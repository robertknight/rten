use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, SliceItem, Tensor, TensorView};

use crate::ops::{
    map_input, static_dims, Input, IntoOpResult, OpError, OpRunContext, Operator, OutputList,
};
use crate::tensor_pool::TensorPool;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PadMode {
    Constant,
    Reflect,
}

pub fn pad<T: Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    padding: &NdTensorView<i32, 1>,
    mode: PadMode,
    const_val: T,
) -> Result<Tensor<T>, OpError> {
    if padding.size(0) != input.ndim() * 2 {
        return Err(OpError::InvalidValue(
            "padding length should be 2 * input dims",
        ));
    }
    if !padding.iter().all(|x| *x >= 0) {
        return Err(OpError::InvalidValue("Pad only supports positive pads"));
    }

    let out_shape: Vec<_> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(i, size)| {
            let start_pad = padding[i] as usize;
            let end_pad = padding[[input.ndim() + i]] as usize;
            start_pad + size + end_pad
        })
        .collect();

    // Just copy the tensor if no padding is required.
    if out_shape == input.shape() {
        return Ok(input.to_tensor_in(pool));
    }

    let output = match mode {
        PadMode::Constant => {
            let non_pad_region: Vec<SliceItem> = input
                .shape()
                .iter()
                .enumerate()
                .map(|(i, size)| {
                    let start_pad = padding[i] as usize;
                    (start_pad..start_pad + size).into()
                })
                .collect();

            let mut output = Tensor::full_in(pool, &out_shape, const_val);
            output
                .slice_mut(non_pad_region.as_slice())
                .copy_from(&input);
            output
        }
        PadMode::Reflect => {
            const PAD_DIMS: usize = 2;
            let batch_dims = input.ndim().saturating_sub(PAD_DIMS);
            if out_shape[..batch_dims] != input.shape()[..batch_dims] {
                return Err(OpError::UnsupportedValue(
                    "Pad only supports reflect padding of last 2 dims",
                ));
            }

            if input.shape()[batch_dims..].contains(&0) {
                return Err(OpError::InvalidValue(
                    "Padded dimension for reflect padding is empty",
                ));
            }

            let pad_dims = input.ndim() - batch_dims;
            let (pad_top, pad_left) = if pad_dims == 1 {
                (0, padding[[batch_dims]] as usize)
            } else {
                (
                    padding[[batch_dims]] as usize,
                    padding[[batch_dims + 1]] as usize,
                )
            };

            let mut input = input.view();
            let mut output = Tensor::uninit_in(pool, &out_shape);

            // For inputs with fewer dims than the padding inner loop, insert
            // extra 1-sized dims at the start.
            while input.ndim() < PAD_DIMS {
                input.insert_axis(0);
                output.insert_axis(0);
            }

            for (mut out_img, in_img) in output
                .inner_iter_mut::<PAD_DIMS>()
                .zip(input.inner_iter::<PAD_DIMS>())
            {
                let out_rows = out_img.size(0);
                let out_cols = out_img.size(1);

                let src_rows = in_img.size(0);
                let src_cols = in_img.size(1);

                for y in 0..out_rows {
                    let src_y = reflect_pad_src(y, src_rows, pad_top);

                    for x in 0..out_cols {
                        let src_x = reflect_pad_src(x, src_cols, pad_left);

                        // Safety:
                        //  - y and x are valid coords.
                        //  - src_y and src_x are valid coords since
                        //    `reflect_pad_src` returns values in [0, len)
                        unsafe {
                            out_img
                                .get_unchecked_mut([y, x])
                                .write(*in_img.get_unchecked([src_y, src_x]));
                        }
                    }
                }
            }

            while output.ndim() > out_shape.len() {
                output.remove_axis(0);
            }

            // Safety: We filled all elements of output.
            unsafe { output.assume_init() }
        }
    };

    Ok(output)
}

/// Compute the coordinate for the source element when applying reflection
/// padding along a single axis.
///
/// `x` is the destination coordinate, `len` is the size of the dimension and
/// `pad_start` is the number of padding elements added at the start of the
/// dimension.
fn reflect_pad_src(x: usize, len: usize, pad_start: usize) -> usize {
    let x = x as isize;
    let len = len as isize;
    let pad_start = pad_start as isize;

    // Compute all possible values and then pick one, so this gets compiled to
    // conditional moves instead of branches.
    let src_x_start = pad_start - x;
    let src_x_mid = x - pad_start;
    let src_x_end = len - (x - len - pad_start) - 2;

    let src_x = if x < pad_start {
        src_x_start
    } else if x < len + pad_start {
        src_x_mid
    } else {
        src_x_end
    };

    // Use `rem_euclid` so that `src_x ∈ [0, len)` if src_x < 0.
    src_x.rem_euclid(len) as usize
}

#[derive(Debug)]
pub struct Pad {
    pub mode: PadMode,
}

impl Operator for Pad {
    fn name(&self) -> &str {
        "Pad"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let pads = inputs.require_as::<i32>(1)?;
        let pads = static_dims!(pads, 1)?;
        let axes = inputs.get_as::<i32>(3)?;

        if axes.is_some() {
            return Err(OpError::UnsupportedValue(
                "Pad operator does not yet support `axes` input",
            ));
        }

        map_input!(input, x, {
            let const_val = inputs.get_as_scalar(2)?.unwrap_or_default();
            pad(ctx.pool(), x, &pads, self.mode, const_val).into_op_result()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;

    use crate::ops::tests::new_pool;
    use crate::ops::{pad, OpError, OperatorExt, Pad, PadMode};

    fn from_slice<T: Clone>(data: &[T]) -> Tensor<T> {
        Tensor::from_data(&[data.len()], data.to_vec())
    }

    #[test]
    fn test_pad() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Same padding around each edge.
        let input = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = Tensor::from_data(
            &[4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
        let const_pads = &[1, 1, 1, 1];
        let result = pad(
            &pool,
            input.view(),
            &const_pads.into(),
            PadMode::Constant,
            0.0,
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // Zero padding (no-op)
        let zero_pads = &[0, 0, 0, 0];
        let result = pad(
            &pool,
            input.view(),
            &zero_pads.into(),
            PadMode::Constant,
            0.0,
        )
        .unwrap();
        expect_equal(&result, &input)?;

        // Un-even padding
        let input = Tensor::from_data(&[1, 2, 2], vec![1, 2, 3, 4]);
        let pads = &[0, 0, 0, 0, 1, 0];
        let result = pad(&pool, input.view(), &pads.into(), PadMode::Constant, 0).unwrap();
        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(result.data().unwrap(), &[1, 2, 3, 4, 0, 0]);

        Ok(())
    }

    #[test]
    fn test_pad_constant_val() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = Tensor::from_data(
            &[4, 4],
            vec![
                9., 9., 9., 9., 9., 1., 2., 9., 9., 3., 4., 9., 9., 9., 9., 9.,
            ],
        );
        let const_pads = &[1, 1, 1, 1];
        let result = pad(
            &pool,
            input.view(),
            &const_pads.into(),
            PadMode::Constant,
            9.,
        )
        .unwrap();
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_pad_reflect() -> Result<(), Box<dyn Error>> {
        #[derive(Debug)]
        struct Case {
            input: Tensor,
            pads: NdTensor<i32, 1>,
            expected: Result<Tensor, OpError>,
        }

        let cases = [
            // Test case from ONNX spec.
            //
            // Pad start columns of a 2D tensor.
            Case {
                input: [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]].into(),
                pads: [0, 2, 0, 0].into(),
                expected: Ok(Tensor::from([
                    [1.0, 1.2, 1.0, 1.2],
                    [2.3, 3.4, 2.3, 3.4],
                    [4.5, 5.7, 4.5, 5.7],
                ])),
            },
            // Pad end columns of a 2D tensor.
            Case {
                input: [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]].into(),
                pads: [0, 0, 0, 2].into(),
                expected: Ok(Tensor::from([
                    [1.0, 1.2, 1.0, 1.2],
                    [2.3, 3.4, 2.3, 3.4],
                    [4.5, 5.7, 4.5, 5.7],
                ])),
            },
            // Pad start and end columns of a 2D tensor.
            Case {
                input: [[1., 2., 3., 4., 5.]].into(),
                pads: [0, 3, 0, 3].into(),
                expected: Ok(Tensor::from([[4., 3., 2., 1., 2., 3., 4., 5., 4., 3., 2.]])),
            },
            // Pad start and end rows of a 2D tensor.
            Case {
                input: Tensor::from([1., 2., 3., 4., 5.]).into_shape([5, 1].as_slice()),
                pads: [3, 0, 3, 0].into(),
                expected: Ok(Tensor::from([4., 3., 2., 1., 2., 3., 4., 5., 4., 3., 2.])
                    .into_shape([5 + 2 * 3, 1].as_slice())),
            },
            // Pad start and end of a 1D tensor.
            Case {
                input: [1., 2., 3., 4.].into(),
                pads: [2, 2].into(),
                expected: Ok(Tensor::from([3., 2., 1., 2., 3., 4., 3., 2.])),
            },
            // Scalar input. This is always a noop since there are no dimensions
            // to pad.
            Case {
                input: Tensor::from(2.),
                pads: NdTensor::from([]),
                expected: Ok(Tensor::from(2.)),
            },
            // Pad start columns of a 3D tensor.
            Case {
                input: [[[1., 2., 3.]]].into(),
                pads: [0, 0, 2, 0, 0, 0].into(),
                expected: Ok(Tensor::from([[[3., 2., 1., 2., 3.]]])),
            },
            // Pad end columns of a 3D tensor.
            Case {
                input: [[[1., 2., 3.]]].into(),
                pads: [0, 0, 0, 0, 0, 2].into(),
                expected: Ok(Tensor::from([[[1., 2., 3., 2., 1.]]])),
            },
            // Pad start rows of a 3D tensor.
            Case {
                input: [[[1.], [2.], [3.]]].into(),
                pads: [0, 2, 0, 0, 0, 0].into(),
                expected: Ok(Tensor::from([[[3.], [2.], [1.], [2.], [3.]]])),
            },
            // Pad channel dimension of a 3D tensor.
            Case {
                input: [[[1., 2., 3.]]].into(),
                pads: [0, 0, 0, 2, 0, 0].into(),
                expected: Err(OpError::UnsupportedValue(
                    "Pad only supports reflect padding of last 2 dims",
                )),
            },
            // Pad zero-size dimension.
            Case {
                input: Tensor::zeros(&[3, 0]),
                pads: NdTensor::from([0, 2, 0, 0]),
                expected: Err(OpError::InvalidValue(
                    "Padded dimension for reflect padding is empty",
                )),
            },
        ];

        cases.test_each(|case| {
            let Case {
                input,
                pads,
                expected,
            } = case;

            let pool = new_pool();
            let result = pad(&pool, input.view(), &pads.view(), PadMode::Reflect, 0.);
            match (result, expected) {
                (Ok(result), Ok(expected)) => {
                    expect_equal(&result, &expected).unwrap();
                }
                (result, expected) => assert_eq!(&result, expected),
            }
        });

        Ok(())
    }

    #[test]
    fn test_pad_op() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let pads = from_slice(&[1, 1, 1, 1]);
        let expected = Tensor::from_data(
            &[4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );

        let op = Pad {
            mode: PadMode::Constant,
        };
        let result: Tensor<f32> = op.run_simple((&input, &pads)).unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_pad_invalid_inputs() {
        let input = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let op = Pad {
            mode: PadMode::Constant,
        };

        // Wrong padding vector length.
        let invalid_pads = from_slice(&[1]);
        let result = op.run_simple::<_, Tensor<f32>>((&input, &invalid_pads));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "padding length should be 2 * input dims"
            ))
        );

        // Unsupported padding amounts.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let result = op.run_simple::<_, Tensor<f32>>((&input, &invalid_pads));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Pad only supports positive pads"))
        );

        // Wrong constant value type.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let const_int = Tensor::from(1);
        let result = op.run_simple::<_, Tensor<f32>>((&input, &invalid_pads, &const_int));
        assert_eq!(result.err(), Some(OpError::IncorrectInputType));

        // Constant value not a scalar.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let int_vec = from_slice(&[1.0, 2.0]);
        let result = op.run_simple::<_, Tensor<f32>>((&input, &invalid_pads, &int_vec));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Expected scalar value"))
        );
    }
}
