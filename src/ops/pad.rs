use std::mem::MaybeUninit;

use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, NdTensorViewMut, SliceItem, Tensor, TensorView};

use crate::buffer_pool::BufferPool;
use crate::ops::{
    map_value_view, IntoOpResult, OpError, OpRunContext, Operator, OutputList, ValueView,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PadMode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

pub fn pad<T: Copy + Default + PartialEq>(
    pool: &BufferPool,
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

            let mut output = if const_val == T::default() {
                // Special case zero for platforms that have optimized
                // instructions for this.
                Tensor::zeros_in(pool, &out_shape)
            } else {
                Tensor::full_in(pool, &out_shape, const_val)
            };

            output
                .slice_mut(non_pad_region.as_slice())
                .copy_from(&input);
            output
        }
        // Pad modes which fill the padding region with elements from the
        // source tensor.
        PadMode::Reflect | PadMode::Edge | PadMode::Wrap => {
            const PAD_DIMS: usize = 2;
            let batch_dims = input.ndim().saturating_sub(PAD_DIMS);
            if out_shape[..batch_dims] != input.shape()[..batch_dims] {
                return Err(OpError::UnsupportedValue(
                    "Pad only supports non-constant padding of last 2 dims",
                ));
            }

            if input.shape()[batch_dims..].contains(&0) {
                return Err(OpError::InvalidValue(
                    "Padded dimension for non-constant padding is empty",
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

            for (out_img, in_img) in output
                .inner_iter_mut::<PAD_DIMS>()
                .zip(input.inner_iter::<PAD_DIMS>())
            {
                match mode {
                    PadMode::Reflect => {
                        fill_pad(out_img, in_img, pad_top, pad_left, ReflectPad);
                    }
                    PadMode::Edge => {
                        fill_pad(out_img, in_img, pad_top, pad_left, EdgePad);
                    }
                    PadMode::Wrap => {
                        fill_pad(out_img, in_img, pad_top, pad_left, WrapPad);
                    }
                    // Constant padding is handled separately.
                    PadMode::Constant => unreachable!(),
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

/// Fill `dest` using elements from `src`.
///
/// For each output position, source element coordinates are generated using
/// `src_index` and copied to the output.
fn fill_pad<T: Copy, P: PadSource>(
    mut dest: NdTensorViewMut<MaybeUninit<T>, 2>,
    src: NdTensorView<T, 2>,
    pad_top: usize,
    pad_left: usize,
    src_index: P,
) {
    let out_rows = dest.size(0);
    let out_cols = dest.size(1);

    let src_rows = src.size(0);
    let src_cols = src.size(1);

    for y in 0..out_rows {
        let src_y = src_index.src_index(y, src_rows, pad_top);
        debug_assert!(src_y < src_rows);

        for x in 0..out_cols {
            let src_x = src_index.src_index(x, src_cols, pad_left);
            debug_assert!(src_x < src_cols);

            // Safety:
            //  - y and x are valid coords.
            //  - src_y and src_x are valid coords since `src_index` returns
            //    values in [0, len).
            unsafe {
                dest.get_unchecked_mut([y, x])
                    .write(*src.get_unchecked([src_y, src_x]));
            }
        }
    }
}

/// Computes the source elements to use when filling a tensor with padding.
///
/// # Safety
///
/// The `src_index` method must return indexes in `[0, len)`.
unsafe trait PadSource {
    /// Compute the coordinate of the source element to copy when filling a
    /// tensor with padding.
    ///
    /// `dest` is the destination coordinate in [0, len), `len` is the size of
    /// the dimension (always >= 1) and `pad_start` is the number of padding
    /// elements added at the start of the dimension.
    ///
    /// The returned index must be in [0, len).
    fn src_index(&self, dest: usize, len: usize, pad_start: usize) -> usize;
}

// Fill the padding region by reflecting the start or end of the source axis.
struct ReflectPad;

// Safety: `src_index` result is in [0, len).
unsafe impl PadSource for ReflectPad {
    fn src_index(&self, dest: usize, len: usize, pad_start: usize) -> usize {
        let x = dest as isize;
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
}

/// Fill the padding region by taking elements from the start or end of the
/// source axis.
struct EdgePad;

// Safety: `src_index` result is in [0, len).
unsafe impl PadSource for EdgePad {
    fn src_index(&self, dest: usize, len: usize, pad_start: usize) -> usize {
        let len = len as isize;
        let dest = dest as isize;
        let src = dest - pad_start as isize;
        src.clamp(0, len - 1) as usize
    }
}

/// Fill the padding region by wrapping coordinates around to the start or end
/// of the source axis.
struct WrapPad;

// Safety: `src_index` result is in [0, len).
unsafe impl PadSource for WrapPad {
    fn src_index(&self, dest: usize, len: usize, pad_start: usize) -> usize {
        let len = len as isize;
        let dest = dest as isize;
        let pad_start = pad_start as isize;
        (dest - pad_start).rem_euclid(len) as usize
    }
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
        let pads = inputs.require_as(1)?;
        let axes: Option<NdTensorView<i32, 1>> = inputs.get_as(3)?;

        if axes.is_some() {
            return Err(OpError::UnsupportedValue(
                "Pad operator does not yet support `axes` input",
            ));
        }

        map_value_view!(input, x, {
            let const_val = inputs.get_as(2)?.unwrap_or_default();
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
    use crate::ops::{pad, CastError, DataType, OpError, OperatorExt, Pad, PadMode, Value};

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

    #[derive(Debug)]
    struct Case {
        input: Tensor,
        pads: NdTensor<i32, 1>,
        mode: PadMode,
        expected: Result<Tensor, OpError>,
    }

    fn test_pad_mode(cases: &[Case]) {
        cases.test_each(|case| {
            let Case {
                input,
                pads,
                mode,
                expected,
            } = case;

            let pool = new_pool();
            let result = pad(&pool, input.view(), &pads.view(), *mode, 0.);
            match (result, expected) {
                (Ok(result), Ok(expected)) => {
                    expect_equal(&result, &expected).unwrap();
                }
                (result, expected) => assert_eq!(&result, expected),
            }
        });
    }

    #[test]
    fn test_pad_constant() {
        let cases = [
            // Test case from ONNX spec.
            Case {
                input: [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]].into(),
                pads: [0, 2, 0, 0].into(),
                mode: PadMode::Constant,
                expected: Ok(Tensor::from([
                    [0.0, 0.0, 1.0, 1.2],
                    [0.0, 0.0, 2.3, 3.4],
                    [0.0, 0.0, 4.5, 5.7],
                ])),
            },
        ];

        test_pad_mode(&cases);
    }

    #[test]
    fn test_pad_reflect() {
        let cases = [
            // Test case from ONNX spec.
            //
            // Pad start columns of a 2D tensor.
            Case {
                input: [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]].into(),
                pads: [0, 2, 0, 0].into(),
                mode: PadMode::Reflect,
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
                mode: PadMode::Reflect,
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
                mode: PadMode::Reflect,
                expected: Ok(Tensor::from([[4., 3., 2., 1., 2., 3., 4., 5., 4., 3., 2.]])),
            },
            // Pad start and end rows of a 2D tensor.
            Case {
                input: Tensor::from([1., 2., 3., 4., 5.]).into_shape([5, 1].as_slice()),
                pads: [3, 0, 3, 0].into(),
                mode: PadMode::Reflect,
                expected: Ok(Tensor::from([4., 3., 2., 1., 2., 3., 4., 5., 4., 3., 2.])
                    .into_shape([5 + 2 * 3, 1].as_slice())),
            },
            // Pad start and end of a 1D tensor.
            Case {
                input: [1., 2., 3., 4.].into(),
                pads: [2, 2].into(),
                mode: PadMode::Reflect,
                expected: Ok(Tensor::from([3., 2., 1., 2., 3., 4., 3., 2.])),
            },
            // Scalar input. This is always a noop since there are no dimensions
            // to pad.
            Case {
                input: Tensor::from(2.),
                pads: NdTensor::from([]),
                mode: PadMode::Reflect,
                expected: Ok(Tensor::from(2.)),
            },
            // Pad start columns of a 3D tensor.
            Case {
                input: [[[1., 2., 3.]]].into(),
                pads: [0, 0, 2, 0, 0, 0].into(),
                mode: PadMode::Reflect,
                expected: Ok(Tensor::from([[[3., 2., 1., 2., 3.]]])),
            },
            // Pad end columns of a 3D tensor.
            Case {
                input: [[[1., 2., 3.]]].into(),
                pads: [0, 0, 0, 0, 0, 2].into(),
                mode: PadMode::Reflect,
                expected: Ok(Tensor::from([[[1., 2., 3., 2., 1.]]])),
            },
            // Pad start rows of a 3D tensor.
            Case {
                input: [[[1.], [2.], [3.]]].into(),
                pads: [0, 2, 0, 0, 0, 0].into(),
                mode: PadMode::Reflect,
                expected: Ok(Tensor::from([[[3.], [2.], [1.], [2.], [3.]]])),
            },
            // Pad channel dimension of a 3D tensor.
            Case {
                input: [[[1., 2., 3.]]].into(),
                pads: [0, 0, 0, 2, 0, 0].into(),
                mode: PadMode::Reflect,
                expected: Err(OpError::UnsupportedValue(
                    "Pad only supports non-constant padding of last 2 dims",
                )),
            },
            // Pad zero-size dimension.
            Case {
                input: Tensor::zeros(&[3, 0]),
                pads: NdTensor::from([0, 2, 0, 0]),
                mode: PadMode::Reflect,
                expected: Err(OpError::InvalidValue(
                    "Padded dimension for non-constant padding is empty",
                )),
            },
        ];

        test_pad_mode(&cases);
    }

    // The Reflect, Edge and Wrap pad modes share most of the implementation.
    // Most tests use the reflect mode, plus there are a smaller subset to test
    // the different behaviors of Edge and Wrap.

    #[test]
    fn test_pad_edge() {
        let cases = [
            // Test case from ONNX spec.
            Case {
                input: [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]].into(),
                pads: [0, 2, 0, 0].into(),
                mode: PadMode::Edge,
                expected: Ok(Tensor::from([
                    [1.0, 1.0, 1.0, 1.2],
                    [2.3, 2.3, 2.3, 3.4],
                    [4.5, 4.5, 4.5, 5.7],
                ])),
            },
        ];

        test_pad_mode(&cases);
    }

    #[test]
    fn test_pad_wrap() {
        let cases = [
            // Test case from ONNX spec.
            Case {
                input: [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]].into(),
                pads: [2, 1, 1, 1].into(),
                mode: PadMode::Wrap,
                expected: Ok(Tensor::from([
                    [3.4, 2.3, 3.4, 2.3],
                    [5.7, 4.5, 5.7, 4.5],
                    [1.2, 1.0, 1.2, 1.0],
                    [3.4, 2.3, 3.4, 2.3],
                    [5.7, 4.5, 5.7, 4.5],
                    [1.2, 1.0, 1.2, 1.0],
                ])),
            },
        ];

        test_pad_mode(&cases);
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
        #[derive(Debug)]
        struct Case {
            input: Tensor<f32>,
            pads: Tensor<i32>,
            const_val: Option<Value>,
            expected_error: OpError,
        }

        let input = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let op = Pad {
            mode: PadMode::Constant,
        };

        let cases = [
            // Wrong padding vector length.
            Case {
                input: input.clone(),
                pads: from_slice(&[1]),
                const_val: None,
                expected_error: OpError::InvalidValue("padding length should be 2 * input dims"),
            },
            // Unsupported padding amounts.
            Case {
                input: input.clone(),
                pads: from_slice(&[1, 1, 1, -1]),
                const_val: None,
                expected_error: OpError::InvalidValue("Pad only supports positive pads"),
            },
            // Wrong constant value type.
            Case {
                input: input.clone(),
                pads: from_slice(&[1, 1, 1, -1]),
                const_val: Some(Tensor::from(1).into()),
                expected_error: OpError::InputCastFailed {
                    index: 2,
                    error: CastError::WrongType {
                        actual: DataType::Int32,
                        expected: DataType::Float,
                    },
                },
            },
            // Constant value not a scalar.
            Case {
                input: input.clone(),
                pads: from_slice(&[1, 1, 1, -1]),
                const_val: Some(from_slice(&[1.0, 2.0]).into()),
                expected_error: OpError::InputCastFailed {
                    index: 2,
                    error: CastError::WrongRank {
                        actual: 1,
                        expected: 0,
                    },
                },
            },
        ];

        cases.test_each(|case| {
            let Case {
                input,
                pads,
                const_val,
                expected_error,
            } = case;

            let result = if let Some(const_val) = const_val {
                op.run_simple::<_, Tensor<f32>>((input, pads, const_val))
            } else {
                op.run_simple::<_, Tensor<f32>>((input, pads))
            };

            assert_eq!(result.err().as_ref(), Some(expected_error));
        });
    }
}
