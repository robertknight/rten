use std::mem::MaybeUninit;

use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, NdTensorViewMut, Tensor, TensorView};

use crate::buffer_pool::BufferPool;
use crate::ops::{
    map_value_view, IntoOpResult, OpError, OpRunContext, Operator, OutputList, ValueView,
};

fn trilu_kernel<T: Copy + Default, const UPPER: bool>(
    mut out_mat: NdTensorViewMut<MaybeUninit<T>, 2>,
    in_mat: NdTensorView<T, 2>,
    k: i32,
) {
    assert_eq!(out_mat.shape(), in_mat.shape());
    let [rows, cols] = out_mat.shape();

    for y in 0..rows {
        for x in 0..cols {
            let delta = y as i32 + k - x as i32;
            let copy = if UPPER { delta <= 0 } else { delta >= 0 };

            // Safety: y and x are in-bounds for the output, and also the input
            // since it has the same shape.
            let value = unsafe { *in_mat.get_unchecked([y, x]) };
            unsafe { out_mat.get_unchecked_mut([y, x]) }.write(if copy {
                value
            } else {
                T::default()
            });
        }
    }
}

pub fn trilu<T: Copy + Default>(
    pool: &BufferPool,
    input: TensorView<T>,
    k: i32,
    upper: bool,
) -> Result<Tensor<T>, OpError> {
    if input.ndim() < 2 {
        return Err(OpError::InvalidValue("Input must have >= 2 dims"));
    }

    let mut output = Tensor::uninit_in(pool, input.shape());

    for (out_mat, in_mat) in output.inner_iter_mut::<2>().zip(input.inner_iter::<2>()) {
        if upper {
            trilu_kernel::<_, true>(out_mat, in_mat, k)
        } else {
            trilu_kernel::<_, false>(out_mat, in_mat, k)
        }
    }

    // Safety: All inner matrices were initialized by `trilu_kernel`.
    let output = unsafe { output.assume_init() };

    Ok(output)
}

#[derive(Debug)]
pub struct Trilu {
    pub upper: bool,
}

impl Operator for Trilu {
    fn name(&self) -> &str {
        "Trilu"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let k = inputs.get_as(1)?.unwrap_or(0);

        map_value_view!(input, input, [FloatTensor, Int32Tensor], {
            trilu(ctx.pool(), input, k, self.upper).into_op_result()
        })
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use crate::ops::tests::new_pool;
    use crate::ops::{trilu, OpError};

    #[test]
    fn test_trilu() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<i32>,
            expected: Tensor<i32>,
            upper: bool,
            k: i32,
        }

        let in_3x3 = Tensor::arange(1, 10, None).into_shape([3, 3].as_slice());

        let cases = [
            // k = 0, upper = true
            Case {
                input: in_3x3.clone(),
                expected: [[1, 2, 3], [0, 5, 6], [0, 0, 9]].into(),
                k: 0,
                upper: true,
            },
            // +ve k, upper = true
            Case {
                input: in_3x3.clone(),
                expected: [[0, 2, 3], [0, 0, 6], [0, 0, 0]].into(),
                k: 1,
                upper: true,
            },
            // -ve k, upper = true
            Case {
                input: in_3x3.clone(),
                expected: [[1, 2, 3], [4, 5, 6], [0, 8, 9]].into(),
                k: -1,
                upper: true,
            },
            // k = 0, upper = false
            Case {
                input: in_3x3.clone(),
                expected: [[1, 0, 0], [4, 5, 0], [7, 8, 9]].into(),
                k: 0,
                upper: false,
            },
            // +ve k, upper = false
            Case {
                input: in_3x3.clone(),
                expected: [[1, 2, 0], [4, 5, 6], [7, 8, 9]].into(),
                k: 1,
                upper: false,
            },
            // -ve k, upper = false
            Case {
                input: in_3x3.clone(),
                expected: [[0, 0, 0], [4, 0, 0], [7, 8, 0]].into(),
                k: -1,
                upper: false,
            },
            // Batch of matrices
            Case {
                input: [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                ]
                .into(),
                expected: [
                    [[1, 2, 3], [0, 5, 6], [0, 0, 9]],
                    [[9, 8, 7], [0, 5, 4], [0, 0, 1]],
                ]
                .into(),
                k: 0,
                upper: true,
            },
            // Non-square (wide) matrix
            Case {
                input: Tensor::arange(1, 16, None).into_shape([3, 5].as_slice()),
                expected: [[1, 2, 3, 4, 5], [0, 7, 8, 9, 10], [0, 0, 13, 14, 15]].into(),
                k: 0,
                upper: true,
            },
            // Non-square (tall) matrix
            Case {
                input: Tensor::arange(1, 16, None).into_shape([5, 3].as_slice()),
                expected: [[1, 2, 3], [0, 5, 6], [0, 0, 9], [0, 0, 0], [0, 0, 0]].into(),
                k: 0,
                upper: true,
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let result = trilu(&pool, case.input.view(), case.k, case.upper).unwrap();
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_trilu_invalid() {
        let pool = new_pool();
        let input = Tensor::from([1]);
        let result = trilu(&pool, input.view(), 0, true /* upper */);
        assert_eq!(
            result,
            Err(OpError::InvalidValue("Input must have >= 2 dims"))
        );
    }
}
