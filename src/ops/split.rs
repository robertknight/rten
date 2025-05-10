use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};

use crate::iter_util::range_chunks;
use crate::ops::{
    map_input, resolve_axis, static_dims, Input, OpError, OpRunContext, Operator, OutputList,
};
use crate::tensor_pool::TensorPool;

#[derive(Clone, Debug)]
pub enum SplitSizes<'a> {
    /// Split a tensor into pieces with sizes specified by a vector. The sum of
    /// the piece sizes must match the size of the axis.
    Sizes(NdTensorView<'a, i32, 1>),
    /// Split a tensor into N equal-sized pieces. If the size of the axis being
    /// split is not evenly divisible by N, the last chunk will be smaller.
    NumSplits(u32),
}

impl<'a> From<&'a [i32]> for SplitSizes<'a> {
    fn from(val: &'a [i32]) -> Self {
        Self::Sizes(val.into())
    }
}

pub fn split<T: Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    axis: isize,
    split: SplitSizes,
) -> Result<Vec<Tensor<T>>, OpError> {
    let axis = resolve_axis(input.ndim(), axis)?;

    let outputs = match split {
        SplitSizes::Sizes(split) => {
            if split.iter().any(|size| *size < 0) {
                return Err(OpError::InvalidValue("Split sizes must be >= 0"));
            }
            let split_sum = split.iter().sum::<i32>() as usize;
            if split_sum != input.size(axis) {
                return Err(OpError::InvalidValue(
                    "Split sizes do not sum to dimension size",
                ));
            }

            let mut split_start = 0;
            split
                .iter()
                .map(|&split_size| {
                    let split_size = split_size as usize;
                    let split_range = split_start..split_start + split_size;
                    split_start += split_size;
                    input.slice_axis(axis, split_range).to_tensor_in(pool)
                })
                .collect()
        }
        SplitSizes::NumSplits(n_splits) => {
            let n_splits = n_splits as usize;
            if n_splits == 0 {
                return Err(OpError::InvalidValue("num_outputs must be > 0"));
            }
            let dim_size = input.size(axis);
            if n_splits > dim_size {
                return Err(OpError::InvalidValue("num_outputs exceeds dim size"));
            }
            let chunk_size = dim_size.div_ceil(n_splits);
            range_chunks(0..dim_size, chunk_size)
                .map(|chunk| input.slice_axis(axis, chunk).to_tensor_in(pool))
                .collect()
        }
    };

    Ok(outputs)
}

#[derive(Debug)]
pub struct Split {
    pub axis: isize,
    pub num_outputs: Option<u32>,
}

impl Operator for Split {
    fn name(&self) -> &str {
        "Split"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let splits = ctx.inputs().get_as::<i32>(1)?;

        let split_sizes = if let Some(splits) = splits {
            let splits = static_dims!(splits, 1)?;
            SplitSizes::Sizes(splits)
        } else if let Some(num_outputs) = self.num_outputs {
            SplitSizes::NumSplits(num_outputs)
        } else {
            return Err(OpError::InvalidValue(
                "Either `num_outputs` or `splits` must be set",
            ));
        };

        map_input!(input, x, {
            split(ctx.pool(), x, self.axis, split_sizes)
                .map(|tensors| tensors.into_iter().map(|t| t.into()).collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use crate::ops::tests::new_pool;
    use crate::ops::{split, OpError};

    use super::SplitSizes;

    #[test]
    fn test_split() {
        let input = Tensor::from([[0., 1.], [2., 3.], [4., 5.], [6., 7.], [8., 9.]]);

        #[derive(Debug)]
        struct Case<'a> {
            axis: isize,
            splits: SplitSizes<'a>,
            expected: Vec<Tensor>,
        }

        let cases = [
            // Positive axis
            Case {
                axis: 1,
                splits: [1, 1].as_slice().into(),
                expected: [
                    Tensor::from([[0.], [2.], [4.], [6.], [8.]]),
                    Tensor::from([[1.], [3.], [5.], [7.], [9.]]),
                ]
                .into(),
            },
            // Negative axis
            Case {
                axis: -1,
                splits: [1, 1].as_slice().into(),
                expected: [
                    Tensor::from([[0.], [2.], [4.], [6.], [8.]]),
                    Tensor::from([[1.], [3.], [5.], [7.], [9.]]),
                ]
                .into(),
            },
            // Splits specified as count
            Case {
                axis: 0,
                splits: SplitSizes::NumSplits(3),
                expected: [
                    Tensor::from([[0., 1.], [2., 3.]]),
                    Tensor::from([[4., 5.], [6., 7.]]),
                    Tensor::from([[8., 9.]]),
                ]
                .into(),
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let results = split(&pool, input.view(), case.axis, case.splits.clone()).unwrap();
            let expected_splits = match case.splits {
                SplitSizes::NumSplits(n) => n as usize,
                SplitSizes::Sizes(sizes) => sizes.len(),
            };
            assert_eq!(results.len(), expected_splits);
            assert_eq!(results, case.expected);
        })
    }

    #[test]
    fn test_split_invalid_inputs() {
        let input = Tensor::from([[0., 1.], [2., 3.], [4., 5.], [6., 7.], [8., 9.]]);

        #[derive(Debug)]
        struct Case<'a> {
            axis: isize,
            splits: SplitSizes<'a>,
            expected: OpError,
        }

        let cases = [
            Case {
                axis: 2,
                splits: [1, 1].as_slice().into(),
                expected: OpError::InvalidValue("Axis is invalid"),
            },
            Case {
                axis: 1,
                splits: [1, 2].as_slice().into(),
                expected: OpError::InvalidValue("Split sizes do not sum to dimension size"),
            },
            Case {
                axis: 1,
                splits: [1, -2].as_slice().into(),
                expected: OpError::InvalidValue("Split sizes must be >= 0"),
            },
            Case {
                axis: 1,
                splits: SplitSizes::NumSplits(0),
                expected: OpError::InvalidValue("num_outputs must be > 0"),
            },
            Case {
                axis: 1,
                splits: SplitSizes::NumSplits(3),
                expected: OpError::InvalidValue("num_outputs exceeds dim size"),
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let result = split(&pool, input.view(), case.axis, case.splits.clone());
            assert_eq!(result.err().as_ref(), Some(&case.expected));
        })
    }
}
