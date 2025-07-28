use rten_base::iter::range_chunks;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};

use crate::buffer_pool::BufferPool;
use crate::ops::{
    map_value_view, resolve_axis, OpError, OpRunContext, Operator, OutputList, ValueView,
};

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
    pool: &BufferPool,
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

        // In Split v18+, the operator should specify either a vector of split
        // sizes or a count of outputs to produce. Older versions of the Split
        // operator could omit both of these, in which case the number of
        // outputs was determined by looking at the operator node's number of
        // outputs.
        //
        // See https://github.com/robertknight/rten/issues/689.
        let splits = ctx.inputs().get_as(1)?;
        let num_outputs = self.num_outputs.or(ctx.num_outputs());

        let split_sizes = if let Some(splits) = splits {
            SplitSizes::Sizes(splits)
        } else if let Some(num_outputs) = num_outputs {
            SplitSizes::NumSplits(num_outputs)
        } else {
            return Err(OpError::InvalidValue(
                "Either `num_outputs` or `splits` must be set",
            ));
        };

        map_value_view!(input, x, {
            split(ctx.pool(), x, self.axis, split_sizes)
                .map(|tensors| tensors.into_iter().map(|t| t.into()).collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;

    use crate::ops::tests::new_pool;
    use crate::ops::{split, InputList, OpError, OpRunContext, Operator};

    use super::{Split, SplitSizes};

    #[test]
    fn test_split() {
        let input = Tensor::from([[0., 1.], [2., 3.], [4., 5.], [6., 7.], [8., 9.]]);

        #[derive(Debug)]
        struct Case {
            axis: isize,
            splits: Option<NdTensor<i32, 1>>,
            num_outputs: Option<u32>,

            // Number of outputs the Split node has in the graph.
            graph_outputs: Option<u32>,

            expected: Vec<Tensor>,
        }

        let cases = [
            // Positive axis, splits specified via input.
            Case {
                axis: 1,
                splits: Some([1, 1].into()),
                num_outputs: None,
                graph_outputs: None,
                expected: [
                    Tensor::from([[0.], [2.], [4.], [6.], [8.]]),
                    Tensor::from([[1.], [3.], [5.], [7.], [9.]]),
                ]
                .into(),
            },
            // Negative axis, splits specified via input.
            Case {
                axis: -1,
                splits: Some([1, 1].into()),
                num_outputs: None,
                graph_outputs: None,
                expected: [
                    Tensor::from([[0.], [2.], [4.], [6.], [8.]]),
                    Tensor::from([[1.], [3.], [5.], [7.], [9.]]),
                ]
                .into(),
            },
            // Split count specified via `num_outputs` attribute.
            Case {
                axis: 0,
                splits: None,
                num_outputs: Some(3),
                graph_outputs: None,
                expected: [
                    Tensor::from([[0., 1.], [2., 3.]]),
                    Tensor::from([[4., 5.], [6., 7.]]),
                    Tensor::from([[8., 9.]]),
                ]
                .into(),
            },
            // Split count inferred from graph outputs
            Case {
                axis: 1,
                splits: None,
                num_outputs: None,
                graph_outputs: Some(2),
                expected: [
                    Tensor::from([[0.], [2.], [4.], [6.], [8.]]),
                    Tensor::from([[1.], [3.], [5.], [7.], [9.]]),
                ]
                .into(),
            },
        ];

        cases.test_each(|case| {
            let split_op = Split {
                axis: case.axis,
                num_outputs: case.num_outputs,
            };

            let mut inputs: InputList = input.view().into();
            inputs.push_optional(case.splits.as_ref().map(|s| s.view()));
            let pool = new_pool();
            let mut ctx = OpRunContext::new(&pool, &inputs);
            if let Some(n_outputs) = case.graph_outputs {
                ctx.set_num_outputs(n_outputs);
            }
            let results = split_op.run(&ctx).unwrap();
            let results: Vec<Tensor> = results.into_iter().map(|o| o.try_into().unwrap()).collect();

            let expected_splits = match (case.splits.as_ref(), case.num_outputs) {
                (None, Some(n)) => n as usize,
                (Some(sizes), None) => sizes.len(),
                (None, None) => case.graph_outputs.unwrap() as usize,
                (Some(_), Some(_)) => 0,
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
