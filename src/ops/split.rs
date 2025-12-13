use rten_base::iter::range_chunks;
use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};

use crate::buffer_pool::BufferPool;
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList, OutputTypesContext,
};
use crate::ops::{map_value_view, resolve_axis};
use crate::value::ValueView;

#[derive(Clone, Debug)]
pub enum SplitSizes<'a> {
    /// Split a tensor into pieces with a given size. If the axis size is not
    /// evenly divisible by the size, the last piece will be smaller.
    Size(i32),
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
    let axis_size = input.size(axis);

    let split_with_chunk_size = |chunk_size| {
        range_chunks(0..axis_size, chunk_size)
            .map(|split_range| input.slice_axis(axis, split_range).to_tensor_in(pool))
            .collect()
    };

    let outputs = match split {
        SplitSizes::Size(size) => {
            if size < 1 {
                return Err(OpError::InvalidValue("Split size must be >= 1"));
            }
            split_with_chunk_size(size as usize)
        }
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
            if n_splits > axis_size {
                return Err(OpError::InvalidValue("num_outputs exceeds dim size"));
            }
            split_with_chunk_size(axis_size.div_ceil(n_splits))
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

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
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

    fn output_types(&self, ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some(OutputTypeList::from_elem(
            OutputType::CopyFromInput(0),
            ctx.num_outputs,
        ))
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    Split,
    op,
    shape_ops::Split {
        axis: op.axis as i32,
        num_outputs: op.num_outputs
    }
);

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;

    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpError, OpRunContext, Operator};

    use super::{Split, SplitSizes, split};

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

            let inputs = InputList::from_iter([
                Some(input.view().into()),
                case.splits.as_ref().map(|s| s.view().into()),
            ]);
            let pool = BufferPool::new();
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
            let pool = BufferPool::new();
            let result = split(&pool, input.view(), case.axis, case.splits.clone());
            assert_eq!(result.err().as_ref(), Some(&case.expected));
        })
    }
}
