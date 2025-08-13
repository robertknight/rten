use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use crate::buffer_pool::BufferPool;
use crate::ops::split::split;
use crate::ops::split::SplitSizes;
use crate::ops::{
    map_value_view, resolve_axis, resolve_index, Concat, InputList, IntoOpResult, OpError,
    OpRunContext, Operator, OutputList,
};
use crate::value::{CastError, DataType, Sequence, Value, ValueView};

#[derive(Debug)]
pub struct SequenceEmpty {
    pub dtype: Option<DataType>,
}

impl Operator for SequenceEmpty {
    fn name(&self) -> &str {
        "SequenceEmpty"
    }

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let dtype = self.dtype.unwrap_or(DataType::Float);
        Value::from(Sequence::new(dtype)).into_op_result()
    }
}

#[derive(Debug)]
pub struct SequenceAt {}

impl Operator for SequenceAt {
    fn name(&self) -> &str {
        "SequenceAt"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let seq: &Sequence = ctx.inputs().require_as(0)?;
        let pos: i32 = ctx.inputs().require_as(1)?;
        let pos = resolve_index(seq.len(), pos as isize)
            .ok_or(OpError::InvalidValue("Sequence position is invalid"))?;
        seq.at(pos)
            .unwrap()
            .to_owned_in(ctx.pool())
            .into_op_result()
    }
}

/// Cast `value` to the same tensor type as `like`.
fn cast_like<'a, T>(
    value: ValueView<'a>,
    #[allow(unused_variables)] like: &TensorView<T>,
) -> Result<TensorView<'a, T>, CastError>
where
    for<'b> TensorView<'b, T>: TryFrom<ValueView<'b>, Error = CastError>,
{
    value.try_into()
}

#[derive(Debug)]
pub struct SequenceConstruct {}

impl Operator for SequenceConstruct {
    fn name(&self) -> &str {
        "SequenceConstruct"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let first = ctx.inputs().require(0)?;
        let rest = ctx.inputs().iter().flatten().skip(1);

        let sequence = map_value_view!(first, first, {
            let mut items = Vec::with_capacity(ctx.inputs().len());
            items.push(first.to_tensor_in(ctx.pool()));

            for value in rest {
                let tensor = cast_like(value, &first)?;
                items.push(tensor.to_tensor_in(ctx.pool()));
            }

            Ok(Sequence::from(items))
        })?;

        Value::from(sequence).into_op_result()
    }
}

fn sequence_erase(mut seq: Sequence, pos: Option<i32>) -> Result<Sequence, OpError> {
    let Some(max_index) = seq.len().checked_sub(1) else {
        return Err(OpError::InvalidValue(
            "Cannot remove element from empty sequence",
        ));
    };

    let pos = pos
        .map(|pos| {
            resolve_index(seq.len(), pos as isize)
                .ok_or(OpError::InvalidValue("Sequence position is invalid"))
        })
        .transpose()?
        .unwrap_or(max_index);

    seq.remove(pos).unwrap();

    Ok(seq)
}

#[derive(Debug)]
pub struct SequenceErase {}

impl Operator for SequenceErase {
    fn name(&self) -> &str {
        "SequenceErase"
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let seq: &Sequence = ctx.inputs().require_as(0)?;
        let pos: Option<i32> = ctx.inputs().get_as(1)?;
        sequence_erase(seq.clone(), pos)
            .map(Value::from)
            .into_op_result()
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let seq: Sequence = input.try_into()?;
        let pos: Option<i32> = ctx.inputs().get_as(0)?;
        sequence_erase(seq, pos).map(Value::from)
    }
}

fn sequence_insert(
    pool: &BufferPool,
    mut seq: Sequence,
    pos: Option<i32>,
    val: ValueView,
) -> Result<Sequence, OpError> {
    if seq.dtype() != val.dtype() {
        return Err(OpError::InvalidValue(
            "Tensor type does not match sequence type",
        ));
    }
    let pos = pos
        .map(|pos| {
            resolve_index(seq.len() + 1, pos as isize)
                .ok_or(OpError::InvalidValue("Sequence position is invalid"))
        })
        .transpose()?
        .unwrap_or(seq.len());

    seq.insert(pos, val.to_owned_in(pool)).unwrap();

    Ok(seq)
}

#[derive(Debug)]
pub struct SequenceInsert {}

impl Operator for SequenceInsert {
    fn name(&self) -> &str {
        "SequenceInsert"
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let seq: &Sequence = ctx.inputs().require_as(0)?;
        let value = ctx.inputs().require(1)?;
        let pos: Option<i32> = ctx.inputs().get_as(2)?;
        sequence_insert(ctx.pool(), seq.clone(), pos, value)
            .map(Value::from)
            .into_op_result()
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let seq: Sequence = input.try_into()?;
        let value = ctx.inputs().require(0)?;
        let pos: Option<i32> = ctx.inputs().get_as(1)?;
        sequence_insert(ctx.pool(), seq, pos, value).map(Value::from)
    }
}

#[derive(Debug)]
pub struct SequenceLength {}

impl Operator for SequenceLength {
    fn name(&self) -> &str {
        "SequenceLength"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let seq: &Sequence = ctx.inputs().require_as(0)?;
        let len = seq.len() as i32;
        Tensor::from(len).into_op_result()
    }
}

#[derive(Debug)]
pub struct ConcatFromSequence {
    pub axis: i32,
    pub new_axis: bool,
}

impl Operator for ConcatFromSequence {
    fn name(&self) -> &str {
        "ConcatFromSequence"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let seq: &Sequence = ctx.inputs().require_as(0)?;

        // Prepare inputs for Concat op.
        //
        // The spec requires that all inputs must have the same shape, except
        // for the axis along which concatenation is happening. Here we rely
        // on the Concat op to check this.
        let values: Result<Vec<ValueView>, OpError> = seq
            .iter()
            .map(|value| {
                // The Concat op only supports concatenating on an existing axis, so
                // if `new_axis` is set, add a 1-sized axis to each of the values.
                if self.new_axis {
                    let resolved_axis = resolve_axis(value.ndim(), self.axis as isize)?;
                    map_value_view!(value, tensor, {
                        let mut tensor = tensor;
                        tensor.insert_axis(resolved_axis);
                        Ok(tensor.into())
                    })
                } else {
                    Ok(value)
                }
            })
            .collect();
        let values = values?;

        // Execute Concat op
        let concat_op = Concat {
            axis: self.axis as isize,
        };
        let concat_inputs = InputList::from(&values);
        let concat_ctx = ctx.with_new_inputs(&concat_inputs);
        concat_op.run(&concat_ctx)
    }
}

#[derive(Debug)]
pub struct SplitToSequence {
    pub axis: i32,
    pub keep_dims: bool,
}

impl Operator for SplitToSequence {
    fn name(&self) -> &str {
        "SplitToSequence"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let splits: Option<TensorView<i32>> = ctx.inputs().get_as(1)?;
        let axis = resolve_axis(input.ndim(), self.axis as isize)?;

        let split_sizes = if let Some(splits) = &splits {
            // Check both `ndim` and `item` because `item` only checks if tensor
            // has exactly one element.
            match (splits.ndim(), splits.item()) {
                (0, Some(&size)) => {
                    if size >= 1 {
                        SplitSizes::Size(size)
                    } else {
                        return Err(OpError::InvalidValue("Split size must be >= 1"));
                    }
                }
                (1, _) => SplitSizes::Sizes(splits.nd_view()),
                _ => {
                    return Err(OpError::InvalidValue("Split size must be scalar or vector"));
                }
            }
        } else {
            SplitSizes::Size(1)
        };

        // `keep_dim` is ignored if the split input is set. The dimension can
        // only be removed if it has size 1, which is true when split is unset.
        let keep_dim = if splits.is_none() {
            self.keep_dims
        } else {
            true
        };

        let sequence = map_value_view!(input, input, {
            split(ctx.pool(), input, self.axis as isize, split_sizes).map(|mut pieces| {
                if !keep_dim {
                    for item in &mut pieces {
                        item.remove_axis(axis);
                    }
                }
                Sequence::from(pieces)
            })
        })?;

        Value::from(sequence).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use super::{
        ConcatFromSequence, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase,
        SequenceInsert, SequenceLength, SplitToSequence,
    };
    use crate::ops::{InputList, OpError, OperatorExt};
    use crate::value::{CastError, DataType, Sequence, Value, ValueView};

    #[test]
    fn test_sequence_empty() {
        #[derive(Debug)]
        struct Case {
            dtype: Option<DataType>,
            expected: DataType,
        }

        let cases = [
            Case {
                dtype: None,
                expected: DataType::Float,
            },
            Case {
                dtype: Some(DataType::Int32),
                expected: DataType::Int32,
            },
        ];

        cases.test_each(|case| {
            let op = SequenceEmpty { dtype: case.dtype };
            let value: Sequence = op.run_simple(InputList::default()).unwrap();
            assert_eq!(value.dtype(), case.expected);
            assert_eq!(value.len(), 0);
        });
    }

    #[test]
    fn test_sequence_at() {
        #[derive(Debug)]
        struct Case {
            seq: Sequence,
            pos: i32,
            expected: Result<Value, OpError>,
        }

        let cases = [
            // Positive index
            Case {
                seq: [1., 2.].map(Tensor::from).into(),
                pos: 0,
                expected: Ok(Tensor::from(1.).into()),
            },
            // Negative index
            Case {
                seq: [1., 2.].map(Tensor::from).into(),
                pos: -1,
                expected: Ok(Tensor::from(2.).into()),
            },
            // Out-of-bounds index
            Case {
                seq: [1., 2.].map(Tensor::from).into(),
                pos: 2,
                expected: Err(OpError::InvalidValue("Sequence position is invalid")),
            },
        ];

        cases.test_each(|case| {
            let op = SequenceAt {};
            let seq = ValueView::Sequence(&case.seq);
            let pos = Tensor::from(case.pos);
            let value = op.run_simple_no_cast((seq, pos.view()));
            assert_eq!(value, case.expected);
        });
    }

    #[test]
    fn test_sequence_construct() {
        #[derive(Debug)]
        struct Case {
            values: Vec<Value>,
            expected: Result<Sequence, OpError>,
        }

        let cases = [
            Case {
                values: [Tensor::from(1i32)].map(Value::from).into(),
                expected: Ok(Sequence::from([Tensor::from(1i32)])),
            },
            // We need at least one input to know what kind of sequence to
            // construct.
            Case {
                values: [].into(),
                expected: Err(OpError::MissingInputs),
            },
            Case {
                values: [
                    Value::from(Tensor::from(1i32)),
                    Value::from(Tensor::from(1.0)),
                ]
                .into(),
                expected: Err(OpError::CastFailed(CastError::WrongType {
                    actual: DataType::Float,
                    expected: DataType::Int32,
                })),
            },
        ];

        cases.test_each(|case| {
            let op = SequenceConstruct {};
            let mut inputs = InputList::new();
            for value in &case.values {
                inputs.push(value.as_view());
            }
            let result: Result<Sequence, _> = op.run_simple(inputs);
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_sequence_erase() {
        #[derive(Debug)]
        struct Case {
            seq: Sequence,
            pos: Option<i32>,
            expected: Result<Sequence, OpError>,
        }

        let test_seq: Sequence = [1., 2., 3.].map(Tensor::from).into();

        let cases = [
            // Removal via explicit position
            Case {
                seq: test_seq.clone(),
                pos: Some(0),
                expected: Ok([2., 3.].map(Tensor::from).into()),
            },
            // Removal via implicit position
            Case {
                seq: test_seq.clone(),
                pos: None,
                expected: Ok([1., 2.].map(Tensor::from).into()),
            },
            // Removal from non-empty sequence with invalid position
            Case {
                seq: test_seq.clone(),
                pos: Some(5),
                expected: Err(OpError::InvalidValue("Sequence position is invalid")),
            },
            // Removal from empty sequence
            Case {
                seq: Sequence::new(DataType::Int32),
                pos: None,
                expected: Err(OpError::InvalidValue(
                    "Cannot remove element from empty sequence",
                )),
            },
        ];

        cases.test_each(|case| {
            let op = SequenceErase {};
            let seq = ValueView::Sequence(&case.seq);
            let pos = case.pos.map(Tensor::from);
            let mut inputs = InputList::new();
            inputs.push(seq);
            inputs.push_optional(pos.as_ref().map(|p| p.view()));
            let new_seq: Result<Sequence, OpError> = op.run_simple(inputs);
            assert_eq!(new_seq, case.expected);
        });
    }

    #[test]
    fn test_sequence_insert() {
        #[derive(Debug)]
        struct Case {
            seq: Sequence,
            pos: Option<i32>,
            value: Value,
            expected: Result<Sequence, OpError>,
        }

        let test_seq: Sequence = [1., 2.].map(Tensor::from).into();
        let test_seq_extended: Sequence = [1., 2., 3.].map(Tensor::from).into();

        let cases = [
            // Insert at start
            Case {
                seq: test_seq.clone(),
                pos: Some(0),
                value: Tensor::from(0.).into(),
                expected: Ok([0., 1., 2.].map(Tensor::from).into()),
            },
            // Insert at end via explicit positive position.
            Case {
                seq: test_seq.clone(),
                pos: Some(2),
                value: Tensor::from(3.).into(),
                expected: Ok(test_seq_extended.clone()),
            },
            // Insert at end via explicit negative position.
            Case {
                seq: test_seq.clone(),
                pos: Some(-1),
                value: Tensor::from(3.).into(),
                expected: Ok(test_seq_extended.clone()),
            },
            // Insert at end via implicit position.
            Case {
                seq: test_seq.clone(),
                pos: None,
                value: Tensor::from(3.).into(),
                expected: Ok(test_seq_extended.clone()),
            },
            // Out-of-bounds index
            Case {
                seq: [1., 2.].map(Tensor::from).into(),
                pos: Some(5),
                value: Tensor::from(3.).into(),
                expected: Err(OpError::InvalidValue("Sequence position is invalid")),
            },
            // Data type mismatch
            Case {
                seq: [1., 2.].map(Tensor::from).into(),
                pos: Some(2),
                value: Tensor::from(3i32).into(),
                expected: Err(OpError::InvalidValue(
                    "Tensor type does not match sequence type",
                )),
            },
        ];

        cases.test_each(|case| {
            let op = SequenceInsert {};
            let seq = ValueView::Sequence(&case.seq);
            let pos = case.pos.map(Tensor::from);
            let mut inputs = InputList::new();
            inputs.push(seq);
            inputs.push(case.value.as_view());
            inputs.push_optional(pos.as_ref().map(|p| p.view()));
            let new_seq: Result<Sequence, OpError> = op.run_simple(inputs);
            assert_eq!(new_seq, case.expected);
        });
    }

    #[test]
    fn test_sequence_length() {
        let op = SequenceLength {};
        let seq = Value::from(Sequence::from([1i32, 2, 3].map(Tensor::from)));
        let result: Tensor<i32> = op.run_simple(seq.as_view()).unwrap();
        assert_eq!(result.item().copied(), Some(seq.len() as i32));
    }

    #[test]
    fn test_concat_from_sequence() {
        #[derive(Debug)]
        struct Case {
            seq: Sequence,
            axis: i32,
            new_axis: bool,
            expected: Result<Value, OpError>,
        }

        let cases = [
            // Concat along existing axis.
            Case {
                seq: [[0], [1], [2]].map(Tensor::from).into(),
                axis: 0,
                new_axis: false,
                expected: Ok(Tensor::from([0, 1, 2]).into()),
            },
            // Concat along new axis.
            Case {
                seq: [[0], [1], [2]].map(Tensor::from).into(),
                axis: 0,
                new_axis: true,
                expected: Ok(Tensor::from([[0], [1], [2]]).into()),
            },
            // Invalid axis
            Case {
                seq: [[0], [1], [2]].map(Tensor::from).into(),
                axis: 3,
                new_axis: true,
                expected: Err(OpError::InvalidValue("Axis is invalid")),
            },
        ];

        cases.test_each(|case| {
            let op = ConcatFromSequence {
                axis: case.axis,
                new_axis: case.new_axis,
            };
            let result = op.run_simple_no_cast(ValueView::Sequence(&case.seq));
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_split_to_sequence() {
        #[derive(Debug)]
        struct Case {
            input: Value,
            splits: Option<Tensor<i32>>,
            axis: i32,
            keep_dims: bool,
            expected: Result<Sequence, OpError>,
        }

        let cases = [
            // Scalar splits
            Case {
                input: Tensor::from([1, 2, 3, 4, 5]).into(),
                splits: Some(Tensor::from(2)),
                axis: 0,
                keep_dims: true,
                expected: Ok([
                    Tensor::from([1, 2]),
                    Tensor::from([3, 4]),
                    Tensor::from([5]),
                ]
                .into()),
            },
            // No splits, keep_dims=true
            Case {
                input: Tensor::from([1, 2, 3]).into(),
                splits: None,
                axis: 0,
                keep_dims: true,
                expected: Ok([[1], [2], [3]].map(Tensor::from).into()),
            },
            // No splits, keep_dims=false
            Case {
                input: Tensor::from([1, 2, 3]).into(),
                splits: None,
                axis: 0,
                keep_dims: false,
                expected: Ok([1, 2, 3].map(Tensor::from).into()),
            },
            // Vector splits
            Case {
                input: Tensor::from([1, 2, 3, 4]).into(),
                splits: Some(Tensor::from([3, 1])),
                axis: 0,
                keep_dims: true,
                expected: Ok([Tensor::from([1, 2, 3]), Tensor::from([4])].into()),
            },
            // Invalid split size
            Case {
                input: Tensor::from([1, 2, 3]).into(),
                splits: Some(Tensor::from(0)),
                axis: 0,
                keep_dims: true,
                expected: Err(OpError::InvalidValue("Split size must be >= 1")),
            },
            // Invalid split rank
            Case {
                input: Tensor::from([1, 2, 3]).into(),
                splits: Some(Tensor::from([[1]])),
                axis: 0,
                keep_dims: true,
                expected: Err(OpError::InvalidValue("Split size must be scalar or vector")),
            },
        ];

        cases.test_each(|case| {
            let op = SplitToSequence {
                axis: case.axis,
                keep_dims: case.keep_dims,
            };
            let mut inputs = InputList::new();
            inputs.push(case.input.as_view());
            if let Some(splits) = &case.splits {
                inputs.push(splits.view());
            }
            let result: Result<Sequence, _> = op.run_simple(inputs);
            assert_eq!(result, case.expected);
        });
    }
}
