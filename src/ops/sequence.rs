use rten_tensor::Tensor;

use crate::ops::{resolve_index, IntoOpResult, OpError, OpRunContext, Operator, OutputList};
use crate::value::{DataType, Sequence, Value, ValueView};

#[derive(Debug)]
pub struct SequenceEmpty {
    pub dtype: Option<DataType>,
}

impl Operator for SequenceEmpty {
    fn name(&self) -> &str {
        "SequenceEmpty"
    }

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let seq: Sequence = match self.dtype {
            Some(DataType::Int32) => Vec::<Tensor<i32>>::new().into(),
            Some(DataType::Int8) => Vec::<Tensor<i8>>::new().into(),
            Some(DataType::UInt8) => Vec::<Tensor<u8>>::new().into(),
            None | Some(DataType::Float) => Vec::<Tensor<f32>>::new().into(),
        };
        Value::from(seq).into_op_result()
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
        seq.at(pos).unwrap().to_owned().into_op_result()
    }
}

fn sequence_insert(
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

    seq.insert(pos, val.to_owned()).unwrap();

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
        sequence_insert(seq.clone(), pos, value)
            .map(Value::from)
            .into_op_result()
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let seq: Sequence = input.try_into()?;
        let value = ctx.inputs().require(0)?;
        let pos: Option<i32> = ctx.inputs().get_as(1)?;
        sequence_insert(seq, pos, value).map(Value::from)
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use super::{SequenceAt, SequenceEmpty, SequenceInsert};
    use crate::ops::{InputList, OpError, OperatorExt};
    use crate::value::{DataType, Sequence, Value, ValueView};

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
}
