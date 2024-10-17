use rten_tensor::prelude::*;

use crate::ops::{DataType, Input, InputList, IntoOpResult, OpError, Operator, Output, OutputList};
use crate::tensor_pool::TensorPool;

fn cast(pool: &TensorPool, input: Input, dtype: DataType) -> Result<Output, OpError> {
    match dtype {
        DataType::Int32 => match input {
            Input::Int32Tensor(t) => Ok(t.to_tensor_in(pool).into()),
            Input::FloatTensor(t) => Ok(t.map_in(pool, |x| *x as i32).into()),
            Input::Int8Tensor(t) => Ok(t.map_in(pool, |x| *x as i32).into()),
            Input::UInt8Tensor(t) => Ok(t.map_in(pool, |x| *x as i32).into()),
        },
        DataType::Float => match input {
            Input::FloatTensor(t) => Ok(t.to_tensor_in(pool).into()),
            Input::Int32Tensor(t) => Ok(t.map_in(pool, |x| *x as f32).into()),
            Input::Int8Tensor(t) => Ok(t.map_in(pool, |x| *x as f32).into()),
            Input::UInt8Tensor(t) => Ok(t.map_in(pool, |x| *x as f32).into()),
        },
        DataType::Int8 => match input {
            Input::FloatTensor(t) => Ok(t.map_in(pool, |x| *x as i8).into()),
            Input::Int32Tensor(t) => Ok(t.map_in(pool, |x| *x as i8).into()),
            Input::Int8Tensor(t) => Ok(t.to_tensor_in(pool).into()),
            Input::UInt8Tensor(t) => Ok(t.map_in(pool, |x| *x as i8).into()),
        },
        DataType::UInt8 => match input {
            Input::FloatTensor(t) => Ok(t.map_in(pool, |x| *x as u8).into()),
            Input::Int32Tensor(t) => Ok(t.map_in(pool, |x| *x as u8).into()),
            Input::Int8Tensor(t) => Ok(t.map_in(pool, |x| *x as u8).into()),
            Input::UInt8Tensor(t) => Ok(t.to_tensor_in(pool).into()),
        },
    }
}

#[derive(Debug)]
pub struct Cast {
    pub to: DataType,
}

impl Operator for Cast {
    fn name(&self) -> &str {
        "Cast"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        cast(pool, input, self.to).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        _: InputList,
    ) -> Result<Output, OpError> {
        match (input, self.to) {
            (Output::Int32Tensor(t), DataType::Int32) => Ok(t.into()),
            (Output::FloatTensor(t), DataType::Float) => Ok(t.into()),
            (Output::Int8Tensor(t), DataType::Int8) => Ok(t.into()),
            (Output::UInt8Tensor(t), DataType::UInt8) => Ok(t.into()),
            (input, _) => {
                let converted = cast(pool, input.as_input(), self.to)?;
                input.add_to_pool(pool);
                Ok(converted)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::Tensor;

    use crate::ops::tests::new_pool;
    use crate::ops::{Cast, DataType, Operator, Output};

    #[test]
    fn test_cast() -> Result<(), Box<dyn Error>> {
        struct Case {
            input: Output,
            dtype: DataType,
            expected: Output,
        }

        let cases = [
            // i32 -> f32
            Case {
                input: Tensor::from([1, 2, 3]).into(),
                dtype: DataType::Float,
                expected: Tensor::from([1., 2., 3.]).into(),
            },
            // i32 -> i32
            Case {
                input: Tensor::from([1, 2, 3]).into(),
                dtype: DataType::Int32,
                expected: Tensor::from([1, 2, 3]).into(),
            },
            // i32 -> i8
            Case {
                input: Tensor::from([i8::MIN as i32, 0, i8::MAX as i32]).into(),
                dtype: DataType::Int8,
                expected: Tensor::from([i8::MIN, 0, i8::MAX]).into(),
            },
            // i32 -> u8
            Case {
                input: Tensor::from([u8::MIN as i32, 0, u8::MAX as i32]).into(),
                dtype: DataType::UInt8,
                expected: Tensor::from([u8::MIN, 0, u8::MAX]).into(),
            },
            // f32 -> i32
            Case {
                input: Tensor::from([1., 2., 3.]).into(),
                dtype: DataType::Int32,
                expected: Tensor::from([1, 2, 3]).into(),
            },
            // f32 -> f32
            Case {
                input: Tensor::from([1., 2., 3.]).into(),
                dtype: DataType::Float,
                expected: Tensor::from([1., 2., 3.]).into(),
            },
            // Int -> float out of range. This will lose precision.
            Case {
                input: Tensor::from([i32::MIN, i32::MAX]).into(),
                dtype: DataType::Float,
                expected: Tensor::from([-2147483600.0, 2147483600.0]).into(),
            },
            // Float -> int out of range.
            //
            // In RTen this saturates following the behavior of Rust's `as`
            // operator. This is different than C++ / PyTorch / NumPy where
            // the behavior of such conversions is undefined.
            // See https://github.com/robertknight/rten/pull/387#issuecomment-2420343989.
            Case {
                input: Tensor::from([f32::MIN, f32::MAX]).into(),
                dtype: DataType::Int32,
                expected: Tensor::from([i32::MIN, i32::MAX]).into(),
            },
        ];

        let pool = new_pool();
        for Case {
            input,
            dtype,
            expected,
        } in cases
        {
            let cast_op = Cast { to: dtype };
            let result = cast_op.run(&pool, (&input).into()).unwrap().remove(0);
            assert_eq!(result, expected);
        }

        Ok(())
    }
}
