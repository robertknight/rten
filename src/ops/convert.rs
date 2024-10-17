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

    use rten_tensor::test_util::expect_equal;
    use rten_tensor::Tensor;

    use crate::ops::tests::new_pool;
    use crate::ops::{Cast, DataType, Operator};

    #[test]
    fn test_cast() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let int_input = Tensor::from([1, 2, 3]);
        let float_input = Tensor::from([1.0, 2.0, 3.0]);

        // No-op cast from int32 => int32
        let cast_to_int = Cast {
            to: DataType::Int32,
        };
        let result = cast_to_int
            .run(&pool, (&int_input).into())
            .unwrap()
            .remove(0)
            .into_tensor::<i32>()
            .unwrap();

        // Flooring cast from float => int32
        assert_eq!(result, int_input);
        let result = cast_to_int
            .run(&pool, (&float_input).into())
            .unwrap()
            .remove(0)
            .into_tensor::<i32>()
            .unwrap();
        assert_eq!(&result, &int_input);

        // No-op cast from float => float
        let cast_to_float = Cast {
            to: DataType::Float,
        };
        let result = cast_to_float
            .run(&pool, (&float_input).into())
            .unwrap()
            .remove(0)
            .into_tensor::<f32>()
            .unwrap();
        expect_equal(&result, &float_input)?;

        // Cast from int32 => float
        let result = cast_to_float
            .run(&pool, (&int_input).into())
            .unwrap()
            .remove(0)
            .into_tensor::<f32>()
            .unwrap();
        expect_equal(&result, &float_input)?;

        Ok(())
    }

    #[test]
    fn test_cast_out_of_range() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let int_input = Tensor::from([i32::MIN, i32::MAX]);

        // Out-of-range cast from int => float. This will simply lose some
        // significant digits.
        let cast_to_float = Cast {
            to: DataType::Float,
        };
        let result = cast_to_float
            .run(&pool, (&int_input).into())
            .unwrap()
            .remove(0)
            .into_tensor::<f32>()
            .unwrap();
        expect_equal(&result, &Tensor::from([-2147483600.0, 2147483600.0]))?;

        // Out-of-range cast from float => int.
        let float_input = Tensor::from([f32::MIN, f32::MAX]);
        let cast_to_int = Cast {
            to: DataType::Int32,
        };
        let result = cast_to_int
            .run(&pool, (&float_input).into())
            .unwrap()
            .remove(0)
            .into_tensor::<i32>()
            .unwrap();
        assert_eq!(&result, &Tensor::from([i32::MIN, i32::MAX]));

        Ok(())
    }
}
