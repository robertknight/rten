use rten_tensor::prelude::*;

use crate::ops::{
    DataType, Input, InputList, IntoOpResult, OpError, OpRunContext, Operator, Output, OutputList,
};
use crate::tensor_pool::TensorPool;

fn cast(pool: &TensorPool, input: Input, dtype: DataType) -> Result<Output, OpError> {
    macro_rules! cast_as {
        ($x:ident) => {
            $x.to_tensor_in(pool).into()
        };

        ($x:ident, $dest_ty:ty) => {
            $x.map_in(pool, |x| *x as $dest_ty).into()
        };
    }

    let result: Output = match dtype {
        DataType::Int32 => match input {
            Input::Int32Tensor(t) => cast_as!(t),
            Input::FloatTensor(t) => cast_as!(t, i32),
            Input::Int8Tensor(t) => cast_as!(t, i32),
            Input::UInt8Tensor(t) => cast_as!(t, i32),
        },
        DataType::Float => match input {
            Input::FloatTensor(t) => cast_as!(t),
            Input::Int32Tensor(t) => cast_as!(t, f32),
            Input::Int8Tensor(t) => cast_as!(t, f32),
            Input::UInt8Tensor(t) => cast_as!(t, f32),
        },
        DataType::Int8 => match input {
            Input::Int8Tensor(t) => cast_as!(t),
            Input::FloatTensor(t) => cast_as!(t, i8),
            Input::Int32Tensor(t) => cast_as!(t, i8),
            Input::UInt8Tensor(t) => cast_as!(t, i8),
        },
        DataType::UInt8 => match input {
            Input::UInt8Tensor(t) => cast_as!(t),
            Input::FloatTensor(t) => cast_as!(t, u8),
            Input::Int32Tensor(t) => cast_as!(t, u8),
            Input::Int8Tensor(t) => cast_as!(t, u8),
        },
    };

    Ok(result)
}

#[derive(Debug)]
pub struct Cast {
    pub to: DataType,
}

impl Operator for Cast {
    fn name(&self) -> &str {
        "Cast"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        cast(ctx.pool(), input, self.to).into_op_result()
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
        if input.dtype() == self.to {
            Ok(input)
        } else {
            let converted = cast(pool, input.as_input(), self.to)?;
            input.add_to_pool(pool);
            Ok(converted)
        }
    }
}

#[derive(Debug)]
pub struct CastLike {}

impl Operator for CastLike {
    fn name(&self) -> &str {
        "CastLike"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let to_type = inputs.require(1)?.dtype();
        cast(ctx.pool(), input, to_type).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        let to_type = other.require(0)?.dtype();

        if input.dtype() == to_type {
            Ok(input)
        } else {
            let converted = cast(pool, input.as_input(), to_type)?;
            input.add_to_pool(pool);
            Ok(converted)
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use crate::ops::{Cast, CastLike, DataType, OperatorExt, Output};

    #[test]
    fn test_cast() {
        #[derive(Debug)]
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

        cases.test_each(|case| {
            let cast_op = Cast { to: case.dtype };
            let result: Output = cast_op.run_simple_no_cast(&case.input).unwrap();
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_cast_like() {
        #[derive(Debug)]
        struct Case {
            input: Output,
            other: Output,
            expected: Output,
        }

        // `CastLike` uses the same conversions as the `Cast` operator,
        // so these tests don't check all data type combinations, only that the
        // target type is taken from the second argument.
        let cases = [
            // i32 -> f32
            Case {
                input: Tensor::from([0i32, 1, 2]).into(),
                other: Tensor::from([0f32]).into(),
                expected: Tensor::from([0., 1., 2.]).into(),
            },
        ];

        cases.test_each(|case| {
            let cast_op = CastLike {};
            let result = cast_op
                .run_simple_no_cast((&case.input, &case.other))
                .unwrap();
            assert_eq!(result, case.expected);
        })
    }
}
