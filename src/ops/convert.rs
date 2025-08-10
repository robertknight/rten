use rten_tensor::prelude::*;

use crate::buffer_pool::BufferPool;
use crate::ops::{
    DataType, IntoOpResult, OpError, OpRunContext, Operator, OutputList, Value, ValueView,
};

fn cast(pool: &BufferPool, input: ValueView, dtype: DataType) -> Result<Value, OpError> {
    macro_rules! cast_as {
        ($x:ident) => {
            Ok($x.to_tensor_in(pool).into())
        };

        ($x:ident, $dest_ty:ty) => {
            Ok($x.map_in(pool, |x| *x as $dest_ty).into())
        };
    }

    match dtype {
        DataType::Int32 => match input {
            ValueView::Int32Tensor(t) => cast_as!(t),
            ValueView::FloatTensor(t) => cast_as!(t, i32),
            ValueView::Int8Tensor(t) => cast_as!(t, i32),
            ValueView::UInt8Tensor(t) => cast_as!(t, i32),

            // The ONNX Cast op doesn't support sequences, although logically
            // this could be supported by casting each tensor in the sequence.
            ValueView::Sequence(_) => Err(OpError::UnsupportedType),
        },
        DataType::Float => match input {
            ValueView::FloatTensor(t) => cast_as!(t),
            ValueView::Int32Tensor(t) => cast_as!(t, f32),
            ValueView::Int8Tensor(t) => cast_as!(t, f32),
            ValueView::UInt8Tensor(t) => cast_as!(t, f32),
            ValueView::Sequence(_) => Err(OpError::UnsupportedType),
        },
        DataType::Int8 => match input {
            ValueView::Int8Tensor(t) => cast_as!(t),
            ValueView::FloatTensor(t) => cast_as!(t, i8),
            ValueView::Int32Tensor(t) => cast_as!(t, i8),
            ValueView::UInt8Tensor(t) => cast_as!(t, i8),
            ValueView::Sequence(_) => Err(OpError::UnsupportedType),
        },
        DataType::UInt8 => match input {
            ValueView::UInt8Tensor(t) => cast_as!(t),
            ValueView::FloatTensor(t) => cast_as!(t, u8),
            ValueView::Int32Tensor(t) => cast_as!(t, u8),
            ValueView::Int8Tensor(t) => cast_as!(t, u8),
            ValueView::Sequence(_) => Err(OpError::UnsupportedType),
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

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        cast(ctx.pool(), input, self.to).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        if input.dtype() == self.to {
            Ok(input)
        } else {
            let converted = cast(ctx.pool(), input.as_view(), self.to)?;
            input.add_to_pool(ctx.pool());
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

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let to_type = ctx.inputs().require(0)?.dtype();

        if input.dtype() == to_type {
            Ok(input)
        } else {
            let converted = cast(ctx.pool(), input.as_view(), to_type)?;
            input.add_to_pool(ctx.pool());
            Ok(converted)
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use crate::ops::{Cast, CastLike, DataType, OperatorExt, Value};

    #[test]
    fn test_cast() {
        #[derive(Debug)]
        struct Case {
            input: Value,
            dtype: DataType,
            expected: Value,
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
            let result: Value = cast_op.run_simple_no_cast(&case.input).unwrap();
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_cast_like() {
        #[derive(Debug)]
        struct Case {
            input: Value,
            other: Value,
            expected: Value,
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
