use std::borrow::Cow;

use rten_base::bit_set::BitSet;
use rten_base::byte_cast::FromByteArray;
use rten_base::num;

use rayon::prelude::*;
use rten_simd::SimdOp;
use rten_tensor::Tensor;
use rten_tensor::prelude::*;
use rten_vecmath as vecmath;

use crate::buffer_pool::BufferPool;
use crate::infer_shapes::{
    InferShapes, InferShapesContext, InferShapesError, SymTensor, SymbolGen, UnaryOp,
};
use crate::operator::{
    InPlaceInputs, IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType,
    OutputTypeList, OutputTypesContext,
};
use crate::value::{DataType, Value, ValueType, ValueView};

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

/// Cast a tensor from type T to U in-place.
///
/// T and U must have the same size and alignment.
fn cast_tensor<T, U>(mut data: Tensor<T>) -> Tensor<U>
where
    T: FromByteArray + num::Cast<U>,
    U: FromByteArray<Bytes = T::Bytes, Align = T::Align>,
{
    data.apply(|x| num::Cast::<U>::cast(*x).cast_bytes());
    data.bit_cast()
}

/// Cast elements of `input` to a given dtype in place, or return the input
/// value if the cast is not possible.
fn cast_in_place(input: Value, dtype: DataType) -> Result<Value, Value> {
    match dtype {
        DataType::Int32 => match input {
            Value::Int32Tensor(t) => Ok(t.into()),
            Value::FloatTensor(t) => Ok(cast_tensor::<_, i32>(t).into()),
            _ => Err(input),
        },
        DataType::Float => match input {
            Value::FloatTensor(t) => Ok(t.into()),
            Value::Int32Tensor(t) => Ok(cast_tensor::<_, f32>(t).into()),
            _ => Err(input),
        },
        DataType::Int8 => match input {
            Value::Int8Tensor(t) => Ok(t.into()),
            Value::UInt8Tensor(t) => Ok(cast_tensor::<_, i8>(t).into()),
            _ => Err(input),
        },
        DataType::UInt8 => match input {
            Value::UInt8Tensor(t) => Ok(t.into()),
            Value::Int8Tensor(t) => Ok(cast_tensor::<_, u8>(t).into()),
            _ => Err(input),
        },
    }
}

/// Target type of a [`Cast`] operator.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CastTarget {
    /// Convert to a data type which RTen supports natively.
    Type(DataType),

    /// Convert to `f32` and round to the precision of `f16`.
    ///
    /// RTen represents `f16` tensors as `f32`.
    F16,

    /// Convert to `f32` and round to the precision of `bf16`.
    ///
    /// RTen represents `bf16` tensors as `f32`.
    BF16,
}

impl CastTarget {
    /// Return the data type of the cast's output.
    pub fn dtype(&self) -> DataType {
        match self {
            Self::Type(dtype) => *dtype,
            Self::F16 | Self::BF16 => DataType::Float,
        }
    }

    /// Round each element of a float tensor to the precision of the target
    /// type.
    ///
    /// Values of other types are left unchanged.
    fn round(&self, pool: &BufferPool, value: Value) -> Value {
        let round_chunk: fn(&mut [f32]) = match self {
            Self::Type(_) => return value,
            Self::F16 => |chunk| {
                vecmath::RoundToF16::new(chunk).dispatch();
            },
            Self::BF16 => |chunk| {
                vecmath::RoundToBF16::new(chunk).dispatch();
            },
        };
        let Value::FloatTensor(mut t) = value else {
            return value;
        };
        t.make_contiguous_in(pool);
        t.data_mut()
            .expect("should be contiguous")
            .par_chunks_mut(ROUND_CHUNK_SIZE)
            .for_each(round_chunk);
        t.into()
    }
}

/// Minimum number of elements in a chunk that is rounded on a single thread.
const ROUND_CHUNK_SIZE: usize = 32 * 1024;

impl From<DataType> for CastTarget {
    fn from(dtype: DataType) -> Self {
        Self::Type(dtype)
    }
}

#[derive(Debug)]
pub struct Cast {
    pub to: CastTarget,
}

impl Operator for Cast {
    fn name(&self) -> &str {
        "Cast"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let output = cast(ctx.pool(), input, self.to.dtype())?;
        self.to.round(ctx.pool(), output).into_op_result()
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        // Cast can run in place if the input's dtype already matches `self.to`
        // or both dtypes have the same element size.
        BitSet::from_indices([0])
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let input = in_place.into_single();
        let output = match cast_in_place(input, self.to.dtype()) {
            Ok(output) => output,
            Err(input) => {
                let converted = cast(ctx.pool(), input.as_view(), self.to.dtype())?;
                input.add_to_pool(ctx.pool());
                converted
            }
        };
        self.to.round(ctx.pool(), output).into_op_result()
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(self.to.dtype()))].into())
    }
}

impl InferShapes for Cast {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;

        // If this is a no-op cast from int to int, preserve symbolic values.
        // Otherwise preserve just the shape like a generic unary operator.
        let value = if data.values().is_some() && self.to == CastTarget::Type(DataType::Int32) {
            data.clone()
        } else if let Some(shape) = data.shape() {
            SymTensor::from_shape(shape.collect())
        } else {
            SymTensor::unknown("unknown input shape")
        };

        Ok([value].into())
    }
}

#[derive(Debug)]
pub struct CastLike {}

impl Operator for CastLike {
    fn name(&self) -> &str {
        "CastLike"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let target_type = ctx.inputs().require(1)?;
        let ValueType::Tensor(to) = target_type.dtype() else {
            return Err(OpError::invalid_value(
                "expected target_type to be a tensor",
            ));
        };
        Cast { to: to.into() }.run(ctx)
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        BitSet::from_indices([0])
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let target_type = ctx.inputs().require(1)?;
        let ValueType::Tensor(to) = target_type.dtype() else {
            return Err(OpError::invalid_value(
                "expected target_type to be a tensor",
            ));
        };
        Cast { to: to.into() }.run_in_place(in_place, ctx)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(1)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }
}

const SIZE_MISMATCH: OpError = OpError::UnsupportedValue(Cow::Borrowed(
    "source and target types have different sizes",
));

/// Reinterpret the bits of a tensor as a different data type.
///
/// See <https://onnx.ai/onnx/operators/onnx__BitCast.html>.
#[derive(Debug)]
pub struct BitCast {
    pub to: DataType,
}

impl Operator for BitCast {
    fn name(&self) -> &str {
        "BitCast"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let bits = input
            .to_owned_in(ctx.pool())
            .into_bits()
            .ok_or(OpError::UnsupportedType)?;
        bits.into_value(self.to)
            .ok_or(SIZE_MISMATCH)
            .into_op_result()
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        BitSet::from_indices([0])
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        _ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let input = in_place.into_single();
        let bits = input.into_bits().ok_or(OpError::UnsupportedType)?;
        bits.into_value(self.to)
            .ok_or(SIZE_MISMATCH)
            .into_op_result()
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(self.to))].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use super::{BitCast, Cast, CastLike, CastTarget};
    use crate::operator::{InputList, OpError, OperatorExt};
    use crate::value::{DataType, Value, ValueType};

    #[test]
    fn test_bit_cast() {
        #[derive(Debug)]
        struct Case {
            input: Value,
            dtype: DataType,
            expected: Result<Value, OpError>,
        }

        let cases = [
            // f32 -> i32
            Case {
                input: Tensor::from([1.0f32, -1.0]).into(),
                dtype: DataType::Int32,
                expected: Ok(Tensor::from([0x3f80_0000i32, -0x4080_0000]).into()),
            },
            // i32 -> f32
            Case {
                input: Tensor::from([0x3f80_0000i32, -0x4080_0000]).into(),
                dtype: DataType::Float,
                expected: Ok(Tensor::from([1.0f32, -1.0]).into()),
            },
            // f32 -> f32
            Case {
                input: Tensor::from([1.0f32, -1.0]).into(),
                dtype: DataType::Float,
                expected: Ok(Tensor::from([1.0f32, -1.0]).into()),
            },
            // i8 -> u8
            Case {
                input: Tensor::from([-1i8, 0, 127]).into(),
                dtype: DataType::UInt8,
                expected: Ok(Tensor::from([255u8, 0, 127]).into()),
            },
            // u8 -> i8
            Case {
                input: Tensor::from([255u8, 0, 127]).into(),
                dtype: DataType::Int8,
                expected: Ok(Tensor::from([-1i8, 0, 127]).into()),
            },
            // Source and target types with different sizes.
            Case {
                input: Tensor::from([1.0f32]).into(),
                dtype: DataType::UInt8,
                expected: Err(OpError::unsupported_value(
                    "source and target types have different sizes",
                )),
            },
        ];

        cases.test_each(|case| {
            let op = BitCast { to: case.dtype };

            // Copying bit-cast.
            let result: Result<Value, OpError> = op.run_simple(&case.input);
            assert_eq!(result, case.expected);

            // In-place bit-cast.
            let result: Result<Value, OpError> =
                op.run_simple_in_place(case.input.clone(), InputList::new());
            assert_eq!(result, case.expected);
        })
    }

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
            // Copying cast.
            let cast_op = Cast {
                to: case.dtype.into(),
            };
            let result: Value = cast_op.run_simple(&case.input).unwrap();
            assert_eq!(result, case.expected);

            let input_dtype = match case.input.dtype() {
                ValueType::Tensor(dtype) => Some(dtype),
                _ => None,
            };

            // In-place cast.
            if input_dtype.unwrap().size() == case.dtype.size() {
                let result: Value = cast_op
                    .run_simple_in_place(case.input.clone(), InputList::new())
                    .unwrap();
                assert_eq!(result, case.expected);
            }
        })
    }

    #[test]
    fn test_cast_to_half_float() {
        #[derive(Debug)]
        struct Case {
            input: Value,
            to: CastTarget,
            expected: Tensor<f32>,
        }

        let cases = [
            // Float -> f16. Values are rounded to nearest, with ties to even.
            Case {
                input: Tensor::from([1.0f32, 1.0 + 2f32.powi(-11), 1.01, 65520., 1e-8]).into(),
                to: CastTarget::F16,
                expected: Tensor::from([1.0, 1.0, 1.0097656, f32::INFINITY, 0.]),
            },
            // Int -> f16.
            Case {
                input: Tensor::from([2048i32, 2049, 2051]).into(),
                to: CastTarget::F16,
                expected: Tensor::from([2048., 2048., 2052.]),
            },
            // Float -> bf16.
            Case {
                input: Tensor::from([1.0f32, 1.0 + 2f32.powi(-8), 1.01, -5.03]).into(),
                to: CastTarget::BF16,
                expected: Tensor::from([1.0, 1.0, 1.0078125, -5.03125]),
            },
            // Int -> bf16.
            Case {
                input: Tensor::from([256i32, 257, 259]).into(),
                to: CastTarget::BF16,
                expected: Tensor::from([256., 256., 260.]),
            },
        ];

        cases.test_each(|case| {
            let cast_op = Cast { to: case.to };
            let expected: Value = case.expected.clone().into();

            // Copying cast.
            let result: Value = cast_op.run_simple(&case.input).unwrap();
            assert_eq!(result, expected);

            // In-place cast.
            let result: Value = cast_op
                .run_simple_in_place(case.input.clone(), InputList::new())
                .unwrap();
            assert_eq!(result, expected);
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
            let result: Value = cast_op.run_simple((&case.input, &case.other)).unwrap();
            assert_eq!(result, case.expected);
        })
    }
}
