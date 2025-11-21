//! The `ops` module exposes the various operators available for machine-learning
//! models.
//!
//! Most operators correspond to an [ONNX
//! Operator](https://onnx.ai/onnx/operators/) of the same name, though RTen
//! does not support all ONNX operators, data types or attributes.
//!
//! Operators are primarily invoked by RTen as part of executing a
//! [Model](crate::Model), however they are also exposed as standalone
//! functions and tensor methods for use in code that pre-processes model
//! inputs and post-processes model outputs.

use std::fmt::Debug;

use smallvec::SmallVec;

use crate::operator::OpError;
use crate::value::DataType;

mod attention;
mod binary_elementwise;
mod concat;
mod control_flow;
mod conv;
mod conv_transpose;
mod convert;
mod einsum;
mod gather;
mod generate;
mod grid_sample;
mod identity;
mod layout;
mod matmul;
mod non_max_suppression;
mod norm;
mod pad;
mod pooling;
mod quantize;

#[cfg(feature = "fft")]
mod fft;

#[cfg(feature = "random")]
mod random;

mod reduce;
mod resize;
mod rnn;
mod sequence;
mod slice;
mod split;
mod trilu;
mod unary_elementwise;
mod variadic_elementwise;

// Fused operations
pub(crate) mod transform_inputs;

// Operator structs. These are re-exported for internal use by the model loader
// and tests.
#[cfg(feature = "fft")]
pub(crate) use fft::STFT;
#[cfg(feature = "random")]
pub(crate) use random::{
    Dropout, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike,
};
pub(crate) use {
    attention::{AddSoftmax, RepeatInterleave},
    binary_elementwise::{
        Add, And, Div, Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Mod, Mul, Or, Pow, Sub,
        Where, Xor,
    },
    concat::{Concat, Tile},
    control_flow::{If, Loop},
    conv::{Conv, ConvInteger},
    conv_transpose::ConvTranspose,
    convert::{Cast, CastLike},
    einsum::Einsum,
    gather::{Gather, GatherElements, GatherND, ScatterElements, ScatterND, ScatterReduction},
    generate::{ConstantOfShape, EyeLike, OneHot, Range},
    grid_sample::GridSample,
    identity::Identity,
    layout::{DepthToSpace, Expand, Flatten, Reshape, Shape, Size, Squeeze, Transpose, Unsqueeze},
    matmul::{
        AccuracyLevel, FusedMatMul, Gemm, MatMul, MatMulInteger, MatMulIntegerToFloat, MatMulNBits,
    },
    non_max_suppression::NonMaxSuppression,
    norm::{
        BatchNormalization, InstanceNormalization, LayerNormalization, LogSoftmax,
        RmsNormalization, Softmax,
    },
    pad::Pad,
    pooling::{AveragePool, GlobalAveragePool, GlobalMaxPool, MaxPool},
    quantize::{DequantizeLinear, DynamicQuantizeLinear, QuantizeLinear},
    reduce::{
        ArgMax, ArgMin, CumSum, NonZero, ReduceL2, ReduceMax, ReduceMean, ReduceMin, ReduceProd,
        ReduceSum, ReduceSumSquare, TopK,
    },
    resize::Resize,
    rnn::{GRU, LSTM},
    sequence::{
        ConcatFromSequence, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase,
        SequenceInsert, SequenceLength, SplitToSequence,
    },
    slice::Slice,
    split::Split,
    trilu::Trilu,
    unary_elementwise::{
        Abs, Acos, Asin, Atan, Ceil, Clip, Cos, Elu, Erf, Exp, Floor, Gelu, HardSigmoid, HardSwish,
        IsInf, IsNaN, LeakyRelu, Log, Neg, Not, PRelu, Reciprocal, Relu, Round, Sigmoid, Sign,
        Silu, Sin, Softplus, Sqrt, Swish, Tan, Tanh,
    },
    variadic_elementwise::{Max, Mean, Min, Sum},
};

// Operators as functions. These are exported for use by pre/post-processing
// code in applications. Some are also used internally in higher-level
// operators.
//
// These may be removed from the public API of the crate in future.
// See https://github.com/robertknight/rten/issues/911.
pub use binary_elementwise::{
    DivMode, add, and, div, equal, greater, greater_or_equal, less, less_or_equal, mod_op, mul, or,
    pow, sub, where_op, xor,
};
pub use concat::{concat, tile};
pub use conv::{conv, conv_integer};
pub use conv_transpose::conv_transpose;
pub use einsum::einsum;
pub use gather::{gather, gather_elements, gather_nd, scatter_elements, scatter_nd};
pub use generate::{constant_of_shape, onehot, range};
pub use layout::{DepthToSpaceMode, depth_to_space, expand, flatten, reshape, squeeze};
pub use matmul::{gemm, matmul};
pub use non_max_suppression::{BoxOrder, non_max_suppression};
pub use norm::{
    batch_norm, instance_normalization, layer_normalization, log_softmax, rms_normalization,
    softmax,
};
pub use pad::{PadMode, pad};
pub use pooling::{average_pool, global_average_pool, max_pool};
pub use quantize::{dequantize_linear, dynamic_quantize_linear, quantize_linear};

#[cfg(feature = "fft")]
pub use fft::stft;

pub use reduce::{
    arg_max, arg_min, cum_sum, nonzero, reduce_l2, reduce_max, reduce_mean, reduce_min,
    reduce_prod, reduce_sum, reduce_sum_square, topk,
};
pub use resize::{CoordTransformMode, NearestMode, ResizeMode, ResizeTarget, resize, resize_image};
pub use rnn::{Direction, gru, lstm};
pub use slice::slice;
pub use split::split;
pub use trilu::trilu;
pub use variadic_elementwise::{max, mean, min, sum};

mod operators;
pub use operators::{FloatOperators, Operators};

#[derive(Clone, Debug, PartialEq)]
pub enum Padding {
    /// Apply enough padding such that the output and input have the same size.
    ///
    /// If the required amount of padding along each dimension is even, it is
    /// divided equally between the start and the end. If it is odd, one more
    /// unit is added on the end than the start. This matches the ONNX spec
    /// for the "SAME_UPPER" value for the `auto_pad` attribute.
    Same,

    /// Apply a given amount of padding to each side of the input. Paddings
    /// are specified in the order `[start, end]` for 1D padding,
    /// `[top, left, bottom, right]` for 2D and so on.
    Fixed(SmallVec<[usize; 4]>),
}

impl Padding {
    /// Return fixed zero padding for an N-dimensional shape.
    pub fn zero<const N: usize>() -> Padding {
        Padding::Fixed(SmallVec::from_elem(0, N * 2))
    }

    /// Expand padding for a 1D operation to 2D.
    pub fn expand_1d_to_2d(&self) -> Result<Padding, OpError> {
        match self {
            Padding::Same => Ok(Padding::Same),
            Padding::Fixed(pads) => match pads.as_slice() {
                &[pad_start, pad_end] => Ok([0, pad_start, 0, pad_end].into()),
                _ => Err(OpError::InvalidValue("expected 2 pad values")),
            },
        }
    }
}

/// Construct a [`Padding::Fixed`] from a slice of paddings for each size.
impl<S: AsRef<[usize]>> From<S> for Padding {
    fn from(val: S) -> Padding {
        Padding::Fixed(val.as_ref().into())
    }
}

/// Resolve an index given as a value in `[-len, len-1]` to a positive index in
/// `[0, len)`, or return None if the index is out of bounds.
fn resolve_index(len: usize, index: isize) -> Option<usize> {
    let len = len as isize;
    if index < -len || index >= len {
        return None;
    }

    if index >= 0 {
        Some(index as usize)
    } else {
        Some((len + index) as usize)
    }
}

/// Resolve an axis given as a value in `[-ndim, ndim-1]` to the zero-based
/// dimension of a tensor with `ndim` dimensions.
///
/// Negative axis values count backwards from the last dimension.
fn resolve_axis(ndim: usize, axis: isize) -> Result<usize, OpError> {
    resolve_index(ndim, axis).ok_or(OpError::InvalidValue("Axis is invalid"))
}

/// Resolve a sequence of axes values in `[-ndim, ndim-1]` to zero-based dimension
/// indexes in a tensor with `ndim` dimensions.
///
/// Negative axis values count backwards from the last dimension.
pub fn resolve_axes<'a, I: ExactSizeIterator<Item = &'a i32>>(
    ndim: usize,
    axes: I,
) -> Result<SmallVec<[usize; 4]>, OpError> {
    let mut resolved_axes = SmallVec::with_capacity(axes.len());
    for axis in axes {
        let resolved = resolve_axis(ndim, *axis as isize)?;
        resolved_axes.push(resolved);
    }
    Ok(resolved_axes)
}

/// Extract a typed tensor view from a [`ValueView`] and pass it to a block.
///
/// The result of the macro is the result of the block, hence the block must
/// return a value of the same type regardless of the input type. This result
/// type must be a `Result<_, OpError>`.
///
/// A list of supported tensor types can optionally be specified, as a list of
/// [`ValueView`] variant names.
///
/// Only tensor types are currently supported. For sequence types this always
/// returns an error.
macro_rules! map_value_view {
    ($input:expr, $typed_input:ident, $block:tt) => {
        match $input {
            ValueView::FloatTensor($typed_input) => $block,
            ValueView::Int32Tensor($typed_input) => $block,
            ValueView::UInt8Tensor($typed_input) => $block,
            ValueView::Int8Tensor($typed_input) => $block,
            ValueView::Sequence(_) => Err(OpError::UnsupportedType)
        }
    };

    ($input:expr, $typed_input:ident, [$($variant:ident),+], $block:tt) => {
            match $input {
                $(ValueView::$variant($typed_input) => $block),+,
                _ => {
                    return Err(OpError::UnsupportedType);
                }
            }
    };
}

use map_value_view;

/// Evaluate a block with a type alias defined that matches a [`DataType`].
///
/// For example if `$dtype` is [`DataType::Int32`] then the block will be
/// evaluated with a type named `$type` in scope which is an alias for `i32`.
macro_rules! map_dtype {
    ($dtype:expr, $type:ident, $block:tt) => {{
        use $crate::ops::DataType;

        match $dtype {
            DataType::Int32 => {
                type $type = i32;
                $block
            }
            DataType::Float => {
                type $type = f32;
                $block
            }
            DataType::UInt8 => {
                type $type = u8;
                $block
            }
            DataType::Int8 => {
                type $type = i8;
                $block
            }
        }
    }};
}

use map_dtype;

/// Extract a typed owned tensor from a [`Value`] and pass it to a block.
///
/// The result of the macro is the result of the block, hence the block must
/// return a value of the same type regardless of the input type. This result
/// type must be a `Result<_, OpError>`.
///
/// A list of supported tensor types can optionally be specified, as a list of
/// [`Value`] variant names.
///
/// Only tensor types are currently supported. For sequence types this always
/// returns an error.
macro_rules! map_value {
    ($input:expr, $typed_input:ident, $block:tt) => {
        match $input {
            #[allow(unused_mut)]
            Value::FloatTensor(mut $typed_input) => $block,
            #[allow(unused_mut)]
            Value::Int32Tensor(mut $typed_input) => $block,
            #[allow(unused_mut)]
            Value::UInt8Tensor(mut $typed_input) => $block,
            #[allow(unused_mut)]
            Value::Int8Tensor(mut $typed_input) => $block,
            Value::Sequence(_) => Err(OpError::UnsupportedType),
        }
    };

    ($input:expr, $typed_input:ident, [$($variant:ident),+], $block:tt) => {
            match $input {
                $(
                    #[allow(unused_mut)]
                    Value::$variant(mut $typed_input) => $block
                ),+,
                _ => {
                    return Err(OpError::UnsupportedType);
                }
            }
    };
}

use map_value;

/// Check that an operator input or attribute is valid or return an [`OpError`]
/// if not.
///
/// This is similar to [`assert`] but it returns an error instead of panicking
/// if the condition evaluates to false.
macro_rules! check_value {
    ($condition:expr, $err_variant:ident, $err_msg:expr) => {
        if !$condition {
            return Err(OpError::$err_variant($err_msg));
        }
    };
}

use check_value;

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;
    use rten_tensor::test_util::{ExpectEqualError, expect_equal_with_tolerance};

    /// Compare two f32 tensors with a higher absolute tolerance (1e-4) than
    /// the default (1e-5).
    ///
    /// Tests that use this generally ought to use a lower tolerance, but
    /// their test expectations will often need updating to a higher precision.
    pub fn expect_eq_1e4<V: AsView<Elem = f32>>(
        result: &V,
        expected: &V,
    ) -> Result<(), ExpectEqualError> {
        expect_equal_with_tolerance(result, expected, 1e-4, 0.)
    }

    /// Increase the rank of a tensor by inserting leading 1-sized dimensions.
    pub trait IntoNDim<const N: usize> {
        /// Variant of `Self` with N dimensions.
        type Output;

        /// Insert leading 1-sized dimensions into the shape of `self` so that
        /// it has N dimensions.
        ///
        /// Panics if `self` already has more than N dimensions.
        fn into_ndim(self) -> Self::Output;
    }

    impl<T: Clone, const M: usize, const N: usize> IntoNDim<N> for NdTensor<T, M> {
        type Output = NdTensor<T, N>;

        fn into_ndim(self) -> Self::Output {
            assert!(N >= M);
            let new_dims = N - M;
            let shape = self.shape();
            let new_shape =
                std::array::from_fn(|d| if d < new_dims { 1 } else { shape[d - new_dims] });
            self.into_shape(new_shape)
        }
    }
}
