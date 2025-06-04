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
//! inputs and post-processes model outputs. Some standalone operator functions
//! come into two flavors, one which operates in-place on an existing tensor,
//! and one which takes a view as input and returns a new tensor as output.

use std::any::Any;
use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};

use smallvec::SmallVec;

use rten_tensor::prelude::*;
use rten_tensor::{
    DynLayout, MutLayout, NdTensor, NdTensorView, Storage, Tensor, TensorBase, TensorView, ViewData,
};

use crate::downcast::impl_downcastdyn;
use crate::gemm::PackedBMatrix;
use crate::graph::{CaptureEnv, Graph, RunError, RunOptions};
use crate::tensor_pool::{ExtractBuffer, TensorPool};
use crate::timing::Profiler;
use crate::weight_cache::WeightCache;

mod binary_elementwise;
mod concat;
mod control_flow;
mod conv;
mod convert;
mod einsum;
mod gather;
mod generate;
mod identity;
mod layout;
mod matmul;
mod non_max_suppression;
mod norm;
mod pad;
mod pooling;
mod quantize;

#[cfg(feature = "random")]
mod random;

mod reduce;
mod resize;
mod rnn;
mod slice;
mod split;
mod trilu;
mod unary_elementwise;
mod variadic_elementwise;

// Fused operations
pub(crate) mod transform_inputs;

pub use binary_elementwise::{
    add, and, div, equal, greater, greater_or_equal, less, less_or_equal, mod_op, mul, or, pow,
    sub, where_op, xor, Add, And, Div, DivMode, Equal, Greater, GreaterOrEqual, Less, LessOrEqual,
    Mod, Mul, Or, Pow, Sub, Where, Xor,
};
pub use concat::{concat, tile, Concat, Tile};
pub use control_flow::If;
pub use conv::{conv, conv_integer, conv_transpose, Conv, ConvInteger, ConvTranspose};
pub use convert::{Cast, CastLike};
pub use einsum::{einsum, Einsum};
pub use gather::{
    gather, gather_elements, gather_nd, scatter_elements, scatter_nd, Gather, GatherElements,
    GatherND, ScatterElements, ScatterND, ScatterReduction,
};
pub use generate::{constant_of_shape, onehot, range, ConstantOfShape, OneHot, Range};
pub use identity::Identity;
pub use layout::{
    depth_to_space, expand, flatten, reshape, squeeze, DepthToSpace, DepthToSpaceMode, Expand,
    Flatten, Reshape, Shape, Size, Squeeze, Transpose, Unsqueeze,
};
pub use matmul::{gemm_op, matmul, FusedMatMul, Gemm, MatMul, MatMulInteger};
pub use non_max_suppression::{non_max_suppression, BoxOrder, NonMaxSuppression};
pub use norm::{
    batch_norm, instance_normalization, layer_normalization, log_softmax, rms_normalization,
    softmax, BatchNormalization, InstanceNormalization, LayerNormalization, LogSoftmax,
    RmsNormalization, Softmax,
};
pub use pad::{pad, Pad, PadMode};
pub use pooling::{
    average_pool, global_average_pool, max_pool, AveragePool, GlobalAveragePool, MaxPool,
};
pub use quantize::{
    dequantize_linear, dynamic_quantize_linear, quantize_linear, DequantizeLinear,
    DynamicQuantizeLinear, QuantizeLinear,
};

#[cfg(feature = "random")]
pub use random::{Dropout, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike};

pub use reduce::{
    arg_max, arg_min, cum_sum, nonzero, reduce_l2, reduce_max, reduce_mean, reduce_min,
    reduce_prod, reduce_sum, reduce_sum_square, topk, ArgMax, ArgMin, CumSum, NonZero, ReduceL2,
    ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, TopK,
};
pub use resize::{
    resize, resize_image, CoordTransformMode, NearestMode, Resize, ResizeMode, ResizeTarget,
};
pub use rnn::{gru, lstm, Direction, GRU, LSTM};
pub use slice::{slice, Slice};
pub use split::{split, Split};
pub use trilu::{trilu, Trilu};
pub use unary_elementwise::{
    abs, acos, asin, atan, ceil, clip, cos, elu, erf, exp, floor, gelu, hard_sigmoid, hard_swish,
    leaky_relu, log, neg, not, reciprocal, relu, round, sigmoid, sign, silu, sin, softplus, sqrt,
    swish, tan, tanh, Abs, Acos, Asin, Atan, Ceil, Clip, Cos, Elu, Erf, Exp, Floor, Gelu,
    HardSigmoid, HardSwish, LeakyRelu, Log, Neg, Not, Reciprocal, Relu, Round, Sigmoid, Sign, Silu,
    Sin, Softplus, Sqrt, Swish, Tan, Tanh,
};
pub use variadic_elementwise::{max, mean, min, sum, Max, Mean, Min, Sum};

mod operators;
pub use operators::{FloatOperators, Operators};

#[derive(Clone, Debug)]
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

/// Enum specifying the data type of a tensor.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DataType {
    Int32,
    Float,
    Int8,
    UInt8,
}

/// Get the [`DataType`] that corresponds to a given type.
pub trait DataTypeOf {
    /// Return the data type that corresponds to the `Self` type.
    fn dtype_of() -> DataType;
}

macro_rules! impl_data_type_of {
    ($type:ty, $dtype:ident) => {
        impl DataTypeOf for $type {
            fn dtype_of() -> DataType {
                DataType::$dtype
            }
        }
    };
}

impl_data_type_of!(f32, Float);
impl_data_type_of!(i32, Int32);
impl_data_type_of!(i8, Int8);
impl_data_type_of!(u8, UInt8);

impl std::fmt::Display for DataType {
    /// Format this enum value in the style of the corresponding Rust type (eg.
    /// "i32" for `DataType::Int32`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DataType::Float => "f32",
                DataType::Int32 => "i32",
                DataType::Int8 => "i8",
                DataType::UInt8 => "u8",
            }
        )
    }
}

/// Generate the body of a [`Layout`] impl for a type which wraps an
/// underlying layout.
macro_rules! impl_proxy_layout {
    () => {
        type Index<'b> = <DynLayout as Layout>::Index<'b>;
        type Indices = <DynLayout as Layout>::Indices;

        fn ndim(&self) -> usize {
            self.layout().ndim()
        }

        fn try_offset(&self, index: Self::Index<'_>) -> Option<usize> {
            self.layout().try_offset(index)
        }

        fn len(&self) -> usize {
            self.layout().len()
        }

        fn is_empty(&self) -> bool {
            self.layout().is_empty()
        }

        fn shape(&self) -> Self::Index<'_> {
            self.layout().shape()
        }

        fn size(&self, dim: usize) -> usize {
            self.layout().size(dim)
        }

        fn strides(&self) -> Self::Index<'_> {
            self.layout().strides()
        }

        fn stride(&self, dim: usize) -> usize {
            self.layout().stride(dim)
        }

        fn indices(&self) -> Self::Indices {
            self.layout().indices()
        }
    };
}

/// Errors when casting a [`Value`] or [`ValueView`] to a tensor of a specific
/// type and/or rank.
#[derive(Debug, Eq, PartialEq)]
pub enum CastError {
    /// The number of dimensions does not match.
    WrongRank { actual: usize, expected: usize },

    /// The data type of elements does not match.
    WrongType {
        actual: DataType,
        expected: DataType,
    },
}

impl Display for CastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongRank { actual, expected } => {
                write!(
                    f,
                    "expected tensor with {} dims must has {} dims",
                    expected, actual
                )
            }
            Self::WrongType { actual, expected } => {
                write!(
                    f,
                    "expected tensor with type {} but has type {}",
                    expected, actual
                )
            }
        }
    }
}

impl Error for CastError {}

impl From<CastError> for OpError {
    fn from(val: CastError) -> OpError {
        OpError::CastFailed(val)
    }
}

/// Metadata about a tensor.
///
/// This is used in profiling and errors which need to contain metadata about
/// a tensor but not the content.
#[derive(Debug, Eq, PartialEq)]
pub struct ValueMeta {
    pub(crate) dtype: DataType,
    pub(crate) shape: Vec<usize>,
}

impl Display for ValueMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Produces strings such as "f32 [1, 16, 256]"
        write!(f, "{} {:?}", self.dtype, self.shape)
    }
}

/// A view of a tensor with runtime-determined type and rank.
///
/// This type is used for operator inputs.
#[derive(Clone)]
pub enum ValueView<'a> {
    FloatTensor(TensorView<'a, f32>),
    Int32Tensor(TensorView<'a, i32>),
    Int8Tensor(TensorView<'a, i8>),
    UInt8Tensor(TensorView<'a, u8>),
}

impl ValueView<'_> {
    /// Return the data type of elements in this tensor.
    pub fn dtype(&self) -> DataType {
        match self {
            Self::FloatTensor(_) => DataType::Float,
            Self::Int32Tensor(_) => DataType::Int32,
            Self::Int8Tensor(_) => DataType::Int8,
            Self::UInt8Tensor(_) => DataType::UInt8,
        }
    }

    pub fn to_owned(&self) -> Value {
        match self {
            ValueView::FloatTensor(t) => t.to_tensor().into(),
            ValueView::Int32Tensor(t) => t.to_tensor().into(),
            ValueView::Int8Tensor(t) => t.to_tensor().into(),
            ValueView::UInt8Tensor(t) => t.to_tensor().into(),
        }
    }

    /// Extract shape and data type information from this tensor.
    pub fn to_meta(&self) -> ValueMeta {
        ValueMeta {
            shape: self.shape().to_vec(),
            dtype: self.dtype(),
        }
    }

    fn layout(&self) -> &DynLayout {
        match self {
            ValueView::FloatTensor(t) => t.layout(),
            ValueView::Int32Tensor(t) => t.layout(),
            ValueView::Int8Tensor(t) => t.layout(),
            ValueView::UInt8Tensor(t) => t.layout(),
        }
    }
}

impl Layout for ValueView<'_> {
    impl_proxy_layout!();
}

macro_rules! impl_input_conversions {
    ($variant:ident, $element_type:ty) => {
        impl<'a> TryFrom<ValueView<'a>> for TensorView<'a, $element_type> {
            type Error = CastError;

            fn try_from(
                input: ValueView<'a>,
            ) -> Result<TensorView<'a, $element_type>, Self::Error> {
                match input {
                    ValueView::$variant(t) => Ok(t),
                    _ => Err(CastError::WrongType {
                        actual: input.dtype(),
                        expected: <$element_type as DataTypeOf>::dtype_of(),
                    }),
                }
            }
        }

        impl<'a, const N: usize> TryFrom<ValueView<'a>> for NdTensorView<'a, $element_type, N> {
            type Error = CastError;

            fn try_from(
                input: ValueView<'a>,
            ) -> Result<NdTensorView<'a, $element_type, N>, Self::Error> {
                let ndim = input.ndim();
                match input {
                    ValueView::$variant(t) => t.try_into().map_err(|_| CastError::WrongRank {
                        actual: ndim,
                        expected: N,
                    }),
                    _ => Err(CastError::WrongType {
                        actual: input.dtype(),
                        expected: <$element_type as DataTypeOf>::dtype_of(),
                    }),
                }
            }
        }

        impl<'a> TryFrom<ValueView<'a>> for $element_type {
            type Error = CastError;

            fn try_from(input: ValueView<'a>) -> Result<$element_type, Self::Error> {
                let tensor: TensorView<'a, _> = input.try_into()?;
                tensor.item().copied().ok_or(CastError::WrongRank {
                    actual: tensor.ndim(),
                    expected: 0,
                })
            }
        }

        impl<'a> From<&'a Tensor<$element_type>> for ValueView<'a> {
            fn from(t: &'a Tensor<$element_type>) -> ValueView<'a> {
                ValueView::$variant(t.view())
            }
        }

        impl<'a> From<TensorView<'a, $element_type>> for ValueView<'a> {
            fn from(t: TensorView<'a, $element_type>) -> ValueView<'a> {
                ValueView::$variant(t)
            }
        }

        impl<'a, const N: usize> From<NdTensorView<'a, $element_type, N>> for ValueView<'a> {
            fn from(t: NdTensorView<'a, $element_type, N>) -> ValueView<'a> {
                ValueView::$variant(t.as_dyn())
            }
        }
    };
}

impl_input_conversions!(FloatTensor, f32);
impl_input_conversions!(Int32Tensor, i32);
impl_input_conversions!(Int8Tensor, i8);
impl_input_conversions!(UInt8Tensor, u8);

impl<'a> From<&'a Value> for ValueView<'a> {
    fn from(output: &'a Value) -> ValueView<'a> {
        match output {
            Value::FloatTensor(t) => ValueView::FloatTensor(t.view()),
            Value::Int32Tensor(t) => ValueView::Int32Tensor(t.view()),
            Value::Int8Tensor(t) => ValueView::Int8Tensor(t.view()),
            Value::UInt8Tensor(t) => ValueView::UInt8Tensor(t.view()),
        }
    }
}

/// An operator input which has been pre-packed for more efficient use during
/// inference.
pub enum PrepackedInput {
    /// Prepacked RHS / B input for matrix multiplication with f32 weights.
    FloatBMatrix(PackedBMatrix<f32>),

    /// Prepacked RHS / B input for matrix multiplication with i8 weights.
    Int8BMatrix(PackedBMatrix<i8>),
}

impl PrepackedInput {
    fn dtype(&self) -> DataType {
        match self {
            Self::FloatBMatrix(_) => DataType::Float,
            Self::Int8BMatrix(_) => DataType::Int8,
        }
    }
}

macro_rules! impl_prepacked_input_conversions {
    ($type:ty, $variant:ident) => {
        impl From<PackedBMatrix<$type>> for PrepackedInput {
            fn from(value: PackedBMatrix<$type>) -> Self {
                PrepackedInput::$variant(value)
            }
        }

        impl<'a> TryFrom<&'a PrepackedInput> for &'a PackedBMatrix<$type> {
            type Error = CastError;

            fn try_from(ppi: &'a PrepackedInput) -> Result<Self, Self::Error> {
                match ppi {
                    PrepackedInput::$variant(packed) => Ok(packed),
                    _ => Err(CastError::WrongType {
                        actual: ppi.dtype(),
                        expected: <$type as DataTypeOf>::dtype_of(),
                    }),
                }
            }
        }
    };
}
impl_prepacked_input_conversions!(f32, FloatBMatrix);
impl_prepacked_input_conversions!(i8, Int8BMatrix);

/// An owned tensor with runtime-determined type and rank.
///
/// This value is used to represent operator outputs.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    FloatTensor(Tensor<f32>),
    Int32Tensor(Tensor<i32>),
    Int8Tensor(Tensor<i8>),
    UInt8Tensor(Tensor<u8>),
}

impl Value {
    /// Return the data type of elements in this tensor.
    pub fn dtype(&self) -> DataType {
        match self {
            Self::FloatTensor(_) => DataType::Float,
            Self::Int32Tensor(_) => DataType::Int32,
            Self::Int8Tensor(_) => DataType::Int8,
            Self::UInt8Tensor(_) => DataType::UInt8,
        }
    }

    /// Return a borrowed view of this tensor.
    pub fn as_view(&self) -> ValueView {
        match self {
            Self::FloatTensor(ft) => ValueView::FloatTensor(ft.view()),
            Self::Int32Tensor(it) => ValueView::Int32Tensor(it.view()),
            Self::Int8Tensor(it) => ValueView::Int8Tensor(it.view()),
            Self::UInt8Tensor(it) => ValueView::UInt8Tensor(it.view()),
        }
    }

    /// Extract shape and data type information from this tensor.
    pub fn to_meta(&self) -> ValueMeta {
        ValueMeta {
            shape: self.shape().to_vec(),
            dtype: self.dtype(),
        }
    }

    /// Move this tensor's buffer into a pool.
    pub(crate) fn add_to_pool(self, pool: &TensorPool) {
        match self {
            Self::FloatTensor(t) => t.extract_buffer().map(|buf| pool.add(buf)),
            Self::Int32Tensor(t) => t.extract_buffer().map(|buf| pool.add(buf)),
            Self::Int8Tensor(t) => t.extract_buffer().map(|buf| pool.add(buf)),
            Self::UInt8Tensor(t) => t.extract_buffer().map(|buf| pool.add(buf)),
        };
    }

    /// Convert this output into a tensor with a given element type.
    ///
    /// Returns `None` if the element type does not match `T`.
    pub fn into_tensor<T>(self) -> Option<Tensor<T>>
    where
        Tensor<T>: TryFrom<Self>,
    {
        self.try_into().ok()
    }

    /// Convert a reference to this output into a tensor view with a given
    /// element type.
    ///
    /// Returns `None` if the element type does not match `T`.
    pub fn as_tensor_view<'a, T>(&'a self) -> Option<TensorView<'a, T>>
    where
        TensorView<'a, T>: TryFrom<&'a Self>,
    {
        self.try_into().ok()
    }

    fn layout(&self) -> &DynLayout {
        match self {
            Value::Int32Tensor(t) => t.layout(),
            Value::Int8Tensor(t) => t.layout(),
            Value::UInt8Tensor(t) => t.layout(),
            Value::FloatTensor(t) => t.layout(),
        }
    }
}

impl Layout for Value {
    impl_proxy_layout!();
}

/// Declare conversions between `Value` and `Tensor<T>` / `NdTensor<T, N>`.
macro_rules! impl_value_conversions {
    ($variant:ident, $element_type:ty) => {
        // Tensor<T> => Value
        impl From<Tensor<$element_type>> for Value {
            fn from(t: Tensor<$element_type>) -> Value {
                Value::$variant(t)
            }
        }

        // NdTensor<T> => Value
        impl<const N: usize> From<NdTensor<$element_type, N>> for Value {
            fn from(t: NdTensor<$element_type, N>) -> Value {
                Value::$variant(t.into_dyn())
            }
        }

        // Value => Tensor<T>
        impl TryFrom<Value> for Tensor<$element_type> {
            type Error = CastError;

            fn try_from(o: Value) -> Result<Tensor<$element_type>, Self::Error> {
                let dtype = o.dtype();
                match o {
                    Value::$variant(t) => Ok(t),
                    _ => Err(CastError::WrongType {
                        actual: dtype,
                        expected: <$element_type as DataTypeOf>::dtype_of(),
                    }),
                }
            }
        }

        // Value => NdTensor<T, N>
        impl<const N: usize> TryFrom<Value> for NdTensor<$element_type, N> {
            type Error = CastError;

            fn try_from(o: Value) -> Result<NdTensor<$element_type, N>, CastError> {
                let tensor: Tensor<_> = o.try_into()?;
                let ndim = tensor.ndim();
                tensor.try_into().map_err(|_| CastError::WrongRank {
                    actual: ndim,
                    expected: N,
                })
            }
        }

        // Value => TensorView<T>
        impl<'a> TryFrom<&'a Value> for TensorView<'a, $element_type> {
            type Error = CastError;

            fn try_from(o: &'a Value) -> Result<TensorView<'a, $element_type>, CastError> {
                match o {
                    Value::$variant(t) => Ok(t.view()),
                    _ => Err(CastError::WrongType {
                        actual: o.dtype(),
                        expected: <$element_type as DataTypeOf>::dtype_of(),
                    }),
                }
            }
        }

        // Value => NdTensorView<T, N>
        impl<'a, const N: usize> TryFrom<&'a Value> for NdTensorView<'a, $element_type, N> {
            type Error = CastError;

            fn try_from(o: &'a Value) -> Result<NdTensorView<'a, $element_type, N>, CastError> {
                let view: TensorView<'a, _> = o.try_into()?;
                let ndim = view.ndim();
                view.try_into().map_err(|_| CastError::WrongRank {
                    actual: ndim,
                    expected: N,
                })
            }
        }
    };
}

impl_value_conversions!(FloatTensor, f32);
impl_value_conversions!(Int32Tensor, i32);
impl_value_conversions!(Int8Tensor, i8);
impl_value_conversions!(UInt8Tensor, u8);

/// A value that is either a tensor view ([`ValueView`]) or an owned tensor
/// ([`Value`]).
#[derive(Clone)]
pub enum ValueOrView<'a> {
    /// A tensor view (like a slice)
    View(ValueView<'a>),
    /// An owned tensor (like a `Vec<T>`)
    Value(Value),
}

impl ValueOrView<'_> {
    /// Convert this value to a tensor view.
    pub fn as_view(&self) -> ValueView {
        match self {
            ValueOrView::View(inp) => inp.clone(),
            ValueOrView::Value(outp) => outp.as_view(),
        }
    }

    /// Convert this value to an owned tensor.
    pub fn to_owned(&self) -> Value {
        match self {
            ValueOrView::View(inp) => inp.to_owned(),
            ValueOrView::Value(outp) => outp.clone(),
        }
    }

    pub fn layout(&self) -> &DynLayout {
        match self {
            Self::View(inp) => inp.layout(),
            Self::Value(outp) => outp.layout(),
        }
    }
}

impl<'a> From<ValueView<'a>> for ValueOrView<'a> {
    fn from(val: ValueView<'a>) -> Self {
        ValueOrView::View(val)
    }
}

impl<'a, T: 'static, S: Storage<Elem = T>, L: MutLayout> From<&'a TensorBase<S, L>>
    for ValueOrView<'a>
where
    ValueView<'a>: From<TensorView<'a, T>>,
{
    fn from(val: &'a TensorBase<S, L>) -> Self {
        ValueOrView::View(val.as_dyn().into())
    }
}

impl<'a, T, L: MutLayout> From<TensorBase<ViewData<'a, T>, L>> for ValueOrView<'a>
where
    ValueView<'a>: From<TensorView<'a, T>>,
{
    fn from(val: TensorBase<ViewData<'a, T>, L>) -> Self {
        ValueOrView::View(val.as_dyn().into())
    }
}

impl<T, L: MutLayout> From<TensorBase<Vec<T>, L>> for ValueOrView<'static>
where
    Value: From<Tensor<T>>,
    DynLayout: From<L>,
{
    fn from(val: TensorBase<Vec<T>, L>) -> Self {
        ValueOrView::Value(val.into_dyn().into())
    }
}

impl From<Value> for ValueOrView<'static> {
    fn from(val: Value) -> Self {
        ValueOrView::Value(val)
    }
}

impl<'a> From<&'a Value> for ValueOrView<'a> {
    fn from(val: &'a Value) -> Self {
        let inp: ValueView<'a> = ValueView::from(val);
        inp.into()
    }
}

impl Layout for ValueOrView<'_> {
    impl_proxy_layout!();
}

/// Trait for values that can be converted into the result type used by
/// [`Operator::run`].
pub trait IntoOpResult {
    fn into_op_result(self) -> Result<OutputList, OpError>;
}

impl IntoOpResult for Result<Value, OpError> {
    fn into_op_result(self) -> Result<OutputList, OpError> {
        self.map(|out| [out].into())
    }
}

impl IntoOpResult for Value {
    fn into_op_result(self) -> Result<OutputList, OpError> {
        Ok([self].into())
    }
}

impl<S: Storage, L: MutLayout> IntoOpResult for TensorBase<S, L>
where
    Value: From<TensorBase<S, L>>,
{
    fn into_op_result(self) -> Result<OutputList, OpError> {
        let output: Value = self.into();
        Ok([output].into())
    }
}

impl<S: Storage, L: MutLayout> IntoOpResult for Result<TensorBase<S, L>, OpError>
where
    Value: From<TensorBase<S, L>>,
{
    fn into_op_result(self) -> Result<OutputList, OpError> {
        self.map(|tensor| [tensor.into()].into())
    }
}

impl<T> IntoOpResult for Result<Vec<T>, OpError>
where
    Value: From<T>,
{
    fn into_op_result(self) -> Result<OutputList, OpError> {
        self.map(|tensors| tensors.into_iter().map(|t| t.into()).collect())
    }
}

/// Possible reasons why an operator may fail on a given input.
#[derive(Eq, PartialEq, Debug)]
pub enum OpError {
    /// Casting a tensor to an expected type or rank failed.
    CastFailed(CastError),

    /// Casting an input to an expected type or rank failed.
    InputCastFailed { index: usize, error: CastError },

    /// A tensor has an unsupported type.
    UnsupportedType,

    /// Input tensor shapes are not compatible with each other or operator
    /// attributes.
    IncompatibleInputShapes(&'static str),

    /// The number of inputs was less than the required number.
    MissingInputs,

    /// An input has a value that is incorrect.
    InvalidValue(&'static str),

    /// An input or attribute has a value that is valid, but not currently supported.
    UnsupportedValue(&'static str),
}

impl Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::CastFailed(err) => write!(f, "{}", err),
            OpError::InputCastFailed { index, error } => {
                write!(f, "conversion error for input {}: {}", index, error)
            }
            OpError::IncompatibleInputShapes(details) => {
                write!(f, "incompatible input shapes: {}", details)
            }
            OpError::MissingInputs => write!(f, "required inputs were missing"),
            OpError::InvalidValue(details) => {
                write!(f, "input or attribute has invalid value: {}", details)
            }
            OpError::UnsupportedValue(details) => {
                write!(f, "unsupported input or attribute value: {}", details)
            }
            OpError::UnsupportedType => {
                write!(f, "unsupported input type")
            }
        }
    }
}

impl Error for OpError {}

/// Convert a tensor with dynamic dimension count to a view with a static
/// dimension count.
///
/// If the conversion fails an `OpError::InvalidValue` error will be returned
/// with a message that includes the name of the tensor and, optionally, the
/// names of the expected dimensions (eg. "NCHW").
macro_rules! static_dims {
    ($tensor:ident, $ndim:literal, $dim_names:literal) => {{
        use rten_tensor::prelude::*;

        if $tensor.ndim() != $ndim {
            Err(OpError::InvalidValue(concat!(
                stringify!($tensor),
                " must have ",
                stringify!($ndim),
                " dims (",
                $dim_names,
                ")"
            )))
        } else {
            Ok($tensor.nd_view::<$ndim>())
        }
    }};

    ($tensor:ident, $ndim:literal) => {{
        use rten_tensor::prelude::*;

        if $tensor.ndim() != $ndim {
            Err(OpError::InvalidValue(concat!(
                stringify!($tensor),
                " must have ",
                stringify!($ndim),
                " dims"
            )))
        } else {
            Ok($tensor.nd_view::<$ndim>())
        }
    }};

    ($tensor:ident?, $ndim: expr) => {
        if let Some($tensor) = $tensor.as_ref() {
            Some(static_dims!($tensor, $ndim))
        } else {
            None
        }
    };
}

pub(crate) use static_dims;

/// Context passed to [`Operator::run`] containing the information needed for
/// the operator to execute.
pub struct OpRunContext<'a, 'i> {
    pool: &'a TensorPool,
    inputs: &'a InputList<'i>,
    n_outputs: Option<u32>,
    name: Option<&'a str>,
}

impl<'a, 'i> OpRunContext<'a, 'i> {
    pub fn new(pool: &'a TensorPool, inputs: &'a InputList<'i>) -> Self {
        OpRunContext {
            pool,
            inputs,
            n_outputs: None,
            name: None,
        }
    }

    /// The pool which should be used to allocate large buffers.
    pub fn pool(&self) -> &TensorPool {
        self.pool
    }

    /// Inputs to the operator execution.
    ///
    /// For in-place execution via [`Operator::run_in_place`] this contains
    /// the non in-place inputs.
    pub fn inputs(&self) -> &InputList<'i> {
        self.inputs
    }

    /// Set the requested number of outputs.
    ///
    /// This can be used to skip generating outputs that are unused, or in
    /// the rare cases that the output count cannot be determined from the
    /// operator's inputs and attributes alone.
    pub fn set_num_outputs(&mut self, n: u32) {
        self.n_outputs = Some(n);
    }

    /// Return the number of requested outputs or `None` if this has not been
    /// specified.
    pub fn num_outputs(&self) -> Option<u32> {
        self.n_outputs
    }

    /// Set the name of the current node in the graph.
    pub fn set_name(&mut self, name: Option<&'a str>) {
        self.name = name;
    }

    /// Return the name of the current node in the graph.
    pub fn name(&self) -> Option<&str> {
        self.name
    }
}

/// Outputs from an operator.
///
/// This avoids allocations in the common case where an operator produces
/// exactly one output.
pub type OutputList = SmallVec<[Value; 1]>;

/// An Operator performs a computation step when executing a data flow graph.
///
/// Operators take zero or more dynamic input values, plus a set of static
/// attributes and produce one or more output values.
///
/// Operators are usually named after the ONNX operator that they implement.
/// See <https://onnx.ai/onnx/operators/>.
pub trait Operator: Any + Debug {
    /// Return a display name for the operator.
    fn name(&self) -> &str;

    /// Execute the operator.
    ///
    /// `ctx` provides access to operator inputs and the [`TensorPool`] from
    /// which the output and temporary buffers should be allocated.
    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError>;

    /// Return true if this operator supports in-place execution via
    /// `run_in_place`.
    ///
    /// In-place execution returns results by modifying an existing tensor
    /// instead of allocating a new one. Reducing memory allocations can
    /// significantly speed up graph runs.
    fn can_run_in_place(&self) -> bool {
        false
    }

    /// Return true if this operator is commutative, meaning that its inputs
    /// can be re-ordered without affecting the result.
    ///
    /// If true, the graph executor may swap inputs before calling the
    /// [`Operator::run_in_place`] implementation.
    fn is_commutative(&self) -> bool {
        false
    }

    /// Return true if this operator's outputs depend only on its inputs.
    ///
    /// The default implementation returns true, since most operators are
    /// deterministic. Operators such as random number generators however are
    /// not.
    ///
    /// The definition of _deterministic_ used here excludes minor differences
    /// due to eg. the order in which results from parallel sub-problems are
    /// accumulated. It also does not guarantee exact consistency across devices.
    fn is_deterministic(&self) -> bool {
        true
    }

    /// Execute this operator in-place on an existing tensor.
    ///
    /// This may only be called if `can_run_in_place` returns true.
    ///
    /// `input` is the first input, which the implementation may modify and
    /// return as the output. `ctx.inputs()` contains the remaining inputs.
    ///
    /// Operators may fall back to allocating a new output if some property of
    /// the input data or shapes means in-place operation is not possible. In
    /// this case they should return the input buffer to the pool, and allocate
    /// the new output buffer from it. The pool should also be used for any
    /// temporary buffers created during execution.
    fn run_in_place(
        &self,
        #[allow(unused)] input: Value,
        #[allow(unused)] ctx: &OpRunContext,
    ) -> Result<Value, OpError> {
        Err(OpError::InvalidValue("In-place execution not supported"))
    }

    /// Return true if this operator executes a subgraph.
    fn has_subgraph(&self) -> bool {
        !self.subgraphs().is_empty()
    }

    /// Return a list of subgraphs used by this operator.
    fn subgraphs(&self) -> SmallVec<[&Graph; 2]> {
        SmallVec::new()
    }

    /// Return the IDs of inputs which can be pre-packed using [`prepack`](Operator::prepack).
    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        SmallVec::new()
    }

    /// Pre-pack an input for more efficient inference later.
    ///
    /// `index` specifies the input ID and should be one of the inputs returned
    /// by [`prepack_inputs`](Operator::prepack_inputs).
    fn prepack(
        &self,
        #[allow(unused)] index: usize,
        #[allow(unused)] input: ValueView,
    ) -> Option<PrepackedInput> {
        None
    }

    /// Execute the operator with the given inputs and captured values.
    ///
    /// This method will be called instead of `run` if the operator reports that
    /// it runs a subgraph (see [`has_subgraph`](Operator::has_subgraph)).
    /// Compared to `run`, it takes an additional `captures` argument which
    /// provides access to values captured from the surrounding scope (like a
    /// closure in Rust) and it returns a [`RunError`] instead of an
    /// [`OpError`].
    ///
    /// The default implementation delegates to `run`. In other words it treats
    /// the operator as a subgraph with a single node.
    fn run_subgraph<'a>(
        &'a self,
        ctx: &OpRunContext,
        #[allow(unused)] captures: CaptureEnv,
        #[allow(unused)] weight_cache: Option<&[WeightCache]>,
        #[allow(unused)] profiler: Option<&mut Profiler<'a>>,
        #[allow(unused)] run_opts: Option<RunOptions>,
    ) -> Result<OutputList, RunError> {
        self.run(ctx)
            .map_err(|error| RunError::op_error(self.name(), error, ctx))
    }
}

impl_downcastdyn!(Operator);

/// Convenience methods that make it easier to run operators in tests.
pub trait OperatorExt: Operator {
    /// Run an operator and extract the first output as a tensor with a given
    /// type.
    ///
    /// `inputs` is a tuple of tensor references or other values that can be
    /// converted to [`ValueView`].
    fn run_simple<'a, I: Into<InputList<'a>>, O: TryFrom<Value, Error = CastError>>(
        &self,
        inputs: I,
    ) -> Result<O, OpError> {
        let result = self.run_simple_no_cast(inputs)?;
        let typed_result = result.try_into()?;
        Ok(typed_result)
    }

    /// Run an operator and extract the first output.
    fn run_simple_no_cast<'a, I: Into<InputList<'a>>>(&self, inputs: I) -> Result<Value, OpError> {
        let pool = TensorPool::new();
        let inputs = inputs.into();
        let ctx = OpRunContext::new(&pool, &inputs);
        let mut outputs = self.run(&ctx)?;
        Ok(outputs.remove(0))
    }

    fn run_simple_in_place<I: Into<Value>, O: TryFrom<Value, Error = CastError>>(
        &self,
        input: I,
    ) -> Result<O, OpError> {
        let pool = TensorPool::new();
        let inputs = InputList::new();
        let ctx = OpRunContext::new(&pool, &inputs);
        let output = self.run_in_place(input.into(), &ctx)?;
        let typed_output = output.try_into()?;
        Ok(typed_output)
    }
}

impl<O: ?Sized + Operator> OperatorExt for O {}

/// List of inputs for an operator evaluation.
///
/// Conceptually this is a `Cow<[Option<ValueView>]>` with methods to conveniently
/// extract inputs and produce appropriate errors if inputs are missing or of
/// the wrong type.
///
/// An InputList can be constructed from a tensor reference or tuple of tensor
/// references using `into`.
#[derive(Clone)]
pub struct InputList<'a> {
    inputs: Cow<'a, [Option<ValueView<'a>>]>,

    /// Callback that retrieves the pre-packed copy of an input with a given
    /// index.
    get_prepacked: Option<&'a dyn Fn(usize) -> Option<&'a PrepackedInput>>,

    /// True if the input list does not contain the first operator input because
    /// it is being passed separately. In this case input indices are offset by
    /// one (eg. `inputs.require(0)` will return the second input to the operator).
    first_input_omitted: bool,
}

impl<'a> InputList<'a> {
    /// Construct an empty input list.
    pub fn new() -> InputList<'a> {
        InputList {
            inputs: Cow::Owned(vec![]),
            get_prepacked: None,
            first_input_omitted: false,
        }
    }

    /// Mark this input list as not containing the first input to the operator.
    ///
    /// This is used together with [`Operator::run_in_place`] where the first
    /// input is passed separately. When this flag is set the input index is
    /// adjusted in errors to reflect the real index.
    pub fn with_first_input_omitted(mut self, offset: bool) -> Self {
        self.first_input_omitted = offset;
        self
    }

    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Append an input to the list.
    ///
    /// This will copy the existing inputs into a new owned vector.
    pub fn push<I: Into<ValueView<'a>>>(&mut self, inp: I) {
        self.inputs.to_mut().push(Some(inp.into()))
    }

    /// Append an optional input to the list.
    ///
    /// This will copy the existing inputs into a new owned vector.
    pub fn push_optional<I: Into<ValueView<'a>>>(&mut self, inp: Option<I>) {
        self.inputs.to_mut().push(inp.map(|inp| inp.into()))
    }

    /// Construct an input list from a slice of non-optional inputs.
    ///
    /// This copies the inputs into a new vector of `Option<ValueView>`s. Using
    /// [`from_optional`](Self::from_optional) is more efficient.
    pub fn from(inputs: &[ValueView<'a>]) -> InputList<'a> {
        InputList {
            inputs: inputs.iter().cloned().map(Some).collect(),
            get_prepacked: None,
            first_input_omitted: false,
        }
    }

    /// Construct an input list from a slice of optional inputs.
    ///
    /// This is a cheap conversion that borrows `inputs`.
    pub fn from_optional(inputs: &'a [Option<ValueView<'a>>]) -> InputList<'a> {
        InputList {
            inputs: Cow::Borrowed(inputs),
            get_prepacked: None,
            first_input_omitted: false,
        }
    }

    /// Configure a callback that will get or create a pre-packed copy of the
    /// input with a given index.
    pub fn with_prepacked(
        mut self,
        lookup: &'a dyn Fn(usize) -> Option<&'a PrepackedInput>,
    ) -> Self {
        self.get_prepacked = Some(lookup);
        self
    }

    /// Get an optional input.
    pub fn get(&self, index: usize) -> Option<ValueView<'a>> {
        self.inputs.get(index).cloned().flatten()
    }

    /// Get the pre-packed version of a weight input, if available.
    pub fn get_prepacked(&self, index: usize) -> Option<&'a PrepackedInput> {
        self.get_prepacked.and_then(|gp| gp(index))
    }

    /// Get a mutable reference to an input.
    ///
    /// This will convert the list into an owned list of inputs first.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ValueView<'a>> {
        self.inputs.to_mut().get_mut(index)?.as_mut()
    }

    /// Convert an optional input into a tensor or scalar.
    pub fn get_as<T>(&self, index: usize) -> Result<Option<T>, OpError>
    where
        T: TryFrom<ValueView<'a>, Error = CastError>,
    {
        self.get(index)
            .map(|input| {
                input.try_into().map_err(|error| OpError::InputCastFailed {
                    index: self.to_real_index(index),
                    error,
                })
            })
            .transpose()
    }

    /// Get a required operator input.
    pub fn require(&self, index: usize) -> Result<ValueView<'a>, OpError> {
        self.get(index).ok_or(OpError::MissingInputs)
    }

    /// Convert a required input into a tensor or scalar.
    pub fn require_as<T>(&self, index: usize) -> Result<T, OpError>
    where
        T: TryFrom<ValueView<'a>, Error = CastError>,
    {
        self.require(index).and_then(|input| {
            input.try_into().map_err(|error| OpError::InputCastFailed {
                index: self.to_real_index(index),
                error,
            })
        })
    }

    /// Return an iterator over provided inputs.
    ///
    /// Use [`Iterator::flatten`] to skip missing optional inputs.
    pub fn iter<'b>(&'b self) -> impl Iterator<Item = Option<ValueView<'a>>> + 'b {
        self.inputs.iter().cloned()
    }

    /// Map an index into this input list back to an index in the full
    /// sequence of operator inputs.
    fn to_real_index(&self, index: usize) -> usize {
        if self.first_input_omitted {
            index + 1
        } else {
            index
        }
    }
}

impl Default for InputList<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, I: Into<ValueView<'a>>> From<I> for InputList<'a> {
    fn from(val: I) -> InputList<'a> {
        InputList::from(&[val.into()])
    }
}

impl<'a, I1: Into<ValueView<'a>>, I2: Into<ValueView<'a>>> From<(I1, I2)> for InputList<'a> {
    fn from((a, b): (I1, I2)) -> InputList<'a> {
        InputList::from(&[a.into(), b.into()])
    }
}

impl<'a, I1: Into<ValueView<'a>>, I2: Into<ValueView<'a>>, I3: Into<ValueView<'a>>>
    From<(I1, I2, I3)> for InputList<'a>
{
    fn from((a, b, c): (I1, I2, I3)) -> InputList<'a> {
        InputList::from(&[a.into(), b.into(), c.into()])
    }
}

#[derive(Debug)]
pub enum Scalar {
    Int(i32),
    Float(f32),
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
/// return a value of the same type regardless of the input type.
///
/// A list of supported tensor types can optionally be specified, as a list of
/// [`ValueView`] variant names.
macro_rules! map_value_view {
    ($input:expr, $typed_input:ident, $block:tt) => {
        match $input {
            ValueView::FloatTensor($typed_input) => $block,
            ValueView::Int32Tensor($typed_input) => $block,
            ValueView::UInt8Tensor($typed_input) => $block,
            ValueView::Int8Tensor($typed_input) => $block,
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

/// Extract a typed owned tensor from a [`Value`] and pass it to a block.
///
/// The result of the macro is the result of the block, hence the block must
/// return a value of the same type regardless of the input type.
///
/// A list of supported tensor types can optionally be specified, as a list of
/// [`Value`] variant names.
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

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::test_util::{expect_equal_with_tolerance, ExpectEqualError};
    use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};

    use super::{CastError, Operator, Value, ValueView};
    use crate::downcast::DowncastDyn;
    use crate::ops::{Add, DataType, InputList, OpError, Sub};
    use crate::tensor_pool::TensorPool;

    /// Create an empty tensor pool.
    ///
    /// This is a wrapper that provides a place to customize the behavior of
    /// the pool in tests.
    pub fn new_pool() -> TensorPool {
        TensorPool::new()
    }

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

    #[test]
    fn test_input_from_tensor() {
        let tensor = NdTensor::<i32, 3>::zeros([1, 2, 3]);
        let input: ValueView = tensor.view().into();
        assert!(matches!(input, ValueView::Int32Tensor(_)));
        assert_eq!(input.shape(), &[1, 2, 3]);

        let tensor = NdTensor::<f32, 2>::zeros([5, 5]);
        let input: ValueView = tensor.view().into();
        assert!(matches!(input, ValueView::FloatTensor(_)));
        assert_eq!(input.shape(), &[5, 5]);
    }

    #[test]
    fn test_input_list_first_input_omitted() {
        let tensor = Tensor::<f32>::zeros(&[2, 2]);

        let inputs = InputList::from(&[tensor.view().into()]).with_first_input_omitted(false);
        let err = inputs.require_as::<TensorView<i32>>(0).err().unwrap();
        assert!(matches!(err, OpError::InputCastFailed { index: 0, .. }));

        let inputs = InputList::from(&[tensor.view().into()]).with_first_input_omitted(true);
        let err = inputs.require_as::<TensorView<i32>>(0).err().unwrap();
        assert!(matches!(err, OpError::InputCastFailed { index: 1, .. }));
    }

    #[test]
    fn test_tensor_from_output() {
        let original = NdTensor::from([[1., 2.], [3., 4.]]);
        let output: Value = original.clone().into();

        let mat_dyn: Tensor<f32> = output.clone().try_into().unwrap();
        assert_eq!(mat_dyn, original);

        let mat: NdTensor<f32, 2> = output.clone().try_into().unwrap();
        assert_eq!(mat, original);

        let err: Result<NdTensor<i32, 2>, _> = output.clone().try_into();
        assert_eq!(
            err,
            Err(CastError::WrongType {
                actual: DataType::Float,
                expected: DataType::Int32,
            })
        );

        let err: Result<NdTensor<f32, 3>, _> = output.clone().try_into();
        assert_eq!(
            err,
            Err(CastError::WrongRank {
                actual: 2,
                expected: 3
            })
        );
    }

    #[test]
    fn test_tensor_view_from_output() {
        let original = NdTensor::from([[1., 2.], [3., 4.]]);
        let output: Value = original.clone().into();

        let mat_dyn: TensorView<f32> = (&output).try_into().unwrap();
        assert_eq!(mat_dyn, original);

        let mat: NdTensorView<f32, 2> = (&output).try_into().unwrap();
        assert_eq!(mat, original);

        let err: Result<NdTensorView<i32, 2>, _> = (&output).try_into();
        assert_eq!(
            err,
            Err(CastError::WrongType {
                actual: DataType::Float,
                expected: DataType::Int32,
            })
        );

        let err: Result<NdTensorView<f32, 3>, _> = (&output).try_into();
        assert_eq!(
            err,
            Err(CastError::WrongRank {
                actual: 2,
                expected: 3
            })
        );
    }

    #[test]
    fn test_downcast_operator() {
        let add_op = Add {};
        let sub_op = Sub {};

        let add_op_dyn: &dyn Operator = &add_op;
        let sub_op_dyn: &dyn Operator = &sub_op;

        assert!(add_op_dyn.downcast_ref::<Add>().is_some());
        assert!(sub_op_dyn.downcast_ref::<Sub>().is_some());
    }
}
