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

// Fused operators.
pub(crate) mod fused;

pub use binary_elementwise::{
    add, add_in_place, and, div, div_in_place, equal, greater, greater_or_equal, less,
    less_or_equal, mod_op, mul, mul_in_place, or, pow, pow_in_place, sub, sub_in_place, where_op,
    xor, Add, And, Div, DivMode, Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Mod, Mul, Or,
    Pow, Sub, Where, Xor,
};
pub use concat::{concat, tile, Concat, Tile};
pub use control_flow::If;
pub use conv::{conv, conv_integer, conv_transpose, Conv, ConvInteger, ConvTranspose};
pub use convert::Cast;
pub use einsum::{einsum, Einsum};
pub use gather::{
    gather, gather_elements, gather_nd, scatter_elements, scatter_nd, Gather, GatherElements,
    GatherND, ScatterElements, ScatterND, ScatterReduction,
};
pub use generate::{constant_of_shape, onehot, range, ConstantOfShape, OneHot, Range};
pub use identity::Identity;
pub use layout::{
    depth_to_space, expand, flatten, reshape, squeeze, squeeze_in_place, DepthToSpace,
    DepthToSpaceMode, Expand, Flatten, Reshape, Shape, Size, Squeeze, Transpose, Unsqueeze,
};
pub use matmul::{gemm_op, matmul, FusedMatMul, Gemm, MatMul, MatMulInteger};
pub use non_max_suppression::{non_max_suppression, BoxOrder, NonMaxSuppression};
pub use norm::{
    batch_norm, batch_norm_in_place, instance_normalization, layer_normalization, log_softmax,
    rms_normalization, softmax, BatchNormalization, InstanceNormalization, LayerNormalization,
    LogSoftmax, RmsNormalization, Softmax,
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
pub use random::{RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike};

pub use reduce::{
    arg_max, arg_min, cum_sum, nonzero, reduce_l2, reduce_max, reduce_mean, reduce_min,
    reduce_prod, reduce_sum, reduce_sum_square, topk, ArgMax, ArgMin, CumSum, NonZero, ReduceL2,
    ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, TopK,
};
pub use resize::{
    resize, resize_image, CoordTransformMode, NearestMode, Resize, ResizeMode, ResizeTarget,
};
pub use rnn::{gru, lstm, Direction, GRU, LSTM};
pub use slice::{slice, slice_in_place, Slice};
pub use split::{split, Split};
pub use trilu::{trilu, Trilu};
pub use unary_elementwise::{
    abs, abs_in_place, acos, acos_in_place, asin, asin_in_place, atan, atan_in_place, ceil,
    ceil_in_place, clip, clip_in_place, cos, cos_in_place, elu, elu_in_place, erf, erf_in_place,
    exp, exp_in_place, floor, floor_in_place, gelu, gelu_in_place, hard_sigmoid,
    hard_sigmoid_in_place, hard_swish, hard_swish_in_place, leaky_relu, leaky_relu_in_place, log,
    log_in_place, neg, neg_in_place, not, not_in_place, reciprocal, reciprocal_in_place, relu,
    relu_in_place, round, round_in_place, sigmoid, sigmoid_in_place, sign, sign_in_place, silu,
    silu_in_place, sin, sin_in_place, softplus, softplus_in_place, sqrt, sqrt_in_place, swish,
    swish_in_place, tan, tan_in_place, tanh, tanh_in_place, Abs, Acos, Asin, Atan, Ceil, Clip, Cos,
    Elu, Erf, Exp, Floor, Gelu, HardSigmoid, HardSwish, LeakyRelu, Log, Neg, Not, Reciprocal, Relu,
    Round, Sigmoid, Sign, Silu, Sin, Softplus, Sqrt, Swish, Tan, Tanh,
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
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DataType {
    Int32,
    Float,
    Int8,
    UInt8,
}

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

/// Enum of the different types of tensor view that can be used as a model or
/// operator input.
#[derive(Clone)]
pub enum Input<'a> {
    FloatTensor(TensorView<'a, f32>),
    Int32Tensor(TensorView<'a, i32>),
    Int8Tensor(TensorView<'a, i8>),
    UInt8Tensor(TensorView<'a, u8>),
}

impl Input<'_> {
    pub fn to_output(&self) -> Output {
        match self {
            Input::FloatTensor(t) => t.to_tensor().into(),
            Input::Int32Tensor(t) => t.to_tensor().into(),
            Input::Int8Tensor(t) => t.to_tensor().into(),
            Input::UInt8Tensor(t) => t.to_tensor().into(),
        }
    }

    fn layout(&self) -> &DynLayout {
        match self {
            Input::FloatTensor(t) => t.layout(),
            Input::Int32Tensor(t) => t.layout(),
            Input::Int8Tensor(t) => t.layout(),
            Input::UInt8Tensor(t) => t.layout(),
        }
    }
}

impl Layout for Input<'_> {
    impl_proxy_layout!();
}

macro_rules! impl_input_conversions {
    ($variant:ident, $element_type:ty) => {
        impl<'a> TryFrom<Input<'a>> for TensorView<'a, $element_type> {
            type Error = OpError;

            fn try_from(input: Input<'a>) -> Result<TensorView<'a, $element_type>, Self::Error> {
                match input {
                    Input::$variant(t) => Ok(t),
                    _ => Err(OpError::IncorrectInputType),
                }
            }
        }

        impl<'a> TryFrom<Input<'a>> for $element_type {
            type Error = OpError;

            fn try_from(input: Input<'a>) -> Result<$element_type, Self::Error> {
                let tensor: TensorView<'a, _> = input.try_into()?;
                tensor
                    .item()
                    .copied()
                    .ok_or(OpError::InvalidValue("Expected scalar value"))
            }
        }

        impl<'a> From<&'a Tensor<$element_type>> for Input<'a> {
            fn from(t: &'a Tensor<$element_type>) -> Input<'a> {
                Input::$variant(t.view())
            }
        }

        impl<'a> From<TensorView<'a, $element_type>> for Input<'a> {
            fn from(t: TensorView<'a, $element_type>) -> Input<'a> {
                Input::$variant(t)
            }
        }

        impl<'a, const N: usize> From<NdTensorView<'a, $element_type, N>> for Input<'a> {
            fn from(t: NdTensorView<'a, $element_type, N>) -> Input<'a> {
                Input::$variant(t.as_dyn())
            }
        }
    };
}

impl_input_conversions!(FloatTensor, f32);
impl_input_conversions!(Int32Tensor, i32);
impl_input_conversions!(Int8Tensor, i8);
impl_input_conversions!(UInt8Tensor, u8);

impl<'a> From<&'a Output> for Input<'a> {
    fn from(output: &'a Output) -> Input<'a> {
        match output {
            Output::FloatTensor(t) => Input::FloatTensor(t.view()),
            Output::Int32Tensor(t) => Input::Int32Tensor(t.view()),
            Output::Int8Tensor(t) => Input::Int8Tensor(t.view()),
            Output::UInt8Tensor(t) => Input::UInt8Tensor(t.view()),
        }
    }
}

/// An operator input which has been pre-packed for more efficient use during
/// inference.
pub enum PrepackedInput {
    /// Prepacked RHS / B input for matrix multiplication with f32 weights.
    FloatBMatrix(PackedBMatrix<f32>),
}

/// Enum of the different types of output tensor that a model or operator can
/// return.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    FloatTensor(Tensor<f32>),
    Int32Tensor(Tensor<i32>),
    Int8Tensor(Tensor<i8>),
    UInt8Tensor(Tensor<u8>),
}

impl Output {
    pub fn as_input(&self) -> Input {
        match self {
            Self::FloatTensor(ft) => Input::FloatTensor(ft.view()),
            Self::Int32Tensor(it) => Input::Int32Tensor(it.view()),
            Self::Int8Tensor(it) => Input::Int8Tensor(it.view()),
            Self::UInt8Tensor(it) => Input::UInt8Tensor(it.view()),
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
            Output::Int32Tensor(t) => t.layout(),
            Output::Int8Tensor(t) => t.layout(),
            Output::UInt8Tensor(t) => t.layout(),
            Output::FloatTensor(t) => t.layout(),
        }
    }
}

impl Layout for Output {
    impl_proxy_layout!();
}

/// Declare conversions between `Output` and `Tensor<T>` / `NdTensor<T, N>`.
macro_rules! impl_output_conversions {
    ($variant:ident, $element_type:ty) => {
        // Tensor<T> => Output
        impl From<Tensor<$element_type>> for Output {
            fn from(t: Tensor<$element_type>) -> Output {
                Output::$variant(t)
            }
        }

        // NdTensor<T> => Output
        impl<const N: usize> From<NdTensor<$element_type, N>> for Output {
            fn from(t: NdTensor<$element_type, N>) -> Output {
                Output::$variant(t.into_dyn())
            }
        }

        // Output => Tensor<T>
        impl TryFrom<Output> for Tensor<$element_type> {
            type Error = OpError;

            fn try_from(o: Output) -> Result<Tensor<$element_type>, OpError> {
                match o {
                    Output::$variant(t) => Ok(t),
                    _ => Err(OpError::IncorrectOutputType),
                }
            }
        }

        // Output => NdTensor<T, N>
        impl<const N: usize> TryFrom<Output> for NdTensor<$element_type, N> {
            type Error = OpError;

            fn try_from(o: Output) -> Result<NdTensor<$element_type, N>, OpError> {
                let tensor: Tensor<_> = o.try_into()?;
                let ndim = tensor.ndim();
                tensor.try_into().map_err(|_| OpError::IncorrectOutputRank {
                    actual: ndim,
                    expected: N,
                })
            }
        }

        // Output => TensorView<T>
        impl<'a> TryFrom<&'a Output> for TensorView<'a, $element_type> {
            type Error = OpError;

            fn try_from(o: &'a Output) -> Result<TensorView<'a, $element_type>, OpError> {
                match o {
                    Output::$variant(t) => Ok(t.view()),
                    _ => Err(OpError::IncorrectOutputType),
                }
            }
        }

        // Output => NdTensorView<T, N>
        impl<'a, const N: usize> TryFrom<&'a Output> for NdTensorView<'a, $element_type, N> {
            type Error = OpError;

            fn try_from(o: &'a Output) -> Result<NdTensorView<'a, $element_type, N>, OpError> {
                let view: TensorView<'a, _> = o.try_into()?;
                let ndim = view.ndim();
                view.try_into().map_err(|_| OpError::IncorrectOutputRank {
                    actual: ndim,
                    expected: N,
                })
            }
        }
    };
}

impl_output_conversions!(FloatTensor, f32);
impl_output_conversions!(Int32Tensor, i32);
impl_output_conversions!(Int8Tensor, i8);
impl_output_conversions!(UInt8Tensor, u8);

/// A value that is either a tensor view ([`Input`]) or an owned tensor
/// ([`Output`]). The names originate from the usage of these types as model
/// inputs and outputs.
#[derive(Clone)]
pub enum InputOrOutput<'a> {
    /// A tensor view (like a slice)
    Input(Input<'a>),
    /// An owned tensor (like a `Vec<T>`)
    Output(Output),
}

impl InputOrOutput<'_> {
    /// Convert this value to a tensor view.
    pub fn as_input(&self) -> Input {
        match self {
            InputOrOutput::Input(inp) => inp.clone(),
            InputOrOutput::Output(outp) => outp.as_input(),
        }
    }

    /// Convert this value to an owned tensor.
    pub fn to_output(&self) -> Output {
        match self {
            InputOrOutput::Input(inp) => inp.to_output(),
            InputOrOutput::Output(outp) => outp.clone(),
        }
    }

    pub fn layout(&self) -> &DynLayout {
        match self {
            Self::Input(inp) => inp.layout(),
            Self::Output(outp) => outp.layout(),
        }
    }
}

impl<'a> From<Input<'a>> for InputOrOutput<'a> {
    fn from(val: Input<'a>) -> Self {
        InputOrOutput::Input(val)
    }
}

impl<'a, T: 'static, S: Storage<Elem = T>, L: MutLayout> From<&'a TensorBase<S, L>>
    for InputOrOutput<'a>
where
    Input<'a>: From<TensorView<'a, T>>,
{
    fn from(val: &'a TensorBase<S, L>) -> Self {
        InputOrOutput::Input(val.as_dyn().into())
    }
}

impl<'a, T, L: MutLayout> From<TensorBase<ViewData<'a, T>, L>> for InputOrOutput<'a>
where
    Input<'a>: From<TensorView<'a, T>>,
{
    fn from(val: TensorBase<ViewData<'a, T>, L>) -> Self {
        InputOrOutput::Input(val.as_dyn().into())
    }
}

impl<T, L: MutLayout> From<TensorBase<Vec<T>, L>> for InputOrOutput<'static>
where
    Output: From<Tensor<T>>,
    DynLayout: From<L>,
{
    fn from(val: TensorBase<Vec<T>, L>) -> Self {
        InputOrOutput::Output(val.into_dyn().into())
    }
}

impl From<Output> for InputOrOutput<'static> {
    fn from(val: Output) -> Self {
        InputOrOutput::Output(val)
    }
}

impl<'a> From<&'a Output> for InputOrOutput<'a> {
    fn from(val: &'a Output) -> Self {
        let inp: Input<'a> = Input::from(val);
        inp.into()
    }
}

impl Layout for InputOrOutput<'_> {
    impl_proxy_layout!();
}

/// Trait for values that can be converted into the result type used by
/// [`Operator::run`].
pub trait IntoOpResult {
    fn into_op_result(self) -> Result<OutputList, OpError>;
}

impl IntoOpResult for Result<Output, OpError> {
    fn into_op_result(self) -> Result<OutputList, OpError> {
        self.map(|out| [out].into())
    }
}

impl IntoOpResult for Output {
    fn into_op_result(self) -> Result<OutputList, OpError> {
        Ok([self].into())
    }
}

impl<S: Storage, L: MutLayout> IntoOpResult for TensorBase<S, L>
where
    Output: From<TensorBase<S, L>>,
{
    fn into_op_result(self) -> Result<OutputList, OpError> {
        let output: Output = self.into();
        Ok([output].into())
    }
}

impl<S: Storage, L: MutLayout> IntoOpResult for Result<TensorBase<S, L>, OpError>
where
    Output: From<TensorBase<S, L>>,
{
    fn into_op_result(self) -> Result<OutputList, OpError> {
        self.map(|tensor| [tensor.into()].into())
    }
}

impl<T> IntoOpResult for Result<Vec<T>, OpError>
where
    Output: From<T>,
{
    fn into_op_result(self) -> Result<OutputList, OpError> {
        self.map(|tensors| tensors.into_iter().map(|t| t.into()).collect())
    }
}

/// Possible reasons why an operator may fail on a given input.
#[derive(Eq, PartialEq, Debug)]
pub enum OpError {
    /// Input tensors have an element type that is unsupported or incompatible
    /// with other inputs.
    IncorrectInputType,

    /// Could not convert operator output to the expected type.
    IncorrectOutputType,

    /// Could not convert operator output to the expected static rank.
    IncorrectOutputRank { actual: usize, expected: usize },

    /// Input tensor shapes are not compatible with each other or operator
    /// attributes.
    IncompatibleInputShapes(&'static str),

    /// The number of inputs was less than the required number.
    MissingInputs,

    /// An input has a value that is incorrect.
    InvalidValue(&'static str),

    /// An input or attribute has a value that is valid, but not currently supported.
    UnsupportedValue(&'static str),

    /// An input has a type that is valid for the operator, but not currently
    /// supported.
    UnsupportedType,
}

impl Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::IncorrectInputType => write!(f, "incorrect or unsupported input type"),
            OpError::IncorrectOutputRank { actual, expected } => {
                write!(
                    f,
                    "expected output to have {} dims but it has {}",
                    expected, actual
                )
            }
            OpError::IncorrectOutputType => write!(f, "output type mismatch"),
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

/// Outputs from an operator.
///
/// This avoids allocations in the common case where an operator produces
/// exactly one output.
pub type OutputList = SmallVec<[Output; 1]>;

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

    /// Execute the operator with the given inputs.
    ///
    /// The output, and any large intermediate buffers used by the operation,
    /// should be allocated from `pool`.
    fn run(&self, pool: &TensorPool, input: InputList) -> Result<OutputList, OpError>;

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
    /// return as the output. `other` are the remaining inputs.
    ///
    /// Operators may fall back to allocating a new output if some property of
    /// the input data or shapes means in-place operation is not possible. In
    /// this case they should return the input buffer to the pool, and allocate
    /// the new output buffer from it. The pool should also be used for any
    /// temporary buffers created during execution.
    fn run_in_place(
        &self,
        _pool: &TensorPool,
        _input: Output,
        _other: InputList,
    ) -> Result<Output, OpError> {
        unimplemented!("in-place execution not supported")
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
        #[allow(unused)] input: Input,
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
    fn run_subgraph(
        &self,
        pool: &TensorPool,
        input: InputList,
        #[allow(unused)] captures: CaptureEnv,
        #[allow(unused)] weight_cache: Option<&[WeightCache]>,
        #[allow(unused)] run_opts: Option<RunOptions>,
    ) -> Result<OutputList, RunError> {
        self.run(pool, input)
            .map_err(|error| RunError::OperatorError {
                name: self.name().to_string(),
                error,
            })
    }
}

impl_downcastdyn!(Operator);

/// List of inputs for an operator evaluation.
///
/// Conceptually this is a `Cow<[Option<Input>]>` with methods to conveniently
/// extract inputs and produce appropriate errors if inputs are missing or of
/// the wrong type.
///
/// An InputList can be constructed from a tensor reference or tuple of tensor
/// references using `into`.
pub struct InputList<'a> {
    inputs: Cow<'a, [Option<Input<'a>>]>,

    /// Callback that retrieves the pre-packed copy of an input with a given
    /// index.
    get_prepacked: Option<&'a dyn Fn(usize) -> Option<&'a PrepackedInput>>,
}

impl<'a> InputList<'a> {
    /// Construct an empty input list.
    pub fn new() -> InputList<'a> {
        InputList {
            inputs: Cow::Owned(vec![]),
            get_prepacked: None,
        }
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
    pub fn push<I: Into<Input<'a>>>(&mut self, inp: I) {
        self.inputs.to_mut().push(Some(inp.into()))
    }

    /// Construct an input list from a slice of non-optional inputs.
    ///
    /// This copies the inputs into a new vector of `Optional<Input>`s. Using
    /// [`from_optional`](Self::from_optional) is more efficient.
    pub fn from(inputs: &[Input<'a>]) -> InputList<'a> {
        InputList {
            inputs: inputs.iter().cloned().map(Some).collect(),
            get_prepacked: None,
        }
    }

    /// Construct an input list from a slice of optional inputs.
    ///
    /// This is a cheap conversion that borrows `inputs`.
    pub fn from_optional(inputs: &'a [Option<Input<'a>>]) -> InputList<'a> {
        InputList {
            inputs: Cow::Borrowed(inputs),
            get_prepacked: None,
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
    pub fn get(&self, index: usize) -> Option<Input<'a>> {
        self.inputs.get(index).cloned().flatten()
    }

    /// Get the pre-packed version of a weight input, if available.
    pub fn get_prepacked(&self, index: usize) -> Option<&'a PrepackedInput> {
        self.get_prepacked.and_then(|gp| gp(index))
    }

    /// Get a mutable reference to an input.
    ///
    /// This will convert the list into an owned list of inputs first.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Input<'a>> {
        self.inputs.to_mut().get_mut(index)?.as_mut()
    }

    /// Get an optional input as a tensor.
    pub fn get_as<T>(&self, index: usize) -> Result<Option<TensorView<'a, T>>, OpError>
    where
        TensorView<'a, T>: TryFrom<Input<'a>, Error = OpError>,
    {
        self.get(index).map(|input| input.try_into()).transpose()
    }

    /// Get an optional input as a scalar value.
    pub fn get_as_scalar<T: Copy + 'a>(&self, index: usize) -> Result<Option<T>, OpError>
    where
        TensorView<'a, T>: TryFrom<Input<'a>, Error = OpError>,
    {
        let tensor = self.get_as::<T>(index)?;
        tensor
            .map(|t| {
                t.item()
                    .copied()
                    .ok_or(OpError::InvalidValue("Expected scalar value"))
            })
            .transpose()
    }

    /// Get a required operator input.
    pub fn require(&self, index: usize) -> Result<Input<'a>, OpError> {
        self.get(index).ok_or(OpError::MissingInputs)
    }

    /// Get a required operator input as a tensor.
    pub fn require_as<T>(&self, index: usize) -> Result<TensorView<'a, T>, OpError>
    where
        TensorView<'a, T>: TryFrom<Input<'a>, Error = OpError>,
    {
        self.require(index).and_then(|input| input.try_into())
    }

    /// Get a required input as a scalar value.
    pub fn require_as_scalar<T>(&self, index: usize) -> Result<T, OpError>
    where
        T: 'a + Copy + TryFrom<Input<'a>, Error = OpError>,
    {
        self.require(index).and_then(|input| input.try_into())
    }

    /// Return an iterator over provided inputs.
    ///
    /// If the InputList was constructed with `from_optional`, this will skip
    /// over any missing inputs.
    pub fn iter<'b>(&'b self) -> impl Iterator<Item = Input<'a>> + 'b {
        self.inputs.iter().filter_map(|inp| inp.clone())
    }
}

impl Default for InputList<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, I: Into<Input<'a>>> From<I> for InputList<'a> {
    fn from(val: I) -> InputList<'a> {
        InputList::from(&[val.into()])
    }
}

impl<'a, I1: Into<Input<'a>>, I2: Into<Input<'a>>> From<(I1, I2)> for InputList<'a> {
    fn from((a, b): (I1, I2)) -> InputList<'a> {
        InputList::from(&[a.into(), b.into()])
    }
}

impl<'a, I1: Into<Input<'a>>, I2: Into<Input<'a>>, I3: Into<Input<'a>>> From<(I1, I2, I3)>
    for InputList<'a>
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

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::test_util::{expect_equal_with_tolerance, ExpectEqualError};
    use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};

    use super::{Input, InputList, OpError, Operator, Output};
    use crate::downcast::DowncastDyn;
    use crate::ops::{Add, Sub};
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

    /// Utility to simplify running a single-output [`Operator`] with a list of
    /// typed inputs.
    ///
    /// Usage is:
    ///
    /// ```text
    /// let result: NdTensor<f32, 2> = run_op(&op, (data.view(), arg.view()))
    /// ```
    pub fn run_op<'a, I: Into<InputList<'a>>, O: TryFrom<Output, Error = OpError>>(
        op: &dyn Operator,
        inputs: I,
    ) -> Result<O, OpError> {
        let pool = new_pool();
        op.run(&pool, inputs.into())?.remove(0).try_into()
    }

    #[test]
    fn test_input_from_tensor() {
        let tensor = NdTensor::<i32, 3>::zeros([1, 2, 3]);
        let input: Input = tensor.view().into();
        assert!(matches!(input, Input::Int32Tensor(_)));
        assert_eq!(input.shape(), &[1, 2, 3]);

        let tensor = NdTensor::<f32, 2>::zeros([5, 5]);
        let input: Input = tensor.view().into();
        assert!(matches!(input, Input::FloatTensor(_)));
        assert_eq!(input.shape(), &[5, 5]);
    }

    #[test]
    fn test_tensor_from_output() {
        let original = NdTensor::from([[1., 2.], [3., 4.]]);
        let output: Output = original.clone().into();

        let mat_dyn: Tensor<f32> = output.clone().try_into().unwrap();
        assert_eq!(mat_dyn, original);

        let mat: NdTensor<f32, 2> = output.clone().try_into().unwrap();
        assert_eq!(mat, original);

        let err: Result<NdTensor<i32, 2>, _> = output.clone().try_into();
        assert_eq!(err, Err(OpError::IncorrectOutputType));

        let err: Result<NdTensor<f32, 3>, _> = output.clone().try_into();
        assert_eq!(
            err,
            Err(OpError::IncorrectOutputRank {
                actual: 2,
                expected: 3
            })
        );
    }

    #[test]
    fn test_tensor_view_from_output() {
        let original = NdTensor::from([[1., 2.], [3., 4.]]);
        let output: Output = original.clone().into();

        let mat_dyn: TensorView<f32> = (&output).try_into().unwrap();
        assert_eq!(mat_dyn, original);

        let mat: NdTensorView<f32, 2> = (&output).try_into().unwrap();
        assert_eq!(mat, original);

        let err: Result<NdTensorView<i32, 2>, _> = (&output).try_into();
        assert_eq!(err, Err(OpError::IncorrectOutputType));

        let err: Result<NdTensorView<f32, 3>, _> = (&output).try_into();
        assert_eq!(
            err,
            Err(OpError::IncorrectOutputRank {
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
