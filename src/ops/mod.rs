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
    DynLayout, MutLayout, NdTensor, NdTensorView, Tensor, TensorBase, TensorView, ViewData,
};

use crate::downcast::impl_downcastdyn;
use crate::tensor_pool::TensorPool;

// Modules containing ops that correspond to ONNX operators.
mod binary_elementwise;
mod concat;
mod conv;
mod convert;
mod gather;
mod generate;
mod identity;
mod layout;
mod matmul;
mod non_max_suppression;
mod norm;
mod pad;
mod pooling;

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
pub use conv::{conv, conv_transpose, Conv, ConvTranspose};
pub use convert::Cast;
pub use gather::{
    gather, gather_elements, gather_nd, scatter_elements, scatter_nd, Gather, GatherElements,
    GatherND, ScatterElements, ScatterND, ScatterReduction,
};
pub use generate::{constant_of_shape, onehot, range, ConstantOfShape, OneHot, Range};
pub use identity::Identity;
pub use layout::{
    expand, flatten, reshape, squeeze, squeeze_in_place, Expand, Flatten, Reshape, Shape, Size,
    Squeeze, Transpose, Unsqueeze,
};
pub use matmul::{gemm_op, matmul, Gemm, MatMul};
pub use non_max_suppression::{non_max_suppression, BoxOrder, NonMaxSuppression};
pub use norm::{
    batch_norm, batch_norm_in_place, instance_normalization, layer_normalization, log_softmax,
    softmax, BatchNormalization, InstanceNormalization, LayerNormalization, LogSoftmax, Softmax,
};
pub use pad::{pad, Pad};
pub use pooling::{
    average_pool, global_average_pool, max_pool, AveragePool, GlobalAveragePool, MaxPool,
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
    exp, exp_in_place, floor, floor_in_place, hard_sigmoid, hard_sigmoid_in_place, hard_swish,
    hard_swish_in_place, leaky_relu, leaky_relu_in_place, log, log_in_place, neg, neg_in_place,
    not, not_in_place, reciprocal, reciprocal_in_place, relu, relu_in_place, round, round_in_place,
    sigmoid, sigmoid_in_place, sign, sign_in_place, sin, sin_in_place, softplus, softplus_in_place,
    sqrt, sqrt_in_place, tan, tan_in_place, tanh, tanh_in_place, Abs, Acos, Asin, Atan, Ceil, Clip,
    Cos, Elu, Erf, Exp, Floor, HardSigmoid, HardSwish, LeakyRelu, Log, Neg, Not, Reciprocal, Relu,
    Round, Sigmoid, Sign, Sin, Softplus, Sqrt, Tan, Tanh,
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

/// Construct a [Padding::Fixed] from a slice of paddings for each size.
impl<S: AsRef<[usize]>> From<S> for Padding {
    fn from(val: S) -> Padding {
        Padding::Fixed(val.as_ref().into())
    }
}

#[derive(Copy, Clone, Debug)]
pub enum DataType {
    Int32,
    Float,
}

/// Enum of the different types of input tensor that an operator can accept.
#[derive(Clone)]
pub enum Input<'a> {
    FloatTensor(TensorView<'a, f32>),
    IntTensor(TensorView<'a, i32>),
}

impl<'a> Input<'a> {
    pub fn to_output(&self) -> Output {
        match self {
            Input::FloatTensor(t) => t.to_tensor().into(),
            Input::IntTensor(t) => t.to_tensor().into(),
        }
    }

    fn layout(&self) -> &DynLayout {
        match self {
            Input::FloatTensor(t) => t.layout(),
            Input::IntTensor(t) => t.layout(),
        }
    }
}

impl<'a> Layout for Input<'a> {
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
}

impl<'a> TryFrom<Input<'a>> for TensorView<'a, f32> {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<TensorView<'a, f32>, Self::Error> {
        match input {
            Input::FloatTensor(t) => Ok(t),
            _ => Err(OpError::IncorrectInputType),
        }
    }
}

impl<'a> TryFrom<Input<'a>> for TensorView<'a, i32> {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<TensorView<'a, i32>, Self::Error> {
        match input {
            Input::IntTensor(t) => Ok(t),
            _ => Err(OpError::IncorrectInputType),
        }
    }
}

impl<'a> TryFrom<Input<'a>> for f32 {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<f32, Self::Error> {
        let tensor: TensorView<'a, _> = input.try_into()?;
        tensor
            .item()
            .copied()
            .ok_or(OpError::InvalidValue("Expected scalar value"))
    }
}

impl<'a> TryFrom<Input<'a>> for i32 {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<i32, Self::Error> {
        let tensor: TensorView<'a, _> = input.try_into()?;
        tensor
            .item()
            .copied()
            .ok_or(OpError::InvalidValue("Expected scalar value"))
    }
}

macro_rules! impl_input_conversions {
    ($variant:ident, $element_type:ty) => {
        impl<'a> From<&'a Tensor<$element_type>> for Input<'a> {
            fn from(t: &'a Tensor<$element_type>) -> Input {
                Input::$variant(t.view())
            }
        }

        impl<'a> From<TensorView<'a, $element_type>> for Input<'a> {
            fn from(t: TensorView<'a, $element_type>) -> Input {
                Input::$variant(t)
            }
        }

        impl<'a, const N: usize> From<NdTensorView<'a, $element_type, N>> for Input<'a> {
            fn from(t: NdTensorView<'a, $element_type, N>) -> Input {
                Input::$variant(t.as_dyn())
            }
        }
    };
}

impl_input_conversions!(FloatTensor, f32);
impl_input_conversions!(IntTensor, i32);

impl<'a> From<&'a Output> for Input<'a> {
    fn from(output: &'a Output) -> Input {
        match output {
            Output::FloatTensor(t) => Input::FloatTensor(t.view()),
            Output::IntTensor(t) => Input::IntTensor(t.view()),
        }
    }
}

/// Enum of the different types of output tensor that an operator can produce.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    FloatTensor(Tensor<f32>),
    IntTensor(Tensor<i32>),
}

impl Output {
    pub fn as_input(&self) -> Input {
        match self {
            Self::FloatTensor(ft) => Input::FloatTensor(ft.view()),
            Self::IntTensor(it) => Input::IntTensor(it.view()),
        }
    }

    pub fn into_int(self) -> Option<Tensor<i32>> {
        if let Output::IntTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn as_int_ref(&self) -> Option<&Tensor<i32>> {
        if let Output::IntTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn into_float(self) -> Option<Tensor<f32>> {
        if let Output::FloatTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn as_float_ref(&self) -> Option<&Tensor<f32>> {
        if let Output::FloatTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    fn layout(&self) -> &DynLayout {
        match self {
            Output::IntTensor(t) => t.layout(),
            Output::FloatTensor(t) => t.layout(),
        }
    }
}

impl Layout for Output {
    type Index<'a> = <DynLayout as Layout>::Index<'a>;
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
                tensor.try_into().map_err(|_| OpError::IncorrectOutputType)
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
                view.try_into().map_err(|_| OpError::IncorrectOutputType)
            }
        }
    };
}

impl_output_conversions!(FloatTensor, f32);
impl_output_conversions!(IntTensor, i32);

/// A value that is either a tensor view ([`Input`]) or an owned tensor
/// ([`Output`]).
#[derive(Clone)]
pub enum InputOrOutput<'a> {
    Input(Input<'a>),
    Output(Output),
}

impl<'a> InputOrOutput<'a> {
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

impl<'a> Layout for InputOrOutput<'a> {
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
}

/// Trait for values that can be converted into the result type used by
/// `Operator::run`.
pub trait IntoOpResult {
    fn into_op_result(self) -> Result<Vec<Output>, OpError>;
}

impl IntoOpResult for Result<Output, OpError> {
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        self.map(|out| [out].into())
    }
}

impl IntoOpResult for Output {
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        Ok([self].into())
    }
}

impl<T> IntoOpResult for Tensor<T>
where
    Output: From<Tensor<T>>,
{
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        let output: Output = self.into();
        Ok([output].into())
    }
}

impl<T, const N: usize> IntoOpResult for NdTensor<T, N>
where
    Output: From<Tensor<T>>,
{
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        let output: Output = self.into_dyn().into();
        Ok([output].into())
    }
}

impl<T> IntoOpResult for Result<Tensor<T>, OpError>
where
    Output: From<Tensor<T>>,
{
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        self.map(|tensor| [tensor.into()].into())
    }
}

impl<T> IntoOpResult for Result<Vec<Tensor<T>>, OpError>
where
    Output: From<Tensor<T>>,
{
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
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
            OpError::IncorrectInputType => write!(f, "incorrect or unsupported input type"),
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
        }
    }
}

impl Error for OpError {}

/// Check that a tensor has an expected number of dimensions, or return an
/// `OpError::InvalidValue`.
///
/// Can be used with `check_dims!(input, expected_rank)` if `input` is a
/// `Tensor<T>` or `check_dims!(input?, expected_rank)` if `input` is an
/// `Option<Tensor<T>>`.
///
/// If `$ndim` is a literal, the macro returns an array of `$ndim` sizes for
/// each dimension. This conveniently allows checking the rank of a tensor
/// and extracting the sizes of dimension in one call. For example:
/// `let [rows, cols] = check_dims!(matrix, 2)`. When `$ndim` is a literal,
/// a third argument can also be passed to specify the names of the dimensions,
/// eg. "NCHW" or "dir, batch, seq". This can produce more helpful errors if
/// the input does not match the expected shape.
#[doc(hidden)]
#[macro_export]
macro_rules! check_dims {
    ($tensor:ident, $ndim:literal, $dim_names:literal) => {{
        let shape: [usize; $ndim] = $tensor.shape().try_into().map_err(|_| {
            OpError::InvalidValue(concat!(
                stringify!($tensor),
                " must have ",
                stringify!($ndim),
                " dims (",
                $dim_names,
                ")"
            ))
        })?;
        shape
    }};

    ($tensor:ident, $ndim:literal) => {{
        let shape: [usize; $ndim] = $tensor.shape().try_into().map_err(|_| {
            OpError::InvalidValue(concat!(
                stringify!($tensor),
                " must have ",
                stringify!($ndim),
                " dims"
            ))
        })?;
        shape
    }};

    ($tensor:ident, $ndim:expr) => {
        if $tensor.ndim() != $ndim {
            return Err(OpError::InvalidValue(concat!(
                stringify!($tensor),
                " must have ",
                stringify!($ndim),
                " dims"
            )));
        }
    };

    ($tensor:ident?, $ndim: expr) => {
        if let Some($tensor) = $tensor.as_ref() {
            check_dims!($tensor, $ndim);
        }
    };
}

/// Convert a tensor with dynamic dimension count to an `NdTensorView`, or
/// return an `OpError::InvalidValue` if the dimension count is incorrect.
#[doc(hidden)]
#[macro_export]
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
}

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
    fn run(&self, pool: &TensorPool, input: InputList) -> Result<Vec<Output>, OpError>;

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
    /// [Operator::run_in_place] implementation.
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
    /// that case they should allocate the output from `pool`. The pool should
    /// also be used for any temporary buffers created during execution.
    fn run_in_place(
        &self,
        _pool: &TensorPool,
        _input: Output,
        _other: InputList,
    ) -> Result<Output, OpError> {
        unimplemented!("in-place execution not supported")
    }
}

impl_downcastdyn!(Operator);

/// List of inputs for an operator evaluation.
///
/// Conceptually this is like a `&[Option<Input>]` with methods to conveniently
/// extract inputs and produce appropriate errors if inputs are missing or of
/// the wrong type.
///
/// An InputList can be constructed from a tensor reference or tuple of tensor
/// references using `into`.
pub struct InputList<'a> {
    inputs: Cow<'a, [Option<Input<'a>>]>,
}

impl<'a> InputList<'a> {
    /// Construct an empty input list.
    pub fn new() -> InputList<'static> {
        InputList {
            inputs: Cow::Owned(vec![]),
        }
    }

    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    pub fn from(inputs: &[Input<'a>]) -> InputList<'a> {
        InputList {
            inputs: inputs.iter().cloned().map(Some).collect(),
        }
    }

    pub fn from_optional(inputs: &'a [Option<Input<'a>>]) -> InputList<'a> {
        InputList {
            inputs: Cow::Borrowed(inputs),
        }
    }

    /// Get an optional input.
    pub fn get(&self, index: usize) -> Option<Input<'a>> {
        self.inputs.get(index).cloned().flatten()
    }

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
) -> Result<Vec<usize>, OpError> {
    let mut resolved_axes = Vec::with_capacity(axes.len());
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
    use rten_tensor::NdTensor;

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

    /// Utility to simplify running a single-output [Operator] with a list of
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
        assert!(matches!(input, Input::IntTensor(_)));
        assert_eq!(input.shape(), &[1, 2, 3]);

        let tensor = NdTensor::<f32, 2>::zeros([5, 5]);
        let input: Input = tensor.view().into();
        assert!(matches!(input, Input::FloatTensor(_)));
        assert_eq!(input.shape(), &[5, 5]);
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
