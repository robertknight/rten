use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};

use wasnn_tensor::{DynLayout, Layout, Tensor, TensorCommon};

mod binary_elementwise;
mod concat;
mod conv;
mod convert;
mod gather;
mod generate;
mod identity;
mod layout;
mod matmul;
mod norm;
mod pad;
mod pooling;
mod reduce;
mod resize;
mod rnn;
mod slice;
mod split;
mod unary_elementwise;

pub use binary_elementwise::{
    add, add_in_place, div, div_in_place, equal, greater, less, less_or_equal, mod_op, mul,
    mul_in_place, pow, pow_in_place, sub, sub_in_place, where_op, Add, Div, DivMode, Equal,
    Greater, Less, LessOrEqual, Mod, Mul, Pow, Sub, Where,
};
pub use concat::{concat, Concat};
pub use conv::{conv, conv_transpose, Conv, ConvTranspose};
pub use convert::Cast;
pub use gather::{gather, Gather};
pub use generate::{constant_of_shape, range, ConstantOfShape, Range};
pub use identity::Identity;
pub use layout::{
    expand, flatten, reshape, squeeze, squeeze_in_place, Expand, Flatten, Reshape, Shape, Squeeze,
    Transpose, Unsqueeze,
};
pub use matmul::{gemm_op, matmul, Gemm, MatMul};
pub use norm::{
    batch_norm, batch_norm_in_place, log_softmax, softmax, BatchNormalization, LogSoftmax, Softmax,
};
pub use pad::{pad, Pad};
pub use pooling::{
    average_pool, global_average_pool, max_pool, AveragePool, GlobalAveragePool, MaxPool,
};
pub use reduce::{
    arg_max, arg_min, cum_sum, nonzero, reduce_l2, reduce_max, reduce_mean, reduce_min,
    reduce_prod, reduce_sum, ArgMax, ArgMin, CumSum, NonZero, ReduceL2, ReduceMax, ReduceMean,
    ReduceMin, ReduceProd, ReduceSum,
};
pub use resize::{resize, CoordTransformMode, NearestMode, Resize, ResizeMode, ResizeTarget};
pub use rnn::{gru, lstm, Direction, GRU, LSTM};
pub use slice::{slice, slice_in_place, Slice};
pub use split::{split, Split};
pub use unary_elementwise::{
    clip, clip_in_place, cos, cos_in_place, erf, erf_in_place, leaky_relu, leaky_relu_in_place,
    log, log_in_place, relu, relu_in_place, sigmoid, sigmoid_in_place, sin, sin_in_place, sqrt,
    sqrt_in_place, tanh, tanh_in_place, Clip, Cos, Erf, LeakyRelu, Log, Relu, Sigmoid, Sin, Sqrt,
    Tanh,
};

#[derive(Copy, Clone, Debug)]
pub enum Padding {
    /// Apply enough padding such that the output and input have the same size.
    ///
    /// If the required amount of padding along each dimension is even, it is
    /// divided equally between the start and the end. If it is odd, one more
    /// unit is added on the end than the start. This matches the ONNX spec
    /// for the "SAME_UPPER" value for the `auto_pad` attribute.
    Same,

    /// Apply a given amount of padding to the top, left, bottom and right of
    /// the input.
    Fixed([usize; 4]),
}

#[derive(Copy, Clone, Debug)]
pub enum DataType {
    Int32,
    Float,
}

/// Enum of the different types of input tensor that an operator can accept.
#[derive(Clone, Copy)]
pub enum Input<'a> {
    FloatTensor(&'a Tensor<f32>),
    IntTensor(&'a Tensor<i32>),
}

impl<'a> Input<'a> {
    fn layout(&self) -> &'a DynLayout {
        match self {
            Input::FloatTensor(t) => t.layout(),
            Input::IntTensor(t) => t.layout(),
        }
    }
}

impl<'a> Layout for Input<'a> {
    type Index<'b> = <DynLayout as Layout>::Index<'b> where Self: 'b;
    type Indices = <DynLayout as Layout>::Indices;

    fn ndim(&self) -> usize {
        self.layout().ndim()
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

impl<'a> TryFrom<Input<'a>> for &'a Tensor<f32> {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<&'a Tensor<f32>, Self::Error> {
        match input {
            Input::FloatTensor(t) => Ok(t),
            _ => Err(OpError::IncorrectInputType),
        }
    }
}

impl<'a> TryFrom<Input<'a>> for &'a Tensor<i32> {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<&'a Tensor<i32>, Self::Error> {
        match input {
            Input::IntTensor(t) => Ok(t),
            _ => Err(OpError::IncorrectInputType),
        }
    }
}

impl<'a> TryFrom<Input<'a>> for f32 {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<f32, Self::Error> {
        let tensor: &Tensor<_> = input.try_into()?;
        tensor
            .item()
            .copied()
            .ok_or(OpError::InvalidValue("Expected scalar value"))
    }
}

impl<'a> TryFrom<Input<'a>> for i32 {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<i32, Self::Error> {
        let tensor: &Tensor<_> = input.try_into()?;
        tensor
            .item()
            .copied()
            .ok_or(OpError::InvalidValue("Expected scalar value"))
    }
}

impl<'a> From<&'a Tensor<f32>> for Input<'a> {
    fn from(t: &'a Tensor<f32>) -> Input {
        Input::FloatTensor(t)
    }
}

impl<'a> From<&'a Tensor<i32>> for Input<'a> {
    fn from(t: &'a Tensor<i32>) -> Input {
        Input::IntTensor(t)
    }
}

impl<'a> From<&'a Output> for Input<'a> {
    fn from(output: &'a Output) -> Input {
        match output {
            Output::FloatTensor(ref t) => Input::FloatTensor(t),
            Output::IntTensor(ref t) => Input::IntTensor(t),
        }
    }
}

/// Enum of the different types of output tensor that an operator can produce.
pub enum Output {
    FloatTensor(Tensor<f32>),
    IntTensor(Tensor<i32>),
}

impl Output {
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

impl From<Tensor<f32>> for Output {
    fn from(t: Tensor<f32>) -> Output {
        Output::FloatTensor(t)
    }
}

impl From<Tensor<i32>> for Output {
    fn from(t: Tensor<i32>) -> Output {
        Output::IntTensor(t)
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
        if let Some($tensor) = $tensor {
            check_dims!($tensor, $ndim);
        }
    };
}

/// Convert a tensor with dynamic dimension count to an `NdTensorView`, or
/// return an `OpError::InvalidValue` if the dimension count is incorrect.
#[macro_export]
macro_rules! static_dims {
    ($tensor:ident, $ndim:literal, $dim_names:literal) => {{
        use wasnn_tensor::{Layout, TensorCommon};

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
        use wasnn_tensor::{Layout, TensorCommon};

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
pub trait Operator: Debug {
    /// Return a display name for the operator.
    fn name(&self) -> &str;

    /// Execute the operator with the given inputs.
    fn run(&self, input: InputList) -> Result<Vec<Output>, OpError>;

    /// Return true if this operator supports in-place execution via
    /// `run_in_place`.
    ///
    /// In-place execution returns results by modifying an existing tensor
    /// instead of allocating a new one. Reducing memory allocations can
    /// significantly speed up graph runs.
    fn can_run_in_place(&self) -> bool {
        false
    }

    /// Execute this operator in-place on an existing tensor.
    ///
    /// This may only be called if `can_run_in_place` returns true.
    ///
    /// `input` is the first input, which the implementation may modify and
    /// return as the output. `other` are the remaining inputs.
    fn run_in_place(&self, _input: Output, _other: InputList) -> Result<Output, OpError> {
        unimplemented!("in-place execution not supported")
    }
}

/// List of inputs for an operator evaluation.
///
/// Conceptually this is like a `&[Option<Input>]` with methods to conveniently
/// extract inputs and produce appropriate errors if inputs are missing or of
/// the wrong type.
pub struct InputList<'a> {
    inputs: Vec<Option<Input<'a>>>,
}

impl<'a> InputList<'a> {
    pub fn from<'b>(inputs: &'b [Input<'b>]) -> InputList<'b> {
        InputList {
            inputs: inputs.iter().copied().map(Some).collect(),
        }
    }

    pub fn from_optional<'b>(inputs: &'b [Option<Input<'b>>]) -> InputList<'b> {
        InputList {
            inputs: inputs.to_vec(),
        }
    }

    /// Get an optional input.
    pub fn get(&self, index: usize) -> Option<Input<'a>> {
        self.inputs.get(index).copied().flatten()
    }

    /// Get an optional input as a tensor.
    pub fn get_as<T>(&self, index: usize) -> Result<Option<&'a Tensor<T>>, OpError>
    where
        &'a Tensor<T>: TryFrom<Input<'a>, Error = OpError>,
    {
        self.get(index).map(|input| input.try_into()).transpose()
    }

    /// Get an optional input as a scalar value.
    pub fn get_as_scalar<T: Copy + 'a>(&self, index: usize) -> Result<Option<T>, OpError>
    where
        &'a Tensor<T>: TryFrom<Input<'a>, Error = OpError>,
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
    pub fn require_as<T>(&self, index: usize) -> Result<&'a Tensor<T>, OpError>
    where
        &'a Tensor<T>: TryFrom<Input<'a>, Error = OpError>,
    {
        self.require(index).and_then(|input| input.try_into())
    }

    /// Get a required input as a scalar value.
    pub fn require_as_scalar<T: Copy + 'a>(&self, index: usize) -> Result<T, OpError>
    where
        T: TryFrom<Input<'a>, Error = OpError>,
    {
        self.require(index).and_then(|input| input.try_into())
    }

    /// Return an iterator over provided inputs.
    ///
    /// If the InputList was constructed with `from_optional`, this will skip
    /// over any missing inputs.
    pub fn iter(&'a self) -> impl Iterator<Item = Input<'a>> + 'a {
        self.inputs.iter().filter_map(|inp| *inp)
    }
}

#[derive(Debug)]
pub enum Scalar {
    Int(i32),
    Float(f32),
}

/// Resolve an axis given as a value in `[-ndim, ndim-1]` to the zero-based
/// dimension of a tensor with `ndim` dimensions.
///
/// Negative axis values count backwards from the last dimension.
fn resolve_axis(ndim: usize, axis: isize) -> Result<usize, OpError> {
    let rank = ndim as isize;
    if axis < -rank || axis >= rank {
        return Err(OpError::InvalidValue("Axis is invalid"));
    }

    if axis >= 0 {
        Ok(axis as usize)
    } else {
        Ok((rank + axis) as usize)
    }
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
