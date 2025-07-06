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

use rten_tensor::errors::DimensionError;
use rten_tensor::{MutLayout, Storage, TensorBase};

use crate::downcast::impl_downcastdyn;
use crate::gemm::PackedBMatrix;
use crate::graph::{CaptureEnv, Graph, RunError, RunOptions};
use crate::tensor_pool::TensorPool;
use crate::timing::Profiler;
use crate::value::{CastError, DataType, DataTypeOf, Value, ValueOrView, ValueView};
use crate::weight_cache::WeightCache;

mod attention;
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

pub use attention::AddSoftmax;
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
pub use generate::{constant_of_shape, onehot, range, ConstantOfShape, EyeLike, OneHot, Range};
pub use identity::Identity;
pub use layout::{
    depth_to_space, expand, flatten, reshape, squeeze, DepthToSpace, DepthToSpaceMode, Expand,
    Flatten, Reshape, Shape, Size, Squeeze, Transpose, Unsqueeze,
};
pub use matmul::{gemm_op, matmul, FusedMatMul, Gemm, MatMul, MatMulInteger, MatMulIntegerToFloat};
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

#[deprecated = "renamed to `ValueOrView`"]
pub type InputOrOutput<'a> = ValueOrView<'a>;

#[deprecated = "renamed to `ValueView`"]
pub type Input<'a> = ValueView<'a>;

#[deprecated = "renamed to `Value`"]
pub type Output = Value;

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

impl OpError {
    /// Associate this error with a given operator input.
    fn with_input_index(self, index: usize) -> OpError {
        match self {
            Self::CastFailed(error) => OpError::InputCastFailed { index, error },
            Self::InputCastFailed { error, .. } => OpError::InputCastFailed { index, error },
            other => other,
        }
    }
}

impl From<DimensionError> for OpError {
    fn from(val: DimensionError) -> OpError {
        OpError::CastFailed(val.into())
    }
}

impl From<CastError> for OpError {
    fn from(val: CastError) -> OpError {
        OpError::CastFailed(val)
    }
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

    /// Run an operator with a mutable input and extract the first output.
    fn run_simple_in_place<
        'a,
        M: Into<Value>,
        I: Into<InputList<'a>>,
        O: TryFrom<Value, Error = CastError>,
    >(
        &self,
        mut_input: M,
        inputs: I,
    ) -> Result<O, OpError> {
        let pool = TensorPool::new();
        let inputs = inputs.into();
        let ctx = OpRunContext::new(&pool, &inputs);
        let output = self.run_in_place(mut_input.into(), &ctx)?;
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

impl<'a> From<()> for InputList<'a> {
    fn from(_: ()) -> InputList<'a> {
        Self::default()
    }
}

impl<'a, I1: Into<ValueView<'a>>> From<(I1,)> for InputList<'a> {
    fn from((a,): (I1,)) -> InputList<'a> {
        InputList::from(&[a.into()])
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
    use rten_tensor::{Tensor, TensorView};

    use super::Operator;
    use crate::downcast::DowncastDyn;
    use crate::ops::{Add, InputList, OpError, Sub};
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
    fn test_downcast_operator() {
        let add_op = Add {};
        let sub_op = Sub {};

        let add_op_dyn: &dyn Operator = &add_op;
        let sub_op_dyn: &dyn Operator = &sub_op;

        assert!(add_op_dyn.downcast_ref::<Add>().is_some());
        assert!(sub_op_dyn.downcast_ref::<Sub>().is_some());
    }
}
