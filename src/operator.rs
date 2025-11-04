//! The [`Operator`] trait for defining operators.

use std::any::Any;
use std::borrow::Cow;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};

use rten_gemm::PackedBMatrix;
use rten_tensor::errors::DimensionError;
use rten_tensor::{Layout, Storage, TensorBase};
use smallvec::SmallVec;

use crate::BufferPool;
use crate::graph::{CaptureEnv, Graph, RunError, RunOptions};
use crate::timing::Profiler;
use crate::value::{DataType, DataTypeOf, TryFromValueError, Value, ValueType, ValueView};
use crate::weight_cache::WeightCache;

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
            type Error = TryFromValueError;

            fn try_from(ppi: &'a PrepackedInput) -> Result<Self, Self::Error> {
                match ppi {
                    PrepackedInput::$variant(packed) => Ok(packed),
                    _ => Err(TryFromValueError::WrongType {
                        actual: ValueType::Tensor(ppi.dtype()),
                        expected: ValueType::Tensor(<$type as DataTypeOf>::dtype_of()),
                    }),
                }
            }
        }
    };
}
impl_prepacked_input_conversions!(f32, FloatBMatrix);
impl_prepacked_input_conversions!(i8, Int8BMatrix);

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

impl<S: Storage, L: Layout> IntoOpResult for TensorBase<S, L>
where
    Value: From<TensorBase<S, L>>,
{
    fn into_op_result(self) -> Result<OutputList, OpError> {
        let output: Value = self.into();
        Ok([output].into())
    }
}

impl<S: Storage, L: Layout> IntoOpResult for Result<TensorBase<S, L>, OpError>
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
    CastFailed(TryFromValueError),

    /// Casting an input to an expected type or rank failed.
    InputCastFailed {
        index: usize,
        error: TryFromValueError,
    },

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
    pub fn with_input_index(self, index: usize) -> OpError {
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

impl From<TryFromValueError> for OpError {
    fn from(val: TryFromValueError) -> OpError {
        OpError::CastFailed(val)
    }
}

impl From<Infallible> for OpError {
    fn from(x: Infallible) -> OpError {
        match x {}
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
    pool: &'a BufferPool,
    inputs: &'a InputList<'i>,
    n_outputs: Option<u32>,
    name: Option<&'a str>,
}

impl<'a, 'i> OpRunContext<'a, 'i> {
    pub fn new(pool: &'a BufferPool, inputs: &'a InputList<'i>) -> Self {
        OpRunContext {
            pool,
            inputs,
            n_outputs: None,
            name: None,
        }
    }

    /// Construct a new context with the same properties but different inputs.
    ///
    /// This is useful when one operator wants to delegate to another.
    pub fn with_new_inputs<'b, 'il>(&self, inputs: &'b InputList<'il>) -> OpRunContext<'b, 'il>
    where
        'a: 'b,
    {
        OpRunContext { inputs, ..*self }
    }

    /// The pool which should be used to allocate large buffers.
    pub fn pool(&self) -> &BufferPool {
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
    /// `ctx` provides access to operator inputs and the [`BufferPool`] from
    /// which the output and temporary buffers should be allocated.
    ///
    /// For operators which have subgraphs (see
    /// [`as_subgraph_op`](Operator::as_subgraph_op)), the
    /// [`SubgraphOperator::run_subgraph`] method should be used instead.
    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError>;

    /// Return the maximum number of inputs this operator accepts.
    ///
    /// This can return `None` for variadic inputs with no limit.
    fn max_inputs(&self) -> Option<usize>;

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

    /// Return the [`SubgraphOperator`] implementation for this operator, if
    /// this operator has subgraphs.
    fn as_subgraph_op(&self) -> Option<&dyn SubgraphOperator> {
        None
    }
}

impl dyn Operator {
    /// Downcast this operator to a concrete type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref()
    }
}

impl dyn Operator + Send + Sync {
    /// Downcast this operator to a concrete type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref()
    }
}

/// Trait for operators which contain subgraphs, such as `If`, `Loop` etc.
pub trait SubgraphOperator: Operator {
    /// Return a list of subgraphs used by this operator.
    fn subgraphs(&self) -> SmallVec<[&Graph; 2]> {
        SmallVec::new()
    }

    /// Execute the operator with the given inputs and captured values.
    ///
    /// This should be used instead of [`Operator::run`] for operators that
    /// implement this trait.
    fn run_subgraph<'a>(
        &'a self,
        ctx: &OpRunContext,
        #[allow(unused)] captures: CaptureEnv,
        #[allow(unused)] weight_cache: Option<&[WeightCache]>,
        #[allow(unused)] profiler: Option<&mut Profiler<'a>>,
        #[allow(unused)] run_opts: Option<RunOptions>,
    ) -> Result<OutputList, RunError>;
}

/// Convenience methods that make it easier to run operators in tests.
#[cfg(test)]
pub trait OperatorExt: Operator {
    /// Run an operator and extract the first output as a tensor with a given
    /// type.
    ///
    /// `inputs` is a tuple of tensor references or other values that can be
    /// converted to [`ValueView`].
    fn run_simple<'a, I: Into<InputList<'a>>, O: TryFrom<Value>>(
        &self,
        inputs: I,
    ) -> Result<O, OpError>
    where
        OpError: From<<O as TryFrom<Value>>::Error>,
    {
        let pool = BufferPool::new();
        let inputs = inputs.into();
        let ctx = OpRunContext::new(&pool, &inputs);
        let mut outputs = self.run(&ctx)?;
        Ok(outputs.remove(0).try_into()?)
    }

    /// Run an operator with a mutable input and extract the first output.
    fn run_simple_in_place<'a, M: Into<Value>, I: Into<InputList<'a>>, O: TryFrom<Value>>(
        &self,
        mut_input: M,
        inputs: I,
    ) -> Result<O, OpError>
    where
        OpError: From<<O as TryFrom<Value>>::Error>,
    {
        let pool = BufferPool::new();
        let inputs = inputs.into();
        let ctx = OpRunContext::new(&pool, &inputs);
        let output = self.run_in_place(mut_input.into(), &ctx)?;
        let typed_output = output.try_into()?;
        Ok(typed_output)
    }
}

#[cfg(test)]
impl<O: ?Sized + Operator> OperatorExt for O {}

/// List of inputs for an operator evaluation.
///
/// This is an owned or borrowed collection of `Option<ValueView>`s with methods
/// to conveniently extract inputs and produce appropriate errors if inputs are
/// missing or of the wrong type.
///
/// An InputList can be constructed from tuples of `impl Into<ValueView>` types
/// (eg. `TensorView`, `&Tensor`) via `Into`. It can also be created or
/// extended from iterators of `ValueView`s or `Option<ValueView>`s.
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
        T: TryFrom<ValueView<'a>, Error = TryFromValueError>,
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
        T: TryFrom<ValueView<'a>, Error = TryFromValueError>,
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

impl<'a> Extend<ValueView<'a>> for InputList<'a> {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = ValueView<'a>>,
    {
        for item in iter {
            self.push(item);
        }
    }
}

impl<'a> Extend<Option<ValueView<'a>>> for InputList<'a> {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Option<ValueView<'a>>>,
    {
        for item in iter {
            self.push_optional(item);
        }
    }
}

impl<'a, A> FromIterator<A> for InputList<'a>
where
    InputList<'a>: Extend<A>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = A>,
    {
        let mut list = InputList::new();
        list.extend(iter);
        list
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{Tensor, TensorView};

    use crate::operator::{InputList, OpError, Operator};
    use crate::ops::{Add, Sub};

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
