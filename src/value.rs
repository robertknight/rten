//! Value types used for operator inputs and outputs.

use std::error::Error;
use std::fmt;
use std::fmt::Display;

use rten_tensor::errors::DimensionError;
use rten_tensor::{
    Alloc, AsView, DynIndices, DynLayout, GlobalAlloc, Layout, MutLayout, NdTensor, NdTensorView,
    Storage, Tensor, TensorBase, TensorView, ViewData,
};
use smallvec::SmallVec;

use crate::buffer_pool::{Buffer, BufferPool, ExtractBuffer};

/// Enum specifying the data type of a tensor.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DataType {
    Int32,
    Float,
    Int8,
    UInt8,
}

impl DataType {
    /** Return the size of elements of this type in bytes. */
    pub fn size(self) -> u8 {
        match self {
            DataType::Int32 | DataType::Float => 4,
            DataType::Int8 | DataType::UInt8 => 1,
        }
    }
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

/// Metadata about a tensor.
///
/// This is used in profiling and errors which need to contain metadata about
/// a tensor but not the content.
#[derive(Clone, Debug, Eq, PartialEq)]
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

/// Errors when casting a [`Value`] or [`ValueView`] to a tensor of a specific
/// type and/or rank.
#[derive(Debug, Eq, PartialEq)]
pub enum CastError {
    /// The number of dimensions does not match.
    WrongRank {
        actual: usize,
        expected: usize,
    },

    /// The data type of elements does not match.
    WrongType {
        actual: DataType,
        expected: DataType,
    },

    ExpectedSequence,
}

impl Display for CastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongRank { actual, expected } => {
                write!(
                    f,
                    "expected tensor with {} dims but has {} dims",
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
            Self::ExpectedSequence => {
                write!(f, "value is not a sequence")
            }
        }
    }
}

impl Error for CastError {}

impl From<DimensionError> for CastError {
    fn from(val: DimensionError) -> CastError {
        let DimensionError { actual, expected } = val;
        CastError::WrongRank { actual, expected }
    }
}

enum ValueLayout<'a> {
    /// Layout used for tensor values.
    Tensor(&'a DynLayout),

    /// Layout used for sequences, where the value is the length and the stride
    /// is always 1.
    Vector(usize),
}

impl<'a> From<&'a DynLayout> for ValueLayout<'a> {
    fn from(layout: &'a DynLayout) -> Self {
        Self::Tensor(layout)
    }
}

/// Generate the body of a [`Layout`] impl for a type which wraps an
/// underlying layout.
macro_rules! impl_proxy_layout {
    () => {
        type Index<'b> = SmallVec<[usize; 4]>;
        type Indices = DynIndices;

        fn ndim(&self) -> usize {
            match self.layout() {
                ValueLayout::Tensor(layout) => layout.ndim(),
                ValueLayout::Vector(_) => 1,
            }
        }

        fn offset(&self, index: Self::Index<'_>) -> Option<usize> {
            match self.layout() {
                ValueLayout::Tensor(layout) => layout.offset(&index),
                ValueLayout::Vector(len) => index
                    .get(0)
                    .and_then(|&idx| if idx < len { Some(idx) } else { None }),
            }
        }

        fn len(&self) -> usize {
            match self.layout() {
                ValueLayout::Tensor(layout) => layout.len(),
                ValueLayout::Vector(len) => len,
            }
        }

        fn is_empty(&self) -> bool {
            match self.layout() {
                ValueLayout::Tensor(layout) => layout.is_empty(),
                ValueLayout::Vector(len) => len == 0,
            }
        }

        fn shape(&self) -> Self::Index<'_> {
            match self.layout() {
                ValueLayout::Tensor(layout) => SmallVec::from_slice(layout.shape()),
                ValueLayout::Vector(len) => SmallVec::from_slice(&[len]),
            }
        }

        fn size(&self, dim: usize) -> usize {
            match self.layout() {
                ValueLayout::Tensor(layout) => layout.size(dim),
                ValueLayout::Vector(len) => [len][dim],
            }
        }

        fn strides(&self) -> Self::Index<'_> {
            match self.layout() {
                ValueLayout::Tensor(layout) => SmallVec::from_slice(layout.strides()),
                ValueLayout::Vector(_) => SmallVec::from_slice(&[1]),
            }
        }

        fn stride(&self, dim: usize) -> usize {
            match self.layout() {
                ValueLayout::Tensor(layout) => layout.stride(dim),
                ValueLayout::Vector(_) => [1][dim],
            }
        }

        fn indices(&self) -> Self::Indices {
            match self.layout() {
                ValueLayout::Tensor(layout) => layout.indices(),
                ValueLayout::Vector(len) => DynIndices::from_shape(&[len]),
            }
        }
    };
}

/// A borrowed value that can be used as a model or operator input.
///
/// Each `ValueView` variant has a counterpart [`Value`] that is the owned value
/// of the same type.
#[derive(Clone)]
#[non_exhaustive]
pub enum ValueView<'a> {
    FloatTensor(TensorView<'a, f32>),
    Int32Tensor(TensorView<'a, i32>),
    Int8Tensor(TensorView<'a, i8>),
    UInt8Tensor(TensorView<'a, u8>),
    Sequence(&'a Sequence),
}

impl ValueView<'_> {
    /// Return the data type of elements in this tensor.
    pub fn dtype(&self) -> DataType {
        match self {
            Self::FloatTensor(_) => DataType::Float,
            Self::Int32Tensor(_) => DataType::Int32,
            Self::Int8Tensor(_) => DataType::Int8,
            Self::UInt8Tensor(_) => DataType::UInt8,
            Self::Sequence(seq) => seq.dtype(),
        }
    }

    /// Return an owned copy of this value.
    pub fn to_owned(&self) -> Value {
        self.to_owned_in(GlobalAlloc::new())
    }

    /// Variant of [`to_owned`](Self::to_owned) that takes an allocator.
    pub fn to_owned_in<A: Alloc>(&self, alloc: A) -> Value {
        match self {
            ValueView::FloatTensor(t) => t.to_tensor_in(alloc).into(),
            ValueView::Int32Tensor(t) => t.to_tensor_in(alloc).into(),
            ValueView::Int8Tensor(t) => t.to_tensor_in(alloc).into(),
            ValueView::UInt8Tensor(t) => t.to_tensor_in(alloc).into(),
            ValueView::Sequence(seq) => (*seq).clone().into(),
        }
    }

    /// Extract shape and data type information from this tensor.
    pub fn to_meta(&self) -> ValueMeta {
        ValueMeta {
            shape: self.shape().to_vec(),
            dtype: self.dtype(),
        }
    }

    fn layout(&self) -> ValueLayout<'_> {
        match self {
            ValueView::FloatTensor(t) => t.layout().into(),
            ValueView::Int32Tensor(t) => t.layout().into(),
            ValueView::Int8Tensor(t) => t.layout().into(),
            ValueView::UInt8Tensor(t) => t.layout().into(),
            ValueView::Sequence(seq) => ValueLayout::Vector(seq.len()),
        }
    }
}

impl Layout for ValueView<'_> {
    impl_proxy_layout!();
}

macro_rules! impl_value_view_conversions {
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

impl_value_view_conversions!(FloatTensor, f32);
impl_value_view_conversions!(Int32Tensor, i32);
impl_value_view_conversions!(Int8Tensor, i8);
impl_value_view_conversions!(UInt8Tensor, u8);

impl<'a> From<&'a Value> for ValueView<'a> {
    fn from(output: &'a Value) -> ValueView<'a> {
        match output {
            Value::FloatTensor(t) => ValueView::FloatTensor(t.view()),
            Value::Int32Tensor(t) => ValueView::Int32Tensor(t.view()),
            Value::Int8Tensor(t) => ValueView::Int8Tensor(t.view()),
            Value::UInt8Tensor(t) => ValueView::UInt8Tensor(t.view()),
            Value::Sequence(seq) => ValueView::Sequence(seq),
        }
    }
}

/// An owned value that can be used as an operator input or output.
///
/// Each `Value` variant has a [`ValueView`] counterpart that represents a
/// borrowed value of the same type.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Value {
    FloatTensor(Tensor<f32>),
    Int32Tensor(Tensor<i32>),
    Int8Tensor(Tensor<i8>),
    UInt8Tensor(Tensor<u8>),
    Sequence(Sequence),
}

impl Value {
    /// Return the data type of elements in this tensor.
    ///
    /// For sequence values, this returns the data type of tensors in the
    /// sequence.
    pub fn dtype(&self) -> DataType {
        match self {
            Self::FloatTensor(_) => DataType::Float,
            Self::Int32Tensor(_) => DataType::Int32,
            Self::Int8Tensor(_) => DataType::Int8,
            Self::UInt8Tensor(_) => DataType::UInt8,
            Self::Sequence(seq) => seq.dtype(),
        }
    }

    /// Return a borrowed view of this tensor.
    pub fn as_view(&self) -> ValueView<'_> {
        match self {
            Self::FloatTensor(ft) => ValueView::FloatTensor(ft.view()),
            Self::Int32Tensor(it) => ValueView::Int32Tensor(it.view()),
            Self::Int8Tensor(it) => ValueView::Int8Tensor(it.view()),
            Self::UInt8Tensor(it) => ValueView::UInt8Tensor(it.view()),
            Self::Sequence(seq) => ValueView::Sequence(seq),
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
    pub(crate) fn add_to_pool(self, pool: &BufferPool) {
        match self {
            Self::FloatTensor(t) => t.extract_buffer().map(|buf| pool.add(buf)),
            Self::Int32Tensor(t) => t.extract_buffer().map(|buf| pool.add(buf)),
            Self::Int8Tensor(t) => t.extract_buffer().map(|buf| pool.add(buf)),
            Self::UInt8Tensor(t) => t.extract_buffer().map(|buf| pool.add(buf)),
            Self::Sequence(seq) => {
                seq.add_to_pool(pool);
                Some(())
            }
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

    fn layout(&self) -> ValueLayout<'_> {
        match self {
            Value::Int32Tensor(t) => t.layout().into(),
            Value::Int8Tensor(t) => t.layout().into(),
            Value::UInt8Tensor(t) => t.layout().into(),
            Value::FloatTensor(t) => t.layout().into(),
            Value::Sequence(seq) => ValueLayout::Vector(seq.len()),
        }
    }
}

impl Layout for Value {
    impl_proxy_layout!();
}

impl ExtractBuffer for Value {
    fn extract_buffer(self) -> Option<Buffer> {
        match self {
            Value::Int32Tensor(t) => t.extract_buffer(),
            Value::Int8Tensor(t) => t.extract_buffer(),
            Value::UInt8Tensor(t) => t.extract_buffer(),
            Value::FloatTensor(t) => t.extract_buffer(),
            Value::Sequence(_) => {
                // We can't implement `ExtractBuffer` for sequences because
                // there may be more than one buffer.
                None
            }
        }
    }
}

/// Declare conversions between `Value` and `Tensor<T>` / `NdTensor<T, N>`.
macro_rules! impl_value_conversions {
    ($variant:ident, $element_type:ty) => {
        // T => Value
        impl From<$element_type> for Value {
            fn from(scalar: $element_type) -> Value {
                Value::$variant(Tensor::from(scalar))
            }
        }

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

impl From<Sequence> for Value {
    fn from(seq: Sequence) -> Value {
        Value::Sequence(seq)
    }
}

/// An owned or borrowed value that can be used as a model or operator input.
#[derive(Clone)]
pub enum ValueOrView<'a> {
    /// A borrowed view (like a slice)
    View(ValueView<'a>),
    /// An owned value (like a `Vec<T>`)
    Value(Value),
}

impl ValueOrView<'_> {
    /// Convert this value to a tensor view.
    pub fn as_view(&self) -> ValueView<'_> {
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

    /// Convert this value to an owned tensor.
    pub fn into_owned(self) -> Value {
        match self {
            ValueOrView::View(view) => view.to_owned(),
            ValueOrView::Value(value) => value,
        }
    }

    fn layout(&self) -> ValueLayout<'_> {
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

impl ExtractBuffer for ValueOrView<'_> {
    fn extract_buffer(self) -> Option<Buffer> {
        match self {
            Self::View(_) => None,
            Self::Value(val) => val.extract_buffer(),
        }
    }
}

/// A scalar value with runtime-determined type.
#[derive(Debug, PartialEq)]
pub enum Scalar {
    Int(i32),
    Float(f32),
}

/// Errors from operations on [`Sequence`] values.
#[derive(Debug)]
pub enum SequenceError {
    InvalidPosition,
    InvalidType,
}

/// A list of tensors.
///
/// The type of list is dynamic but tensors within a list all have the same
/// type. The rank and shape of each tensor in the list can vary.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Sequence {
    Float(Vec<Tensor<f32>>),
    Int32(Vec<Tensor<i32>>),
    Int8(Vec<Tensor<i8>>),
    UInt8(Vec<Tensor<u8>>),
}

impl Sequence {
    /// Create an empty sequence with the given data type.
    pub fn new(dtype: DataType) -> Sequence {
        match dtype {
            DataType::Int32 => Vec::<Tensor<i32>>::new().into(),
            DataType::Int8 => Vec::<Tensor<i8>>::new().into(),
            DataType::UInt8 => Vec::<Tensor<u8>>::new().into(),
            DataType::Float => Vec::<Tensor<f32>>::new().into(),
        }
    }

    /// Return the data type of elements in each item of the sequence.
    pub fn dtype(&self) -> DataType {
        match self {
            Self::Float(_) => DataType::Float,
            Self::Int32(_) => DataType::Int32,
            Self::Int8(_) => DataType::Int8,
            Self::UInt8(_) => DataType::UInt8,
        }
    }

    /// Return the number of items in the sequence.
    pub fn len(&self) -> usize {
        match self {
            Self::Float(floats) => floats.len(),
            Self::Int32(ints) => ints.len(),
            Self::Int8(ints) => ints.len(),
            Self::UInt8(ints) => ints.len(),
        }
    }

    /// Return true if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the value at a given index.
    pub fn at(&self, index: usize) -> Option<ValueView<'_>> {
        match self {
            Self::Float(floats) => Self::at_impl(floats, index),
            Self::Int32(ints) => Self::at_impl(ints, index),
            Self::Int8(ints) => Self::at_impl(ints, index),
            Self::UInt8(ints) => Self::at_impl(ints, index),
        }
    }

    fn at_impl<T>(items: &[T], index: usize) -> Option<ValueView<'_>>
    where
        for<'a> ValueView<'a>: From<&'a T>,
    {
        items.get(index).map(|it| it.into())
    }

    /// Insert an element into the given position in this sequence.
    ///
    /// The operation will fail if the position is not in the range `[0,
    /// self.len()]` or the value has a different type than the sequence.
    pub fn insert(&mut self, index: usize, val: Value) -> Result<(), SequenceError> {
        if index > self.len() {
            return Err(SequenceError::InvalidPosition);
        }
        match (self, val) {
            (Self::Float(floats), Value::FloatTensor(val)) => floats.insert(index, val),
            (Self::Int32(ints), Value::Int32Tensor(val)) => ints.insert(index, val),
            (Self::Int8(ints), Value::Int8Tensor(val)) => ints.insert(index, val),
            (Self::UInt8(ints), Value::UInt8Tensor(val)) => ints.insert(index, val),
            _ => {
                return Err(SequenceError::InvalidType);
            }
        }
        Ok(())
    }

    /// Remove the element at the given position in the sequence.
    pub fn remove(&mut self, index: usize) -> Result<Value, SequenceError> {
        if index >= self.len() {
            return Err(SequenceError::InvalidPosition);
        }
        let value: Value = match self {
            Self::Float(floats) => floats.remove(index).into(),
            Self::Int32(ints) => ints.remove(index).into(),
            Self::Int8(ints) => ints.remove(index).into(),
            Self::UInt8(ints) => ints.remove(index).into(),
        };
        Ok(value)
    }

    /// Return an iterator over values in the sequence.
    pub fn iter(&self) -> impl Iterator<Item = ValueView<'_>> {
        (0..self.len()).map(|i| self.at(i).unwrap())
    }

    /// Extract the underlying buffers from tensors in this sequence and add
    /// them to `pool`.
    fn add_to_pool(self, pool: &BufferPool) {
        match self {
            Self::Float(floats) => Self::add_items_to_pool(floats, pool),
            Self::Int32(ints) => Self::add_items_to_pool(ints, pool),
            Self::Int8(ints) => Self::add_items_to_pool(ints, pool),
            Self::UInt8(ints) => Self::add_items_to_pool(ints, pool),
        }
    }

    fn add_items_to_pool<T: ExtractBuffer>(items: Vec<T>, pool: &BufferPool) {
        for item in items {
            if let Some(buf) = item.extract_buffer() {
                pool.add(buf);
            }
        }
    }
}

macro_rules! impl_sequence_conversions {
    ($variant:ident, $seq_type:ty) => {
        impl From<Vec<$seq_type>> for Sequence {
            fn from(val: Vec<$seq_type>) -> Sequence {
                Sequence::$variant(val)
            }
        }

        impl<const N: usize> From<[$seq_type; N]> for Sequence {
            fn from(val: [$seq_type; N]) -> Sequence {
                Sequence::$variant(val.into())
            }
        }
    };
}
impl_sequence_conversions!(Float, Tensor<f32>);
impl_sequence_conversions!(Int32, Tensor<i32>);
impl_sequence_conversions!(Int8, Tensor<i8>);
impl_sequence_conversions!(UInt8, Tensor<u8>);

impl<'a> TryFrom<ValueView<'a>> for &'a Sequence {
    type Error = CastError;

    fn try_from(val: ValueView<'a>) -> Result<Self, Self::Error> {
        match val {
            ValueView::Sequence(seq) => Ok(seq),
            _ => Err(CastError::ExpectedSequence),
        }
    }
}

impl TryFrom<Value> for Sequence {
    type Error = CastError;

    fn try_from(val: Value) -> Result<Self, Self::Error> {
        match val {
            Value::Sequence(seq) => Ok(seq),
            _ => Err(CastError::ExpectedSequence),
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};

    use super::{CastError, DataType, Value, ValueView};

    #[test]
    fn test_value_view_from_tensor() {
        let tensor = NdTensor::<i32, 3>::zeros([1, 2, 3]);
        let input: ValueView = tensor.view().into();
        assert!(matches!(input, ValueView::Int32Tensor(_)));
        assert_eq!(input.shape().as_slice(), &[1, 2, 3]);

        let tensor = NdTensor::<f32, 2>::zeros([5, 5]);
        let input: ValueView = tensor.view().into();
        assert!(matches!(input, ValueView::FloatTensor(_)));
        assert_eq!(input.shape().as_slice(), &[5, 5]);
    }

    #[test]
    fn test_tensor_from_value() {
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
    fn test_tensor_view_from_value() {
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
}
