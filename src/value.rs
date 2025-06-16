//! Value types used for operator inputs and outputs.

use std::error::Error;
use std::fmt;
use std::fmt::Display;

use rten_tensor::errors::DimensionError;
use rten_tensor::{
    AsView, DynLayout, Layout, MutLayout, NdTensor, NdTensorView, Storage, Tensor, TensorBase,
    TensorView, ViewData,
};

use crate::tensor_pool::{ExtractBuffer, TensorPool};

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

impl From<DimensionError> for CastError {
    fn from(val: DimensionError) -> CastError {
        let DimensionError { actual, expected } = val;
        CastError::WrongRank { actual, expected }
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
        }
    }
}

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
        assert_eq!(input.shape(), &[1, 2, 3]);

        let tensor = NdTensor::<f32, 2>::zeros([5, 5]);
        let input: ValueView = tensor.view().into();
        assert!(matches!(input, ValueView::FloatTensor(_)));
        assert_eq!(input.shape(), &[5, 5]);
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
