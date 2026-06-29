//! Dynamically-typed tensor containers.
//!
//! Serialization formats can store tensors of various element types within a
//! single file. [`Value`] and [`View`] are enums that hold a tensor of
//! any [supported element type](Element), allowing readers to return tensors
//! whose type is only known at runtime, and writers to accept a heterogeneous
//! collection of tensors.

use std::error::Error;
use std::fmt;

use rten_tensor::storage::ViewData;
use rten_tensor::{AsView, DynLayout, Layout, Scalar, Tensor, TensorBase, TensorView};

mod private {
    /// Prevents [`Element`](super::Element) from being implemented outside this
    /// crate.
    pub trait Sealed {}
}

/// The element type of a tensor stored in a serialization format.
///
/// This enumerates the element types that have a corresponding Rust primitive
/// and are supported by all of this crate's formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DataType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

/// Error returned when a [`Value`] or [`View`] is accessed as the wrong element
/// type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeError {
    /// The element type that was requested.
    pub expected: DataType,
    /// The element type the value actually holds.
    pub actual: DataType,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected tensor with data type {:?}, but found {:?}",
            self.expected, self.actual
        )
    }
}

impl Error for TypeError {}

/// A Rust primitive type that can be stored in a [`Value`] or [`View`].
///
/// This trait is sealed and implemented for `bool`, the signed and unsigned
/// integer types up to 64 bits, and `f32`/`f64`.
pub trait Element: Scalar + private::Sealed + Sized {
    /// Extract an owned tensor, or fail if `value` holds a different type.
    ///
    /// Prefer [`Value::into_type`], which calls this.
    #[doc(hidden)]
    fn tensor_from_value(value: Value) -> Result<Tensor<Self>, TypeError>;

    /// Extract the typed view from `view`, or fail if it holds a different type.
    ///
    /// Prefer [`View::into_type`] / [`View::as_type`], which call this.
    #[doc(hidden)]
    fn view_from_view<'a>(view: View<'a>) -> Result<TensorView<'a, Self>, TypeError>;
}

/// Generate the [`Value`] and [`View`] enums plus the per-type [`Element`] impls
/// and `From`/`TryFrom` conversions.
///
/// Each row maps an enum variant to its Rust element type and [`DataType`].
macro_rules! define_value_types {
    ($($variant:ident => $ty:ty, $dtype:ident;)*) => {
        /// An owned tensor whose element type is determined at runtime.
        ///
        /// A typed [`Tensor`] can be converted into a `Value` using
        /// [`From`]/[`Into`]:
        ///
        /// ```
        /// use rten_serialize::Value;
        /// use rten_tensor::Tensor;
        ///
        /// let tensor = Tensor::from([1i32, 2, 3]);
        /// let value: Value = tensor.clone().into();
        /// assert_eq!(value.into_type::<i32>().unwrap(), tensor);
        /// ```
        ///
        /// To go the other way, use [`Value::into_type`] to extract an owned
        /// tensor or [`Value::as_type`] to borrow a typed view, both of which
        /// fail if the value holds a different element type.
        #[derive(Clone, Debug)]
        #[non_exhaustive]
        pub enum Value {
            $($variant(Tensor<$ty>),)*
        }

        /// A borrowed view of a tensor whose element type is determined at
        /// runtime.
        ///
        /// A typed [`TensorView`] can be converted into a `View` using
        /// [`From`]/[`Into`]:
        ///
        /// ```
        /// use rten_serialize::View;
        /// use rten_tensor::Tensor;
        /// use rten_tensor::prelude::*;
        ///
        /// let tensor = Tensor::from([1i32, 2, 3]);
        /// let view: View = tensor.view().into();
        /// assert_eq!(view.into_type::<i32>().unwrap().to_vec(), [1, 2, 3]);
        /// ```
        ///
        /// To go the other way, use [`View::into_type`] / [`View::as_type`] to
        /// extract a typed view, both of which fail if the view holds a
        /// different element type.
        #[derive(Clone, Debug)]
        #[non_exhaustive]
        pub enum View<'a> {
            $($variant(TensorView<'a, $ty>),)*
        }

        impl Value {
            /// Return the element type of this tensor.
            pub fn dtype(&self) -> DataType {
                match self {
                    $(Value::$variant(_) => DataType::$dtype,)*
                }
            }

            /// Return a borrowed view of this tensor.
            pub fn view(&self) -> View<'_> {
                match self {
                    $(Value::$variant(t) => View::$variant(t.view()),)*
                }
            }
        }

        impl<'a> View<'a> {
            /// Return the element type of this tensor.
            pub fn dtype(&self) -> DataType {
                match self {
                    $(View::$variant(_) => DataType::$dtype,)*
                }
            }
        }

        $(
            impl private::Sealed for $ty {}

            impl Element for $ty {
                fn tensor_from_value(value: Value) -> Result<Tensor<Self>, TypeError> {
                    match value {
                        Value::$variant(t) => Ok(t),
                        other => Err(TypeError {
                            expected: DataType::$dtype,
                            actual: other.dtype(),
                        }),
                    }
                }

                fn view_from_view<'a>(view: View<'a>) -> Result<TensorView<'a, Self>, TypeError> {
                    match view {
                        View::$variant(t) => Ok(t),
                        other => Err(TypeError {
                            expected: DataType::$dtype,
                            actual: other.dtype(),
                        }),
                    }
                }
            }

            impl From<Tensor<$ty>> for Value {
                fn from(tensor: Tensor<$ty>) -> Value {
                    Value::$variant(tensor)
                }
            }

            impl<'a, L: Layout + Into<DynLayout>> From<TensorBase<ViewData<'a, $ty>, L>> for View<'a> {
                fn from(view: TensorBase<ViewData<'a, $ty>, L>) -> View<'a> {
                    View::$variant(view.into_dyn())
                }
            }

            impl TryFrom<Value> for Tensor<$ty> {
                type Error = TypeError;

                fn try_from(value: Value) -> Result<Self, TypeError> {
                    <$ty as Element>::tensor_from_value(value)
                }
            }

            impl<'a> TryFrom<View<'a>> for TensorView<'a, $ty> {
                type Error = TypeError;

                fn try_from(view: View<'a>) -> Result<Self, TypeError> {
                    <$ty as Element>::view_from_view(view)
                }
            }
        )*
    };
}

define_value_types! {
    Bool => bool, Bool;
    Int8 => i8, Int8;
    Int16 => i16, Int16;
    Int32 => i32, Int32;
    Int64 => i64, Int64;
    UInt8 => u8, UInt8;
    UInt16 => u16, UInt16;
    UInt32 => u32, UInt32;
    UInt64 => u64, UInt64;
    Float32 => f32, Float32;
    Float64 => f64, Float64;
}

/// Evaluate `$body` with `$T` bound to the Rust element type corresponding to a
/// runtime [`DataType`].
#[cfg(any(feature = "npy", feature = "npz"))]
macro_rules! dispatch_data_type {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::value::DataType::Bool => {
                type $T = bool;
                $body
            }
            $crate::value::DataType::Int8 => {
                type $T = i8;
                $body
            }
            $crate::value::DataType::Int16 => {
                type $T = i16;
                $body
            }
            $crate::value::DataType::Int32 => {
                type $T = i32;
                $body
            }
            $crate::value::DataType::Int64 => {
                type $T = i64;
                $body
            }
            $crate::value::DataType::UInt8 => {
                type $T = u8;
                $body
            }
            $crate::value::DataType::UInt16 => {
                type $T = u16;
                $body
            }
            $crate::value::DataType::UInt32 => {
                type $T = u32;
                $body
            }
            $crate::value::DataType::UInt64 => {
                type $T = u64;
                $body
            }
            $crate::value::DataType::Float32 => {
                type $T = f32;
                $body
            }
            $crate::value::DataType::Float64 => {
                type $T = f64;
                $body
            }
        }
    };
}
#[cfg(any(feature = "npy", feature = "npz"))]
pub(crate) use dispatch_data_type;

/// Evaluate `$body` with `$v` bound to the typed [`TensorView`] held by a
/// [`View`].
#[cfg(any(feature = "npy", feature = "npz"))]
macro_rules! match_view {
    ($view:expr, $v:ident => $body:expr) => {
        match $view {
            $crate::value::View::Bool($v) => $body,
            $crate::value::View::Int8($v) => $body,
            $crate::value::View::Int16($v) => $body,
            $crate::value::View::Int32($v) => $body,
            $crate::value::View::Int64($v) => $body,
            $crate::value::View::UInt8($v) => $body,
            $crate::value::View::UInt16($v) => $body,
            $crate::value::View::UInt32($v) => $body,
            $crate::value::View::UInt64($v) => $body,
            $crate::value::View::Float32($v) => $body,
            $crate::value::View::Float64($v) => $body,
        }
    };
}
#[cfg(any(feature = "npy", feature = "npz"))]
pub(crate) use match_view;

impl Value {
    /// Convert this value into a tensor of type `T`, or fail if it holds a
    /// different type.
    pub fn into_type<T: Element>(self) -> Result<Tensor<T>, TypeError> {
        T::tensor_from_value(self)
    }

    /// Borrow this value as a tensor view of type `T`, or fail if it holds a
    /// different type.
    pub fn as_type<T: Element>(&self) -> Result<TensorView<'_, T>, TypeError> {
        T::view_from_view(self.view())
    }
}

impl<'a> View<'a> {
    /// Convert this view into a tensor view of type `T`, or fail if it holds a
    /// different type.
    pub fn into_type<T: Element>(self) -> Result<TensorView<'a, T>, TypeError> {
        T::view_from_view(self)
    }

    /// Borrow this view as a tensor view of type `T`, or fail if it holds a
    /// different type.
    pub fn as_type<T: Element>(&self) -> Result<TensorView<'_, T>, TypeError> {
        T::view_from_view(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{DataType, Value, View};
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;

    #[test]
    fn test_value_from_into_type() {
        let tensor: Tensor<i32> = [[1, 2], [3, 4]].into();
        let value: Value = tensor.clone().into();

        assert_eq!(value.dtype(), DataType::Int32);
        assert_eq!(value.into_type::<i32>().unwrap(), tensor);
    }

    #[test]
    fn test_value_into_type_wrong_type_errors() {
        let value: Value = Tensor::<f32>::zeros(&[2, 2]).into();

        let err = value.into_type::<i32>().unwrap_err();

        assert_eq!(err.expected, DataType::Int32);
        assert_eq!(err.actual, DataType::Float32);
    }

    #[test]
    fn test_value_as_type_borrows() {
        let value: Value = Tensor::from([1i64, 2, 3]).into();

        let view = value.as_type::<i64>().unwrap();

        assert_eq!(view.to_vec(), [1, 2, 3]);
    }

    #[test]
    fn test_value_as_type_wrong_type_errors() {
        let value: Value = Tensor::from([1i64, 2, 3]).into();

        assert!(value.as_type::<u8>().is_err());
    }

    #[test]
    fn test_view_into_type() {
        let tensor: Tensor<f32> = [1., 2., 3.].into();
        let view: View = tensor.view().into();

        assert_eq!(view.dtype(), DataType::Float32);
        assert_eq!(view.into_type::<f32>().unwrap().to_vec(), [1., 2., 3.]);
    }
}
