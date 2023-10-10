//! wasnn_tensor provides multi-dimensional arrays, commonly referred to as
//! _tensors_ in a machine learning context. The tensor types are divided into two
//! sets, one for tensors where the dimension count is specified at compile
//! time, and one where it is determined at runtime.
//!
//! Each tensor is a combination of data and a layout. The data can be owned,
//! borrowed or mutably borrowed. This is analagous to `Vec<T>`, `&[T]` and
//! `&mut [T]` for 1D arrays. The layout determines the range of valid indices
//! for each array and how indices are mapped to elements in the data.
//!
//! # Static rank tensors
//!
//! [NdTensorBase] is the base type for tensors with a static dimension count.
//! It is normally used via one of the type aliases [NdTensor], [NdTensorView]
//! or [NdTensorViewMut]. [NdTensor] owns its elements, [NdTensorView] is an
//! immutable view of an `&[T]` and [NdTensorViewMut] is a mutable view of an
//! `&mut [T]`.
//!
//! All static-rank tensor variants implement the [Layout] trait which allows
//! querying the layout and the [NdView] trait which allows read-only operations
//! on all variants. Read-only views ([NdTensorView]) provides specialized
//! versions of [View] methods which preserves lifetimes of the underlying
//! buffer.
//!
//! # Dynamic rank tensors
//!
//! [TensorBase] is the base type for tensors with a dynamic dimension count.
//! It is normally used via one of the type aliases [Tensor], [TensorView] or
//! [TensorViewMut]. [Tensor] owns its elements, [TensorView] is an immutable
//! view of an `&[T]` and [TensorViewMut] is a mutable view of an `&mut [T]`.
//!
//! All dynamic-rank tensor variants implement the [Layout] trait which allows
//! querying the layout and the [View] trait which allows read-only operations
//! on all variants. Read-only views ([TensorView]) provides specialized
//! versions of [View] methods which preserves lifetimes of the underlying
//! buffer.

mod errors;
mod index_iterator;
mod iterators;
mod layout;
mod macros;
mod ndtensor;
mod overlap;
mod range;
mod tensor;

pub use index_iterator::{DynIndices, Indices, NdIndices};
pub use iterators::{AxisIter, AxisIterMut, BroadcastIter, Iter, IterMut, Offsets};
pub use layout::{is_valid_permutation, DynLayout, Layout, MatrixLayout};
pub use ndtensor::{
    Matrix, MatrixMut, NdTensor, NdTensorBase, NdTensorView, NdTensorViewMut, NdView,
};
pub use range::{IntoSliceItems, SliceItem, SliceRange};
pub use tensor::{Tensor, TensorBase, TensorView, TensorViewMut, View};

// These modules are public for use by other crates in this repo, but
// currently considered internal to the project.
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod test_util;
