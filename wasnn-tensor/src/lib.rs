//! wasnn_tensor provides multi-dimensional arrays, commonly referred to as
//! _tensors_ in a machine learning context. The tensor types are divided into two
//! sets, one for tensors where the dimension count is specified at compile
//! time, and one where it is determined at runtime.
//!
//! Each tensor is a combination of data and a layout. The data can be owned,
//! borrowed or mutably borrowed. This is analagous to `Vec<T>`, `&[T]` and
//! `&mut [T]` for 1D arrays. The layout determines the range of valid indices
//! for each array and how indices are mapped to elements in the data.
//! Information about the layout of a tensor is provided via the [Layout] trait.
//!
//! # Static rank tensors
//!
//! [NdTensorBase] is the base type for tensors with a static dimension count.
//! It is normally used via one of the type aliases [NdTensor], [NdTensorView]
//! or [NdTensorViewMut]. [NdTensor] owns its elements, [NdTensorView] is an
//! immutable view of an `&[T]` and [NdTensorViewMut] is a mutable view of an
//! `&mut [T]`.
//!
//! # Dynamic rank tensors
//!
//! [TensorBase] is the base type for tensors with a dynamic dimension count.
//! It is normally used via one of the type aliases [Tensor], [TensorView] or
//! [TensorViewMut]. [Tensor] owns its elements, [TensorView] is an immutable
//! view of an `&[T]` and [TensorViewMut] is a mutable view of an `&mut [T]`.

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
    Matrix, MatrixMut, NdTensor, NdTensorBase, NdTensorCommon, NdTensorView, NdTensorViewMut,
};
pub use range::{IntoSliceItems, SliceItem, SliceRange};
pub use tensor::{Tensor, TensorBase, TensorCommon, TensorView, TensorViewMut};

// These modules are public for use by other crates in this repo, but
// currently considered internal to the project.
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod test_util;
