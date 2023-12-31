//! rten_tensor provides multi-dimensional arrays, commonly referred to as
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
//! immutable view and [NdTensorViewMut] is a mutable view.
//!
//! [NdTensorBase] implements the [Layout] and [NdView] traits. The [Layout]
//! trait provides information about the shape, rank and strides of the tensor.
//! The [NdView] trait provides methods for reading and slicing the tensor.
//! Read-only views ([NdTensorView]) provide specialized versions of [NdView]
//! methods which preserve lifetimes of the data.
//!
//! # Dynamic rank tensors
//!
//! [TensorBase] is the base type for tensors with a dynamic dimension count.
//! It is normally used via one of the type aliases [Tensor], [TensorView] or
//! [TensorViewMut]. [Tensor] owns its elements, [TensorView] is an immutable
//! view and [TensorViewMut] is a mutable view.
//!
//! [TensorBase] implements the [Layout] and [View] traits. The [Layout]
//! trait provides information about the shape, rank and strides of the tensor.
//! The [View] trait provides methods for reading and slicing the tensor.
//! Read-only views ([TensorView]) provide specialized versions of [View]
//! methods which preserve lifetimes of the data.

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
pub use iterators::{
    AxisChunks, AxisChunksMut, AxisIter, AxisIterMut, BroadcastIter, InnerIter, InnerIterMut, Iter,
    IterMut, Lanes, LanesMut, Offsets,
};
pub use layout::{is_valid_permutation, DynLayout, Layout, MatrixLayout};
pub use ndtensor::{
    Matrix, MatrixMut, NdTensor, NdTensorBase, NdTensorView, NdTensorViewMut, NdView,
};
pub use range::{to_slice_items, DynSliceItems, IntoSliceItems, SliceItem, SliceRange};
pub use tensor::{RandomSource, Tensor, TensorBase, TensorView, TensorViewMut, View};

/// This module provides a convenient way to import the most common traits
/// from this library via a glob import.
pub mod prelude {
    pub use super::{Layout, NdView, View};
}

// These modules are public for use by other crates in this repo, but
// currently considered internal to the project.
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod test_util;
