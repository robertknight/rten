//! rten_tensor provides multi-dimensional arrays, commonly referred to as
//! _tensors_ in a machine learning context.
//!
//! Each tensor is a combination of data and a layout. The data can be owned,
//! borrowed or mutably borrowed. This is analagous to `Vec<T>`, `&[T]` and
//! `&mut [T]` for 1D arrays. The layout determines the number of dimensions
//! (the _rank_), the size of each dimension, and the strides (gap between
//! successive indices along a given dimension).
//!
//! # Key types and traits
//!
//! The base type for all tensors is [TensorBase]. This is not normally used
//! directly but instead via a type alias, depending on whether the number of
//! dimensions (the _rank_) of the tensor is known at compile time or only
//! at runtime, as well as whether the tensor owns, borrows or mutably borrows
//! its data.
//!
//! | Rank    | Owned (like `Vec<T>`) | Borrowed (like `&[T]`) | Mutably borrowed |
//! | ----    | --------------------- | ---------------------- | ---------------- |
//! | Static  | [NdTensor] | [NdTensorView] | [NdTensorViewMut] |
//! | Dynamic | [Tensor]   | [TensorView]   | [TensorViewMut]   |
//!
//! All tensors implement the [Layout] trait, which provide methods to query
//! the shape, dimension count and strides of the tensor. Tensor views provide
//! various methods for indexing, iterating, slicing and transforming them.
//! The [AsView] trait provides access to these methods for owned and mutably
//! borrowed tensors. Conceptually it is similar to how [Deref](std::ops::Deref)
//! allows accesing methods for `&[T]` on a `Vec<T>`. The preferred way to
//! import the traits is via the prelude:
//!
//! ```
//! use rten_tensor::prelude::*;
//! use rten_tensor::NdTensor;
//!
//! let tensor = NdTensor::from([[1, 2], [3, 4]]);
//!
//! let transposed_elems: Vec<_> = tensor.transposed().iter().copied().collect();
//! assert_eq!(transposed_elems, [1, 3, 2, 4]);
//! ```
//!
//! # Serialization
//!
//! Tensors can be serialized and deserialized using [serde](https://serde.rs)
//! if the `serde` feature is enabled. The serialized representation of a
//! tensor includes its shape and elements in row-major (C) order. The JSON
//! serialization of a matrix (`NdTensor<f32, 2>`) looks like this for example:
//!
//! ```json
//! {
//!   "shape": [2, 2],
//!   "data": [0.5, 1.0, 1.5, 2.0]
//! }
//! ```

mod copy;
pub mod errors;
mod index_iterator;
mod iterators;
mod layout;
mod macros;
mod overlap;
mod slice_range;
mod storage;
pub mod type_num;

mod impl_debug;
#[cfg(feature = "serde")]
mod impl_serialize;
mod tensor;

/// Trait for sources of random data for tensors, for use with [`Tensor::rand`].
pub trait RandomSource<T> {
    /// Generate the next random value.
    fn next(&mut self) -> T;
}

/// Storage allocation trait.
///
/// This is used by various methods on [`TensorBase`] with an `_in` suffix,
/// which allow the caller to control the allocation of the data buffer for
/// the returned owned tensor.
pub trait Alloc {
    /// Allocate storage for an owned tensor.
    ///
    /// The returned `Vec` should be empty but have the given capacity.
    fn alloc<T>(&self, capacity: usize) -> Vec<T>;
}

impl<A: Alloc> Alloc for &A {
    fn alloc<T>(&self, capacity: usize) -> Vec<T> {
        A::alloc(self, capacity)
    }
}

/// Implementation of [`Alloc`] which wraps the global allocator.
pub struct GlobalAlloc {}

impl GlobalAlloc {
    pub const fn new() -> GlobalAlloc {
        GlobalAlloc {}
    }
}

impl Default for GlobalAlloc {
    fn default() -> Self {
        Self::new()
    }
}

impl Alloc for GlobalAlloc {
    fn alloc<T>(&self, capacity: usize) -> Vec<T> {
        Vec::with_capacity(capacity)
    }
}

pub use index_iterator::{DynIndices, Indices, NdIndices};
pub use iterators::{
    AxisChunks, AxisChunksMut, AxisIter, AxisIterMut, InnerIter, InnerIterMut, Iter, IterMut,
    Lanes, LanesMut,
};
pub use layout::{
    is_valid_permutation, DynLayout, IntoLayout, Layout, MatrixLayout, MutLayout, NdLayout,
    OverlapPolicy, ResizeLayout,
};
pub use slice_range::{to_slice_items, DynSliceItems, IntoSliceItems, SliceItem, SliceRange};

pub use tensor::{
    AsView, Matrix, MatrixMut, NdTensor, NdTensorView, NdTensorViewMut, Scalar, Tensor, TensorBase,
    TensorView, TensorViewMut, WeaklyCheckedView,
};

pub use storage::{CowData, IntoStorage, Storage, StorageMut, ViewData, ViewMutData};

/// This module provides a convenient way to import the most common traits
/// from this library via a glob import.
pub mod prelude {
    pub use super::{AsView, Layout};
}

// These modules are public for use by other crates in this repo, but
// currently considered internal to the project.
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod test_util;
