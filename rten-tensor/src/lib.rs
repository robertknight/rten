//! rten_tensor provides multi-dimensional arrays, commonly referred to as
//! _tensors_ in a machine learning context.
//!
//! # Storage and layout
//!
//! A tensor is a combination of data storage and a layout. The data storage
//! determines the element type and how the data is owned. A tensor can be:
//!
//! - Owned (like `Vec<T>`)
//! - Borrowed (like `&[T]` or `&mut [T]`)
//! - Maybe-owned (like `Cow[T]`)
//! - Shared / reference-counted (like `Arc<[T]>`)
//!
//! The layout determines the number of dimensions (the _rank_) and size of each
//! (the _shape_) and how indices map to offsets in the storage. The dimension
//! count can be static (fixed at compile time) or dynamic (variable at
//! runtime).
//!
//! # Tensor types and traits
//!
//! The base type for all tensors is [TensorBase]. This is not normally used
//! directly but instead via a type alias which specifies the data ownership
//! and layout:
//!
//! | Rank    | Owned | Borrowed | Mutably borrowed | Owned or borrowed | Reference counted |
//! | ----    | ----- | -------- | ---------------- | ----------------- | ----------------- |
//! | Static  | [NdTensor] | [NdTensorView] | [NdTensorViewMut] | [CowNdTensor] | [ArcNdTensor] |
//! | Dynamic | [Tensor]   | [TensorView]   | [TensorViewMut]   | [CowTensor]   | [ArcTensor]   |
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

mod assume_init;
mod contiguous;
mod copy;
pub mod errors;
mod index_iterator;
pub mod iterators;
pub mod layout;
mod overlap;
pub mod slice_range;
pub mod storage;
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

// Re-exports for convenience.
pub use assume_init::AssumeInit;
pub use contiguous::Contiguous;
pub use index_iterator::{DynIndices, Indices, NdIndices};
pub use layout::{DynLayout, InsertDim, Layout, MatrixLayout, NdLayout};
pub use slice_range::{SliceItem, SliceRange};
pub use storage::Storage;
pub use tensor::{
    ArcNdTensor, ArcTensor, AsView, CowNdTensor, CowTensor, Matrix, MatrixMut, NdTensor,
    NdTensorView, NdTensorViewMut, Scalar, Tensor, TensorBase, TensorView, TensorViewMut,
    WeaklyCheckedView,
};

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
