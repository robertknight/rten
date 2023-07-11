mod index_iterator;
mod iterators;
mod layout;
mod macros;
mod ndtensor;
mod overlap;
mod range;
mod tensor;
mod vec_with_offset;

pub use index_iterator::{DynIndices, Indices, NdIndices};
pub use iterators::{AxisIter, AxisIterMut, BroadcastIter, Iter, IterMut, Offsets};
pub use layout::{is_valid_permutation, Layout};
pub use ndtensor::{
    Matrix, MatrixLayout, MatrixMut, NdTensor, NdTensorBase, NdTensorLayout, NdTensorView,
    NdTensorViewMut,
};
pub use range::{IntoSliceItems, SliceItem, SliceRange};
pub use tensor::{Tensor, TensorBase, TensorLayout, TensorView, TensorViewMut};

// These modules are public for use by other crates in this repo, but
// currently considered internal to the project.
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod test_util;
