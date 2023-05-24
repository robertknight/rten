mod graph;
mod linalg;
mod model;
mod number;
mod tensor;
mod timer;
mod wasm_api;

// Temporarily included in this crate. These functions should be moved into
// a separate crate in future.
pub mod ctc;
pub mod geometry;
pub mod page_layout;

pub mod ops;

pub use graph::{Dimension, RunOptions};
pub use model::Model;
pub use ops::{Input, Output};
pub use tensor::{
    BroadcastElements, Elements, NdTensor, NdTensorBase, NdTensorLayout, NdTensorView,
    NdTensorViewMut, SliceItem, SliceRange, Tensor, TensorBase, TensorLayout, TensorView,
    TensorViewMut,
};

#[allow(clippy::extra_unused_lifetimes, dead_code, unused_imports)]
mod schema_generated;

#[cfg(test)]
mod rng;

#[cfg(test)]
mod model_builder;

#[cfg(test)]
mod test_util;
