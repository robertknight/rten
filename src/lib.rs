mod graph;
mod linalg;
mod model;
mod number;
mod tensor;
mod timer;
mod wasm_api;

pub mod ops;

pub use graph::{Dimension, RunOptions};
pub use model::Model;
pub use ops::{Input, Output};
pub use tensor::{
    BroadcastElements, Elements, SliceItem, SliceRange, Tensor, TensorLayout, TensorView,
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
