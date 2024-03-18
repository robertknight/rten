//! rten is a runtime for machine learning models.
//!
//! RTen uses models that are exported from other frameworks such as
//! PyTorch into [ONNX](https://onnx.ai) format and then converted into the
//! inference-optimized `.rten` format by the tools in this repository.
//!
//! The basic workflow for loading and running a model is:
//!
//! 1. Load the model using [Model::load].
//! 2. Load the input data (images, audio, text etc.)
//! 3. Pre-process the input data to convert it into tensors in the format the
//!    model expects. For this you can use RTen's own tensor types (see
//!    [rten-tensor](rten_tensor)) and pre-processing libraries, or popular Rust
//!    crates such as [ndarray](https://docs.rs/ndarray/latest/ndarray/#).
//! 4. Execute the model using [Model::run] (or one of the other `run_` methods)
//! 5. Post-process the results to convert them into meaningful outputs.
//!
//! See the example projects in [rten-examples][rten_examples] to see how all
//! these pieces fit together.
//!
//! [rten_examples]: https://github.com/robertknight/rten/tree/main/rten-examples
#![cfg_attr(
    feature = "avx512",
    feature(stdarch_x86_avx512),
    feature(avx512_target_feature)
)]

#[allow(unused)] // Docs only
use rten_tensor::{NdTensor, Tensor};

mod gemm;
mod graph;
mod iter_util;
mod model;
mod model_metadata;
mod number;
mod slice_reductions;
mod timer;
mod timing;

#[cfg(feature = "wasm_api")]
mod wasm_api;

// Temporarily included in this crate. These functions should be moved into
// a separate crate in future.
pub mod ctc;

pub mod ops;

pub use graph::{Dimension, NodeId, RunOptions};
pub use model::{DefaultOperatorFactory, Model, ModelLoadError, NodeInfo, OpRegistry};
pub use model_metadata::ModelMetadata;
pub use ops::{FloatOperators, Input, Operators, Output};
pub use timer::Timer;
pub use timing::TimingSort;

#[allow(dead_code, unused_imports)]
mod schema_generated;

// This is currently exposed for use in ocrs tests. That crate should probably
// create an abstraction around model execution instead.
#[doc(hidden)]
pub mod model_builder;
