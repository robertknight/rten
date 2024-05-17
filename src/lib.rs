//! rten is a runtime for machine learning models.
//!
//! # Preparing models
//!
//! To use a model trained with a framework such as
//! [PyTorch](https://pytorch.org), it needs to first be exported into
//! [ONNX](https://onnx.ai) format and then converted into `.rten` format using
//! the [`rten-convert`](https://pypi.org/project/rten-convert/) tool.
//!
//! # Loading and running models
//!
//! The basic workflow for loading and running a model is:
//!
//! 1. Load the model using [`Model::load_file`].
//! 2. Load the input data (images, audio, text etc.)
//! 3. Pre-process the input data to convert it into tensors in the format the
//!    model expects. For this you can use RTen's own tensor types (see
//!    [rten-tensor](rten_tensor)) or
//!    [ndarray](https://docs.rs/ndarray/latest/ndarray/#).
//!
//!    If using ndarray, you will need to convert to RTen tensor types before
//!    running the model and convert the output back to ndarray types
//!    afterwards. See
//!    [rten-ndarray-demo](https://github.com/robertknight/rten-ndarray-demo)
//!    for an example.
//!
//! 4. Execute the model using [`Model::run`]
//! 5. Post-process the results to convert them into meaningful outputs.
//!
//! See the example projects in [rten-examples][rten_examples] to see how all
//! these pieces fit together.
//!
//! ## Threading
//!
//! RTen automatically executes models using multiple threads. For this purpose
//! it creates its own Rayon
//! [ThreadPool](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html)
//! which is sized to match the number of physical cores. You can access this
//! pool using [threading::thread_pool] if you want to run your own tasks in
//! this pool.
//!
//! # Supported models and hardware
//!
//! ## Hardware
//!
//! RTen currently executes models on the CPU. It can build for most
//! architectures that the Rust compiler supports. SIMD acceleration is
//! available for x86-64, Arm Neon and WebAssembly. For x86-64, AVX-512 support
//! is available but requires Nightly Rust and enabling the `avx512` crate
//! feature.
//!
//! ## Data types
//!
//! RTen supports `f32` and `i32` data types. Models with `i64` and `bool`
//! tensors are supported, but these are converted to `i32` by the conversion
//! tool. Supported for lower-precision types (16-bit floats, 8-bit integers
//! etc.) is planned for the future.
//!
//! ## Operators
//!
//! RTen currently implements a subset of [ONNX operators][onnx_operators]. See
//! the [`schema.fbs` FlatBuffers schema][schema_fbs] for currently supported
//! operators and attributes.
//!
//! Some operators require additional dependencies and are only available if
//! certain crate features are enabled:
//!
//! - The `random` feature enables operators that generate random numbers (eg.
//!   `RandomUniform`).
//!
//! # Inspecting models
//!
//! The [rten-cli](https://crates.io/crates/rten-cli) tool can be used to query
//! basic information about a `.rten` model.
//!
//! # Performance
//!
//! See the [performance
//! guide](https://github.com/robertknight/rten/blob/main/docs/performance.md) for
//! information on profiling and improving model execution performance.
//!
//! [rten_examples]: https://github.com/robertknight/rten/tree/main/rten-examples
//! [onnx_operators]: https://onnx.ai/onnx/operators/
//! [schema_fbs]: https://github.com/robertknight/rten/blob/main/src/schema.fbs
#![cfg_attr(
    feature = "avx512",
    feature(stdarch_x86_avx512),
    feature(avx512_target_feature)
)]

#[allow(unused)] // Docs only
use rten_tensor::{NdTensor, Tensor};

mod constant_storage;
mod env;
mod gemm;
mod graph;
mod iter_util;
mod model;
mod model_metadata;
mod number;
mod slice_reductions;
mod tensor_pool;
mod threading;
mod timer;
mod timing;

#[cfg(feature = "wasm_api")]
mod wasm_api;

// Temporarily included in this crate. These functions should be moved into
// a separate crate in future.
pub mod ctc;

pub mod ops;

pub use graph::{Dimension, NodeId, RunOptions};
pub use model::{Model, ModelLoadError, ModelOptions, NodeInfo, OpRegistry, ReadOp, ReadOpError};
pub use model_metadata::ModelMetadata;
pub use ops::{FloatOperators, Input, Operators, Output};
pub use tensor_pool::{ExtractBuffer, PoolRef, TensorPool};
pub use threading::thread_pool;
pub use timer::Timer;
pub use timing::TimingSort;

#[allow(dead_code, unused_imports)]
mod schema_generated;

// This is currently exposed for use in ocrs tests. That crate should probably
// create an abstraction around model execution instead.
#[doc(hidden)]
pub mod model_builder;
