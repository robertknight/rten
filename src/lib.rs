//! rten is a runtime for machine learning models.
//!
//! # Preparing models
//!
//! To use a model trained with a framework such as
//! [PyTorch](https://pytorch.org), it needs to first be exported into
//! [ONNX](https://onnx.ai) format and then converted into `.rten` format using
//! the [`rten-convert`](https://pypi.org/project/rten-convert/) tool. See the
//! [rten model format][file_format] docs for more details on the file format.
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
//! available for x86-64, Arm 64 and WebAssembly. For x86-64, AVX-512 support
//! is available but requires Nightly Rust and enabling the `avx512` crate
//! feature.
//!
//! ## Data types
//!
//! RTen supports tensors with the following data types:
//!
//! - `f32`, `i32`, `i8`, `u8`
//! - `i64` and `bool` tensors are supported by converting them to `i32` as
//! part of the model conversion process. When preparing model inputs that
//! expect these data types in ONNX, you will need to convert them to `i32`.
//!
//! Some operators support a more limited set of data types than described in
//! the ONNX specification. Please file an issue if you need an operator to
//! support additional data types.
//!
//! Support for additional types (eg. `f16`, `bf16`) is planned for the
//! future.
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
//! ## Quantized models
//!
//! RTen supports quantized models where activations are in uint8 format and
//! weights are in int8 format. This combination is the default when an ONNX
//! model is quantized using [dynamic
//! quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#dynamic-quantization).
//! The `tools/ort-quantize.py` script in the RTen repository can be used to
//! quantize an existing model with float tensors into this format.
//!
//! # Inspecting models
//!
//! The [rten-cli](https://crates.io/crates/rten-cli) tool can be used to query
//! basic information about a `.rten` model, such as the inputs and outputs.
//! It can also be used to test model compatibility and inference performance
//! by running models with randomly generated inputs.
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
//! [file_format]: https://github.com/robertknight/rten/blob/main/docs/rten-file-format.md
#![cfg_attr(
    feature = "avx512",
    feature(stdarch_x86_avx512),
    feature(avx512_target_feature)
)]

#[allow(unused)] // Docs only
use rten_tensor::{NdTensor, Tensor};

mod constant_storage;
mod downcast;
mod env;
mod gemm;
mod graph;
mod header;
mod iter_util;
mod model;
mod model_metadata;
mod number;
mod op_registry;
mod optimize;
mod slice_cast;
mod slice_reductions;
mod tensor_pool;
mod threading;
mod timing;
mod weight_cache;

#[cfg(feature = "wasm_api")]
mod wasm_api;

// Temporarily included in this crate. These functions should be moved into
// a separate crate in future.
pub mod ctc;

pub mod ops;

pub use graph::{Dimension, NodeId, RunError, RunOptions};
pub use model::{Model, ModelLoadError, ModelOptions, NodeInfo};
pub use model_metadata::ModelMetadata;
pub use op_registry::{OpRegistry, ReadOp, ReadOpError};
pub use ops::{DataType, FloatOperators, Input, InputOrOutput, Operators, Output};
pub use tensor_pool::{ExtractBuffer, PoolRef, TensorPool};
pub use threading::{thread_pool, ThreadPool};
pub use timing::TimingSort;

#[allow(dead_code, unused_imports)]
mod schema_generated;

// This is currently exposed for use in ocrs tests. That crate should probably
// create an abstraction around model execution instead.
#[doc(hidden)]
pub mod model_builder;
