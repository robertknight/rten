//! rten is an inference runtime for machine learning models.
//!
//! It enables you to take machine learning models trained using PyTorch
//! or other frameworks and run them in Rust.
//!
//! # Preparing models
//!
//! To use a model trained with a framework such as
//! [PyTorch](https://pytorch.org), it needs to first be exported into
//! [ONNX](https://onnx.ai) format. There are several ways to obtain models
//! in this format:
//!
//! - The model authors may already provide the model in ONNX
//!   format. On [Hugging Face](https://huggingface.co/) you can find models
//!   available in ONNX format by searching for the [ONNX
//!   tag](https://huggingface.co/models?library=onnx&sort=trending).
//!
//! - Hugging Face provides a tool called
//!   [Optimum](https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model)
//!   which takes as input a Hugging Face model repository URL and exports an
//!   ONNX model. This is a convenient way to export many popular pre-trained
//!   models to ONNX format.
//!
//! - PyTorch has built-in [ONNX export functions](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html).
//!   This can be used to convert custom models or any other model which is not
//!   available in ONNX format via another means.
//!
//! RTen can load and run ONNX models directly, but it also supports a custom
//! [`.rten` file format][rten_format]. Models can be converted from ONNX to
//! this format via [rten-convert](https://pypi.org/project/rten-convert/). The
//! `.rten` format can be faster to load and supports large (> 2GB) models in a
//! single file, whereas ONNX models of this size must use external files for
//! weights. It is recommended to start with the ONNX format and consider
//! `.rten` later if you need these benefits.
//!
//! See the [model formats][model_formats] documentation for more details on
//! the format differences.
//!
//! # Loading and running models
//!
//! The basic workflow for loading and running a model is:
//!
//! 1. Load the model using [`Model::load_file`] or [`Model::load_mmap`].
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
//! available for x86-64, Arm 64 and WebAssembly.
//!
//! ## Data types
//!
//! RTen supports tensors with the following data types:
//!
//! - `f32`, `i32`, `i8`, `u8`
//! - `i64` and `bool` tensors are supported by converting them to `i32`
//!   tensors, on the assumption that the values in `i64` tensors will be in the
//!   `i32` range. When preparing model inputs that expect these data types in
//!   ONNX, you will need to convert them to `i32`.
//! - `f64` tensors are supported by converting them to `f32`.
//!
//! Some operators support a more limited set of data types than described in
//! the ONNX specification. Please file an issue if you need an operator to
//! support additional data types.
//!
//! Support for additional types (eg. `f16`, `bf16`) is planned for the
//! future.
//!
//! ## Supported operators
//!
//! RTen supports most ONNX operators. See the [tracking
//! issue](https://github.com/robertknight/rten/issues/14) for details.
//!
//! Some operators require additional dependencies and are only available if
//! certain crate features are enabled:
//!
//! - The `fft` feature enables operators related to the Fast Fourier Transform
//!   (eg. STFT) using [rustfft](https://docs.rs/crate/rustfft).
//! - The `random` feature enables operators that generate random numbers (eg.
//!   `RandomUniform`) using [fastrand](https://docs.rs/crate/fastrand).
//!
//! As a convenience, the `all-ops` feature enables all of the above features.
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
//! See the [quantization
//! guide](https://github.com/robertknight/rten/blob/main/docs/quantization.md)
//! for a tutorial on how to quantize models and more information about
//! quantization in ONNX and the nuances of quantization support in RTen.
//!
//! # Inspecting models
//!
//! The [rten-cli](https://crates.io/crates/rten-cli) tool can be used to query
//! basic information about a `.rten` or `.onnx` model, such as the inputs and
//! outputs. It can also be used to test model compatibility and inference
//! performance by running models with randomly generated inputs.
//!
//! To examine a `.onnx` model in more detail, the [Netron](https://netron.app/)
//! application is very useful. It shows the complete model graph and enables
//! inspecting individual nodes.
//!
//! # Performance
//!
//! See the [performance
//! guide](https://github.com/robertknight/rten/blob/main/docs/performance.md) for
//! information on profiling and improving model execution performance.
//!
//! [model_formats]: https://github.com/robertknight/rten/blob/main/docs/model-formats.md
//! [onnx_operators]: https://onnx.ai/onnx/operators/
//! [rten_examples]: https://github.com/robertknight/rten/tree/main/rten-examples
//! [rten_format]: https://github.com/robertknight/rten/blob/main/docs/rten-file-format.md
//! [schema_fbs]: https://github.com/robertknight/rten/blob/main/src/schema.fbs

#[allow(unused)] // Docs only
use rten_tensor::{NdTensor, Tensor};

mod buffer_pool;
mod constant_storage;
mod env;
mod graph;
mod model;
mod op_registry;
mod optimize;
mod shift_cast;
mod slice_reductions;
mod threading;
mod timing;
mod value;
mod weight_cache;

#[cfg(feature = "wasm_api")]
mod wasm_api;

// Temporarily included in this crate. These functions should be moved into
// a separate crate in future.
pub mod ctc;

pub mod ops;

pub use buffer_pool::{BufferPool, ExtractBuffer, PoolRef};
pub use graph::{Dimension, NodeId, RunError, RunOptions};
pub use model::{Model, ModelLoadError, ModelMetadata, ModelOptions, NodeInfo};
pub use op_registry::{OpRegistry, ReadOpError};
pub use ops::{FloatOperators, Operators};
pub use threading::{ThreadPool, thread_pool};
pub use timing::TimingSort;
pub use value::{DataType, Sequence, Value, ValueOrView, ValueView};

// Deprecated aliases for `ValueView`, `ValueOrView` and `Value`.
#[allow(deprecated)]
pub use ops::{Input, InputOrOutput, Output};

// This is currently exposed for use in ocrs tests. That crate should probably
// create an abstraction around model execution instead.
#[doc(hidden)]
#[cfg(any(test, feature = "model_builder"))]
pub use model::rten_builder as model_builder;
