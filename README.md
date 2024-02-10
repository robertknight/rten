# RTen

RTen (the _Rusty Tensor engine_) † is a runtime for machine learning models
converted from [ONNX](https://onnx.ai) format, which you can export from
PyTorch and other frameworks.

The project also provides supporting libraries for common pre-processing and
post-processing tasks in various domains. This makes RTen a more complete
toolkit for running models in Rust applications.

† _The name is also a reference to PyTorch's ATen library._

## Goals

- Provide a (relatively) small and efficient neural network runtime that makes
  it easy to take models created in frameworks such as PyTorch and run them in
  Rust applications.
- Be easy to compile and run on a variety of platforms, including WebAssembly
- End-to-end Rust. This project and all of its required dependencies are
  written in Rust.

## Limitations

This project has a number of limitations to be aware of. Addressing them is
planned for the future:

- Supports CPU inference only. There is currently no support for running models
  on GPUs or other accelerators.
- Not all ONNX operators are currently supported. See `OperatorType` in
  [src/schema.fbs](src/schema.fbs) for currently supported operators. For
  implemented operators, some attributes or input shapes may not be supported.
- A limited set of data types are supported: float32 and int32 tensors. int64
  and boolean tensors are converted to int32.
- RTen is not as well optimized as more mature runtimes such as ONNX Runtime
  or TensorFlow Lite. The performance difference depends on the operators used,
  model structure, CPU architecture and platform.

## Getting started

The best way to get started is to clone this repository and try running some of
the examples locally. The conversion scripts use popular Python machine learning
libraries, so you will need Python >= 3.10 installed.

The examples are located in the [rten-examples/](rten-examples/) directory.
See the [README](rten-examples/) for descriptions of all the examples and steps
to run them. As a quick-start, here are the steps to run the image
classification example:

```sh
git clone https://github.com/robertknight/rten.git
cd rten

# Install dependencies for model conversion
pip install -e rten-convert

# Install dependencies for Python scripts
pip install -r tools/requirements.txt

# Export an ONNX model. We're using resnet-50, a classic image classification model.
python -m tools.export-timm-model timm/resnet50.a1_in1k

# Convert model to this library's format
rten-convert resnet50.a1_in1k.onnx resnet50.rten

# Run image classification example. Replace `image.png` with your own image.
cargo run -p rten-examples --release --bin imagenet mobilenet resnet50.rten image.png
```

## Converting ONNX models

RTen does not load ONNX models directly. ONNX models must be run through a
conversion tool which produces an optimized model in a
[FlatBuffers](https://google.github.io/flatbuffers/)-based format (`.rten`) that
the engine can load. This is conceptually similar to the `.tflite` and `.ort`
formats that TensorFlow Lite and ONNX Runtime use.

The conversion tool requires Python >= 3.10. To convert an existing ONNX model,
run:

```sh
git clone https://github.com/robertknight/rten.git
pip install -e rten/rten-convert
rten-convert your-model.onnx output.rten
```

The RTen model format does not yet guarantee long-term backwards compatibility,
so be aware that you may need to recompile models from ONNX for new releases.

## Usage in JavaScript

To use this library in a JavaScript application, there are two approaches:

1. Prepare model inputs in JavaScript and use the rten library's built-in
   WebAssembly API to run the model and return a tensor which will then need
   to be post-processed in JS. This approach may be easiest for tasks where
   the pre-processing is simple.

   The [image classification](js-examples/image-classification/) example uses
   this approach.

2. Create a Rust library that uses rten and does pre-processing of inputs and
   post-processing of outputs on the Rust side, exposing a domain-specific
   WebAssembly API. This approach is more suitable if you have complex and/or
   computationally intensive pre/post-processing to do.

Before running the examples, you will need to follow the steps under ["Building
the WebAssembly library"](#building-the-webassembly-library) below.

The general steps for using RTen's built-in WebAssembly API to run models in
a JavaScript project are:

1.  Develop a model or find a pre-trained one that you want to run. Pre-trained
    models in ONNX format can be obtained from the [ONNX Model Zoo](https://github.com/onnx/models)
    or [Hugging Face](https://huggingface.co/docs/transformers/serialization).
2.  If the model is not already in ONNX format, convert it to ONNX. PyTorch
    users can use [torch.onnx](https://pytorch.org/docs/stable/onnx.html) for this.
3.  Use the `rten-convert` package in this repository to convert the model
    to the optimized format RTen uses. See the section above on converting models.
4.  In your JavaScript code, fetch the WebAssembly binary and initialize RTen
    using the `init` function.
5.  Fetch the prepared `.rten` model and use it to an instantiate the `Model`
    class from this library.
6.  Each time you want to run the model, prepare one or more `Float32Array`s
    containing input data in the format expected by the model, and call
    `Model.run`. This will return a `TensorList` that provides access to the
    shapes and data of the outputs.

After building the library, API documentation for the `Model` and `TensorList`
classes is available in `dist/rten.d.ts`.

## Building the WebAssembly library

### Prerequisites

To build RTen for WebAssembly you will need:

- A recent stable version of Rust
- `make`
- (Optional) The `wasm-opt` tool from [Binaryen](https://github.com/WebAssembly/binaryen)
  can be used to optimize `.wasm` binaries for improved performance
- (Optional) A recent version of Node for running demos

### Building rten

```sh
git clone https://github.com/robertknight/rten.git
cd rten
make wasm-all
```

The `make wasm-all` command will build two versions of the library, one for
browsers that support SIMD (Chrome 91, Firefox 89, Safari 16.4) and one for
those which do not (primarily older Safari releases). See the [WebAssembly
Roadmap](https://webassembly.org/roadmap/) for a full list of which features
different engines support. **The SIMD build is significantly faster**.

During development, you can speed up the testing cycle by running `make wasm`
to build only the SIMD version, or `make wasm-nosimd` for the non-SIMD version.

At runtime, you can find out which build is supported by calling the `binaryName()`
function exported by this package.
