# RTen
[![Latest Version]][crates.io] [![Documentation]][docs.rs]

[Latest Version]: https://img.shields.io/crates/v/rten.svg
[Documentation]: https://img.shields.io/docsrs/rten
[docs.rs]: https://docs.rs/rten
[crates.io]: https://crates.io/crates/rten

RTen (the _Rust Tensor engine_) † is a machine learning runtime. It supports
models in [ONNX](https://onnx.ai) format. RTen enables you to take machine
learning models which have been trained in Python using frameworks such as
PyTorch and run them in Rust.

In addition to ML inference, the project also provides supporting libraries for
common pre-processing and post-processing tasks in various domains. This makes
RTen a more complete toolkit for running models in Rust applications.

† _The name is also a reference to PyTorch's ATen library._

## Goals

- Provide a (relatively) small and efficient neural network runtime that makes
  it easy to take models created in frameworks such as PyTorch and run them in
  Rust applications.
- Be easy to compile and run on a variety of platforms, including WebAssembly
- End-to-end Rust. This project and all of its required dependencies are
  written in Rust. This simplifies the build and deployment process.

## Supported devices

RTen currently supports CPU inference only. It supports SIMD via AVX2, AVX-512,
Arm Neon and WebAssembly SIMD. Inference uses multiple threads by default,
defaulting to the number of physical cores (or performance cores). This can be
customized.

## Supported models

### Operators

RTen supports most standard ONNX operators. See [this tracking
issue](https://github.com/robertknight/rten/issues/14) for details. Please open
an issue if you find that you cannot run a model because an operator is not
supported.

### Data types

RTen supports models with float32 weights as well as quantized models with int8
or uint8 weights. Quantized models can take advantage of CPU features such
as VNNI (x86) and UDOT / i8mm (Arm) for better performance.

### Model formats

RTen can load models in ONNX format directly. It also supports a custom `.rten`
format which can offer faster load times and supports arbitrarily large models
in a single file. See the [rten file format
documentation](docs/rten-file-format.md) for more details on the format and
information on how to convert models.

## Getting started

The best way to get started is to clone this repository and try running some of
the examples locally. Many of the examples use Hugging Face's
[Optimum](https://github.com/huggingface/optimum) or other Python-based tools to
export the ONNX model, so you will need a recent Python version installed.

The examples are located in the [rten-examples/](rten-examples/) directory.
See the [README](rten-examples/) for descriptions of all the examples and steps
to run them. As a quick-start, here are the steps to run the image
classification example:

```sh
git clone https://github.com/robertknight/rten.git
cd rten

# Install dependencies for Python scripts
pip install -r tools/requirements.txt

# Export an ONNX model. We're using resnet-50, a classic image classification model.
python -m tools.export-timm-model timm/resnet50.a1_in1k

# Run image classification example. Replace `image.png` with your own image.
cargo run -p rten-examples --release --bin imagenet resnet50.a1_in1k.onnx image.png
```

**Model format note:** Support for running `.onnx` models directly is new in
RTen v0.23. To run models with earlier versions you need to convert them to the
`.rten` format first using [rten-convert](https://pypi.org/project/rten-convert/).

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

1.  In your JavaScript code, fetch the WebAssembly binary and initialize RTen
    using the `init` function.
2.  Fetch the prepared `.onnx` model and use it to an instantiate the `Model`
    class from this library.
3.  Each time you want to run the model, prepare one or more `Float32Array`s
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
make wasm
```

The build created by `make wasm` requires support for WebAssembly SIMD,
available since Chrome 91, Firefox 89 and Safari 16.4. It is possible to
build the library without WebAssembly SIMD support using `make wasm-nosimd`,
or both using `make wasm-all`. The non-SIMD builds are significantly slower.

At runtime, you can find out which build is supported by calling the
`binaryName()` function exported by this package.
