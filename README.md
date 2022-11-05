# Wasnn

Wasnn is a neural network inference engine for running [ONNX
models](https://onnx.ai). It has a particular focus on use in browsers and
other environments that support WebAssembly.

Wasnn is written in portable Rust and has minimal dependencies.

## Goals

 - Provide a small and reasonably efficient neural network runtime that is
   well-suited to the needs of running small models in browsers
 - Be easy to compile and run on a variety of platforms.

## Limitations

 - Only a subset of ONNX operators are currently supported.
 - There is no support for running models on the GPU or other neural network
   accelerators.
 - Wasnn is fast enough to be useful for many applications, but not as well
   optimized as more mature runtimes such as ONNX Runtime or TensorFlow
   Lite.

## Preparing ONNX models

Wasnn does not load ONNX models directly. ONNX models must be run through a
conversion tool which produces an optimized model in a
[FlatBuffers](https://google.github.io/flatbuffers/)-based format that the
engine can load.

The conversion tool requires Python >= 3.10. To convert an existing ONNX model,
run:

```sh
git clone https://github.com/robertknight/wasnn.git
pip install -r wasnn/tools/requirements.txt
wasnn/tools/convert-onnx.py your-model.onnx output.model
```

The optimized Wasnn model format is not yet backwards compatible, so models
should be converted from ONNX for the specific Wasnn release that the model is
going to be used with, typically as part of your project's build process.

## Building the library

### Prerequisites

To build Wasnn you will need:

 - A recent stable version of Rust
 - `make`
 - (Optional) The `wasm-opt` tool from [Binaryen](https://github.com/WebAssembly/binaryen)
   can be used to optimize the `.wasm` binary for improved performance
 - (Optional) A recent version of Node for running demos

### Building wasnn

```sh
git clone https://github.com/robertknight/wasnn.git
cd wasnn
make wasm
```
