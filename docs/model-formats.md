# Model formats

RTen supports loading and running models in two formats: ONNX (`.onnx`
extension) and rten (`.rten` extension).

The ONNX format is the main format that models are distributed in by model
authors and the format that PyTorch can directly export to. It is therefore the
most convenient format to use. ONNX models contain the model's architecture and
its weights. The weights can be stored within the `.onnx` file or separately.

The rten format contains the same information as ONNX models but in a different
format that is optimized for efficient loading. rten models are produced from
ONNX models using [rten-convert](https://pypi.org/project/rten-convert/).

As a general recommendation, start with the ONNX format and then consider using
the rten format if you need its benefits.

## Graph and weight storage comparison

The main difference between the two formats for inference purposes is how
weights are stored. This affects loading the model. Once models are loaded they
function the same.

### Memory-mapping support

rten models store the graph followed by the weights in a single file, and the
weights are stored at appropriate offsets for them to be loaded by
memory-mapping the entire file using `mmap`, or by reading the entire file into
one buffer. ONNX models store the graph in the `.onnx` file and the weights can
either be embedded in the same file or stored in separate external data files.
Embedded weights inside `.onnx` files cannot be memory-mapped because they are
not appropriately aligned. Instead each tensor must be copied into an
appropriately aligned buffer when the model is loaded. Weights in external data
files however **can** be memory-mapped.

### File size limits and renaming

rten files can be as large as supported by the file system. ONNX files with
embedded weights can be up to 2GB in size. Larger ONNX models must store weights
in separate files. rten files can be easily renamed. ONNX files can be renamed,
but their external data files cannot. This is because `.onnx` files reference
external data files by their full file name.

### Graph representation

rten uses a more compact representation of node IDs and operator attributes when
describing the model graph. This can improve model load times slightly, although
the impact is usually small because model load time is dominated by the time
loading or processing weights.

## Tooling and ecosystem comparison

ONNX models can be visualized using [Netron](https://netron.app/) and parsed or
modified using the [onnx](https://pypi.org/project/onnx/) Python package. ONNX
models can be run using [ONNX Runtime](https://onnxruntime.ai/) and many
other engines.

By comparison there are far fewer tools available for the `.rten` format. The
`rten-model-file` crate can be used to parse rten models and the `rten` CLI tool
can be used to inspect and run them. RTen is the only inference engine that
supports this format. Due to the better availability of tooling for ONNX, if you
generate `.rten`-format models, always retain the ONNX version.

## Format documentation

### ONNX

ONNX files are serialized Protocol Buffers messages, which may reference tensor
data stored in external data files. See the [onnx.ai
site](https://onnx.ai/onnx/intro/python.html#serialization) for an overview and
the `onnx.proto` files in the [onnx
repository](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3).

### RTen

The rten file format is described by the [RTen format](rten-file-format.md)
document. It consists of a header, model graph in FlatBuffers format followed
by weights.
