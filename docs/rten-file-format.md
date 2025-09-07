# RTen model format

RTen model files (`.rten`) contain the computation graph for a machine learning
model, model metadata and weights. The format is designed to be efficient to
load and to minimize additional memory required, beyond the size of the file
itself.

RTen files are produced by exporting models from a machine learning framework
such as PyTorch or Keras into [ONNX](https://onnx.ai) format, and converting the
ONNX model to `.rten` using the
[rten-convert](https://pypi.org/project/rten-convert/) tool.

## Compatibility

The `rten-convert` tool and `rten` Rust crate have version numbers that are
aligned. A `.rten` model produced by version X of `rten-convert` can be read by
version X of the `rten` crate or newer. Models produced by version X of
`rten-convert` _may_ work with earlier versions of `rten` as long as the model
does not rely on operators or attributes that were added in version X.

## History

There are two versions of the RTen model format. The second version added
support for models larger than 2GB. RTen can load models in either format. The
`rten-convert` tool generates the V2 format by default, and will generate the V1
format if the `--v1` flag is passed.

## V2 format

### Overall structure

The overall structure of a `.rten` file is:

```
[header] … [model_data] … [tensor_data]
```

### Header

The header identifies the file type, the major version of the format and
contains the offsets of the other sections. The structure of the header is:

```
[magic:u8x4] [version:u32] [model_data_offset:u64] [model_data_len:u64] [tensor_data_offset:u64]
```

All numbers are encoded in little-endian order.

- `magic` - The ASCII bytes `RTEN`
- `version` - Currently 2
- `model_data_offset` - Offset of the data describing the model
- `model_data_len` - Length of the data describing the model
- `tensor_data_offset` - Offset of the start of tensor data. Tensor references in
  the model data are relative to this.

### Model data

The model data is a [FlatBuffers](https://flatbuffers.dev) buffer which
describes the computation graph for the model. It also contains metadata about
the model.

The computation graph consists of three kinds of nodes: constants (weights,
biases etc.), values (inputs or outputs from computation steps) and operators
(computation steps such as matrix multiplication). The operators correspond
closely to operators in the [ONNX
specification](https://onnx.ai/onnx/operators/). Constant nodes describe the
data type and shape of tensors. The data for a tensor can either be stored
inline in the model or externally in the tensor data section.

The FlatBuffers schema can be found in `rten-model-file/src/schema.fbs`.

### Tensor data

The tensor data section is a block of bytes referenced by the model data. The
shape of tensors, type of elements and other metadata is contained in the model
data.

## V1 format

The first version of the `.rten` model format consisted of just the model
data without the header or tensor data sections. The FlatBuffers schema used by
V1 is the same as V2.

This was changed due to FlatBuffers having a 2GB file
size limit, and also to enable more control over the alignment of tensor data.
