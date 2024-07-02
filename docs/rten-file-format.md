# RTen model format

RTen model files (`.rten`) contain the computation graph for a model, model
metadata and weights.

RTen files are produced by exporting models from a machine learning framework
such as PyTorch or Keras into ONNX format, and converting the ONNX model to
`.rten` using the rten-convert tool.

## Format structure

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

`magic` - The ASCII bytes `RTEN`
`version` - Currently 2
`model_data_offset` - Offset of the FlatBuffers data describing the model
`model_data_len` - Length of the FlatBuffers data describing the model
`tensor_data_offset` - Offset of the start of tensor data. Tensor references in
the model buffer are relative to this.

### Model data

The model data is a FlatBuffers buffer which describes the computation graph for
the model, its inputs and outputs, metadata and the format and location of data
for individual tensors within the tensor data section.

The computation graph consists of three kinds of nodes: constants (weights,
biases etc.), values (inputs or outputs from computation steps) and operators
(computation steps such as matrix multiplication). This structure and in
particular the operators correspond closely to the
[ONNX](https://onnx.ai/onnx/operators/) format.

The FlatBuffers schema can be found in `src/schema.fbs`.

### Tensor data

The tensor data section is a block of bytes whose interpretation is described
by _constant_ nodes in the graph.

## Earlier versions

The initial version of the `.rten` model format consisted of just the model
data without the header or tensor data sections.

This was changed due to FlatBuffers having a 2GB file
size limit, but also to enable more control over the alignment of tensor data.
