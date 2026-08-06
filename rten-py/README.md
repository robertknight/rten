# rten (Python)

Python bindings for [RTen](https://github.com/robertknight/rten), a machine
learning runtime for models in ONNX format.

## Usage

```python
import numpy as np
import rten

model = rten.Model("model.onnx")

for input in model.get_inputs():
    print(input.name, input.type, input.shape)

outputs = model.run(None, {"input": np.zeros((1, 3, 224, 224), np.float32)})
```

`Model.run` takes a list of output names, or `None` for all of the model's
outputs, and a dict mapping input names to numpy arrays. It returns a list of
numpy arrays.

## Porting from ONNX Runtime

The API is similar to ONNX Runtime. Basic use of `ort.InferenceSession` can
be ported with a simple class name change:

```python
session = ort.InferenceSession("model.onnx")  # ONNX Runtime
model = rten.Model("model.onnx")              # RTen
```

## Caveats

- RTen represents `int64` and `bool` tensors as `int32`, and `float64` tensors
  as `float32`. Inputs in those dtypes are converted for you, but **outputs that
  a model declares as `int64` or `bool` come back as `int32`**, and
  `get_inputs()` / `get_outputs()` report them as `tensor(i32)`. An `int64`
  input containing a value too large for `int32` raises `OverflowError`.

- `float16` inputs are not supported. Convert them with
  `array.astype(np.float32)`.

Errors that RTen reports when loading or running a model raise `rten.RtenError`.

## Development

To build and test the bindings locally, first install
[uv](https://docs.astral.sh/uv/). Then build the bindings and run tests with:

```sh
make test-py
```
