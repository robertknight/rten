# rten-convert

rten-convert converts ONNX models to `.rten` format, for use with the
[RTen](https://github.com/robertknight/rten) machine learning runtime. It can
also convert `.rten` models back to ONNX.

## Installation

The conversion tool requires Python >= 3.10. To install the tool, run:

```sh
pip install rten-convert
```

## Usage

```sh
rten-convert your-model.onnx your-model.rten
```

The second argument is optional. If omitted the output filename will be the
input filename with the `.onnx` extension replaced with `.rten`.

### Converting rten models to ONNX

The direction of the conversion is determined by the input file, so to convert
a model back to ONNX, run:

```sh
rten-convert your-model.rten your-model.onnx
```

The `.rten` format is [deprecated][rten-format-deprecation]. This conversion
exists to migrate models for which the original ONNX file is no longer
available. **If you still have the original ONNX model, use that instead.**
`.rten` models do not contain all of the information that ONNX models do, so
the generated model will differ from the original in several ways:

- Tensors are converted to the types that rten uses. `int64` and `bool` tensors
  are stored as `int32` in `.rten` models. They are converted back to `int64`,
  and `Cast` operations are inserted where ONNX requires a `bool` input. As a
  result, an `int32` input or output may become `int64`.
- `float16` and `float64` tensors were converted to `float32` when the `.rten`
  model was generated. This cannot be reversed.
- Operators are generated using a recent opset version, with the semantics that
  rten implements. Where an operator's behavior changed between opset versions,
  such as the effect of `ceil_mode` on `AveragePool`, the generated model
  follows the newer behavior.
- `SAME_LOWER` auto padding is replaced by `SAME_UPPER`, and operators that
  ONNX has since deprecated (`Upsample`, `Scatter`) are replaced by their
  modern equivalents.
- Model inputs and outputs for which the `.rten` model records no data type are
  assumed to be `float32`.

[rten-format-deprecation]: https://github.com/robertknight/rten/issues/1470

## Versioning

The `rten-convert` tool and `rten` library use common version numbering. A
model produced by `rten-convert` version X can be executed by `rten` version X
or newer.

## Development

To install this tool from a checkout of the Git repository, run:

```sh
pip install -e .
```

After making changes, run the QA checks. First, install the development
dependencies:

```
pip install -r requirements.dev.txt
```

Then run:

```
make check
```
