# rten-convert

rten-convert converts ONNX models to `.rten` format, for use with the
[RTen](https://github.com/robertknight/rten) machine learning runtime.

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
