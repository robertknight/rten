# rten-cli

rten-cli is a CLI tool for inspecting ONNX models and running them with
randomly generated inputs.

The CLI is useful for testing the compatibility and performance of an ONNX model
with RTen.

## Installation

Install the tool using Cargo:

```sh
cargo install rten-cli
```

This installs a binary called `rten`.

## Running a model

Run `rten <model_path>` to load an ONNX model in RTen, print details about its
inputs and outputs and then run the model with randomly generated inputs.

```
$ rten mnist.onnx
Model summary: 1 inputs, 1 outputs, 26.5 K params

Inputs
  input: tensor(f32) [1, 1, 28, 28]

Outputs
  logits: tensor(f32) [1, 10]

Metadata:
  producer_name: pytorch
  producer_version: 2.7.1

Running model with random inputs...
  Input "input" shape [1, 1, 28, 28]

  #1 - Model returned 1 outputs in 1.03ms.

  Output "logits" shape [1, 10]
```

### Dynamic input sizes

Many models have inputs with dynamically sized dimensions, such as batch size
or sequence length. These default to a size of 1. Use `-s` / `--size` to set
the size of a dimension, either for all inputs (`dim_name=size`) or for a
specific input (`input_name.dim_name=size`):

```sh
rten model.onnx -s batch_size=1 -s sequence_length=128
```

### Input values

By default input values are randomly generated. Use `-r` / `--range` to
control the range of generated values for an input. For example, to generate
token IDs in the range `[0, 1000)`:

```sh
rten model.onnx -r input_ids=0:1000
```

To use specific input values instead, save them in a
[Safetensors](https://huggingface.co/docs/safetensors) file, with tensor names
matching the model's input names, and pass the file using `-i` / `--inputs`.
Use `--check-outputs` with a second Safetensors file containing expected
outputs to compare the model's outputs against them, for example against
outputs produced by another runtime:

```sh
rten model.onnx -i inputs.safetensors --check-outputs expected.safetensors
```

Use `--print` to print the values of output tensors.

## Benchmarking and profiling

Use `-n` to run the model multiple times. When run more than once, rten-cli
prints statistics about the duration of each run. Use `-p` / `--profile` to
show a breakdown of execution time by operator, or `-p -p` for a more detailed
breakdown by operator and input shape. The number of threads used can be set
with `-t` / `--num-threads`.

```sh
rten model.onnx -n 10 -p
```

## Further help

Run `rten --help` to see all available options.
