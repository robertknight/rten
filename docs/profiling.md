# Profiling and optimizing

This document provides strategies for profiling and optimizing graph execution
performance in RTen.

## Build settings

Performance sensitive numeric code like RTen will run extremely slowly in debug
builds, to the point of being unusable. For profiling, make sure you build your
application in release mode.

## Profiling using built-in logging

To find out where model inference is spending time, you can enable the built-in
timing logs, either by setting flags in the `RunOptions` passed to `Model::run`
or by setting the `RTEN_TIMING` environment variable.

Example output for the MobileViT model with timing enabled:

```
Graph run of 878 ops finished in 172.944ms

Sigmoid          64.15ms (37.09%)
Conv             50.39ms (29.14%)
MatMul           20.09ms (11.62%)
Softmax          12.92ms (7.47%)
[Mem alloc/free] 7.38ms  (4.27%)
Mul              5.80ms  (3.36%)
Transpose        3.11ms  (1.80%)
Add              1.95ms  (1.13%)
[Other]          1.55ms  (0.90%)
Sub              1.44ms  (0.83%)
Div              0.97ms  (0.56%)
ReduceMean       0.97ms  (0.56%)
Pow              0.68ms  (0.39%)
Gemm             0.64ms  (0.37%)
Concat           0.29ms  (0.17%)
Gather           0.25ms  (0.14%)
Unsqueeze        0.17ms  (0.10%)
Shape            0.11ms  (0.06%)
Reshape          0.08ms  (0.05%)
Sqrt             0.01ms  (0.01%)
```

In this case most time is spent in convolution and matrix multiplication ops
as expected, but `Sigmoid` and `Softmax` are much more expensive than expected
and need optimization. The entries in square brackets show time spent outside
operators, eg. allocating and de-allocating memory.

### `RTEN_TIMING` syntax

If the `RTEN_TIMING` environment variable is defined, a summary of operator
timings will be output at the each of each graph run. You can control the
level of granularity in the output and the sort order by adding configuration
values to the variable's value. Values are specified as a space-separated list
of `key=value` pairs. For boolean values, "1", "true", "yes" are interpreted
as `true` and "0", "false" or "no" as `false`.

Valid keys are:

- `sort` controls whether the summary is sorted by `time` (the default) or
  operator `name`.
- `by-shape` is a boolean that determines whether the summary includes a
  breakdown of timings by each distinct input shape, for each operator.

For example, to sort timings by operator name and show a breakdown by shape:

```sh
export RTEN_TIMING="sort=name by-shape=1"
```

## Profiling using sampling profilers

To dive deeper into execution time, you will need to use a profiler. A
convenient open source tool for Linux and macOS is
[samply](https://github.com/mstange/samply). To use it, first make sure the
binary you want to profile is built:

```sh
cargo build -r --example imagenet
```

Then run it using `samply record`:

```sh
samply record cargo run -r --example imagenet mobilevit mobilevit.rten image.jpg
```

Here `cargo run` would automatically rebuild the binary if needed, but we build
the binary separately first because we don't want the profiling output to
include time spent compiling Rust code.

After samply runs it will produce a profile and serve it in a web application
that you can view using Firefox or Chrome.

## PyTorch and ONNX Runtime baselines

It is often helpful to write a Python script that runs inference on the same
model, in order to get a baseline for what "good" performance on the same
system can look like.

See PyTorch's [profiler
tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
for information on how to get a breakdown of operator timings in PyTorch. With
PyTorch you may want to compare eager mode execution (ie. the default execution
mode) vs execution with a model optimized via `torch.compile`.

When comparing with ONNX Runtime, it is worth comparing with [graph
optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
turned on and off, as well as with the number of threads varied. This can show
whether the difference in runtime performance is due to individual operator
performance, graph-level optimizations (ie. combining or "fusing" multiple
operations together), how effectively different runtimes are able to exploit
parallelism or other factors.

## Optimizing inference

RTen does not currently have many turn-key solutions for optimizing inference,
like `torch.compile` for PyTorch or ONNX Runtime's [graph
optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).
These are planned for the future.

Some ways to speed up inference without changing RTen's code are:

- If choosing from a family of models with different sizes, you can trade
  accuracy for performance by using a smaller model.
- If you can break your problem up into chunks, use
  [Rayon](https://github.com/rayon-rs/rayon) to execute the model on separate
  chunks in parallel.
- Computer vision models may accept inputs of different sizes. Reducing the
  input size can speed up inference significantly, although accuracy can degrade
  significantly if the input is much smaller than what the model was trained
  on.
- For autoregressive or recurrent models which are run repeatedly inside a loop,
  you can use `Model::partial_run` outside the loop to evaluate parts of the model
  which depend only on loop-invariant inputs. Then inside the loop you pass the
  results of `partial_run` together with the loop-varying inputs to `Model::run`
  to compute the remainder of the graph.

If you find that an operator is unexpectedly slow compared to other runtimes,
and the issue can be reproduced using an open source model and code, please
[file an issue](https://github.com/robertknight/rten/issues).

## Platform-specific optimizations

### AVX-512

On modern x64 CPUs which support AVX-512, you can get better performance by
enabling the `avx512` feature. As of Rust v1.75, this requires nightly Rust.
