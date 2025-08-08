# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.21.1] - 2025-08-08

- Fix a bug where model converter would produce an empty model if the input is
  larger than 2GB in size and shape inference is enabled (https://github.com/robertknight/rten/pull/852)

## [0.21.0] - 2025-08-05

This release improves int8 matrix multiplication performance for all targets,
but especially low-end Arm CPUs without dotprod support and high-end Arm CPUs
with i8mm support. It also adds support for `IsInf` and `IsNaN` operators.

Two new internal crates have been added, `rten-base` and `rten-gemm`.

### All crates

- Fixed `mismatched_lifetime_syntaxes` warnings that appeared when compiling
  with nightly Rust (https://github.com/robertknight/rten/pull/814)

### rten-base

- Created a new `rten-base` crate at the root of the dependency tree that
  contains various shared utilities.

### rten-convert

- Run shape inference as part of ONNX model conversion. This enables additional
  graph optimizations (https://github.com/robertknight/rten/pull/820)

- Don't warn about models which use `-i64::MAX` to represent `-Infinity` when
  slicing tensors (https://github.com/robertknight/rten/pull/821)

### rten-examples

- Report time spent in encoder, prompt processing and decoder in Whisper
  example (https://github.com/robertknight/rten/pull/842)

### rten-gemm

- Improve int8 GEMM performance by increasing depth blocking size
  (https://github.com/robertknight/rten/pull/840)

- Added UMLAL-based kernel for older/low-end Arm CPUs. This gives a ~50%
  improvement on eg. Cortex A53 compared to the previous kernel
  (https://github.com/robertknight/rten/pull/832)

- Added i8mm kernel for newer/high-end Arm CPUs. This gives a ~1.6x
  performance improvement on eg. Graviton 4 compared to the previous kernel
  (https://github.com/robertknight/rten/pull/824, https://github.com/robertknight/rten/pull/829)

- Extracted f32 and int8 matrix multiplication kernels into a new `rten-gemm`
  crate.

### rten

- Impl `Default` and `Clone` for `ModelOptions` (https://github.com/robertknight/rten/pull/844)

- Support IsInf and IsNaN operators (https://github.com/robertknight/rten/pull/837)

- Fixed WASM relaxed-simd build (https://github.com/robertknight/rten/pull/834)

- Optimized handling of zero points in int8 matmul (https://github.com/robertknight/rten/pull/812)

- Improved int8 LHS packing efficiency (https://github.com/robertknight/rten/pull/809)

- Added graph optimization that can determine when `Slice(Shape(X))` operations
  produce a constant even if some dimensions of X are dynamic. This enables
  better fusion for scaled-dot-product operations in some models
  (https://github.com/robertknight/rten/pull/805).

- Parallelize weight prepacking (https://github.com/robertknight/rten/pull/802)

- Add graph optimization to eliminate common identity operations (eg. `Mul(X, 1)`)
  (https://github.com/robertknight/rten/pull/798, https://github.com/robertknight/rten/pull/799)

- Optimized Trilu operator (https://github.com/robertknight/rten/pull/797)

- Support fusing MatMulInteger + Cast + Mul subgraphs produced by dynamic
  quantization (https://github.com/robertknight/rten/pull/795)

### rten-cli

- Support setting `--num-iters=0`. This loads the model, runs optimization and
  prints metadata, but skips inference (https://github.com/robertknight/rten/pull/801)

## [0.20.0] - 2025-07-06

### rten-examples

- Support stereo .wav files in Whisper example (https://github.com/robertknight/rten/pull/785)

### rten-generate

- Support pre-allocation of KV-cache capacity (https://github.com/robertknight/rten/pull/786)

### rten-tensor

- Fixed invalid use of `memcpy` on non-`Copy` types when copying tensors with
  contiguous last axis (https://github.com/robertknight/rten/pull/773)

- Make `fold` and fold-based operations on iterators more efficient
  (https://github.com/robertknight/rten/pull/772)

- Removed inherent methods of layouts which duplicated trait methods
  (https://github.com/robertknight/rten/pull/771)

- Support tensors with borrowed layouts (https://github.com/robertknight/rten/pull/770)

### rten

- Support fusing Add + Softmax where `axis` attribute is positive
  (https://github.com/robertknight/rten/pull/794)

- Support fusing LayerNormalization and RmsNormalization ops where `ReduceMean`
  op has `axes` specified via input rather than attribute and/or a positive
  value (https://github.com/robertknight/rten/pull/793)

- Improve buffer re-use if `Slice` op falls back to non in-place operation
  (https://github.com/robertknight/rten/pull/789)

- Added `Reciprocal` op fusion (https://github.com/robertknight/rten/pull/790)

- Allow ops with fused input transposes to run in place. This fixes a regression
  in performance for the Whisper example (https://github.com/robertknight/rten/pull/787)

- Make TensorPool usable across threads (https://github.com/robertknight/rten/pull/782)

- Parallelize QuantizeLinear and DynamicQuantizeLinear (https://github.com/robertknight/rten/pull/780)

- Fixed issue where operator fusions were not applied correctly for subgraphs
  (https://github.com/robertknight/rten/pull/779)

- Fixed timings for operators in subgraphs being counted multiple times (https://github.com/robertknight/rten/pull/778)

- Parallelize BatchNormalization, InstanceNormalization (https://github.com/robertknight/rten/pull/777)

- Fixed depthwise ConvInteger (https://github.com/robertknight/rten/pull/776)

- Optimize ConvTranspose via parallelized and more efficient col2im step
  (https://github.com/robertknight/rten/pull/775)

- Prevent fusion of subgraphs with shared nodes (https://github.com/robertknight/rten/pull/768)

- Handle non-contiguous mask tensor more efficiently in fused Add + Softmax
  (https://github.com/robertknight/rten/pull/766)

## [0.19.0] - 2025-06-16

**Breaking changes:**

The `Input`, `Output` and `InputOrOutput` types have been renamed to
`ValueView`, `Value` and `ValueOrView` respectively. This reflects the fact these
types are owned or borrowed values of runtime-determined type, rather than just
model inputs or outputs.

### rten

- Improved `Gelu` performance by using native `abs(x)` instruction
  (https://github.com/robertknight/rten/pull/755)

- Added `Add -> Softmax` fusion (https://github.com/robertknight/rten/pull/754)

- Enabled `Transpose -> {Split, Slice, Concat, Expand}` fusion
  (https://github.com/robertknight/rten/pull/747)

- Fixed panic during graph optimization if an operator takes the same input
  value in multiple positions (https://github.com/robertknight/rten/pull/746)

- Fixed warning about `stdarch_x86_avx512` feature opt-in not being required on
  latest Rust nightly builds (https://github.com/robertknight/rten/pull/744)

- Implemented `EyeLike` operator (https://github.com/robertknight/rten/pull/741)

- Improved formatting of tensor types and shapes in errors (https://github.com/robertknight/rten/pull/735)

- Fixed unnecessary large deallocation in `Expand` operator (https://github.com/robertknight/rten/pull/734)

- Added `Tile` fast path for when innermost dimensions are not repeated
  (https://github.com/robertknight/rten/pull/733)

- Renamed `Input` and `Output` types (https://github.com/robertknight/rten/pull/731).
  The previous names are retained as deprecated aliases (https://github.com/robertknight/rten/pull/752).

- Improved error messages when operator inputs are invalid due to a rank
  mismatch (https://github.com/robertknight/rten/pull/730)

- Reduced graph optimization time by reducing number of graph traversals
  (https://github.com/robertknight/rten/pull/726)

- Support `Pow` operator with different base and exponent types
  (https://github.com/robertknight/rten/pull/724)

- Added information about operator input shapes and data types to operator
  errors (https://github.com/robertknight/rten/pull/721,
  https://github.com/robertknight/rten/pull/723)

- Combined profiling outputs for main graph and subgraphs (https://github.com/robertknight/rten/pull/718)

- Added support for `approximate="tanh"` attribute in `Gelu` operator and
  support fusing approximate Gelu subgraphs (https://github.com/robertknight/rten/pull/717)

- Improved efficiency of applying unary op fusions to graphs
  (https://github.com/robertknight/rten/pull/716)

- Added API and CLI options to set number of threads used for inference
  (https://github.com/robertknight/rten/pull/712)

- Updated AVX-512 support for latest Rust nightly (https://github.com/robertknight/rten/pull/711)

- Fixed optimizer issue where optimized graph could run the same operator
  multiple times if intermediate values in fused subgraphs were used outside
  the subgraph (https://github.com/robertknight/rten/pull/709)

- Fixed performance regression in int8 model inference on Arm under Rust v1.87+
  (https://github.com/robertknight/rten/pull/706)

- Fixed issue where performance of vectorized unary operators could significantly
  degrade in certain contexts (https://github.com/robertknight/rten/pull/705)

- Removed the `{operation}_in_place` functions for most unary operators
  (https://github.com/robertknight/rten/pull/703)

- Parallelized `LayerNormalization`, `RmsNormalization` operators
  (https://github.com/robertknight/rten/pull/698)

- Support fusing `Transpose` + `MatMul` subgraphs where both inputs are
  transposed (https://github.com/robertknight/rten/pull/696)

- Support `Split` v13 operators that don't specify the expected number of
  outputs (https://github.com/robertknight/rten/pull/692)

- Improved execution planning error messages, e.g. when there is no way to
  compute the requested outputs given the inputs (https://github.com/robertknight/rten/pull/688,
  https://github.com/robertknight/rten/pull/690)

- Improved input validation in `Slice` op (https://github.com/robertknight/rten/pull/687)

- Fixed issue where fast paths were not always used in `Slice` op
  (https://github.com/robertknight/rten/pull/686)

### rten-cli

- Fixed issue where run timings were rounded down to nearest millisecond
  (https://github.com/robertknight/rten/pull/756)

- Output detailed profiling information if `-p`/`--profile` flag is repeated
  (https://github.com/robertknight/rten/pull/753)

- Improved formatting of errors in CLI (https://github.com/robertknight/rten/pull/729)

### rten-generate

- Added `Metrics::token_count` API which provides a convenient way to get the
  total number of generated tokens (https://github.com/robertknight/rten/pull/699)

### rten-tensor

- Improved `Iterator::nth` worst-case performance for several iterators
  (https://github.com/robertknight/rten/pull/750)

- Optimized `TensorBase::inner_iter` and related iterators (https://github.com/robertknight/rten/pull/748)

- Added `DoubleEndedIterator` support to several iterators (https://github.com/robertknight/rten/pull/720,
  https://github.com/robertknight/rten/pull/749)

- Optimized tensor iteration for non-contiguous tensors (https://github.com/robertknight/rten/pull/713,
  https://github.com/robertknight/rten/pull/757)

## [0.18.0] - 2025-05-08

### rten

- Optimized `Softmax`, `Erf` and `Gelu` using reduced-range exp function
  (https://github.com/robertknight/rten/pull/682,
  https://github.com/robertknight/rten/pull/684)

- Optimized copying of transposed and partly-contiguous tensors
  (https://github.com/robertknight/rten/pull/681)

- Optimized int8 matrix multiplication on Arm using indexed UDOT instructions
  (https://github.com/robertknight/rten/pull/680)

- Optimized normalization operators with fast paths in kernels
  (https://github.com/robertknight/rten/pull/677, https://github.com/robertknight/rten/pull/678)

- Optimized min/max reductions in `DynamicQuantizeLinear` (https://github.com/robertknight/rten/pull/676)

- Fixed deserialization of `CastLike` operators (https://github.com/robertknight/rten/pull/673)

- Improved model load errors by adding node names and more attribute details
  (https://github.com/robertknight/rten/pull/672, https://github.com/robertknight/rten/pull/674)

- Optimized reductions in `LayerNormalization` and `Softmax` operators by
  improving instruction level parallelism (https://github.com/robertknight/rten/pull/671,
  https://github.com/robertknight/rten/pull/683)

- Fixed "instruction requires: dotprod" error in Linux build on Arm and added
  Arm Linux CI (https://github.com/robertknight/rten/pull/670,
  https://github.com/robertknight/rten/pull/669)

- Optimized f32 matrix multiplication on Arm by adjusting tile size and using
  indexed FMLA instructions (https://github.com/robertknight/rten/pull/666,
  https://github.com/robertknight/rten/pull/679)

- Optimized `AveragePool` and `MaxPool` by using separate loops for padding and
  non-padding regions of input (https://github.com/robertknight/rten/pull/665)

- Support 1D and 2D inputs in `BatchNormalization` (https://github.com/robertknight/rten/pull/663)

- Optimized `BatchNormalization` and `InstanceNormalization` by removing tensor
  slicing overhead (https://github.com/robertknight/rten/pull/661,
  https://github.com/robertknight/rten/pull/662)

- Fixed error when reducing multiple axes if reduced chunks are not contiguous
  (https://github.com/robertknight/rten/pull/660)

### rten-generate

- Improved performance of ArgMax sampler (https://github.com/robertknight/rten/pull/667)

## [0.17.0] - 2025-04-09

This release is largely an internal refactoring to reduce the amount of unsafe
code in SIMD vectorized kernels and prepare for supporting more data types in
future. This has been achieved by creating a new portable SIMD API in the
rten-simd crate, which uses witness types to enable writing vectorized
operations with safe code. See the rten-simd crate docs for more details.

### rten

- Support `num_outputs` attribute in `Split` operator (https://github.com/robertknight/rten/pull/658)

- Support `Dropout` operator (inference mode only) (https://github.com/robertknight/rten/pull/652)

- Added option to use TorchDynamo ONNX export in `tools/export-timm-model.py`
  script (https://github.com/robertknight/rten/pull/651)

- Support `value_{float, floats, int, ints}` attributes in
  `Constant` operator (https://github.com/robertknight/rten/pull/649)

- Support `start` and `end` attributes for `Shape` operator (https://github.com/robertknight/rten/pull/648)

- Support `pytorch_half_pixel` value for `coordinate_transform_mode` in `Resize`
  operator (https://github.com/robertknight/rten/pull/647,
  https://github.com/robertknight/rten/pull/654)

- Support `CastLike` operator (https://github.com/robertknight/rten/pull/646)

- Fixed broken links to model files in ImageNet example (https://github.com/robertknight/rten/pull/603)

- Fixed broken `ArgMin` operator (https://github.com/robertknight/rten/pull/592)

- Support vector inputs in `MatMulInteger` (https://github.com/robertknight/rten/pull/589)

- Reduced use of unsafe code in vectorized SIMD kernels. See tracking issue
  in https://github.com/robertknight/rten/issues/549.

### rten-tensor

- Add `Tensor::{slice_axis, slice_axis_mut}` methods (https://github.com/robertknight/rten/pull/657)

- Export `AssumeInit` utility trait (https://github.com/robertknight/rten/pull/598)

- Add `CowTensor` and `CowNdTensor` alias for maybe-owned tensors (https://github.com/robertknight/rten/pull/568)

- Make `Tensor::broadcast` panic error message more helpful (https://github.com/robertknight/rten/pull/588)

- Provide more detailed messages in debug builds if argument to
  `Tensor::{size, stride}` is invalid (https://github.com/robertknight/rten/pull/565)

### rten-simd

The APIs for this internal crate have changed completely to support defining
operations without `unsafe`.

### rten-vecmath

Operations in this internal crate were changed to use the new safe SIMD API in
rten-simd.

## [0.16.0] - 2025-02-08

This release adds support for running models that have been quantized to int8,
taking advantage of dot product CPU instructions if available.
A guide to quantization support in RTen has been added in `docs/quantization.md`.
The `tools/ort-quantize.py` script in the rten repository has been updated to
provide an easy way to quantize ONNX models with recommended settings.

Further optimizations for quantized model inference will come in future releases.

### rten

- Added a guide to quantization support in RTen (https://github.com/robertknight/rten/pull/584)

- Support `axis` equal to input rank in `Flatten` operator (https://github.com/robertknight/rten/pull/577)

- Support all tensor types in `Split` operator (https://github.com/robertknight/rten/pull/576)

- Support `auto_pad=VALID` in `Conv` and other operators (https://github.com/robertknight/rten/pull/575)

- Support `ConvInteger` operator (https://github.com/robertknight/rten/pull/566,
  https://github.com/robertknight/rten/pull/570)

- Set default thread count on macOS to match performance core count rather than
  total core count (https://github.com/robertknight/rten/pull/552)

- `ort-quantize.py` now avoids quantizing `Conv` operators by default to work
  around an issue in ONNX Runtime. If the produced model does not need to be
  compatible with ORT, quantization can be enabled using `--quantize-conv`
  (https://github.com/robertknight/rten/pull/550,
  https://github.com/robertknight/rten/pull/566)

- Support models with subgraphs in `tools/ort-quantize.py` script and adjust
  configuration so that it produces usable results with more models
  (https://github.com/robertknight/rten/pull/530)

- Added initial optimized implementation of `MatMulInteger` for x64 (AVX2,
  AVX512 VNNI) and Arm 64 (with dotprod extensions)
  (https://github.com/robertknight/rten/pull/528,
  https://github.com/robertknight/rten/pull/535,
  https://github.com/robertknight/rten/pull/537,
  https://github.com/robertknight/rten/pull/541,
  https://github.com/robertknight/rten/pull/542,
  https://github.com/robertknight/rten/pull/543)

- Optimized and vectorized `DynamicQuantizeLinear` and `QuantizeLinear` operations
  (https://github.com/robertknight/rten/pull/531,
  https://github.com/robertknight/rten/pull/532,
  https://github.com/robertknight/rten/pull/538)

- Fixed edge case bug with incorrect handling of fused MatMul-Add operations
  when K dimension (LHS column count) is zero (https://github.com/robertknight/rten/pull/526)

- Fixed panic in `Conv` operator if group count is zero (https://github.com/robertknight/rten/pull/523)

- Support `MatMulInteger` operators where zero point is a vector (https://github.com/robertknight/rten/pull/521,
  https://github.com/robertknight/rten/pull/572)

### rten-examples

- Added ModernBERT masked word prediction example (https://github.com/robertknight/rten/pull/520)

### rten-tensor

- Added `CowTensor` and `CowNdTensor` type aliases for tensors which can be
  either owned or borrowed (https://github.com/robertknight/rten/pull/568)

## [0.15.1] - 2025-01-06

### rten

- Refactored matrix multiplication internals to prepare for supporting additional
  data types and architectures (https://github.com/robertknight/rten/pull/510,
  https://github.com/robertknight/rten/pull/511,
  https://github.com/robertknight/rten/pull/513,
  https://github.com/robertknight/rten/pull/519)

- Optimized Softmax by using multiplication-by-reciprocal instead of
  division (https://github.com/robertknight/rten/pull/516)

- Optimized matrix multiplication with specialized code for edge tiles
  (https://github.com/robertknight/rten/pull/505), more efficient indexing
  into LHS / A input (https://github.com/robertknight/rten/pull/512) and
  more aggressive unrolling (https://github.com/robertknight/rten/pull/518)

- Fuse RMSNorm subgraphs (https://github.com/robertknight/rten/pull/497)

- Optimized Gather with fast path for common case of axis=0 and faster general case
  (https://github.com/robertknight/rten/pull/496)

### rten-cli

- Error if a dimension size specified with `--size` does not match any model
  input (https://github.com/robertknight/rten/pull/517)

- Made output less noisy when a dimension size repeated in many inputs is not
  specified and is defaulted to 1 (https://github.com/robertknight/rten/pull/517)

- Prefix timing for each run with a run number (https://github.com/robertknight/rten/pull/508)

## [0.15.0] - 2024-12-28

### rten

- Fuse and vectorize Swish activation function used in CLIP and other models
  (https://github.com/robertknight/rten/pull/493).

- Avoid redundant zeroing of output buffer in `Gather` operator
  (https://github.com/robertknight/rten/pull/492)

- Fuse `MatMul` + `Mul` or `Div` by constant on either inputs or outputs
  (https://github.com/robertknight/rten/pull/487,
  https://github.com/robertknight/rten/pull/489).  In Transformers this occurs
  in the context of Scaled Dot Product Attention.

- Fix panic if `Model::run` is passed an input or output node ID which refers to
  an operator node rather than a value or constant node
  (https://github.com/robertknight/rten/pull/485).

- Support prepacked weights. This increases model load time and memory usage but
  decreases inference time. Weight pre-packing is disabled by default and can be
  enabled via `ModelOptions::prepack_weights`
  (https://github.com/robertknight/rten/pull/483).

- Support fusing LayerNormalization operator variants that don't use a bias,
  such as found in ModernBERT and other models (https://github.com/robertknight/rten/pull/470).

- Support `DepthToSpace` operator (https://github.com/robertknight/rten/pull/468)

- Support fusing `Add(MatMul(a, b), bias)` subgraphs (https://github.com/robertknight/rten/pull/462)

- Improved Where operator performance by removing an old "fast" path that is
  now slower than the standard path (https://github.com/robertknight/rten/pull/460)

- Optimized ReduceMean, ReduceL2 operators using SIMD (
  https://github.com/robertknight/rten/pull/457)

- Unified and optimized implementation of normalization operators (BatchNormalization,
  InstanceNormalization, LayerNormalization) using SIMD (
  https://github.com/robertknight/rten/pull/456, https://github.com/robertknight/rten/pull/457,
  https://github.com/robertknight/rten/pull/465, https://github.com/robertknight/rten/pull/469,
  https://github.com/robertknight/rten/pull/471).

- Added Nougat PDF-to-Markdown OCR example (https://github.com/robertknight/rten/pull/448).

- Make Depth Anything example support variants with 3D (instead of 4D) outputs
  (https://github.com/robertknight/rten/pull/447).

- Make error message more helpful if converting a model `Output` into an
  `NdTensor` fails due to a rank (dimension count) mismatch
  (https://github.com/robertknight/rten/pull/446)

- Release buffer back to memory pool in Concat op if in-place concatenation
  is not possible (https://github.com/robertknight/rten/pull/426)

- Enable Resize operations to be very cheap if the target size is the same as
  the input size (https://github.com/robertknight/rten/pull/423)

- Reduced some unnecessary memory reservation when constructing model graph
  (https://github.com/robertknight/rten/pull/422).

- Added CLIP example (https://github.com/robertknight/rten/pull/421). This
  computes similarity between images and text labels.

- Added data type information to model inputs and outputs
  (https://github.com/robertknight/rten/pull/420)

- Support vector inputs in MatMul operator (https://github.com/robertknight/rten/pull/418)

- Support additional DETR-based models in the DETR example, such as
  Table Transformer (https://github.com/robertknight/rten/pull/413)

### rten-tensor

**Breaking changes:** The result of `TensorBase::reshaped` now has a shorter
lifetime as it may be an owned tensor instead of a view. Method call chains that
used `reshaped` in the middle may need to be split into separate statements.

- Support indexing 1D tensors using scalars instead of arrays
  (https://github.com/robertknight/rten/pull/480).

- Support using slice ranges with steps in `TensorBase::slice`
  (https://github.com/robertknight/rten/pull/464)

- `TensorBase::reshaped` now copies its input instead of panicking if non
  contiguous. As a result it returns a copy-on-write (maybe owned) tensor with a
  shorter lifetime.

### rten-text

- Support `Lowercase`, `Replace`, `Sequence` normalizers
  (https://github.com/robertknight/rten/pull/451)

- Support all the Unicode normalization normalizers (NFC, NFD, NFKC, NFKD)
  (https://github.com/robertknight/rten/pull/450)

- Support `Digits`, `Sequence`, `Split` pre-tokenizers
  (https://github.com/robertknight/rten/pull/449)

- Add `Tokenizer::from_file` convenience method
  (https://github.com/robertknight/rten/pull/445)

- Added `Tokenizer::{encode, decode}` methods for more ergonomic tokenization
  and de-tokenization of text (https://github.com/robertknight/rten/pull/429)

- Started to revise tokenization pipeline to follow the one used by HuggingFace
  Tokenizers (https://github.com/robertknight/rten/pull/428,
  https://github.com/robertknight/rten/pull/429, https://github.com/robertknight/rten/pull/430,
  https://github.com/robertknight/rten/pull/440, https://github.com/robertknight/rten/pull/441,
  https://github.com/robertknight/rten/pull/443, https://github.com/robertknight/rten/pull/444,
  https://github.com/robertknight/rten/pull/452)

- Support `end_of_word_suffix` in BPE model
  (https://github.com/robertknight/rten/pull/425)

## [0.14.1] - 2024-11-16

This release adds Serde support for rten tensors and several optimizations which
allow the Whisper example to run significantly faster.

### rten-tensor

- Support (de-)serializing tensors using Serde (https://github.com/robertknight/rten/pull/402)

### rten

### Examples

- Output transcription speed as a multiple of real-time in Whisper example
  (https://github.com/robertknight/rten/pull/403)

- Support longer audio inputs and normalize inputs in wav2vec2 speech
  recognition example (https://github.com/robertknight/rten/pull/400)

### Bug fixes

- Fixed an issue where metadata associated with output value nodes was lost
  after a graph fusion. In the Whisper example this prevented several Transpose-MatMul
  fusions from being used (https://github.com/robertknight/rten/pull/401).

### Performance improvements

- Added fast path for ArgMin / ArgMax for case when axis has unit stride
  (https://github.com/robertknight/rten/pull/411)

- Optimized GatherND by avoiding redundant zeroing of output and adding fast
  path for contiguous inputs (https://github.com/robertknight/rten/pull/410)

- Optimized copying of tensors with 5+ dimensions (https://github.com/robertknight/rten/pull/409)

- Operators in subgraphs which capture their first input from a parent graph can
  now run in-place (https://github.com/robertknight/rten/pull/407)

- After the initial execution plan is created, it is now re-ordered to enable
  more operations to run in-place
  (https://github.com/robertknight/rten/pull/405)

### rten-generate

- The strategy for reserving capacity for KV-cache growth has been modified to
  work with models that don't append to KV-cache inputs on the first run.
  This benefits Hugging Face "merged" transformer models with "past" and "no-past"
  branches (https://github.com/robertknight/rten/pull/408)

## [0.14.0] - 2024-10-27

### Breaking changes

- The `NodeId` type used to identify model inputs and outputs is now an opaque
  `u32`-sized type instead of a `usize`
  (https://github.com/robertknight/rten/pull/381)

- The tensor slicing APIs (`TensorBase::slice` etc.) now infer the rank of the
  output automatically, instead of requiring the caller to specify. See
  https://github.com/robertknight/rten/pull/367.

### rten

#### New features

- Added Whisper speech recognition example (https://github.com/robertknight/rten/pull/397)

- Added background removal example using RMBG (https://github.com/robertknight/rten/pull/344)

- Support i8 and u8 tensors in operator inputs, outputs and model weights
  (https://github.com/robertknight/rten/pull/345).

- Support 8-bit int tensors in Cast, Gather, GatherElements, GatherND,
  ScatterElements, ScatterND, Expand, Flatten, Reshape, Squeeze, Transpose,
  Pad, Unsqueeze ops (https://github.com/robertknight/rten/pull/387)

- Implement `QuantizeLinear`, `DequantizeLinear` and `DynamicQuantizeLinear`
  ops (https://github.com/robertknight/rten/pull/346)

- Added reference implementation of `MatMulInteger`. Quantized models using this
  operator will now run, but very slowly. Optimized execution for quantized
  models will come in future releases (https://github.com/robertknight/rten/pull/356).

- Support f16 models in model converter by widening to f32 (https://github.com/robertknight/rten/pull/372).
  This is an interim measure until f16 tensors are properly supported in RTen.

- Added YOLOv11 support to YOLO example (https://github.com/robertknight/rten/pull/374)

#### Bug fixes

- Fixed AVX-512 build (https://github.com/robertknight/rten/pull/376)

- Fixed graph optimizations not being applied correctly when a fused operation
  feeds directly into a subsequent fused operation (https://github.com/robertknight/rten/pull/369)

- Fixed errors when running WebAssembly builds compiled without SIMD support (https://github.com/robertknight/rten/pull/348)

#### Performance improvements

- Made `NodeId` a u32-sized type with a niche, reducing the size of various
  internal data structures (https://github.com/robertknight/rten/pull/381)

- Optimized `Cast` op when source and dest types are the same (https://github.com/robertknight/rten/pull/388)

- Avoid unnecessary copying in `Squeeze` and `Unsqueeze` ops (https://github.com/robertknight/rten/pull/339,
  https://github.com/robertknight/rten/pull/340)

### rten-cli

- Added `--no-optimize` flag to enable testing impact of graph optimizations
  (https://github.com/robertknight/rten/pull/368)

### rten-generate

- Added more context to token generation errors (https://github.com/robertknight/rten/pull/396)

- Support `cache_position` input in models exported from Optimum
  (https://github.com/robertknight/rten/pull/395)

- Added API for modifying model outputs ("logits") before sampling
  (https://github.com/robertknight/rten/pull/393,
  https://github.com/robertknight/rten/pull/394)

- Support the new `merges` format in tokenizer.json files exported by
  current versions of HuggingFace Transformers
  (https://github.com/robertknight/rten/pull/392)

### rten-imageproc

- Added `normalize_image` utility (https://github.com/robertknight/rten/pull/343)

### rten-tensor

- Improved debug formatting of tensors (https://github.com/robertknight/rten/pull/377)

- Changed `TensorBase::slice` to infer the rank of the output based on the
  rank of the input and the number of index entries in the slice arguments
  (https://github.com/robertknight/rten/pull/367).

## [0.13.1] - 2024-08-30

### rten

#### New features

- Added speech detection example using Silero VAD
  (https://github.com/robertknight/rten/pull/338)

- Support int tensors in ArgMin and ArgMax ops
  (https://github.com/robertknight/rten/pull/329)

- Support "reflect" padding mode (https://github.com/robertknight/rten/pull/326)

#### Bug fixes

- Fixed panic with certain combinations of input, kernel size and padding in
  depthwise convolution (https://github.com/robertknight/rten/pull/336)

- Fixed attempted out-of-bounds slice in depthwise convolution when input tensor
  has a row stride that exceeds the row length
  (https://github.com/robertknight/rten/pull/335)

- Fixed conversion of `auto_pad` attribute for Conv operator
  (https://github.com/robertknight/rten/pull/333)

- Round timings to microseconds in verbose log
  (https://github.com/robertknight/rten/pull/331)

- Fixed panic when slicing empty tensors
  (https://github.com/robertknight/rten/pull/325)

- Fixed 1D convolution failing with non-contiguous inputs
  (https://github.com/robertknight/rten/pull/324)

- Fixed conversion of shape information for scalar tensors
  (https://github.com/robertknight/rten/pull/323)

- Fixed panic in softmax if the size of the normalized axis is zero
  (https://github.com/robertknight/rten/pull/322)

### rten-cli

- Added `--mmap` flag to load model using memory mapping instead of reading
  whole file into a buffer (https://github.com/robertknight/rten/pull/330)

## [0.13.0] - 2024-08-24

This release adds the infrastructure to support subgraphs, which are used in
control flow operators like `If`, plus an implementation of the `If` operator
and a TrOCR example which uses it.

### rten

- Added [`Model::load_static_slice`](https://github.com/robertknight/rten/pull/316)
  API which can be used to load models embedded in the binary with
  `include_bytes!`. Thanks @hsfzxjy.

- Added TrOCR example (https://github.com/robertknight/rten/pull/304)

- Support `If` operator (https://github.com/robertknight/rten/pull/306)

- Added full support for `Einsum` operator (https://github.com/robertknight/rten/pull/297,
  https://github.com/robertknight/rten/pull/299, https://github.com/robertknight/rten/pull/300,
  https://github.com/robertknight/rten/pull/302, https://github.com/robertknight/rten/pull/303)

### rten-cli

- Added `--quiet` flag (https://github.com/robertknight/rten/pull/313)

- Inputs named `use_cache_branch` now get a default value of `0` (ddf4109)

### rten-generate

- Support models with cross-attention KV caches that are computed on the first
  run of the decoder (https://github.com/robertknight/rten/pull/318). This
  is used by Hugging Face models for encoder-decoder systems.

- Support models without a KV cache (https://github.com/robertknight/rten/pull/305)

### rten-tensor

- Added `Tensor::remove_axis` (b823d46)
- Added `Tensor::from_storage_and_layout` (54d2941)

### rten-text

- The BPE tokenizer no longer complains if a tokenizer contains tokens in the
  vocabulary which are never generated by merges and are not added special
  tokens (18e9b2a)

## [0.12.0] - 2024-07-30

### rten

#### Breaking changes

- The `rten-convert` tool now generates models in the V2 format by default
  (https://github.com/robertknight/rten/pull/272).
  These models can only be loaded by RTen version 0.11.0 or later. The V1
  format can be generated by specifying the `--v1` flag. The `rten` crate can
  load both V1 and V2 format models.

  See [the `.rten` file format documentation](https://github.com/robertknight/rten/blob/main/docs/rten-file-format.md)
  for more details.

- The `reduce_{max, min, sum}` tensor methods have moved from the
  `FloatOperators` trait to the `Operators` trait (https://github.com/robertknight/rten/pull/274).

#### Examples and documentation

- Added Segment Anything example (https://github.com/robertknight/rten/pull/295).
  This supports the original SAM models plus several derivatives with
  lighter-weight image encoders.

- Added chatbot example using Qwen2 (https://github.com/robertknight/rten/pull/282).
  This also works with [SmolLM](https://huggingface.co/blog/smollm).

- `Model::load_mmap` docs now have a better explanation of the memory and
  performance impact (ce0b717)

#### New features

- Added partial support for `Einsum` operator (https://github.com/robertknight/rten/pull/295).

#### Performance improvements

- Avoid allocations in most cases when broadcasting tensor shapes (c4b5f26).

- Strides of size-1 dimensions are ignored when determining whether a tensor is
  contiguous (https://github.com/robertknight/rten/pull/292). This allows more
  operations to use fast paths for contiguous tensors.

- Optimized `LayerNormalization` and `ReduceMean` (https://github.com/robertknight/rten/pull/291)

- Added fast-path for `Resize` operator when input scale is 1 (https://github.com/robertknight/rten/pull/290)

- Return input buffer to pool in `Cast` operator if input needs to be copied
  (https://github.com/robertknight/rten/pull/289).

- Implemented `LayerNormalization` fusion (https://github.com/robertknight/rten/pull/280)

- Implemented `GELU` fusion (https://github.com/robertknight/rten/pull/277)

### rten-cli

- Inputs with names matching the pattern `*_ids` now use zero as the
  auto-generated input value (78cd621)

### rten-generate

- `TopKSampler` now supports specifying a temperature (65b837b)

- Added `Generator::append_prompt` to append to prompt after initial generation.
  This is useful for chat-like applications (5ef3cb2)

- Fixed an issue where `attention_mask` input had the wrong size (cae6134)

### rten-tensor

#### Breaking changes

- The `tensor` and `ndtensor` macros have been deprecated in favor of
  `Tensor::from` and `NdTensor::from` (https://github.com/robertknight/rten/pull/286).

#### Other changes

- `Tensor::from` now supports creating tensors from scalar values (d2ca876)

- `Tensor::lanes` iterator performance was improved by making them exact-sized
  and fused (9e31556)

### rten-text

- Token IDs are now represented as `u32` rather than `usize`, for consistency
  with rten-generate (https://github.com/robertknight/rten/pull/288).

- The `vocab` mapping in `tokenizer.json` files is now used to determine token
  IDs when decoding (https://github.com/robertknight/rten/pull/287).

## [0.11.1] - 2024-07-17

### rten

- Fixed a crash in WebAssembly due to unsupported use of `Instant::now`
  (https://github.com/robertknight/rten/pull/283).

## [0.11.0] - 2024-07-05

### rten

#### Breaking changes

- The `inputs` argument to `Model::run` now accepts a `Vec<(NodeId,
  InputOrOutput)>` instead of `&[(NodeId, Input)]`, where `InputOrOutput` is an
  enum that is either an owned `Tensor` or a `TensorView`. This enables passing
  ownership of an input to `Model::run`, which is in turn enables efficient
  in-place updates to cache-like inputs.

  The `InputOrOutput` type implements `From` for tensors and tensor views, so
  code such as:

  ```rs
  model.run(&[(input_id, tensor_view.into())], output_ids, None)
  ```

  Becomes:

  ```rs
  model.run(vec![(input_id, tensor_view.into())], output_ids, None)
  ```

#### New features

- Add a new version of the `.rten` file format which supports models over 2GB
  in size. The `rten-convert` tool still generates V1 models by default but
  will generate the V2 format if the `--v2` flag is provided
  (https://github.com/robertknight/rten/pull/260).

- Support `Gelu` operator (https://github.com/robertknight/rten/pull/248)

#### Bug fixes

- Prevent `Model::partial_run` from propagating values through randomized
  operators (https://github.com/robertknight/rten/pull/240).

- Improved accuracy of timing metrics and eliminated unaccounted for
  ("[Other]") time https://github.com/robertknight/rten/pull/254.

#### Performance improvements

This release adds a new graph optimization step as part of loading models. This
performs fusions and other optimizations to speed up inference. These
optimizations are enabled by default, but can be disabled via options in
`ModelOptions`.

- Improved parallelism in the `Softmax` operator (https://github.com/robertknight/rten/pull/258)

- Made `Tensor::inner_iter` faster (https://github.com/robertknight/rten/pull/259)

- Made `Gather`, `Concat` and `Unsqueeze` operators faster for small inputs.
  These operations are common in subgraphs that operator on tensor shapes.
  https://github.com/robertknight/rten/pull/255,
  https://github.com/robertknight/rten/pull/256,
  https://github.com/robertknight/rten/pull/257.

- Optimized vector-matrix multiplication (https://github.com/robertknight/rten/pull/250,
  https://github.com/robertknight/rten/pull/253). This benefits transformer
  decoder inference when the batch size is 1.

- Fuse `Mul(X, Sigmoid(X))` subgraphs into a `Silu` operation. This speeds up
  YOLOv8 by 8%. See https://github.com/robertknight/rten/pull/246.

- Further reduce small allocations during graph execution
  (https://github.com/robertknight/rten/pull/243,
  https://github.com/robertknight/rten/pull/245).

- Fuse `MatMul(Transpose(X), Y)` subgraphs to avoid materializing the transposed
  matrix (https://github.com/robertknight/rten/pull/242).

- Perform constant propagation when loading models
  (https://github.com/robertknight/rten/pull/241).

- Enabled `Concat` operator to run in-place if the caller has specifically
  reserved space in the first input's buffer
  (https://github.com/robertknight/rten/pull/239).

- Cache the last-used execution plan. This avoids recomputing the sequence of
  execution steps when a model is run in a loop
  (https://github.com/robertknight/rten/pull/234).

- Improved performance of unary operators for non-contiguous inputs
  (https://github.com/robertknight/rten/pull/223)

- Optimized `Where` operator for non-contiguous inputs
  (https://github.com/robertknight/rten/pull/213)

- Optimized variadic operators (https://github.com/robertknight/rten/pull/212)

- Optimized `Pow` operator (https://github.com/robertknight/rten/pull/219)

### rten-examples

- Added GPT-2 text generation example (https://github.com/robertknight/rten/pull/228)
- Added DistilViT image captioning example (https://github.com/robertknight/rten/pull/230)

### rten-generate

This is a new crate which provides a convenient `Iterator`-based interface for
running auto-regressive decoder models. See the `gpt2` and `distilvit` examples
in the `rten-examples` crate for code samples.

### rten-tensor

- Support more primitive element types in `NdTensor::from`
  (https://github.com/robertknight/rten/pull/226).

### rten-text

- Added Byte Pair Encoding (BPE) tokenizer (https://github.com/robertknight/rten/pull/227)

## [0.10.0] - 2024-05-25

### rten

#### Breaking changes

- RTen now creates its own Rayon thread pool where the number of threads is
  configured to match the physical rather than logical core count, rather than
  using the global Rayon thread pool. This improves performance on systems with
  Simultaneous Multi-Threading (aka. SMT, Hyper-Threading) (most x86_64 CPUs),
  but can lead to contention if the calling application has its own
  multi-threaded parallelism. Applications may need to adjust their own use of
  threading to avoid this. RTen provides functions for applications to run
  their own tasks within this thread pool.

  See https://github.com/robertknight/rten/pull/183.

#### Bug fixes

- Fixed conversion of `Transpose` operators without a `perm` attribute
  (https://github.com/robertknight/rten/pull/201)

- The `RunError` type returned by `Model::run` is now exported
  (https://github.com/robertknight/rten/pull/206)

#### Performance improvements

- Made `Resize` operator parallel over rows. This benefits resize operations on
  images with large spatial dimensions and few channels
  (https://github.com/robertknight/rten/pull/208).

- Improved performance of `Conv` operator on Intel CPUs with a mitigation for
  the Gather Data Sampling /
  "[Downfall](https://en.wikipedia.org/wiki/Downfall_(security_vulnerability))"
  vulnerability applied. This affects most 6th-11th generation Intel CPUs
  (https://github.com/robertknight/rten/pull/204).

- Optimized `Concat` operator when input is not contiguous (eg. following a
  `Slice` op) (https://github.com/robertknight/rten/pull/204)

- Improved performance of `GRU` operator by combining operations on separate
  gates (https://github.com/robertknight/rten/pull/188)

- Improved performance of binary operators on non-contiguous tensors
(https://github.com/robertknight/rten/pull/190)

### rten-cli

- Added `--n_iters` flag to control how many times the model is run (https://github.com/robertknight/rten/pull/202)

- Optimize model by performing constant propagation before running the model
  (https://github.com/robertknight/rten/pull/202)

- Made it easier to specify sizes for dynamic inputs. The new syntax is
  `--size dim_name=size`. Additionally the size for dynamic dimensions defaults
  to 1. See https://github.com/robertknight/rten/pull/182.

- Added `--version` flag (https://github.com/robertknight/rten/pull/181)

### rten-imageproc

- Added `serde_traits` feature which implements serde `Serialize` and
  `Deserialize` traits for geometry types
  (Thanks @luketpeterson, https://github.com/robertknight/rten/pull/198)

### rten-tensor

- Added `Tensor::split_at` and `Tensor::split_at_mut` (
  https://github.com/robertknight/rten/pull/205, https://github.com/robertknight/rten/pull/207)

- `Tensor::{axis_chunks, axis_chunks_mut}` iterators now preserve the layout
  in their output type (https://github.com/robertknight/rten/pull/207).

### rten-vecmath, rten-simd

- The internal crate providing portable SIMD and vectorized math functions
  was split into two. rten-simd now contains the portable SIMD code.
  rten-vecmath contains the vectorized math functions.

## [0.9.0] - 2024-05-16

### Breaking Changes

This release contains breaking changes to the model loading APIs and code using
the `TensorBase` type directly (as opposed to aliases like `Tensor`). See the
notes for the `rten` and `rten-tensor` crates respectively.

### rten

#### Breaking changes

- The `Model::load` API now takes a `Vec<u8>` rather than `&[u8]` as an
  argument. This enables it to avoid copying data internally. For the most
  common use case of loading a model from disk, use the new `Model::load_file`
  API.

- The `Model::load_with_ops` API has been replaced by `ModelOptions::with_ops`.

#### New features

- Added `Model::load_file` API for more convenient loading of a model from
  a file (https://github.com/robertknight/rten/pull/174)

- Added `Model::load_mmap` API for zero-copy loading of models by using
  memory maps. This can be faster than `Model::load` for very large models
  (https://github.com/robertknight/rten/pull/174).

- Added Piper text-to-speech example (https://github.com/robertknight/rten/pull/161)

- Support 1D inputs and padding in `ConvTranspose` (https://github.com/robertknight/rten/pull/156)

- Support `GatherND` operator (https://github.com/robertknight/rten/pull/155)

- Support `Softplus` operator (https://github.com/robertknight/rten/pull/146)

- Support converting ONNX models containing unnamed operator nodes
  (https://github.com/robertknight/rten/pull/143)

- Support `RandomNormal`, `RandomNormalLike`, `RandomUniformLike` operators
  (https://github.com/robertknight/rten/pull/144)

#### Bug fixes

- Fixed incorrect calculation of update slice size in `ScatterND` operator
  (https://github.com/robertknight/rten/pull/157)

- Fixed incorrect conversion of `axis` attribute for `ArgMin` and `ArgMax`
  operators (https://github.com/robertknight/rten/pull/142)

- Fixed uninitialized read in `Gemm` operator when `alpha != 1` and `beta == 0`
  (https://github.com/robertknight/rten/pull/150)

- Fixed `NonMaxSuppression` operator missing overlap of boxes due to confusion
  of X/Y coordinates (https://github.com/robertknight/rten/pull/177)

#### Optimizations

- Optimize `Gather`, `NonZero` operator by allocating from memory pool
  (https://github.com/robertknight/rten/pull/168)

- Optimize `Slice` operator when slice ranges contain negative steps
  (https://github.com/robertknight/rten/pull/167)

- Optimize `Pad` operator by making copying of non-contiguous views more
  efficient (https://github.com/robertknight/rten/pull/166)

- Optimize `Conv` operator by avoiding redundant zeroing of packing buffers,
  optimizing `im2col` setup (https://github.com/robertknight/rten/pull/165)

- Optimize `ConvTranspose` by fusing bias addition into `col2im` transform
  (https://github.com/robertknight/rten/pull/159)

- Parallelize `AveragePool` operator (https://github.com/robertknight/rten/pull/138)

- Improved model loading performance by avoiding copying weights in `Model::load`
  (https://github.com/robertknight/rten/pull/174)

### rten-imageproc

- The mask matrix argument to `find_contours` now uses `bool` instead of `i32`
  for elements. This improves performance / reduces memory usage for large masks.

### rten-tensor

#### Breaking changes

This release changes the signature of the `TensorBase` struct from
`TensorBase<T, S: AsRef<[T]>, L: MutLayout>` to `TensorBase<S: Storage, L:
MutLayout>`. The element type is now available via `S::Elem`. The type of `S`
used by views has changed from slices to new custom types. The
`TensorBase::from_data` method still accepts both `Vec<T>` and slices as the
`data` argument, and will convert to the appropriate storage struct.

Code using the type aliases (`Tensor`, `TensorView`, `TensorViewMut` etc.)
does not need to change.

#### New features

- Added `TensorBase::{as_cow, into_cow}` (named after `std::borrow::Cow`) to
  convert tensor storage to a type which is `Cow`-like. This is useful for
  writing code which works with either borrowed or owned tensors
  (https://github.com/robertknight/rten/pull/153).

#### Bug fixes

- Added missing checks for equality between old/new layout lengths in
  reshape operations (https://github.com/robertknight/rten/pull/170,
  https://github.com/robertknight/rten/pull/171)

- Improved internal checks that storage slicing does not lead to out-of-bounds
  accesses (https://github.com/robertknight/rten/pull/163)

- Refactored tensor storage types to fix a violation of Rust's unique ownership
  rules for mutable slices. This enables tests for rten-tensor and code using
  this crate to be run under Miri
  (https://github.com/robertknight/rten/pull/148).

### rten-vecmath

- Revised SIMD traits to make working with masks more ergonomic and efficient
  (https://github.com/robertknight/rten/pull/152). Integer and floating point
  types with the same number of lanes will now use the same mask type.

## [0.8.0] - 2024-04-29

### rten-tensor

- Added `Alloc` trait which provides a simple allocator interface, and
  `*_in`-suffixed variants of several `TensorBase` methods, which allows
  specifying an allocator for the returned tensor's data buffer
  (https://github.com/robertknight/rten/pull/123).

### rten-vecmath

- Fixed crashes in several functions when running on pre-AVX2 x64 CPUs (see
  `rten` changes)

### rten

#### New features

- Support `Elu` operator (https://github.com/robertknight/rten/pull/132)

- Support `Reduce*` operators that take `axes` as a dynamic input rather than
  static attribute (https://github.com/robertknight/rten/pull/132)

#### Bug fixes

- Fixed crash in several operators when running on x64 CPUs that do not
  support AVX-2 instructions (https://github.com/robertknight/rten/pull/131,
  https://github.com/robertknight/rten/pull/134)

#### Performance improvements

- Added a buffer pool that enables reuse of operator output and temporary
  buffers, avoiding the overhead of allocating and freeing large buffers using
  the system allocator (https://github.com/robertknight/rten/pull/108).

  Statistics about buffer pool usage are printed as part of `RTEN_TIMING`
  output.

- Fixed a `MatMul` performance regression introduced in v0.7.0 due to virtual
  calls to get kernel tile size (https://github.com/robertknight/rten/pull/101)

- Optimize convolutions by using SIMD operations for im2col transform
  (https://github.com/robertknight/rten/pull/104)

- Parallelize depthwise convolution (https://github.com/robertknight/rten/pull/102)

- Avoid redundant of zeroing buffers in `Conv`, `OneHot`, and various unary
  operations (https://github.com/robertknight/rten/pull/97,
  https://github.com/robertknight/rten/pull/99,
  https://github.com/robertknight/rten/pull/101,
  https://github.com/robertknight/rten/pull/106)

- Optimize `Unsqueeze` by running in-place where possible (https://github.com/robertknight/rten/pull/96)

- Optimize vector-matrix products where matrix is transposed (https://github.com/robertknight/rten/pull/94)

- Reduced graph execution overhead by using faster hashing (https://github.com/robertknight/rten/pull/92)

- Optimize `ScatterND` (https://github.com/robertknight/rten/pull/91)

- Support AVX-512 acceleration for `Exp`, `Sigmoid`, `Tanh`, `Softmax` and
  `Erf` operators (https://github.com/robertknight/rten/pull/131). This
  requires nightly Rust and the `avx512` feature enabled.

## [0.7.0] - 2024-04-12

### rten-tensor

- Add `Tensor::merge_axes` method to simplify layouts (https://github.com/robertknight/rten/pull/78)

- Add `Tensor::{uninit, assume_init}` methods for working with uninitialized
  buffers (https://github.com/robertknight/rten/pull/82)

### rten

- Reduced `Graph::run` overhead by reducing allocations
  (https://github.com/robertknight/rten/pull/89)

- Added `Model::partial_run` API to speed up autoregressive / recurrent models
  by precomputing parts of the graph that depend only on inputs that are
  unchanging across loop iterations (https://github.com/robertknight/rten/pull/86)

- Optimize `MatMul` and binary operators by avoiding unnecessary zeroing of
  output buffers (https://github.com/robertknight/rten/pull/82,
  https://github.com/robertknight/rten/pull/88)

- Fixed incorrect output from `Gemm` operator when the bias is zero and the "C"
  input contained infinities / NaNs (https://github.com/robertknight/rten/pull/81)

- Optimize matrix packing operations on Intel CPUs using AVX-2 instructions
  (https://github.com/robertknight/rten/pull/80)

- Optimize `Transpose` operations where input dimensions are powers of 2 by
  using blocking and tiling (https://github.com/robertknight/rten/pull/78)

- Exclude test files and tools from published crate
  (https://github.com/robertknight/rten/pull/77)

- Optimize RNN operators for the case where the input sequence is short, by
  avoiding prepacking of weights in this case
  (https://github.com/robertknight/rten/pull/74)

## [0.6.0] - 2024-03-31

### rten

- Updated AVX-512 support to work with latest Rust nightly releases
  (https://github.com/robertknight/rten/pull/58)

- Improved performance of vector-matrix product operations
  (https://github.com/robertknight/rten/pull/61)

- Slightly improved WASM matrix multiplication performance with a dedicated
  kernel (https://github.com/robertknight/rten/pull/64)

- Fixed conversion of RNN operators (LSTM, GRU) that explicitly declare the
  direction as forward (https://github.com/robertknight/rten/pull/67)

- Support tensors with 3 or 5+ dimensions in `BatchNormalization` operator
  (https://github.com/robertknight/rten/pull/68)

- Support `RandomUniform` operator (https://github.com/robertknight/rten/pull/69)

- Improve matrix prepacking performance by eliminating unnecessary
  zero-initialization of buffers (https://github.com/robertknight/rten/pull/70)

## [0.5.0] - 2024-02-29

### rten

 - Changed `OperatorType` enum in .rten schema from byte to ubyte, to allow for
   more operator types in future (https://github.com/robertknight/rten/pull/56)

 - Made `Model` instances `Send`, enabling use with PyO3 (https://github.com/robertknight/rten/pull/55)

 - The ONNX => rten model conversion tool is now an installable Python package
   called `rten-convert` (https://github.com/robertknight/rten/pull/53)

 - Implemented `ReduceSumSquare` operator (36bbf89f)

## [0.4.0] - 2024-02-08

### rten

 - Support `count_include_pad` attr in AveragePool operator (09ecb729)

 - Support license/version/provenance metadata in RTen models
   (https://github.com/robertknight/rten/pull/48)

 - Fix error when a negative index was used with `Gather` operator (573ded4c)

 - Improve performance of `MatMul` operator when row count of LHS is small and
   batch size is large (https://github.com/robertknight/rten/pull/51)

### rten-imageproc

 - Optimized `find_contours` for large images (c471a6c, 7a14f43)

### rten-tensor

 - Optimize `TensorBase::map` for contiguous tensors (5562fd23)
 - Add `TensorBase::{from_fn, from_simple_fn}` (5e654ea0)
 - Add `TensorBase::try_from_data` (18817907)
 - Support `get_unchecked` on owned/mutable tensors (06b02eaf)

## [0.3.1] - 2024-01-23

 - Updated rten-vecmath dependency to latest version

## [0.3.0] - 2024-01-23

### Breaking changes

The static and dynamic tensor types (`NdTensorBase`, `TensorBase`) have been
unified into a single implementation. Most code uses these via type aliases
(`NdTensor`, `Tensor` etc.), which remain the same. However there have been some
API changes as a result:

 - The `View` and `NdView` traits were combined into `AsView`. The recommended
   way to import this trait is via the prelude (`use rten_tensor::prelude::*`)

 - Some inherent methods of `TensorBase` moved to the `AsView` trait. You may
   need to add additional imports of this trait or the prelude.

 - `NdTensor::from_data` now has the same API signature as `Tensor::from_data`.
   This means the order of arguments is reversed compared to before. It is now
   `from_data(shape, data)`. Creating tensors with custom strides is now done
   via `from_data_with_strides` or `from_slice_with_strides`.

 - Tensor methods for broadcasting and reshaping tensors now determine the rank
   of the result from the type of the shape argument. If passed an array, they
   return a static-rank view. If passed a slice, they return a dynamic-rank
   view.

 - Methods that insert, remove or swap axes now have an `_axis` suffix (eg.
   `move_axis`). Previously some of these methods had a `_dim` suffix.

 - The `slice` method now always returns a static rank view. Usage is
   `tensor.slice::<M, _>(range)` where `M` is the rank of the result. To create
   a view with a dynamic dimension count, use `tensor.slice_dyn(range)` instead.

## New features

 - Implemented LayerNormalization operator
   ([#44](https://github.com/robertknight/rten/pull/44))
 - Added "Depth Anything" monocular depth estimation example
   ([#44](https://github.com/robertknight/rten/pull/44))
 - Added support for `align_corners` value for `coordinate_transformation_mode`
   attr in Resize operator ([#44](https://github.com/robertknight/rten/pull/44)).

## Performance improvements

 - Optimized index iteration for tensors (d3fd3c9)
 - Optimized col2im transform used by ConvTranspose (fbc541b)
 - Optimized depthwise convolution (20e83e8)
 - Improved performance on Arm via a better optimized GEMM kernel
   ([#32](https://github.com/robertknight/rten/pull/32)) and vectorized kernels
   for other functions ([#31](https://github.com/robertknight/rten/pull/31)).

## [0.2.0] - 2024-01-03

 - Improved inference performance on ARM [#30](https://github.com/robertknight/rten/pull/30)

## [0.1.1] - 2024-01-01

 - Fix softmax operator on non-x64 / wasm32 platforms (59f4815)

## [0.1.0] - 2023-12-31

Initial release.
