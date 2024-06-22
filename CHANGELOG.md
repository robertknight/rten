# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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

#### Bug fixes

- Prevent `Model::partial_run` from propagating values through randomized
  operators (https://github.com/robertknight/rten/pull/240).

#### Performance improvements

This release adds a new graph optimization step as part of loading models. This
performs fusions and other optimizations to speed up inference. These
optimizations are enabled by default, but can be disabled via options in
`ModelOptions`.

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
