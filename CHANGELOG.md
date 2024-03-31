# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
