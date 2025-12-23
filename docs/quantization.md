# Quantization support

Quantization is a technique for reducing model size and improving inference
performance. For smaller models (<1GB) quantization can offer a significant
speedup (1.5-3x is typical). As models get larger the impact increases. For LLMs
with billions of parameters, quantization is essential for reasonable
performance on consumer hardware.

This document provides an overview of how quantization works, how
quantization is represented in ONNX models and the support RTen has for running
quantized models. It also explains how to quantize models for use with RTen
using recommended settings.

For help with creating or running quantized models, please [start a
discussion](https://github.com/robertknight/rten/discussions) on GitHub.

## How quantization works

Quantizing a model involves representing the model's weights, and optionally
activations, using a smaller data type. This reduces the amount of memory
bandwidth needed to transfer data into compute cores and also enables the use of
specific hardware features for accelerating operations on the smaller type.

A quantized tensor represents each float value from the original tensor using a
small integer (eg. 8 or 4-bit) and associated scale and zero point such that:

```
float_value = (int_value - zero_point) * scale
```

Where the zero point is an integer of the same bit-width as `int_value` and the
scale is a float. Each zero point and scale are shared across many tensor
elements, enabling the quantized tensor to be smaller than the original. The
_granularity_ of quantization can be:

 - One scale and zero-point per fixed-sized block of each row or column. This is
   known as _blockwise_ quantization.
 - One scale and zero-point per row or column. This is known as _per-channel_ or
   _per-row_ quantization.
 - One scale and zero-point per tensor. This is known as _per-tensor_
   quantization.

Using more fine-grained quantization improves accuracy but increases the model
size and amount of computation required. The general trend over time, as models
have gotten larger, has been to use smaller weights (eg. going from int8 to
int4) combed with more fine-grained scaling.

The zero point can be chosen as zero (_symmetric_ quantization) or
allowed to be non-zero (_asymmetric_).

Quantization can be applied to the weights only, or both the weights and results
of internal computations (the _activations_). 

When activations are quantized, the zero point and scale can be computed during
inference, known as _dynamic_ quantization, or offline beforehand, known as
_static_ quantization. When using static quantization, example inputs must
be provided as calibration data. Dynamic quantization is simpler to use and
more accurate, but adds some overhead during inference.

Due to a quirk of older x86 CPUs (those without the "VNNI" or "DL Boost"
feature), it may be necessary due to restrict the range of quantized 8-bit
integers so that only 7 bits are actually used, in order to avoid saturation
during computation. Quantizing to 7-bit weights is referred to as "range
reduced" quantization in some contexts.

## How quantization affects performance

Quantization can improve performance in two ways:

- By reducing the memory bandwidth required to move weights and activations
  from memory into compute cores.
- By enabling the use of specific hardware instructions for accelerating int8
  matrix products and matrix-vector products.

There are additional computation steps involved in quantized model inference,
which reduces the gain compared to the unquantized model. These include
converting tensors between integer and float types, and calculating the
quantization parameters to use, if using dynamic quantization.

The impact of each of these depends on whether performance is bottlenecked
primarily by compute or memory bandwidth. For small models running on
hardware without dot product instructions, the benefit over f32 may be minimal.
For LLM models with billions of parameters, memory bandwidth is the dominant
factor affecting performance and quantization has a huge impact.

## Quantization in ONNX

ONNX models have a number of operators that can be used to express quantization
and computation involving quantized values. You can inspect a model to
understand which it is using by using tools such as
[Netron](https://netron.app).

1. The `QuantizeLinear`, `DequantizeLinear` and `DynamicQuantizeLinear`
   operators convert between float values (f32, f16 etc.) and quantized
   values (int8, int4 etc). The `QuantizeLinear` and `DequantizeLinear`
   operators can do per-tensor, per-row or per-block (de-)quantization.
   `DynamicQuantizeLinear` is limited to per-tensor quantization.
2. The `MatMulInteger` and `ConvInteger` operators perform matrix multiplication
   with int8 inputs and produce int32 outputs. These operators use per-row/column
   or per-tensor weights.
3. The ONNX standard defines a handful of "Q"-prefixed operators
   (`QLinearMatMul`, `QLinearConv`). Which are effectively combinations of
   operators from (1) and (2).
4. Large language models (LLMs) often use the non-standard
   [`MatMulNBits`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits)
   operator to perform matrix multiplication with f32 activations, int8 or int4
   weights and produce f32 outputs. If the `accuracy_level` attribute is unset
   or 0, multiplication is internally done using f32 values. If set to 4,
   multiplication internally uses int8 values, which can be much more efficient.

## Quantization support in RTen

### Supported CPU instructions

All CPUs can run quantized models, however performance is significantly improved
if int8 dot product or matrix multiplication instructions are available.

#### Arm

The Arm [dot product
extensions](https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/exploring-the-arm-dot-product-instructions)
(aka. SDOT / UDOT) are available in all CPUs that support Arm v8.4 and some
earlier Arm v8.2+ CPUs.

More recent Arm CPUs support the
[i8mm](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction)
integer matrix multiplication extension. On some CPUs (eg. Neoverse), this
offers a significant improvement over SDOT. On others (eg. Apple M-series) it
performs the same as SDOT.

#### x64

The dot product instructions on x86_64 are known as
[VNNI](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html)
or "DL Boost". VNNI comes in several flavors:
[AVX512-VNNI](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512_CPU_compatibility_table)
and
[AVX-VNNI](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI,_AVX-IFMA).

RTen currently supports the AVX512-VNNI variant. Future updates will add
AVX-VNNI support.

#### WebAssembly

The CPU's dot product instructions are exposed in WebAssembly if the
`relaxed-simd` target feature is enabled, but RTen does not currently take
advantage of this.

### Supported data types

ONNX quantization allows the use of different int8 signed-ness (`uint8` vs `int8`)
for weights and activations.

For models using the `ConvInteger` and `MatMulInteger` operations, RTen is
currently optimized for the case where activations are uint8 and weights are
int8. This is the default choice used by ONNX's dynamic quantization tool and
[ort-quantize.py][ort-quantize]. Other combinations are supported, but may
encounter slower performance due to RTen internally converting to the preferred
format.

### Supported granularity

RTen supports per-tensor and per-channel quantization for the `QuantizeLinear`,
`DequantizeLinear` and `MatMulInteger` operations. There is no performance
advantage to using per-tensor quantization over per-channel.

### Supported symmetry

For `MatMulInteger` and `ConvInteger`, RTen always uses assumes asymmetric
quantization internally (ie. it assumes the zero point may be non-zero). Hence
there is no performance advantage to using symmetric quantization, as there may
be in some other runtimes.

For `MatMulNBits`, only symmetric quantization (no zero point) is currently
supported.

### Supported operators

RTen supports the quantization operators
`QuantizeLinear`, `DequantizeLinear` and `DynamicQuantizeLinear`, as well as
the integer compute operators `MatMulInteger` and `ConvInteger`. It does not
support the "Q" operators (`QLinearMatMul`, `QLinearConv`). Support for
`MatMulNBits` has been added quite recently, and is not yet fully optimized.

### Saturation hazard on x86_64 CPUs

On x64 systems which do not support VNNI / DL Boost, int8 matrix multiplication
uses a CPU instruction (`VPMADDUBSW`) which can encounter saturation when adding
pairs of int16 values. The workaround for this issue in ONNX is to ensure that
quantized weights are actually 7-bit integers ([-64, 63] for i8, [-128, 127] for
u8) by enabling the "range reduction" setting in the quantization tool. The
[ort-quantize.py][ort-quantize] script in this repository will do this
automatically.

See [Intel's
documentation](https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html)
for more information about the issue.

### Support for weights-only quantization

RTen can run models which use weights-only quantization via either `MatMul(A,
DequantizeLinear(B))` or `MatMulNBits` with `accuracy_level=0`. The combination
of `MatMul` + `DequantizeLinear` is not currently well optimized.

## Using an existing quantized model

You can use quantized models downloaded from the internet, provided they rely
on supported combinations of operators and data types.

If you encounter a quantized ONNX model which contains quantization features
that RTen does not yet support, you can try downloading the un-quantized model
and re-quantizing it using the [ort-quantize.py][ort-quantize] script, which
will use a supported combination of features.

On model hosting platforms such as Hugging Face, models using `MatMulInteger` or
`ConvInteger` with per-row quantization will often have a "int8", "uint8" or
"quantized" suffix. Models using `MatMulNBits` will often have a "q4" suffix.

## Quantizing a model

The easiest way to quantize an existing fp32 or fp16 model is to use the
[ort-quantize.py][ort-quantize] script in the rten repository. To quantize a
model, run:

```
pip install onnx onnxruntime
python ort-quantize.py [mode] model.onnx
```

This will produce a quantized model called `model.quant.onnx` and associated
external data files. There are two options for `mode`:

- `dynamic` produces a model using the `MatMulInteger`, `ConvInteger` and
  `DynamicQuantizeLinear` operators for int8 per-row quantization.

- `nbits` produces a model using the `MatMulNBits` operator with int4 blockwise
  quantization.

For smaller models the `dynamic` mode produces the fastest models but the use of
per-row quantization can limit accuracy. The `nbits` mode produces models which
are slower but the use of more fine-grained quantization improves accuracy.

As models get larger, the benefits of using int4 over int8 quantization
increases. Models quantized with `nbits` will be both more accurate _and_ faster
above a certain size. For LLMs with billions of parameters, you will always want
to use the `nbits` mode.

The `ort-quantize.py` script prioritizes making the quantization process simple
and producing models which work across a range of hardware. If you need more
control, you can use the underlying ONNX quantization tools and APIs directly.

### Quantizing convolution operators

`ort-quantize.py` does not quantize `Conv` operators by default in order to
produce models which work in ONNX Runtime as well as RTen (see [ONNX Runtime
issue](https://github.com/microsoft/onnxruntime/issues/15888)).

If you only intend to use the model with `rten`, you can pass the
`--quantize-conv` flag which will enable the use of the quantized `ConvInteger`
operator. This can reduce the model size and improve inference performance
of convolution operations.

## Further reading

- [ONNX quantization
guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [oneDNN - Nuances of int8 computation on CPU and
GPU](https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html#processors-with-the-intel-avx2-or-intel-avx-512-support)

[ort-quantize]: ../tools/ort-quantize.py

## Appendix: Understanding the structure of quantized ONNX models

This section explains how quantization is represented as operators in ONNX
models, which is useful to understand when inspecting a quantized model using a
tool such as eg. [Netron](https://netron.app).

Weights-only quantization is expressed in ONNX by using graphs with a structure
like:

```
MatMul(activations, DequantizeLinear(weights))
```

When weight and activations are both quantized using dynamic quantization, this
produces graphs such as:

```
X_quant, X_scale, X_zero_point = DynamicQuantizeLinear(X)
Y_quant = MatMulInteger(X_quant, W_quant, X_zero_point, W_zero_point)
Y = Cast(Y_quant, fp32)
Y_scaled = Mul(Y, X_scale)
```

To achieve optimal performance, the runtime may "fuse" several steps together.
RTen currently has very limited fusion for quantization operators and depending
on the model this will have varying cost. Better support for fusion of
quantization operators is planned for the future.
