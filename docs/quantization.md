# Quantization support

RTen supports ONNX models that have been quantized to int8 format. Quantization
reduces the file size of the model and can improve inference performance,
depending on the model size and hardware.

The RTen repository contains tools ([ort-quantize.py][ort-quantize]) to assist
with creating quantized models.

This guide explains:

 - How quantization works and how it affects performance
 - How to quantize ONNX models for use with RTen
 - Nuances of quantization support in RTen

To get started with creating and running quantized models using recommended
settings, jump to the "[Quantizing a model](#quantizing-a-model)" section below.

## How quantization works

Quantization means converting a model's weights, and optionally internal
computations, to a smaller data type in order to reduce model size and improve
performance. To do this, float values are mapped to int8 values and associated
scale and zero point such that:

```
float_value = (int8_value - zero_point) * scale
```

Where the zero point is an int8 value and the scale is a float. The zero point
and scale are shared across many tensor elements. There can be one zero point
and scale per row or column, per channel (for an image) or one for the whole
tensor. The zero point can be chosen as zero (_symmetric_ quantization) or
allowed to be non-zero (_asymmetric_).

Quantization can be applied to the weights only, or both the weights and results
of internal computations (the _activations_). 

When activations are quantized, the zero point and scale can be computed during
inference, known as _dynamic_ quantization, or offline beforehand, known as
_static_ quantization. When using static quantization, example inputs must
be provided as calibration data. Dynamic quantization is simpler to use and
more accurate, but adds some overhead during inference.

## How quantization affects performance

Quantization can improve performance in two ways:

- By reducing the memory bandwidth required to move weights and activations
  from memory into CPU cores for computation.
- By enabling the use of specific hardware instructions for accelerating int8
  matrix products and matrix-vector products.

There are however additional computation steps involved in quantized model
inference, which reduces the gain compared to the theoretical maximum. These
include converting tensors between int8 and float types, and calculating the
quantization parameters to use, if using dynamic quantization.

The impact of each of these depends on whether performance is bottlenecked
primarily by compute or memory bandwidth. For small models running on
hardware without dot product instructions, the benefit over f32 may be minimal.
For LLM models with billions of parameters, memory bandwidth is the dominant
factor affecting performance and quantization has a huge impact.

## Quantization support in RTen

### Supported CPU instructions

All CPUs can run int8-quantized models, however performance is significantly
improved if int8 dot product instructions are available.

#### Arm

The Arm [dot product
extensions](https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/exploring-the-arm-dot-product-instructions)
(aka. SDOT / UDOT) are available in all CPUs that support Arm v8.4 and some
earlier Arm v8.2+ CPUs.

#### x64

The dot product instructions on x86_64 are known as
[VNNI](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html)
or "DL Boost". VNNI comes in several flavors:
[AVX512-VNNI](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512_CPU_compatibility_table)
and
[AVX-VNNI](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI,_AVX-IFMA).

RTen currently supports the AVX512-VNNI variant. Future updates will add
AVX-VNNI support. Enabling VNNI requires compiling rten with nightly Rust and
the `avx512` feature enabled.

#### WebAssembly

The CPU's dot product instructions are exposed in WebAssembly if the
`relaxed-simd` target feature is enabled, but RTen does not currently take
advantage of this.

### Supported data types

ONNX quantization allows the use of different int8 signed-ness (`uint8` vs `int8`)
for weights and activations.

RTen is currently optimized for the case where activations are uint8 and weights
are int8. This is the default choice used by ONNX's dynamic quantization tool
and [ort-quantize.py][ort-quantize]. Other combinations are supported, but may
encounter slower performance due to RTen internally converting to the preferred
format.

### Supported quantization granularity

The granularity of quantization (ie.  which elements are quantized together and
share a scale and zero point) can be per-tensor, per-channel or per-block. RTen
currently supports per-tensor and per-channel quantization. There is no
performance advantage to using per-tensor quantization over per-channel.

### Supported quantization symmetry

RTen always uses assumes asymmetric quantization internally (ie. it assumes the
zero point may be non-zero). Hence there is no performance advantage to using
symmetric quantization, as there may be in some other runtimes.

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

### Supported quantization operators

ONNX has different operators that can be used to represent quantized operations.
RTen supports the "Tensor-oriented" operators (`QuantizeLinear`,
`DequantizeLinear`, `DynamicQuantizeLinear`) as well as integer matrix
multiplication and convolution (`MatMulInteger`, `ConvInteger`). It does not
currently support "QOperator" operators (`QLinearMatMul` and `QLinearConv`).

RTen only supports quantization operators that are part of the ONNX standard
(see [operator list](https://onnx.ai/onnx/operators/)). It does not support
custom operators which are specific to particular runtimes. You may encounter
this when trying to use a quantized model published on the internet to which
"model optimizations" have been applied, as these optimizations may include the
use of runtime-specific operators. If you encounter a problem trying to convert
an existing quantized ONNX model to RTen's format, you can try downloading the
un-quantized model and converting it using the [ort-quantize.py][ort-quantize]
script.

### Weights-only quantization

RTen is currently optimized for running models where both weights and
activations of matrix multiplication and convolutions are quantized using
`MatMulInteger` and `ConvInteger`. Models using weight-only quantization will
run, but with sub-optimal performance because RTen does not yet fuse together
dequantization (`DequantizeLinear`) with the computation operation (`MatMul`,
`Conv` etc.)

## Using an existing quantized model

You can use quantized models downloaded from the internet, provided they only
use standard ONNX operators (see section on supported operators above). If
multiple quantized variants are available, the preferred choice is the one that
uses dynamic quantization with uint8 activations, int8 weights and range
reduction enabled.

If it is unclear which settings were used when quantizing a model, you can
download the model and inspect it using
[Netron](https://github.com/lutzroeder/netron). Search for `MatMulInteger` or
`ConvInteger` operators in the model and see what data type the weights (second
input) have. You can check whether dynamic or static quantization is used by
searching for `DynamicQuantizeLinear` operators. This operator indicates dynamic
quantization. To understand whether range reduction was used, you can check
the range of values for weights and see if they lie in [0, 127] for uint8 or
[-64, 63] for int8.

## Quantizing a model

The easiest way to quantize an fp32 or fp16 model is to use the
[ort-quantize.py][ort-quantize] script in the rten repository. This
will produce an ONNX model which is compatible with both RTen and other ONNX
runtimes.

```
pip install onnx onnxruntime
python tools/ort-quantize.py model.onnx
```

This command will produce a `model.quant.onnx` file, which can then be converted
to `.rten` format for use with RTen using:

```
pip install rten-convert
rten-convert model.quant.onnx
```

This will produce `model.quant.rten`, which you can load and run in RTen
in the same way as an fp32 model.

This script uses quantization settings which prioritizes making the quantization
process simple and producing models which work across a range of hardware.
It may be possible to improve performance and accuracy slightly by using the
underlying ONNX quantization tools with custom settings. For example by using
static rather than dynamic quantization to reduce overhead during inference,
and disabling range reduction if you are targeting hardware not affected by
the x64 saturation hazard.

### Quantizing convolution operators

`ort-quantize.py` does not quantize `Conv` operators by default in order to
produce models which work in ONNX Runtime as well as RTen (see [ONNX Runtime
issue](https://github.com/microsoft/onnxruntime/issues/15888)).

If you only intend to use the model with `rten`, you can pass the
`--quantize-conv` flag which will enable the use of the quantized `ConvInteger`
operator. This can reduce the model size and improve inference performance
of convolution operations.

## Understanding the structure of quantized ONNX models

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

## Further reading

- [ONNX quantization
guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [oneDNN - Nuances of int8 computation on CPU and
GPU](https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html#processors-with-the-intel-avx2-or-intel-avx-512-support)

[ort-quantize]: ../tools/ort-quantize.py
