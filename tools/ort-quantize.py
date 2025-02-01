from argparse import ArgumentParser

import onnx
from onnxruntime.quantization import quantize_dynamic

parser = ArgumentParser(description="Quantize ONNX models using dynamic quantization.")
parser.add_argument("input")
parser.add_argument("output", nargs="?")
parser.add_argument(
    "--quantize-conv",
    action="store_true",
    help="""
Enable quantization of `Conv` operators.

This is disabled by default to avoid producing models that don't work
in ONNX Runtime. See https://github.com/microsoft/onnxruntime/issues/15888.
""",
)
args = parser.parse_args()

output = args.output or args.input.replace(".onnx", ".quant.onnx")

# Quantized operation types we support.
#
# See https://github.com/microsoft/onnxruntime/blob/1fc9c4823d7c2e8f0d07a09315a0755dd7c58ef8/onnxruntime/python/tools/quantization/quantize.py#L828 for the default list that ORT uses.
#
# See https://github.com/microsoft/onnxruntime/blob/1fc9c4823d7c2e8f0d07a09315a0755dd7c58ef8/onnxruntime/python/tools/quantization/registry.py#L66 for registries of different ops that
# will be quantized depending on the quantization type.
op_types_to_quantize = [
    # Supported ops from `CommonOpsRegistry`. These support int8 types directly.
    #
    # There are other operators which support int8 types that we could list
    # here but don't because `quantize_dynamic` doesn't attempt to quantize them.
    "Gather",
    "Transpose",
    # Supported ops from `IntegerOpsRegistry`. These get replaced during quantization.
    "MatMul",  # Replaced by MatMulInteger
]

if args.quantize_conv:
    op_types_to_quantize.append("Conv")  # Replaced by ConvInteger

quantize_dynamic(
    args.input,
    output,
    op_types_to_quantize=op_types_to_quantize,
    # Avoid a saturation issue on x86-64 systems that don't support VNNI by
    # reducing the range of quantized values from 8 to 7 bits.
    #
    # Specifically the VPMADDUBSW instruction used in int8 matmul operations
    # can saturate when adding pairs of signed i16 values.
    #
    # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-to-use-reduce-range-and-per-channel-quantization.
    reduce_range=True,
    # Use per-channel rather than per-tensor quantization.
    #
    # The effect of this is that separate zero points and scales are used per
    # row or column of an input matrix in quantized matmuls (`MatMulInteger`
    # ops).
    #
    # Turning this on increases compute slightly, but allows tolerating a
    # wider range of weight values in a tensor. Since transformer models are
    # prone to having outlier weights, this seems like a good idea. Also
    # RTen internally broadcasts scalar zero points to vectors anyway.
    per_channel=True,
    extra_options={
        # Enable quantization of models with control flow operators. This
        # includes Hugging Face "merged" transformer decoder models, which is
        # what various RTen examples use.
        "EnableSubgraph": True,
    },
)
