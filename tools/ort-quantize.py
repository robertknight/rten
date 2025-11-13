from argparse import ArgumentParser
import os

from onnxruntime.quantization import quantize_dynamic, quantize
import onnxruntime.quantization.matmul_nbits_quantizer as nbits
from onnxruntime.quantization.quant_utils import QuantFormat

parser = ArgumentParser(description="Quantize ONNX models.")
mode_parsers = parser.add_subparsers(
    title="mode",
    required=True,
    dest="mode",
    help="""
Quantizer to use. Use `{mode} --help` to see mode-specific options.
""",
)

dynamic_parser = mode_parsers.add_parser(
    "dynamic",
    help="""
int8 per-channel quantization using DynamicQuantizeLinear and MatMulInteger operators.
Widely supported by ONNX runtimes.
""",
)
dynamic_parser.add_argument(
    "--quantize-conv",
    action="store_true",
    help="""
Enable quantization of `Conv` operators.

Disabled by default for ONNX Runtime compatibility. See https://github.com/microsoft/onnxruntime/issues/15888.
""",
)
dynamic_parser.add_argument("input", help="Path to un-quantized input model")
dynamic_parser.add_argument("output", nargs="?", help="Path to quantized output model")

nbits_parser = mode_parsers.add_parser(
    "nbits",
    help="""
Newer blocked quantization method using MatMulNBits (or DequantizeLinear +
MatMul).

This can preserve accuracy better and is often used for LLMs. This is less-widely
supported than the "dynamic" mode.
""",
)
nbits_parser.add_argument("input", help="Path to un-quantized input model")
nbits_parser.add_argument("output", nargs="?", help="Path to quantized output model")
nbits_parser.add_argument(
    "--qdq",
    action="store_true",
    help="""
Represent quantization using QuantizeLinear + DequantizeLinear ("QDQ") rather
than operators with built-in quantization ("QOperator").

This enables representing quantization with only standard operators, but may
affect model load time and accuracy. See https://github.com/robertknight/rten/issues/578.
""",
)

# The combination of i8 compute and block size of 32 was chosen to match the
# settings used for pre-created ONNX models uploaded to Hugging Face such as
# https://huggingface.co/onnx-community/Llama-3.2-3B-Instruct-ONNX.
nbits_parser.add_argument(
    "--accuracy",
    choices=["f32", "i8"],
    default="i8",
    help="""
Minimum accuracy for internal compute of matrix multiplications.

This affects whether the activations are dynamically quantized in MatMulNBits
operators. Quantization can significantly improve performance, but may impact accuracy.

See MatMulNBits in https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md.
""",
)
nbits_parser.add_argument(
    "--block-size",
    type=int,
    default=32,
    help="""
Block size for quantization along the K dimension.

Values must be a power of 2 that is >= 16. Using smaller values can improve
accuracy at the cost of increasing model size and slightly reducing inference
speed.
""",
)
args = parser.parse_args()

output = args.output or args.input.replace(".onnx", ".quant.onnx")

# If an external data file already exists, the quantizer will append to
# it instead of overwriting. This is not the behavior we want.
data_file_path = f"{output}.data"
try:
    os.remove(data_file_path)
except FileNotFoundError:
    pass


def do_dynamic_quantize(args):
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
        if args.mode != "dynamic":
            raise Exception("--quantize-conv not supported for this quantization mode")

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


def do_nbits_quantize(args):
    # Always use symmetric quantization because RTen's MatMulNBits
    # implementation doesn't support zero points yet.
    symmetric = True
    block_size = args.block_size

    levels = {
        "f32": 0,
        "i8": 4,
    }
    accuracy_level = levels[args.accuracy]

    config_kwargs = {}
    if args.qdq:
        config_kwargs["quant_format"] = QuantFormat.QDQ

    config = nbits.DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=symmetric,
        accuracy_level=accuracy_level,
        **config_kwargs,
    )
    quantize(args.input, output, config)


match args.mode:
    case "dynamic":
        do_dynamic_quantize(args)
    case "nbits":
        do_nbits_quantize(args)
