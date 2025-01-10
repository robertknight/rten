from argparse import ArgumentParser

import onnx
from onnxruntime.quantization import quantize_dynamic

parser = ArgumentParser()
parser.add_argument("input")
parser.add_argument("output", nargs="?")
args = parser.parse_args()

output = args.output or args.input.replace(".onnx", ".quant.onnx")

quantize_dynamic(
    args.input,
    output,
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
