from argparse import ArgumentParser

import onnx
from onnxruntime.quantization import quantize_dynamic

parser = ArgumentParser()
parser.add_argument("input")
parser.add_argument("output", nargs="?")
args = parser.parse_args()

output = args.output or args.input.replace(".onnx", ".quant.onnx")

quantize_dynamic(args.input, output)
