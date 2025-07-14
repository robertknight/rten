from argparse import ArgumentParser

import onnx

parser = ArgumentParser(description="Apply shape inference to a model")
parser.add_argument("input")
parser.add_argument("output", nargs="?")
args = parser.parse_args()

output = args.output or args.input.replace(".onnx", ".shaped.onnx")

model = onnx.load(args.input)

# See https://onnx.ai/onnx/api/shape_inference.html
updated_model = onnx.shape_inference.infer_shapes(model, data_prop=True)

onnx.save(updated_model, output)
