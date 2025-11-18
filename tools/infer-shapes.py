from argparse import ArgumentParser

import onnx

parser = ArgumentParser(
    description="""
Apply shape inference to a model.

This adds metadata to the model about the shapes of values, potentially
enabling additional optimizations when running the model.
"""
)
parser.add_argument("input")
parser.add_argument("output", nargs="?", help="Output file path")
parser.add_argument(
    "-i", "--in-place", action="store_true", help="Modify the input model"
)
args = parser.parse_args()

if args.in_place:
    if args.output:
        raise Exception("Cannot specify both `--in-place` and output path")
    output = args.input
else:
    output = args.output or args.input.replace(".onnx", ".shaped.onnx")

# Use `onnx.shape_inference.infer_shapes_path` to support models that
# are more than 2GB in size. This function will generate a new ".onnx"
# file that shares the same external data (if any) as the input file.
#
# See https://onnx.ai/onnx/api/shape_inference.html#infer-shapes-path and
# https://github.com/robertknight/rten/issues/851.
onnx.shape_inference.infer_shapes_path(args.input, output, data_prop=True)
