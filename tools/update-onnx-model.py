from argparse import ArgumentParser

import onnx
from onnx import version_converter


def main():
    parser = ArgumentParser()
    parser.add_argument("input_model", help="Input ONNX model")
    parser.add_argument("output_model", help="Output ONNX model")
    parser.add_argument(
        "--opset_version", type=int, default=11, help="ONNX opset version to upgrade to"
    )
    args = parser.parse_args()

    original_model = onnx.load(args.input_model)

    # A full list of supported adapters can be found here:
    # https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
    converted_model = version_converter.convert_version(
        original_model, args.opset_version
    )

    onnx.save(converted_model, args.output_model)


if __name__ == "__main__":
    main()
