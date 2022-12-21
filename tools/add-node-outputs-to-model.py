from argparse import ArgumentParser

import onnx


# This script adds all of the operator output value nodes to the model's list
# of outputs.
#
# This enables examining intermediate outputs from the model when run using
# ONNX Runtime. See https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-979901463.
#
# For Wasnn this step is not necessary since any value node in the graph can
# be specfied as the output for a model execution.
def main():
    parser = ArgumentParser(
        description="Add intermediate outputs in an ONNX graph to the model's outputs"
    )
    parser.add_argument("onnx_model")
    parser.add_argument("out_model")
    args = parser.parse_args()

    model = onnx.load(args.onnx_model)
    initial_outputs = [val.name for val in model.graph.output]

    for node in model.graph.node:
        for output in node.output:
            if output not in initial_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    onnx.save(model, args.out_model)


if __name__ == "__main__":
    main()
