from argparse import ArgumentParser
from collections import Counter

import onnx
from onnx import TensorProto, TypeProto


def iter_graphs(graph):
    """Yield `graph` and all of the subgraphs nested inside it."""
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                yield from iter_graphs(attr.g)
            for subgraph in attr.graphs:
                yield from iter_graphs(subgraph)


def dtype_name(elem_type: int) -> str:
    """Return the name of a `TensorProto.DataType` value (eg. "float16")."""
    return TensorProto.DataType.Name(elem_type).lower()


def type_name(type_: TypeProto) -> str:
    """Return a description of the type of a model input or output."""
    match type_.WhichOneof("value"):
        case "tensor_type":
            return dtype_name(type_.tensor_type.elem_type)
        case "sparse_tensor_type":
            return f"sparse[{dtype_name(type_.sparse_tensor_type.elem_type)}]"
        case "sequence_type":
            return f"sequence[{type_name(type_.sequence_type.elem_type)}]"
        case "optional_type":
            return f"optional[{type_name(type_.optional_type.elem_type)}]"
        case "map_type":
            key = dtype_name(type_.map_type.key_type)
            return f"map[{key}, {type_name(type_.map_type.value_type)}]"
        case _:
            return "unknown"


def list_operators(model, args):
    """List the unique operators used by a model."""

    op_counts = Counter(
        (node.domain, node.op_type)
        for graph in iter_graphs(model.graph)
        for node in graph.node
    )

    for (domain, op_type), count in sorted(op_counts.items()):
        line = f"{domain}.{op_type}" if domain else op_type
        if args.count:
            line += f" {count}"
        print(line)


def list_dtypes(model, args):
    """List the data types of a model's inputs, outputs and initializers."""

    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    inputs = [inp for inp in graph.input]

    sections = [
        ("Inputs", [(inp.name, type_name(inp.type)) for inp in inputs]),
        ("Outputs", [(out.name, type_name(out.type)) for out in graph.output]),
        (
            "Initializers",
            [(init.name, dtype_name(init.data_type)) for init in graph.initializer],
        ),
    ]

    for label, values in sections:
        print(f"{label}:")
        if args.list:
            for name, dtype in values:
                print(f"  {name} {dtype}")
        else:
            # Summarize value types in each section.
            for dtype, count in sorted(Counter(dtype for _, dtype in values).items()):
                print(f"  {dtype} {count}")


def main():
    parser = ArgumentParser(description="Query information about an ONNX model.")
    subparsers = parser.add_subparsers(required=True)

    ops_parser = subparsers.add_parser("ops", help=list_operators.__doc__)
    ops_parser.add_argument("model", help="Input ONNX model")
    ops_parser.add_argument(
        "-c", "--count", action="store_true", help="Show the number of uses of each op"
    )
    ops_parser.set_defaults(command=list_operators)

    dtypes_parser = subparsers.add_parser("dtypes", help=list_dtypes.__doc__)
    dtypes_parser.add_argument("model", help="Input ONNX model")
    dtypes_parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List each value instead of counting values by type",
    )
    dtypes_parser.set_defaults(command=list_dtypes)

    args = parser.parse_args()

    # Load without external data, as only the model structure is needed.
    model = onnx.load(args.model, load_external_data=False)

    args.command(model, args)


if __name__ == "__main__":
    main()
