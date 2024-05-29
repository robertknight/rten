from argparse import ArgumentParser
import json
from typing import TypedDict, cast


class TypeShape(TypedDict):
    """
    Type and shape for a tensor.

    There will be one key specifying the type, and it's value is the shape.
    """

    float: list[int]
    int64: list[int]


class FenceNodeArgs(TypedDict):
    """Node args for a `*_fence_{before, after}` profile node."""

    op_name: str


class KernelTimeNodeArgs(TypedDict):
    """Node args for a `*_kernel_time` profile node."""

    op_name: str
    """
    ONNX operator name, eg. "MatMul".
    """

    input_type_shape: list[TypeShape]

    # Other fields:
    # - provider (execution provider)
    # - thread_scheduling_stats
    # - output_size (bytes?)
    # - parameter_size (bytes?)


class OtherNodeArgs(TypedDict):
    """
    Node args for other types of profile node.

    This is for profile nodes that don't have any arguments or aren't used by
    this script.
    """

    pass


class ProfNode(TypedDict):
    """Entry in an ORT JSON profile."""

    name: str
    """
    Profile node name.

    For nodes that relate to execution of an operator, this is a concatenation
    of the ONNX node name and sub-task, eg. "/module.0/layer.0/MatMul_kernel_time".
    """

    args: FenceNodeArgs | KernelTimeNodeArgs | OtherNodeArgs
    """Detailed information for this node."""

    dur: int
    """Duration in microseconds"""


def summarize_profile(profile: list[ProfNode]):
    # Extract nodes containing details for execution of an operator.
    op_execution_nodes = [
        node for node in profile if node["name"].endswith("_kernel_time")
    ]
    nodes_by_op = {}
    for node in op_execution_nodes:
        args = cast(KernelTimeNodeArgs, node["args"])
        op_name = args["op_name"]
        if op_name not in nodes_by_op:
            nodes_by_op[op_name] = []
        nodes_by_op[op_name].append(node)

    # Calculate duration metrics for each operator type.
    op_time_stats = []
    for op_name, nodes in nodes_by_op.items():
        total_dur_us = sum(n["dur"] for n in nodes)
        mean_dur_us = float(total_dur_us) / float(len(nodes))

        op_time_stats.append(
            {
                "op_name": op_name,
                "mean": mean_dur_us,
                "total": total_dur_us,
            }
        )

    op_time_stats.sort(key=lambda entry: entry["total"], reverse=True)
    sum_total = sum(n["total"] for n in op_time_stats)

    # Print a summary by operator type.
    for stats in op_time_stats:
        op_name = stats["op_name"]
        total_ms = stats["total"] / 1000.0
        mean_ms = stats["mean"] / 1000.0
        percent = float(stats["total"]) * 100.0 / float(sum_total)
        print(f"{op_name: <16}\t{total_ms: <10.3f}{mean_ms:<8.3f}{percent:.2f}%")


parser = ArgumentParser(description="Analyze profiler output from ONNX Runtime")
parser.add_argument("profile", help="Path to JSON profile")
args = parser.parse_args()

with open(args.profile) as profile_fp:
    profile_data = json.load(profile_fp)
    summarize_profile(profile_data)
