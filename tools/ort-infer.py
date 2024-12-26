from argparse import ArgumentParser
import sys
from time import perf_counter

import numpy as np
import onnxruntime as ort

OPT_LEVELS = {
    "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def run_model(
    model_path: str,
    *,
    dynamic_dims: dict[str, int] | None = None,
    enable_profiling=False,
    execution_provider: str | None = None,
    inter_op_threads: int | None = None,
    intra_op_threads: int | None = None,
    n_evals: int = 10,
    opt_level: str | None = None,
    optimized_path: str | None = None,
):
    """
    Run the ONNX model in `model_path` with randomly generated inputs.

    :param n_evals: Number of times to run inference
    :param dynamic_dims: Dict of dimension name to size for dimensions with dynamic size
    """

    if dynamic_dims is None:
        dynamic_dims = {}

    sess_opts = ort.SessionOptions()

    # Flags to customize optmizations. By default, all optimizations are enabled.
    #
    # See https://onnxruntime.ai/docs/api/python/api_summary.html

    # Parallelism flags.
    # See also https://onnxruntime.ai/docs/performance/tune-performance/threading.html.
    if inter_op_threads:
        sess_opts.inter_op_num_threads = inter_op_threads
        sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    if intra_op_threads:
        sess_opts.intra_op_num_threads = intra_op_threads

    # Graph optimization flags (eg. operator fusion).
    # See also https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html.
    if opt_level:
        sess_opts.graph_optimization_level = OPT_LEVELS[opt_level]

    # Memory usage flags
    # sess_opts.enable_cpu_mem_arena = False
    # sess_opts.enable_mem_pattern = False
    # sess_opts.enable_mem_reuse = False

    # Additional optimization controls. See
    # https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
    #
    # sess_opts.add_session_config_entry("session.disable_prepacking", "1")

    sess_opts.enable_profiling = enable_profiling

    if optimized_path:
        sess_opts.optimized_model_filepath = optimized_path

    providers = ["CPUExecutionProvider"]
    if execution_provider and execution_provider != "CPU":
        providers = [execution_provider + "ExecutionProvider"] + providers

    session = ort.InferenceSession(
        model_path, providers=providers, sess_options=sess_opts
    )

    # Print summary of model inputs and outputs and generate random data for
    # model inputs.
    inputs = {}
    dynamic_dims_without_sizes = set()

    print("Inputs:")
    for node in session.get_inputs():
        type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }

        resolved_shape = []
        for d in node.shape:
            if isinstance(d, int):
                resolved_shape.append(d)
            elif d in dynamic_dims:
                resolved_shape.append(dynamic_dims[d])
            else:
                dynamic_dims_without_sizes.add(d)
                resolved_shape.append(1)
            value = np.random.rand(*resolved_shape).astype(type_map[node.type])
            inputs[node.name] = value

        print(f"  {node.name}: {node.type} {node.shape}")

    if dynamic_dims_without_sizes:
        print()
        for dim in dynamic_dims_without_sizes:
            print(
                f'  Size not specified for dynamic dimension "{dim}". Defaulting to 1.'
            )
    print()

    print("Outputs:")
    for node in session.get_outputs():
        print(f"  {node.name}: {node.type} {node.shape}")
    print()

    # Run model and collect timing statistics
    output_names = [node.name for node in session.get_outputs()]
    durations = []
    for _ in range(0, n_evals):
        start = perf_counter()
        session.run(output_names, inputs)
        elapsed = perf_counter() - start
        durations.append(elapsed * 1000.0)
        print("Model eval time: {:.2f}ms".format(elapsed * 1000))

    # Print duration statistics
    mean = sum(durations) / n_evals
    variance = sum((dur - mean) ** 2 for dur in durations) / n_evals
    std_dev = variance**0.5
    min_dur = min(durations)
    max_dur = max(durations)
    print()
    print(
        f"Duration stats: mean {mean:.2f}ms, min {min_dur:.2f}ms, max {max_dur:.2f}ms, std dev {std_dev:.2f}ms"
    )


parser = ArgumentParser(description="Run an ONNX model using ONNX Runtime")
parser.add_argument("model", help="Path to .onnx model")
parser.add_argument(
    "-s",
    "--size",
    type=str,
    action="append",
    help="Specify size for dynamic input dim as `dim_name=size`",
)
parser.add_argument(
    "-e", "--exec-provider", type=str, help='Execution provider (eg. "CPU", "CoreML")'
)
parser.add_argument(
    "-n", "--n_evals", type=int, help="Number of inference iterations", default=10
)
parser.add_argument(
    "-o",
    "--opt-level",
    choices=["none", "basic", "extended", "all"],
    help="Graph optimization level",
)
parser.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
parser.add_argument(
    "--save-optimized", type=str, help="Save optimized model to given path"
)
parser.add_argument(
    "-t", "--intra-threads", type=int, help="Number of threads to use within ops"
)
parser.add_argument(
    "-T",
    "--inter-threads",
    type=int,
    help="Number of threads to use to run ops in parallel",
)
args = parser.parse_args()

provider_names = ", ".join(
    p.replace("ExecutionProvider", "") for p in ort.get_available_providers()
)

print(f"ONNX Runtime version {ort.get_version_string()}")
print("Available execution providers:", provider_names)
print("Device:", ort.get_device())
print()

# Map of dimension name to size for dynamic dims. The parsing here should
# roughly match rten-cli.
dynamic_dims: dict[str, int] = {}
for size_spec in args.size or []:
    try:
        name, size = size_spec.split("=")
        size = int(size)
        if size < 0:
            print(f'Dim size "{name}" must be positive')
            sys.exit(1)
    except ValueError:
        print(
            f'Dimension size argument "{size_spec}" does not have expected format. Expected "name=size".'
        )
        sys.exit(1)

    dynamic_dims[name] = size

run_model(
    args.model,
    dynamic_dims=dynamic_dims,
    enable_profiling=args.profile,
    execution_provider=args.exec_provider,
    inter_op_threads=args.inter_threads,
    intra_op_threads=args.intra_threads,
    n_evals=args.n_evals,
    opt_level=args.opt_level,
    optimized_path=args.save_optimized,
)
