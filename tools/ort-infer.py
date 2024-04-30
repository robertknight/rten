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

    sess_opts.enable_profiling = enable_profiling

    if optimized_path:
        sess_opts.optimized_model_filepath = optimized_path

    providers = ["CPUExecutionProvider"]
    if execution_provider and execution_provider != "CPU":
        providers = [execution_provider + "ExecutionProvider"] + providers

    session = ort.InferenceSession(
        model_path, providers=providers, sess_options=sess_opts
    )

    def resolve_dim(dim: str) -> int:
        if dim in dynamic_dims:
            return dynamic_dims[dim]
        else:
            raise ValueError(
                f"Missing size for dynamic dim `{dim}`. Specify it with `-d {dim}=size`"
            )

    inputs = {}
    output_names = [node.name for node in session.get_outputs()]
    for node in session.get_inputs():
        type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
        }
        resolved_shape = [
            d if isinstance(d, int) else resolve_dim(d) for d in node.shape
        ]
        value = np.random.rand(*resolved_shape).astype(type_map[node.type])
        inputs[node.name] = value

    print(
        "Model inputs:",
        [(inp[0], inp[1].shape, inp[1].dtype.name) for inp in inputs.items()],
    )
    print("Outputs: ", output_names)

    # Run the model multiple times. The first few runs can act as a warmup.
    total_elapsed = 0.0
    for _ in range(0, n_evals):
        start = perf_counter()
        outputs = session.run(output_names, inputs)
        elapsed = perf_counter() - start
        total_elapsed += elapsed
        print("Model eval time: {}ms".format(elapsed * 1000))

    mean_elapsed = total_elapsed / n_evals
    print("Mean eval time: {}ms".format(mean_elapsed * 1000))


parser = ArgumentParser(description="Run an ONNX model using ONNX Runtime")
parser.add_argument("model", help="Path to .onnx model")
parser.add_argument(
    "-d",
    "--dim",
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
    "-s", "--save-optimized", type=str, help="Save optimized model to given path"
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


print(f"ONNX Runtime version {ort.get_version_string()}")
print("Available execution providers:", ort.get_available_providers())
print(f"Device:", ort.get_device())

dynamic_dims = {}
for dim_str in args.dim or []:
    try:
        name, size = dim_str.split("=")
        size = int(size)
    except ValueError:
        print(
            f'Dim argument "{dim_str}" does not have expected format. Expected "name=size".'
        )
        sys.exit(1)

    if size < 0:
        print(f"Dim size must be positive")
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
