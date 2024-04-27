from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import onnxruntime as ort


def run_model(model_path: str, n_evals: int = 10, enable_profiling=False):
    """
    Run the ONNX model in `model_path` with randomly generated inputs.
    """

    sess_opts = ort.SessionOptions()

    # Flags to customize optmizations. By default, all optimizations are enabled.
    #
    # See https://onnxruntime.ai/docs/api/python/api_summary.html

    # Parallelism flags
    # sess_opts.inter_op_num_threads = 1
    # sess_opts.intra_op_num_threads = 1

    # Graph optimization flags (eg. operator fusion)
    # sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Memory usage flags
    # sess_opts.enable_cpu_mem_arena = False
    # sess_opts.enable_mem_pattern = False
    # sess_opts.enable_mem_reuse = False

    sess_opts.enable_profiling = enable_profiling

    session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"], sess_options=sess_opts
    )

    inputs = {}
    output_names = [node.name for node in session.get_outputs()]
    for node in session.get_inputs():
        type_map = {
            "tensor(float)": np.float32,
        }
        value = np.random.rand(*node.shape).astype(type_map[node.type])
        inputs[node.name] = value

    print("Model inputs:", [(inp.shape, inp.dtype) for inp in inputs.values()])
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
    "-n", "--n_evals", type=int, help="Number of inference iterations", default=10
)
parser.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
args = parser.parse_args()


print(f"ONNX Runtime version {ort.get_version_string()}")
print(f"Execution providers:", ort.get_available_providers())
print(f"Device:", ort.get_device())

run_model(args.model, n_evals=args.n_evals, enable_profiling=args.profile)
