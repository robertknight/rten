from argparse import ArgumentParser
import json
import sys

import numpy as np

from debug_utils import read_tensor

def read_json_tensor(path: str):
    """
    Load a tensor from a JSON file.

    The JSON data format is `{ "data": [elements...], "shape": [dims...] }`.
    This matches rten-tensor's serde serialization for the `Tensor` type.
    """
    with open(path) as tensor_fp:
        tensor_json = json.load(tensor_fp)
        return np.array(tensor_json["data"]).reshape(tensor_json["shape"])


def main():
    parser = ArgumentParser(description="Compare two binary tensors")
    parser.add_argument('tensor_a', help="File containing first tensor")
    parser.add_argument('tensor_b', help="File containing second_tensor")
    args = parser.parse_args()

    if args.tensor_a.endswith(".json"):
        x = read_json_tensor(args.tensor_a)
    else:
        x = read_tensor(args.tensor_a)

    if args.tensor_b.endswith(".json"):
        y = read_json_tensor(args.tensor_b)
    else:
        y = read_tensor(args.tensor_b)

    print(f"X shape {x.shape} Y shape {y.shape}")

    if x.shape != y.shape:
        print("Tensor shapes do not match")
        sys.exit(1)

    abs_diff = np.absolute(x - y)
    print(f"Average diff {abs_diff.sum() / x.size}")
    print(f"Max diff {abs_diff.max()}")
    print(f"Total diff {abs_diff.sum()}")


if __name__ == '__main__':
    main()
