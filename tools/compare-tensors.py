from argparse import ArgumentParser
import sys

import numpy as np

from debug_utils import read_tensor

def main():
    parser = ArgumentParser(description="Compare two binary tensors")
    parser.add_argument('tensor_a', help="File containing first tensor")
    parser.add_argument('tensor_b', help="File containing second_tensor")
    args = parser.parse_args()

    x = read_tensor(args.tensor_a)
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
