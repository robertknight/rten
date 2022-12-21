import math
import struct

import numpy as np

def read_tensor(path: str) -> np.ndarray:
    """
    Read a tensor from a file.

    The file is expected to contain the tensor data in the little-endian
    binary format:

    [rank:u32][dims:u32 * rank][data:f32 * product(dims)]
    """
    with open(path, 'rb') as file:
        ndim, = struct.unpack('<I', file.read(4))
        dims = struct.unpack('<' + 'I'*ndim, file.read(4 * ndim))
        nelts = math.prod(dims)
        data = struct.unpack('<' + 'f'*nelts, file.read(4 * nelts))
        return np.array(data).reshape(dims)


def write_tensor(tensor: np.ndarray, path: str):
    """
    Write a tensor to a file.

    This writes the tensor in the same binary format that `read_tensor` reads.
    """
    with open(path, 'wb') as file:
        file.write(struct.pack('<I', tensor.ndim))
        file.write(struct.pack('<' + 'I'*tensor.ndim, *tensor.shape))
        file.write(struct.pack('<' + 'f'*tensor.size, *list(tensor.flatten())))

