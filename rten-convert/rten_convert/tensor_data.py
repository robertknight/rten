from typing import BinaryIO

import numpy as np

from rten_convert.util import round_up, write_padding


class TensorDataBuilder:
    offset: int
    """End offset of written data from start of tensor data."""

    tensors: list[np.ndarray]
    """List of tensors to write."""

    align: int
    """Alignment of each tensor's data, relative to the start of the tensor data."""

    def __init__(self):
        self.offset = 0
        self.tensors = []
        self.tensor_offsets = []
        self.tensor_lengths = []
        self.align = 64

    def add_tensor(self, array: np.ndarray, dtype=None) -> int:
        """
        Add a tensor to be written to the tensor data segment.

        Returns the offset that the data will be stored at, relative to the
        start of the tensor data segment.
        """
        self.tensors.append(array)

        match array.dtype:
            case np.float32 | np.int32:
                element_size = 4
            case np.int8 | np.uint8:
                element_size = 1
            case _:
                raise ValueError("Unsupported NumPy array type {}".format(array.dtype))

        prev_offset = self.offset
        padding = round_up(prev_offset, self.align) - prev_offset
        tensor_len = array.size * element_size

        self.offset += padding
        self.tensor_offsets.append(self.offset)
        self.tensor_lengths.append(tensor_len)
        self.offset += tensor_len

        return self.tensor_offsets[-1]

    def write(self, fp: BinaryIO):
        """
        Write out tensor data to a file.
        """

        offset = 0

        for i, tensor in enumerate(self.tensors):
            expected_offset = self.tensor_offsets[i]
            padding = round_up(offset, self.align) - offset

            assert (
                expected_offset == offset + padding
            ), f"actual offset {offset} of tensor {i} does not match expected offset {expected_offset}"

            write_padding(fp, padding)
            offset += padding

            tensor_data = tensor.tobytes()
            assert (
                len(tensor_data) == self.tensor_lengths[i]
            ), f"actual length {len(tensor_data)} of tensor {i} does not match expected length {self.tensor_lengths[i]}"

            fp.write(tensor_data)
            offset += len(tensor_data)
