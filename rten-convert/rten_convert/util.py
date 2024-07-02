import math
from typing import BinaryIO


def round_up(value: int, base: int) -> int:
    """Round up `value` to the next multiple of `base`."""
    return base * math.ceil(value / base)


def write_padding(fp: BinaryIO, n: int, max_padding=1024):
    """
    Write `n` bytes of zero padding at the end of a file.

    :param max_padding:
        Maximum value for `n`. This is a sanity check to catch unexpectedly
        large padding sizes.
    """

    if n < 0 or n >= max_padding:
        raise ValueError(f"Padding size {n} is out of range")

    if n == 0:
        return
    fp.write(b"\x00" * n)
