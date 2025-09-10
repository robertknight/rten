import math
from typing import Any, BinaryIO
import sys


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


EMITTED_WARNINGS: set[str] = set()


def warn_once(format_str: str, *args: Any):
    """
    Emit a warning if not already emitted.

    This is used to reduce output noise if the same problem arises many times
    when converting a model.

    :param format_str: Message format string
    :param args: Arguments passed to `format_str.format`
    """
    if format_str in EMITTED_WARNINGS:
        return
    EMITTED_WARNINGS.add(format_str)
    msg = format_str.format(*args)
    print(f"WARNING: {msg}", file=sys.stderr)
