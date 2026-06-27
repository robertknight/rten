#!/usr/bin/env python3
"""Generate NPY fixtures for row-major and column-major ordering tests."""

from pathlib import Path

import numpy as np


FIXTURE_DIR = Path(__file__).resolve().parent


def main() -> None:
    array = np.arange(24, dtype=np.int32).reshape(2, 3, 4)

    np.save(FIXTURE_DIR / "order_c_i32.npy", np.ascontiguousarray(array))
    np.save(FIXTURE_DIR / "order_fortran_i32.npy", np.asfortranarray(array))


if __name__ == "__main__":
    main()
