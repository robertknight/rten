#!/usr/bin/env python3

"""
Group the output of `cargo llvm-lines` by function, ignoring generic parameters.

`cargo llvm-lines` reports each monomorphization of a generic function
separately (eg. `gather_nd::<i32>` and `gather_nd::<i8>`). This script strips
generic argument lists from function names and sums the line and copy counts
of all instantiations which map to the same name.

Usage:

    cargo llvm-lines --release -p rten > llvm-lines.txt
    python tools/group-llvm-lines.py llvm-lines.txt
"""

from argparse import ArgumentParser
from collections import defaultdict
import re
import signal
import sys

# Matches a data row, eg:
#   "  1729 (0.2%,  1.1%)      1 (0.0%,  0.0%)  rten::ops::conv::conv_impl::<i8, u8, i32>"
ROW_RE = re.compile(r"^\s*(\d+)\s+\([^)]*\)\s+(\d+)\s+\([^)]*\)\s+(.*)$")

# Matches crate disambiguator hashes, eg. the `[c3832d8ff6c99cb]` in
# `rten[c3832d8ff6c99cb]::ops`. The lookbehind avoids matching slice types such
# as `[f32]`, which are never preceded by an identifier character.
CRATE_HASH_RE = re.compile(r"(?<=[A-Za-z0-9_])\[[0-9a-f]{6,}\]")


def is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def skip_generic_args(name: str, start: int) -> int:
    """
    Return the index after the `>` which closes the `<` at `name[start]`.

    For example:

        skip_generic_args("foo::<Vec<fn(u8) -> i32>>::bar", 5)

    Returns 25, the index of `::bar`.
    """
    depth = 0
    i = start
    while i < len(name):
        ch = name[i]
        if ch == "-" and name.startswith("->", i):
            # Return type arrow in `fn(A) -> B` or `Fn(A) -> B`.
            i += 2
            continue
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(name)


def strip_generics(name: str) -> str:
    """
    Remove generic argument lists from a demangled Rust symbol name.

    For example:

        <rten_tensor::TensorBase<Vec<f32>, DynLayout> as rten_tensor::AsView>::map_in::<i32, rten::ops::relu::{closure#0}>

    Becomes:

        <rten_tensor::TensorBase as rten_tensor::AsView>::map_in

    Generic argument lists (`Foo<T>`, `foo::<T>`) are removed, but qualified
    paths such as `<Foo as Trait>::method` are kept, with generics stripped
    from the types inside them.
    """
    out = []
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == "<":
            turbofish = name.endswith("::", 0, i)
            after_ident = i > 0 and is_ident_char(name[i - 1])
            if turbofish or after_ident:
                if turbofish:
                    # Drop the `::` preceding `<`.
                    out.pop()
                    out.pop()
                i = skip_generic_args(name, i)
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def normalize_name(name: str) -> str:
    return strip_generics(CRATE_HASH_RE.sub("", name))


def main():
    parser = ArgumentParser(
        description="Group `cargo llvm-lines` output across generic instantiations."
    )
    parser.add_argument(
        "file", nargs="?", help="Output of `cargo llvm-lines`. Defaults to stdin."
    )
    args = parser.parse_args()

    # Exit quietly if the output pipe is closed early (eg. when piping to `head`).
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if args.file:
        with open(args.file) as fp:
            text = fp.read()
    else:
        text = sys.stdin.read()

    lines: dict[str, int] = defaultdict(int)
    copies: dict[str, int] = defaultdict(int)
    variants: dict[str, set[str]] = defaultdict(set)
    total_lines = 0
    total_copies = 0

    for row in text.splitlines():
        match = ROW_RE.match(row)
        if not match:
            continue
        n_lines, n_copies, name = match.groups()
        if name == "(TOTAL)":
            continue
        key = normalize_name(name)
        lines[key] += int(n_lines)
        copies[key] += int(n_copies)
        variants[key].add(name)
        total_lines += int(n_lines)
        total_copies += int(n_copies)

    names = sorted(lines, key=lambda k: (-lines[k], k))

    def pct(value: int, total: int) -> float:
        return 100 * value / total if total else 0.0

    print(f"  {'Lines':<22} {'Copies':<22} {'Insts':>6}  Function name")
    print(f"  {'-----':<22} {'------':<22} {'-----':>6}  -------------")
    print(f"  {total_lines:<22} {total_copies:<22} {'':>6}  (TOTAL)")

    cum_lines = 0
    cum_copies = 0
    for name in names:
        cum_lines += lines[name]
        cum_copies += copies[name]
        lines_col = (
            f"{lines[name]:>8} ({pct(lines[name], total_lines):.1f}%,"
            f"{pct(cum_lines, total_lines):5.1f}%)"
        )
        copies_col = (
            f"{copies[name]:>6} ({pct(copies[name], total_copies):.1f}%,"
            f"{pct(cum_copies, total_copies):5.1f}%)"
        )
        print(f"{lines_col:<24} {copies_col:<22} {len(variants[name]):>6}  {name}")


if __name__ == "__main__":
    main()
