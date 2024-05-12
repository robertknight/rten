# rten-vecmath

This crate provides portable SIMD types that abstract over SIMD intrinsics on
different architectures. Unlike
[`std::simd`](https://doc.rust-lang.org/std/simd/index.html) this works on
stable Rust. There is also functionality to detect the available instructions
at runtime and dispatch to the optimal implementation.

This crate also contains SIMD-vectorized versions of math functions such as exp,
erf, tanh, softmax etc. that are performance-critical in machine-learning
models.
