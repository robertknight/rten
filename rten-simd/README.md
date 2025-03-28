# rten-simd

Portable SIMD library for stable Rust.

rten-simd is a library for defining operations that are accelerated using
[SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)
instruction sets such as AVX2, Arm Neon or WebAssembly SIMD. Operations are
defined once using safe, portable APIs, then _dispatched_ at runtime to
evaluate the operation using the best available SIMD instruction set (ISA)
on the current CPU.
                                                                            
The design is inspired by Google's
[Highway](https://github.com/google/highway) library for C++ and the
[pulp](https://docs.rs/pulp/latest/pulp/) crate.
