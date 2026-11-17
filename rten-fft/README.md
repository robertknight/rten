# rten-fft

This crate contains a SIMD-vectorized Fast Fourier Transform for `f32` complex
signals, used to implement the FFT-related operators (DFT, STFT) in the rten
crate.

Complex signals are represented in a decomposed form, with separate slices for
the real and imaginary parts. Batches of equal-length signals can be transformed
together, one signal per SIMD lane.

SIMD operations are implemented using portable SIMD types from the rten-simd
crate.
