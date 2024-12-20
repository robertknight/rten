# rten-vecmath

This crate contains SIMD-vectorized kernels ("vectorized math") for various
operations used in machine learning models. This includes:

 - Math functions such as exp, erf, tanh
 - Activation function such as gelu
 - Normalization functions such as softmax and mean-variance normalization
 - Reduction functions such as sums and sum-of-square

SIMD operations are implemented using portable SIMD types from the rten-simd
crate.
