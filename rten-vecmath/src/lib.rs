//! SIMD-vectorized implementations of various math functions that are commonly
//! used in neural networks.
//!
//! For each function in this library there are multiple variants, which
//! typically include:
//!
//!  - A version that operates on scalars
//!  - A version that reads values from an input slice and writes to the
//!    corresponding position in an equal-length output slice. These have a
//!    `vec_` prefix.
//!  - A version that reads values from a mutable input slice and writes
//!    the computed values back in-place. These have a `vec_` prefix and
//!    `_in_place` suffix.
//!
//! All variants use the same underlying implementation and should have the
//! same accuracy.
//!
//! See the source code for comments on accuracy.

mod erf;
mod exp;
mod shift_scale;
mod softmax;
mod sum;
mod tanh;

#[cfg(test)]
mod ulp;

#[cfg(test)]
mod testing;

pub use erf::{erf, gelu, vec_erf, vec_erf_in_place, vec_gelu, vec_gelu_in_place};
pub use exp::{
    exp, sigmoid, silu, vec_exp, vec_exp_in_place, vec_sigmoid, vec_sigmoid_in_place, vec_silu,
    vec_silu_in_place,
};
pub use shift_scale::vec_shift_scale_in_place;
pub use softmax::{vec_softmax, vec_softmax_in_place};
pub use sum::{vec_sum, vec_sum_square};
pub use tanh::{tanh, vec_tanh, vec_tanh_in_place};
