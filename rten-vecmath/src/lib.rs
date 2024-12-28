//! SIMD-vectorized implementations of functions used in neural networks.
//!
//! The functions are implemented by structs which implement the SIMD operation
//! traits from [rten-simd](rten_simd).
//!
//! ## Examples
//!
//! ### Applying a vectorized unary function
//!
//! ```
//! use std::mem::MaybeUninit;
//!
//! use rten_simd::dispatch::SimdUnaryOp;
//! use rten_vecmath::Erf;
//!
//! // Apply the error function to each element of `data`.
//! let mut data = [1., 0.5, 2.0];
//! let erf_op = Erf {};
//! erf_op.map_mut(&mut data);
//!
//! // Apply the error function to each element of `src`, writing to `dest`.
//! let src = [1., 0.5, 2.0];
//! let mut dest = [MaybeUninit::uninit(); 3];
//! erf_op.map(&src, &mut dest);
//! ```

mod erf;
mod exp;
mod normalize;
mod softmax;
mod sum;
mod tanh;

#[cfg(test)]
mod ulp;

#[cfg(test)]
mod testing;

// Unary functions.
pub use erf::{Erf, Gelu};
pub use exp::{Exp, Sigmoid, Silu, Swish};
pub use tanh::Tanh;

// Normalization and reduction functions.
pub use normalize::{normalize, normalize_mut};
pub use softmax::{softmax, softmax_mut};
pub use sum::{sum, sum_square, sum_square_sub};
