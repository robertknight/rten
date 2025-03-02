//! SIMD-vectorized implementations of operations used in neural networks.
//!
//! These implementations are used as kernels for operations in the
//! [rten](https://crates.io/crates/rten) crate.
//!
//! ## Constructing and dispatching operations
//!
//! The operations are implemented by structs which implement the SIMD operation
//! traits from [rten-simd](rten_simd). To apply an operation to data, first
//! construct the operation using the struct from this crate, then use a
//! dispatch method from the [`SimdOp`](rten_simd::dispatch::SimdOp) or
//! [`SimdUnaryOp`](rten_simd::dispatch::SimdUnaryOp) traits to execute the
//! operation using the preferred SIMD instruction set.
//!
//! ## In-place versus mutating operations
//!
//! Some operations support both updating data in place or reading input from
//! one slice and writing to another. For unary operations this is controlled by
//! dispatching with either [`map`](rten_simd::dispatch::SimdUnaryOp::map) or
//! [`map_mut`](rten_simd::dispatch::SimdUnaryOp::map_mut). For other operations
//! this is handled by exposing different constructors for the in-place and
//! mutating cases, such as [`Softmax::new`] and [`Softmax::new_mut`].
//!
//! ## Examples
//!
//! ### Applying a vectorized unary function
//!
//! ```
//! use std::mem::MaybeUninit;
//!
//! use rten_simd::safe::SimdUnaryOp;
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
//!
//! ### Applying softmax in place
//!
//! ```
//! use rten_simd::safe::SimdOp;
//! use rten_vecmath::Softmax;
//!
//! let mut data = [1., 0.5, 2.0];
//! Softmax::new_mut(&mut data).dispatch();
//! ```
//!
//! ### Applying softmax with separate input and output buffers
//!
//! This example reads data from an input and writes to an uninitialized output
//! buffer. The softmax operation returns the initialized slice.
//!
//! ```
//! use rten_simd::safe::SimdOp;
//! use rten_vecmath::Softmax;
//!
//! let data = [1., 0.5, 2.0];
//! let mut output = Vec::with_capacity(data.len());
//! let output_uninit = &mut output.spare_capacity_mut()[..data.len()];
//! let output_init = Softmax::new(&data, output_uninit).dispatch();
//!
//! // Safety: The softmax operation initialized all output elements.
//! let init_len = output_init.len();
//! unsafe { output.set_len(init_len) };
//! ```
//!
//! ### Computing the sum of a list of floats
//!
//! ```
//! use rten_simd::safe::SimdOp;
//! use rten_vecmath::Sum;
//!
//! let data = [1., 0.5, 2.0];
//! let sum = Sum::new(&data).dispatch();
//! ```

mod erf;
mod exp;
mod min_max;
mod normalize;
mod quantize;
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
pub use quantize::Quantize;
pub use tanh::Tanh;

// Normalization and reduction functions.
pub use min_max::MinMax;
pub use normalize::{Normalize, NormalizeOptions};
pub use softmax::Softmax;
pub use sum::{Sum, SumSquare, SumSquareSub};
