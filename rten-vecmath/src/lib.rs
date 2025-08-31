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
//! dispatch method from the [`SimdOp`](rten_simd::SimdOp) or
//! [`SimdUnaryOp`](rten_simd::SimdUnaryOp) traits to execute
//! the operation.
//!
//! ## In-place and non in-place operations
//!
//! Some operations support both updating data in place or reading input from
//! one slice and writing to another. For unary operations this is controlled by
//! dispatching with either [`map`](rten_simd::SimdUnaryOp::map) or
//! [`map_mut`](rten_simd::SimdUnaryOp::map_mut). For other operations
//! this is handled by exposing different constructors for the in-place and
//! mutating cases, such as [`Softmax::new`] and [`Softmax::new_mut`].
//!
//! For operations which use a separate source and destination, the destination
//! is expected to be an uninitialized slice (`[MaybeUninit<T>]`). This allows
//! the caller to control allocation of the buffer and avoid the overhead of
//! initializing elements which the operation will overwrite. The [`ExtendInit`]
//! trait provides a safe API for the common task of filling a new `Vec` with
//! the result of the operation.
//!
//! ## Examples
//!
//! ### Applying a vectorized unary function
//!
//! ```
//! use std::mem::MaybeUninit;
//!
//! use rten_simd::SimdUnaryOp;
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
//! This example applies the softmax function in-place to a mutable slice.
//!
//! ```
//! use rten_simd::SimdOp;
//! use rten_vecmath::Softmax;
//!
//! let mut data = [1., 0.5, 2.0];
//! Softmax::new_mut(&mut data).dispatch();
//! ```
//!
//! ### Applying softmax with separate input and output buffers
//!
//! This example reads data from an input and writes to an uninitialized output
//! buffer (`&mut [MaybeUninit<f32>]`), obtained from the uninitialized portion
//! of a `Vec<f32>`. To update the length of the `Vec<f32>` after it is
//! initialized, the helper `ExtendInit` trait is used.
//!
//! ```
//! use rten_simd::SimdOp;
//! use rten_vecmath::{Softmax, ExtendInit};
//!
//! let data = [1., 0.5, 2.0];
//! let mut output = Vec::with_capacity(data.len());
//! output.extend_init(|output_uninit| {
//!     // `output_uninit` is the uninitialized part of `output`, as returned by
//!     // `output.spare_capacity_mut()`.
//!     //
//!     // The `dispatch` call initializes it and returns the initialized slice.
//!     Softmax::new(&data, output_uninit).dispatch()
//! });
//! assert_eq!(output.len(), 3);
//! ```
//!
//! ### Computing the sum of a list of floats
//!
//! ```
//! use rten_simd::SimdOp;
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
mod relu;
mod sin_cos;
mod softmax;
mod sum;
mod tanh;

#[cfg(test)]
mod ulp;

#[cfg(test)]
mod testing;

mod extend_init;

// Unary functions.
pub use erf::{ApproxGelu, Erf, Gelu};
pub use exp::{Elu, Exp, Sigmoid, Silu, Swish};
pub use quantize::Quantize;
pub use relu::LeakyRelu;
pub use sin_cos::{Cos, Sin};
pub use tanh::Tanh;

// Normalization and reduction functions.
pub use min_max::{MaxNum, MinMax, MinNum};
pub use normalize::{Normalize, NormalizeOptions};
pub use softmax::Softmax;
pub use sum::{Sum, SumSquare, SumSquareSub};

// Utilities
pub use extend_init::ExtendInit;
