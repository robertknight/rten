//! Portable SIMD library.
//!
//! _This module contains a new portable SIMD API which is in development. The
//! focus is to revise the API to reduce the amount of unsafe code needed when
//! using it._
//!
//! rten-simd is a library for defining operations that are accelerated using
//! [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)
//! instructions such as AVX2, Arm Neon or WebAssembly SIMD. These operations
//! can be defined once and then evaluated using the preferred instruction set
//! at runtime, depending on the platform and available CPU instructions.
//!
//! The design is inspired by Google's
//! [Highway](https://github.com/google/highway) library for C++ and the
//! [pulp](https://docs.rs/pulp/latest/pulp/) crate.
//!
//! ## Example
//!
//! This code defines an operation which squares each value in a slice and
//! evaluates it on a vector of floats:
//!
//! ```
//! use rten_simd::safe::{Isa, SimdOp, SimdOps};
//! use rten_simd::safe::functional::simd_map;
//!
//! struct Square<'a> {
//!     xs: &'a mut [f32],
//! }
//!
//! impl<'a> SimdOp for Square<'a> {
//!     type Output = &'a mut [f32];
//!
//!     #[inline(always)]
//!     fn eval<I: Isa>(self, isa: I) -> Self::Output {
//!         let ops = isa.f32();
//!         simd_map(ops, self.xs, |x| ops.mul(x, x))
//!     }
//! }
//!
//! let mut buf: Vec<_> = (0..32).map(|x| x as f32).collect();
//! let expected: Vec<_> = buf.iter().map(|x| *x * *x).collect();
//! let squared = Square { xs: &mut buf }.dispatch();
//! assert_eq!(squared, &expected);
//! ```
//!
//! The above may use AVX2 on an x86 system, Arm Neon on aarch64 and WASM SIMD
//! on wasm32. In the `simd_map` callback, `x` is the SIMD vector for the chosen
//! instruction set.
//!
//! This example shows the basic steps to define a vectorized operation:
//!
//! 1. Create a struct containing the operation's parameters.
//! 2. Implement the [`SimdOp`] trait for the struct to define how to evaluate
//!    the operation.
//! 3. Call [`SimdOp::dispatch`] to select the preferred SIMD instruction set and
//!    evaluate the operation using it.
//!
//! ## Separation of SIMD vector types and operations
//!
//! SIMD vectors are effectively arrays (like `[T; N]`) with a specific
//! alignment. A SIMD vector type can be created whether or not the associated
//! instructions are supported on the system.
//!
//! Performing a SIMD operation however requires the caller to first ensure that
//! the instructions are supported on the current system. To enforce this,
//! operations are separated from the vector type, and types providing access to
//! SIMD operations ([`Isa`]) can only be instantiated if the instruction set is
//! supported.
//!
//! ## Key traits
//!
//! The [`SimdOp`] trait defines an _operation_ which can be vectorized using
//! different SIMD instruction sets.
//!
//! An instance of the [`Isa`] trait is passed to the operation when it is
//! evaluated. This instance provides access to different implementations of
//! the [`SimdOps`] trait and sub-traits, which provide operations on SIMD
//! vectors with different data types. The [`SimdOps`] trait provides operations
//! that are available on all SIMD vectors. The sub-traits [`SimdFloatOps`]
//! and [`SimdIntOps`] provide operations that are only available on SIMD
//! vectors with float and integer elements respectively.
//!
//! ## Applying SIMD operations to slices
//!
//! SIMD operations are usually applied to a slice of elements. To assist this,
//! the [`SimdIterable`] trait provides a way to iterate over SIMD vector-sized
//! chunks of a slice.
//!
//! The [`functional`] module provides utilities for defining vectorized
//! transforms on slices (eg. [`simd_map`](functional::simd_map)).
mod arch;
mod dispatch;
pub mod functional;
mod iter;
mod vec;

pub use dispatch::{SimdOp, SimdUnaryOp};
pub use iter::{Iter, SimdIterable};
pub use vec::{Elem, Isa, Mask, Simd, SimdFloatOps, SimdIntOps, SimdOps};

#[cfg(test)]
pub(crate) use dispatch::test_simd_op;

/// Test that two [`Simd`] vectors are equal according to a [`PartialEq`]
/// comparison of their array representations.
#[cfg(test)]
macro_rules! assert_simd_eq {
    ($x:expr, $y:expr) => {
        assert_eq!($x.to_array(), $y.to_array());
    };
}

#[cfg(test)]
pub(crate) use assert_simd_eq;

#[cfg(test)]
mod tests {
    use super::functional::simd_map;
    use super::{Isa, SimdOp, SimdOps};

    #[test]
    fn test_simd_f32_op() {
        struct Square<'a> {
            xs: &'a mut [f32],
        }

        impl<'a> SimdOp for Square<'a> {
            type Output = &'a mut [f32];

            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                let ops = isa.f32();
                simd_map(ops, self.xs, |x| ops.mul(x, x))
            }
        }

        let mut buf: Vec<_> = (0..32).map(|x| x as f32).collect();
        let expected: Vec<_> = buf.iter().map(|x| *x * *x).collect();

        let squared = Square { xs: &mut buf }.dispatch();

        assert_eq!(squared, &expected);
    }

    #[test]
    fn test_simd_i32_op() {
        struct Square<'a> {
            xs: &'a mut [i32],
        }

        impl<'a> SimdOp for Square<'a> {
            type Output = &'a mut [i32];

            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                let ops = isa.i32();
                simd_map(ops, self.xs, |x| ops.mul(x, x))
            }
        }

        let mut buf: Vec<_> = (0..32).collect();
        let expected: Vec<_> = buf.iter().map(|x| *x * *x).collect();

        let squared = Square { xs: &mut buf }.dispatch();

        assert_eq!(squared, &expected);
    }
}
