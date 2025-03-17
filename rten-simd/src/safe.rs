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
//! use rten_simd::safe::{Isa, SimdOp, NumOps};
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
//!         simd_map(ops, self.xs, #[inline(always)] |x| ops.mul(x, x))
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
//! Note the use of the `#[inline(always)]` attribute on closures and any
//! functions called within `eval`. See the section on inlining below for an
//! explanation.
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
//! different SIMD instruction sets. This trait has a
//! [`dispatch`](SimdOp::dispatch) method to perform the operation.
//!
//! An instance of the [`Isa`] trait is passed to the operation when it is
//! evaluated. The type of ISA will depend on the selected instruction set.  The
//! ISA provides access to different implementations of the [`NumOps`] trait
//! and sub-traits. These in turn provide operations on SIMD vectors with
//! different data types. The [`NumOps`] trait provides operations that are
//! available on all SIMD vectors. The sub-traits [`FloatOps`] and
//! [`SignedIntOps`] provide operations that are only available on SIMD vectors
//! with float and signed integer elements respectively.
//!
//! ## Applying SIMD operations to slices
//!
//! SIMD operations are usually applied to a slice of elements. To support this,
//! the [`SimdIterable`] trait provides a way to iterate over SIMD vector-sized
//! chunks of a slice.
//!
//! The [`functional`] module provides utilities for defining vectorized
//! transforms on slices (eg. [`simd_map`](functional::simd_map)).
//!
//! The [`SliceWriter`] utility provides a way to incrementally initialize the
//! contents of a slice with the results of SIMD operations, by writing one
//! SIMD vector at a time.
//!
//! ## The importance of inlining
//!
//! In the above example `#[inline(always)]` attributes are applied to ensure
//! that the whole operation is compiled to a single function, with one instance
//! generated per enabled ISA on each platform. This is required in current
//! stable versions of Rust to ensure that the low-level intrinsics (eg.
//! `_mm256_add_ps` to add two f32 SIMD vectors on x64) are compiled to direct
//! instructions with no function call overhead.
//!
//! Failure to inline these intrinsics will significantly harm performance,
//! since most of the runtime will be spend in function call overhead rather
//! than actual computation. This issue affects platforms where the availability
//! of the SIMD instruction set is not guaranteed at compile time.  This
//! includes AVX2 and AVX-512 on x86-64, but not Arm Neon or WASM SIMD.
//!
//! If a vectorized operation performs more slowly than expected, it is
//! recommended to use a profiler such as
//! [samply](https://github.com/mstange/samply) to verify that the intrinsics
//! have been inlined and thus do not appear in the list of called functions.
//!
//! The need for this comprehensive and aggressive approach to inlining is
//! expected to change in future with updates to how Rust's [`target_feature`
//! attribute](https://github.com/rust-lang/rust/issues/69098) works.
mod arch;
mod dispatch;
pub mod functional;
mod iter;
mod vec;
mod writer;

/// Target-specific [`Isa`] implementations.
///
/// Most code using this library will not need to use these types. Instead the
/// appropriate ISA will be constructed when using a dispatch method such as
/// [`SimdOp::dispatch`]. These types are exported for use in downstream code
/// which uses the portable SIMD APIs but also has ISA-specific properties.
pub mod isa {
    pub use super::arch::generic::GenericIsa;

    #[cfg(target_arch = "aarch64")]
    pub use super::arch::aarch64::ArmNeonIsa;

    #[cfg(target_arch = "x86_64")]
    pub use super::arch::x86_64::Avx2Isa;

    #[cfg(target_arch = "x86_64")]
    #[cfg(feature = "avx512")]
    pub use super::arch::x86_64::Avx512Isa;

    #[cfg(target_arch = "wasm32")]
    #[cfg(target_feature = "simd128")]
    pub use super::arch::wasm32::Wasm32Isa;
}

pub use dispatch::{SimdOp, SimdUnaryOp};
pub use iter::{Iter, SimdIterable};
pub use vec::{
    Elem, Extend, FloatOps, Isa, Mask, MaskOps, NarrowSaturate, NumOps, SignedIntOps, Simd,
};
pub use writer::SliceWriter;

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
    use super::{Isa, NumOps, SimdOp};

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
