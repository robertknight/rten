//! Portable SIMD library.
//!
//! rten-simd is a library for defining operations that are accelerated using
//! [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)
//! instruction sets such as AVX2, Arm Neon or WebAssembly SIMD. Operations are
//! defined once using safe, portable APIs, then _dispatched_ at runtime to
//! evaluate the operation using the best available SIMD instruction set (ISA)
//! on the current CPU.
//!
//! The design is inspired by Google's
//! [Highway](https://github.com/google/highway) library for C++ and the
//! [pulp](https://docs.rs/pulp/latest/pulp/) crate.
//!
//! ## Differences from `std::simd`
//!
//! In nightly Rust the standard library has a built-in portable SIMD API,
//! `std::simd`. This library differs in several ways:
//!
//! 1. It is available on stable Rust
//! 2. The instruction set is selected at runtime rather than compile time. On
//!    x86 an operation may be compiled for AVX-512, AVX2 and generic (SSE). If
//!    the binary is run on a system supporting AVX-512 that version will be
//!    used. The same binary on an older system may use the generic (SSE)
//!    version.
//! 3. Operations use the full available SIMD vector width, which varies by
//!    instruction set, as opposed to specifying a fixed width in the code.
//!    For example a SIMD vector with f32 elements has 4 lanes on Arm Neon and
//!    16 lanes under AVX-512.
//!
//!    The API is designed to support scalable vector ISAs such as [Arm
//!    SVE](https://developer.arm.com/Architectures/Scalable%20Vector%20Extensions)
//!    and RVV in future, where the vector length is known only at runtime.
//!
//! 4. Semantics are chosen to be "performance portable". This means that the
//!    behavior is chosen based on what maps well to the hardware, rather than
//!    strictly matching Rust behaviors for scalars as `std::simd` generally
//!    does. It also means some operations may have different behaviors in edge
//!    cases on different platforms. This is similar to [WebAssembly Relaxed
//!    SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md).
//!
//! ## Supported architectures
//!
//! The currently supported SIMD ISAs are:
//!
//! - AVX2
//! - AVX-512 (requires nightly Rust and `avx512` feature enabled)
//! - Arm Neon
//! - WebAssembly SIMD (including relaxed SIMD)
//!
//! There is also a generic fallback implemented using 128-bit arrays which is
//! designed to be autovectorization-friendly (ie. it compiles on all platforms,
//! and should enable the compiler to use SSE or similar instructions).
//!
//! ## Example
//!
//! This code defines an operation which squares each value in a slice and
//! evaluates it on a vector of floats:
//!
//! ```
//! use rten_simd::{Isa, SimdOp};
//! use rten_simd::ops::NumOps;
//! use rten_simd::functional::simd_map;
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
//! This example shows the basic steps to define a vectorized operation:
//!
//! 1. Create a struct containing the operation's parameters.
//! 2. Implement the [`SimdOp`] trait for the struct to define how to evaluate
//!    the operation.
//! 3. Call [`SimdOp::dispatch`] to evaluate the operation using the best
//!    available instruction set. Here "best" refers to the ISA with the
//!    widest vectors, and thus the maximum amount of parallelism.
//!
//! Note the use of the `#[inline(always)]` attribute on closures and functions
//! called within `eval`. See the section on inlining below for an explanation.
//!
//! ## Separation of vector types and operations
//!
//! SIMD vectors are effectively arrays (like `[T; N]`) with a larger alignment.
//! A SIMD vector type can be created whether or not the associated instructions
//! are supported on the system.
//!
//! Performing a SIMD operation however requires the caller to first ensure that
//! the instructions are supported on the current system. To enforce this,
//! operations are separated from the vector type, and types providing access to
//! SIMD operations ([`Isa`]) can only be instantiated if the instruction set is
//! supported.
//!
//! ## Overview of key traits
//!
//! The [`SimdOp`] trait defines an _operation_ which can be vectorized using
//! different SIMD instruction sets. This trait has a
//! [`dispatch`](SimdOp::dispatch) method to perform the operation.
//!
//! An implementation of the [`Isa`] trait is passed to [`SimdOp::eval`]. The
//! [`Isa`] is the entry point for operations on SIMD vectors. It provides
//! access to implementations of the [`NumOps`](ops::NumOps) trait and
//! sub-traits for each element type. For example [`Isa::f32`] provides
//! operations on SIMD vectors with `f32` elements.
//!
//! The [`NumOps`](ops::NumOps) trait provides operations that are available on
//! all SIMD vectors. The sub-traits [`FloatOps`](ops::FloatOps) and
//! [`IntOps`](ops::IntOps) provide operations that are only available on SIMD
//! vectors with float and integer elements respectively. There is also
//! [`SignedIntOps`](ops::SignedIntOps) for signed integer operations. Finally
//! there are additional traits for operations only available for other subsets
//! of element types. For example [`Extend`](ops::Extend) widens each lane to
//! one with twice the bit-width.
//!
//! SIMD operations (eg. [`NumOps::add`](ops::NumOps::add) take SIMD vectors as
//! arguments. These vectors are either platform-specific types (eg.
//! `float32x4_t` on Arm) or transparent wrappers around them. The [`Simd`]
//! trait is implemented for all vector types. The [`Elem`] trait is implemented
//! for supported element types, providing required numeric operations.
//!
//! ## Use with slices
//!
//! SIMD operations are usually applied to a slice of elements. To support this,
//! the [`SimdIterable`] trait provides a way to iterate over SIMD vector-sized
//! chunks of a slice, using padding or masking to handle slice lengths that are
//! not a multiple of the vector size.
//!
//! The [`functional`] module provides utilities for defining vectorized
//! transforms on slices (eg. [`simd_map`](functional::simd_map)).
//!
//! The [`SliceWriter`] utility provides a way to incrementally initialize the
//! contents of a slice with the results of SIMD operations, by writing one
//! SIMD vector at a time.
//!
//! The [`SimdUnaryOp`] trait provides a convenient way to define unary
//! operations (like [`Iterator::map`]) on slices.
//!
//! ## Importance of inlining
//!
//! In the above example `#[inline(always)]` attributes are used to ensure
//! that the whole `eval` implementation is compiled to a single function. This
//! is required to ensure that the platform-specific intrinsics (from
//! [`core::arch`]) are compiled to direct instructions with no function call
//! overhead.
//!
//! Failure to inline these intrinsics will significantly harm performance,
//! since most of the runtime will be spent in function call overhead rather
//! than actual computation. This issue affects platforms where the availability
//! of the SIMD instruction set is not guaranteed at compile time.  This
//! includes AVX2 and AVX-512 on x86-64, but not Arm Neon or WASM SIMD.
//!
//! If a vectorized operation performs more slowly than expected, use a profiler
//! such as [samply](https://github.com/mstange/samply) to verify that the
//! intrinsics have been inlined and thus do not appear in the list of called
//! functions.
//!
//! The need for this forced inlining is expected to change in future with
//! updates to how Rust's [`target_feature`
//! attribute](https://github.com/rust-lang/rust/issues/69098) works.
//!
//! ## Generic operations
//!
//! It is possible to define operations which are generic over the element type
//! by using the [`GetNumOps`](ops::GetNumOps) trait and related traits. These
//! are implemented for supported element types and provide a way to get the
//! [`NumOps`](ops::NumOps) implementation for that element type from an `Isa`.
//! This can be used to define [`SimdOp`]s which are generic over the element
//! type.
//!
//! This example defines an operation which can sum a slice of any supported
//! element type:
//!
//! ```
//! use std::iter::Sum;
//! use rten_simd::{Isa, Simd, SimdIterable, SimdOp};
//! use rten_simd::ops::{GetNumOps, NumOps};
//!
//! struct SimdSum<'a, T>(&'a [T]);
//!
//! impl<'a, T: GetNumOps + Sum> SimdOp for SimdSum<'a, T> {
//!     type Output = T;
//!
//!     #[inline(always)]
//!     fn eval<I: Isa>(self, isa: I) -> Self::Output {
//!         let ops = T::num_ops(isa);
//!         let partial_sums = self.0.simd_iter(ops).fold(
//!             ops.zero(),
//!             |sum, x| ops.add(sum, x)
//!         );
//!         partial_sums.to_array().into_iter().sum()
//!     }
//! }
//!
//! assert_eq!(SimdSum(&[1.0f32, 2.0, 3.0]).dispatch(), 6.0);
//! assert_eq!(SimdSum(&[1i32, 2, 3]).dispatch(), 6);
//! assert_eq!(SimdSum(&[1u8, 2, 3]).dispatch(), 6u8);
//! ```

#![cfg_attr(
    feature = "avx512",
    feature(stdarch_x86_avx512),
    feature(avx512_target_feature)
)]

mod arch;
mod dispatch;
mod elem;
pub mod functional;
pub mod isa_detection;
mod iter;
pub mod ops;
mod simd;
pub mod span;
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
pub use elem::Elem;
pub use iter::{Iter, SimdIterable};
pub use ops::Isa;
pub use simd::{Mask, Simd};
pub use writer::SliceWriter;

#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
pub use isa_detection::is_avx512_supported;

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

/// Test that two [`Simd`] vectors are not equal according to a [`PartialEq`]
/// comparison of their array representations.
#[cfg(test)]
macro_rules! assert_simd_ne {
    ($x:expr, $y:expr) => {
        assert_ne!($x.to_array(), $y.to_array());
    };
}

#[cfg(test)]
pub(crate) use {assert_simd_eq, assert_simd_ne};

#[cfg(test)]
mod tests {
    use super::functional::simd_map;
    use super::ops::NumOps;
    use super::{Isa, SimdOp};

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
