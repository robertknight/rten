//! Portable SIMD library.
//!
//! _This module contains a new portable SIMD API which is in development. The
//! focus is to revise the API to reduce the amount of unsafe code needed when
//! using it._
//!
//! ## Usage
//!
//! The steps to define a vectorized operation using this module are:
//!
//! 1. Create a struct containing the operation's parameters.
//! 2. Implement the [`SimdOp`] trait for the struct to define how to evaluate
//!    the operation.
//! 3. Call [`SimdOp::dispatch`] to select the preferred SIMD instruction set and
//!    evaluate the operation using it.
//!
//! ## Functional utilities
//!
//! The [`functional`] module provides utilities for defining vectorized
//! transforms ([`simd_map`](functional::simd_map)) and reductions
//! ([`simd_fold`](functional::simd_fold)).
mod arch;
mod dispatch;
pub mod functional;
mod vec;

pub use dispatch::{SimdOp, SimdUnaryFloatOp};
pub use vec::{Elem, Isa, MakeSimd, Mask, Simd, SimdF32, SimdFloat, SimdInt};

#[cfg(test)]
mod tests {
    use super::functional::simd_map;
    use super::{Isa, SimdOp};

    #[test]
    fn test_simd_f32_op() {
        struct Square<'a> {
            xs: &'a mut [f32],
        }

        impl<'a> SimdOp for Square<'a> {
            type Output = &'a mut [f32];

            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                simd_map(isa.f32(), self.xs.into(), |x| x * x)
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
                simd_map(isa.i32(), self.xs.into(), |x| x * x)
            }
        }

        let mut buf: Vec<_> = (0..32).collect();
        let expected: Vec<_> = buf.iter().map(|x| *x * *x).collect();

        let squared = Square { xs: &mut buf }.dispatch();

        assert_eq!(squared, &expected);
    }
}
