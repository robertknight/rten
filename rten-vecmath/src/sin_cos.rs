// We reuse constants exactly from the XNNPACK implementation.
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]

use rten_base::hint::unlikely;
use rten_simd::ops::{FloatOps, MaskOps, NumOps};
use rten_simd::{Isa, Simd, SimdUnaryOp};

// Values taken from the XNNPACK vsin implementation that was used as a
// reference.
const PI: f32 = 3.1415927;
const INV_2_PI: f32 = 0.15915494;
const HALF_PI: f32 = 1.5707964;

// Threshold for large inputs. If abs(x) exceeds this value, the implementation
// falls back to the standard library.
const LARGE_THRESHOLD: f32 = 48_000.0;

/// Computes the sine function.
///
/// The implementation has a maximum absolute error of 2.98e-7 (2.5 * f32::EPSILON).
#[derive(Default)]
pub struct Sin(SinCos<false>);

impl Sin {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SimdUnaryOp<f32> for Sin {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        self.0.eval(isa, x)
    }
}

/// Computes the cosine function.
///
/// The implementation has a maximum absolute error of 4.17e-7 (3.5 * f32::EPSILON).
#[derive(Default)]
pub struct Cos(SinCos<true>);

impl Cos {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SimdUnaryOp<f32> for Cos {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        self.0.eval(isa, x)
    }
}

/// Computes the sine or cosine function.
#[derive(Default)]
struct SinCos<const COS: bool> {}

impl<const COS: bool> SimdUnaryOp<f32> for SinCos<COS> {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();
        let mask_ops = isa.m32();

        // For large inputs the vectorized algorithm can produce results
        // outside the [-1, 1] range. Fall back to scalar evaluation for such
        // inputs to avoid this.
        let large = ops.ge(ops.abs(x), ops.splat(LARGE_THRESHOLD));
        if unlikely(mask_ops.any(large)) {
            let mut y = x.to_array();
            for i in 0..ops.len() {
                y[i] = if COS { y[i].cos() } else { y[i].sin() };
            }
            return ops.load(y.as_ref());
        }

        // The implementation here is based on XNNPACK.
        // See https://github.com/google/XNNPACK/blob/master/src/f32-vsin/rational-5-4.c.in.

        // Range reduction constants.
        let inv_2_pi = ops.splat(INV_2_PI);
        let two_pi_hi = ops.splat(6.28125);
        let two_pi_lo = ops.splat(1.9353072e-3);
        let pi = ops.splat(PI);

        // Rational approximation numerator constants.
        let a3 = ops.splat(-1.3314664364e-01);
        let a5 = ops.splat(3.2340581529e-03);
        let one = ops.splat(1.0);

        // Rational approximation denominator constants.
        let b2 = ops.splat(3.3519912511e-02);
        let b4 = ops.splat(4.8770775902e-04);

        // Compute range-reduced `x_rr` such that `x_rr ∈ [−π, π]`.
        let k = ops.round_ties_even(ops.mul(x, inv_2_pi));
        let x_rr = ops.mul_sub_from(k, two_pi_hi, x);
        let mut x_rr = ops.mul_sub_from(k, two_pi_lo, x_rr);

        if COS {
            let pi_half = ops.splat(HALF_PI);
            x_rr = ops.sub(pi_half, x_rr);
        }

        // Further reduce range to [-π/2, π/2].
        let x_rr = ops.min(x_rr, ops.sub(pi, x_rr));
        let x_rr = ops.max(x_rr, ops.sub(ops.neg(pi), x_rr));
        let x_rr = ops.min(x_rr, ops.sub(pi, x_rr));

        // Approximate sin via a rational approximation.
        let x_rr_sq = ops.mul(x_rr, x_rr);

        // Numerator polynomial
        let p = ops.mul_add(x_rr_sq, a5, a3);
        let p = ops.mul_add(x_rr_sq, p, one);
        let p = ops.mul(p, x_rr);

        // Denominator polynomial
        let q = ops.mul_add(x_rr_sq, b4, b2);
        let q = ops.mul_add(x_rr_sq, q, one);

        ops.div(p, q)
    }
}

#[cfg(test)]
mod tests {
    use super::LARGE_THRESHOLD;
    use crate::testing::{ARange, Tolerance, UnaryOpTester, arange};
    use crate::{Cos, Sin};

    // Maximum error of `SinCos` compared to `f32::sin` and `f32::cos` in the
    // `SMALL_X` range.
    const MAX_ERROR_FOR_SMALL_X: f32 = 2.0 * std::f32::EPSILON; // 2.38e-7

    // Range of small/medium X values which we expect most inputs will be in.
    const SMALL_X: ARange<f32> = arange(-10., 10., 0.1f32);

    // Multiples of π are the worst case for range reduction.
    fn multiples_of_pi() -> impl Iterator<Item = f32> + Clone {
        (-5..5).map(|n| (n as f32) * super::PI)
    }

    // Generate all float values in the range [min, max].
    fn all_floats_in_range(min: f32, max: f32) -> impl Iterator<Item = f32> + Clone {
        std::iter::successors(Some(min), |f| Some(f.next_up())).take_while(move |x| *x <= max)
    }

    #[test]
    fn test_sin() {
        let test = UnaryOpTester {
            reference: f32::sin,
            simd: Sin::new(),
            range: SMALL_X.chain(multiples_of_pi()),
            tolerance: Tolerance::Absolute(MAX_ERROR_FOR_SMALL_X),
        };
        test.run();
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_sin_exhaustive() {
        let test = UnaryOpTester {
            reference: f32::sin,
            simd: Sin::new(),
            range: all_floats_in_range(-LARGE_THRESHOLD, LARGE_THRESHOLD),
            tolerance: Tolerance::Absolute(3e-7),
        };
        test.run_with_progress();
    }

    #[test]
    fn test_cos() {
        let test = UnaryOpTester {
            reference: f32::cos,
            simd: Cos::new(),
            range: SMALL_X.chain(multiples_of_pi()),
            tolerance: Tolerance::Absolute(MAX_ERROR_FOR_SMALL_X),
        };
        test.run();
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_cos_exhaustive() {
        let test = UnaryOpTester {
            reference: f32::cos,
            simd: Cos::new(),
            range: all_floats_in_range(-LARGE_THRESHOLD, LARGE_THRESHOLD),
            // Maximum error for cos is larger than for sin because cos has an
            // extra subtraction.
            tolerance: Tolerance::Absolute(5e-7),
        };
        test.run_with_progress();
    }
}
