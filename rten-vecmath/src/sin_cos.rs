// We reuse constants exactly from the XNNPACK implementation.
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]

use rten_simd::ops::{FloatOps, NumOps};
use rten_simd::{Isa, Simd, SimdUnaryOp};

// Values taken from the XNNPACK vsin implementation that was used as a
// reference.
const PI: f32 = 3.1415927;
const INV_2_PI: f32 = 0.15915494;
const HALF_PI: f32 = 1.5707964;

/// Computes the sine function.
#[derive(Default)]
pub struct Sin(SinCos<false>);

impl Sin {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SimdUnaryOp<f32> for Sin {
    #[inline(always)]
    fn eval<I: Isa, S: Simd<Elem = f32, Isa = I>>(&self, isa: I, x: S) -> S {
        self.0.eval(isa, x)
    }
}

/// Computes the cosine function.
#[derive(Default)]
pub struct Cos(SinCos<true>);

impl Cos {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SimdUnaryOp<f32> for Cos {
    #[inline(always)]
    fn eval<I: Isa, S: Simd<Elem = f32, Isa = I>>(&self, isa: I, x: S) -> S {
        self.0.eval(isa, x)
    }
}

/// Computes the sine or cosine function.
#[derive(Default)]
struct SinCos<const COS: bool> {}

impl<const COS: bool> SimdUnaryOp<f32> for SinCos<COS> {
    #[inline(always)]
    fn eval<I: Isa, S: Simd<Elem = f32, Isa = I>>(&self, isa: I, x: S) -> S {
        let ops = isa.f32();
        let x = x.same_cast();

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

        ops.div(p, q).same_cast()
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::{arange, AllF32s, Tolerance, UnaryOpTester};
    use crate::{Cos, Sin};

    // Maximum error of `SinCos` compared to `f32::sin` and `f32::cos` in the
    // range [-10, 10].
    const MAX_SIN_COS_ERROR: f32 = 2.0 * std::f32::EPSILON; // 2.38e-7

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_sin_exhaustive() {
        // FIXME: This implementation can produce out of range values for large
        // finite inputs. This test captures the current accuracy.
        let limit = 48817.0;
        let test = UnaryOpTester {
            reference: f32::sin,
            simd: Sin::new(),
            range: AllF32s::new().filter(|x| !x.is_finite() || x.abs() < limit),
            tolerance: Tolerance::Absolute(3e-7),
        };
        test.run_with_progress();
    }

    #[test]
    fn test_sin() {
        let test = UnaryOpTester {
            reference: f32::sin,
            simd: Sin::new(),
            range: arange(-10., 10., 0.1f32),
            tolerance: Tolerance::Absolute(MAX_SIN_COS_ERROR),
        };
        test.run();
    }

    #[test]
    fn test_cos() {
        let test = UnaryOpTester {
            reference: f32::cos,
            simd: Cos::new(),
            range: arange(-10., 10., 0.1f32),
            tolerance: Tolerance::Absolute(MAX_SIN_COS_ERROR),
        };
        test.run();
    }
}
