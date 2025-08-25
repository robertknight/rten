//! Vectorized version of the exponential and closely related functions.

#![allow(clippy::excessive_precision)]

use rten_simd::ops::{FloatOps, IntOps, NumOps};
use rten_simd::{Isa, Simd, SimdUnaryOp};

const INV_LOG2: f32 = std::f32::consts::LOG2_E; // aka. 1 / ln2
const ROUNDING_MAGIC: f32 = 12582912.; // 0x3 << 22

// `log(2)` split into large and small parts for Cody-Waite range reduction.
const LOG2_HI: f32 = -6.93145752e-1;
const LOG2_LO: f32 = -1.42860677e-6;

// Coefficients of polynomial used to approximate `exp(x)` in `[0, ln2/2]`.
//
// These values are very close to, but not exactly the same as the coefficients
// of the Taylor series around 0 (1, 1/1!, 1/2!, 1/3!, 1/4! ...).
const EXP_POLY_0: f32 = 1.0;
const EXP_POLY_1: f32 = 1.0;
const EXP_POLY_2: f32 = 4.99999851e-1; // ~ 1/2!
const EXP_POLY_3: f32 = 1.66664720e-1; // ~ 1/3! or 1/6
const EXP_POLY_4: f32 = 4.16695364e-2; // ~ 1/4! or 1/24
const EXP_POLY_5: f32 = 8.37312452e-3; // ~ 1/5! or 1/120
const EXP_POLY_6: f32 = 1.37805939e-3; // ~ 1/6! or 1/720

/// Vectorized exponential function.
///
/// This has a maximum error of 1 ULP compared to `f32::exp` in the Rust standard
/// library.
#[derive(Default)]
pub struct Exp {}

// Implementation based on work by Norbert Juffa in
// https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528.
//
// See also
// https://justinwillmert.com/articles/2020/numerically-computing-the-exponential-function-with-polynomial-approximations/.
//
// Method outline:
//
//  1. Use the identity `exp(a + b) = exp(a) * exp(b)` to reduce the range for
//     which a polynomial approximation needs to be valid:
//
//     ```text
//        exp(x) = exp(ln2 * k) * exp(r);
//               = 2**k * exp(r)
//     ```
//
//     Such that `k` is an integer and `|r| <= 1/2 ln 2`.
//
//     ```text
//             k = rintf(x / ln2)
//             r = x - k * ln 2
//     ```
//
//  2. Compute `exp(r)` using a polynomial approximation.
//
//  3. Compute result as `exp(x) = exp(r) * 2**k`. The reconstruction is split
//     into multiple steps to extend the domain.
impl SimdUnaryOp<f32> for Exp {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();
        let int_ops = isa.i32();

        // Load constants
        let inv_log_2 = ops.splat(INV_LOG2);
        let rounding_magic = ops.splat(ROUNDING_MAGIC);
        let ln2_hi = ops.splat(LOG2_HI);
        let ln2_lo = ops.splat(LOG2_LO);

        let p6 = ops.splat(EXP_POLY_6);
        let p5 = ops.splat(EXP_POLY_5);
        let p4 = ops.splat(EXP_POLY_4);
        let p3 = ops.splat(EXP_POLY_3);
        let p2 = ops.splat(EXP_POLY_2);
        let p1 = ops.splat(EXP_POLY_1);
        let p0 = ops.splat(EXP_POLY_0);

        // Compute `k = rintf(x / ln2), r = x - k * ln2`.
        let j = ops.mul_add(x, inv_log_2, rounding_magic);
        let j = ops.sub(j, rounding_magic);
        let r = ops.mul_add(j, ln2_hi, x);
        let r = ops.mul_add(j, ln2_lo, r);
        let k = ops.to_int_trunc(j);

        // Approximate `exp(r)` on interval [-ln2 / 2, +ln2 / 2]
        let mut tmp = p6;
        tmp = ops.mul_add(tmp, r, p5);
        tmp = ops.mul_add(tmp, r, p4);
        tmp = ops.mul_add(tmp, r, p3);
        tmp = ops.mul_add(tmp, r, p2);
        tmp = ops.mul_add(tmp, r, p1);
        let r = ops.mul_add(tmp, r, p0);

        // Reconstruct `exp(x) = 2**k * exp(r`).
        //
        // Reconstruction is split into steps to extend the input domain of the
        // function. The split reconstruction is effectively:
        //
        //   When k > 0:  exp(r) * exp2(127) * exp2(k - 127)
        //   When k <= 0: exp(r) * exp2(-123) * exp2(k + 123)
        //
        // Where 127 is the exponent bias for f32.
        let ia = int_ops.gt(k, int_ops.zero());
        let x7f = int_ops.splat(0x7f000000);
        #[allow(overflowing_literals)]
        let x83 = int_ops.splat(0x83000000);
        let ia = int_ops.select(int_ops.zero(), x83, ia);
        let is = int_ops.add(ia, x7f);

        let it = int_ops.shift_left::<23>(k);
        let it = int_ops.sub(it, ia);

        let s: I::F32 = is.reinterpret_cast();
        let t: I::F32 = it.reinterpret_cast();
        let r = ops.mul(r, s);
        let r = ops.mul(r, t);

        // Handle overflow and underflow when `x.abs() >= 104.`
        let overflow_mask = ops.ge(x, ops.splat(104.0));
        let underflow_mask = ops.le(x, ops.splat(-104.0));
        let r = ops.select(ops.splat(f32::INFINITY), r, overflow_mask);
        ops.select(ops.zero(), r, underflow_mask)
    }
}

/// Cutoff value chosen such that if `k = round(x / ln2)`, `2**k` is a normal
/// number.
const EXP_LOWER_CUTOFF: f32 = -126.5 * std::f32::consts::LN_2 + 0.01; // ~87.67

/// A simplified and faster version of [`Exp`] with a reduced domain and range.
///
/// 1. The input value must be <= 0
/// 2. The lower cutoff for which `exp(x)` returns 0 is higher (~87.67 instead of ~104).
#[derive(Default)]
pub struct ReducedRangeExp {}

impl SimdUnaryOp<f32> for ReducedRangeExp {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();
        let int_ops = isa.i32();

        // Load constants
        let inv_log_2 = ops.splat(INV_LOG2);
        let rounding_magic = ops.splat(ROUNDING_MAGIC);
        let ln2_hi = ops.splat(LOG2_HI);
        let ln2_lo = ops.splat(LOG2_LO);

        let p6 = ops.splat(EXP_POLY_6);
        let p5 = ops.splat(EXP_POLY_5);
        let p4 = ops.splat(EXP_POLY_4);
        let p3 = ops.splat(EXP_POLY_3);
        let p2 = ops.splat(EXP_POLY_2);
        let p1 = ops.splat(EXP_POLY_1);
        let p0 = ops.splat(EXP_POLY_0);

        // Compute `k = rintf(x / ln2), r = x - k * ln2`.
        //
        // Since x <= 0, also k <= 0.
        let j = ops.mul_add(x, inv_log_2, rounding_magic);
        let j = ops.sub(j, rounding_magic);
        let r = ops.mul_add(j, ln2_hi, x);
        let r = ops.mul_add(j, ln2_lo, r);
        let k = ops.to_int_trunc(j);

        // Approximate `exp(r)` on interval [-ln2 / 2, +ln2 / 2]
        let mut tmp = p6;
        tmp = ops.mul_add(tmp, r, p5);
        tmp = ops.mul_add(tmp, r, p4);
        tmp = ops.mul_add(tmp, r, p3);
        tmp = ops.mul_add(tmp, r, p2);
        tmp = ops.mul_add(tmp, r, p1);
        let r = ops.mul_add(tmp, r, p0);

        // Reconstruct `exp(x) = 2**k * exp(r)`.
        //
        // This is valid as long as `k >= -126`, so that `2**k` as f32 is a
        // normal number.
        let exponent_bias = int_ops.splat(127);
        let k_pow2 = int_ops.shift_left::<23>(int_ops.add(k, exponent_bias));
        let k_pow2: I::F32 = k_pow2.reinterpret_cast();
        let r = ops.mul(r, k_pow2);

        // Handle underflow. We don't need to handle overflow since x <= 0.
        let underflow_mask = ops.lt(x, ops.splat(EXP_LOWER_CUTOFF));
        ops.select(ops.zero(), r, underflow_mask)
    }
}

/// Computes the [sigmoid function][sigmoid], aka. the standard logistic function, `1. /
/// (1. + (-x).exp())`.
///
/// This has a maximum error of 4 ULPs compared to a reference implementation
/// using `1. / (1. + (-x).exp())`.
///
/// [sigmoid]: https://en.wikipedia.org/wiki/Logistic_function#Mathematical_properties
#[derive(Default)]
pub struct Sigmoid {}

impl SimdUnaryOp<f32> for Sigmoid {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();

        // 1. + exp(-x)
        let denom = ops.add(ops.one(), Exp::apply(isa, ops.neg(x)));
        ops.reciprocal(denom)
    }
}

/// Vectorized Sigmoid Linear Unit (SiLU) function.
///
/// This computes `x * sigmoid(x)` for all lanes in `x`.
pub struct Silu {}

impl SimdUnaryOp<f32> for Silu {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();

        // 1. + exp(-x)
        let denom = ops.add(ops.one(), Exp::apply(isa, ops.neg(x)));
        ops.div(x, denom)
    }
}

/// Vectorized Swish function.
///
/// This computes `x * sigmoid(beta * x)` for each element.
pub struct Swish {
    pub beta: f32,
}

impl SimdUnaryOp<f32> for Swish {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();

        let beta = ops.splat(self.beta);
        ops.mul(x, Sigmoid::apply(isa, ops.mul(x, beta)))
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::SimdUnaryOp;

    use super::{ReducedRangeExp, EXP_LOWER_CUTOFF};
    use crate::testing::{arange, benchmark_op, AllF32s, Tolerance, UnaryOpTester};
    use crate::{Exp, Sigmoid, Silu, Swish};

    // Maximum error of `Exp` compared to Rust standard library implementation.
    const MAX_EXP_ERROR_ULPS: f32 = 1.0;

    // Maximum error of `Sigmoid` compared to reference implementation below.
    const MAX_SIGMOID_ERROR_ULPS: f32 = 4.0;

    fn reference_sigmoid(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    fn reference_silu(x: f32) -> f32 {
        x * reference_sigmoid(x)
    }

    fn reference_swish(x: f32, beta: f32) -> f32 {
        x * reference_sigmoid(beta * x)
    }

    #[test]
    fn test_exp_basic() {
        // A few simple test cases, including "typical" +/-ve inputs with
        // |x| above/below ln2, zero and values below/above min/max cutoffs.
        let cases = [-2.0f32, -1., -0.5, 0.1, 0., 0.1, 0.5, 1., 2., -105., 105.];

        let exp_op = Exp {};
        for case in cases {
            let expected = case.exp();
            let actual = exp_op.scalar_eval(case);
            let diff = (expected - actual).abs();

            if actual.is_infinite() || expected.is_infinite() {
                assert_eq!(actual, expected);
            } else {
                // The expected precision is less than 1 ULP, so the diff should
                // be exactly zero.
                assert_eq!(diff, 0.);
            };
        }
    }

    #[test]
    fn test_exp() {
        let test = UnaryOpTester {
            reference: f32::exp,
            simd: Exp {},
            range: arange(-6., 6., 0.001),
            tolerance: Tolerance::Ulp(MAX_EXP_ERROR_ULPS),
        };
        test.run();
    }

    #[test]
    fn test_reduced_range_exp() {
        let test = UnaryOpTester {
            reference: f32::exp,
            simd: ReducedRangeExp {},
            range: arange(EXP_LOWER_CUTOFF, 0., 0.015),
            tolerance: Tolerance::Ulp(MAX_EXP_ERROR_ULPS),
        };
        test.run();
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_exp_exhaustive() {
        let test = UnaryOpTester {
            reference: f32::exp,
            simd: Exp {},
            range: AllF32s::new(),
            tolerance: Tolerance::Ulp(MAX_EXP_ERROR_ULPS),
        };
        test.run_with_progress();
    }

    #[test]
    fn test_sigmoid() {
        let test = UnaryOpTester {
            reference: reference_sigmoid,
            simd: Sigmoid {},
            range: arange(-6., 6., 0.001),
            tolerance: Tolerance::Ulp(MAX_SIGMOID_ERROR_ULPS),
        };
        test.run();
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_sigmoid_exhaustive() {
        let test = UnaryOpTester {
            reference: reference_sigmoid,
            simd: Sigmoid {},
            range: AllF32s::new(),
            tolerance: Tolerance::Ulp(MAX_SIGMOID_ERROR_ULPS),
        };
        test.run_with_progress();
    }

    #[test]
    fn test_silu() {
        let test = UnaryOpTester {
            reference: reference_silu,
            simd: Silu {},
            range: arange(-6., 6., 0.001),
            tolerance: Tolerance::Ulp(MAX_SIGMOID_ERROR_ULPS),
        };
        test.run();
    }

    #[test]
    fn test_swish() {
        let beta = 1.7;
        let test = UnaryOpTester {
            reference: |x| reference_swish(x, beta),
            simd: Swish { beta },
            range: arange(-6., 6., 0.001),
            tolerance: Tolerance::Ulp(MAX_SIGMOID_ERROR_ULPS),
        };
        test.run();
    }

    #[test]
    #[ignore]
    fn bench_exp() {
        benchmark_op(
            |xs, ys| xs.iter().zip(ys.iter_mut()).for_each(|(x, y)| *y = x.exp()),
            |xs, ys| Exp {}.map(xs, ys),
        );
    }

    #[test]
    #[ignore]
    fn bench_sigmoid() {
        benchmark_op(
            |xs, ys| {
                xs.iter()
                    .zip(ys.iter_mut())
                    .for_each(|(x, y)| *y = reference_sigmoid(*x))
            },
            |xs, ys| Sigmoid {}.map(xs, ys),
        );
    }
}
