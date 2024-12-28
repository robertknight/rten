//! Vectorized version of the exponential and closely related functions.

#![allow(clippy::excessive_precision)]

use std::mem::MaybeUninit;

use rten_simd::dispatch::{dispatch_map_op, dispatch_map_op_in_place, SimdUnaryOp};
use rten_simd::{Simd, SimdFloat, SimdInt};

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

/// Computes e^val. Functionally equivalent to [`f32::exp`].
///
/// This is scalar variant of [`vec_exp`] that uses exactly the same algorithm.
/// It has no performance or correctness advantage over [`f32::exp`] on most systems.
pub fn exp(val: f32) -> f32 {
    // Safety: f32 is available on all systems.
    unsafe { simd_exp(val) }
}

/// Vectorized implementation of exponential function.
///
/// Based on work by Norbert Juffa in
/// https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528.
///
/// See also
/// https://justinwillmert.com/articles/2020/numerically-computing-the-exponential-function-with-polynomial-approximations/.
///
/// Method outline:
///
///  1. Use the identity `exp(a + b) = exp(a) * exp(b)` to reduce the range for
///     which a polynomial approximation needs to be valid:
///
///     ```text
///        exp(x) = exp(ln2 * k) * exp(r);
///               = 2**k * exp(r)
///     ```
///
///     Such that `k` is an integer and `|r| <= 1/2 ln 2`.
///
///     ```text
///             k = rintf(x / ln2)
///             r = x - k * ln 2
///     ```
///
///  2. Compute `exp(r)` using a polynomial approximation.
///
///  3. Compute result as `exp(x) = exp(r) * 2**k`. The reconstruction is split
///     into multiple steps to extend the domain.
///
/// This has a maximum error of 1 ULP compared to `f32::exp` in the Rust standard
/// library.
///
/// Safety: The caller must ensure the `SimdFloat` impl is usable on the current system.
#[inline(always)]
pub(crate) unsafe fn simd_exp<S: SimdFloat>(x: S) -> S {
    // Load constants
    let inv_log_2 = S::splat(INV_LOG2);
    let rounding_magic = S::splat(ROUNDING_MAGIC);
    let ln2_hi = S::splat(LOG2_HI);
    let ln2_lo = S::splat(LOG2_LO);

    let p6 = S::splat(EXP_POLY_6);
    let p5 = S::splat(EXP_POLY_5);
    let p4 = S::splat(EXP_POLY_4);
    let p3 = S::splat(EXP_POLY_3);
    let p2 = S::splat(EXP_POLY_2);
    let p1 = S::splat(EXP_POLY_1);
    let p0 = S::splat(EXP_POLY_0);

    // Compute `k = rintf(x / ln2), r = x - k * ln2`.
    let j = x.mul_add(inv_log_2, rounding_magic);
    let j = j.sub(rounding_magic);
    let r = j.mul_add(ln2_hi, x);
    let r = j.mul_add(ln2_lo, r);
    let k = j.to_int_trunc();

    // Approximate `exp(r)` on interval [-ln2 / 2, +ln2 / 2]
    let mut tmp = p6;
    tmp = tmp.mul_add(r, p5);
    tmp = tmp.mul_add(r, p4);
    tmp = tmp.mul_add(r, p3);
    tmp = tmp.mul_add(r, p2);
    tmp = tmp.mul_add(r, p1);
    let r = tmp.mul_add(r, p0);

    // Reconstruct `exp(x) = 2**k * exp(r`).
    //
    // Reconstruction is split into steps to extend the input domain of the
    // function. The split reconstruction is effectively:
    //
    //   When k > 0:  exp(r) * exp2(127) * exp2(k - 127)
    //   When k <= 0: exp(r) * exp2(-123) * exp2(k + 123)
    //
    // Where 127 is the exponent bias for f32.
    let ia = k.gt(S::Int::zero());
    let x7f = S::Int::splat(0x7f000000);
    #[allow(overflowing_literals)]
    let x83 = S::Int::splat(0x83000000);
    let ia = x83.blend(S::Int::zero(), ia);
    let is = ia.add(x7f);

    let it = k.shl::<23>();
    let it = it.sub(ia);

    let s = is.reinterpret_as_float();
    let t = it.reinterpret_as_float();
    let r = r.mul(s);
    let r = r.mul(t);

    // Handle overflow and underflow when `x.abs() >= 104.`
    let overflow_mask = x.ge(S::splat(104.0));
    let underflow_mask = x.le(S::splat(-104.0));
    let r = r.blend(S::splat(f32::INFINITY), overflow_mask);
    r.blend(S::zero(), underflow_mask)
}

struct SimdExp {}
impl SimdUnaryOp for SimdExp {
    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S {
        simd_exp(x)
    }
}

/// Vectorized exponential function.
///
/// This is a vectorized version of [`exp`] that computes the function for each
/// element in `xs` and writes the result to `out`. `xs` and `out` must be equal
/// in length.
///
/// `out` will be fully initialized after this function returns.
pub fn vec_exp(xs: &[f32], out: &mut [MaybeUninit<f32>]) {
    dispatch_map_op(xs, out, SimdExp {});
}

/// Variant of [`vec_exp`] that modifies elements in-place.
pub fn vec_exp_in_place(xs: &mut [f32]) {
    dispatch_map_op_in_place(xs, SimdExp {});
}

/// Compute sigmoid of each element in a SIMD vector.
///
/// ie. This computes `1. / (1. + exp(-x))`.
///
/// This has a maximum error of 4 ULPs compared to a reference implementation
/// using `1. / (1. + (-x).exp())`.
///
/// Safety: The caller must ensure the `SimdFloat` impl is usable on the current system.
#[inline(always)]
unsafe fn simd_sigmoid<S: SimdFloat>(x: S) -> S {
    // 1. + exp(-x)
    let denom = S::one().add(simd_exp(x.neg()));
    denom.reciprocal()
}

/// Computes the [sigmoid function][sigmoid], aka. the standard logistic function, `1. /
/// (1. + (-x).exp())`.
///
/// This is a scalar variant of [`vec_sigmoid`] that uses the same algorithm.
///
/// [sigmoid]: https://en.wikipedia.org/wiki/Logistic_function#Mathematical_properties
pub fn sigmoid(x: f32) -> f32 {
    // f32 is available on all systems
    unsafe { simd_sigmoid(x) }
}

struct SimdSigmoid {}
impl SimdUnaryOp for SimdSigmoid {
    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S {
        simd_sigmoid(x)
    }
}

/// Vectorized sigmoid function.
///
/// This is a vectorized version of [`sigmoid`] that computes the function for
/// each element in `xs` and writes the result to `out`. `xs` and `out` must be
/// equal in length.
///
/// `out` will be fully initialized after this function returns.
pub fn vec_sigmoid(xs: &[f32], out: &mut [MaybeUninit<f32>]) {
    dispatch_map_op(xs, out, SimdSigmoid {});
}

/// Variant of [`vec_sigmoid`] that modifies elements in-place.
pub fn vec_sigmoid_in_place(xs: &mut [f32]) {
    dispatch_map_op_in_place(xs, SimdSigmoid {});
}

/// Compute Sigmoid Linear Unit (SiLU) function.
///
/// This computes `x * sigmoid(x)` for all lanes in `x`.
#[inline(always)]
unsafe fn simd_silu<S: SimdFloat>(x: S) -> S {
    x.mul(simd_sigmoid(x))
}

struct SimdSilu {}
impl SimdUnaryOp for SimdSilu {
    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S {
        simd_silu(x)
    }
}

/// Vectorized Sigmoid Linear Unit (SiLU) function.
///
/// This computes `x * sigmoid(x)` for each element.
pub fn vec_silu(xs: &[f32], out: &mut [MaybeUninit<f32>]) {
    dispatch_map_op(xs, out, SimdSilu {});
}

/// Vectorized Sigmoid Linear Unit (SiLU) function.
///
/// This computes `x * sigmoid(x)` for each element.
pub fn vec_silu_in_place(xs: &mut [f32]) {
    dispatch_map_op_in_place(xs, SimdSilu {});
}

/// Sigmoid Linear Unit (SiLU) function. This computes `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    // Safety: f32 is available on all systems
    unsafe { simd_silu(x) }
}

struct SimdSwish {
    beta: f32,
}

impl SimdUnaryOp for SimdSwish {
    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S {
        let beta = S::splat(self.beta);
        x.mul(simd_sigmoid(x.mul(beta)))
    }
}

/// Vectorized Swish function.
///
/// This computes `x * sigmoid(beta * x)` for each element.
pub fn vec_swish(xs: &[f32], out: &mut [MaybeUninit<f32>], beta: f32) {
    dispatch_map_op(xs, out, SimdSwish { beta });
}

/// Vectorized Swish function.
///
/// This computes `x * sigmoid(beta * x)` for each element.
pub fn vec_swish_in_place(xs: &mut [f32], beta: f32) {
    dispatch_map_op_in_place(xs, SimdSwish { beta });
}

/// Swish function. This computes `x * sigmoid(beta * x)`.
pub fn swish(x: f32, beta: f32) -> f32 {
    // Safety: f32 is available on all systems
    let op = SimdSwish { beta };
    unsafe { op.eval(x) }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use crate::testing::{
        arange, benchmark_op, check_f32s_are_equal_ulps, check_with_all_f32s, AsUninit,
    };
    use crate::{exp, vec_exp, vec_sigmoid, vec_silu, vec_swish};

    // Maximum error of `vec_expf` compared to Rust standard library
    // implementation.
    const MAX_EXP_ERROR_ULPS: f32 = 1.0;

    // Maximum error of `vec_sigmoid` compared to reference implementation
    // below.
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

    /// Check the results of a SIMD implementation of a unary operator against
    /// a reference implementation.
    fn check_simd_vs_reference<
        F: Fn(&[f32], &mut [MaybeUninit<f32>]),
        R: Fn(f32) -> f32,
        I: Iterator<Item = f32>,
    >(
        simd_op: F,
        reference_op: R,
        max_error_ulps: f32,
        values: I,
    ) {
        let cases: Vec<_> = values.collect();
        let expected: Vec<_> = cases.iter().copied().map(reference_op).collect();
        let mut actual = cases.clone();

        simd_op(&cases, actual.as_mut_slice().as_uninit());

        let results = cases
            .iter()
            .zip(actual.iter().zip(expected.iter()))
            .map(|(x, (actual, expected))| (*x, *actual, *expected));
        check_f32s_are_equal_ulps(results, max_error_ulps);
    }

    #[test]
    fn test_expf() {
        // A few simple test cases, including "typical" +/-ve inputs with
        // |x| above/below ln2, zero and values below/above min/max cutoffs.
        let cases = [-2.0f32, -1., -0.5, 0.1, 0., 0.1, 0.5, 1., 2., -105., 105.];

        for case in cases {
            let expected = case.exp();
            let actual = exp(case);
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
    fn test_vec_expf() {
        check_simd_vs_reference(
            vec_exp,
            f32::exp,
            MAX_EXP_ERROR_ULPS,
            arange(-6., 6., 0.001f32),
        );
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_expf_exhaustive() {
        check_with_all_f32s(|x| (exp(x), x.exp()), MAX_EXP_ERROR_ULPS, "testing exp");
        check_with_all_f32s(
            |x| {
                let mut y = [0.; 1];
                vec_exp(&[x], y.as_mut().as_uninit());
                (y[0], x.exp())
            },
            MAX_EXP_ERROR_ULPS,
            "testing vec_expf",
        );
    }

    #[test]
    fn test_sigmoid() {
        check_simd_vs_reference(
            vec_sigmoid,
            reference_sigmoid,
            MAX_SIGMOID_ERROR_ULPS,
            arange(-6., 6., 0.001f32),
        );
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_sigmoid_exhaustive() {
        check_with_all_f32s(
            |x| {
                let mut y = [0.; 1];
                vec_sigmoid(&[x], y.as_mut().as_uninit());
                (y[0], reference_sigmoid(x))
            },
            MAX_SIGMOID_ERROR_ULPS,
            "testing vec_sigmoid",
        );
    }

    #[test]
    fn test_silu() {
        check_simd_vs_reference(
            vec_silu,
            reference_silu,
            MAX_SIGMOID_ERROR_ULPS,
            arange(-6., 6., 0.001f32),
        );
    }

    #[test]
    fn test_swish() {
        let beta = 1.7;
        check_simd_vs_reference(
            |src, dest| vec_swish(src, dest, beta),
            |x| reference_swish(x, beta),
            MAX_SIGMOID_ERROR_ULPS,
            arange(-6., 6., 0.001f32),
        )
    }

    #[test]
    #[ignore]
    fn bench_expf() {
        benchmark_op(
            |xs, ys| xs.iter().zip(ys.iter_mut()).for_each(|(x, y)| *y = x.exp()),
            vec_exp,
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
            vec_sigmoid,
        );
    }
}
