#![allow(clippy::excessive_precision)]

use std::mem::MaybeUninit;

use crate::dispatch_unary_op;
use crate::exp::simd_exp;
use crate::simd_vec::SimdFloat;

/// Computes the [error function](https://en.wikipedia.org/wiki/Error_function).
pub fn erf(x: f32) -> f32 {
    // Safety: f32 is available on all platforms
    unsafe { simd_erf(x) }
}

/// Vectorized implementation of error function (erf).
///
/// The implementation uses an approximation from Abramowitz and Stegun,
/// see https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions.
///
/// This has a maximum absolute error of 6.631017e-7 when comparing to
/// `libm::erff` as a source of truth.
///
/// Safety: The caller must ensure the `SimdFloat` impl is usable on the current system.
#[inline(always)]
unsafe fn simd_erf<S: SimdFloat>(x: S) -> S {
    let neg_mask = x.lt(S::zero());

    // x = x.abs()
    let x = x.blend(x.neg(), neg_mask);

    let p = S::splat(0.3275911);

    // Coefficients for polynomial approximation.
    let a0 = S::splat(0.254829592);
    let a1 = S::splat(-0.284496736);
    let a2 = S::splat(1.421413741);
    let a3 = S::splat(-1.453152027);
    let a4 = S::splat(1.061405429);

    // t = 1. / (1. + p * x);
    let t = x.mul_add(p, S::one()).reciprocal();
    let at = t.poly_eval(&[a0, a1, a2, a3, a4]);

    // exp_mx2 = e^(-x^2)
    let x_m2 = x.mul(x).neg();
    let exp_mx2 = simd_exp(x_m2);

    // y = 1. - at * exp_mx2;
    let y = S::one().sub(at.mul(exp_mx2));

    // Approximation is valid only for x >= 0. For negative values approximation
    // can be computed as -erf(-x).
    y.blend(y.neg(), neg_mask)
}

/// Vectorized error function.
///
/// This is a vectorized version of [erf] that computes the function for each
/// element in `xs` and writes the result to `out`. `xs` and `out` must be equal
/// in length.
pub fn vec_erf(xs: &[f32], out: &mut [MaybeUninit<f32>]) {
    dispatch_unary_op!(xs, out, simd_erf, erf);
}

/// Variant of [vec_erf] that modifies elements in-place.
pub fn vec_erf_in_place(xs: &mut [f32]) {
    dispatch_unary_op!(xs, simd_erf, erf);
}

#[cfg(test)]
mod tests {
    use super::{erf, vec_erf};

    use crate::testing::{
        arange, benchmark_op, check_f32s_are_equal_atol, triples, AllF32s, AsUninit, Progress,
    };

    // Maximum difference between our erf function and `libm::erf` found
    // through an exhaustive test.
    //
    // We use a max-difference test rather than comparing ULPs because the ULP
    // difference is large when the input is near zero, but the absolute
    // difference is still small enough to be acceptable for the practical uses
    // this library is most concerned with.
    const MAX_EXPECTED_DIFF: f32 = 6.631017e-7;

    #[test]
    fn test_erf() {
        // This range is sufficient to cover the regions where the function
        // is not saturated and where it is saturated at +/- 1.
        let input: Vec<_> = arange(-6., 6., 0.001f32).collect();
        let mut actual = vec![0.; input.len()];
        let expected: Vec<_> = input.iter().copied().map(libm::erff).collect();

        vec_erf(&input, actual.as_mut_slice().as_uninit());

        check_f32s_are_equal_atol(triples(&input, &actual, &expected), MAX_EXPECTED_DIFF);
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_erf_exhaustive() {
        let mut max_diff = 0.0f32;
        for x in Progress::wrap(AllF32s::new(), "testing erf") {
            let (actual, expected) = (erf(x), libm::erff(x));
            let diff = (actual - expected).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(max_diff <= MAX_EXPECTED_DIFF);
    }

    #[test]
    #[ignore]
    fn bench_erf() {
        benchmark_op(
            |xs, ys| {
                xs.iter()
                    .zip(ys.iter_mut())
                    .for_each(|(x, y)| *y = libm::erff(*x))
            },
            vec_erf,
        );
    }
}
