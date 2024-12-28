//! Error function ("erf") and closely related operations.

#![allow(clippy::excessive_precision)]

use std::f32::consts::SQRT_2;

use rten_simd::dispatch::SimdUnaryOp;
use rten_simd::SimdFloat;

use crate::exp::simd_exp;

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

/// Vectorized error function (erf).
///
/// The implementation uses an approximation from Abramowitz and Stegun,
/// see <https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions>.
///
/// This has a maximum absolute error of 6.631017e-7 when comparing to
/// `libm::erff` as a source of truth.
pub struct Erf {}

impl SimdUnaryOp for Erf {
    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S {
        simd_erf(x)
    }
}

const SQRT_2_RCP: f32 = 1.0 / SQRT_2;

#[inline(always)]
unsafe fn simd_gelu<S: SimdFloat>(x: S) -> S {
    let half_x = x.mul(S::splat(0.5));
    let sqrt_2_rcp = S::splat(SQRT_2_RCP);
    let y = x.mul(sqrt_2_rcp);
    let y = simd_erf(y).add(S::splat(1.0));
    half_x.mul(y)
}

/// Computes the [GELU](https://onnx.ai/onnx/operators/onnx__Gelu.html)
/// function.
pub struct Gelu {}
impl SimdUnaryOp for Gelu {
    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S {
        simd_gelu(x)
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::dispatch::SimdUnaryOp;

    use super::{Erf, Gelu};
    use crate::testing::{
        arange, benchmark_op, check_f32s_are_equal_atol, triples, AllF32s, AsUninit, Progress,
    };

    fn reference_gelu(x: f32) -> f32 {
        0.5 * x * (1. + libm::erff(x / (2.0f32).sqrt()))
    }

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

        Erf {}.map(&input, actual.as_mut_slice().as_uninit());

        check_f32s_are_equal_atol(triples(&input, &actual, &expected), MAX_EXPECTED_DIFF);
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_erf_exhaustive() {
        let mut max_diff = 0.0f32;
        let op = Erf {};
        for x in Progress::wrap(AllF32s::new(), "testing erf") {
            let (actual, expected) = (op.scalar_eval(x), libm::erff(x));
            let diff = (actual - expected).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(max_diff <= MAX_EXPECTED_DIFF);
    }

    #[test]
    fn test_gelu() {
        let input: Vec<_> = arange(-6., 6., 0.001f32).collect();
        let mut actual = vec![0.; input.len()];
        let expected: Vec<_> = input.iter().copied().map(reference_gelu).collect();

        Gelu {}.map(&input, actual.as_mut_slice().as_uninit());

        // Gelu uses erf, so its error is constrained by this.
        check_f32s_are_equal_atol(triples(&input, &actual, &expected), MAX_EXPECTED_DIFF);
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
            |xs, ys| Erf {}.map(xs, ys),
        );
    }
}
