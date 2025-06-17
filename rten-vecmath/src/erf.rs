//! Error function ("erf") and closely related operations.

#![allow(clippy::excessive_precision)]

use std::f32::consts::SQRT_2;

use rten_simd::ops::{FloatOps, NumOps};
use rten_simd::{Isa, Simd, SimdUnaryOp};

use crate::exp::ReducedRangeExp;
use crate::tanh::Tanh;

/// Vectorized error function (erf).
///
/// The implementation uses an approximation from Abramowitz and Stegun,
/// see <https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions>.
///
/// This has a maximum absolute error of 6.631017e-7 when comparing to
/// `libm::erff` as a source of truth.
#[derive(Default)]
pub struct Erf {}

impl SimdUnaryOp<f32> for Erf {
    #[inline(always)]
    fn eval<I: Isa, S: Simd<Elem = f32, Isa = I>>(&self, isa: I, x: S) -> S {
        let ops = isa.f32();
        let x = x.same_cast();

        let neg_mask = ops.lt(x, ops.zero());

        let x = ops.abs(x);

        let p = ops.splat(0.3275911);

        // Coefficients for polynomial approximation.
        let a0 = ops.splat(0.254829592);
        let a1 = ops.splat(-0.284496736);
        let a2 = ops.splat(1.421413741);
        let a3 = ops.splat(-1.453152027);
        let a4 = ops.splat(1.061405429);

        // t = 1. / (1. + p * x);
        let t = ops.reciprocal(ops.mul_add(x, p, ops.one()));
        let at = ops.poly_eval(t, &[a0, a1, a2, a3, a4]);

        // exp_mx2 = e^(-x^2). `-(x^2)` is always <= 0, so we can use
        // reduced-range exp.
        let x_m2 = ops.neg(ops.mul(x, x));
        let exp_mx2 = ReducedRangeExp::apply(isa, x_m2);

        // y = 1. - at * exp_mx2;
        let y = ops.sub(ops.one(), ops.mul(at, exp_mx2));

        // Approximation is valid only for x >= 0. For negative values approximation
        // can be computed as -erf(-x).
        ops.select(ops.neg(y), y, neg_mask).same_cast()
    }
}

const SQRT_2_RCP: f32 = 1.0 / SQRT_2;

/// Computes the [GELU](https://onnx.ai/onnx/operators/onnx__Gelu.html)
/// function.
pub struct Gelu {}

impl SimdUnaryOp<f32> for Gelu {
    #[inline(always)]
    fn eval<I: Isa, S: Simd<Elem = f32, Isa = I>>(&self, isa: I, x: S) -> S {
        let ops = isa.f32();
        let x = x.same_cast();

        let half_x = ops.mul(x, ops.splat(0.5));
        let sqrt_2_rcp = ops.splat(SQRT_2_RCP);
        let y = ops.mul(x, sqrt_2_rcp);
        let y = ops.add(Erf::apply(isa, y), ops.splat(1.0));
        ops.mul(half_x, y).same_cast()
    }
}

// sqrt(2 / pi)
const SQRT_2_PI: f32 = 0.7978845608028654;

/// Approximate Gelu function.
///
/// See <https://onnx.ai/onnx/operators/onnx__Gelu.html>.
pub struct ApproxGelu {}

impl SimdUnaryOp<f32> for ApproxGelu {
    #[inline(always)]
    fn eval<I: Isa, S: Simd<Elem = f32, Isa = I>>(&self, isa: I, x: S) -> S {
        let ops = isa.f32();
        let x = x.same_cast();

        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let half_x = ops.mul(x, ops.splat(0.5));
        let x_cubed = ops.mul(ops.mul(x, x), x);
        let y = ops.mul_add(x_cubed, ops.splat(0.044715), x);
        let y = ops.mul(y, ops.splat(SQRT_2_PI));
        let y = Tanh::apply(isa, y);
        let y = ops.add(y, ops.splat(1.));
        let y = ops.mul(half_x, y);

        y.same_cast()
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::SimdUnaryOp;

    use super::{ApproxGelu, Erf, Gelu};
    use crate::testing::{
        AllF32s, AsUninit, Progress, arange, benchmark_op, check_f32s_are_equal_atol, triples,
    };

    fn reference_gelu(x: f32) -> f32 {
        0.5 * x * (1. + libm::erff(x / (2.0f32).sqrt()))
    }

    fn reference_approx_gelu(x: f32) -> f32 {
        let x_cubed = x * x * x;
        let approx_erf = ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x_cubed)).tanh();
        0.5 * x * (1. + approx_erf)
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
    fn test_approx_gelu() {
        let input: Vec<_> = arange(-6., 6., 0.001f32).collect();
        let mut actual = vec![0.; input.len()];
        let expected: Vec<_> = input.iter().copied().map(reference_approx_gelu).collect();

        ApproxGelu {}.map(&input, actual.as_mut_slice().as_uninit());

        let max_diff = 5e-7;
        check_f32s_are_equal_atol(triples(&input, &actual, &expected), max_diff);
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
