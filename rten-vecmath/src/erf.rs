//! Error function ("erf") and closely related operations.

#![allow(clippy::excessive_precision)]

use std::f32::consts::SQRT_2;

use rten_simd::safe::{Isa, Simd, SimdFloatOps, SimdOps, SimdUnaryOp};

use crate::Exp;

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

        // x = x.abs()
        let x = ops.select(ops.neg(x), x, neg_mask);

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

        // exp_mx2 = e^(-x^2)
        let x_m2 = ops.neg(ops.mul(x, x));
        let exp_mx2 = Exp::apply(isa, x_m2);

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

#[cfg(test)]
mod tests {
    use rten_simd::safe::SimdUnaryOp;

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
