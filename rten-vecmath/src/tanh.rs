#![allow(clippy::excessive_precision)]

use rten_simd::dispatch::SimdUnaryOp;
use rten_simd::SimdFloat;

use crate::Exp;

/// Vectorized tanh implementation.
pub struct Tanh {}

impl SimdUnaryOp for Tanh {
    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S {
        let x_negative = x.le(S::zero());
        let abs_x = x.abs();

        // Cutoff beyond which `f32::tanh(x)` saturates at +/- 1.0.
        let x_cutoff = abs_x.ge(S::splat(9.02));

        // tanh(x) ~ x when |x| is very small.
        let x_tiny = abs_x.le(S::splat(0.0004));

        // Threshold below which `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)` method
        // produces errors >= 2 ULP.
        let x_small = abs_x.le(S::splat(0.55));

        // For small x, use polynomial approximation. Computed using Sollya with
        // `P = fpminimax(f, [|1, 3, 5, 7, 9|], [|SG...|], [0, 0.6])`.
        const P1: f32 = 0.999999940395355224609375;
        const P3: f32 = -0.33332359790802001953125;
        const P5: f32 = 0.13310669362545013427734375;
        const P7: f32 = -5.21197654306888580322265625e-2;
        const P9: f32 = 1.5497927553951740264892578125e-2;

        let p1 = S::splat(P1);
        let p3 = S::splat(P3);
        let p5 = S::splat(P5);
        let p7 = S::splat(P7);
        let p9 = S::splat(P9);

        let x_sqr = x.mul(x);
        let y_small = p9.mul_add(x_sqr, p7);
        let y_small = y_small.mul_add(x_sqr, p5);
        let y_small = y_small.mul_add(x_sqr, p3);
        let y_small = y_small.mul_add(x_sqr, p1);
        let y_small = y_small.mul(abs_x);

        // For medium x, compute `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`.
        let x2 = abs_x.mul(S::splat(2.0));
        let exp_2x = Exp::apply(x2);
        let exp_2x_m1 = exp_2x.sub(S::one());
        let exp_2x_p1 = exp_2x.add(S::one());
        let y_medium = exp_2x_m1.div(exp_2x_p1);

        // Select output to use depending on |x|.
        let y = S::one().select(y_medium, x_cutoff);
        let y = y_small.select(y, x_small);
        let y = abs_x.select(y, x_tiny);

        // Flip sign if input was negative.
        y.neg().select(y, x_negative)
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::dispatch::SimdUnaryOp;

    use crate::testing::{
        arange, benchmark_op, check_f32s_are_equal_ulps, check_with_all_f32s, AsUninit,
    };
    use crate::Tanh;

    // Maximum error of `vec_tanh` compared to `f32::tanh`.
    const MAX_TANH_ERROR_ULPS: f32 = 3.0;

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_tanh_exhaustive() {
        check_with_all_f32s(
            |x| {
                let mut y = [0.; 1];
                Tanh {}.map(&[x], y.as_mut().as_uninit());
                (y[0], x.tanh())
            },
            MAX_TANH_ERROR_ULPS,
            "testing vec_tanh",
        );
    }

    #[test]
    fn test_tanh() {
        let cases: Vec<f32> = arange(-8., 8., 0.001f32).collect();
        let expected: Vec<_> = cases.iter().copied().map(|x| x.tanh()).collect();
        let mut actual = cases.clone();
        Tanh {}.map(&cases, actual.as_mut_slice().as_uninit());

        let results = cases
            .iter()
            .zip(actual.iter().zip(expected.iter()))
            .map(|(x, (actual, expected))| (*x, *actual, *expected));
        check_f32s_are_equal_ulps(results, MAX_TANH_ERROR_ULPS);
    }

    #[test]
    #[ignore]
    fn bench_tanh() {
        benchmark_op(
            |xs, ys| {
                xs.iter()
                    .zip(ys.iter_mut())
                    .for_each(|(x, y)| *y = x.tanh())
            },
            |xs, ys| Tanh {}.map(xs, ys),
        );
    }
}
