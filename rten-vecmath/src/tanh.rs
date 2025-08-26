#![allow(clippy::excessive_precision)]

use rten_simd::ops::{FloatOps, NumOps};
use rten_simd::{Isa, SimdUnaryOp};

use crate::Exp;

/// Computes the hyperbolic tangent function.
#[derive(Default)]
pub struct Tanh {}

impl SimdUnaryOp<f32> for Tanh {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();

        let x_negative = ops.le(x, ops.zero());
        let abs_x = ops.abs(x);

        // Cutoff beyond which `f32::tanh(x)` saturates at +/- 1.0.
        let x_cutoff = ops.ge(abs_x, ops.splat(9.02));

        // tanh(x) ~ x when |x| is very small.
        let x_tiny = ops.le(abs_x, ops.splat(0.0004));

        // Threshold below which `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)` method
        // produces errors >= 2 ULP.
        let x_small = ops.le(abs_x, ops.splat(0.55));

        // For small x, use polynomial approximation. Computed using Sollya with
        // `P = fpminimax(f, [|1, 3, 5, 7, 9|], [|SG...|], [0, 0.6])`.
        const P1: f32 = 0.999999940395355224609375;
        const P3: f32 = -0.33332359790802001953125;
        const P5: f32 = 0.13310669362545013427734375;
        const P7: f32 = -5.21197654306888580322265625e-2;
        const P9: f32 = 1.5497927553951740264892578125e-2;

        let p1 = ops.splat(P1);
        let p3 = ops.splat(P3);
        let p5 = ops.splat(P5);
        let p7 = ops.splat(P7);
        let p9 = ops.splat(P9);

        let x_sqr = ops.mul(x, x);
        let y_small = ops.mul_add(p9, x_sqr, p7);
        let y_small = ops.mul_add(y_small, x_sqr, p5);
        let y_small = ops.mul_add(y_small, x_sqr, p3);
        let y_small = ops.mul_add(y_small, x_sqr, p1);
        let y_small = ops.mul(y_small, abs_x);

        // For medium x, compute `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`.
        let x2 = ops.mul(abs_x, ops.splat(2.0));
        let exp_2x = Exp::apply(isa, x2);
        let exp_2x_m1 = ops.sub(exp_2x, ops.one());
        let exp_2x_p1 = ops.add(exp_2x, ops.one());
        let y_medium = ops.div(exp_2x_m1, exp_2x_p1);

        // Select output to use depending on |x|.
        let y = ops.select(ops.one(), y_medium, x_cutoff);
        let y = ops.select(y_small, y, x_small);
        let y = ops.select(abs_x, y, x_tiny);

        // Flip sign if input was negative.
        ops.select(ops.neg(y), y, x_negative)
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::SimdUnaryOp;

    use crate::Tanh;
    use crate::testing::{AllF32s, Tolerance, UnaryOpTester, arange, benchmark_op};

    // Maximum error of `Tanh` compared to `f32::tanh`.
    const MAX_TANH_ERROR_ULPS: f32 = 3.0;

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_tanh_exhaustive() {
        let test = UnaryOpTester {
            reference: f32::tanh,
            simd: Tanh {},
            range: AllF32s::new(),
            tolerance: Tolerance::Ulp(MAX_TANH_ERROR_ULPS),
        };
        test.run_with_progress();
    }

    #[test]
    fn test_tanh() {
        let test = UnaryOpTester {
            reference: f32::tanh,
            simd: Tanh {},
            range: arange(-8., 8., 0.001),
            tolerance: Tolerance::Ulp(MAX_TANH_ERROR_ULPS),
        };
        test.run();
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
