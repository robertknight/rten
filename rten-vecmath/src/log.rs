//! Vectorized version of the natural logarithm and closely related functions.

#![allow(clippy::excessive_precision)]

use rten_simd::ops::{BitOps, FloatOps, IntOps, MaskOps, NumOps, ToFloat};
use rten_simd::{Isa, Simd, SimdUnaryOp};

// `ln(2)` split into large and small parts. The large part has zeros in the low
// bits of the mantissa, so multiplying it by a small integer is exact.
const LN2_HI: f32 = 0.693359375;
const LN2_LO: f32 = -2.12194440e-4;

// Coefficients of the polynomial `P` used to approximate `ln(1 + f)` as
// `f - f^2/2 + f^3 * P(f)` for `f` in `[sqrt(2)/2 - 1, sqrt(2) - 1]`.
//
// These, and the split of `ln(2)` above, are from the single precision `logf`
// in the Cephes math library (<https://netlib.org/cephes/>,
// <https://github.com/jeremybarnes/cephes/blob/master/single/logf.c>).
const LN_POLY_0: f32 = 3.3333331174E-1; // ~ 1/3
const LN_POLY_1: f32 = -2.4999993993E-1; // ~ -1/4
const LN_POLY_2: f32 = 2.0000714765E-1; // ~ 1/5
const LN_POLY_3: f32 = -1.6668057665E-1; // ~ -1/6
const LN_POLY_4: f32 = 1.4249322787E-1; // ~ 1/7
const LN_POLY_5: f32 = -1.2420140846E-1; // ~ -1/8
const LN_POLY_6: f32 = 1.1676998740E-1;
const LN_POLY_7: f32 = -1.1514610310E-1;
const LN_POLY_8: f32 = 7.0376836292E-2;

// Power of two by which subnormal inputs are scaled to make them normal. The
// smallest subnormal is `2**-149` and the smallest normal value is `2**-126`.
const SUBNORMAL_SCALE_LOG2: u32 = 23;

/// Compute `ln(x)` for lanes containing positive, normal floats.
///
/// Results for zero, negative, infinite, NaN and subnormal inputs are
/// unspecified.
///
/// Method outline:
///
///  1. Decompose the input as `x = 2**k * m`, where `k` is an integer and `m`
///     is in `[sqrt(2)/2, sqrt(2))`, and use `ln(a * b) = ln(a) + ln(b)`:
///
///     ```text
///        ln(x) = k * ln2 + ln(m)
///     ```
///
///  2. Compute `ln(m) = ln(1 + f)`, where `f = m - 1` is small, using a
///     polynomial approximation.
#[inline(always)]
fn ln_normal<I: Isa>(isa: I, x: I::F32) -> I::F32 {
    let ops = isa.f32();
    let int_ops = isa.i32();

    // Split the input into the exponent `k` and mantissa `m` in `[1, 2)`.
    let bits: I::I32 = x.reinterpret_cast();
    let k = int_ops.sub(int_ops.shift_right::<23>(bits), int_ops.splat(127));
    let m: I::F32 = int_ops
        .or(
            int_ops.and(bits, int_ops.splat(0x007fffff)),
            int_ops.splat(0x3f800000),
        )
        .reinterpret_cast();

    // Shift the mantissa into `[sqrt(2)/2, sqrt(2))` so that `f = m - 1` is
    // centered on zero. Both the halving and the subtraction of one are exact.
    let m_large = ops.ge(m, ops.splat(std::f32::consts::SQRT_2));
    let m = ops.select(ops.mul(m, ops.splat(0.5)), m, m_large);
    let f = ops.sub(m, ops.one());

    let k = int_ops.to_float(k);
    let k = ops.select(ops.add(k, ops.one()), k, m_large);

    // Approximate `ln(1 + f) = f - f**2 / 2 + f**3 * P(f)`.
    let mut poly = ops.splat(LN_POLY_8);
    poly = ops.mul_add(poly, f, ops.splat(LN_POLY_7));
    poly = ops.mul_add(poly, f, ops.splat(LN_POLY_6));
    poly = ops.mul_add(poly, f, ops.splat(LN_POLY_5));
    poly = ops.mul_add(poly, f, ops.splat(LN_POLY_4));
    poly = ops.mul_add(poly, f, ops.splat(LN_POLY_3));
    poly = ops.mul_add(poly, f, ops.splat(LN_POLY_2));
    poly = ops.mul_add(poly, f, ops.splat(LN_POLY_1));
    poly = ops.mul_add(poly, f, ops.splat(LN_POLY_0));

    let f_sqr = ops.mul(f, f);
    let y = ops.mul(ops.mul(poly, f), f_sqr);

    // Reconstruct `ln(x) = k * ln2 + ln(1 + f)`, adding terms from smallest to
    // largest to limit rounding error.
    let y = ops.mul_add(k, ops.splat(LN2_LO), y);
    let y = ops.mul_add(f_sqr, ops.splat(-0.5), y);
    let y = ops.add(f, y);
    ops.mul_add(k, ops.splat(LN2_HI), y)
}

/// Vectorized natural logarithm.
///
/// This has a maximum error of 1 ULP compared to `f32::ln` in the Rust
/// standard library.
#[derive(Default)]
pub struct Ln {}

impl SimdUnaryOp<f32> for Ln {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();
        let masks = isa.m32();

        // If every lane is in the domain of `ln_normal`, none of the fixups
        // below would change the result, so skip them. This is the common case
        // and saves around a quarter of the work.
        let in_domain = masks.and(
            ops.ge(x, ops.splat(f32::MIN_POSITIVE)),
            ops.le(x, ops.splat(f32::MAX)),
        );
        if masks.all(in_domain) {
            return ln_normal(isa, x);
        }

        // Scale subnormal inputs into the normal range, since `ln_normal` can't
        // decompose them, and subtract `ln(2**23)` from the result afterwards.
        // The mask also selects zero and negative inputs, but results for those
        // are replaced below.
        let subnormal = ops.lt(x, ops.splat(f32::MIN_POSITIVE));
        let scale = (1 << SUBNORMAL_SCALE_LOG2) as f32;
        let x_scaled = ops.select(ops.mul(x, ops.splat(scale)), x, subnormal);

        let y = ln_normal(isa, x_scaled);
        let y = ops.select(
            ops.sub(
                y,
                ops.splat(SUBNORMAL_SCALE_LOG2 as f32 * std::f32::consts::LN_2),
            ),
            y,
            subnormal,
        );

        // Handle inputs outside the domain of `ln_normal`. Negative inputs, as
        // well as NaN, produce NaN.
        let y = ops.select(y, ops.splat(f32::NAN), ops.ge(x, ops.zero()));
        let y = ops.select(ops.splat(f32::NEG_INFINITY), y, ops.eq(x, ops.zero()));
        ops.select(
            ops.splat(f32::INFINITY),
            y,
            ops.eq(x, ops.splat(f32::INFINITY)),
        )
    }
}

/// Compute `ln(1 + x)` for lanes containing finite values greater than -1.
///
/// Results for other inputs are unspecified. Callers which need to support them
/// should use [`Ln1p`], which handles them.
#[inline(always)]
pub(crate) fn ln_1p_finite<I: Isa>(isa: I, x: I::F32) -> I::F32 {
    let ops = isa.f32();
    let one = ops.one();

    // Computing `ln(1 + x)` directly loses the low bits of `x` when `|x|` is
    // small. Compensate by scaling the result by the ratio between `x` and the
    // value that was actually added to one. See Goldberg, "What Every Computer
    // Scientist Should Know About Floating-Point Arithmetic", theorem 4.
    let u = ops.add(one, x);
    let d = ops.sub(u, one);
    let y = ops.mul(ln_normal(isa, u), ops.div(x, d));

    // When `1 + x` rounds to one, `ln(1 + x) == x` and the scaling above
    // divides by zero.
    ops.select(x, y, ops.eq(u, one))
}

/// Vectorized natural logarithm of one plus the input, `ln(1 + x)`.
///
/// This has a maximum error of 2 ULPs compared to `f32::ln_1p` in the Rust
/// standard library.
#[derive(Default)]
pub struct Ln1p {}

impl SimdUnaryOp<f32> for Ln1p {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();
        let y = ln_1p_finite(isa, x);

        // Handle inputs outside the domain of `ln_1p_finite`. Inputs below -1,
        // as well as NaN, produce NaN.
        let y = ops.select(y, ops.splat(f32::NAN), ops.ge(x, ops.splat(-1.)));
        let y = ops.select(ops.splat(f32::NEG_INFINITY), y, ops.eq(x, ops.splat(-1.)));
        ops.select(
            ops.splat(f32::INFINITY),
            y,
            ops.eq(x, ops.splat(f32::INFINITY)),
        )
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::SimdUnaryOp;

    use crate::testing::{
        AllF32s, Tolerance, UnaryOpTester, arange, benchmark_op, check_f32s_are_equal_ulps, triples,
    };
    use crate::{Ln, Ln1p};

    // Maximum error of `Ln` compared to `f32::ln`.
    const MAX_LN_ERROR_ULPS: f32 = 1.0;

    // Maximum error of `Ln1p` compared to `f32::ln_1p`.
    const MAX_LN_1P_ERROR_ULPS: f32 = 2.0;

    #[test]
    fn test_ln_special_values() {
        let cases = [
            (1., 0.),
            (0., f32::NEG_INFINITY),
            (-0., f32::NEG_INFINITY),
            (-1., f32::NAN),
            (f32::INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::NAN),
            (f32::NAN, f32::NAN),
        ];

        let op = Ln {};
        for (x, expected) in cases {
            let actual = op.scalar_eval(x);
            if expected.is_nan() {
                assert!(actual.is_nan(), "expected NaN for x = {x}, got {actual}");
            } else {
                assert_eq!(actual, expected, "mismatch for x = {x}");
            }
        }
    }

    #[test]
    fn test_ln_mixed_lanes() {
        // Interleave in-domain values with each kind of input needing a fixup,
        // so that vectors of any width contain a mix of the two.
        let specials = [
            0.,
            -0.,
            -1.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(1),
            f32::MIN_POSITIVE / 2.,
        ];
        let mut input = Vec::new();
        for (i, special) in specials.into_iter().enumerate() {
            for j in 0..7 {
                input.push(1. + (i * 7 + j) as f32);
            }
            input.push(special);
        }
        let expected: Vec<f32> = input.iter().map(|x| x.ln()).collect();

        let mut actual = input.clone();
        Ln {}.map_mut(&mut actual);

        check_f32s_are_equal_ulps(triples(&input, &actual, &expected), MAX_LN_ERROR_ULPS);
    }

    #[test]
    fn test_ln() {
        // Subnormal values, which have to be scaled into the normal range.
        let test = UnaryOpTester {
            reference: f32::ln,
            simd: Ln {},
            range: arange(
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                f32::from_bits(1) * 1e5,
            ),
            tolerance: Tolerance::Ulp(MAX_LN_ERROR_ULPS),
        };
        test.run();

        // Typical values.
        let test = UnaryOpTester {
            reference: f32::ln,
            simd: Ln {},
            range: arange(1e-6, 100., 0.001),
            tolerance: Tolerance::Ulp(MAX_LN_ERROR_ULPS),
        };
        test.run();

        // Large values.
        let test = UnaryOpTester {
            reference: f32::ln,
            simd: Ln {},
            range: arange(0., 1e30, 1e26),
            tolerance: Tolerance::Ulp(MAX_LN_ERROR_ULPS),
        };
        test.run();
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_ln_exhaustive() {
        let test = UnaryOpTester {
            reference: f32::ln,
            simd: Ln {},
            range: AllF32s::new(),
            tolerance: Tolerance::Ulp(MAX_LN_ERROR_ULPS),
        };
        test.run_with_progress();
    }

    #[test]
    #[ignore]
    fn bench_ln() {
        // Note that half of the benchmark inputs are negative. That makes the
        // reference return NaN early, and puts every vector on this kernel's
        // slow path, so this ratio understates the speedup for typical inputs.
        benchmark_op(
            |xs, ys| xs.iter().zip(ys.iter_mut()).for_each(|(x, y)| *y = x.ln()),
            |xs, ys| {
                Ln {}.map(xs, ys);
            },
        );
    }

    #[test]
    fn test_ln_1p_special_values() {
        let cases = [
            (0., 0.),
            (-0., -0.),
            (-1., f32::NEG_INFINITY),
            (-1.5, f32::NAN),
            (f32::INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::NAN),
            (f32::NAN, f32::NAN),
        ];

        let op = Ln1p {};
        for (x, expected) in cases {
            let actual = op.scalar_eval(x);
            if expected.is_nan() {
                assert!(actual.is_nan(), "expected NaN for x = {x}, got {actual}");
            } else {
                assert_eq!(actual, expected, "mismatch for x = {x}");
                assert_eq!(
                    actual.is_sign_negative(),
                    expected.is_sign_negative(),
                    "sign mismatch for x = {x}"
                );
            }
        }
    }

    #[test]
    fn test_ln_1p() {
        // Values close to zero, where the result is dominated by the input.
        let test = UnaryOpTester {
            reference: f32::ln_1p,
            simd: Ln1p {},
            range: arange(-1e-6, 1e-6, 1e-9),
            tolerance: Tolerance::Ulp(MAX_LN_1P_ERROR_ULPS),
        };
        test.run();

        // Values close to -1, where `ln(1 + x)` tends to -infinity.
        let test = UnaryOpTester {
            reference: f32::ln_1p,
            simd: Ln1p {},
            range: arange(-0.9999, -0.99, 1e-7),
            tolerance: Tolerance::Ulp(MAX_LN_1P_ERROR_ULPS),
        };
        test.run();

        // Typical values.
        let test = UnaryOpTester {
            reference: f32::ln_1p,
            simd: Ln1p {},
            range: arange(-0.99, 100., 0.001),
            tolerance: Tolerance::Ulp(MAX_LN_1P_ERROR_ULPS),
        };
        test.run();

        // Large values.
        let test = UnaryOpTester {
            reference: f32::ln_1p,
            simd: Ln1p {},
            range: arange(0., 1e30, 1e26),
            tolerance: Tolerance::Ulp(MAX_LN_1P_ERROR_ULPS),
        };
        test.run();
    }

    #[test]
    #[ignore] // Ignored by default due to long runtime
    fn test_ln_1p_exhaustive() {
        let test = UnaryOpTester {
            reference: f32::ln_1p,
            simd: Ln1p {},
            range: AllF32s::new(),
            tolerance: Tolerance::Ulp(MAX_LN_1P_ERROR_ULPS),
        };
        test.run_with_progress();
    }

    #[test]
    #[ignore]
    fn bench_ln_1p() {
        benchmark_op(
            |xs, ys| {
                xs.iter()
                    .zip(ys.iter_mut())
                    .for_each(|(x, y)| *y = x.ln_1p())
            },
            |xs, ys| {
                Ln1p {}.map(xs, ys);
            },
        );
    }
}
