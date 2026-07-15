use std::mem::MaybeUninit;

use rten_simd::functional::{simd_apply, simd_map};
use rten_simd::ops::{BitOps, FloatOps, NumOps};
use rten_simd::span::SrcDest;
use rten_simd::{Isa, Simd, SimdIterable, SimdOp, SimdUnaryOp};

use crate::exp::ReducedRangeExp;

/// Computes the [softmax][softmax] function over a slice of floats.
///
/// The implementation uses a three-pass approach for numerical stability.
/// See <https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html>.
/// and <https://arxiv.org/abs/2001.04438>.
///
/// [softmax]: <https://en.wikipedia.org/wiki/Softmax_function>
pub struct Softmax<'src, 'dst> {
    src_dest: SrcDest<'src, 'dst, f32>,
    flush_nans_to_zero: bool,
}

impl<'src, 'dst> Softmax<'src, 'dst> {
    /// Construct a softmax operation which reads `input` and writes to to
    /// `output`.
    #[track_caller]
    pub fn new(input: &'src [f32], output: &'dst mut [MaybeUninit<f32>]) -> Self {
        Softmax {
            src_dest: (input, output).into(),
            flush_nans_to_zero: false,
        }
    }

    /// Construct a softmax operation which updates `input` in place.
    pub fn new_mut(input: &'dst mut [f32]) -> Self
    where
        'dst: 'src,
    {
        Softmax {
            src_dest: input.into(),
            flush_nans_to_zero: false,
        }
    }

    /// Replace NaN values in the output with zeros.
    ///
    /// This option exists to changing handling of the case where the input
    /// values are all negative infinity. In that case the normal output would
    /// be NaN.
    ///
    /// In the context of attention operations which use negative infinity to
    /// represent masked token positions, it is preferable to produce zeros as
    /// the output if _all_ input positions are masked. See
    /// <https://github.com/pytorch/pytorch/issues/41508>.
    pub fn flush_nans_to_zero(mut self, flush: bool) -> Self {
        self.flush_nans_to_zero = flush;
        self
    }
}

impl<'dst> SimdOp for Softmax<'_, 'dst> {
    /// The normalized elements.
    type Output = &'dst mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();

        let max_val = max(ops, self.src_dest.src());

        // Compute `y = exp(x - max(x))` and `sum(y)`.
        let (dest, exp_sum) = exp_sum_minus_max(isa, self.src_dest, max_val);

        // Divide by `exp_sum`.
        let exp_sum = ops.splat(exp_sum);
        let inv_exp_sum = ops.reciprocal(exp_sum);
        const UNROLL: usize = 2;
        let zero = ops.zero();

        if self.flush_nans_to_zero {
            simd_apply::<_, _, _, UNROLL>(
                ops,
                dest,
                #[inline(always)]
                |x| {
                    let y = ops.mul(x, inv_exp_sum);
                    let not_nan = ops.eq(y, y);
                    ops.select(y, zero, not_nan)
                },
            );
        } else {
            simd_apply::<_, _, _, UNROLL>(
                ops,
                dest,
                #[inline(always)]
                |x| ops.mul(x, inv_exp_sum),
            );
        }

        dest
    }
}

/// Computes the log softmax function over a slice of floats.
///
/// This is conceptually `log(softmax(x))` which can be rewritten as
/// `log(exp(x) / sum(exp(x)))`.
pub struct LogSoftmax<'src, 'dst> {
    src_dest: SrcDest<'src, 'dst, f32>,
}

impl<'src, 'dst> LogSoftmax<'src, 'dst> {
    /// Construct a log softmax operation which reads `input` and writes to
    /// `output`.
    pub fn new(input: &'src [f32], output: &'dst mut [MaybeUninit<f32>]) -> Self {
        LogSoftmax {
            src_dest: (input, output).into(),
        }
    }

    /// Construct a log softmax operation which updates `input` in place.
    pub fn new_mut(input: &'dst mut [f32]) -> Self
    where
        'dst: 'src,
    {
        LogSoftmax {
            src_dest: input.into(),
        }
    }
}

impl<'dst> SimdOp for LogSoftmax<'_, 'dst> {
    /// The normalized elements.
    type Output = &'dst mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();

        // The maximum is subtracted from the input for numerical stability.
        // Log identities are used to simplify the result:
        //
        // log(exp(xi - xmax) / sum(exp(x - xmax)))
        //  = log(exp(xi - xmax)) - log(sum(exp(x - xmax)))
        //  = xi - xmax - log(sum(exp(x - xmax)))
        let max_val = max(ops, self.src_dest.src());

        // Compute `sum(exp(x - max(x)))`.
        let max_vec = ops.splat(max_val);
        let exp_sum = self.src_dest.src().simd_iter(ops).fold(
            ops.zero(),
            #[inline(always)]
            |exp_sum, x| {
                // Use faster `exp(x)` since input is known to be <= 0.
                let y = ReducedRangeExp::apply(isa, ops.sub(x, max_vec));
                ops.add(exp_sum, y)
            },
        );
        let exp_sum: f32 = exp_sum.to_array().into_iter().sum();

        // Compute `y = (x - x_max) - log(exp_sum)`.
        //
        // We use two separate subtractions inside the loop instead of computing
        // `x_max + log(exp_sum)` once and then using a single subtraction. This
        // reduces rounding error when `|x_max|` is large and `log(exp_sum)` is
        // small.
        let log_exp_sum = ops.splat(exp_sum.ln());
        simd_map(
            ops,
            self.src_dest,
            #[inline(always)]
            |x| ops.sub(ops.sub(x, max_vec), log_exp_sum),
        )
    }
}

/// Returns the maximum value in `xs`, or [`f32::MIN`] if `xs` is empty.
#[inline(always)]
fn max<O: FloatOps<f32>>(ops: O, xs: &[f32]) -> f32 {
    let max_val = xs.simd_iter(ops).fold_unroll::<4>(
        ops.splat(f32::MIN),
        #[inline(always)]
        |max, x| ops.max(max, x),
        #[inline(always)]
        |max, x| ops.max(max, x),
    );
    max_val
        .to_array()
        .into_iter()
        .fold(f32::MIN, |max, x| max.max(x))
}

/// Computes `y = exp(x - max(x))` and `sum(y)` in a single pass.
#[inline(always)]
fn exp_sum_minus_max<'dst, I: Isa>(
    isa: I,
    src_dest: SrcDest<'_, 'dst, f32>,
    max_val: f32,
) -> (&'dst mut [f32], f32) {
    let ops = isa.f32();

    let max_val = ops.splat(max_val);

    // *x = (*x - max_val).exp()
    let mut prev_exp_sum = ops.zero();
    let mut exp_sum = ops.zero();
    let dest = simd_map(
        ops,
        src_dest,
        #[inline(always)]
        |x| {
            // Use faster `exp(x)` since input is known to be <= 0.
            let y = ReducedRangeExp::apply(isa, ops.sub(x, max_val));
            prev_exp_sum = exp_sum;
            exp_sum = ops.add(exp_sum, y);
            y
        },
    );

    // Undo the last update to `exp_sum` for unused lanes.
    let remainder = dest.len() % ops.len();
    if remainder != 0 {
        let remainder_mask = ops.first_n_mask(remainder);
        exp_sum = ops.select(exp_sum, prev_exp_sum, remainder_mask);
    }
    let exp_sum = exp_sum.to_array().into_iter().sum();

    (dest, exp_sum)
}

#[cfg(test)]
mod tests {
    use rten_simd::SimdOp;

    use super::{LogSoftmax, Softmax};
    use crate::testing::{AsUninit, benchmark_op, check_f32s_are_equal_ulps, triples};

    fn reference_log_softmax(xs: &[f32], ys: &mut [f32]) {
        let max = xs.iter().copied().fold(f32::MIN, |max, x| max.max(x));
        let log_exp_sum = xs
            .iter()
            .fold(0., |exp_sum: f32, x| exp_sum + (x - max).exp())
            .ln();
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = (*x - max) - log_exp_sum;
        }
    }

    fn reference_softmax(xs: &[f32], ys: &mut [f32]) {
        let max = xs.iter().copied().fold(f32::MIN, |max, x| max.max(x));

        let mut exp_sum = 0.;
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = (*x - max).exp();
            exp_sum += *y;
        }

        for el in ys.iter_mut() {
            *el /= exp_sum;
        }
    }

    #[test]
    fn test_softmax() {
        // Test against reference values.
        let input = vec![0.1634, 0.8647, 0.6401, 0.8265, 0.0560, 0.2304];
        let expected = &([
            0.11715934, 0.23623686, 0.18871443, 0.2273828, 0.10522857, 0.12527795,
        ]);
        let mut actual = vec![0.; input.len()];

        Softmax::new(&input, actual.as_mut_slice().as_uninit()).dispatch();
        check_f32s_are_equal_ulps(triples(&input, &actual, expected), 1. /* max ULPs */);

        // Test against reference implementation for various lengths.
        for len in 1..20 {
            let input: Vec<f32> = (0..len).map(|x| x as f32 + 0.1).collect();
            let mut expected = vec![0.; input.len()];
            reference_softmax(&input, &mut expected);

            let mut actual = vec![0.; input.len()];
            Softmax::new(&input, actual.as_mut_slice().as_uninit()).dispatch();

            check_f32s_are_equal_ulps(triples(&input, &actual, &expected), 3. /* max ULPs */);
        }
    }

    #[test]
    fn test_softmax_flush_nans_to_zero() {
        let mut input = [f32::NEG_INFINITY; 3];
        Softmax::new_mut(&mut input).dispatch();
        assert!(input.iter().all(|x| x.is_nan()));

        let mut input = [f32::NEG_INFINITY; 3];
        Softmax::new_mut(&mut input)
            .flush_nans_to_zero(true)
            .dispatch();
        assert_eq!(input, [0.; 3]);
    }

    #[test]
    fn test_log_softmax() {
        // Test against reference implementation for various lengths.
        for len in 1..20 {
            let input: Vec<f32> = (0..len).map(|x| x as f32 + 0.1).collect();
            let mut expected = vec![0.; input.len()];
            reference_log_softmax(&input, &mut expected);

            let mut actual = vec![0.; input.len()];
            LogSoftmax::new(&input, actual.as_mut_slice().as_uninit()).dispatch();

            check_f32s_are_equal_ulps(triples(&input, &actual, &expected), 3. /* max ULPs */);
        }
    }

    #[test]
    fn test_log_softmax_in_place() {
        let input: Vec<f32> = (0..32).map(|x| x as f32 * 0.5 - 8.).collect();
        let mut expected = vec![0.; input.len()];
        reference_log_softmax(&input, &mut expected);

        let mut actual = input.clone();
        LogSoftmax::new_mut(&mut actual).dispatch();

        check_f32s_are_equal_ulps(triples(&input, &actual, &expected), 3. /* max ULPs */);
    }

    #[test]
    fn test_log_softmax_sums_to_one() {
        // `exp(log_softmax(x))` is a probability distribution, so it must sum
        // to one regardless of the input scale.
        for scale in [1e-3, 1., 10., 100.] {
            let input: Vec<f32> = (0..64).map(|x| (x as f32 - 32.) * scale).collect();
            let mut actual = input.clone();
            LogSoftmax::new_mut(&mut actual).dispatch();

            let sum: f32 = actual.iter().map(|x| x.exp()).sum();
            assert!((sum - 1.).abs() < 1e-4, "scale {scale} sum {sum}");
        }
    }

    #[test]
    #[ignore]
    fn bench_softmax() {
        benchmark_op(reference_softmax, |src, dest| {
            Softmax::new(src, dest).dispatch();
        });
    }

    #[test]
    #[ignore]
    fn bench_log_softmax() {
        benchmark_op(reference_log_softmax, |src, dest| {
            LogSoftmax::new(src, dest).dispatch();
        });
    }
}
