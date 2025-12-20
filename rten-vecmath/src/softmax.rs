use std::mem::MaybeUninit;

use rten_simd::functional::{simd_apply, simd_map};
use rten_simd::ops::{FloatOps, NumOps};
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

        let max_val = self.src_dest.src().simd_iter(ops).fold_unroll::<4>(
            ops.splat(f32::MIN),
            #[inline(always)]
            |max, x| ops.max(max, x),
            #[inline(always)]
            |max, x| ops.max(max, x),
        );
        let max_val = max_val
            .to_array()
            .into_iter()
            .fold(f32::MIN, |max, x| max.max(x));

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

/// Computes the [softmax][softmax] function over a slice of floats using the
/// online algorithm.
///
/// This implementation uses a two-pass approach based on the "Online
/// normalizer calculation for softmax" paper (Milakov & Gimelshein, 2018,
/// <https://arxiv.org/abs/1805.02867>). The first pass computes the maximum
/// and sum of exponentials simultaneously by rescaling the running sum when
/// a new maximum is found. The second pass computes the final softmax values.
///
/// This reduces memory bandwidth requirements compared to the three-pass
/// approach in [Softmax], at the cost of using full-range `exp` instead of
/// reduced-range `exp` in the first pass.
///
/// [softmax]: <https://en.wikipedia.org/wiki/Softmax_function>
pub struct OnlineSoftmax<'src, 'dst> {
    src_dest: SrcDest<'src, 'dst, f32>,
    flush_nans_to_zero: bool,
}

impl<'src, 'dst> OnlineSoftmax<'src, 'dst> {
    /// Construct a softmax operation which reads `input` and writes to `output`.
    #[track_caller]
    pub fn new(input: &'src [f32], output: &'dst mut [MaybeUninit<f32>]) -> Self {
        OnlineSoftmax {
            src_dest: (input, output).into(),
            flush_nans_to_zero: false,
        }
    }

    /// Construct a softmax operation which updates `input` in place.
    pub fn new_mut(input: &'dst mut [f32]) -> Self
    where
        'dst: 'src,
    {
        OnlineSoftmax {
            src_dest: input.into(),
            flush_nans_to_zero: false,
        }
    }

    /// Replace NaN values in the output with zeros.
    ///
    /// See [Softmax::flush_nans_to_zero] for details.
    pub fn flush_nans_to_zero(mut self, flush: bool) -> Self {
        self.flush_nans_to_zero = flush;
        self
    }
}

impl<'dst> SimdOp for OnlineSoftmax<'_, 'dst> {
    /// The normalized elements.
    type Output = &'dst mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();
        let src = self.src_dest.src();

        // Pass 1: Compute max and sum of exponentials in a single pass.
        //
        // The key insight from the online softmax paper is that when we
        // encounter a new maximum, we can rescale the running sum:
        //   d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
        //
        // This allows computing both max and sum in one pass over the data.
        //
        // Optimization: We only need ONE exp call per element, not two:
        // - If x > max_acc: new_max = x, so exp(x - new_max) = 1
        // - If x <= max_acc: new_max = max_acc, so exp(max_acc - new_max) = 1
        // Both arguments are always <= 0, so we can use ReducedRangeExp.
        let one = ops.one();
        let [max_vec, sum_vec] = src.simd_iter(ops).fold_n(
            [ops.splat(f32::NEG_INFINITY), ops.zero()],
            #[inline(always)]
            |[max_acc, sum_acc], x| {
                let new_max = ops.max(max_acc, x);
                let x_is_new_max = ops.gt(x, max_acc);

                // Compute exp(smaller - larger) = exp(min - max).
                // The argument is always <= 0, so we can use ReducedRangeExp.
                let min_val = ops.min(max_acc, x);
                let exp_val = ReducedRangeExp::apply(isa, ops.sub(min_val, new_max));

                // When x > max_acc: new_sum = sum_acc * exp_val + 1
                // When x <= max_acc: new_sum = sum_acc + exp_val
                let sum_if_new_max = ops.mul_add(sum_acc, exp_val, one);
                let sum_if_not = ops.add(sum_acc, exp_val);
                let new_sum = ops.select(sum_if_new_max, sum_if_not, x_is_new_max);
                [new_max, new_sum]
            },
        );

        // Reduce SIMD vectors to scalars.
        //
        // When reducing across lanes, we need to rescale sums from lanes with
        // smaller maximums using the same online algorithm.
        let max_arr = max_vec.to_array();
        let sum_arr = sum_vec.to_array();
        let mut max_val = f32::NEG_INFINITY;
        let mut exp_sum = 0.0f32;
        for i in 0..ops.len() {
            let lane_max = max_arr[i];
            let lane_sum = sum_arr[i];
            if lane_max > max_val {
                // Rescale accumulated sum and add this lane's contribution
                exp_sum = exp_sum * (max_val - lane_max).exp() + lane_sum;
                max_val = lane_max;
            } else {
                // Add this lane's contribution, scaled to current max
                exp_sum += lane_sum * (lane_max - max_val).exp();
            }
        }

        // Pass 2: Compute exp(x - max) / sum
        let max_val_vec = ops.splat(max_val);
        let inv_sum = ops.splat(1.0 / exp_sum);
        let zero = ops.zero();

        let dest = if self.flush_nans_to_zero {
            simd_map(
                ops,
                self.src_dest,
                #[inline(always)]
                |x| {
                    let y = ReducedRangeExp::apply(isa, ops.sub(x, max_val_vec));
                    let result = ops.mul(y, inv_sum);
                    let not_nan = ops.eq(result, result);
                    ops.select(result, zero, not_nan)
                },
            )
        } else {
            simd_map(
                ops,
                self.src_dest,
                #[inline(always)]
                |x| {
                    let y = ReducedRangeExp::apply(isa, ops.sub(x, max_val_vec));
                    ops.mul(y, inv_sum)
                },
            )
        };

        dest
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::SimdOp;

    use super::{OnlineSoftmax, Softmax};
    use crate::testing::{AsUninit, benchmark_op, check_f32s_are_equal_ulps, triples};

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
    #[ignore]
    fn bench_softmax() {
        benchmark_op(reference_softmax, |src, dest| {
            Softmax::new(src, dest).dispatch();
        });
    }

    #[test]
    fn test_online_softmax() {
        // Test against reference values.
        let input = vec![0.1634, 0.8647, 0.6401, 0.8265, 0.0560, 0.2304];
        let expected = &([
            0.11715934, 0.23623686, 0.18871443, 0.2273828, 0.10522857, 0.12527795,
        ]);
        let mut actual = vec![0.; input.len()];

        OnlineSoftmax::new(&input, actual.as_mut_slice().as_uninit()).dispatch();
        check_f32s_are_equal_ulps(triples(&input, &actual, expected), 2. /* max ULPs */);

        // Test against reference implementation for various lengths.
        for len in 1..20 {
            let input: Vec<f32> = (0..len).map(|x| x as f32 + 0.1).collect();
            let mut expected = vec![0.; input.len()];
            reference_softmax(&input, &mut expected);

            let mut actual = vec![0.; input.len()];
            OnlineSoftmax::new(&input, actual.as_mut_slice().as_uninit()).dispatch();

            check_f32s_are_equal_ulps(triples(&input, &actual, &expected), 4. /* max ULPs */);
        }
    }

    #[test]
    fn test_online_softmax_flush_nans_to_zero() {
        let mut input = [f32::NEG_INFINITY; 3];
        OnlineSoftmax::new_mut(&mut input).dispatch();
        assert!(input.iter().all(|x| x.is_nan()));

        let mut input = [f32::NEG_INFINITY; 3];
        OnlineSoftmax::new_mut(&mut input)
            .flush_nans_to_zero(true)
            .dispatch();
        assert_eq!(input, [0.; 3]);
    }

    #[test]
    #[ignore]
    fn bench_online_softmax() {
        benchmark_op(reference_softmax, |src, dest| {
            OnlineSoftmax::new(src, dest).dispatch();
        });
    }
}
