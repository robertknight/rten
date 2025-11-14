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
}

impl<'src, 'dst> Softmax<'src, 'dst> {
    /// Construct a softmax operation which reads `input` and writes to to
    /// `output`.
    #[track_caller]
    pub fn new(input: &'src [f32], output: &'dst mut [MaybeUninit<f32>]) -> Self {
        Softmax {
            src_dest: (input, output).into(),
        }
    }

    /// Construct a softmax operation which updates `input` in place.
    pub fn new_mut(input: &'dst mut [f32]) -> Self
    where
        'dst: 'src,
    {
        Softmax {
            src_dest: input.into(),
        }
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
        simd_apply::<_, _, _, UNROLL>(
            ops,
            dest,
            #[inline(always)]
            |x| ops.mul(x, inv_exp_sum),
        );

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

#[cfg(test)]
mod tests {
    use rten_simd::SimdOp;

    use super::Softmax;
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
    #[ignore]
    fn bench_softmax() {
        benchmark_op(reference_softmax, |src, dest| {
            Softmax::new(src, dest).dispatch();
        });
    }
}
