use std::mem::MaybeUninit;

use rten_simd::dispatch::{dispatch, SimdOp};
use rten_simd::functional::{simd_fold, simd_map};
use rten_simd::span::{MutPtrLen, PtrLen};
use rten_simd::SimdFloat;

use crate::exp::simd_exp;

/// Apply the softmax operation over elements in `xs` and write results to
/// `out`.
///
/// The implementation uses a three-pass approach for numerical stability.
/// See https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html
/// and https://arxiv.org/abs/2001.04438.
#[inline(always)]
unsafe fn simd_softmax<S: SimdFloat>(input: PtrLen<f32>, out: MutPtrLen<MaybeUninit<f32>>) {
    let max_val = simd_fold(
        input,
        S::splat(f32::MIN),
        #[inline(always)]
        |max, x| max.max(x),
        f32::MIN, /* pad */
    );
    let max_val = max_val.fold_splat(f32::MIN, |max: f32, x: f32| max.max(x));

    // *x = (*x - max_val).exp()
    let mut exp_sum = S::zero();
    let exp_pad = f32::NEG_INFINITY; // exp(-inf) = 0, so won't affect `exp_sum`
    simd_map(
        input,
        out,
        #[inline(always)]
        |x: S| {
            let y = simd_exp(x.sub(max_val));
            exp_sum = exp_sum.add(y);
            y
        },
        exp_pad,
    );

    // *x /= exp_sum
    let exp_sum = exp_sum.fold_splat(0., |sum, x| sum + x);
    simd_map(
        out.assume_init().into(),
        out,
        #[inline(always)]
        |x: S| x.div(exp_sum),
        1., /* pad */
    );
}

struct SimdSoftmax {
    input: PtrLen<f32>,
    output: MutPtrLen<MaybeUninit<f32>>,
}

impl SimdOp for SimdSoftmax {
    type Output = ();

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        simd_softmax::<S>(self.input, self.output)
    }
}

/// Computes the [softmax][softmax] function over a slice of floats.
///
/// `out` will be fully initialized after this function returns.
///
/// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
pub fn vec_softmax(xs: &[f32], out: &mut [MaybeUninit<f32>]) {
    let op = SimdSoftmax {
        input: xs.into(),
        output: out.into(),
    };
    dispatch(op)
}

/// Computes the [softmax][softmax] function over a slice of floats.
///
/// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
pub fn vec_softmax_in_place(xs: &mut [f32]) {
    let out: MutPtrLen<f32> = xs.into();
    let op = SimdSoftmax {
        input: xs.into(),
        output: out.as_uninit(),
    };
    dispatch(op)
}

#[cfg(test)]
mod tests {
    use super::vec_softmax;

    use crate::testing::{benchmark_op, check_f32s_are_equal_ulps, triples, AsUninit};

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
    fn test_vec_softmax() {
        let input = vec![0.1634, 0.8647, 0.6401, 0.8265, 0.0560, 0.2304];
        let expected = &([
            0.11715934, 0.23623686, 0.18871443, 0.2273828, 0.10522857, 0.12527795,
        ]);
        let mut actual = vec![0.; input.len()];

        vec_softmax(&input, actual.as_mut_slice().as_uninit());

        check_f32s_are_equal_ulps(triples(&input, &actual, expected), 0. /* max ULPs */);
    }

    #[test]
    #[ignore]
    fn bench_softmax() {
        benchmark_op(reference_softmax, vec_softmax);
    }
}
