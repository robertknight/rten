use rten_simd::dispatch::{dispatch, SimdOp};
use rten_simd::functional::simd_fold;
use rten_simd::SimdFloat;

struct SimdSum<'a> {
    input: &'a [f32],
}

impl SimdOp for SimdSum<'_> {
    type Output = f32;

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let vec_sum = simd_fold(
            self.input.into(),
            S::zero(),
            #[inline(always)]
            |sum, x| sum.add(x),
            0., /* pad */
        );
        vec_sum.sum()
    }
}

/// Compute the sum of a slice of floats.
///
/// This is more efficient than `xs.iter().sum()` as it computes multiple
/// partial sums in parallel using SIMD and then sums across the SIMD lanes at
/// the end. This will produce very slightly different results because the
/// additions are happening in a different order.
pub fn vec_sum(xs: &[f32]) -> f32 {
    let op = SimdSum { input: xs };
    dispatch(op)
}

struct SimdSumSquare<'a> {
    input: &'a [f32],
}

impl SimdOp for SimdSumSquare<'_> {
    type Output = f32;

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let vec_sum = simd_fold(
            self.input.into(),
            S::zero(),
            #[inline(always)]
            |sum, x| x.mul_add(x, sum),
            0., /* pad */
        );
        vec_sum.sum()
    }
}

/// Compute the sum of the squares of elements in `xs`.
///
/// Conceptually this is like `xs.iter().map(|&x| x * x).sum()` but more
/// efficient as it computes multiple partial sums in parallel and then sums
/// across SIMD lanes at the end. The results will also be slightly different
/// because the additions are happening in a different order.
pub fn vec_sum_square(xs: &[f32]) -> f32 {
    let op = SimdSumSquare { input: xs };
    dispatch(op)
}

#[cfg(test)]
mod tests {
    use super::{vec_sum, vec_sum_square};

    #[test]
    fn test_vec_sum() {
        let xs: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let expected_sum: f32 = xs.iter().sum();
        let sum = vec_sum(&xs);
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn test_vec_sum_square() {
        let xs: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let expected_sum: f32 = xs.iter().copied().map(|x| x * x).sum();
        let sum = vec_sum_square(&xs);
        assert_eq!(sum, expected_sum);
    }
}
