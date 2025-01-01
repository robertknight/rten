use rten_simd::dispatch::SimdOp;
use rten_simd::functional::simd_fold;
use rten_simd::SimdFloat;

/// Computes the sum of a sequence of numbers.
///
/// This is more efficient than `slice.iter().sum()` as it computes multiple
/// partial sums in parallel using SIMD and then sums across the SIMD lanes at
/// the end. This will produce very slightly different results because the
/// additions are happening in a different order.
pub struct Sum<'a> {
    input: &'a [f32],
}

impl<'a> Sum<'a> {
    pub fn new(input: &'a [f32]) -> Self {
        Sum { input }
    }
}

impl SimdOp for Sum<'_> {
    type Output = f32;

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let vec_sum = simd_fold(
            self.input.into(),
            S::zero(),
            #[inline(always)]
            |sum, x| sum.add(x),
        );
        vec_sum.sum()
    }
}

/// Computes the sum of squares of a sequence of numbers.
///
/// This is conceptually equivalent to `slice.iter().map(|&x| x * x).sum()` but
/// more efficient as it computes multiple partial sums in parallel using SIMD
/// and then sums across the SIMD lanes at the end. This will produce very
/// slightly different results because the additions are happening in a
/// different order.
pub struct SumSquare<'a> {
    input: &'a [f32],
}

impl<'a> SumSquare<'a> {
    pub fn new(input: &'a [f32]) -> Self {
        SumSquare { input }
    }
}

impl SimdOp for SumSquare<'_> {
    type Output = f32;

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let vec_sum = simd_fold(
            self.input.into(),
            S::zero(),
            #[inline(always)]
            |sum, x| x.mul_add(x, sum),
        );
        vec_sum.sum()
    }
}

/// Compute the sum of squares of input with a bias subtracted.
///
/// This is a variant of [`SumSquare`] which subtracts a constant value from each
/// element before squaring it. A typical use case is to compute the variance of
/// a sequence, which is defined as `mean((X - x_mean)^2)`.
pub struct SumSquareSub<'a> {
    input: &'a [f32],
    offset: f32,
}

impl<'a> SumSquareSub<'a> {
    pub fn new(input: &'a [f32], offset: f32) -> Self {
        SumSquareSub { input, offset }
    }
}

impl SimdOp for SumSquareSub<'_> {
    type Output = f32;

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let offset_vec = S::splat(self.offset);
        let vec_sum = simd_fold(
            self.input.into(),
            S::zero(),
            #[inline(always)]
            |sum, x| {
                let x_offset = x.sub(offset_vec);
                x_offset.mul_add(x_offset, sum)
            },
        );
        vec_sum.sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{Sum, SumSquare, SumSquareSub};
    use rten_simd::dispatch::SimdOp;

    // Chosen to not be a multiple of vector size, so that tail handling is
    // exercised.
    const LEN: usize = 100;

    #[test]
    fn test_sum() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let expected_sum: f32 = xs.iter().sum();
        let sum = Sum::new(&xs).dispatch();
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn test_sum_square() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let expected_sum: f32 = xs.iter().copied().map(|x| x * x).sum();
        let sum = SumSquare::new(&xs).dispatch();
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn test_sum_square_sub() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let expected_sum: f32 = xs.iter().copied().map(|x| (x - mean) * (x - mean)).sum();
        let sum = SumSquareSub::new(&xs, mean).dispatch();
        assert_eq!(sum, expected_sum);
    }
}
