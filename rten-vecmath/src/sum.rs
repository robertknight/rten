use rten_simd::ops::NumOps;
use rten_simd::{Isa, Simd, SimdIterable, SimdOp};

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
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();
        let vec_sum = self
            .input
            .simd_iter(ops)
            .fold(ops.zero(), |sum, x| ops.add(sum, x));
        vec_sum.to_array().into_iter().sum()
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
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();
        let vec_sum = self
            .input
            .simd_iter(ops)
            .fold(ops.zero(), |sum, x| ops.mul_add(x, x, sum));
        vec_sum.to_array().into_iter().sum()
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
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();
        let offset_vec = ops.splat(self.offset);

        let vec_sum = self.input.simd_iter(ops).fold(ops.zero(), |sum, x| {
            let x_offset = ops.sub(x, offset_vec);
            ops.mul_add(x_offset, x_offset, sum)
        });

        vec_sum.to_array().into_iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use crate::ulp::assert_ulp_diff_le;

    use super::{Sum, SumSquare, SumSquareSub};
    use rten_simd::SimdOp;

    // Chosen to not be a multiple of vector size, so that tail handling is
    // exercised.
    const LEN: usize = 100;

    #[test]
    fn test_sum() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let expected_sum: f64 = xs.iter().map(|x| *x as f64).sum();
        let sum = Sum::new(&xs).dispatch();
        assert_ulp_diff_le!(sum, expected_sum as f32, 1.0);
    }

    #[test]
    fn test_sum_square() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let expected_sum: f64 = xs.iter().copied().map(|x| x as f64 * x as f64).sum();
        let sum = SumSquare::new(&xs).dispatch();
        assert_ulp_diff_le!(sum, expected_sum as f32, 2.0);
    }

    #[test]
    fn test_sum_square_sub() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let expected_sum: f64 = xs
            .iter()
            .copied()
            .map(|x| (x as f64 - mean as f64) * (x as f64 - mean as f64))
            .sum();
        let sum = SumSquareSub::new(&xs, mean).dispatch();
        assert_ulp_diff_le!(sum, expected_sum as f32, 2.0);
    }
}
