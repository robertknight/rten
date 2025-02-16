use rten_simd::dispatch::SimdOp;
use rten_simd::functional::simd_fold_array;
use rten_simd::SimdFloat;

/// Compute the minimum and maximum values in a slice of floats.
pub struct MinMax<'a> {
    input: &'a [f32],
}

impl<'a> MinMax<'a> {
    pub fn new(input: &'a [f32]) -> Self {
        MinMax { input }
    }
}

impl SimdOp for MinMax<'_> {
    type Output = (f32, f32);

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let [vec_min, vec_max] = simd_fold_array(
            self.input,
            [S::splat(f32::MAX), S::splat(f32::MIN)],
            #[inline(always)]
            |[min, max], x| [x.min(min), x.max(max)],
        );
        let min = vec_min
            .to_array()
            .as_ref()
            .iter()
            .fold(f32::MAX, |min, x| x.min(min));
        let max = vec_max
            .to_array()
            .as_ref()
            .iter()
            .fold(f32::MIN, |max, x| x.max(max));
        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::MinMax;
    use rten_simd::dispatch::SimdOp;

    // Chosen to not be a multiple of vector size, so that tail handling is
    // exercised.
    const LEN: usize = 100;

    fn reference_min_max(xs: &[f32]) -> (f32, f32) {
        let min = xs.iter().fold(f32::MAX, |min, x| x.min(min));
        let max = xs.iter().fold(f32::MIN, |max, x| x.max(max));
        (min, max)
    }

    #[test]
    fn test_min_max() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let expected = reference_min_max(&xs);
        let min_max = MinMax::new(&xs).dispatch();
        assert_eq!(min_max, expected);
    }
}
