use rten_simd::ops::NumOps;
use rten_simd::{Isa, Simd, SimdIterable, SimdOp};

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
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();
        let [vec_min, vec_max] = self.input.simd_iter(ops).fold_n_unroll::<2, 4>(
            [ops.splat(f32::MAX), ops.splat(f32::MIN)],
            #[inline(always)]
            |[min, max], x| [ops.min(x, min), ops.max(x, max)],
            #[inline(always)]
            |[min_a, max_a], [min_b, max_b]| [ops.min(min_a, min_b), ops.max(max_a, max_b)],
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

/// Compute the maximum value in a slice, propagating NaNs.
pub struct MaxNum<'a, T> {
    input: &'a [T],
}

impl<'a, T> MaxNum<'a, T> {
    pub fn new(input: &'a [T]) -> Self {
        MaxNum { input }
    }
}

impl<'a> SimdOp for MaxNum<'a, f32> {
    type Output = f32;

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();

        let max_num = |max, x| {
            let not_nan = ops.eq(x, x);
            let new_max = ops.max(max, x);
            ops.select(new_max, x, not_nan)
        };

        let vec_max =
            self.input
                .simd_iter(ops)
                .fold_unroll::<2>(ops.splat(f32::MIN), max_num, max_num);

        vec_max
            .to_array()
            .as_ref()
            .iter()
            .copied()
            .fold(f32::MIN, |max, x| {
                if x.is_nan() {
                    x
                } else if max.is_nan() {
                    max
                } else {
                    x.max(max)
                }
            })
    }
}

/// Compute the minimum value in a slice, propagating NaNs.
pub struct MinNum<'a, T> {
    input: &'a [T],
}

impl<'a, T> MinNum<'a, T> {
    pub fn new(input: &'a [T]) -> Self {
        MinNum { input }
    }
}

impl<'a> SimdOp for MinNum<'a, f32> {
    type Output = f32;

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();

        let min_num = |min, x| {
            let not_nan = ops.eq(x, x);
            let new_min = ops.min(min, x);
            ops.select(new_min, x, not_nan)
        };

        let vec_min = self.input.simd_iter(ops).fold(ops.splat(f32::MAX), min_num);

        vec_min
            .to_array()
            .as_ref()
            .iter()
            .copied()
            .fold(f32::MAX, |min, x| {
                if x.is_nan() {
                    x
                } else if min.is_nan() {
                    min
                } else {
                    x.min(min)
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{MaxNum, MinMax, MinNum};
    use rten_simd::SimdOp;

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

    #[test]
    fn test_max_num() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let (_, expected_max) = reference_min_max(&xs);
        let max = MaxNum::new(&xs).dispatch();
        assert_eq!(max, expected_max);

        let xs = [0.1, 1.0, 0.2, f32::NAN, 0.4, 0.5, 0.6];
        let max = MaxNum::new(&xs).dispatch();
        assert!(max.is_nan());
    }

    #[test]
    fn test_min_num() {
        let xs: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.1).collect();
        let (expected_min, _) = reference_min_max(&xs);
        let min = MinNum::new(&xs).dispatch();
        assert_eq!(min, expected_min);

        let xs = [0.1, 1.0, 0.2, f32::NAN, 0.4, 0.5, 0.6];
        let min = MinNum::new(&xs).dispatch();
        assert!(min.is_nan());
    }
}
