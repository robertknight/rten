//! Optimized reductions of slices and iterators of numbers.
//!
//! Library APIs like `std::iter::Sum` reduce elements in-order. For float
//! values this is not optimal for performance as each step has a dependency on
//! the previous step, inhibiting Instruction Level Parallelism and
//! autovectorization. The functions in this module re-order operations to
//! enable better performance.
//!
//! Related reading:
//!
//! - <https://blog.zachbjornson.com/2019/08/11/fast-float-summation.html>

use crate::number::MinMax;

/// Return the sum of a slice of numbers.
pub fn slice_max<T: Copy + MinMax>(xs: &[T]) -> T {
    const CHUNK_SIZE: usize = 8;
    xs.chunks(CHUNK_SIZE)
        .map(|chunk| {
            if chunk.len() == CHUNK_SIZE {
                // Writing the code this way encourages better autovectorization.
                let a0 = chunk[0].max(chunk[1]);
                let a1 = chunk[2].max(chunk[3]);
                let a2 = chunk[4].max(chunk[5]);
                let a3 = chunk[6].max(chunk[7]);

                let b0 = a0.max(a1);
                let b1 = a2.max(a3);

                b0.max(b1)
            } else {
                chunk.iter().copied().fold(T::min_val(), |x, y| x.max(y))
            }
        })
        .fold(T::min_val(), |x, y| x.max(y))
}

/// Return the sum of a slice of numbers.
pub fn slice_sum<T: Copy + Default + std::ops::Add<Output = T>>(xs: &[T]) -> T {
    const CHUNK_SIZE: usize = 8;
    xs.chunks(CHUNK_SIZE)
        .map(|chunk| {
            if chunk.len() == CHUNK_SIZE {
                // Writing the code this way encourages better autovectorization.
                let x = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let y = [chunk[4], chunk[5], chunk[6], chunk[7]];
                let z = [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]];
                z[0] + z[1] + z[2] + z[3]
            } else {
                chunk.iter().copied().fold(T::default(), |x, y| x + y)
            }
        })
        .fold(T::default(), |x, y| x + y)
}

/// Return the sum of an iterator of numbers.
pub fn iter_sum<T: Copy + Default + std::ops::Add<Output = T>, I: ExactSizeIterator<Item = T>>(
    mut iter: I,
) -> T {
    let zero = T::default();
    let len = iter.len();
    let mut sum = zero;
    let mut n = len;

    while n > 4 {
        n -= 4;

        let a = iter.next().unwrap_or(zero);
        let b = iter.next().unwrap_or(zero);
        let c = iter.next().unwrap_or(zero);
        let d = iter.next().unwrap_or(zero);

        let ab = a + b;
        let cd = c + d;
        let abcd = ab + cd;
        sum = sum + abcd;
    }

    for x in iter {
        sum = sum + x;
    }

    sum
}

#[cfg(test)]
mod tests {
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::ApproxEq;

    use super::{slice_max, slice_sum};

    #[test]
    fn test_slice_max() {
        let mut rng = XorShiftRng::new(1234);
        let xs: Vec<_> = std::iter::from_fn(|| Some(rng.next_f32()))
            .take(256)
            .collect();
        let expected = xs.iter().fold(f32::NEG_INFINITY, |x, y| x.max(*y));
        let actual = slice_max(&xs);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_slice_sum() {
        let mut rng = XorShiftRng::new(1234);
        let xs: Vec<_> = std::iter::from_fn(|| Some(rng.next_f32()))
            .take(256)
            .collect();
        assert!(xs.iter().sum::<f32>().approx_eq(&slice_sum(&xs)));
    }
}
