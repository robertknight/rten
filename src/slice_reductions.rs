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

use rten_base::num::MinMax;

/// Fold the contents of a slice under the assumption that the fold operation
/// is associative.
///
/// This is similar to `xs.iter().fold(init, f)` but more efficient if `f` is
/// an operation which is not seen by the compiler as associative, but can be
/// treated as such for a particular use. This includes many float operations
/// (min, max etc.).
pub fn slice_fold_assoc<T: Copy, F: Fn(T, T) -> T>(xs: &[T], init: T, f: F) -> T {
    const CHUNK_SIZE: usize = 8;

    let (chunks, tail) = xs.as_chunks::<CHUNK_SIZE>();
    let acc = chunks.iter().fold(init, |acc, chunk| {
        // Use associativity assumption to perform a tree reduction. This
        // enables better instruction-level parallelism.
        let a0 = f(chunk[0], chunk[1]);
        let a1 = f(chunk[2], chunk[3]);
        let a2 = f(chunk[4], chunk[5]);
        let a3 = f(chunk[6], chunk[7]);

        let b0 = f(a0, a1);
        let b1 = f(a2, a3);

        let chunk_acc = f(b0, b1);
        f(acc, chunk_acc)
    });

    tail.iter().copied().fold(acc, &f)
}

/// Return the maximum of a slice of numbers.
pub fn slice_max<T: Copy + MinMax>(xs: &[T]) -> T {
    slice_fold_assoc(xs, T::min_val(), |acc, x| acc.max(x))
}

/// Return the sum of a slice of numbers.
pub fn slice_sum<T: Copy + Default + std::ops::Add<Output = T>>(xs: &[T]) -> T {
    slice_fold_assoc(xs, T::default(), |acc, x| acc + x)
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
