//! Optimized reductions of slices of numbers.

/// Return the sum of a slice of primitive types.
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

#[cfg(test)]
mod tests {
    use wasnn_tensor::rng::XorShiftRng;
    use wasnn_tensor::test_util::ApproxEq;

    use super::slice_sum;

    #[test]
    fn test_slice_sum() {
        let mut rng = XorShiftRng::new(1234);
        let xs: Vec<_> = std::iter::from_fn(|| Some(rng.next_f32()))
            .take(256)
            .collect();
        assert!(xs.iter().sum::<f32>().approx_eq(&slice_sum(&xs)));
    }
}
