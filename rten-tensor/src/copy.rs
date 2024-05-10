use std::mem::MaybeUninit;
use std::ops::Range;

use crate::{AsView, Layout};
use crate::{
    Matrix, MatrixLayout, MatrixMut, NdTensorView, NdTensorViewMut, TensorView, TensorViewMut,
};

/// Iterator returned by [range_chunks].
pub struct RangeChunks {
    remainder: Range<usize>,
    chunk_size: usize,
}

impl Iterator for RangeChunks {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.remainder.is_empty() {
            let start = self.remainder.start;
            let end = (start + self.chunk_size).min(self.remainder.end);
            self.remainder.start += self.chunk_size;
            Some(start..end)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remainder.len().div_ceil(self.chunk_size);
        (len, Some(len))
    }
}

impl ExactSizeIterator for RangeChunks {}

impl std::iter::FusedIterator for RangeChunks {}

/// Return an iterator over sub-ranges of `range`. If `range.len()` is not a
/// multiple of `chunk_size` then the final chunk will be shorter.
#[inline]
pub fn range_chunks(range: Range<usize>, chunk_size: usize) -> RangeChunks {
    RangeChunks {
        remainder: range,
        chunk_size,
    }
}

/// Tile size for blocked copy. A tile should fit in registers for 32-bit
/// values.
const TILE_SIZE: usize = 4;

/// Block size for blocked copy. A source and dest block should fit in the cache
/// for 32-bit values.
const BLOCK_SIZE: usize = 64;

/// Copy elements from `src` into `dest`.
///
/// `src` and `dest` must have the same shape but can (should) have different
/// strides. This function uses blocking to avoid the cache conflicts that can
/// arise in a naive copy if `src` is transposed.
fn copy_blocked<T: Clone>(src: Matrix<T>, mut dest: MatrixMut<MaybeUninit<T>>) {
    // Ensure src and dest have same index range.
    assert!(src.shape() == dest.shape());

    // Ensure tiles are always full.
    assert!(dest.rows() % TILE_SIZE == 0);
    assert!(dest.cols() % TILE_SIZE == 0);

    for row_block in range_chunks(0..dest.rows(), BLOCK_SIZE) {
        for col_block in range_chunks(0..dest.cols(), BLOCK_SIZE) {
            for row_tile in range_chunks(row_block.clone(), TILE_SIZE) {
                for col_tile in range_chunks(col_block.clone(), TILE_SIZE) {
                    debug_assert!(row_tile.len() == TILE_SIZE);
                    debug_assert!(col_tile.len() == TILE_SIZE);

                    for y in 0..TILE_SIZE {
                        for x in 0..TILE_SIZE {
                            // Safety: Max values of `idx` are in-bounds for
                            // `src` and `dest`.
                            unsafe {
                                let idx = [row_tile.start + y, col_tile.start + x];
                                let src_el = src.get_unchecked(idx).clone();
                                dest.get_unchecked_mut(idx).write(src_el);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Copy elements of `src` into a contiguous destination slice with the same
/// length.
///
/// Returns `dest` as an initialized slice.
pub fn copy_into_slice<'a, T: Clone>(
    src: TensorView<T>,
    dest: &'a mut [MaybeUninit<T>],
) -> &'a [T] {
    assert!(dest.len() == src.len());

    // Merge axes to increase the chance that we can use the fast path and
    // also maximize the iteration count of the innermost loops.
    let mut src = src.clone();
    src.merge_axes();

    if src.ndim() > 4 {
        for (dst, src) in dest.iter_mut().zip(src.iter()) {
            dst.write(src.clone());
        }
        // Safety: Loop above initialized all elements of `dest`.
        return unsafe { std::mem::transmute(dest) };
    }

    while src.ndim() < 4 {
        src.insert_axis(0);
    }

    let src: NdTensorView<T, 4> = src.nd_view();

    // As a heuristic, use a blocked copy if the source stride is likely to lead
    // to cache conflicts. Otherwise a simple direct copy is probably going to
    // be faster. With a better optimized blocked copy path, we might be able to
    // use it all the time.
    let use_blocked_copy = src.stride(3).count_ones() == 1
        && src.stride(3) >= 32
        && src.size(2) % TILE_SIZE == 0
        && src.size(3) % TILE_SIZE == 0;

    if use_blocked_copy {
        let mut dest = NdTensorViewMut::from_data(src.shape(), dest);
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                let src = src.slice::<2, _>([i0, i1]);
                let dest = dest.slice_mut::<2, _>([i0, i1]);
                copy_blocked(src, dest);
            }
        }
        // Safety: Loop above initialized all elements of `dest`.
        let data = dest.data().unwrap();
        unsafe { std::mem::transmute(data) }
    } else {
        let mut dest_offset = 0;
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                for i2 in 0..src.size(2) {
                    for i3 in 0..src.size(3) {
                        unsafe {
                            let elt = src.get_unchecked([i0, i1, i2, i3]).clone();
                            dest.get_unchecked_mut(dest_offset).write(elt);
                            dest_offset += 1;
                        }
                    }
                }
            }
        }
        // Safety: Loop above initialized all elements of `dest`.
        unsafe { std::mem::transmute(dest) }
    }
}

/// Clone elements of `src` into `dest`.
///
/// This is functionally equivalent to:
///
/// ```text
/// src.iter().zip(dest.iter_mut()).for_each(|(y, x)| *y = x.clone())
/// ```
///
/// But more efficient, especially when `src` or `dest` are not contiguous.
pub fn copy_into<T: Clone>(mut src: TensorView<T>, mut dest: TensorViewMut<T>) {
    assert!(src.shape() == dest.shape());

    while src.ndim() < 4 {
        src.insert_axis(0);
        dest.insert_axis(0);
    }

    // Efficiency could be improved here by sorting dims so that those with
    // the smallest stride are innermost. Also it could use the blocked copy
    // that `copy_into_slice` uses to avoid cache conflicts when inputs are
    // transposed.

    src.inner_iter::<4>()
        .zip(dest.inner_iter_mut::<4>())
        .for_each(|(src, mut dest)| {
            for i0 in 0..src.size(0) {
                for i1 in 0..src.size(1) {
                    for i2 in 0..src.size(2) {
                        for i3 in 0..src.size(3) {
                            unsafe {
                                *dest.get_unchecked_mut([i0, i1, i2, i3]) =
                                    src.get_unchecked([i0, i1, i2, i3]).clone();
                            }
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::{copy_into, copy_into_slice};
    use crate::rng::XorShiftRng;
    use crate::{AsView, Layout, Tensor, TensorView};

    /// Return the elements of `src` as a contiguous vector, in the same order they
    /// would be yielded by `src.iter()`.
    ///
    /// This function assumes that the caller has already checked if `src` is
    /// contiguous and used more efficient methods to copy the data in that case.
    ///
    /// This is equivalent to `src.iter().cloned().collect::<Vec<_>>()` but
    /// faster.
    fn copy_into_vec<T: Clone>(src: TensorView<T>) -> Vec<T> {
        let src_len = src.len();
        let mut result = Vec::with_capacity(src_len);
        copy_into_slice(src, &mut result.spare_capacity_mut()[..src_len]);

        // Safety: `copy_contiguous` initialized `src_len` elements of result.
        unsafe { result.set_len(src_len) };

        result
    }

    #[test]
    fn test_copy_into() {
        let mut rng = XorShiftRng::new(1234);
        for ndim in 0..5 {
            let shape: Vec<_> = (0..ndim).map(|d| d + 1).collect();
            let src = Tensor::rand(&shape, &mut rng);
            let src = src.transposed();

            let mut dest = Tensor::zeros(src.shape());
            copy_into(src.view(), dest.view_mut());

            assert_eq!(dest, src);
        }
    }

    #[test]
    fn test_copy_into_slice() {
        // <= 4 dims
        let x = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        assert_eq!(copy_into_vec(x.view()), [1, 2, 3, 4]);
        assert_eq!(copy_into_vec(x.transposed()), [1, 3, 2, 4]);

        // > 4 dims
        let x = Tensor::from_data(&[1, 1, 1, 2, 2], vec![1, 2, 3, 4]);
        assert_eq!(copy_into_vec(x.view()), [1, 2, 3, 4]);
        assert_eq!(copy_into_vec(x.transposed()), [1, 3, 2, 4]);

        // Transposed matrices of varying sizes. This includes:
        //
        // - Zero
        // - Powers of 2
        // - Non-powers of 2
        // - Values above and below threshold for using blocked copy
        for size in [0usize, 2, 4, 8, 15, 16, 32, 64, 65, 68] {
            let x = Tensor::<i32>::arange(0, (size * size) as i32, None);
            let x = x.reshaped([size, size]);
            let transposed = copy_into_vec(x.transposed().as_dyn());
            let expected = x.transposed().iter().copied().collect::<Vec<_>>();
            assert_eq!(transposed, expected);
        }
    }
}
