use std::mem::{transmute, MaybeUninit};
use std::ops::Range;

use smallvec::SmallVec;

use crate::slice_range::{IndexRange, SliceItem};
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

pub struct RangeChunksExact {
    remainder: Range<usize>,
    chunk_size: usize,
}

impl RangeChunksExact {
    /// Return the part of the range that has not yet been visited.
    pub fn remainder(&self) -> Range<usize> {
        self.remainder.clone()
    }
}

impl Iterator for RangeChunksExact {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remainder.len() >= self.chunk_size {
            let start = self.remainder.start;
            let end = start + self.chunk_size;
            self.remainder.start += self.chunk_size;
            Some(start..end)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remainder.len() / self.chunk_size;
        (len, Some(len))
    }
}

impl ExactSizeIterator for RangeChunksExact {}

impl std::iter::FusedIterator for RangeChunksExact {}

/// Return an iterator over sub-ranges of `range`. If `range.len()` is not a
/// multiple of `chunk_size` then there will be a remainder after iteration
/// completes, available via [RangeChunksExact::remainder].
pub fn range_chunks_exact(range: Range<usize>, chunk_size: usize) -> RangeChunksExact {
    RangeChunksExact {
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

    for row_block in range_chunks(0..dest.rows(), BLOCK_SIZE) {
        for col_block in range_chunks(0..dest.cols(), BLOCK_SIZE) {
            let mut row_tiles = range_chunks_exact(row_block.clone(), TILE_SIZE);
            for row_tile in row_tiles.by_ref() {
                // Handle full height + width tiles.
                let mut col_tiles = range_chunks_exact(col_block.clone(), TILE_SIZE);
                for col_tile in col_tiles.by_ref() {
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

                // Handle full height but narrow edge tiles.
                for y in 0..TILE_SIZE {
                    for x in col_tiles.remainder() {
                        // Safety: Max values of `idx` are in-bounds for
                        // `src` and `dest`.
                        unsafe {
                            let idx = [row_tile.start + y, x];
                            let src_el = src.get_unchecked(idx).clone();
                            dest.get_unchecked_mut(idx).write(src_el);
                        }
                    }
                }
            }

            // Handle short edge tiles.
            for y in row_tiles.remainder() {
                for x in col_block.clone() {
                    // Safety: Max values of `idx` are in-bounds for
                    // `src` and `dest`.
                    unsafe {
                        let idx = [y, x];
                        let src_el = src.get_unchecked(idx).clone();
                        dest.get_unchecked_mut(idx).write(src_el);
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
        return unsafe { transmute::<&mut [MaybeUninit<T>], &[T]>(dest) };
    }

    while src.ndim() < 4 {
        src.insert_axis(0);
    }

    let src: NdTensorView<T, 4> = src.nd_view();

    // As a heuristic, use a blocked copy if the source stride is likely to lead
    // to cache conflicts. Otherwise a simple direct copy is probably going to
    // be faster. With a better optimized blocked copy path, we might be able to
    // use it all the time.
    let use_blocked_copy = src.stride(3) % 16 == 0 && src.stride(3) >= 32;

    if use_blocked_copy {
        let mut dest = NdTensorViewMut::from_data(src.shape(), dest);
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                let src = src.slice_with([i0, i1]);
                let dest = dest.slice_with_mut([i0, i1]);
                copy_blocked(src, dest);
            }
        }
        // Safety: Loop above initialized all elements of `dest`.
        let data = dest.data().unwrap();
        unsafe { transmute::<&[MaybeUninit<T>], &[T]>(data) }
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
        unsafe { transmute::<&[MaybeUninit<T>], &[T]>(dest) }
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

    // Efficiency could be improved here by sorting dims so that those with
    // the smallest stride are innermost. Also it could use the blocked copy
    // that `copy_into_slice` uses to avoid cache conflicts when inputs are
    // transposed.

    let copy_into_4d = |src: NdTensorView<T, 4>, mut dest: NdTensorViewMut<T, 4>| {
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                for i2 in 0..src.size(2) {
                    for i3 in 0..src.size(3) {
                        // Safety: `dest` and `src` have the same shape,
                        // and i0..i3 are in `[0, src.size(i))`.
                        unsafe {
                            *dest.get_unchecked_mut([i0, i1, i2, i3]) =
                                src.get_unchecked([i0, i1, i2, i3]).clone();
                        }
                    }
                }
            }
        }
    };

    if src.ndim() <= 4 {
        while src.ndim() < 4 {
            src.insert_axis(0);
            dest.insert_axis(0);
        }
        let src: NdTensorView<T, 4> = src.nd_view();
        let dest: NdTensorViewMut<T, 4> = dest.nd_view_mut();
        copy_into_4d(src, dest);
    } else {
        src.inner_iter::<4>()
            .zip(dest.inner_iter_mut::<4>())
            .for_each(|(src, dest)| copy_into_4d(src, dest));
    }
}

/// Clone elements of `src` into `dest`.
///
/// This is functionally equivalent to:
///
/// ```text
/// src.iter().zip(dest.iter_mut()).for_each(|(y, x)| y.write(x.clone()))
/// ```
///
/// But more efficient, especially when `src` or `dest` are not contiguous.
pub fn copy_into_uninit<T: Clone>(mut src: TensorView<T>, mut dest: TensorViewMut<MaybeUninit<T>>) {
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
                            // Safety: `dest` and `src` have the same shape,
                            // and i0..i3 are in `[0, src.size(i))`.
                            unsafe {
                                dest.get_unchecked_mut([i0, i1, i2, i3])
                                    .write(src.get_unchecked([i0, i1, i2, i3]).clone());
                            }
                        }
                    }
                }
            }
        });
}

/// Apply `f` to each element of `src` and write the output to `dest` in
/// contiguous order.
pub fn map_into_slice<T, R, F: Fn(&T) -> R>(
    mut src: TensorView<T>,
    dest: &mut [MaybeUninit<R>],
    f: F,
) {
    assert!(src.len() == dest.len());

    while src.ndim() < 4 {
        src.insert_axis(0);
    }

    // This would benefit from the same optimizations that `copy_into_slice` has
    // for eg. transposed inputs, preferably without generating a ton of
    // duplicate code for each map function `F`.

    let mut out_offset = 0;
    src.inner_iter::<4>().for_each(|src| {
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                for i2 in 0..src.size(2) {
                    for i3 in 0..src.size(3) {
                        // Safety: i0..i3 are in `[0, src.size(i))`.
                        let x = unsafe { src.get_unchecked([i0, i1, i2, i3]) };
                        let y = f(x);

                        // Safety: We write to `src.len()` successive output
                        // elements, and `src` and `dest` have the same length.
                        unsafe {
                            dest.get_unchecked_mut(out_offset).write(y);
                        }
                        out_offset += 1;
                    }
                }
            }
        }
    });

    debug_assert!(out_offset == src.len());
}

/// Copy a slice of `src` specified by `ranges` into `dest` in contiguous order.
pub fn copy_range_into_slice<T: Clone>(
    mut src: TensorView<T>,
    dest: &mut [MaybeUninit<T>],
    ranges: &[SliceItem],
) {
    assert!(ranges.len() <= src.ndim());

    // Pad shape to at least 4 dims.
    let mut added_dims = 0;
    while src.ndim() < 4 {
        src.insert_axis(0);
        added_dims += 1;
    }

    // Resolve slice ranges to stepped index ranges over source view.
    let index_ranges: SmallVec<[IndexRange; 4]> = src
        .shape()
        .iter()
        .enumerate()
        .map(|(i, &size)| {
            if i < added_dims {
                SliceItem::Index(0).index_range(size)
            } else {
                let full_range = SliceItem::full_range();
                let range = ranges.get(i - added_dims).unwrap_or(&full_range);
                range.index_range(size)
            }
        })
        .collect();

    copy_range_into_slice_inner(src, dest, &index_ranges);
}

fn copy_range_into_slice_inner<T: Clone>(
    src: TensorView<T>,
    mut dest: &mut [MaybeUninit<T>],
    ranges: &[IndexRange],
) {
    assert!(ranges.len() >= 4);

    if ranges.len() == 4 {
        // Iterate over innermost 4 dims. This uses static-rank views and
        // nested loops for efficiency.

        let src = src.nd_view::<4>();
        let ranges: [IndexRange; 4] = ranges.try_into().unwrap();

        // Check output length is correct.
        let sliced_len = ranges.iter().map(|s| s.steps()).product();
        assert_eq!(dest.len(), sliced_len, "output too short");

        let mut dest_offset = 0;
        for i0 in ranges[0] {
            for i1 in ranges[1] {
                for i2 in ranges[2] {
                    for i3 in ranges[3] {
                        // Safety: We checked dest offset is < length product
                        // of `index_ranges` iterators.
                        unsafe {
                            dest.get_unchecked_mut(dest_offset)
                                .write(src.get_unchecked([i0, i1, i2, i3]).clone());
                        }
                        dest_offset += 1;
                    }
                }
            }
        }
    } else {
        // Iterate over views of outermost dimension and recurse.
        for i0 in ranges[0] {
            let src_slice = src.slice_dyn(i0);
            let (dest_slice, dest_tail) = dest.split_at_mut(src_slice.len());

            copy_range_into_slice_inner(src_slice, dest_slice, &ranges[1..]);

            dest = dest_tail;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{copy_into, copy_into_slice, copy_into_uninit, copy_range_into_slice};
    use crate::rng::XorShiftRng;
    use crate::{AsView, Layout, NdTensor, SliceItem, Tensor, TensorView};

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
            let src = Tensor::<f32>::rand(&shape, &mut rng);
            let src = src.transposed();

            let mut dest = Tensor::zeros(src.shape());
            copy_into(src.view(), dest.view_mut());

            assert_eq!(dest, src);
        }
    }

    #[test]
    fn test_copy_into_uninit() {
        let mut rng = XorShiftRng::new(1234);
        for ndim in 0..5 {
            let shape: Vec<_> = (0..ndim).map(|d| d + 1).collect();
            let src = Tensor::<f32>::rand(&shape, &mut rng);
            let src = src.transposed();

            let mut dest = Tensor::uninit(src.shape());
            copy_into_uninit(src.view(), dest.view_mut());
            let dest = unsafe { dest.assume_init() };

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

    #[test]
    fn test_copy_range_into_slice() {
        struct Case<'a> {
            tensor: Tensor<i32>,
            range: &'a [SliceItem],
            expected: Vec<i32>,
        }

        let cases = [
            // <= 4 dims
            Case {
                tensor: Tensor::arange(0, 16, None).into_shape([4, 4].as_slice()),
                range: &[
                    // Every other row, in order
                    SliceItem::range(0, Some(5), 2),
                    // Every column, reversed
                    SliceItem::range(-1, Some(-6), -1),
                ],
                expected: Tensor::from([[3, 2, 1, 0], [11, 10, 9, 8]]).into_data(),
            },
            // > 4 dims
            Case {
                tensor: Tensor::arange(0, 32, None).into_shape([2, 2, 2, 2, 2].as_slice()),
                range: &[
                    SliceItem::range(0, Some(2), -1),
                    SliceItem::range(0, Some(2), -1),
                    SliceItem::range(0, Some(2), -1),
                    SliceItem::range(0, Some(2), -1),
                    SliceItem::range(0, Some(2), -1),
                ],
                expected: Tensor::arange(31, -1, Some(-1)).into_data(),
            },
        ];

        for Case {
            tensor,
            range,
            expected,
        } in cases
        {
            let dest_len = expected.len();
            let mut dest = Vec::with_capacity(dest_len);

            copy_range_into_slice(
                tensor.view(),
                &mut dest.spare_capacity_mut()[..dest_len],
                range,
            );

            // Assume `copy_range_into_slice` initialized all elements.
            unsafe {
                dest.set_len(dest_len);
            }

            assert_eq!(dest, expected);
        }
    }

    #[test]
    #[should_panic(expected = "output too short")]
    fn test_copy_range_into_slice_invalid() {
        let mut dest = Vec::new();

        copy_range_into_slice(
            NdTensor::arange(0, 4, None).into_shape([2, 2]).as_dyn(),
            &mut dest.spare_capacity_mut(),
            &[], // Empty range, selects whole tensor
        );
    }
}
