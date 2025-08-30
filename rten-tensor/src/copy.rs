use std::mem::MaybeUninit;

use rten_base::iter::{range_chunks, range_chunks_exact};
use smallvec::SmallVec;

use crate::assume_init::AssumeInit;
use crate::slice_range::{IndexRange, SliceItem};
use crate::storage::{Storage, StorageMut};
use crate::{AsView, Layout};
use crate::{
    Matrix, MatrixLayout, MatrixMut, NdTensorView, NdTensorViewMut, TensorView, TensorViewMut,
};

/// Tile size for blocked copy. A tile should fit in registers for 32-bit
/// values.
const TILE_SIZE: usize = 4;

/// Block size for blocked copy. A source and dest block should fit in the cache
/// for 32-bit values.
const BLOCK_SIZE: usize = 64;

/// Transpose a square tile of size `TILE_SIZE`.
///
/// Safety: Caller must ensure that `src` points to a buffer of size
/// `src_col_stride * (TILE_SIZE - 1) + TILE_SIZE` and `dest` points to a buffer
/// of size `dest_row_stride * (TILE_SIZE - 1) + TILE_SIZE`.
unsafe fn transpose_tile<T: Clone, const TILE_SIZE: usize>(
    src: *const T,
    dest: *mut T,
    src_col_stride: usize,
    dest_row_stride: usize,
) {
    let src_tile: [[T; TILE_SIZE]; TILE_SIZE] = std::array::from_fn(|x| {
        std::array::from_fn(|y| unsafe { (*src.add(x * src_col_stride + y)).clone() })
    });
    for y in 0..TILE_SIZE {
        for x in 0..TILE_SIZE {
            let val = src_tile[x][y].clone();
            unsafe {
                dest.add(y * dest_row_stride + x).write(val);
            }
        }
    }
}

/// Copy elements from `src` into `dest`.
///
/// `src` and `dest` must have the same shape but can (should) have different
/// strides. This function uses blocking to avoid the cache conflicts that can
/// arise in a naive copy if `src` is transposed.
fn copy_blocked<T: Clone>(src: Matrix<T>, mut dest: MatrixMut<MaybeUninit<T>>) {
    // Ensure src and dest have same index range.
    assert!(src.shape() == dest.shape());

    let transpose = src.row_stride() == 1 && dest.col_stride() == 1;

    for row_block in range_chunks(0..dest.rows(), BLOCK_SIZE) {
        for col_block in range_chunks(0..dest.cols(), BLOCK_SIZE) {
            let mut row_tiles = range_chunks_exact(row_block.clone(), TILE_SIZE);
            for row_tile in row_tiles.by_ref() {
                // Handle full height + width tiles.
                let mut col_tiles = range_chunks_exact(col_block.clone(), TILE_SIZE);

                if transpose {
                    let src_ptr = src.storage().as_ptr();
                    let dest_ptr = dest.storage_mut().as_mut_ptr();
                    let src_col_stride = src.col_stride();
                    let dest_row_stride = dest.row_stride();

                    for col_tile in col_tiles.by_ref() {
                        // Safety: `col_tile` and `row_tile` are valid ranges
                        // for `src` and `dst` of size `TILE_SIZE`.
                        unsafe {
                            let src_ptr =
                                src_ptr.add(col_tile.start * src_col_stride + row_tile.start);
                            let dest_ptr =
                                dest_ptr.add(row_tile.start * dest_row_stride + col_tile.start);
                            transpose_tile::<T, TILE_SIZE>(
                                src_ptr,
                                dest_ptr as *mut T,
                                src_col_stride,
                                dest_row_stride,
                            );
                        }
                    }
                } else {
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

/// Return true if `T` is a type that is known to be `Copy`.
fn is_known_copy_type<T>() -> bool {
    [
        typeid::of::<f32>(),
        typeid::of::<i32>(),
        typeid::of::<i16>(),
        typeid::of::<u16>(),
        typeid::of::<u8>(),
        typeid::of::<i8>(),
    ]
    .contains(&typeid::of::<T>())
}

/// Clone elements of `src` into a contiguous destination slice with the same
/// length.
///
/// Returns `dest` as an initialized slice.
pub fn copy_into_slice<'a, T: Clone>(
    mut src: TensorView<T>,
    dest: &'a mut [MaybeUninit<T>],
) -> &'a mut [T] {
    assert!(dest.len() == src.len());

    if dest.is_empty() {
        // Safety: Destination is empty so already initialized.
        return unsafe { dest.assume_init() };
    }

    // Merge axes to increase the chance that we can use the fast path and
    // also maximize the iteration count of the innermost loops.
    src.merge_axes();

    if src.ndim() > 4 {
        let chunk_size = src.shape()[src.ndim() - 4..].iter().product();
        let mut n_init = 0;
        for (src, dest) in src.inner_iter::<4>().zip(dest.chunks_mut(chunk_size)) {
            copy_into_slice(src.as_dyn(), dest);
            n_init += chunk_size;
        }
        assert!(n_init == dest.len());

        // Safety: Loop above initialized all elements of `dest`.
        return unsafe { dest.assume_init() };
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

    // Threshold for copying contiguous inner lane using bulk-copying methods
    // (eg. memcpy).
    let bulk_copy_inner_lane_min_size_bytes = 32;

    if use_blocked_copy {
        // Source is transposed or otherwise not contiguous in the last lane
        let mut dest_mat = NdTensorViewMut::from_data(src.shape(), &mut dest[..]);
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                let src = src.slice([i0, i1]);
                let dest = dest_mat.slice_mut([i0, i1]);
                copy_blocked(src, dest);
            }
        }
    } else if src.stride(3) == 1
        && src.size(3) * size_of::<T>() >= bulk_copy_inner_lane_min_size_bytes
    {
        // Inner lane of source is contiguous and large enough to make it
        // worthwhile to copy contiguous chunks using a memcpy.
        let inner_lane_size = src.size(3);
        let mut dest_chunks = dest.chunks_mut(inner_lane_size);

        let src_ptr = src.storage().as_ptr();

        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                for i2 in 0..src.size(2) {
                    let dest_lane = dest_chunks.next().unwrap();

                    // Safety: `[i0, i1, i2]` is a valid index for the outer axes of `src` and the
                    // inner axis is contiguous and has the same size as `dest_lane`.
                    let src_lane = unsafe {
                        let ptr = src_ptr
                            .add(i0 * src.stride(0) + i1 * src.stride(1) + i2 * src.stride(2));
                        std::slice::from_raw_parts(ptr, dest_lane.len())
                    };

                    // For known copy types, use a `memcpy` which is more efficient than the generic
                    // clone loop.
                    if is_known_copy_type::<T>() {
                        // Safety: `T` is a copy type, `src_lane` and `dest_lane` have same length.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                src_lane.as_ptr(),
                                dest_lane.as_mut_ptr() as *mut T,
                                dest_lane.len(),
                            );
                        }
                    } else {
                        for (src, dst) in src_lane.iter().zip(dest_lane.iter_mut()) {
                            dst.write(src.clone());
                        }
                    }
                }
            }
        }
    } else {
        // General case
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
    }

    // Safety: Loop above initialized all elements of `dest`.
    unsafe { dest.assume_init() }
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
        let sliced_len: usize = ranges.iter().map(|s| s.steps()).product();
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
            let src_slice = src.slice(i0);
            let (dest_slice, dest_tail) = dest.split_at_mut(src_slice.len());

            copy_range_into_slice_inner(src_slice, dest_slice, &ranges[1..]);

            dest = dest_tail;
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{copy_into, copy_into_slice, copy_into_uninit, copy_range_into_slice};
    use crate::rng::XorShiftRng;
    use crate::{AsView, Layout, NdTensor, SliceItem, Tensor, TensorView};

    /// Wrapper around `copy_into_slice` that allocates a Vec to hold the result.
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
        // Contiguous and non-contiguous tensors with <= 4 dims.
        let x = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        assert_eq!(copy_into_vec(x.view()), [1, 2, 3, 4]);
        assert_eq!(copy_into_vec(x.transposed()), [1, 3, 2, 4]);

        // Contiguous and non-contiguous tensors with > 4 dims.
        let x = Tensor::from_data(&[1, 1, 1, 2, 2], vec![1, 2, 3, 4]);
        assert_eq!(copy_into_vec(x.view()), [1, 2, 3, 4]);
        assert_eq!(copy_into_vec(x.transposed()), [1, 3, 2, 4]);

        // Contiguous tensor with zero-size inner dim.
        let x = Tensor::<i32>::zeros(&[3, 4, 0]);
        assert_eq!(copy_into_vec(x.view()), [0i32; 0]);

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
    fn test_copy_into_slice_clone_type() {
        // Tensor with non-Copy element type and stride-1 inner axis.
        let src = Tensor::<String>::from([String::from("one"), "two".into(), "three".into()]);
        let dst = copy_into_vec(src.view());
        assert_eq!(dst, &["one", "two", "three"]);
    }

    #[test]
    fn test_copy_range_into_slice() {
        #[derive(Debug)]
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

        cases.test_each(|case| {
            let Case {
                tensor,
                range,
                expected,
            } = case;

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

            assert_eq!(dest, *expected);
        })
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
