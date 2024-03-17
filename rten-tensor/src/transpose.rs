use std::mem::MaybeUninit;
use std::ops::Range;

use crate::{AsView, Layout};
use crate::{NdLayout, NdTensorView, NdTensorViewMut, TensorView};

/// Iterator returned by [range_chunks].
pub struct RangeChunks {
    remainder: Range<usize>,
    chunk_size: usize,
}

impl Iterator for RangeChunks {
    type Item = Range<usize>;

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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remainder.len().div_ceil(self.chunk_size);
        (len, Some(len))
    }
}

impl ExactSizeIterator for RangeChunks {}

impl std::iter::FusedIterator for RangeChunks {}

/// Return an iterator over sub-ranges of `range`. If `range.len()` is not a
/// multiple of `chunk_size` then the final chunk will be shorter.
fn range_chunks(range: Range<usize>, chunk_size: usize) -> RangeChunks {
    RangeChunks {
        remainder: range,
        chunk_size,
    }
}

/// Return the elements of `src` as a contiguous vector, in the same order they
/// would be yielded by `src.iter()`.
///
/// This is equivalent to `src.iter().cloned().collect::<Vec<_>>()` but
/// faster.
pub fn contiguous_data<T: Clone>(mut src: TensorView<T>) -> Vec<T> {
    if src.ndim() > 4 {
        // Fallback for tensors with too many dims.
        return src.iter().cloned().collect();
    }

    // Pad input to 4 dims.
    while src.ndim() < 4 {
        src.insert_axis(0);
    }

    let src_4d: NdTensorView<_, 4> = src.nd_view();
    let src_ptr = src.non_contiguous_data().as_ptr();
    let src_strides = src_4d.strides();

    let mut data: Vec<T> = Vec::with_capacity(src.len());
    let dest_ptr = data.as_mut_ptr();
    let dest_strides = NdLayout::contiguous_strides(src_4d.shape());

    let block_size = [8, 8, 32, 32];
    let mut tmp_data: Vec<T> = Vec::with_capacity(block_size.iter().product());
    let tmp_ptr = tmp_data.as_mut_ptr();

    // Partition input into blocks which fit in cache.
    for i0_block in range_chunks(0..src_4d.size(0), block_size[0]) {
        for i1_block in range_chunks(0..src_4d.size(1), block_size[1]) {
            for i2_block in range_chunks(0..src_4d.size(2), block_size[2]) {
                for i3_block in range_chunks(0..src_4d.size(3), block_size[3]) {
                    // Copy input into temporary buffer in contiguous order.
                    let mut tmp_off = 0;
                    for i0 in i0_block.clone() {
                        for i1 in i1_block.clone() {
                            for i2 in i2_block.clone() {
                                for i3 in i3_block.clone() {
                                    let src_off = i0 * src_strides[0]
                                        + i1 * src_strides[1]
                                        + i2 * src_strides[2]
                                        + i3 * src_strides[3];
                                    unsafe {
                                        tmp_ptr
                                            .add(tmp_off)
                                            .write(src_ptr.add(src_off).read().clone());
                                    }
                                    tmp_off += 1;
                                }
                            }
                        }
                    }

                    // Copy temporary buffer into destination.
                    tmp_off = 0;
                    for i0 in i0_block.clone() {
                        for i1 in i1_block.clone() {
                            for i2 in i2_block.clone() {
                                for i3 in i3_block.clone() {
                                    let dest_off = i0 * dest_strides[0]
                                        + i1 * dest_strides[1]
                                        + i2 * dest_strides[2]
                                        + i3 * dest_strides[3];
                                    unsafe {
                                        dest_ptr.add(dest_off).write(tmp_ptr.add(tmp_off).read());
                                    }
                                    tmp_off += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Safety: We have initialized all the elements.
    unsafe { data.set_len(data.capacity()) }

    data
}

#[cfg(test)]
mod tests {
    use super::contiguous_data;
    use crate::{AsView, Tensor};

    #[test]
    fn test_contiguous_data() {
        // <= 4 dims
        let x = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        assert_eq!(contiguous_data(x.view()), [1, 2, 3, 4]);
        assert_eq!(contiguous_data(x.transposed()), [1, 3, 2, 4]);

        // > 4 dims
        let x = Tensor::from_data(&[1, 1, 1, 2, 2], vec![1, 2, 3, 4]);
        assert_eq!(contiguous_data(x.view()), [1, 2, 3, 4]);
        assert_eq!(contiguous_data(x.transposed()), [1, 3, 2, 4]);
    }
}
