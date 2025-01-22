use std::mem::MaybeUninit;
use std::ops::Range;

use rten_tensor::{Alloc, Matrix, MatrixLayout, Storage};

use super::kernels::PackedLayout;
use crate::iter_util::range_chunks;
use crate::slice_cast::{cast_pod_mut_slice, cast_pod_slice};

pub mod int8;

/// Helper for incrementally filling a slice.
struct SliceWriter<'a, T> {
    offset: usize,
    slice: &'a mut [MaybeUninit<T>],
}

impl<'a, T> SliceWriter<'a, T> {
    fn new(slice: &'a mut [MaybeUninit<T>]) -> Self {
        SliceWriter { slice, offset: 0 }
    }

    /// Return true if the slice has been fully written.
    fn completed(&self) -> bool {
        self.offset == self.slice.len()
    }

    /// Write the next element in the slice.
    ///
    /// Safety: The number of elements already written must be less than the
    /// length of the slice.
    unsafe fn write_unchecked(&mut self, val: T) {
        debug_assert!(self.offset < self.slice.len());
        self.slice.get_unchecked_mut(self.offset).write(val);
        self.offset += 1;
    }

    /// Write `len` copies of `val` to the slice.
    ///
    /// Safety: The number of elements already written must be less than or
    /// equal to `slice.len() - len`.
    unsafe fn write_n_unchecked(&mut self, len: usize, val: T)
    where
        T: Copy,
    {
        debug_assert!(self.offset + len <= self.slice.len());
        for i in 0..len {
            self.slice.get_unchecked_mut(self.offset + i).write(val);
        }
        self.offset += len;
    }
}

/// Return the required size and other metadata for packing an "A" matrix with
/// [`pack_a_block`].
pub fn packed_a_layout<T, const MR: usize>(rows: usize, cols: usize) -> PackedLayout {
    let size = rows.next_multiple_of(MR) * cols * size_of::<T>();
    let panel_stride = MR * cols * size_of::<T>();
    PackedLayout::new(size, align_of::<T>(), panel_stride)
}

/// Pack a block of the "A" matrix for use by a GEMM kernel, in row-major order.
///
/// The packed buffer is laid out as a sequence of `ceil(rows.len() / MR)` row
/// panels. Each row panel has size `MR * cols.len()` and uses row-major order.
/// If `rows.len()` is not a multiple of `MR`, the final panel is zero-padded.
#[inline] // Allow caller to control `target_feature`s
pub fn pack_a_block<T: Copy + Default, const MR: usize>(
    out: &mut [MaybeUninit<T>],
    a: Matrix<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    assert_eq!(out.len(), rows.len().next_multiple_of(MR) * cols.len());
    let mut out = SliceWriter::new(out);

    for panel_rows in range_chunks(rows, MR) {
        for row in panel_rows.clone() {
            for col in cols.clone() {
                unsafe {
                    out.write_unchecked(*a.get_unchecked([row, col]));
                }
            }
        }

        // Pad wth zeros
        for _ in panel_rows.end..panel_rows.start + MR {
            unsafe { out.write_n_unchecked(cols.len(), T::default()) };
        }
    }

    // Make sure we initialized the entire block.
    assert!(out.completed());
}

/// Return the required size and other metadata for packing a "B" matrix with
/// [`pack_b_block`].
pub fn packed_b_layout<T, const NR: usize>(rows: usize, cols: usize) -> PackedLayout {
    let size = cols.next_multiple_of(NR) * rows * size_of::<T>();
    let panel_stride = NR * rows * size_of::<T>();
    PackedLayout::new(size, align_of::<T>(), panel_stride)
}

/// Pack a block of the "B" matrix for use by a GEMM kernel.
///
/// The packed buffer is laid out as a sequence of `ceil(cols.len() /
/// NR)` column panels. Each column panel has size `rows.len() *
/// NR` and uses row-major order. If `cols.len()` is not a multiple of
/// `NR`, the final panel is zero-padded.
///
/// Panics if the output buffer is not exactly the correct size.
///
/// # Safety
///
/// When this function returns, all elements of `out` will have been initialized
/// either to a value from `b`, or zero.
#[inline] // Allow caller to control `target_feature`s
pub fn pack_b_block<T: Copy + Default, const NR: usize>(
    out: &mut [MaybeUninit<T>],
    b: Matrix<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    let b_cols = cols.len();
    let b_rows = rows.len();
    let b_row_stride = b.row_stride();
    let b_col_stride = b.col_stride();
    let n_panels = b_cols.next_multiple_of(NR) / NR;

    let used_size = n_panels * b_rows * NR;
    assert_eq!(out.len(), used_size);

    // Safety: Loops below must only access valid offsets in `b_data`.
    let b_data = b.storage();

    let mut out = SliceWriter::new(out);

    for panel in 0..n_panels {
        let panel_start_col = panel * NR;

        if b_cols - panel_start_col >= NR {
            // Optimized loop for panels that don't need any padding
            let b_offset =
                rows.start * b_row_stride + (cols.start + panel_start_col) * b_col_stride;

            assert!(
                b_data.len() > b_offset + (b_rows - 1) * b_row_stride + (NR - 1) * b_col_stride
            );

            // Optimize for common case of unit stride, as this makes the inner
            // loop a simple memcpy for which the compiler generates much better
            // code.
            if b_col_stride == 1 {
                for row in 0..b_rows {
                    let in_offset = b_offset + row * b_row_stride;
                    for col in 0..NR {
                        // Safety: Indexes are less than lengths asserted above.
                        unsafe {
                            out.write_unchecked(*b_data.get_unchecked(in_offset + col));
                        }
                    }
                }
            } else {
                for row in 0..b_rows {
                    let in_offset = b_offset + row * b_row_stride;
                    for col in 0..NR {
                        // Safety: Indexes are less than lengths asserted above.
                        unsafe {
                            out.write_unchecked(
                                *b_data.get_unchecked(in_offset + col * b_col_stride),
                            );
                        }
                    }
                }
            }
        } else {
            // Fallback for final panel if padding is required
            for row in 0..b_rows {
                let b_row_offset = (rows.start + row) * b_row_stride;

                for col in 0..NR {
                    let out_col = panel_start_col + col;
                    let b_offset =
                        b_row_offset + (cols.start + panel_start_col + col) * b_col_stride;

                    unsafe {
                        out.write_unchecked(if out_col < b_cols {
                            *b_data.get_unchecked(b_offset)
                        } else {
                            T::default()
                        });
                    }
                }
            }
        }
    }
    assert!(out.completed());
}

// Element type used by [`PackingBuffer`]. This must have an alignment that is
// at least as large as the alignment required by any of the kernels.
pub type PackElem = u32;

/// Buffer used for storing a block of a packed matrix.
///
/// The data type and layout of the contents is determined by the GEMM kernel,
/// subject to the constraints:
///
///  - There is a maximum alignment the kernel can request. See [`PackElem`].
///  - The stored data must all be plain `Copy` types for which any bit pattern
///    is valid.
#[derive(Clone)]
pub struct PackingBuffer {
    buf: Vec<PackElem>,
    used_len: usize,
}

impl PackingBuffer {
    /// Construct an empty packing buffer.
    ///
    /// No allocation happens until `alloc` is called.
    pub const fn new() -> PackingBuffer {
        PackingBuffer {
            buf: Vec::new(),
            used_len: 0,
        }
    }

    /// Clear the buffer and reserve space for a packed input.
    ///
    /// Returns an uninitialized slice of `layout.size()` bytes which the
    /// caller must fill.
    pub fn alloc(&mut self, size: usize, align: usize) -> &mut [MaybeUninit<u8>] {
        assert!(align <= align_of::<PackElem>());

        let buf_len = size.div_ceil(size_of::<PackElem>());
        self.buf.clear();
        self.buf.reserve(buf_len);
        self.used_len = 0;

        let uninit_data = &mut self.buf.spare_capacity_mut()[..buf_len];
        cast_pod_mut_slice(uninit_data).unwrap()
    }

    /// Clear the buffer and allocate a new one using `alloc`.
    ///
    /// When the packing buffer is no longer needed it can be extracted using
    /// [`into_vec`](Self::into_vec) to be returned to the pool that `alloc`
    /// allocates from.
    pub fn alloc_in<A: Alloc>(
        &mut self,
        alloc: A,
        size: usize,
        align: usize,
    ) -> &mut [MaybeUninit<u8>] {
        assert!(align <= align_of::<PackElem>());

        let buf_len = size.div_ceil(size_of::<PackElem>());
        self.buf = alloc.alloc::<PackElem>(buf_len);
        self.used_len = 0;

        let uninit_data = &mut self.buf.spare_capacity_mut()[..buf_len];
        cast_pod_mut_slice(uninit_data).unwrap()
    }

    /// Set the number of bytes in the buffer which have been initialized.
    pub unsafe fn set_len(&mut self, initialized_len: usize) {
        let rounded_len = initialized_len.next_multiple_of(size_of::<PackElem>());
        assert_eq!(rounded_len, initialized_len);

        let buf_len = rounded_len / size_of::<PackElem>();
        assert!(buf_len <= self.buf.capacity());
        self.buf.set_len(buf_len);
        self.used_len = initialized_len;
    }

    /// Return the contents of the buffer as a slice of bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &cast_pod_slice(&self.buf).unwrap()[..self.used_len]
    }

    /// Extract the buffer from self.
    pub fn into_vec(self) -> Vec<PackElem> {
        self.buf
    }
}

impl Default for PackingBuffer {
    fn default() -> Self {
        PackingBuffer::new()
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::{PackedLayout, PackingBuffer};

    #[test]
    fn test_packing_buffer() {
        struct Case {
            size: usize,
            align: usize,
            panel_stride: usize,
        }

        let cases = [Case {
            size: 256,
            align: 4,
            panel_stride: 64,
        }];

        for Case {
            size,
            align,
            panel_stride,
        } in cases
        {
            let mut buf = PackingBuffer::new();
            assert_eq!(buf.as_bytes().len(), 0);

            let layout = PackedLayout::new(size, align, panel_stride);
            let uninit_data = buf.alloc(layout.size(), layout.align());
            assert_eq!(uninit_data.len(), layout.size());

            uninit_data.fill(MaybeUninit::new(0));

            unsafe {
                buf.set_len(layout.size());
            }

            assert_eq!(buf.as_bytes().len(), layout.size());
        }
    }
}
