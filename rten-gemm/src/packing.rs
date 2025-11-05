use std::mem::MaybeUninit;
use std::ops::Range;

use rten_base::byte_cast::{cast_pod_slice, cast_uninit_pod_mut_slice};
use rten_base::iter::range_chunks;
use rten_tensor::storage::Alloc;
use rten_tensor::{AssumeInit, Matrix, MatrixLayout, Storage};

use super::kernels::PackedLayout;
use crate::block_quant::{BlockQuantizedMatrix, nbit_zero_point};

pub mod int8;

/// Packs tiles of a matrix for use by a kernel.
pub trait Packer<'a> {
    /// Pack a tile of the input into an output buffer.
    fn pack(&self, out: &mut [MaybeUninit<u8>], rows: Range<usize>, cols: Range<usize>);
}

/// Helper for incrementally filling a slice.
struct SliceWriter<'a, T> {
    offset: usize,
    slice: &'a mut [MaybeUninit<T>],
}

impl<'a, T> SliceWriter<'a, T> {
    fn new(slice: &'a mut [MaybeUninit<T>]) -> Self {
        SliceWriter { slice, offset: 0 }
    }

    /// Return the initialized portion of the slice.
    fn into_slice(self) -> &'a mut [T] {
        let written = &mut self.slice[0..self.offset];
        unsafe { written.assume_init() }
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
        unsafe { self.slice.get_unchecked_mut(self.offset) }.write(val);
        self.offset += 1;
    }

    /// Write a slice of elements into the buffer.
    fn write_slice(&mut self, vals: &[T])
    where
        T: Copy,
    {
        assert!(self.offset + vals.len() <= self.slice.len());
        for (i, val) in vals.iter().enumerate() {
            // Safety: We checked there are at least `vals.len()` remaining elements in `self.slice`.
            unsafe { self.slice.get_unchecked_mut(self.offset + i).write(*val) };
        }
        self.offset += vals.len();
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
            unsafe { self.slice.get_unchecked_mut(self.offset + i) }.write(val);
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

/// Packer which dequantizes and packs an input matrix that is quantized to N
/// bits along the K dimension.
pub struct BlockQuantizedMatrixPacker<'a, T, const NR: usize> {
    mat: BlockQuantizedMatrix<'a, T>,
}

impl<'a, T: Copy, const NR: usize> BlockQuantizedMatrixPacker<'a, T, NR> {
    pub fn new(mat: BlockQuantizedMatrix<'a, T>) -> Self {
        Self { mat }
    }
}

impl<'a, const NR: usize> BlockQuantizedMatrixPacker<'a, f32, NR> {
    /// Dequantize and pack a region of an n-bit block-quantized matrix.
    ///
    /// The start and count of rows to pack must be a multiple of the matrix's
    /// [block size](BlockQuantizedMatrix::elements_per_block). `cols.start`
    /// must be a multiple of `NR`.
    #[inline] // Allow caller to control `target_feature`s
    fn pack(
        &self,
        out: &'a mut [MaybeUninit<f32>],
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> &'a mut [f32] {
        let n_panels = cols.len().next_multiple_of(NR) / NR;

        let block_size = self.mat.elements_per_block();
        assert!(rows.start.is_multiple_of(block_size) && rows.len().is_multiple_of(block_size));
        let n_blocks = rows.len() / block_size;
        let start_block = rows.start / block_size;

        // Only 4-bit elements are supported. With small changes this could
        // support 2 or 8-bits. ONNX Runtime's implementation only supports 2, 4
        // or 8 bits.
        const N_BITS: u8 = 4;
        const ZERO_POINT: i16 = nbit_zero_point(N_BITS);

        // Data for padding and scales if column count is not a multiple of NR.
        //
        // Values are chosen so that the dequantized values are zero in the
        // padding region.
        let mut pad_data: Vec<u8> = Vec::new();
        let mut pad_scales: Vec<f32> = Vec::new();

        if !cols.len().is_multiple_of(NR) {
            pad_data.resize(block_size * n_blocks, ZERO_POINT as u8);
            pad_scales.resize(n_blocks, 0.);
        }

        let block_bytes = self.mat.bytes_per_block();

        let mut out = SliceWriter::new(out);
        for panel in 0..n_panels {
            let start_col = cols.start + panel * NR;

            // Extract data and scales for column.
            let data: [&[u8]; NR] = std::array::from_fn(|col| {
                self.mat
                    .column_data(start_col + col, start_block, n_blocks)
                    .unwrap_or(&pad_data)
            });

            let scales: [&[f32]; NR] = std::array::from_fn(|col| {
                self.mat
                    .column_scales(start_col + col, start_block, n_blocks)
                    .unwrap_or(&pad_scales)
            });

            // Dequantize K blocks.
            for block_idx in 0..n_blocks {
                let block_scales = scales.map(|bs| *unsafe { bs.get_unchecked(block_idx) });

                for k in 0..block_bytes {
                    let bytes: [u8; NR] = std::array::from_fn(|c| unsafe {
                        *data[c].get_unchecked(block_idx * block_bytes + k)
                    });

                    // First row from low 4 bits.
                    for c in 0..NR {
                        let elem = (bytes[c] & 0x0F) as i16 - ZERO_POINT;
                        let dequant = elem as f32 * block_scales[c];
                        unsafe { out.write_unchecked(dequant) };
                    }

                    // Second row from high 4 bits.
                    for c in 0..NR {
                        let elem = (bytes[c] >> 4) as i16 - ZERO_POINT;
                        let dequant = elem as f32 * block_scales[c];
                        unsafe { out.write_unchecked(dequant) };
                    }
                }
            }
        }

        assert!(out.completed());
        out.into_slice()
    }
}

impl<'a, const NR: usize> Packer<'a> for BlockQuantizedMatrixPacker<'a, f32, NR> {
    fn pack(&self, out: &mut [MaybeUninit<u8>], rows: Range<usize>, cols: Range<usize>) {
        let out = cast_uninit_pod_mut_slice(out).unwrap();
        self.pack(out, rows, cols);
    }
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
        cast_uninit_pod_mut_slice(uninit_data).unwrap()
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
        cast_uninit_pod_mut_slice(uninit_data).unwrap()
    }

    /// Set the number of bytes in the buffer which have been initialized.
    pub unsafe fn set_len(&mut self, initialized_len: usize) {
        let rounded_len = initialized_len.next_multiple_of(size_of::<PackElem>());
        assert_eq!(rounded_len, initialized_len);

        let buf_len = rounded_len / size_of::<PackElem>();
        assert!(buf_len <= self.buf.capacity());
        unsafe { self.buf.set_len(buf_len) };
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
    use rten_tensor::{AsView, NdTensor, NdTensorView};
    use rten_testing::TestCases;
    use std::mem::MaybeUninit;

    use super::{BlockQuantizedMatrixPacker, PackedLayout, PackingBuffer};
    use crate::block_quant::{BlockQuantizedMatrix, nbit_zero_point, pack_4bit_elements};

    #[test]
    fn test_packing_buffer() {
        #[derive(Clone, Debug)]
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

        cases.test_each_clone(|case| {
            let Case {
                size,
                align,
                panel_stride,
            } = case;

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
        })
    }

    #[test]
    fn test_block_quantized_matrix_packer() {
        let cols = 4;
        let k_blocks = 2;
        let n_bits = 4;
        let zero_point = nbit_zero_point(n_bits);

        let elems: Vec<i8> = (-8..8).cycle().take(256).collect();
        let packed_elems = pack_4bit_elements(&elems, zero_point as i8);
        let data = NdTensor::from_data([cols, k_blocks, 16], packed_elems);
        let data = data.to_contiguous();
        let scales = NdTensorView::from_data([cols, k_blocks], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let scales = scales.to_contiguous();
        let mat = BlockQuantizedMatrix::new(data.view(), scales.view(), n_bits).unwrap();

        const NR: usize = 2;
        let packer = BlockQuantizedMatrixPacker::<f32, NR>::new(mat);

        // Dequantized column panels
        let mut dequantized = Vec::with_capacity(elems.len());
        let dequantized = packer.pack(
            dequantized.spare_capacity_mut(),
            0..mat.rows(),
            0..mat.cols(),
        );

        for panel in 0..cols / NR {
            for row in 0..mat.rows() {
                for panel_col in 0..NR {
                    let col = panel * NR + panel_col;
                    let dequant_offset = panel * mat.rows() * NR + row * NR + panel_col;
                    let input_offset = col * mat.rows() + row;

                    let block_idx = row / mat.elements_per_block();
                    let scale = mat.column_scales(col, block_idx, 1).unwrap()[0];

                    assert_eq!(
                        dequantized[dequant_offset],
                        elems[input_offset] as f32 * scale,
                        "mismatch at panel {} row {} col {}",
                        panel,
                        row,
                        panel_col
                    );
                }
            }
        }
    }
}
