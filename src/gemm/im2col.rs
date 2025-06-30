use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::ops::{MaskOps, NumOps};
use rten_simd::{Isa, Mask, Simd};
use rten_tensor::{NdTensorView, Storage};

use super::packing::int8::shift_cast_i8_u8;
use crate::slice_cast::cast_uninit_pod_mut_slice;

/// Maps rows of an [`Im2Col`] matrix to locations in the source image.
///
/// For efficiency when packing the image, the locations are premultiplied by
/// the corresponding stride.
pub struct RowOffsets {
    /// Map of row index to `channel * channel_stride`.
    pub chan: Vec<i32>,

    /// Map of row index to `kernel_y * dilation_y * row_stride`.
    pub y: Vec<i32>,

    /// Map of row index to `kernel_x * dilation_x * col_stride`.
    pub x: Vec<i32>,
}

/// Maps columns of an [`Im2Col`] matrix to locations in the source image.
///
/// For efficiency when packing the image, the locations are premultiplied by
/// the corresponding stride.
pub struct ColOffsets {
    /// Map of column index to `row * row_stride` where `row` is the top Y
    /// coordinate of the patch in the source image.
    pub y: Vec<i32>,

    /// Map of column index to `col * col_stride` where `col` is the left X
    /// coordinate of the patch in the source image.
    pub x: Vec<i32>,
}

/// A matrix formed by unrolling patches of an image into columns.
///
/// Each column of the matrix corresponds to a different spatial patch of the
/// image, and each row is a different location within the patch. The matrix
/// can be used as the right-hand input of a matrix multiplication in order
/// to perform a convolution.
///
/// The input image has shape [C,H,W] and is transformed into a matrix with
/// shape [C * Kh * kW, Oh * Ow] where Kh/Kw are convolution kernel sizes and
/// Oh/Ow are the number of patches in the Y and X directions. Given a weight
/// matrix W of shape `[M, C * Kh * kW]` the matrix multiplication `W @
/// im2col(image)` produces an output of shape `[M, Oh * Ow]` which can be
/// reshaped into the convolution output `[M, Oh, Ow]`.
///
/// The matrix is _virtual_ as it is not materialized fully in memory. Instead
/// blocks of the matrix are materialized during computation.
pub struct Im2Col<'a, T> {
    pub image: NdTensorView<'a, T, 3>,

    /// Map of im2col row index to position within image patch (channel,
    /// kernel_y, kernel_x) pre-multiplied by corresponding stride.
    ///
    /// The arrays may be padded to a multiple of a step size specified by the
    /// GEMM kernel. `n_rows` contains the actual number of rows in the virtual
    /// matrix.
    pub row_offsets: RowOffsets,

    /// Map of im2col column index to (y, x) coordinate of top-level corner of
    /// patch in image, pre-multiplied by corresponding stride.
    ///
    /// The arrays may be padded to a multiple of a step size specified by the
    /// GEMM kernel. `n_cols` contains the actual number of columns in the
    /// virtual matrix.
    pub col_offsets: ColOffsets,

    /// Number of columns in the im2col matrix.
    pub n_cols: usize,

    /// Number of rows in the im2col matrix.
    pub n_rows: usize,

    /// Maximum valid sum of `row_offsets.y + col_offsets.y`. Values above this
    /// correspond to the padding region.
    pub max_y_offset: i32,

    /// Maximum valid sum of `row_offsets.x + col_offsets.x`. Values above this
    /// correspond to the padding region.
    pub max_x_offset: i32,
}

impl<T: Copy + Default> Im2Col<'_, T> {
    /// Return the number of rows in the im2col matrix.
    pub fn rows(&self) -> usize {
        self.n_rows
    }

    /// Return the number of columns in the im2col matrix.
    pub fn cols(&self) -> usize {
        self.n_cols
    }

    /// Pack part of an image into a packing buffer.
    ///
    /// This method is for use by kernels using the "standard" packing buffer
    /// layout for the B / RHS input.
    ///
    /// `NR_REGS` specifies the width of each column panel as a multiple of
    /// `S::LEN` elements. In other words, `panel_width` must exactly equal
    /// `NR_REGS * S::LEN`.
    #[inline(always)]
    pub(super) fn pack_block<I: Isa, const NR_REGS: usize>(
        &self,
        isa: I,
        out: &mut [MaybeUninit<T>],
        panel_width: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        let ops = isa.i32();
        let mask_ops = ops.mask_ops();

        assert_eq!(panel_width, ops.len() * NR_REGS);

        let col_range = cols.start..cols.end.next_multiple_of(panel_width);
        let used_size = rows.len() * col_range.len();
        assert_eq!(out.len(), used_size);

        let col_y_offsets = &self.col_offsets.y[col_range.clone()];
        let col_x_offsets = &self.col_offsets.x[col_range.clone()];
        let row_chan_offsets = &self.row_offsets.chan[rows.clone()];
        let row_y_offsets = &self.row_offsets.y[rows.clone()];
        let row_x_offsets = &self.row_offsets.x[rows.clone()];

        let img_data = self.image.storage();

        // Compute max valid image buffer offset. Used to clamp generated offsets
        // as a form of bounds check.
        let img_len = self.image.storage().len();
        assert!(img_len > 0 && img_len <= i32::MAX as usize);
        let max_img_offset = ops.splat(img_len as i32 - 1);

        // Loop over column panels, then rows, then SIMD-wide column groups
        // within each panel.
        let mut out_offset = 0;

        for start_col in (0..col_y_offsets.len()).step_by(ops.len() * NR_REGS) {
            let col_y_offset: [I::I32; NR_REGS] =
                std::array::from_fn(|i| ops.load(&col_y_offsets[start_col + ops.len() * i..]));
            let col_x_offset: [I::I32; NR_REGS] =
                std::array::from_fn(|i| ops.load(&col_x_offsets[start_col + ops.len() * i..]));
            let max_x_offset = ops.splat(self.max_x_offset);
            let max_y_offset = ops.splat(self.max_y_offset);

            for ((&row_chan_offset, &row_y_offset), &row_x_offset) in row_chan_offsets
                .iter()
                .zip(row_y_offsets.iter())
                .zip(row_x_offsets.iter())
            {
                let row_chan_offset = ops.splat(row_chan_offset);
                let row_y_offset = ops.splat(row_y_offset);
                let row_x_offset = ops.splat(row_x_offset);

                for i in 0..NR_REGS {
                    let y_offset = ops.add(col_y_offset[i], row_y_offset);
                    let x_offset = ops.add(col_x_offset[i], row_x_offset);

                    let offsets = ops.add(ops.add(row_chan_offset, y_offset), x_offset);

                    // Ensure offsets cannot be out of bounds even if row /
                    // column offsets were calculated incorrectly.
                    let offsets = ops.min(ops.max(offsets, ops.zero()), max_img_offset);

                    // Create mask to specify offsets which are valid. Others
                    // correspond to the padding region.
                    let zero = ops.zero();

                    let y_valid =
                        mask_ops.and(ops.ge(y_offset, zero), ops.le(y_offset, max_y_offset));
                    let x_valid =
                        mask_ops.and(ops.ge(x_offset, zero), ops.le(x_offset, max_x_offset));
                    let pad_mask = mask_ops.and(y_valid, x_valid);

                    // Set offsets to zero for padding elements. We require
                    // this offset is always valid.
                    let offsets_array = ops.select(offsets, zero, pad_mask).to_array();
                    let pad_mask_array = pad_mask.to_array();

                    // Gather elements and store in packing buffer.
                    for idx in 0..ops.len() {
                        // Safety: offsets_array[idx] is a valid offset.
                        let src_elem =
                            unsafe { *img_data.get_unchecked(offsets_array[idx] as usize) };

                        // This should be compiled to a conditional move.
                        let elem = if pad_mask_array[idx] {
                            src_elem
                        } else {
                            T::default()
                        };

                        // Safety: `out_offset + i` is valid for `i < ops.len()`.
                        let out_el = unsafe { out.get_unchecked_mut(out_offset + idx) };
                        out_el.write(elem);
                    }

                    out_offset += ops.len();
                }
            }
        }

        // Check we initialized as many elements as used.
        assert_eq!(out_offset, used_size);
    }
}

impl Im2Col<'_, i8> {
    /// Pack part of an image into a packing buffer.
    ///
    /// This method is for use by kernels using int8 dot product instructions
    /// to compute `S::LEN x i32` dot products from two input vectors each
    /// containing `S::LEN x 4 x i8` (or u8) inputs.
    #[inline(always)]
    #[allow(unused)] // Some architectures only
    pub(super) fn pack_block_i8_dot<I: Isa, const NR_REGS: usize>(
        &self,
        isa: I,
        out: &mut [MaybeUninit<i8>],
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        self.pack_block_int8::<_, NR_REGS, false>(isa, out, rows, cols);
    }

    /// Variant of [`pack_block_i8_dot`](Self::pack_block_i8_dot) which shifts
    /// i8 values to u8 by adding 128.
    #[inline(always)]
    #[allow(unused)] // Some architectures only
    pub(super) fn pack_block_i8_dot_cast_u8<I: Isa, const NR_REGS: usize>(
        &self,
        isa: I,
        out: &mut [MaybeUninit<u8>],
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        let out = cast_uninit_pod_mut_slice(out).unwrap();
        self.pack_block_int8::<_, NR_REGS, true>(isa, out, rows, cols);
    }

    #[inline(always)]
    fn pack_block_int8<I: Isa, const NR_REGS: usize, const CAST_B_U8: bool>(
        &self,
        isa: I,
        out: &mut [MaybeUninit<i8>],
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        let ops = isa.i32();
        let mask_ops = ops.mask_ops();

        const K_TILE: usize = size_of::<i32>() / size_of::<i8>();

        debug_assert!(rows.end <= self.rows());
        debug_assert!(cols.end <= self.cols());

        let max_x_offset = ops.splat(self.max_x_offset);
        let max_y_offset = ops.splat(self.max_y_offset);

        let col_x_offsets = &self.col_offsets.x;
        debug_assert_eq!(col_x_offsets.len() % ops.len(), 0);

        let col_y_offsets = &self.col_offsets.y;
        debug_assert_eq!(col_y_offsets.len() % ops.len(), 0);

        let row_x_offsets = &self.row_offsets.x;
        debug_assert_eq!(row_x_offsets.len() % K_TILE, 0);

        let row_y_offsets = &self.row_offsets.y;
        debug_assert_eq!(row_y_offsets.len() % K_TILE, 0);

        let row_chan_offsets = &self.row_offsets.chan;
        debug_assert_eq!(row_chan_offsets.len() % K_TILE, 0);

        let img_data = self.image.storage();

        let mut out_offset = 0;

        for start_col in cols.step_by(ops.len() * NR_REGS) {
            let col_y_offset: [I::I32; NR_REGS] =
                std::array::from_fn(|i| ops.load(&col_y_offsets[start_col + i * ops.len()..]));
            let col_x_offset: [I::I32; NR_REGS] =
                std::array::from_fn(|i| ops.load(&col_x_offsets[start_col + i * ops.len()..]));
            let zero = ops.zero();

            let mut col_sums = [ops.zero().to_array(); NR_REGS];

            for start_row in rows.clone().step_by(4) {
                for i in 0..K_TILE {
                    let k = start_row + i;
                    let row_x_offset = ops.splat(unsafe { *row_x_offsets.get_unchecked(k) });
                    let row_y_offset = ops.splat(unsafe { *row_y_offsets.get_unchecked(k) });
                    let row_chan_offset = ops.splat(unsafe { *row_chan_offsets.get_unchecked(k) });

                    for c_block in 0..NR_REGS {
                        let x_offsets = ops.add(row_x_offset, col_x_offset[c_block]);
                        let y_offsets = ops.add(row_y_offset, col_y_offset[c_block]);
                        let offsets = ops.add(ops.add(x_offsets, y_offsets), row_chan_offset);

                        let y_valid =
                            mask_ops.and(ops.ge(y_offsets, zero), ops.le(y_offsets, max_y_offset));
                        let x_valid =
                            mask_ops.and(ops.ge(x_offsets, zero), ops.le(x_offsets, max_x_offset));
                        let pad_mask = mask_ops.and(y_valid, x_valid);
                        let pad_mask_array = pad_mask.to_array();

                        // Set offsets to zero for padding elements. We require
                        // this offset is always valid.
                        let offsets_array = ops.select(offsets, zero, pad_mask).to_array();

                        for idx in 0..ops.len() {
                            let out_elem = unsafe {
                                out.get_unchecked_mut(
                                    out_offset + (c_block * ops.len() + idx) * K_TILE + i,
                                )
                            };
                            let src_elem =
                                unsafe { *img_data.get_unchecked(offsets_array[idx] as usize) };

                            if CAST_B_U8 {
                                let src_elem = shift_cast_i8_u8(src_elem);
                                let elem = if pad_mask_array[idx] { src_elem } else { 0 };
                                col_sums[c_block][idx] += elem as i32;
                                out_elem.write(elem as i8);
                            } else {
                                let elem = if pad_mask_array[idx] { src_elem } else { 0 };
                                col_sums[c_block][idx] += elem as i32;
                                out_elem.write(elem);
                            }
                        }
                    }
                }
                out_offset += ops.len() * NR_REGS * K_TILE;
            }

            // Store column sums at end of each panel.
            for c_block in 0..NR_REGS {
                let col_sum_ptr = unsafe { out.as_mut_ptr().add(out_offset) as *mut i32 };
                for i in 0..ops.len() {
                    unsafe {
                        *col_sum_ptr.add(i) = col_sums[c_block][i];
                    }
                }
                out_offset += ops.len() * K_TILE;
            }
        }

        // Sanity check
        assert_eq!(out_offset, out.len());
    }
}
