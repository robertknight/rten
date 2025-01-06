use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::{SimdInt, SimdMask};

use rten_tensor::{NdTensorView, Storage};

/// Maps rows of an [`Im2Col`] matrix to locations in the source image.
///
/// For efficiency when packing the image, the locations are premultiplied by
/// the corresponding stride.
pub struct RowOffsets {
    /// Map of row index to `channel * channel_stride`.
    pub chan: Vec<i32>,

    /// Map of row index to `row * row_stride`.
    pub y: Vec<i32>,

    /// Map of row index to `col * col_stride`.
    pub x: Vec<i32>,
}

/// Maps columns of an [`Im2Col`] matrix to locations in the source image.
///
/// For efficiency when packing the image, the locations are premultiplied by
/// the corresponding stride.
pub struct ColOffsets {
    /// Map of column index to `row * row_stride`.
    pub y: Vec<i32>,

    /// Map of column index to `col * col_stride`.
    pub x: Vec<i32>,
}

/// A matrix formed by unrolling patches of an image into columns.
///
/// The input image has shape [C,H,W] and is transformed into a matrix with
/// shape [C * Kh * kW, Oh * Ow] where Kh/Kw are convolution kernel sizes and
/// Oh/Ow are the number of patches in the Y and X directions.
///
/// The matrix is _virtual_ as it is not materialized fully in memory. Instead
/// blocks of the matrix are materialized during computation.
pub struct Im2Col<'a, T> {
    pub image: NdTensorView<'a, T, 3>,

    /// Map of im2col row index to input image coordinate, premultiplied with
    /// the corresponding stride.
    pub row_offsets: RowOffsets,

    /// Map of im2col column index to input image coordinate, premultiplied with
    /// the corresponding stride. The length of arrays in `col_offsets` is
    /// rounded up to the nearest multiple of the panel width. `n_cols` contains
    /// the actual number of columns in the virtual matrix.
    pub col_offsets: ColOffsets,

    /// Number of columns in the im2col matrix.
    pub n_cols: usize,

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
        self.row_offsets.chan.len()
    }

    /// Return the number of columns in the im2col matrix.
    pub fn cols(&self) -> usize {
        self.n_cols
    }

    /// Pack part of an image into a packing buffer.
    ///
    /// `NR_REGS` specifies the width of each column panel as a multiple of
    /// `S::LEN` elements. In other words, `panel_width` must exactly equal
    /// `NR_REGS * S::LEN`.
    ///
    /// # Safety
    ///
    /// Caller must ensure SIMD type is supported.
    #[inline(always)]
    pub(super) unsafe fn pack_block<S: SimdInt, const NR_REGS: usize>(
        &self,
        out: &mut [MaybeUninit<T>],
        panel_width: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        assert_eq!(panel_width, S::LEN * NR_REGS);

        let col_range = cols.start..cols.end.next_multiple_of(panel_width);
        let used_size = rows.len() * col_range.len();
        assert_eq!(out.len(), used_size);

        let col_y_offsets = &self.col_offsets.y[col_range.clone()];
        let col_x_offsets = &self.col_offsets.x[col_range.clone()];
        let row_chan_offsets = &self.row_offsets.chan[rows.clone()];
        let row_y_offsets = &self.row_offsets.y[rows.clone()];
        let row_x_offsets = &self.row_offsets.x[rows.clone()];

        let img_ptr = self.image.storage().as_ptr();

        // Compute max valid image buffer offset. Used to clamp generated offsets
        // as a form of bounds check.
        let img_len = self.image.storage().len();
        assert!(img_len > 0 && img_len <= i32::MAX as usize);
        let max_img_offset = S::splat(img_len as i32 - 1);

        // Loop over column panels, then rows, then `S::LEN`-wide column groups
        // within each panel.
        let out_ptr = out.as_mut_ptr();
        let mut out_offset = 0;

        for start_col in (0..col_y_offsets.len()).step_by(S::LEN * NR_REGS) {
            let col_y_offset: [S; NR_REGS] = std::array::from_fn(|i| {
                S::load(col_y_offsets.as_ptr().add(start_col + S::LEN * i))
            });
            let col_x_offset: [S; NR_REGS] = std::array::from_fn(|i| {
                S::load(col_x_offsets.as_ptr().add(start_col + S::LEN * i))
            });
            let max_x_offset = S::splat(self.max_x_offset);
            let max_y_offset = S::splat(self.max_y_offset);

            for ((&row_chan_offset, &row_y_offset), &row_x_offset) in row_chan_offsets
                .iter()
                .zip(row_y_offsets.iter())
                .zip(row_x_offsets.iter())
            {
                let row_chan_offset = S::splat(row_chan_offset);
                let row_y_offset = S::splat(row_y_offset);
                let row_x_offset = S::splat(row_x_offset);

                for i in 0..NR_REGS {
                    let y_offset = col_y_offset[i].add(row_y_offset);
                    let x_offset = col_x_offset[i].add(row_x_offset);
                    let offsets = row_chan_offset
                        .add(y_offset)
                        .add(x_offset)
                        // Ensure offsets cannot be out of bounds even if row /
                        // column offsets were calculated incorrectly.
                        .max(S::zero())
                        .min(max_img_offset);

                    // Create mask to specify offsets which are valid. Others
                    // correspond to the padding region.
                    let zero = S::zero();
                    let pad_mask = y_offset
                        .ge(zero)
                        .and(y_offset.le(max_y_offset))
                        .and(x_offset.ge(zero))
                        .and(x_offset.le(max_x_offset));

                    // Set offsets to zero for padding elements. We require
                    // this offset is always valid.
                    let offsets_array = zero.blend(offsets, pad_mask).to_array();
                    let pad_mask_array = pad_mask.to_array();

                    // Gather elements and store in packing buffer.
                    for idx in 0..S::LEN {
                        let out_ptr: *mut T = std::mem::transmute(out_ptr.add(out_offset + idx));

                        // Safety: Offsets are clamped so they must be in-bounds.
                        let src_elem = *img_ptr.add(offsets_array[idx] as usize);

                        // This should be compiled to a conditional move.
                        let elem = if pad_mask_array[idx] {
                            src_elem
                        } else {
                            T::default()
                        };

                        out_ptr.write(elem);
                    }

                    out_offset += S::LEN;
                }
            }
        }

        // Check we initialized as many elements as used.
        assert_eq!(out_offset, used_size);
    }
}
