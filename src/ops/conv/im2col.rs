use rten_gemm::{ColOffsets, Im2Col, RowOffsets};
use rten_tensor::NdTensorView;
use rten_tensor::prelude::*;

use crate::ops::Padding;
use crate::ops::pooling::{RoundMode, calc_output_size_and_padding};

/// Build a virtual [`Im2Col`] matrix from an image and convolution parameters.
///
/// The number of columns in the matrix is padded to a multiple of `col_count_step`.
pub fn build_im2col<T>(
    image: NdTensorView<T, 3>,
    kernel: [usize; 2],
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
    col_count_step: usize,
    row_count_step: usize,
) -> Im2Col<T> {
    // Ensure image has at least one cell.
    assert!(image.len() > 0);

    let [chans, h, w] = image.shape();
    let [k_h, k_w] = kernel;
    let [stride_h, stride_w] = strides;
    let [dilation_y, dilation_x] = dilations;
    let [pad_top, pad_left, _pad_bottom, _pad_right] = padding;
    let (y_patches, x_patches, _) = calc_output_size_and_padding(
        (h, w),
        (k_h, k_w),
        (stride_h, stride_w),
        Padding::Fixed(padding.into()),
        Some((dilation_y, dilation_x)),
        RoundMode::default(),
    )
    .expect("invalid im2col params");

    let [im_stride_c, im_stride_h, im_stride_w]: [i32; 3] =
        image.strides().map(|s| s.try_into().unwrap());

    // Build lookup table of row index in the virtual im2col matrix to
    // offsets in the image.
    let n_rows = chans * k_h * k_w;
    let n_rows_padded = n_rows.next_multiple_of(row_count_step);

    let mut row_chan_offsets = Vec::<i32>::with_capacity(n_rows_padded);
    let mut row_y_offsets = Vec::<i32>::with_capacity(n_rows_padded);
    let mut row_x_offsets = Vec::<i32>::with_capacity(n_rows_padded);
    for chan in 0..chans {
        // Offset to image channel
        row_chan_offsets.extend(std::iter::repeat(chan as i32 * im_stride_c).take(k_h * k_w));

        for k_y in 0..k_h {
            // Offset from top-left corner of patch
            row_y_offsets
                .extend(std::iter::repeat(im_stride_h * k_y as i32 * dilation_y as i32).take(k_w));
            row_x_offsets.extend(
                (0..k_w as i32)
                    .map(|k_x| im_stride_w * k_x * dilation_x as i32)
                    .take(k_w),
            );
        }
    }

    // Compute max valid X / Y offsets for testing whether an element is in
    // the padding region or not.
    let max_y_offset: i32 = ((image.size(1) - 1) * image.stride(1))
        .try_into()
        .expect("invalid im2col params");
    let max_x_offset: i32 = ((image.size(2) - 1) * image.stride(2))
        .try_into()
        .expect("invalid im2col params");

    for _ in n_rows..n_rows_padded {
        row_chan_offsets.push(0);
        row_x_offsets.push(max_x_offset + 1);
        row_y_offsets.push(max_y_offset + 1);
    }

    // Build lookup table of column index in the virtual im2col matrix to
    // offsets in the image.
    let n_cols = x_patches * y_patches;
    let n_cols_padded = n_cols.next_multiple_of(col_count_step);

    // Main loop for the used columns.
    let mut col_y_offsets = Vec::with_capacity(n_cols_padded);
    let mut col_x_offsets = Vec::with_capacity(n_cols_padded);
    for patch_y in 0..y_patches {
        let img_y = (patch_y as i32 * stride_h as i32) - pad_top as i32;
        col_y_offsets.extend(std::iter::repeat(img_y * im_stride_h).take(x_patches));
        col_x_offsets.extend((0..x_patches).map(|patch_x| {
            let img_x = (patch_x as i32 * stride_w as i32) - pad_left as i32;
            img_x * im_stride_w
        }));
    }

    // Remainder loop for columns added to pad count to a multiple of
    // `col_count_step`. This is slower as it uses divisions.
    for col in n_cols..n_cols_padded {
        let patch_y = col as i32 / x_patches as i32;
        let patch_x = col as i32 % x_patches as i32;
        let img_x = (patch_x * stride_w as i32) - pad_left as i32;
        let img_y = (patch_y * stride_h as i32) - pad_top as i32;
        col_y_offsets.push(img_y * im_stride_h);
        col_x_offsets.push(img_x * im_stride_w);
    }

    Im2Col {
        image,
        n_rows,
        n_cols,

        row_offsets: RowOffsets {
            chan: row_chan_offsets,
            y: row_y_offsets,
            x: row_x_offsets,
        },
        col_offsets: ColOffsets {
            y: col_y_offsets,
            x: col_x_offsets,
        },

        max_y_offset,
        max_x_offset,
    }
}
