use std::mem::MaybeUninit;
use std::ops::Range;

use rten_tensor::{Matrix, MatrixLayout};

use super::round_up;
use super::Kernel;

/// Pack a block of the "A" matrix for use by kernel K.
///
/// The packed buffer is laid out as a sequence of `ceil(rows.len() / K::MR)`
/// row panels. Each row panel has size `K::MR * cols.len()` and uses
/// column-major order. If `rows.len()` is not a multiple of `K::MR`, the
/// final panel is zero-padded.
///
/// # Safety
///
/// When this function returns, all elements of `out` will have been initialized
/// either to a value from `a`, or zero.
pub fn pack_a_block<K: Kernel>(
    out: &mut [MaybeUninit<f32>],
    a: Matrix,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    let a_rows = rows.len();
    let a_cols = cols.len();

    // Safety: Loops below must only access valid offsets in `a_data`.
    let a_data = a.non_contiguous_data();

    let row_stride = a.row_stride();
    let col_stride = a.col_stride();

    let n_panels = round_up(a_rows, K::MR) / K::MR;
    for panel in 0..n_panels {
        let panel_offset = panel * a_cols * K::MR;
        let panel_start_row = panel * K::MR;

        if a_rows - panel_start_row >= K::MR {
            // Optimized loop for panels that don't need any padding
            let a_offset = (rows.start + panel_start_row) * row_stride + cols.start * col_stride;

            assert!(out.len() > panel_offset + (a_cols - 1) * K::MR + K::MR - 1);
            assert!(a_data.len() > a_offset + (K::MR - 1) * row_stride + (a_cols - 1) * col_stride);

            // Optimize for common case of unit stride as this generates better
            // code.
            if col_stride == 1 {
                for col in 0..a_cols {
                    for row in 0..K::MR {
                        // Safety: Indexes are less than lengths asserted above.
                        unsafe {
                            out.get_unchecked_mut(panel_offset + col * K::MR + row)
                                .write(*a_data.get_unchecked(a_offset + row * row_stride + col));
                        }
                    }
                }
            } else {
                for col in 0..a_cols {
                    for row in 0..K::MR {
                        // Safety: Indexes are less than lengths asserted above.
                        unsafe {
                            out.get_unchecked_mut(panel_offset + col * K::MR + row)
                                .write(
                                    *a_data.get_unchecked(
                                        a_offset + row * row_stride + col * col_stride,
                                    ),
                                );
                        }
                    }
                }
            }
        } else {
            // Fallback for final panel if padding is required
            for col in 0..a_cols {
                let out_col_offset = panel_offset + col * K::MR;
                for row in 0..K::MR {
                    let a_row = rows.start + panel_start_row + row;
                    out[out_col_offset + row].write(if a_row < rows.end {
                        a_data[a_row * row_stride + (cols.start + col) * col_stride]
                    } else {
                        0.0
                    });
                }
            }
        }
    }

    // Initialize any spare capacity in the buffer.
    let n_init = n_panels * a_cols * K::MR;
    for x in &mut out[n_init..] {
        x.write(0.);
    }
}

/// Pack a block of the "B" matrix for use by kernel K.
///
/// The packed buffer is laid out as a sequence of `ceil(cols.len() /
/// K::NR)` column panels. Each column panel has size `rows.len() *
/// K::NR` and uses row-major order. If `cols.len()` is not a multiple of
/// `K::NR`, the final panel is zero-padded.
///
/// # Safety
///
/// When this function returns, all elements of `out` will have been initialized
/// either to a value from `b`, or zero.
pub fn pack_b_block<K: Kernel>(
    out: &mut [MaybeUninit<f32>],
    b: Matrix,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    let b_cols = cols.len();
    let b_rows = rows.len();
    let b_row_stride = b.row_stride();
    let b_col_stride = b.col_stride();

    // Safety: Loops below must only access valid offsets in `b_data`.
    let b_data = b.non_contiguous_data();

    let n_panels = round_up(b_cols, K::NR) / K::NR;
    for panel in 0..n_panels {
        let panel_offset = panel * b_rows * K::NR;
        let panel_start_col = panel * K::NR;

        if b_cols - panel_start_col >= K::NR {
            // Optimized loop for panels that don't need any padding
            let b_offset =
                rows.start * b_row_stride + (cols.start + panel_start_col) * b_col_stride;

            assert!(out.len() >= panel_offset + (b_rows - 1) * K::NR + K::NR);
            assert!(
                b_data.len() > b_offset + (b_rows - 1) * b_row_stride + (K::NR - 1) * b_col_stride
            );

            // Optimize for common case of unit stride, as this makes the inner
            // loop a simple memcpy for which the compiler generates much better
            // code.
            if b_col_stride == 1 {
                for row in 0..b_rows {
                    let out_offset = panel_offset + row * K::NR;
                    let in_offset = b_offset + row * b_row_stride;
                    for col in 0..K::NR {
                        // Safety: Indexes are less than lengths asserted above.
                        unsafe {
                            out.get_unchecked_mut(out_offset + col)
                                .write(*b_data.get_unchecked(in_offset + col));
                        }
                    }
                }
            } else {
                for row in 0..b_rows {
                    let out_offset = panel_offset + row * K::NR;
                    let in_offset = b_offset + row * b_row_stride;
                    for col in 0..K::NR {
                        // Safety: Indexes are less than lengths asserted above.
                        unsafe {
                            out.get_unchecked_mut(out_offset + col)
                                .write(*b_data.get_unchecked(in_offset + col * b_col_stride));
                        }
                    }
                }
            }
        } else {
            // Fallback for final panel if padding is required
            for row in 0..b_rows {
                let out_row_offset = panel_offset + row * K::NR;
                let b_row_offset = (rows.start + row) * b_row_stride;

                for col in 0..K::NR {
                    let out_col = panel_start_col + col;
                    let b_offset =
                        b_row_offset + (cols.start + panel_start_col + col) * b_col_stride;

                    out[out_row_offset + col].write(if out_col < b_cols {
                        b_data[b_offset]
                    } else {
                        0.0
                    });
                }
            }
        }
    }

    // Initialize any spare capacity in the buffer.
    let n_init = n_panels * b_rows * K::NR;
    for x in &mut out[n_init..] {
        x.write(0.);
    }
}
