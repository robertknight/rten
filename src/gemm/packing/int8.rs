//! Packing for int8 x int8 -> int32 matrix multiplications.
//!
//! These require the inputs to be packed into microtiles of size `[MR, 4]` for
//! the LHS or `[4, NR]` for the RHS. These can then be used with SIMD
//! instructions that compute `VEC_LEN x (4 x i8) -> VEC_LEN x i32` dot
//! products.

use std::mem::MaybeUninit;

use rten_tensor::prelude::*;
use rten_tensor::Matrix;

use super::PackedLayout;
use super::SliceWriter;

/// Size of micro tiles of K in the innermost dimension of packed layouts.
const K_TILE: usize = 4;

/// Return the layout of the packing buffer required by `pack_b`.
pub fn packed_b_layout<const NR: usize>(b_rows: usize, b_cols: usize) -> PackedLayout {
    // Packed block is padded to a multiple of NR columns and K_TILE rows.
    let n_panels = b_cols.div_ceil(NR);
    let packed_elements_size = b_rows.div_ceil(K_TILE) * NR * K_TILE;

    // At the end of the buffer are NR x i32 column sums.
    let col_sums_size = NR * 4;
    let panel_stride = packed_elements_size + col_sums_size;
    let size = n_panels * panel_stride;

    // Use i32 alignment for column sums
    let align = align_of::<i32>();

    PackedLayout::new(size, align, panel_stride)
}

// Pack blocks of the B matrix for use by the matmul kernel.
//
// Pack B matrix of shape `[K, N]` into a series of column panels. Each panel
// contains elements from a `[K, NR]` slice of the input and is laid out as `[K
// / 4, NR, 4]` u8 values, followed by `NR` i32 column sums.  In the kernel a
// transposed `[NR, 4]` microtile of `B` is then multiplied with a `[MR, 4]`
// microtile of `A` using dot product instructions. The column sums are used
// to handle subtraction of the zero point.
pub fn pack_b<const NR: usize>(out: &mut [MaybeUninit<i8>], b: Matrix<i8>) {
    let [b_rows, b_cols] = b.shape();
    assert_eq!(out.len(), packed_b_layout::<NR>(b_rows, b_cols).size());

    let mut out = SliceWriter::new(out);

    // Loop over column panels
    for col_tile in 0..b_cols.div_ceil(NR) {
        let mut col_sums = [0i32; NR];
        let col_range = col_tile * NR..(col_tile * NR + NR).min(b_cols);

        // Write panel elements
        for row_tile in 0..b_rows.div_ceil(K_TILE) {
            let row_range = row_tile * K_TILE..(row_tile * K_TILE + K_TILE).min(b_rows);
            if col_range.len() == NR && row_range.len() == K_TILE {
                // Full tile
                for c in 0..NR {
                    for r in 0..K_TILE {
                        let y = row_tile * K_TILE + r;
                        let x = col_tile * NR + c;
                        unsafe {
                            let val = *b.get_unchecked([y, x]);
                            col_sums[c] += val as i32;
                            out.write_unchecked(val);
                        }
                    }
                }
            } else {
                // Partial tile
                for c in 0..col_range.len() {
                    for r in 0..row_range.len() {
                        let y = row_tile * K_TILE + r;
                        let x = col_tile * NR + c;
                        unsafe {
                            let val = *b.get_unchecked([y, x]);
                            col_sums[c] += val as i32;
                            out.write_unchecked(val);
                        }
                    }
                    // Pad to row tile size
                    unsafe { out.write_n_unchecked(K_TILE - row_range.len(), 0) };
                }
                // Pad to column tile size
                unsafe { out.write_n_unchecked((NR - col_range.len()) * K_TILE, 0) };
            }
        }

        // Write column sums
        for c in 0..NR {
            let col_sum_i8 = col_sums[c].to_ne_bytes().map(|b| b as i8);
            for i in 0..4 {
                unsafe {
                    out.write_unchecked(col_sum_i8[i]);
                }
            }
        }
    }

    assert!(out.completed());
}

/// Return the layout of the packing buffer required by `pack_a`.
pub fn packed_a_layout<const MR: usize>(a_rows: usize, a_cols: usize) -> PackedLayout {
    // Packed block is padded to a multiple of MR rows and K_TILE columns.
    let n_panels = a_rows.div_ceil(MR);
    let packed_elements_size = a_cols.div_ceil(K_TILE) * MR * K_TILE;

    // At the end of the buffer are MR x i32 row sums.
    let row_sums_size = MR * 4;
    let panel_stride = packed_elements_size + row_sums_size;
    let size = n_panels * panel_stride;

    // Use i32 alignment for row sums
    let align = align_of::<i32>();

    PackedLayout::new(size, align, panel_stride)
}

// Pack blocks of the A matrix for use by the matmul kernel.
//
// Pack A matrix of shape `[M, K]` into a series of row panels. Each panel
// contains elements from an `[MR, K]` slice of the input and is laid out as `[K
// / 4, MR, 4]` u8 values, followed by `MR` i32 row sums. The row sums are
// used to handle subtraction of the zero point.
pub fn pack_a<const MR: usize>(out: &mut [MaybeUninit<u8>], a: Matrix<u8>) {
    let [a_rows, a_cols] = a.shape();
    assert_eq!(out.len(), packed_a_layout::<MR>(a_rows, a_cols).size());

    let mut out = SliceWriter::new(out);

    // Loop over row panels
    for row_tile in 0..a_rows.div_ceil(MR) {
        let mut row_sums = [0i32; MR];
        let row_range = row_tile * MR..(row_tile * MR + MR).min(a_rows);

        // Write panel elements
        for col_tile in 0..a_cols.div_ceil(K_TILE) {
            let col_range = col_tile * K_TILE..(col_tile * K_TILE + K_TILE).min(a_cols);

            if row_range.len() == MR && col_range.len() == K_TILE {
                // Full tile
                for r in 0..MR {
                    for c in 0..K_TILE {
                        let y = row_tile * MR + r;
                        let x = col_tile * K_TILE + c;
                        unsafe {
                            let val = *a.get_unchecked([y, x]);
                            row_sums[r] += val as i32;
                            out.write_unchecked(val);
                        }
                    }
                }
            } else {
                // Partial tile
                for r in 0..row_range.len() {
                    for c in 0..col_range.len() {
                        let y = row_tile * MR + r;
                        let x = col_tile * K_TILE + c;
                        unsafe {
                            let val = *a.get_unchecked([y, x]);
                            row_sums[r] += val as i32;
                            out.write_unchecked(val);
                        }
                    }
                    // Pad to column tile size
                    unsafe {
                        out.write_n_unchecked(K_TILE - col_range.len(), 0);
                    }
                }
                // Pad to row tile size
                unsafe {
                    out.write_n_unchecked((MR - row_range.len()) * K_TILE, 0);
                }
            }
        }

        // Write row sums
        for r in 0..MR {
            let row_sum_u8 = row_sums[r].to_ne_bytes();
            for i in 0..4 {
                unsafe {
                    out.write_unchecked(row_sum_u8[i]);
                }
            }
        }
    }

    assert!(out.completed());
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::NdTensor;

    use super::{pack_a, pack_b, packed_a_layout, packed_b_layout, K_TILE};

    const MR: usize = 8;
    const NR: usize = 8;

    #[test]
    fn test_pack_a() {
        let mut rng = XorShiftRng::new(5678);

        // Test packing with a range of input sizes and make sure it doesn't panic.
        for m in 1..MR * 2 {
            for k in 1..K_TILE * 2 {
                let mat = NdTensor::rand([m, k], &mut rng);
                let layout = packed_a_layout::<MR>(m, k);
                let mut buf = Vec::with_capacity(layout.size());

                pack_a::<MR>(&mut buf.spare_capacity_mut()[..layout.size()], mat.view());
            }
        }
    }

    #[test]
    fn test_pack_b() {
        let mut rng = XorShiftRng::new(5678);

        // Test packing with a range of input sizes and make sure it doesn't panic.
        for n in 1..NR * 2 {
            for k in 1..K_TILE * 2 {
                let mat = NdTensor::rand([k, n], &mut rng);
                let layout = packed_b_layout::<NR>(k, n);
                let mut buf = Vec::with_capacity(layout.size());

                pack_b::<NR>(&mut buf.spare_capacity_mut()[..layout.size()], mat.view());
            }
        }
    }
}
