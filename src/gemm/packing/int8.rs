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
use crate::slice_cast::cast_pod_slice;

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

/// Pack blocks of the B matrix for use by the matmul kernel.
///
/// Pack B matrix of shape `[K, N]` into a series of column panels. Each panel
/// contains elements from a `[K, NR]` slice of the input and is laid out as `[K
/// / 4, NR, 4]` u8 values, followed by `NR` i32 column sums.  In the kernel a
/// transposed `[NR, 4]` microtile of `B` is then multiplied with a `[MR, 4]`
/// microtile of `A` using dot product instructions. The column sums are used
/// to handle subtraction of the zero point.
#[allow(unused)]
pub fn pack_b<const NR: usize>(out: &mut [MaybeUninit<i8>], b: Matrix<i8>) {
    pack_b_impl::<NR, _>(
        out,
        b,
        |x| x,
        |out, col_sum| {
            let bytes = col_sum.to_ne_bytes();
            for i in 0..4 {
                unsafe {
                    out.write_unchecked(bytes[i] as i8);
                }
            }
        },
    )
}

/// Convert a byte from signed to unsigned and shift the value so that it
/// is the same distance from the minimum value.
///
/// For example `-125` (i8::MIN + 3) becomes `3` (u8::MIN + 3).
#[inline]
pub fn shift_cast_i8_u8(x: i8) -> u8 {
    x as u8 ^ 0x80
}

/// Variant of [`pack_b`] which converts `i8` values to `u8` values during
/// packing, shifting the values by 128 to preserve the position of each value
/// within the numeric range.
#[allow(unused)]
pub fn pack_b_cast_i8_u8<const NR: usize>(out: &mut [MaybeUninit<u8>], b: Matrix<i8>) {
    pack_b_impl::<NR, _>(out, b, shift_cast_i8_u8, |out, col_sum| {
        let bytes = col_sum.to_ne_bytes();
        for i in 0..4 {
            unsafe {
                out.write_unchecked(bytes[i]);
            }
        }
    })
}

/// A type with size and align of 1.
trait Byte: Copy + Default {}
impl Byte for u8 {}
impl Byte for i8 {}

fn pack_b_impl<const NR: usize, T: Byte>(
    out: &mut [MaybeUninit<T>],
    b: Matrix<i8>,
    cast: impl Fn(i8) -> T,
    write_col_sum: impl Fn(&mut SliceWriter<T>, i32),
) where
    i32: From<T>,
{
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
                            let val = cast(*b.get_unchecked([y, x]));
                            col_sums[c] += i32::from(val);
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
                            let val = cast(*b.get_unchecked([y, x]));
                            col_sums[c] += i32::from(val);
                            out.write_unchecked(val);
                        }
                    }
                    // Pad to row tile size
                    unsafe { out.write_n_unchecked(K_TILE - row_range.len(), T::default()) };
                }
                // Pad to column tile size
                unsafe { out.write_n_unchecked((NR - col_range.len()) * K_TILE, T::default()) };
            }
        }

        // Write column sums
        for c in 0..NR {
            write_col_sum(&mut out, col_sums[c]);
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

/// Extract the packed elements and row sums from a buffer packed by [`pack_a`].
pub fn extract_packed_a<const MR: usize>(a: &[u8]) -> (&[u8], &[i32; MR]) {
    let row_sum_offset = a.len() - MR * size_of::<i32>();
    let (packed_elements, row_sums) = a.split_at(row_sum_offset);
    let row_sums: &[i32] = cast_pod_slice(row_sums).unwrap();
    (packed_elements, row_sums.try_into().unwrap())
}

/// Extract the packed elements and column sums from a buffer packed by [`pack_b`].
pub fn extract_packed_b<const NR: usize>(b: &[u8]) -> (&[u8], &[i32; NR]) {
    let col_sum_offset = b.len() - NR * size_of::<i32>();
    let (packed_elements, col_sums) = b.split_at(col_sum_offset);
    let col_sums: &[i32] = cast_pod_slice(col_sums).unwrap();
    (packed_elements, col_sums.try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::{Matrix, MatrixLayout, NdTensor};

    use super::{
        extract_packed_a, extract_packed_b, pack_a, pack_b, pack_b_cast_i8_u8, packed_a_layout,
        packed_b_layout, K_TILE,
    };
    use crate::slice_cast::cast_pod_slice;

    const MR: usize = 8;
    const NR: usize = 8;

    fn pack_a_matrix(mat: Matrix<u8>) -> Vec<u8> {
        let layout = packed_a_layout::<MR>(mat.rows(), mat.cols());

        // Layout must have space for at least each element in the input, plus
        // row sums as i32 values.
        assert!(layout.size() >= mat.rows() * mat.cols() + mat.rows() * 4);

        let mut buf = Vec::with_capacity(layout.size());
        pack_a::<MR>(&mut buf.spare_capacity_mut()[..layout.size()], mat.view());

        // Safety: `pack_a` initialized `layout.size()` elements.
        unsafe { buf.set_len(layout.size()) }

        buf
    }

    fn pack_b_matrix(mat: Matrix<i8>) -> Vec<i8> {
        let layout = packed_b_layout::<NR>(mat.rows(), mat.cols());

        // Layout must have space for at least each element in the input, plus
        // column sums as i32 values.
        assert!(layout.size() >= mat.rows() * mat.cols() + mat.cols() * 4);

        let mut buf = Vec::with_capacity(layout.size());
        pack_b::<NR>(&mut buf.spare_capacity_mut()[..layout.size()], mat.view());

        // Safety: `pack_b` initialized `layout.size()` elements.
        unsafe { buf.set_len(layout.size()) }

        buf
    }

    fn pack_b_matrix_cast_u8<const NR: usize>(mat: Matrix<i8>) -> Vec<u8> {
        let layout = packed_b_layout::<NR>(mat.rows(), mat.cols());

        // Layout must have space for at least each element in the input, plus
        // column sums as i32 values.
        assert!(layout.size() >= mat.rows() * mat.cols() + mat.cols() * 4);

        let mut buf = Vec::with_capacity(layout.size());
        pack_b_cast_i8_u8::<NR>(&mut buf.spare_capacity_mut()[..layout.size()], mat.view());

        // Safety: `pack_b` initialized `layout.size()` elements.
        unsafe { buf.set_len(layout.size()) }

        buf
    }

    #[test]
    fn test_pack_a_various_sizes() {
        let mut rng = XorShiftRng::new(5678);
        for m in 1..MR * 2 {
            for k in 1..K_TILE * 2 {
                let mat = NdTensor::rand([m, k], &mut rng);
                pack_a_matrix(mat.view());
            }
        }
    }

    #[test]
    fn test_extract_packed_a() {
        let mat = NdTensor::<u8, 2>::from([[1, 2], [3, 4]]);
        let packed = pack_a_matrix(mat.view());

        let (packed_elems, row_sums) = extract_packed_a(&packed);

        assert!(packed_elems.len() >= mat.rows() * mat.cols());
        assert_eq!(row_sums, &[3, 7, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_pack_b_various_sizes() {
        let mut rng = XorShiftRng::new(5678);
        for n in 1..NR * 2 {
            for k in 1..K_TILE * 2 {
                let mat = NdTensor::rand([k, n], &mut rng);
                pack_b_matrix(mat.view());
            }
        }
    }

    #[test]
    fn test_extract_packed_b() {
        let mat = NdTensor::<i8, 2>::from([[1, 2], [3, 4]]);
        let packed = pack_b_matrix(mat.view());

        let (packed_elems, col_sums) = extract_packed_b(cast_pod_slice(&packed).unwrap());

        assert!(packed_elems.len() >= mat.rows() * mat.cols());
        assert_eq!(col_sums, &[4, 6, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_pack_b_cast_i8_u8() {
        let mat = NdTensor::<i8, 2>::from([[1, 2], [3, 4]]);
        let packed = pack_b_matrix_cast_u8::<2>(mat.view());

        let (packed_elems, col_sums) = extract_packed_b(&packed);

        assert!(packed_elems.len() >= mat.rows() * mat.cols());
        assert_eq!(packed_elems, &[129, 131, 0, 0, 130, 132, 0, 0]);
        assert_eq!(col_sums, &[129 + 131, 130 + 132]);
    }
}
