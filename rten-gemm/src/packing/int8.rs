//! Packing for int8 x int8 -> int32 matrix multiplications.
//!
//! These require the inputs to be packed into microtiles of size `[MR, 4]` for
//! the LHS or `[4, NR]` for the RHS. These can then be used with SIMD
//! instructions that compute `VEC_LEN x (4 x i8) -> VEC_LEN x i32` dot
//! products.

use std::mem::MaybeUninit;

use rten_base::byte_cast::{AsBytes, FromBytes};
use rten_tensor::layout::AsIndex;
use rten_tensor::prelude::*;
use rten_tensor::storage::ViewData;
use rten_tensor::{Layout, Matrix, NdIndices, NdLayout, TensorBase};

use super::PackedLayout;
use super::SliceWriter;

/// Metadata placed at the end of a packed MR x KC panel of the A input.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct PackedAMeta<const MR: usize> {
    /// Sum of elements in each row.
    pub row_sums: [i32; MR],

    /// Zero points for each row.
    pub zero_points: [i32; MR],
}

// Safety: PackedAMeta meets requirements for AsBytes, FromBytes.
unsafe impl<const MR: usize> AsBytes for PackedAMeta<MR> {}
unsafe impl<const MR: usize> FromBytes for PackedAMeta<MR> {}

/// Metadata placed at the end of a packed KC x NR panel of the B input.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct PackedBMeta<const NR: usize> {
    /// Sum of elements in each column.
    pub col_sums: [i32; NR],

    /// Zero points for each column.
    pub zero_points: [i32; NR],
}

// Safety: PackedAMeta meets requirements for AsBytes, FromBytes.
unsafe impl<const NR: usize> AsBytes for PackedBMeta<NR> {}
unsafe impl<const NR: usize> FromBytes for PackedBMeta<NR> {}

/// Return the layout of the packing buffer required by `pack_b`.
pub fn packed_b_layout<const NR: usize, const K_TILE: usize>(
    b_rows: usize,
    b_cols: usize,
) -> PackedLayout {
    // Packed block is padded to a multiple of NR columns and K_TILE rows.
    let n_panels = b_cols.div_ceil(NR);
    let packed_elements_size = b_rows.div_ceil(K_TILE) * NR * K_TILE;

    // PackedBMeta should have an align of 4, and packed_elements_size is a
    // multiple of K_TILE (4), so this should be true.
    debug_assert_eq!(packed_elements_size % align_of::<PackedBMeta<NR>>(), 0);

    // At the end of the buffer are the column sums and zero points.
    let panel_stride = packed_elements_size + size_of::<PackedBMeta<NR>>();
    let size = n_panels * panel_stride;
    let align = align_of::<PackedBMeta<NR>>();

    PackedLayout::new(size, align, panel_stride)
}

/// Pack blocks of the B matrix for use by int8 matmul kernels using dot
/// product instructions (Arm dotprod, AVX-512 VNNI etc.)
///
/// Pack B matrix of shape `[K, N]` into a series of column panels. Each panel
/// contains elements from a `[K, NR]` slice of the input and is laid out as `[K
/// / K_TILE, NR, K_TILE]` u8 values, followed by `NR` i32 column sums.  In the
/// kernel a transposed `[NR, K_TILE]` microtile of `B` is then multiplied with
/// a `[MR, K_TILE]` microtile of `A` using dot product instructions. The column
/// sums are used to handle subtraction of the zero point.
#[allow(unused)]
pub fn pack_b<const NR: usize, const K_TILE: usize>(
    out: &mut [MaybeUninit<i8>],
    b: Matrix<i8>,
    zero_points: Option<&[i8]>,
) {
    pack_b_impl::<NR, K_TILE, _>(
        out,
        b,
        zero_points,
        |x| x,
        |out, meta| {
            out.write_slice(meta.as_signed_bytes());
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
pub fn pack_b_cast_i8_u8<const NR: usize, const K_TILE: usize>(
    out: &mut [MaybeUninit<u8>],
    b: Matrix<i8>,
    zero_points: Option<&[i8]>,
) {
    pack_b_impl::<NR, K_TILE, _>(out, b, zero_points, shift_cast_i8_u8, |out, meta| {
        out.write_slice(meta.as_bytes())
    })
}

/// A type with size and align of 1.
trait Byte: Copy + Default {
    /// True if this is a signed type.
    const SIGNED: bool;

    fn from_u8(x: u8) -> Self;
    fn to_u8(self) -> u8;
}

impl Byte for u8 {
    const SIGNED: bool = false;

    fn from_u8(x: u8) -> Self {
        x
    }

    fn to_u8(self) -> u8 {
        self
    }
}

impl Byte for i8 {
    const SIGNED: bool = true;

    fn from_u8(x: u8) -> Self {
        x as i8
    }

    fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Return the sum of the bytes in `x`, interpreted as values of type `T`.
#[inline(always)]
fn sum_bytes<T: Byte>(x: u64) -> i32 {
    const M8: u64 = 0x00FF_00FF_00FF_00FF;
    const M16: u64 = 0x0000_FFFF_0000_FFFF;
    let s = (x & M8) + ((x >> 8) & M8);
    let s = (s & M16) + ((s >> 16) & M16);
    let sum = ((s & 0xFFFF_FFFF) + (s >> 32)) as i32;
    if T::SIGNED {
        // Subtract 256 for each byte which is negative when signed.
        let n_neg = ((x >> 7) & 0x0101_0101_0101_0101).wrapping_mul(0x0101_0101_0101_0101) >> 56;
        sum - 256 * n_neg as i32
    } else {
        sum
    }
}

/// Swap the bytes selected by `mask << shift` in `rows[i]` with the bytes
/// selected by `mask` in `rows[j]`.
#[inline(always)]
fn swap_bits(rows: &mut [u64; 8], i: usize, j: usize, shift: u32, mask: u64) {
    let t = ((rows[i] >> shift) ^ rows[j]) & mask;
    rows[j] ^= t;
    rows[i] ^= t << shift;
}

/// Transpose a 4x4 matrix of bytes stored in the low 32 bits of `rows[..4]`,
/// in little-endian order.
fn transpose_4x4(rows: &mut [u64; 8]) {
    const M8: u64 = 0x00FF_00FF;
    const M16: u64 = 0x0000_FFFF;
    swap_bits(rows, 0, 1, 8, M8);
    swap_bits(rows, 2, 3, 8, M8);
    swap_bits(rows, 0, 2, 16, M16);
    swap_bits(rows, 1, 3, 16, M16);
}

/// Transpose an 8x8 matrix of bytes stored in `rows`, in little-endian
/// order.
fn transpose_8x8(rows: &mut [u64; 8]) {
    const M8: u64 = 0x00FF_00FF_00FF_00FF;
    const M16: u64 = 0x0000_FFFF_0000_FFFF;
    const M32: u64 = 0x0000_0000_FFFF_FFFF;
    for i in (0..8).step_by(2) {
        swap_bits(rows, i, i + 1, 8, M8);
    }
    for i in [0, 1, 4, 5] {
        swap_bits(rows, i, i + 2, 16, M16);
    }
    for i in 0..4 {
        swap_bits(rows, i, i + 4, 32, M32);
    }
}

/// Load `N` (4 or 8) bytes from `data` into the low bytes of a `u64`, in
/// little-endian order.
#[inline(always)]
fn load_bytes<const N: usize>(data: &[i8]) -> u64 {
    match N {
        4 => u64::from(u32::from_le_bytes(std::array::from_fn(|i| data[i] as u8))),
        _ => u64::from_le_bytes(std::array::from_fn(|i| data[i] as u8)),
    }
}

/// Return a function which transposes a `[K_TILE, K_TILE]` block of bytes,
/// if supported for this tile size.
fn transpose_fn<const K_TILE: usize>() -> Option<fn(&mut [u64; 8])> {
    match K_TILE {
        4 => Some(transpose_4x4),
        8 => Some(transpose_8x8),
        _ => None,
    }
}

/// Pack the first `n_tiles` `[K_TILE, NR]` tiles of the first `n_panels`
/// column panels of a B matrix whose rows are contiguous.
///
/// `data` is the matrix data, with rows starting at multiples of
/// `row_stride`. Elements are converted by XOR-ing with `cast_zero`, the
/// result of casting zero to `T`. Returns the column sums of the packed
/// elements in each panel, or `None` if the tile size is not supported.
fn pack_b_tiles_contiguous<const NR: usize, const K_TILE: usize, T: Byte>(
    out: &mut [MaybeUninit<T>],
    data: &[i8],
    row_stride: usize,
    n_panels: usize,
    n_tiles: usize,
    panel_stride: usize,
    cast_zero: T,
) -> Option<Vec<[i32; NR]>> {
    let transpose = transpose_fn::<K_TILE>()?;
    if !NR.is_multiple_of(K_TILE) {
        return None;
    }

    // `cast` is either the identity or a flip of the sign bit. Apply it only to
    // the `K_TILE` bytes in use, so the unused bytes stay zero.
    let used_bytes = u64::MAX >> (64 - 8 * K_TILE);
    let cast_mask = (u64::from(cast_zero.to_u8()) * 0x0101_0101_0101_0101) & used_bytes;

    let mut col_sums = vec![[0i32; NR]; n_panels];

    // Process groups of panels which together span a few cache lines of each
    // input row, so that each line loaded is fully used.
    const GROUP_COLS: usize = 256;
    let group_panels = (GROUP_COLS / NR).max(1);

    for group_start in (0..n_panels).step_by(group_panels) {
        let group = group_start..(group_start + group_panels).min(n_panels);
        for row_tile in 0..n_tiles {
            let row_offset = row_tile * K_TILE * row_stride;
            for col_panel in group.clone() {
                let tile_offset = col_panel * panel_stride + row_tile * NR * K_TILE;
                let out_tile = &mut out[tile_offset..tile_offset + NR * K_TILE];
                let panel_sums = &mut col_sums[col_panel];

                for col_block in 0..NR / K_TILE {
                    let col_offset = row_offset + col_panel * NR + col_block * K_TILE;

                    #[cfg(target_arch = "aarch64")]
                    if K_TILE == 8 {
                        let cols = col_block * 8..(col_block + 1) * 8;
                        pack_block_8x8_neon(
                            &mut out_tile[cols.start * 8..cols.end * 8],
                            &mut panel_sums[cols],
                            data,
                            col_offset,
                            row_stride,
                            cast_zero.to_u8(),
                        );
                        continue;
                    }

                    // Load a `[K_TILE, K_TILE]` block with one row per `u64`.
                    let mut block = [0u64; 8];
                    for (r, row) in block.iter_mut().take(K_TILE).enumerate() {
                        let start = col_offset + r * row_stride;
                        *row = load_bytes::<K_TILE>(&data[start..start + K_TILE]) ^ cast_mask;
                    }

                    // Transpose so each `u64` contains one column.
                    transpose(&mut block);

                    for (c, &col) in block.iter().take(K_TILE).enumerate() {
                        let col_idx = col_block * K_TILE + c;
                        panel_sums[col_idx] += sum_bytes::<T>(col);
                        let bytes = col.to_le_bytes();
                        for (dst, &src) in out_tile[col_idx * K_TILE..(col_idx + 1) * K_TILE]
                            .iter_mut()
                            .zip(&bytes)
                        {
                            dst.write(T::from_u8(src));
                        }
                    }
                }
            }
        }
    }

    Some(col_sums)
}

/// Pack an 8x8 block of bytes from `data`, with rows starting at `start` and
/// separated by `row_stride`, into `out` in column-major order. The column sums
/// are added to `col_sums`.
///
/// Elements are converted by XOR-ing with `cast_mask`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn pack_block_8x8_neon<T: Byte>(
    out: &mut [MaybeUninit<T>],
    col_sums: &mut [i32],
    data: &[i8],
    start: usize,
    row_stride: usize,
    cast_mask: u8,
) {
    use core::arch::aarch64::*;

    assert_eq!(out.len(), 64);
    assert_eq!(col_sums.len(), 8);

    // Safety: Each load reads 8 bytes from a slice of length 8, and each store
    // writes 8 bytes to `out`, which has length 64. NEON is always available
    // on aarch64.
    unsafe {
        let mask = vdup_n_u8(cast_mask);
        let row = |r: usize| {
            let row = &data[start + r * row_stride..][..8];
            veor_u8(vld1_u8(row.as_ptr() as *const u8), mask)
        };

        // Transpose the block in three stages, swapping 1, 2 and then
        // 4-byte elements between pairs of rows.
        let a = vtrn_u8(row(0), row(1));
        let b = vtrn_u8(row(2), row(3));
        let c = vtrn_u8(row(4), row(5));
        let d = vtrn_u8(row(6), row(7));

        let u16 = vreinterpret_u16_u8;
        let e0 = vtrn_u16(u16(a.0), u16(b.0));
        let e1 = vtrn_u16(u16(a.1), u16(b.1));
        let g0 = vtrn_u16(u16(c.0), u16(d.0));
        let g1 = vtrn_u16(u16(c.1), u16(d.1));

        let u32 = vreinterpret_u32_u16;
        let c04 = vtrn_u32(u32(e0.0), u32(g0.0));
        let c26 = vtrn_u32(u32(e0.1), u32(g0.1));
        let c15 = vtrn_u32(u32(e1.0), u32(g1.0));
        let c37 = vtrn_u32(u32(e1.1), u32(g1.1));

        let cols = [c04.0, c15.0, c26.0, c37.0, c04.1, c15.1, c26.1, c37.1];
        for (i, col) in cols.into_iter().enumerate() {
            let col = vreinterpret_u8_u32(col);
            col_sums[i] += if T::SIGNED {
                i32::from(vaddlv_s8(vreinterpret_s8_u8(col)))
            } else {
                i32::from(vaddlv_u8(col))
            };
            vst1_u8(out.as_mut_ptr().add(i * 8) as *mut u8, col);
        }
    }
}

fn pack_b_impl<const NR: usize, const K_TILE: usize, T: Byte>(
    out: &mut [MaybeUninit<T>],
    b: Matrix<i8>,
    zero_point: Option<&[i8]>,
    cast: impl Fn(i8) -> T,
    write_meta: impl Fn(&mut SliceWriter<T>, PackedBMeta<NR>),
) where
    i32: From<T>,
{
    let [b_rows, b_cols] = b.shape();
    assert_eq!(
        out.len(),
        packed_b_layout::<NR, K_TILE>(b_rows, b_cols).size()
    );

    // Loop over column panels
    let full_panels = b_cols / NR;
    let tail_cols = b_cols % NR;

    let full_tiles = b_rows / K_TILE;
    let tail_rows = b_rows % K_TILE;

    // If rows are contiguous, pack the full tiles of full panels using a fast
    // path. This produces the column sums for these tiles.
    let fast_col_sums = if b.stride(1) == 1 && full_panels > 0 {
        // Safety: `b` is an immutable view, so no mutable references to its
        // elements exist.
        let data = unsafe { b.storage().as_slice() };
        let panel_stride = packed_b_layout::<NR, K_TILE>(b_rows, b_cols).panel_stride();
        pack_b_tiles_contiguous::<NR, K_TILE, T>(
            out,
            data,
            b.stride(0),
            full_panels,
            full_tiles,
            panel_stride,
            cast(0),
        )
    } else {
        None
    };

    let mut out = SliceWriter::new(out);

    for col_panel in 0..full_panels {
        let mut col_sums = [0i32; NR];

        // Write panel elements
        if let Some(fast_col_sums) = &fast_col_sums {
            col_sums = fast_col_sums[col_panel];
            // Safety: The tiles were initialized by `pack_b_tiles_contiguous`.
            unsafe { out.skip_unchecked(full_tiles * NR * K_TILE) };
        } else {
            for row_tile in 0..full_tiles {
                for c in 0..NR {
                    for r in 0..K_TILE {
                        let y = row_tile * K_TILE + r;
                        let x = col_panel * NR + c;
                        let val = cast(unsafe { *b.get_unchecked([y, x]) });
                        col_sums[c] += i32::from(val);
                        unsafe { out.write_unchecked(val) };
                    }
                }
            }
        }

        if tail_rows != 0 {
            for c in 0..NR {
                for r in 0..tail_rows {
                    let y = full_tiles * K_TILE + r;
                    let x = col_panel * NR + c;
                    unsafe {
                        let val = cast(*b.get_unchecked([y, x]));
                        col_sums[c] += i32::from(val);
                        out.write_unchecked(val);
                    }
                }
                // Write padding
                unsafe { out.write_n_unchecked(K_TILE - tail_rows, T::default()) };
            }
        }

        // Write column sums
        let meta = PackedBMeta {
            col_sums,
            zero_points: if let Some(zp) = zero_point {
                std::array::from_fn(|c| i32::from(cast(zp[col_panel * NR + c])))
            } else {
                [i32::from(cast(0)); NR]
            },
        };
        write_meta(&mut out, meta);
    }

    if tail_cols != 0 {
        let mut col_sums = [0i32; NR];
        let col_panel = full_panels;

        // Write panel elements
        for row_tile in 0..full_tiles {
            for c in 0..tail_cols {
                for r in 0..K_TILE {
                    let y = row_tile * K_TILE + r;
                    let x = col_panel * NR + c;
                    let val = cast(unsafe { *b.get_unchecked([y, x]) });
                    col_sums[c] += i32::from(val);
                    unsafe { out.write_unchecked(val) };
                }
            }
            // Write padding
            unsafe { out.write_n_unchecked((NR - tail_cols) * K_TILE, T::default()) };
        }

        if tail_rows != 0 {
            for c in 0..tail_cols {
                for r in 0..tail_rows {
                    let y = full_tiles * K_TILE + r;
                    let x = col_panel * NR + c;
                    unsafe {
                        let val = cast(*b.get_unchecked([y, x]));
                        col_sums[c] += i32::from(val);
                        out.write_unchecked(val);
                    }
                }
                // Write padding
                unsafe { out.write_n_unchecked(K_TILE - tail_rows, T::default()) };
            }
            // Write padding
            unsafe { out.write_n_unchecked((NR - tail_cols) * K_TILE, T::default()) };
        }

        // Write column sums
        let mut zero_point_array = [0i32; NR];
        if let Some(zp) = zero_point {
            let col_range = col_panel * NR..b_cols;
            for (i, c) in col_range.enumerate() {
                zero_point_array[i] = i32::from(cast(zp[c]));
            }
        } else {
            for i in 0..tail_cols {
                zero_point_array[i] = i32::from(cast(0));
            }
        }
        let meta = PackedBMeta {
            col_sums,
            zero_points: zero_point_array,
        };
        write_meta(&mut out, meta);
    }

    assert!(out.completed());
}

/// Return the layout of the packing buffer required by `pack_a`.
pub fn packed_a_layout<const MR: usize, const K_TILE: usize>(
    a_rows: usize,
    a_cols: usize,
) -> PackedLayout {
    // Packed block is padded to a multiple of MR rows and K_TILE columns.
    let n_panels = a_rows.div_ceil(MR);
    let packed_elements_size = a_cols.div_ceil(K_TILE) * MR * K_TILE;

    // PackedAMeta should have an align of 4, and packed_elements_size is a
    // multiple of K_TILE (4), so this should be true.
    debug_assert_eq!(packed_elements_size % align_of::<PackedAMeta<MR>>(), 0);

    // At the end of the buffer are the row sums and zero points.
    let panel_stride = packed_elements_size + size_of::<PackedAMeta<MR>>();
    let size = n_panels * panel_stride;
    let align = align_of::<PackedAMeta<MR>>();

    PackedLayout::new(size, align, panel_stride)
}

/// A 2D tensor layout where the last dimension has unit stride.
///
/// This is implemented in a way that enables the compiler to understand that
/// successive entries in a row are contiguous, allowing for better vectorization.
/// Otherwise it behaves the same as an [`NdLayout<2>`].
///
/// This layout only supports rank-2 tensors because a) that's all we need and
/// b) it was harder to make this layout generic like `NdLayout` and still have
/// the compiler generate vectorized code.
#[derive(Clone)]
struct RowMajorLayout {
    shape: [usize; 2],
    row_stride: usize,
}

impl RowMajorLayout {
    /// Return a `RowMajorLayout` with the same shape and strides as `layout`
    /// if it has a column stride of 1, or None otherwise.
    fn from_layout(layout: NdLayout<2>) -> Option<Self> {
        if layout.stride(1) == 1 {
            Some(Self {
                shape: layout.shape(),
                row_stride: layout.stride(0),
            })
        } else {
            None
        }
    }

    fn index_valid(&self, index: [usize; 2]) -> bool {
        index[0] < self.shape[0] && index[1] < self.shape[1]
    }
}

impl Layout for RowMajorLayout {
    type Shape<'a> = [usize; 2];
    type Strides<'a> = [usize; 2];
    type Index<'a> = [usize; 2];
    type Indices = NdIndices<2>;

    fn ndim(&self) -> usize {
        2
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    fn offset(&self, index: [usize; 2]) -> Option<usize> {
        self.index_valid(index)
            .then_some(self.offset_unchecked(index))
    }

    #[inline]
    fn offset_unchecked(&self, index: [usize; 2]) -> usize {
        index[0] * self.row_stride + index[1]
    }

    #[inline]
    fn shape(&self) -> Self::Shape<'_> {
        self.shape
    }

    #[inline]
    fn strides(&self) -> Self::Index<'_> {
        [self.row_stride, 1]
    }

    fn indices(&self) -> Self::Indices {
        NdIndices::from_shape(self.shape)
    }
}

impl AsIndex<RowMajorLayout> for [usize; 2] {
    fn as_index(&self) -> [usize; 2] {
        *self
    }
}

// Pack blocks of the A matrix for use by the matmul kernel.
//
// Pack A matrix of shape `[M, K]` into a series of row panels. Each panel
// contains elements from an `[MR, K]` slice of the input and is laid out as `[K
// / 4, MR, 4]` u8 values, followed by `MR` i32 row sums. The row sums are
// used to handle subtraction of the zero point.
pub fn pack_a<const MR: usize, const K_TILE: usize>(
    out: &mut [MaybeUninit<u8>],
    a: Matrix<u8>,
    zero_point: Option<&[u8]>,
) {
    // Specialize for the common case where `a` has unit column stride, as this
    // enables vectorization of inner loops.
    if let Some(layout) = RowMajorLayout::from_layout(*a.layout()) {
        pack_a_impl::<MR, K_TILE, _>(
            out,
            TensorBase::from_storage_and_layout(a.storage(), layout),
            zero_point,
        )
    } else {
        pack_a_impl::<MR, K_TILE, _>(out, a, zero_point)
    }
}

// Disable inlining here because it interferes with vectorization when `L = RowMajorLayout`.
#[inline(never)]
fn pack_a_impl<const MR: usize, const K_TILE: usize, L>(
    out: &mut [MaybeUninit<u8>],
    a: TensorBase<ViewData<u8>, L>,
    zero_point: Option<&[u8]>,
) where
    L: Clone + for<'a> Layout<Shape<'a> = [usize; 2]> + 'static,
    [usize; 2]: AsIndex<L>,
{
    let [a_rows, a_cols] = a.shape();
    assert_eq!(
        out.len(),
        packed_a_layout::<MR, K_TILE>(a_rows, a_cols).size()
    );

    let mut out = SliceWriter::new(out);

    // Loop over row panels.
    let full_panels = a_rows / MR;
    let tail_rows = a_rows % MR;

    let full_tiles = a_cols / K_TILE;
    let tail_cols = a_cols % K_TILE;

    for row_tile in 0..full_panels {
        let mut row_sums = [0i32; MR];

        // Write packed elements
        for col_tile in 0..full_tiles {
            for r in 0..MR {
                for c in 0..K_TILE {
                    let y = row_tile * MR + r;
                    let x = col_tile * K_TILE + c;
                    let val = unsafe { *a.get_unchecked([y, x]) };
                    row_sums[r] += val as i32;
                    unsafe { out.write_unchecked(val) };
                }
            }
        }

        if tail_cols != 0 {
            for r in 0..MR {
                for c in 0..tail_cols {
                    let y = row_tile * MR + r;
                    let x = full_tiles * K_TILE + c;
                    let val = unsafe { *a.get_unchecked([y, x]) };
                    row_sums[r] += val as i32;
                    unsafe { out.write_unchecked(val) };
                }
                // Write padding
                unsafe { out.write_n_unchecked(K_TILE - tail_cols, 0) };
            }
        }

        // Write row sums
        let meta = PackedAMeta {
            row_sums,
            zero_points: if let Some(zp) = zero_point {
                std::array::from_fn(|r| zp[row_tile * MR + r] as i32)
            } else {
                [0; MR]
            },
        };
        out.write_slice(meta.as_bytes());
    }

    if tail_rows != 0 {
        let row_tile = full_panels;
        let mut row_sums = [0i32; MR];
        let row_range = row_tile * MR..(row_tile * MR + MR).min(a_rows);

        // Write packed elements
        for col_tile in 0..full_tiles {
            for r in 0..tail_rows {
                for c in 0..K_TILE {
                    let y = row_tile * MR + r;
                    let x = col_tile * K_TILE + c;
                    let val = unsafe { *a.get_unchecked([y, x]) };
                    row_sums[r] += val as i32;
                    unsafe { out.write_unchecked(val) };
                }
            }
            // Write padding
            unsafe { out.write_n_unchecked((MR - tail_rows) * K_TILE, 0) };
        }
        if tail_cols != 0 {
            for r in 0..tail_rows {
                for c in 0..tail_cols {
                    let y = row_tile * MR + r;
                    let x = full_tiles * K_TILE + c;
                    let val = unsafe { *a.get_unchecked([y, x]) };
                    row_sums[r] += val as i32;
                    unsafe { out.write_unchecked(val) };
                }
                // Write padding
                unsafe { out.write_n_unchecked(K_TILE - tail_cols, 0) };
            }
            // Write padding
            unsafe { out.write_n_unchecked((MR - tail_rows) * K_TILE, 0) };
        }

        // Write row sums
        let mut zero_point_array = [0i32; MR];
        if let Some(zp) = zero_point {
            for (i, r) in row_range.enumerate() {
                zero_point_array[i] = zp[r] as i32;
            }
        }
        let meta = PackedAMeta {
            row_sums,
            zero_points: zero_point_array,
        };
        out.write_slice(meta.as_bytes());
    }

    assert!(out.completed());
}

/// Extract the packed elements and row sums from a buffer packed by [`pack_a`].
pub fn extract_packed_a<const MR: usize>(a: &[u8]) -> (&[u8], &PackedAMeta<MR>) {
    assert!(a.len() >= size_of::<PackedAMeta<MR>>());
    let meta_offset = a.len() - size_of::<PackedAMeta<MR>>();
    let (packed_elements, meta_bytes) = a.split_at(meta_offset);
    (packed_elements, PackedAMeta::from_bytes(meta_bytes))
}

/// Extract the packed elements and column sums from a buffer packed by [`pack_b`].
pub fn extract_packed_b<const NR: usize>(b: &[u8]) -> (&[u8], &PackedBMeta<NR>) {
    assert!(b.len() >= size_of::<PackedBMeta<NR>>());
    let meta_offset = b.len() - size_of::<PackedBMeta<NR>>();
    let (packed_elements, meta_bytes) = b.split_at(meta_offset);
    (packed_elements, PackedBMeta::from_bytes(meta_bytes))
}

#[cfg(test)]
mod tests {
    use rten_base::byte_cast::{AsBytes, cast_slice};
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::{Matrix, MatrixLayout, NdTensor};

    use super::{
        PackedAMeta, PackedBMeta, extract_packed_a, extract_packed_b, pack_a, pack_b,
        pack_b_cast_i8_u8, packed_a_layout, packed_b_layout,
    };

    // K tile size used by kernels that compute dot product of 4x i8 -> i32.
    const K_TILE_I8DOT: usize = 4;

    // K tile use used by Arm i8mm.
    const K_TILE_I8MM: usize = 8;

    fn pack_a_matrix<const MR: usize, const K_TILE: usize>(
        mat: Matrix<u8>,
        zero_point: Option<&[u8]>,
    ) -> Vec<u8> {
        let layout = packed_a_layout::<MR, K_TILE>(mat.rows(), mat.cols());

        // Layout must have space for at least each element in the input, plus
        // row sums and zero points as i32 values.
        assert!(layout.size() >= mat.rows() * mat.cols() + 2 * mat.rows() * 4);

        let mut buf = Vec::with_capacity(layout.size());
        pack_a::<MR, K_TILE>(
            &mut buf.spare_capacity_mut()[..layout.size()],
            mat.view(),
            zero_point,
        );

        // Safety: `pack_a` initialized `layout.size()` elements.
        unsafe { buf.set_len(layout.size()) }

        buf
    }

    // Un-optimized reference implementation of `pack_a`.
    fn reference_pack_a<const MR: usize, const K_TILE: usize>(
        mat: Matrix<u8>,
        zero_point: Option<&[u8]>,
    ) -> Vec<u8> {
        let layout = packed_a_layout::<MR, K_TILE>(mat.rows(), mat.cols());
        let mut buf = Vec::with_capacity(layout.size());

        for row_panel in 0..mat.rows().div_ceil(MR) {
            let mut row_sums = [0i32; MR];
            for k_tile in 0..mat.cols().div_ceil(K_TILE) {
                for r in 0..MR {
                    for c in 0..K_TILE {
                        let y = row_panel * MR + r;
                        let x = k_tile * K_TILE + c;
                        let val = mat.get([y, x]).copied().unwrap_or(0);
                        row_sums[r] += val as i32;
                        buf.push(val);
                    }
                }
            }
            let meta = PackedAMeta {
                row_sums,
                zero_points: std::array::from_fn(|r| {
                    zero_point
                        .and_then(|zp| zp.get(row_panel * MR + r))
                        .copied()
                        .unwrap_or(0) as i32
                }),
            };
            buf.extend_from_slice(meta.as_bytes());
        }

        assert_eq!(buf.len(), layout.size());
        buf
    }

    fn pack_b_matrix<const NR: usize, const K_TILE: usize>(
        mat: Matrix<i8>,
        zero_point: Option<&[i8]>,
    ) -> Vec<i8> {
        let layout = packed_b_layout::<NR, K_TILE>(mat.rows(), mat.cols());

        // Layout must have space for at least each element in the input, plus
        // column sums as i32 values.
        assert!(layout.size() >= mat.rows() * mat.cols() + mat.cols() * 4);

        let mut buf = Vec::with_capacity(layout.size());
        pack_b::<NR, K_TILE>(
            &mut buf.spare_capacity_mut()[..layout.size()],
            mat.view(),
            zero_point,
        );

        // Safety: `pack_b` initialized `layout.size()` elements.
        unsafe { buf.set_len(layout.size()) }

        buf
    }

    // Un-optimized reference implementation of `pack_b`.
    fn reference_pack_b<const NR: usize, const K_TILE: usize>(
        mat: Matrix<i8>,
        zero_point: Option<&[i8]>,
    ) -> Vec<i8> {
        let layout = packed_b_layout::<NR, K_TILE>(mat.rows(), mat.cols());
        let mut buf = Vec::with_capacity(layout.size());

        for col_panel in 0..mat.cols().div_ceil(NR) {
            let mut col_sums = [0i32; NR];
            for k_tile in 0..mat.rows().div_ceil(K_TILE) {
                for c in 0..NR {
                    for r in 0..K_TILE {
                        let y = k_tile * K_TILE + r;
                        let x = col_panel * NR + c;
                        let val = mat.get([y, x]).copied().unwrap_or(0);
                        col_sums[c] += val as i32;
                        buf.push(val);
                    }
                }
            }
            let meta = PackedBMeta {
                col_sums,
                zero_points: std::array::from_fn(|c| {
                    zero_point
                        .and_then(|zp| zp.get(col_panel * NR + c))
                        .copied()
                        .unwrap_or(0) as i32
                }),
            };
            buf.extend_from_slice(cast_slice(meta.as_bytes()).unwrap());
        }

        assert_eq!(buf.len(), layout.size());
        buf
    }

    fn pack_b_matrix_cast_u8<const NR: usize, const K_TILE: usize>(mat: Matrix<i8>) -> Vec<u8> {
        let layout = packed_b_layout::<NR, K_TILE>(mat.rows(), mat.cols());

        // Layout must have space for at least each element in the input, plus
        // column sums as i32 values.
        assert!(layout.size() >= mat.rows() * mat.cols() + mat.cols() * 4);

        let mut buf = Vec::with_capacity(layout.size());
        pack_b_cast_i8_u8::<NR, K_TILE>(
            &mut buf.spare_capacity_mut()[..layout.size()],
            mat.view(),
            None,
        );

        // Safety: `pack_b` initialized `layout.size()` elements.
        unsafe { buf.set_len(layout.size()) }

        buf
    }

    #[test]
    fn test_pack_a_various_sizes() {
        const MR: usize = 8;

        fn test_pack_a<const K_TILE: usize>() {
            let mut rng = XorShiftRng::new(5678);
            for m in 1..MR * 3 {
                // Use distinct per-row zero points so that using the wrong
                // row's zero point is detected.
                let zero_points: Vec<u8> = (1..=m as u8).collect();

                for k in 1..K_TILE_I8DOT * 2 {
                    for zp in [None, Some(zero_points.as_slice())] {
                        // Test row-major and non-row major inputs, as the implementation
                        // has a fast path for row-major layouts.

                        // Row major layout
                        let mat = NdTensor::rand([m, k], &mut rng);
                        let expected = reference_pack_a::<MR, K_TILE_I8DOT>(mat.view(), zp);
                        let actual = pack_a_matrix::<MR, K_TILE_I8DOT>(mat.view(), zp);
                        assert_eq!(
                            actual, expected,
                            "packed buffer mismatch for row-major m={} k={} zp={:?}",
                            m, k, zp
                        );

                        // Column major layout
                        let mat = NdTensor::rand([k, m], &mut rng);
                        let expected =
                            reference_pack_a::<MR, K_TILE_I8DOT>(mat.transposed().view(), zp);
                        let actual = pack_a_matrix::<MR, K_TILE_I8DOT>(mat.transposed().view(), zp);
                        assert_eq!(
                            actual, expected,
                            "packed buffer mismatch for col-major m={} k={} zp={:?}",
                            m, k, zp
                        );
                    }
                }
            }
        }

        test_pack_a::<K_TILE_I8DOT>();
        test_pack_a::<K_TILE_I8MM>();
    }

    #[test]
    fn test_extract_packed_a() {
        fn test_extract_packed_a<const K_TILE: usize>() {
            const MR: usize = 8;

            let mat = NdTensor::<u8, 2>::from([[1, 2], [3, 4]]);
            let packed = pack_a_matrix::<MR, K_TILE_I8DOT>(mat.view(), None);

            let (packed_elems, meta) = extract_packed_a(&packed);

            assert!(packed_elems.len() >= mat.rows() * mat.cols());
            assert_eq!(meta.row_sums, [3, 7, 0, 0, 0, 0, 0, 0]);
        }

        test_extract_packed_a::<K_TILE_I8DOT>();
        test_extract_packed_a::<K_TILE_I8MM>();
    }

    #[test]
    fn test_pack_b_various_sizes() {
        const NR: usize = 8;

        fn test_pack_b<const K_TILE: usize>() {
            let mut rng = XorShiftRng::new(5678);
            for n in 1..NR * 3 {
                // Use distinct per-column zero points so that using the wrong
                // column's zero point is detected.
                let zero_points: Vec<i8> = (1..=n as i8).collect();

                for k in 1..K_TILE * 2 {
                    for zp in [None, Some(zero_points.as_slice())] {
                        let mat = NdTensor::rand([k, n], &mut rng);
                        let expected = reference_pack_b::<NR, K_TILE>(mat.view(), zp);
                        let actual = pack_b_matrix::<NR, K_TILE>(mat.view(), zp);

                        assert_eq!(
                            actual, expected,
                            "packed buffer mismatch for n={} k={} zp={:?}",
                            n, k, zp
                        );
                    }
                }
            }
        }
        test_pack_b::<K_TILE_I8DOT>();
        test_pack_b::<K_TILE_I8MM>();
    }

    #[test]
    fn test_pack_b_contiguous_rows() {
        // Packing uses a separate path for inputs with contiguous rows. Check
        // that it matches the path for other inputs.
        fn test_pack_b<const NR: usize, const K_TILE: usize>() {
            let mut rng = XorShiftRng::new(1234);
            for n in 1..NR * 3 {
                for k in 1..K_TILE * 3 {
                    let mat = NdTensor::<i8, 2>::rand([k, n], &mut rng);
                    let col_major = mat.transposed().to_tensor();
                    let strided = col_major.transposed();
                    let zero_points: Vec<i8> = (1..=n as i8).collect();

                    assert_eq!(
                        pack_b_matrix::<NR, K_TILE>(mat.view(), Some(&zero_points)),
                        pack_b_matrix::<NR, K_TILE>(strided, Some(&zero_points)),
                        "pack_b mismatch for n={} k={}",
                        n,
                        k
                    );
                    assert_eq!(
                        pack_b_matrix_cast_u8::<NR, K_TILE>(mat.view()),
                        pack_b_matrix_cast_u8::<NR, K_TILE>(strided),
                        "pack_b_cast_i8_u8 mismatch for n={} k={}",
                        n,
                        k
                    );
                }
            }
        }
        test_pack_b::<8, K_TILE_I8DOT>();
        test_pack_b::<8, K_TILE_I8MM>();
        test_pack_b::<16, K_TILE_I8DOT>();
    }

    #[test]
    fn test_extract_packed_b() {
        const NR: usize = 8;

        let mat = NdTensor::<i8, 2>::from([[1, 2], [3, 4]]);
        let packed = pack_b_matrix::<NR, K_TILE_I8DOT>(mat.view(), None);

        let (packed_elems, meta) = extract_packed_b(cast_slice(&packed).unwrap());

        assert!(packed_elems.len() >= mat.rows() * mat.cols());
        assert_eq!(meta.col_sums, [4, 6, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_pack_b_cast_i8_u8() {
        let mat = NdTensor::<i8, 2>::from([[1, 2], [3, 4]]);
        let packed = pack_b_matrix_cast_u8::<2, K_TILE_I8DOT>(mat.view());

        let (packed_elems, meta) = extract_packed_b(&packed);

        assert!(packed_elems.len() >= mat.rows() * mat.cols());
        assert_eq!(packed_elems, &[129, 131, 0, 0, 130, 132, 0, 0]);
        assert_eq!(meta.col_sums, [129 + 131, 130 + 132]);
    }
}
