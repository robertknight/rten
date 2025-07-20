//! Packing for int8 x int8 -> int32 matrix multiplications.
//!
//! These require the inputs to be packed into microtiles of size `[MR, 4]` for
//! the LHS or `[4, NR]` for the RHS. These can then be used with SIMD
//! instructions that compute `VEC_LEN x (4 x i8) -> VEC_LEN x i32` dot
//! products.

use std::mem::MaybeUninit;

use rten_base::byte_cast::{AsBytes, FromBytes};
use rten_tensor::prelude::*;
use rten_tensor::{AsIndex, Layout, Matrix, NdIndices, NdLayout, TensorBase, ViewData};

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

/// Size of micro tiles of K in the innermost dimension of packed layouts.
const K_TILE: usize = 4;

/// Return the layout of the packing buffer required by `pack_b`.
pub fn packed_b_layout<const NR: usize>(b_rows: usize, b_cols: usize) -> PackedLayout {
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

/// Pack blocks of the B matrix for use by the matmul kernel.
///
/// Pack B matrix of shape `[K, N]` into a series of column panels. Each panel
/// contains elements from a `[K, NR]` slice of the input and is laid out as `[K
/// / 4, NR, 4]` u8 values, followed by `NR` i32 column sums.  In the kernel a
/// transposed `[NR, 4]` microtile of `B` is then multiplied with a `[MR, 4]`
/// microtile of `A` using dot product instructions. The column sums are used
/// to handle subtraction of the zero point.
#[allow(unused)]
pub fn pack_b<const NR: usize>(
    out: &mut [MaybeUninit<i8>],
    b: Matrix<i8>,
    zero_points: Option<&[i8]>,
) {
    pack_b_impl::<NR, _>(
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
pub fn pack_b_cast_i8_u8<const NR: usize>(
    out: &mut [MaybeUninit<u8>],
    b: Matrix<i8>,
    zero_points: Option<&[i8]>,
) {
    pack_b_impl::<NR, _>(out, b, zero_points, shift_cast_i8_u8, |out, meta| {
        out.write_slice(meta.as_bytes())
    })
}

/// A type with size and align of 1.
trait Byte: Copy + Default {}
impl Byte for u8 {}
impl Byte for i8 {}

fn pack_b_impl<const NR: usize, T: Byte>(
    out: &mut [MaybeUninit<T>],
    b: Matrix<i8>,
    zero_point: Option<&[i8]>,
    cast: impl Fn(i8) -> T,
    write_meta: impl Fn(&mut SliceWriter<T>, PackedBMeta<NR>),
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
        let mut zero_point_array = [0i32; NR];
        if let Some(zp) = zero_point {
            for (i, c) in col_range.enumerate() {
                zero_point_array[i] = i32::from(cast(zp[c]));
            }
        } else {
            for i in 0..col_range.len() {
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
pub fn packed_a_layout<const MR: usize>(a_rows: usize, a_cols: usize) -> PackedLayout {
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
    type Index<'a> = [usize; 2];
    type Indices = NdIndices<2>;

    fn ndim(&self) -> usize {
        2
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    fn try_offset(&self, index: [usize; 2]) -> Option<usize> {
        self.index_valid(index)
            .then_some(self.offset_unchecked(index))
    }

    #[inline]
    fn offset_unchecked(&self, index: [usize; 2]) -> usize {
        index[0] * self.row_stride + index[1]
    }

    #[inline]
    fn shape(&self) -> Self::Index<'_> {
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
pub fn pack_a<const MR: usize>(
    out: &mut [MaybeUninit<u8>],
    a: Matrix<u8>,
    zero_point: Option<&[u8]>,
) {
    // Specialize for the common case where `a` has unit column stride, as this
    // enables vectorization of inner loops.
    if let Some(layout) = RowMajorLayout::from_layout(*a.layout()) {
        pack_a_impl::<MR, _>(
            out,
            TensorBase::from_storage_and_layout(a.storage(), layout),
            zero_point,
        )
    } else {
        pack_a_impl::<MR, _>(out, a, zero_point)
    }
}

// Disable inlining here because it interferes with vectorization when `L = RowMajorLayout`.
#[inline(never)]
fn pack_a_impl<const MR: usize, L>(
    out: &mut [MaybeUninit<u8>],
    a: TensorBase<ViewData<u8>, L>,
    zero_point: Option<&[u8]>,
) where
    L: Clone + for<'a> Layout<Index<'a> = [usize; 2]>,
    [usize; 2]: AsIndex<L>,
{
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
    use rten_base::byte_cast::cast_pod_slice;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::{Matrix, MatrixLayout, NdTensor};

    use super::{
        extract_packed_a, extract_packed_b, pack_a, pack_b, pack_b_cast_i8_u8, packed_a_layout,
        packed_b_layout, K_TILE,
    };

    const MR: usize = 8;
    const NR: usize = 8;

    fn pack_a_matrix(mat: Matrix<u8>) -> Vec<u8> {
        let layout = packed_a_layout::<MR>(mat.rows(), mat.cols());

        // Layout must have space for at least each element in the input, plus
        // row sums and zero points as i32 values.
        assert!(layout.size() >= mat.rows() * mat.cols() + 2 * mat.rows() * 4);

        let mut buf = Vec::with_capacity(layout.size());
        pack_a::<MR>(
            &mut buf.spare_capacity_mut()[..layout.size()],
            mat.view(),
            None,
        );

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
        pack_b::<NR>(
            &mut buf.spare_capacity_mut()[..layout.size()],
            mat.view(),
            None,
        );

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
        pack_b_cast_i8_u8::<NR>(
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
        let mut rng = XorShiftRng::new(5678);
        for m in 1..MR * 2 {
            for k in 1..K_TILE * 2 {
                // Test row-major and non-row major inputs, as the implementation
                // has a fast path for row-major layouts.

                // Row major layout
                let mat = NdTensor::rand([m, k], &mut rng);
                pack_a_matrix(mat.view());

                // Column major layout
                let mat = NdTensor::rand([k, m], &mut rng);
                pack_a_matrix(mat.transposed().view());
            }
        }
    }

    #[test]
    fn test_extract_packed_a() {
        let mat = NdTensor::<u8, 2>::from([[1, 2], [3, 4]]);
        let packed = pack_a_matrix(mat.view());

        let (packed_elems, meta) = extract_packed_a(&packed);

        assert!(packed_elems.len() >= mat.rows() * mat.cols());
        assert_eq!(meta.row_sums, [3, 7, 0, 0, 0, 0, 0, 0]);
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

        let (packed_elems, meta) = extract_packed_b(cast_pod_slice(&packed).unwrap());

        assert!(packed_elems.len() >= mat.rows() * mat.cols());
        assert_eq!(meta.col_sums, [4, 6, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_pack_b_cast_i8_u8() {
        let mat = NdTensor::<i8, 2>::from([[1, 2], [3, 4]]);
        let packed = pack_b_matrix_cast_u8::<2>(mat.view());

        let (packed_elems, meta) = extract_packed_b(&packed);

        assert!(packed_elems.len() >= mat.rows() * mat.cols());
        assert_eq!(packed_elems, &[129, 131, 0, 0, 130, 132, 0, 0]);
        assert_eq!(meta.col_sums, [129 + 131, 130 + 132]);
    }
}
