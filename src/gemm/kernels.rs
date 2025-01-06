use std::mem::MaybeUninit;
use std::ops::Range;

use super::{GemmOutT, Im2Col};
use rten_tensor::Matrix;

pub mod generic;
mod simd_generic;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub mod wasm;

/// LHS / A input to a GEMM kernel.
#[derive(Clone, Copy)]
pub enum Lhs<'a, T> {
    /// Input packed by the kernel's [`pack_a_block`](Kernel::pack_a_block)
    /// impl.
    Packed(&'a [u8]),

    /// Unpacked input with a column stride of 1 and row stride of `row_stride`.
    ///
    /// `data` is a pointer to the first element needed to compute the output
    /// tile. `(data, len)` is not passed as a slice because it is a pointer to
    /// part of a larger tensor and there may be mutable references in existence
    /// to other parts of that tensor.
    Unpacked {
        data: *const T,
        len: usize,
        row_stride: usize,
    },
}

/// Metadata about a packed block of an input matrix.
///
/// The packed block is expected to be organized as a sequence of panels with
/// stride [`panel_stride`](PackedInfo::panel_stride), but the kernel is
/// otherwise free to choose the layout.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedLayout {
    size: usize,
    align: usize,
    panel_stride: usize,

    /// True if the input must be packed to be used by the kernel.
    pub must_pack: bool,
}

impl PackedLayout {
    /// Construct a new packing buffer descriptor.
    ///
    /// `size`, `align` and `panel_stride` specify the minimum size of the
    /// packing buffer, its alignment and the stride between panels
    /// respectively. All units are in bytes. The size must be a multiple of
    /// both the alignment and panel stride.
    pub fn new(size: usize, align: usize, panel_stride: usize) -> PackedLayout {
        debug_assert_eq!(size % align, 0);
        debug_assert_eq!(size % panel_stride, 0);

        PackedLayout {
            size,
            align,
            panel_stride,
            must_pack: false,
        }
    }

    /// Return size of the packed block in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return minimum alignment of the packed block.
    pub fn align(&self) -> usize {
        self.align
    }

    /// Return stride between panels in bytes.
    pub fn panel_stride(&self) -> usize {
        self.panel_stride
    }
}

/// Kernel that computes a small tile of a general matrix multiplication (GEMM)
/// or general matrix-vector multiplication (GEMV).
///
/// This trait is an interface for the architecture-specific inner loop for
/// matrix multiplication and matrix-vector multiplication, as well as the
/// methods that pack the input matrices into a format that is efficient for the
/// kernel to use.
///
/// # Tile size selection
///
/// For a typical f32 kernel using FMA instructions (eg. AVX2), the tile size is
/// chosen such that an `MR x NR` tile of the output fits in registers. Each
/// iteration over the K dimension accumulates into this tile. Additionally one
/// of the dimensions (usually NR) is a multiple of the vector size and the
/// other is large enough such that enough cycles elapse between one update to
/// an accumulator register and the next that it doesn't encounter a delay
/// waiting for the previous update to complete. There is a small overhead for
/// each call into the kernel, so making tiles larger can improve performance by
/// reducing the overall number of tiles that need to be processed. See [^1]
/// for more details.
///
/// # Safety
///
/// - It must only be possible to construct the kernel using `new` if the
///   instructions it uses are supported on the current system.
/// - Kernels must initialize all output elements when `beta` is zero.
///
/// [^1]: https://dl.acm.org/doi/pdf/10.1145/2925987
pub unsafe trait Kernel<LhsT, RhsT, OutT>: Sync {
    /// Construct a new instance of this kernel, if supported on the current
    /// system.
    fn new() -> Option<Self>
    where
        Self: Sized;

    /// Return the number of rows in each tile.
    fn mr(&self) -> usize;

    /// Return the number of columns in each tile.
    fn nr(&self) -> usize;

    /// Return a name for this kernel for use in logging etc.
    fn name(&self) -> &'static str;

    /// Return the layout of a packing buffer required to pack an A / LHS input.
    fn packed_a_layout(&self, a: Matrix<LhsT>, rows: usize, cols: usize) -> PackedLayout;

    /// Pack a block of the LHS / "A" input for use by this kernel.
    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        a: Matrix<LhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
    );

    /// Return the layout of a packing buffer required to pack a block of a "B"
    /// / RHS input of size `rows x cols`.
    ///
    /// Unlike `packed_a_layout` this doesn't take the matrix as an argument.
    /// `packed_a_layout` may use this to indicate that the A input does not
    /// need to be packed. For the B input it is assumed this is always packed.
    fn packed_b_layout(&self, rows: usize, cols: usize) -> PackedLayout;

    /// Pack a block of the RHS / "B" input for use by this kernel.
    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        b: Matrix<RhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
    );

    /// Pack a block of an image as the B input for use by this kernel, using
    /// an im2col transformation to flatten the image into a matrix.
    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<RhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
    );

    /// Compute a tile of the output matrix.
    ///
    /// The output is stored in row-major order with `used_rows` rows and `NR`
    /// columns, a row stride of `tile_row_stride` and column stride of 1. The
    /// maximum size of the tile will be `MR` rows and `NR` columns.
    ///
    /// The `a` input is either an unpacked matrix with a column stride of 1 or
    /// a buffer packed with `pack_a_block`. The `b` input is the RHS matrix
    /// packed with `pack_b_block`.
    ///
    /// `depth` specifies the number of columns of A and rows B that should be
    /// summed over to compute the output tile.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `tile_ptr` points to a buffer of the correct
    /// size. If `beta` is zero then the output may be uninitialized and must
    /// not be read by the implementation. If `beta` is non-zero then the output
    /// must be initialized and the implementation will read from it.
    unsafe fn kernel(
        &self,
        tile_ptr: *mut OutT,
        tile_row_stride: usize,
        a: Lhs<LhsT>,
        b: &[u8],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        alpha: f32,
        beta: OutT,
    );

    /// Compute an output block of a vector-matrix product ("gemv").
    ///
    /// This computes `y = alpha * (a B) + beta * y` where `a` is a row vector
    /// and `B` is a matrix.
    ///
    /// This is a simplified version of the matrix multiplication kernel that
    /// operates on unpacked data, since the overhead of packing outweighs the
    /// benefits for this operation.
    ///
    /// The length of vector `a` must match `b.rows()` and the length of `out`
    /// must match `b.cols()`. The `b` matrix must have a column stride of 1.
    ///
    /// # Safety
    ///
    /// If `beta` is zero then the output may be uninitialized and must not be
    /// read by the implementation. After the kernel has run, all elements will
    /// be initialized.
    fn gemv_kernel(
        &self,
        out: &mut [MaybeUninit<OutT>],
        a: &[LhsT],
        b: Matrix<RhsT>,
        alpha: f32,
        beta: OutT,
    );
}

/// A temporary stack-allocated tile used by some kernels to handle edge tiles
/// which are smaller than the kernel's (MR, NR) size.
///
/// The kernel first allocates a [`TempTile`] and accumulates into it, then
/// calls [`accumulate_into`] to add the results to the matrix multiplication
/// output, for rows and columns that are actually valid.
///
/// The temporary tile is uninitialized, so callers must always set `beta=0`
/// when running the kernel on it. The row stride of the temporary tile is
/// always `NR`.
pub struct TempTile<T: GemmOutT, const MR: usize, const NR: usize> {
    data: [[MaybeUninit<T>; NR]; MR],
}

impl<T: GemmOutT, const MR: usize, const NR: usize> TempTile<T, MR, NR> {
    // Create an uninitialized temporary tile.
    //
    // Since the data is uninitialized, the only cost is reserving stack space.
    pub fn new() -> Self {
        TempTile {
            data: [[MaybeUninit::<T>::uninit(); NR]; MR],
        }
    }

    /// Return a pointer to the temporary tile data.
    pub fn as_mut_ptr(&mut self) -> *mut MaybeUninit<T> {
        self.data.as_mut_ptr() as *mut MaybeUninit<T>
    }

    /// Update the first `n_rows` rows and `n_cols` columns of the tile pointed
    /// to by `dest` with the values from this temporary tile.
    ///
    /// If `beta` is zero, `dest` is treated as uninitialized. Otherwise
    /// it must be initialized.
    pub unsafe fn accumulate_into(
        &self,
        dest: *mut MaybeUninit<T>,
        n_rows: usize,
        n_cols: usize,
        row_stride: usize,
        beta: T,
    ) {
        if beta != T::zero() {
            // Accumulate into initialized output.
            for i in 0..n_rows {
                for j in 0..n_cols {
                    // Safety: Row and column indices are < `n_rows`/`n_cols`
                    unsafe {
                        let out_el = dest.add(row_stride * i + j);
                        let tmp = (*out_el).assume_init();
                        out_el.write(MaybeUninit::new(
                            beta * tmp + self.data.get_unchecked(i).get_unchecked(j).assume_init(),
                        ));
                    }
                }
            }
        } else {
            // Copy into possible uninitialized output.
            for i in 0..n_rows {
                for j in 0..n_cols {
                    // Safety: Row and column indices are < `n_rows`/`n_cols`
                    unsafe {
                        let out_el = dest.add(row_stride * i + j);
                        out_el.write(MaybeUninit::new(
                            self.data.get_unchecked(i).get_unchecked(j).assume_init(),
                        ));
                    }
                }
            }
        }
    }
}
