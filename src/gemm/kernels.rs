use std::mem::MaybeUninit;
use std::ops::Range;

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
    /// Input packed into a contiguous buffer in column major order.
    Packed(&'a [T]),

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

/// Kernel that computes a small tile of a general matrix multiplication (GEMM)
/// or general matrix-vector multiplication (GEMV).
///
/// This trait is an interface for the architecture-specific inner loop for
/// matrix multiplication and matrix-vector multiplication, as well as the
/// methods that pack the input matrices into a format that is efficient for the
/// kernel to use.
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

    /// Return the width of this kernel's tiles.
    fn mr(&self) -> usize;

    /// Return the height of this kernel's tiles.
    fn nr(&self) -> usize;

    /// Return a name for this kernel for use in logging etc.
    fn name(&self) -> &'static str;

    /// Pack a block of the LHS / "A" input for use by this kernel.
    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<LhsT>],
        a: Matrix<LhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
    );

    /// Pack a block of the RHS / "B" input for use
    /// by this kernel.
    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<RhsT>],
        b: Matrix<RhsT>,
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
        used_rows: usize,
        b: &[RhsT],
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
