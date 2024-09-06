use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::{vec_count, SimdFloat};
use rten_tensor::{Matrix, MatrixLayout, Storage};

use crate::gemm::packing::{pack_a_block, pack_b_block};
use crate::iter_util::{range_chunks_exact, unroll_loop};

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub mod wasm;

/// Compute an output block of a vector-matrix product ("gemv" in BLAS APIs).
///
/// Multiple output columns are computed at a time, using `NR_REGS` SIMD
/// registers of type `S`. See [Kernel::gemv_kernel].
///
/// Safety: The `SimdFloat` type must be supported on the current system.
#[inline(always)]
unsafe fn simd_gemv<S: SimdFloat, const NR_REGS: usize>(
    out: &mut [f32],
    a: &[f32],
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    // Handle cases where `b` does not have unit stride.
    if b.row_stride() == 1 {
        return simd_gemv_transposed::<S>(out, a, b, alpha, beta);
    } else if b.col_stride() != 1 {
        return simd_gemv_fallback(out, a, b, alpha, beta);
    }

    assert!(b.col_stride() == 1);
    assert!(a.len() == b.rows());
    assert!(out.len() == b.cols());

    let out_ptr = out.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.storage().as_ptr();
    let b_row_stride = b.row_stride();

    let mut b_tiles = range_chunks_exact(0..b.cols(), NR_REGS * S::LEN);
    for b_tile in b_tiles.by_ref() {
        let mut acc = [S::zero(); NR_REGS];
        unroll_loop!(0..a.len(), k, 4, {
            let a_elt = *a_ptr.add(k);
            let a_elts = S::splat(a_elt);

            // Pre-fetch the current row for the next column tile.
            S::prefetch(b_ptr.add(k * b_row_stride + b_tile.start + NR_REGS + S::LEN));

            for i in 0..NR_REGS {
                let b_elts = S::load(b_ptr.add(k * b_row_stride + b_tile.start + i * S::LEN));
                acc[i] = a_elts.mul_add(b_elts, acc[i]);
            }
        });

        if alpha != 1. {
            let alpha_vec = S::splat(alpha);
            for i in 0..NR_REGS {
                acc[i] = acc[i].mul(alpha_vec);
            }
        }

        let get_out_tile_ptr = |i| out_ptr.add(b_tile.start + i * S::LEN);

        if beta == 0. {
            for i in 0..NR_REGS {
                acc[i].store(get_out_tile_ptr(i));
            }
        } else if beta == 1. {
            for i in 0..NR_REGS {
                let out_tile_ptr = get_out_tile_ptr(i);
                let out_tile = S::load(out_tile_ptr).add(acc[i]);
                out_tile.store(out_tile_ptr);
            }
        } else {
            let beta_vec = S::splat(beta);
            for i in 0..NR_REGS {
                let out_tile_ptr = get_out_tile_ptr(i);
                let out_tile = S::load(out_tile_ptr).mul_add(beta_vec, acc[i]);
                out_tile.store(out_tile_ptr);
            }
        }
    }

    for c in b_tiles.remainder() {
        let mut acc = 0.;
        for k in 0..a.len() {
            acc += *a_ptr.add(k) * *b_ptr.add(k * b_row_stride + c);
        }
        let out_el = out_ptr.add(c);
        let tmp = if beta == 0. { 0. } else { *out_el };
        *out_el = beta * tmp + acc * alpha;
    }
}

/// Variant of [simd_gemv] which handles the case where `b` has unit row stride.
#[inline(always)]
unsafe fn simd_gemv_transposed<S: SimdFloat>(
    out: &mut [f32],
    a: &[f32],
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    assert!(b.row_stride() == 1);
    assert!(a.len() == b.rows());
    assert!(out.len() == b.cols());

    let b_ptr = b.storage().as_ptr();
    let b_col_stride = b.col_stride();

    const COL_TILE: usize = 8;

    let mut col_tiles = range_chunks_exact(0..b.cols(), COL_TILE);
    for col_tile in col_tiles.by_ref() {
        let mut acc = [S::zero(); COL_TILE];

        let mut depth_tiles = range_chunks_exact(0..a.len(), S::LEN);
        for depth_tile in depth_tiles.by_ref() {
            let a_tile = S::load(a.as_ptr().add(depth_tile.start));
            for i in 0..COL_TILE {
                let b_col_ptr = b_ptr.add((col_tile.start + i) * b_col_stride);
                let b_tile = S::load(b_col_ptr.add(depth_tile.start));
                acc[i] = a_tile.mul_add(b_tile, acc[i]);
            }
        }

        let mut acc: [f32; COL_TILE] = std::array::from_fn(|i| acc[i].sum());
        for k in depth_tiles.remainder() {
            let ak = *a.get_unchecked(k);
            for i in 0..COL_TILE {
                let b_col_ptr = b_ptr.add((col_tile.start + i) * b_col_stride);
                let bk = *b_col_ptr.add(k);
                acc[i] = ak.mul_add(bk, acc[i]);
            }
        }

        if beta == 0. {
            for i in 0..COL_TILE {
                out[col_tile.start + i] = alpha * acc[i];
            }
        } else {
            for i in 0..COL_TILE {
                let out_val = alpha * acc[i] + beta * out[col_tile.start + i];
                out[col_tile.start + i] = out_val;
            }
        }
    }

    let last_col_tile = col_tiles.remainder();
    if !last_col_tile.is_empty() {
        simd_gemv_fallback(
            &mut out[last_col_tile.clone()],
            a,
            b.slice::<2, _>((.., last_col_tile)),
            alpha,
            beta,
        );
    }
}

/// Variant of [simd_gemv] which handles the case where `b` has non-unit strides
/// for rows and columns.
///
/// This doesn't benefit from SIMD operations. It is at least inlined so it
/// can benefit from the kernel's instruction set (eg. for FMA operations).
#[inline(always)]
fn simd_gemv_fallback(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
    assert!(a.len() == b.rows());
    assert!(out.len() == b.cols());

    for (col, out) in out.iter_mut().enumerate() {
        let mut acc = 0.;
        for (k, ak) in (0..a.len()).zip(a.iter()) {
            let bk = unsafe { *b.get_unchecked([k, col]) };
            acc = ak.mul_add(bk, acc);
        }
        acc *= alpha;
        if beta == 0. {
            *out = acc;
        } else {
            *out = acc + beta * *out;
        }
    }
}

/// Compute a tile of matrix-multiplication output.
///
/// `S` specifies the SIMD vector type, `MR` is the number of rows in the tile
/// and `NR_REGS` specifies the number of columns in the tile as a multiple of
/// the SIMD register width.
///
/// See [Kernel::kernel].
///
/// Safety: The `SimdFloat` type must be supported on the current system.
#[inline(always)]
unsafe fn simd_gemm<S: SimdFloat, const MR: usize, const NR_REGS: usize>(
    tile_ptr: *mut f32,
    tile_row_stride: usize,
    a: &[f32],
    b: &[f32],
    depth: usize,
    alpha: f32,
    beta: f32,
) {
    // Check that buffer accesses below are going to be valid.
    assert!(a.len() >= depth * MR);
    assert!(b.len() >= depth * NR_REGS * S::LEN);
    assert!(depth > 0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut tmp = [[S::zero(); NR_REGS]; MR];
    let mut b_rows = [S::zero(); NR_REGS];

    unroll_loop!(0..depth - 1, k, 4, {
        let a_off = k * MR;
        let b_off = k * NR_REGS * S::LEN;

        // Prefetch B for the next iteration
        S::prefetch(b_ptr.add((k + 1) * NR_REGS * S::LEN));

        for i in 0..NR_REGS {
            b_rows[i] = S::load(b_ptr.add(b_off + i * S::LEN));
        }

        for i in 0..MR {
            let a_val = *a_ptr.add(a_off + i);
            let a_broadcast = S::splat(a_val);

            for j in 0..NR_REGS {
                tmp[i][j] = a_broadcast.mul_add(b_rows[j], tmp[i][j]);
            }
        }
    });

    // Prefetch output before the final computation loop
    for i in 0..MR {
        S::prefetch_write(tile_ptr.add(tile_row_stride * i));
    }

    // Perform final outer product update.
    let k = depth - 1;
    let a_off = k * MR;
    let b_off = k * NR_REGS * S::LEN;

    for i in 0..NR_REGS {
        b_rows[i] = S::load(b_ptr.add(b_off + i * S::LEN));
    }

    for i in 0..MR {
        let a_val = *a_ptr.add(a_off + i);
        let a_broadcast = S::splat(a_val);

        for j in 0..NR_REGS {
            tmp[i][j] = a_broadcast.mul_add(b_rows[j], tmp[i][j]);
        }
    }

    let get_out_ptr = |i, j| tile_ptr.add(tile_row_stride * i + j * S::LEN);

    // Write to output tile.
    //
    // We have special cases for zero/one values of alpha and beta, both for
    // performance in the common cases where (alpha, beta) are (0, 1) or (1, 1)
    // and because when beta is zero, the destination may be uninitialized and
    // must not be read.
    if beta == 0. && alpha == 1. {
        for i in 0..MR {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                tmp[i][j].store(out_ptr);
            }
        }
    } else if beta == 1. && alpha == 1. {
        for i in 0..MR {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = S::load(out_ptr).add(tmp[i][j]);
                out_val.store(out_ptr);
            }
        }
    } else if beta == 0. {
        let alpha_broadcast = S::splat(alpha);

        for i in 0..MR {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = tmp[i][j].mul(alpha_broadcast);
                out_val.store(out_ptr);
            }
        }
    } else {
        let alpha_broadcast = S::splat(alpha);
        let beta_broadcast = S::splat(beta);

        for i in 0..MR {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = S::load(out_ptr).mul(beta_broadcast);
                let out_val = tmp[i][j].mul_add(alpha_broadcast, out_val);
                out_val.store(out_ptr);
            }
        }
    }
}

/// Kernel that computes a small tile of a matrix multiplication output.
///
/// The matrix multiplication takes input matrices with element types `LhsT`
/// and `RhsT` and produces an output with element type `OutT`.
///
/// The kernel corresponds to Loop 6 (the "microkernel") in Page 4 of [^1]. The
/// tile size depends upon the kernel and is specified by the `MR` and `NR`
/// associated constants. See Section 3.2 [^1] for theory behind choosing the
/// `MR` and `NR` values.
///
/// # Safety
///
/// It must only be possible to construct the kernel using `new` if the
/// instructions it uses are supported on the current system.
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

    /// Compute a tile of the output matrix. The output is stored in row-major
    /// order with `MR` rows and `NR` columns, a row stride of `tile_row_stride`
    /// and column stride of 1.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `tile_ptr` points to a buffer of the correct
    /// size.
    unsafe fn kernel(
        &self,
        tile_ptr: *mut OutT,
        tile_row_stride: usize,
        a: &[LhsT],
        b: &[RhsT],
        depth: usize,
        alpha: f32,
        beta: OutT,
        a_zero_point: LhsT,
        b_zero_point: RhsT,
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
    /// The caller must ensure that the kernel is supported on the current
    /// system.
    fn gemv_kernel(
        &self,
        out: &mut [OutT],
        a: &[LhsT],
        b: Matrix<RhsT>,
        alpha: f32,
        beta: OutT,
        a_zero_point: LhsT,
        b_zero_point: RhsT,
    );
}

/// This is the base kernel that does not use architecture-specific intrinsics
/// but is autovectorization-friendly. It is expected to perform the same as
/// a kernel using SSE intrinsics (or equivalent).
#[derive(Default)]
pub struct BaseKernel {
    _private: (),
}

impl BaseKernel {
    const MR: usize = 8;

    // The base kernel will most likely be compiled to SSE or equivalent. SSE
    // registers are 128 bits wide = 4 x f32, so this should be a multiple of
    // that.
    const NR: usize = 4;
}

// Safety - Base kernel is always supported
unsafe impl Kernel<f32, f32, f32> for BaseKernel {
    fn new() -> Option<Self> {
        Some(BaseKernel { _private: () })
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn name(&self) -> &'static str {
        "base"
    }

    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<f32>],
        a: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        pack_a_block::<f32, { Self::MR }>(out, a, rows, cols);
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<f32>],
        b: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        pack_b_block::<f32, { Self::NR }>(out, b, rows, cols);
    }

    unsafe fn kernel(
        &self,
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
        _a_zero_point: f32,
        _b_zero_point: f32,
    ) {
        const MR: usize = BaseKernel::MR;
        const NR: usize = BaseKernel::NR;
        const NR_REGS: usize = vec_count::<f32>(NR);
        simd_gemm::<f32, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta);
    }

    fn gemv_kernel(
        &self,
        out: &mut [f32],
        a: &[f32],
        b: Matrix<f32>,
        alpha: f32,
        beta: f32,
        _a_zero_point: f32,
        _b_zero_point: f32,
    ) {
        // Safety - f32 "SIMD" type is always supported
        unsafe {
            simd_gemv::<f32, 4>(out, a, b, alpha, beta);
        }
    }
}

pub struct BaseU8S8Kernel {
    _private: (),
}

impl BaseU8S8Kernel {
    const MR: usize = 8;

    // The base kernel will most likely be compiled to SSE or equivalent. SSE
    // registers are 128 bits wide = 4 x f32, so this should be a multiple of
    // that.
    const NR: usize = 4;
}

// Safety - Base kernel is always supported
unsafe impl Kernel<u8, i8, i32> for BaseU8S8Kernel {
    fn new() -> Option<Self> {
        Some(BaseU8S8Kernel { _private: () })
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn name(&self) -> &'static str {
        "base_u8s8"
    }

    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        a: Matrix<u8>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        pack_a_block::<u8, { Self::MR }>(out, a, rows, cols);
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<i8>],
        b: Matrix<i8>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        pack_b_block::<i8, { Self::NR }>(out, b, rows, cols);
    }

    unsafe fn kernel(
        &self,
        tile_ptr: *mut i32,
        tile_row_stride: usize,
        a: &[u8],
        b: &[i8],
        depth: usize,
        _alpha: f32,
        beta: i32,
        a_zero_point: u8,
        b_zero_point: i8,
    ) {
        const MR: usize = BaseU8S8Kernel::MR;
        const NR: usize = BaseU8S8Kernel::NR;

        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);
        assert!(depth > 0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut tmp = [[0i32; NR]; MR];
        let mut b_rows = [0i32; NR];

        for k in 0..depth {
            let a_off = k * MR;
            let b_off = k * NR;

            for i in 0..NR {
                b_rows[i] = *b_ptr.add(b_off + i) as i32 - b_zero_point as i32;
            }

            for i in 0..MR {
                let a_val = *a_ptr.add(a_off + i) as i32 - a_zero_point as i32;

                for j in 0..NR {
                    tmp[i][j] += a_val * b_rows[j];
                }
            }
        }

        let get_out_ptr = |i, j| tile_ptr.add(tile_row_stride * i + j);

        // Write to output tile.
        //
        // We have special cases for zero/one values of alpha and beta, both for
        // performance in the common cases where (alpha, beta) are (0, 1) or (1, 1)
        // and because when beta is zero, the destination may be uninitialized and
        // must not be read.
        if beta == 0 {
            for i in 0..MR {
                for j in 0..NR {
                    *get_out_ptr(i, j) = tmp[i][j];
                }
            }
        } else if beta == 1 {
            for i in 0..MR {
                for j in 0..NR {
                    *get_out_ptr(i, j) += tmp[i][j];
                }
            }
        } else {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = get_out_ptr(i, j);
                    *out_el += *out_el * tmp[i][j];
                }
            }
        }
    }

    fn gemv_kernel(
        &self,
        out: &mut [i32],
        a: &[u8],
        b: Matrix<i8>,
        _alpha: f32,
        beta: i32,
        a_zero_point: u8,
        b_zero_point: i8,
    ) {
        assert!(a.len() == b.rows());
        assert!(out.len() == b.cols());

        for (col, out) in out.iter_mut().enumerate() {
            let mut acc = 0;

            for (k, &ak) in (0..a.len()).zip(a.iter()) {
                let bk = unsafe { *b.get_unchecked([k, col]) };
                acc += (ak as i32 - a_zero_point as i32) * (bk as i32 - b_zero_point as i32);
            }

            if beta == 0 {
                *out = acc;
            } else {
                *out = acc + beta * *out;
            }
        }
    }
}
