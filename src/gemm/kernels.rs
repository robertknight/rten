use std::mem::MaybeUninit;
use std::ops::Range;

use rten_tensor::{Matrix, MatrixLayout};
use rten_vecmath::simd_vec::SimdFloat;

use super::{GemmInputA, GemmInputB};
use crate::iter_util::{range_chunks_exact, unroll_loop};

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// Compute an output block of a vector-matrix product ("gemv" in BLAS APIs).
///
/// Multiple output columns are computed at a time, using `NR_REGS` SIMD
/// registers of type `S`. See [Kernel::gemv_kernel].
#[inline(always)]
unsafe fn simd_gemv<S: SimdFloat, const NR_REGS: usize>(
    out: &mut [f32],
    a: &[f32],
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    assert!(b.col_stride() == 1);
    assert!(a.len() == b.rows());
    assert!(out.len() == b.cols());

    let out_ptr = out.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.non_contiguous_data().as_ptr();
    let b_row_stride = b.row_stride();

    let mut b_tiles = range_chunks_exact(0..b.cols(), NR_REGS * S::LEN);
    for b_tile in b_tiles.by_ref() {
        let mut acc = [S::zero(); NR_REGS];
        for k in 0..a.len() {
            let a_elt = *a_ptr.add(k);
            let a_elts = S::splat(a_elt);

            for i in 0..NR_REGS {
                let b_elts = S::load(b_ptr.add(k * b_row_stride + b_tile.start + i * S::LEN));
                acc[i] = a_elts.mul_add(b_elts, acc[i]);
            }
        }

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
        *out_el = (*out_el * beta) + acc * alpha;
    }
}

/// Compute a tile of matrix-multiplication output.
///
/// `S` specifies the SIMD vector type, `MR` is the number of rows in the tile
/// and `NR_REGS` specifies the number of columns in the tile as a multiple of
/// the SIMD register width.
///
/// See [Kernel::kernel].
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

    // Write to output tile
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
/// The kernel corresponds to Loop 6 (the "microkernel") in Page 4 of [^1]. The
/// tile size depends upon the kernel and is specified by the `MR` and `NR`
/// associated constants. See Section 3.2 [^1] for theory behind choosing the
/// `MR` and `NR` values.
///
/// [^1]: https://dl.acm.org/doi/pdf/10.1145/2925987
pub trait Kernel {
    /// Height of output tiles computed by the kernel.
    const MR: usize;

    /// Width of output tiles computed by the kernel.
    const NR: usize;

    /// Return a name for this kernel for use in logging etc.
    fn name() -> &'static str;

    /// Return true if this kernel is usable on the current system.
    ///
    /// Unsafe functions in this trait can only be called if this returns true.
    fn supported() -> bool;

    /// Compute a tile of the output matrix. The output is stored in row-major
    /// order with `MR` rows and `NR` columns, a row stride of `tile_row_stride`
    /// and column stride of 1.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the kernel is supported on the current
    /// system, and `tile_ptr` points to a buffer of the correct size.
    unsafe fn kernel(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
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
    unsafe fn gemv_kernel(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        simd_gemv::<f32, 4>(out, a, b, alpha, beta);
    }
}

/// Object-safe trait for performing matrix multiplications and packing inputs
/// with a specific kernel.
///
/// # Safety
///
/// The packing functions must initialize all elements of the output buffers
/// passed to them.
pub unsafe trait GemmOps: Sync {
    fn name(&self) -> &str;

    /// Pack elements of `a` into a packing buffer for use by the matrix
    /// multiplication kernel.
    ///
    /// Implementations must initialize all elements of `out`.
    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<f32>],
        a: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    );

    /// Pack elements of `b` into a packing buffer for use by the matrix
    /// multiplication kernel.
    ///
    /// Implementations must initialize all elements of `out`.
    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<f32>],
        a: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    );

    fn gemm(
        &self,
        out_data: &mut [f32],
        out_row_stride: usize,
        a: GemmInputA,
        b: GemmInputB,
        alpha: f32,
        beta: f32,
        bias: Option<&[f32]>,
    );
}

/// Implement `GemmOps` for a specific kernel. A macro is used instead of
/// `impl<K: Kernel> GemmOps for K` to work around const generics limitations in
/// stable Rust.
macro_rules! impl_gemmops {
    ($kernel:ident) => {
        // Safety - The packing functions initialize all elements of their output.
        unsafe impl crate::gemm::kernels::GemmOps for $kernel {
            fn name(&self) -> &str {
                <$kernel as crate::gemm::kernels::Kernel>::name()
            }

            fn pack_a_block(
                &self,
                out: &mut [std::mem::MaybeUninit<f32>],
                a: rten_tensor::Matrix,
                rows: std::ops::Range<usize>,
                cols: std::ops::Range<usize>,
            ) {
                crate::gemm::packing::pack_a_block::<Self>(out, a, rows, cols);
            }

            fn pack_b_block(
                &self,
                out: &mut [std::mem::MaybeUninit<f32>],
                a: rten_tensor::Matrix,
                rows: std::ops::Range<usize>,
                cols: std::ops::Range<usize>,
            ) {
                crate::gemm::packing::pack_b_block::<Self>(out, a, rows, cols);
            }

            fn gemm(
                &self,
                out_data: &mut [f32],
                out_row_stride: usize,
                a: crate::gemm::GemmInputA,
                b: crate::gemm::GemmInputB,
                alpha: f32,
                beta: f32,
                bias: Option<&[f32]>,
            ) {
                crate::gemm::gemm_impl::<Self, { Self::MR * Self::NR }>(
                    out_data,
                    out_row_stride,
                    a,
                    b,
                    alpha,
                    beta,
                    bias,
                )
            }
        }
    };
}

use impl_gemmops;

/// This is the base kernel that does not use architecture-specific intrinsics
/// but is autovectorization-friendly. It is expected to perform the same as
/// a kernel using SSE intrinsics (or equivalent).
#[derive(Default)]
pub struct BaseKernel {}

impl Kernel for BaseKernel {
    const MR: usize = 8;

    // The base kernel will most likely be compiled to SSE or equivalent. SSE
    // registers are 128 bits wide = 4 x f32, so this should be a multiple of
    // that.
    const NR: usize = 4;

    fn name() -> &'static str {
        "base"
    }

    fn supported() -> bool {
        true
    }

    unsafe fn kernel(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = BaseKernel::MR;
        const NR: usize = BaseKernel::NR;
        const NR_REGS: usize = NR / <f32 as SimdFloat>::LEN;
        simd_gemm::<f32, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta);
    }
}

impl_gemmops!(BaseKernel);
