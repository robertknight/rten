use std::ops::Range;

use rten_tensor::Matrix;

use super::{GemmInputA, GemmInputB};

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

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
    /// It is unsafe to call `kernel` if this is false.
    fn supported() -> bool;

    /// Compute a tile of the output matrix. The output is stored in row-major
    /// order with `MR` rows and `NR` columns, a row stride of `tile_row_stride`
    /// and column stride of 1.
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
}

/// Object-safe trait for performing matrix multiplications and packing inputs
/// with a specific kernel.
pub trait GemmOps: Sync {
    fn name(&self) -> &str;
    fn pack_a_block(&self, out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>);
    fn pack_b_block(&self, out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>);
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
        impl crate::gemm::kernels::GemmOps for $kernel {
            fn name(&self) -> &str {
                <$kernel as crate::gemm::kernels::Kernel>::name()
            }

            fn pack_a_block(
                &self,
                out: &mut [f32],
                a: rten_tensor::Matrix,
                rows: std::ops::Range<usize>,
                cols: std::ops::Range<usize>,
            ) {
                crate::gemm::packing::pack_a_block::<Self>(out, a, rows, cols);
            }

            fn pack_b_block(
                &self,
                out: &mut [f32],
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

        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);

        // Accumulate into a fixed-sized array to allow the compiler to generate
        // more efficient code for the loop over `depth`.
        let mut tmp = [[0.0; NR]; MR];
        for k in 0..depth {
            let a_off = k * MR;
            let b_off = k * NR;

            for i in 0..MR {
                for j in 0..NR {
                    tmp[i][j] += a.get_unchecked(a_off + i) * b.get_unchecked(b_off + j);
                }
            }
        }

        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el = tmp[i][j];
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el += tmp[i][j];
                }
            }
        } else {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el = beta * *out_el + alpha * tmp[i][j];
                }
            }
        }
    }
}

impl_gemmops!(BaseKernel);
