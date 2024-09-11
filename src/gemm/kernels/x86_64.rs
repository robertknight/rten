use std::arch::x86_64::__m256;
use std::mem::MaybeUninit;
use std::ops::Range;

#[cfg(feature = "avx512")]
use std::arch::x86_64::__m512;

use rten_simd::vec_count;
use rten_tensor::Matrix;

#[cfg(feature = "avx512")]
use rten_simd::isa_detection::is_avx512_supported;

use super::simd_generic::{simd_gemm, simd_gemv};
use super::Kernel;
use crate::gemm::packing::{pack_a_block, pack_b_block};

/// Optimized kernel for x64 CPUs that support AVX + FMA instructions.
pub struct FmaKernel {
    _private: (),
}

impl FmaKernel {
    const MR: usize = 6;

    // Chosen to fit 2 AVX registers and take advantage of the two FMA
    // execution ports.
    const NR: usize = 16;
}

/// Wrapper for `pack_a_block` which enables AVX instructions.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn pack_a_block_avx<const MR: usize>(
    out: &mut [MaybeUninit<f32>],
    a: Matrix,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    pack_a_block::<f32, MR>(out, a, rows, cols);
}

/// Wrapper for `pack_b_block` which enables AVX instructions.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn pack_b_block_avx<const NR: usize>(
    out: &mut [MaybeUninit<f32>],
    b: Matrix,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    pack_b_block::<f32, NR>(out, b, rows, cols);
}

// Safety - The `new` fn tests for AVX-2 / FMA support.
unsafe impl Kernel<f32, f32, f32> for FmaKernel {
    fn new() -> Option<Self> {
        let supported = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        supported.then_some(FmaKernel { _private: () })
    }

    fn name(&self) -> &'static str {
        "fma"
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<f32>],
        a: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        // Safety: Kernel can only be constructed if AVX is supported.
        unsafe {
            pack_a_block_avx::<{ Self::MR }>(out, a, rows, cols);
        }
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<f32>],
        b: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        // Safety: Kernel can only be constructed if AVX is supported.
        unsafe {
            pack_b_block_avx::<{ Self::NR }>(out, b, rows, cols);
        }
    }

    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn kernel(
        &self,
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = FmaKernel::MR;
        const NR: usize = FmaKernel::NR;
        const NR_REGS: usize = vec_count::<__m256>(NR);

        simd_gemm::<__m256, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta);
    }

    fn gemv_kernel(&self, out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        unsafe fn gemv_kernel_impl(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
            simd_gemv::<__m256, 4>(out, a, b, alpha, beta);
        }
        // Safety: Kernel can only be constructed if supported.
        unsafe {
            gemv_kernel_impl(out, a, b, alpha, beta);
        }
    }
}

/// Optimized kernel for x64 CPUs that support AVX 512 instructions.
#[cfg(feature = "avx512")]
pub struct Avx512Kernel {
    _private: (),
}

#[cfg(feature = "avx512")]
impl Avx512Kernel {
    // The optimal value of MR depends on how many AVX-512 FMA units the CPU has.
    // Client Intel CPUs have one, server CPUs have two. This smaller value is
    // tuned for single-FMA CPUs.
    //
    // See https://github.com/robertknight/rten/issues/17.
    const MR: usize = 6;

    // 2 x 16-f32-wide registers.
    const NR: usize = 32;
}

// Safety - The `new` fn checks for AVX-512 support.
#[cfg(feature = "avx512")]
unsafe impl Kernel<f32, f32, f32> for Avx512Kernel {
    fn new() -> Option<Self> {
        is_avx512_supported().then_some(Avx512Kernel { _private: () })
    }

    fn name(&self) -> &'static str {
        "avx512"
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<f32>],
        a: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        // Safety: We assume AVX-512 implies availability of AVX 2.
        unsafe {
            pack_a_block_avx::<{ Self::MR }>(out, a, rows, cols);
        }
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<f32>],
        b: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        // Safety: We assume AVX-512 implies availability of AVX 2.
        unsafe {
            pack_b_block_avx::<{ Self::NR }>(out, b, rows, cols);
        }
    }

    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn kernel(
        &self,
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = Avx512Kernel::MR;
        const NR: usize = Avx512Kernel::NR;
        const NR_REGS: usize = vec_count::<__m512>(NR);

        simd_gemm::<__m512, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta)
    }

    fn gemv_kernel(&self, out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn gemv_kernel_impl(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
            simd_gemv::<__m512, 2>(out, a, b, alpha, beta);
        }
        // Safety: Kernel can only be constructed if supported.
        unsafe {
            gemv_kernel_impl(out, a, b, alpha, beta);
        }
    }
}
