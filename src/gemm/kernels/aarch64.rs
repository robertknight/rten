use std::arch::aarch64::float32x4_t;

use rten_tensor::Matrix;
use rten_vecmath::simd_vec::SimdFloat;

use super::{simd_gemm, simd_gemv, Kernel};

#[derive(Default)]
pub struct ArmNeonKernel {
    _private: (),
}

// Safety - We assume that Rust code on Arm is always compiled with Arm Neon
// available.
unsafe impl Kernel for ArmNeonKernel {
    const MR: usize = 8;
    const NR: usize = 8;

    fn new() -> Option<Self> {
        Some(ArmNeonKernel { _private: () })
    }

    fn name(&self) -> &'static str {
        "arm-neon"
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
    ) {
        const MR: usize = ArmNeonKernel::MR;
        const NR: usize = ArmNeonKernel::NR;
        const NR_REGS: usize = NR / <float32x4_t as SimdFloat>::LEN;

        simd_gemm::<float32x4_t, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta);
    }

    fn gemv_kernel(&self, out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        // Safety - float32x4_t is supported if this kernel was constructed.
        unsafe {
            simd_gemv::<float32x4_t, 2>(out, a, b, alpha, beta);
        }
    }
}

super::impl_gemmops!(ArmNeonKernel);
