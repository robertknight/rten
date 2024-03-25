use rten_tensor::Matrix;
use rten_vecmath::simd_vec::wasm::v128f;
use rten_vecmath::simd_vec::SimdFloat;

use super::{simd_gemm, simd_gemv, Kernel};

#[derive(Default)]
pub struct WasmKernel {}

impl Kernel for WasmKernel {
    const MR: usize = 8;
    const NR: usize = 8;

    fn name() -> &'static str {
        "wasm32"
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
        const MR: usize = WasmKernel::MR;
        const NR: usize = WasmKernel::NR;
        const NR_REGS: usize = NR / <v128f as SimdFloat>::LEN;

        simd_gemm::<v128f, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta);
    }

    unsafe fn gemv_kernel(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        simd_gemv::<v128f, 4>(out, a, b, alpha, beta);
    }
}

super::impl_gemmops!(WasmKernel);
