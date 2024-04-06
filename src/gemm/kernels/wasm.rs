use rten_tensor::Matrix;
use rten_vecmath::simd_vec::wasm::v128f;
use rten_vecmath::simd_vec::SimdFloat;

use super::{simd_gemm, simd_gemv, Kernel};

#[derive(Default)]
pub struct WasmKernel {
    _private: (),
}

// Safety - Support for used WASM instructions is checked by the runtime when
// the WASM binary is loaded.
unsafe impl Kernel for WasmKernel {
    const MR: usize = 8;
    const NR: usize = 8;

    fn new() -> Option<Self> {
        Some(WasmKernel { _private: () })
    }

    fn name(&self) -> &'static str {
        "wasm32"
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
        const MR: usize = WasmKernel::MR;
        const NR: usize = WasmKernel::NR;
        const NR_REGS: usize = NR / <v128f as SimdFloat>::LEN;

        simd_gemm::<v128f, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta);
    }

    fn gemv_kernel(&self, out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        // Safety - WASM SIMD types are supported if this kernel was constructed.
        unsafe {
            simd_gemv::<v128f, 4>(out, a, b, alpha, beta);
        }
    }
}

super::impl_gemmops!(WasmKernel);
