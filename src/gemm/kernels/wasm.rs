use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::arch::wasm::v128f;
use rten_simd::vec_count;
use rten_tensor::Matrix;

use super::simd_generic::{simd_gemm, simd_gemv};
use super::Kernel;
use crate::gemm::packing::{pack_a_block, pack_b_block};

pub struct WasmKernel {
    _private: (),
}

impl WasmKernel {
    const MR: usize = 8;
    const NR: usize = 8;
}

// Safety - Support for used WASM instructions is checked by the runtime when
// the WASM binary is loaded.
unsafe impl Kernel<f32, f32, f32> for WasmKernel {
    fn new() -> Option<Self> {
        #[cfg(target_feature = "simd128")]
        return Some(WasmKernel { _private: () });

        #[cfg(not(target_feature = "simd128"))]
        None
    }

    fn name(&self) -> &'static str {
        "wasm32"
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
    ) {
        const MR: usize = WasmKernel::MR;
        const NR: usize = WasmKernel::NR;
        const NR_REGS: usize = vec_count::<v128f>(NR);

        simd_gemm::<v128f, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta);
    }

    fn gemv_kernel(&self, out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        // Safety - WASM SIMD types are supported if this kernel was constructed.
        unsafe {
            simd_gemv::<v128f, 4>(out, a, b, alpha, beta);
        }
    }
}
