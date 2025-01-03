use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::arch::wasm::v128f;
use rten_simd::vec_count;
use rten_tensor::{Matrix, MatrixLayout};

use super::simd_generic::{simd_gemm, simd_gemv};
use super::{Kernel, Lhs, PackedLayout, TempTile};
use crate::gemm::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::number::{cast_pod_mut_slice, cast_pod_slice};

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

    fn packed_a_layout(&self, a: Matrix, rows: usize, cols: usize) -> PackedLayout {
        let mut info = packed_a_layout::<f32, { Self::MR }>(rows, cols);
        info.must_pack = a.col_stride() != 1;
        info
    }

    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        a: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        let out = cast_pod_mut_slice(out).unwrap();
        pack_a_block::<f32, { Self::MR }>(out, a, rows, cols);
    }

    fn packed_b_layout(&self, rows: usize, cols: usize) -> PackedLayout {
        packed_b_layout::<f32, { Self::NR }>(rows, cols)
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        b: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        let out = cast_pod_mut_slice(out).unwrap();
        pack_b_block::<f32, { Self::NR }>(out, b, rows, cols);
    }

    unsafe fn kernel(
        &self,
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: Lhs<f32>,
        b: &[u8],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = WasmKernel::MR;
        const NR: usize = WasmKernel::NR;
        const NR_REGS: usize = vec_count::<v128f>(NR);

        if used_cols == NR {
            simd_gemm::<v128f, MR, NR_REGS>(
                tile_ptr,
                tile_row_stride,
                a,
                used_rows,
                cast_pod_slice(b).unwrap(),
                depth,
                alpha,
                beta,
            );
        } else {
            let mut tmp_tile = TempTile::<f32, MR, NR>::new();
            simd_gemm::<v128f, MR, NR_REGS>(
                tmp_tile.as_mut_ptr() as *mut f32,
                NR,
                a,
                used_rows,
                cast_pod_slice(b).unwrap(),
                depth,
                alpha,
                0.,
            );
            tmp_tile.accumulate_into(
                tile_ptr as *mut MaybeUninit<f32>,
                used_rows,
                used_cols,
                tile_row_stride,
                beta,
            );
        }
    }

    fn gemv_kernel(
        &self,
        out: &mut [MaybeUninit<f32>],
        a: &[f32],
        b: Matrix,
        alpha: f32,
        beta: f32,
    ) {
        // Safety - WASM SIMD types are supported if this kernel was constructed.
        unsafe {
            simd_gemv::<v128f, 4>(out, a, b, alpha, beta);
        }
    }
}
