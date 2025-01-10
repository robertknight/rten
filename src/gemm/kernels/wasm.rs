use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::arch::wasm::{v128f, v128i};
use rten_simd::vec_count;
use rten_tensor::{Matrix, MatrixLayout};

use super::simd_generic::{simd_gemv, GemmDispatch};
use super::{Kernel, Lhs, PackedLayout, QuantParams, TempTile};
use crate::gemm::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::gemm::Im2Col;
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

    fn packed_a_layout(
        &self,
        a: Matrix,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<f32>>,
    ) -> PackedLayout {
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
        _quant: Option<QuantParams<f32>>,
    ) {
        let out = cast_pod_mut_slice(out).unwrap();
        pack_a_block::<f32, { Self::MR }>(out, a, rows, cols);
    }

    fn packed_b_layout(
        &self,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<f32>>,
    ) -> PackedLayout {
        packed_b_layout::<f32, { Self::NR }>(rows, cols)
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        b: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
        _quant: Option<QuantParams<f32>>,
    ) {
        let out = cast_pod_mut_slice(out).unwrap();
        pack_b_block::<f32, { Self::NR }>(out, b, rows, cols);
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<f32>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        const NR_REGS: usize = vec_count::<v128f>(WasmKernel::NR);

        // Safety: WASM SIMD types are supported
        let out = cast_pod_mut_slice(out).unwrap();
        unsafe {
            image.pack_block::<v128i, NR_REGS>(out, Self::NR, rows, cols);
        }
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
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        const MR: usize = WasmKernel::MR;
        const NR: usize = WasmKernel::NR;
        const NR_REGS: usize = vec_count::<v128f>(NR);

        let b = cast_pod_slice(b).unwrap();
        let mut tmp_tile = TempTile::<f32, MR, NR>::new();
        let (dest_ptr, dest_row_stride, dest_beta) = if used_cols == NR {
            (tile_ptr, tile_row_stride, beta)
        } else {
            (tmp_tile.as_mut_ptr() as *mut f32, NR, 0.)
        };

        let gemm = GemmDispatch::<v128f, MR, NR_REGS>::new(
            dest_ptr,
            dest_row_stride,
            a,
            b,
            depth,
            alpha,
            dest_beta,
        );

        match used_rows {
            8 => gemm.dispatch::<8>(),
            7 => gemm.dispatch::<7>(),
            6 => gemm.dispatch::<6>(),
            5 => gemm.dispatch::<5>(),
            4 => gemm.dispatch::<4>(),
            3 => gemm.dispatch::<3>(),
            2 => gemm.dispatch::<2>(),
            1 => gemm.dispatch::<1>(),
            _ => panic!("unsupported `used_rows` {}", used_rows),
        }

        if used_cols != NR {
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
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        // Safety - WASM SIMD types are supported if this kernel was constructed.
        unsafe {
            simd_gemv::<v128f, 4>(out, a, b, alpha, beta);
        }
    }
}
