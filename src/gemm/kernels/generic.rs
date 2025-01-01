use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::vec_count;
use rten_tensor::Matrix;

use super::simd_generic::{simd_gemm, simd_gemv};
use super::{Kernel, Lhs, TempTile};
use crate::gemm::packing::{pack_a_block, pack_b_block};

/// This is the base kernel that does not use architecture-specific intrinsics
/// but is autovectorization-friendly. It is expected to perform the same as
/// a kernel using SSE intrinsics (or equivalent).
pub struct GenericKernel {
    _private: (),
}

impl GenericKernel {
    const MR: usize = 8;

    // The base kernel will most likely be compiled to SSE or equivalent. SSE
    // registers are 128 bits wide = 4 x f32, so this should be a multiple of
    // that.
    const NR: usize = 4;
}

// Safety - Base kernel is always supported
unsafe impl Kernel<f32, f32, f32> for GenericKernel {
    fn new() -> Option<Self> {
        Some(GenericKernel { _private: () })
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
        a: Lhs<f32>,
        b: &[f32],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = GenericKernel::MR;
        const NR: usize = GenericKernel::NR;
        const NR_REGS: usize = vec_count::<f32>(NR);

        if used_cols == NR {
            simd_gemm::<f32, MR, NR_REGS>(
                tile_ptr,
                tile_row_stride,
                a,
                used_rows,
                b,
                depth,
                alpha,
                beta,
            );
        } else {
            let mut tmp_tile = TempTile::<f32, MR, NR>::new();
            simd_gemm::<f32, MR, NR_REGS>(
                tmp_tile.as_mut_ptr() as *mut f32,
                NR,
                a,
                used_rows,
                b,
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
        // Safety - f32 "SIMD" type is always supported
        unsafe {
            simd_gemv::<f32, 4>(out, a, b, alpha, beta);
        }
    }
}
