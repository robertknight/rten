use std::arch::aarch64::float32x4_t;
use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::vec_count;
use rten_tensor::{Matrix, MatrixLayout};

use super::simd_generic::{simd_gemm, simd_gemv};
use super::{Kernel, Lhs, PackedLayout, TempTile};
use crate::gemm::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::number::{cast_pod_mut_slice, cast_pod_slice};

pub struct ArmNeonKernel {
    _private: (),
}

impl ArmNeonKernel {
    const MR: usize = 8;
    const NR: usize = 8;
}

// Safety - We assume that Rust code on Arm is always compiled with Arm Neon
// available.
unsafe impl Kernel<f32, f32, f32> for ArmNeonKernel {
    fn new() -> Option<Self> {
        Some(ArmNeonKernel { _private: () })
    }

    fn name(&self) -> &'static str {
        "arm-neon"
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
        let out = cast_pod_mut_slice(out).expect("incorrect alignment for packing buffer");
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
        let out = cast_pod_mut_slice(out).expect("incorrect alignment for packing buffer");
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
        const MR: usize = ArmNeonKernel::MR;
        const NR: usize = ArmNeonKernel::NR;
        const NR_REGS: usize = vec_count::<float32x4_t>(NR);

        let b = cast_pod_slice(b).unwrap();

        if used_cols == NR {
            simd_gemm::<float32x4_t, MR, NR_REGS>(
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
            simd_gemm::<float32x4_t, MR, NR_REGS>(
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
        // Safety - float32x4_t is supported if this kernel was constructed.
        unsafe {
            simd_gemv::<float32x4_t, 4>(out, a, b, alpha, beta);
        }
    }
}
