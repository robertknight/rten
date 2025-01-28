use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::vec_count;
use rten_tensor::{Matrix, MatrixLayout};

use super::simd_generic::{simd_gemv, GemmDispatch};
use super::{Kernel, Lhs, PackedLayout, QuantParams, TempTile};
use crate::gemm::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::gemm::Im2Col;
use crate::slice_cast::{cast_pod_mut_slice, cast_pod_slice, Pod};

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
        const NR_REGS: usize = vec_count::<f32>(GenericKernel::NR);

        // Safety: Scalar "SIMD" types are always supported
        let out = cast_pod_mut_slice(out).unwrap();
        unsafe {
            image.pack_block::<i32, NR_REGS>(out, Self::NR, rows, cols);
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
        const MR: usize = GenericKernel::MR;
        const NR: usize = GenericKernel::NR;
        const NR_REGS: usize = vec_count::<f32>(NR);

        let b = cast_pod_slice(b).unwrap();
        let mut tmp_tile = TempTile::<f32, MR, NR>::new();
        let (dest_ptr, dest_row_stride, dest_beta) = if used_cols == NR {
            (tile_ptr, tile_row_stride, beta)
        } else {
            (tmp_tile.as_mut_ptr() as *mut f32, NR, 0.)
        };

        let gemm = GemmDispatch::<f32, MR, NR_REGS>::new(
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
        // Safety - f32 "SIMD" type is always supported
        unsafe {
            simd_gemv::<f32, 4>(out, a, b, alpha, beta);
        }
    }
}

unsafe impl<LhsT, RhsT> Kernel<LhsT, RhsT, i32> for GenericKernel
where
    LhsT: Pod + Default + Into<i32> + std::fmt::Debug,
    RhsT: Pod + Default + Into<i32> + std::fmt::Debug,
{
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
        "generic-i8"
    }

    fn packed_a_layout(
        &self,
        _a: Matrix<LhsT>,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<LhsT>>,
    ) -> PackedLayout {
        let mut info = packed_a_layout::<u8, { Self::MR }>(rows, cols);
        info.must_pack = true;
        info
    }

    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        a: Matrix<LhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
        _quant: Option<QuantParams<LhsT>>,
    ) {
        let out = cast_pod_mut_slice(out).unwrap();
        pack_a_block::<LhsT, { Self::MR }>(out, a, rows, cols);
    }

    fn packed_b_layout(
        &self,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<RhsT>>,
    ) -> PackedLayout {
        packed_b_layout::<RhsT, { Self::NR }>(rows, cols)
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        b: Matrix<RhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
        _quant: Option<QuantParams<RhsT>>,
    ) {
        let out = cast_pod_mut_slice(out).unwrap();
        pack_b_block::<RhsT, { Self::NR }>(out, b, rows, cols);
    }

    fn pack_im2col(
        &self,
        _out: &mut [MaybeUninit<u8>],
        _image: &Im2Col<RhsT>,
        _rows: Range<usize>,
        _cols: Range<usize>,
    ) {
        unimplemented!("im2col packing not implemented");
    }

    unsafe fn kernel(
        &self,
        tile_ptr: *mut i32,
        tile_row_stride: usize,
        a: Lhs<LhsT>,
        b: &[u8],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        alpha: f32,
        beta: i32,
        a_quant: Option<QuantParams<LhsT>>,
        b_quant: Option<QuantParams<RhsT>>,
    ) {
        assert_eq!(alpha, 1.);
        assert!(beta == 0 || beta == 1, "unsupported beta value");
        assert!(used_rows <= MR);
        assert!(used_cols <= NR);

        const MR: usize = GenericKernel::MR;
        const NR: usize = GenericKernel::NR;

        let a_data = match a {
            Lhs::Packed(packed) => packed,
            Lhs::Unpacked { .. } => panic!("inputs must be packed"),
        };
        let a_data: &[LhsT] = cast_pod_slice(a_data).unwrap();
        let a_row_stride = depth;

        let mut a_zero_point = [0i32; MR];
        if let Some(a_quant) = a_quant {
            #[allow(clippy::manual_memcpy)]
            for row in 0..used_rows {
                a_zero_point[row] = a_quant.zero_point[row].into();
            }
        }
        let mut b_zero_point = [0i32; NR];
        if let Some(b_quant) = b_quant {
            #[allow(clippy::manual_memcpy)]
            for col in 0..used_cols {
                b_zero_point[col] = b_quant.zero_point[col].into();
            }
        }

        let b: &[RhsT] = cast_pod_slice(b).unwrap();
        let use_tmp_tile = used_cols < NR || used_rows < MR;

        let mut tmp_tile = TempTile::<i32, MR, NR>::new();
        let (dest_ptr, dest_row_stride, dest_beta) = if !use_tmp_tile {
            (tile_ptr, tile_row_stride, beta)
        } else {
            (tmp_tile.as_mut_ptr() as *mut i32, NR, 0)
        };

        let mut tmp = [[0i32; NR]; MR];
        for k in 0..depth {
            for row in 0..MR {
                let a_i32: i32 = unsafe { (*a_data.get_unchecked(row * a_row_stride + k)).into() };
                let a_i32 = a_i32 - a_zero_point[row];
                for col in 0..NR {
                    let b_i32: i32 = unsafe { (*b.get_unchecked(k * NR + col)).into() };
                    let b_i32 = b_i32 - b_zero_point[col];
                    tmp[row][col] += a_i32 * b_i32;
                }
            }
        }

        if dest_beta == 0 {
            for row in 0..used_rows {
                for col in 0..used_cols {
                    dest_ptr
                        .add(row * dest_row_stride + col)
                        .write(tmp[row][col]);
                }
            }
        } else {
            // nb. We require that beta is 0 or 1, so here it is 1.
            for row in 0..used_rows {
                for col in 0..used_cols {
                    *dest_ptr.add(row * dest_row_stride + col) += tmp[row][col];
                }
            }
        }

        if use_tmp_tile {
            tmp_tile.accumulate_into(
                tile_ptr as *mut MaybeUninit<i32>,
                used_rows,
                used_cols,
                tile_row_stride,
                beta,
            );
        }
    }

    fn gemv_kernel(
        &self,
        out: &mut [MaybeUninit<i32>],
        a: &[LhsT],
        b: Matrix<RhsT>,
        alpha: f32,
        beta: i32,
        a_quant: Option<QuantParams<LhsT>>,
        b_quant: Option<QuantParams<RhsT>>,
    ) {
        int8_gemv(out, a, b, alpha, beta, a_quant, b_quant)
    }
}

/// Generic implementation of [`Kernel::gemv`] for u8 x i8 -> i32 kernels.
pub fn int8_gemv<LhsT: Copy + Into<i32>, RhsT: Copy + Into<i32>>(
    out: &mut [MaybeUninit<i32>],
    a: &[LhsT],
    b: Matrix<RhsT>,
    alpha: f32,
    beta: i32,
    a_quant: Option<QuantParams<LhsT>>,
    b_quant: Option<QuantParams<RhsT>>,
) {
    assert!(beta == 0 || beta == 1);
    assert_eq!(alpha, 1.);
    assert_eq!(b.rows(), a.len());
    assert_eq!(out.len(), b.cols());

    let a_zero = a_quant.map(|aq| aq.zero_point[0].into()).unwrap_or(0);
    let depth = a.len();

    for (out, col) in out.iter_mut().zip(0..b.cols()) {
        let b_zero = b_quant.map(|bq| bq.zero_point[col].into()).unwrap_or(0);
        let mut acc = 0;
        let mut row_sum = 0;
        let mut col_sum = 0;

        for k in 0..depth {
            let a_el: i32 = unsafe { (*a.get_unchecked(k)).into() };
            let b_el: i32 = unsafe { (*b.get_unchecked([k, col])).into() };
            acc += a_el * b_el;
            row_sum += a_el;
            col_sum += b_el;
        }

        // Subtract zero points. This is equivalent to doing
        // `acc += (a - a_zero) * (b - b_zero)` in the loop over K, but more
        // efficient.
        acc = depth as i32 * a_zero * b_zero + acc - row_sum * b_zero - col_sum * a_zero;

        if beta == 0 {
            out.write(acc);
        } else {
            // Safety: Output is initialized when beta is non-zero
            unsafe {
                out.write(out.assume_init() + acc);
            }
        }
    }
}
