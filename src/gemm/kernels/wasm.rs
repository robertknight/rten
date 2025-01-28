use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::arch::wasm::{v128f, v128i};
use rten_simd::vec_count;
use rten_tensor::{Matrix, MatrixLayout};

use super::simd_generic::{
    simd_gemv, simd_int8_gemm, simd_int8_gemv, GemmDispatch, CAST_I8_TO_U8, CAST_SAME,
};
use super::{extract_zero_points, Kernel, Lhs, PackedLayout, QuantParams, TempTile};
use crate::gemm::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::gemm::{packing, Im2Col};
use crate::slice_cast::{cast_pod_mut_slice, cast_pod_slice};

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

pub struct WasmInt8Kernel {
    _private: (),
}

impl WasmInt8Kernel {
    const MR: usize = 8;
    const NR: usize = 4;
}

macro_rules! impl_wasm_int8_kernel {
    ($lhs:ty, $rhs:ty, $lhs_cast:ident, $rhs_cast:ident) => {
        unsafe impl Kernel<$lhs, $rhs, i32> for WasmInt8Kernel {
            fn new() -> Option<Self> {
                Some(WasmInt8Kernel { _private: () })
            }

            fn name(&self) -> &'static str {
                "wasm-int8"
            }

            fn mr(&self) -> usize {
                Self::MR
            }

            fn nr(&self) -> usize {
                Self::NR
            }

            fn packed_a_layout(
                &self,
                _a: Matrix<$lhs>,
                rows: usize,
                cols: usize,
                _quant: Option<QuantParams<$lhs>>,
            ) -> PackedLayout {
                let mut layout = packing::int8::packed_a_layout::<{ Self::MR }>(rows, cols);
                layout.must_pack = true;
                layout
            }

            fn pack_a_block(
                &self,
                out: &mut [MaybeUninit<u8>],
                a: Matrix<$lhs>,
                rows: Range<usize>,
                cols: Range<usize>,
                _quant: Option<QuantParams<$lhs>>,
            ) {
                let out = cast_pod_mut_slice(out).unwrap();
                packing::int8::pack_a::<{ Self::MR }, $lhs, u8>(out, a.slice((rows, cols)))
            }

            fn packed_b_layout(
                &self,
                rows: usize,
                cols: usize,
                _quant: Option<QuantParams<$rhs>>,
            ) -> PackedLayout {
                packing::int8::packed_b_layout::<{ Self::NR }>(rows, cols)
            }

            fn pack_b_block(
                &self,
                out: &mut [MaybeUninit<u8>],
                b: Matrix<$rhs>,
                rows: Range<usize>,
                cols: Range<usize>,
                _quant: Option<QuantParams<$rhs>>,
            ) {
                let out = cast_pod_mut_slice(out).unwrap();
                packing::int8::pack_b::<{ Self::NR }, $rhs, u8>(out, b.slice((rows, cols)))
            }

            fn pack_im2col(
                &self,
                _out: &mut [MaybeUninit<u8>],
                _image: &Im2Col<$rhs>,
                _rows: Range<usize>,
                _cols: Range<usize>,
            ) {
                unimplemented!("pack_im2col not implemented");
            }

            unsafe fn kernel(
                &self,
                tile_ptr: *mut i32,
                tile_row_stride: usize,
                a: Lhs<$lhs>,
                b: &[u8],
                used_rows: usize,
                used_cols: usize,
                depth: usize,
                _alpha: f32,
                beta: i32,
                a_quant: Option<QuantParams<$lhs>>,
                b_quant: Option<QuantParams<$rhs>>,
            ) {
                let a_data = match a {
                    Lhs::Packed(data) => data,
                    Lhs::Unpacked { .. } => panic!("lhs must be packed"),
                };

                let a_zero_points =
                    extract_zero_points::<$lhs, u8, { Self::MR }>(a_quant, used_rows);
                let b_zero_points =
                    extract_zero_points::<$rhs, u8, { Self::NR }>(b_quant, used_cols);
                let (a_data, a_row_sums) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
                let (b, b_col_sums) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

                simd_int8_gemm::<_, { Self::MR }, { Self::NR }>(
                    tile_ptr,
                    tile_row_stride,
                    a_data,
                    b,
                    used_rows,
                    used_cols,
                    depth,
                    beta != 0, // accumulate
                    a_zero_points,
                    b_zero_points,
                    a_row_sums,
                    b_col_sums,
                    wasm_u8u8_i32_dot_product,
                )
            }

            fn gemv_kernel(
                &self,
                out: &mut [MaybeUninit<i32>],
                a: &[$lhs],
                b: Matrix<$rhs>,
                _alpha: f32,
                beta: i32,
                a_quant: Option<QuantParams<$lhs>>,
                b_quant: Option<QuantParams<$rhs>>,
            ) {
                let a_zero = a_quant.map(|aq| aq.zero_point[0]).unwrap_or(0);
                let b_zero = b_quant.map(|bq| bq.zero_point);
                let accumulate = beta != 0;

                // Safety: Target features were checked when kernel was constructed.
                unsafe {
                    simd_int8_gemv::<_, $lhs, $rhs, $lhs_cast, $rhs_cast>(
                        out,
                        a,
                        b,
                        accumulate,
                        a_zero,
                        b_zero,
                        wasm_u8u8_i32_dot_product,
                    )
                }
            }
        }
    };
}

impl_wasm_int8_kernel!(u8, i8, CAST_SAME, CAST_I8_TO_U8);
impl_wasm_int8_kernel!(u8, u8, CAST_SAME, CAST_SAME);
impl_wasm_int8_kernel!(i8, i8, CAST_I8_TO_U8, CAST_I8_TO_U8);
impl_wasm_int8_kernel!(i8, u8, CAST_I8_TO_U8, CAST_SAME);

/// Compute i32 dot product of each group of 4 u8 integers in `a` and `b` and
/// add to i32x4 accumulator in `c`.
///
/// Adapted from the reference lowing of `i32x4.dot_i8x16_i7x16_add_s` given
/// in https://github.com/WebAssembly/relaxed-simd/issues/52.
#[inline]
fn wasm_u8u8_i32_dot_product(a: v128i, b: v128i, c: v128i) -> v128i {
    use std::arch::wasm32::{
        i32x4_add, i32x4_extadd_pairwise_u16x8, i32x4_shuffle, u16x8_extmul_high_u8x16,
        u16x8_extmul_low_u8x16,
    };

    let mul_lo = u16x8_extmul_low_u8x16(a.0, b.0);
    let mul_hi = u16x8_extmul_high_u8x16(a.0, b.0);

    let pair_sum_lo = i32x4_extadd_pairwise_u16x8(mul_lo);
    let pair_sum_hi = i32x4_extadd_pairwise_u16x8(mul_hi);

    let pair_sum_even = i32x4_shuffle::<0, 2, 4, 6>(pair_sum_lo, pair_sum_hi);
    let pair_sum_odd = i32x4_shuffle::<1, 3, 5, 7>(pair_sum_lo, pair_sum_hi);

    let quad_sum = i32x4_add(pair_sum_even, pair_sum_odd);
    v128i(i32x4_add(quad_sum, c.0))
}
