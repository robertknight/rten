use std::arch::aarch64::{int32x4_t, int8x16_t};
use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::isa::ArmNeonIsa;
use rten_tensor::{Matrix, MatrixLayout};

use super::simd_generic::{simd_gemv, simd_int8_gemm, simd_int8_gemv, GemmDispatch};
use super::{
    extract_zero_points, Int8DotProduct, Kernel, Lhs, MatVecOutput, PackedLayout, QuantParams,
    TempTile,
};
use crate::gemm::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::gemm::{packing, Im2Col};
use crate::slice_cast::{cast_pod_mut_slice, cast_pod_slice};

pub struct ArmNeonKernel {
    isa: ArmNeonIsa,
}

impl ArmNeonKernel {
    const MR: usize = 4;
    const NR: usize = 16;
}

/// Number of 32-bit lanes in an Arm Neon SIMD vector.
const X32_LANES: usize = 4;

// Safety - We assume that Rust code on Arm is always compiled with Arm Neon
// available.
unsafe impl Kernel<f32, f32, f32> for ArmNeonKernel {
    fn new() -> Option<Self> {
        ArmNeonIsa::new().map(|isa| ArmNeonKernel { isa })
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
        let out = cast_pod_mut_slice(out).expect("incorrect alignment for packing buffer");
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
        let out = cast_pod_mut_slice(out).expect("incorrect alignment for packing buffer");
        pack_b_block::<f32, { Self::NR }>(out, b, rows, cols);
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<f32>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        const NR_REGS: usize = ArmNeonKernel::NR / X32_LANES;

        // Safety: Arm Neon instructions are supported
        let out = cast_pod_mut_slice(out).unwrap();
        image.pack_block::<_, NR_REGS>(self.isa, out, Self::NR, rows, cols);
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
        const MR: usize = ArmNeonKernel::MR;
        const NR: usize = ArmNeonKernel::NR;
        const NR_REGS: usize = NR / X32_LANES;

        let b = cast_pod_slice(b).unwrap();

        let mut tmp_tile = TempTile::<f32, MR, NR>::new();
        let (dest_ptr, dest_row_stride, dest_beta) = if used_cols == NR {
            (tile_ptr, tile_row_stride, beta)
        } else {
            (tmp_tile.as_mut_ptr() as *mut f32, NR, 0.)
        };

        let gemm = GemmDispatch::<_, MR, NR_REGS>::new(
            self.isa,
            dest_ptr,
            dest_row_stride,
            a,
            b,
            depth,
            alpha,
            dest_beta,
        );

        debug_assert_eq!(MR, 4);
        match used_rows {
            4 => gemm.dispatch_broadcast_lane::<4>(),
            3 => gemm.dispatch_broadcast_lane::<3>(),
            2 => gemm.dispatch_broadcast_lane::<2>(),
            1 => gemm.dispatch_broadcast_lane::<1>(),
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
        out: MatVecOutput<f32>,
        a: &[f32],
        b: Matrix,
        alpha: f32,
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        simd_gemv::<_, 4>(self.isa, out, a, b, alpha);
    }
}

macro_rules! impl_arm_int8_common {
    ($self_type:ty) => {
        fn mr(&self) -> usize {
            Self::MR
        }

        fn nr(&self) -> usize {
            Self::NR
        }

        fn im2col_row_count_step(&self) -> usize {
            4
        }

        fn packed_a_layout(
            &self,
            _a: Matrix<u8>,
            rows: usize,
            cols: usize,
            _quant: Option<QuantParams<u8>>,
        ) -> PackedLayout {
            let mut layout = packing::int8::packed_a_layout::<{ Self::MR }>(rows, cols);
            layout.must_pack = true;
            layout
        }

        fn pack_a_block(
            &self,
            out: &mut [MaybeUninit<u8>],
            a: Matrix<u8>,
            rows: Range<usize>,
            cols: Range<usize>,
            _quant: Option<QuantParams<u8>>,
        ) {
            let out = cast_pod_mut_slice(out).unwrap();
            packing::int8::pack_a::<{ Self::MR }>(out, a.slice((rows, cols)))
        }

        fn packed_b_layout(
            &self,
            rows: usize,
            cols: usize,
            _quant: Option<QuantParams<i8>>,
        ) -> PackedLayout {
            packing::int8::packed_b_layout::<{ Self::NR }>(rows, cols)
        }

        fn pack_b_block(
            &self,
            out: &mut [MaybeUninit<u8>],
            b: Matrix<i8>,
            rows: Range<usize>,
            cols: Range<usize>,
            _quant: Option<QuantParams<i8>>,
        ) {
            packing::int8::pack_b_cast_i8_u8::<{ Self::NR }>(out, b.slice((rows, cols)))
        }

        fn pack_im2col(
            &self,
            out: &mut [MaybeUninit<u8>],
            image: &Im2Col<i8>,
            rows: Range<usize>,
            cols: Range<usize>,
        ) {
            // Safety: Arm Neon is supported
            const NR_REGS: usize = <$self_type>::NR / X32_LANES;
            image.pack_block_i8_dot_cast_u8::<_, NR_REGS>(self.isa, out, rows, cols)
        }
    };
}

/// 8-bit integer matrix multiplication kernel using Arm dot product
/// instructions.
pub struct ArmInt8DotKernel {
    isa: ArmNeonIsa,
    dot_isa: NeonNativeDotProd,
}

impl ArmInt8DotKernel {
    const MR: usize = 8;
    const NR: usize = 8;
}

unsafe impl Kernel<u8, i8, i32> for ArmInt8DotKernel {
    fn new() -> Option<Self> {
        let isa = ArmNeonIsa::new()?;
        let dot_isa = NeonNativeDotProd::new()?;
        Some(ArmInt8DotKernel { isa, dot_isa })
    }

    fn name(&self) -> &'static str {
        "arm-int8-udot"
    }

    impl_arm_int8_common!(ArmInt8DotKernel);

    #[target_feature(enable = "dotprod")]
    unsafe fn kernel(
        &self,
        tile_ptr: *mut i32,
        tile_row_stride: usize,
        a: Lhs<u8>,
        b: &[u8],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        _alpha: f32,
        beta: i32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        let a_data = match a {
            Lhs::Packed(data) => data,
            Lhs::Unpacked { .. } => panic!("lhs must be packed"),
        };
        let a_zero_points = extract_zero_points(a_quant, used_rows, |x| x);
        let b_zero_points = extract_zero_points(b_quant, used_cols, |zp| zp + I8_U8_SHIFT);
        let (a_data, a_row_sums) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
        let (b, b_col_sums) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

        const NR_REGS: usize = ArmInt8DotKernel::NR / X32_LANES;
        simd_int8_gemm::<_, { Self::MR }, { Self::NR }, NR_REGS>(
            self.isa,
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
            self.dot_isa,
        )
    }

    fn gemv_kernel(
        &self,
        mut out: MatVecOutput<i32>,
        a: &[u8],
        b: Matrix<i8>,
        _alpha: f32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        let a_zero = a_quant.map(|aq| aq.zero_point[0]).unwrap_or(0);
        let b_zero = b_quant.map(|bq| bq.zero_point);
        let out = out.as_bool_beta();
        simd_int8_gemv::<_, true /* CAST_B_U8 */>(self.isa, out, a, b, a_zero, b_zero, self.dot_isa)
    }
}

/// 8-bit integer matrix multiplication kernel for Arm CPUs which don't support
/// dot product instructions.
pub struct ArmInt8Kernel {
    isa: ArmNeonIsa,
}

impl ArmInt8Kernel {
    const MR: usize = 8;
    const NR: usize = 8;
}

unsafe impl Kernel<u8, i8, i32> for ArmInt8Kernel {
    fn new() -> Option<Self> {
        ArmNeonIsa::new().map(|isa| ArmInt8Kernel { isa })
    }

    fn name(&self) -> &'static str {
        "arm-int8"
    }

    impl_arm_int8_common!(ArmInt8Kernel);

    unsafe fn kernel(
        &self,
        tile_ptr: *mut i32,
        tile_row_stride: usize,
        a: Lhs<u8>,
        b: &[u8],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        _alpha: f32,
        beta: i32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        let a_data = match a {
            Lhs::Packed(data) => data,
            Lhs::Unpacked { .. } => panic!("lhs must be packed"),
        };

        let a_zero_points = extract_zero_points(a_quant, used_rows, |x| x);
        let b_zero_points = extract_zero_points(b_quant, used_cols, |zp| zp + I8_U8_SHIFT);
        let (a_data, a_row_sums) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
        let (b, b_col_sums) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

        const NR_REGS: usize = ArmInt8Kernel::NR / X32_LANES;
        simd_int8_gemm::<_, { Self::MR }, { Self::NR }, NR_REGS>(
            self.isa,
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
            NeonDotProd {},
        )
    }

    fn gemv_kernel(
        &self,
        mut out: MatVecOutput<i32>,
        a: &[u8],
        b: Matrix<i8>,
        _alpha: f32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        let a_zero = a_quant.map(|aq| aq.zero_point[0]).unwrap_or(0);
        let b_zero = b_quant.map(|bq| bq.zero_point);
        let out = out.as_bool_beta();
        simd_int8_gemv::<_, true /* CAST_B_U8 */>(
            self.isa,
            out,
            a,
            b,
            a_zero,
            b_zero,
            NeonDotProd {},
        )
    }
}

/// Adjustment to apply to zero points in kernel where corresponding input
/// was shifted from i8 to u8 when packing.
const I8_U8_SHIFT: i32 = 128;

/// Implementation of int8 dot product using the native UDOT / SDOT instructions.
#[derive(Copy, Clone)]
struct NeonNativeDotProd {
    _private: (),
}

impl NeonNativeDotProd {
    fn new() -> Option<Self> {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return None;
        }
        Some(NeonNativeDotProd { _private: () })
    }
}

// Safety: Constructor checked for "dotprod" feature support.
unsafe impl Int8DotProduct for NeonNativeDotProd {
    type X8 = int8x16_t;
    type I32 = int32x4_t;

    #[inline]
    fn dot_product(self, a: Self::X8, b: Self::X8, c: Self::I32) -> Self::I32 {
        #[target_feature(enable = "dotprod")]
        #[inline]
        unsafe fn dot_product(a: int8x16_t, b: int8x16_t, c: int32x4_t) -> int32x4_t {
            use core::arch::aarch64::{
                vreinterpretq_s32_u32, vreinterpretq_u32_s32, vreinterpretq_u8_s8,
            };
            use core::arch::asm;

            let a = vreinterpretq_u8_s8(a);
            let b = vreinterpretq_u8_s8(b);
            let mut c = vreinterpretq_u32_s32(c);

            // Use inline asm here because the `vdotq_u32` intrinsic is not
            // stabilized yet.
            asm! {
                "udot {result:v}.4s, {a:v}.16b, {b:v}.16b",
                result = inout(vreg) c,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack)
            }

            vreinterpretq_s32_u32(c)
        }
        unsafe { dot_product(a, b, c) }
    }
}

/// Fallback implementation of u8 dot product for older Arm CPUs which don't
/// support dot product instructions (UDOT, SDOT).
#[derive(Copy, Clone)]
struct NeonDotProd;

// Safety: Neon instructions used are always supported on aarch64.
unsafe impl Int8DotProduct for NeonDotProd {
    type X8 = int8x16_t;
    type I32 = int32x4_t;

    #[inline]
    fn dot_product(self, a: Self::X8, b: Self::X8, c: Self::I32) -> Self::I32 {
        unsafe {
            use core::arch::aarch64::{
                vaddq_u32, vget_low_u8, vmull_high_u8, vmull_u8, vpaddlq_u16, vpaddq_u32,
                vreinterpretq_s32_u32, vreinterpretq_u32_s32, vreinterpretq_u8_s8,
            };

            let a = vreinterpretq_u8_s8(a);
            let b = vreinterpretq_u8_s8(b);
            let c = vreinterpretq_u32_s32(c);

            let mul_lo = vmull_u8(vget_low_u8(a), vget_low_u8(b));
            let mul_hi = vmull_high_u8(a, b);
            let tmp_lo = vpaddlq_u16(mul_lo);
            let tmp_hi = vpaddlq_u16(mul_hi);
            let dot_prod = vpaddq_u32(tmp_lo, tmp_hi);
            vreinterpretq_s32_u32(vaddq_u32(dot_prod, c))
        }
    }
}
