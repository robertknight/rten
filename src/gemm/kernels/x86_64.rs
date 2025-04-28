use std::mem::MaybeUninit;
use std::ops::Range;

use rten_simd::{isa::Avx2Isa, Isa};
use rten_tensor::{Matrix, MatrixLayout};

#[cfg(feature = "avx512")]
use rten_simd::isa::Avx512Isa;

use super::simd_generic::{simd_gemv, simd_int8_gemm, simd_int8_gemv, GemmDispatch};
use super::{
    extract_zero_points, Int8DotProduct, Kernel, Lhs, MatVecOutput, PackedLayout, QuantParams,
    TempTile,
};
use crate::gemm::packing;
use crate::gemm::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::gemm::Im2Col;
use crate::slice_cast::{cast_pod_mut_slice, cast_pod_slice};

/// Optimized kernel for x64 CPUs that support AVX + FMA instructions.
pub struct FmaKernel {
    isa: Avx2Isa,
}

impl FmaKernel {
    const MR: usize = 6;

    // Chosen to fit 2 AVX registers and take advantage of the two FMA
    // execution ports.
    const NR: usize = 16;
}

/// Number of 32-bit lanes in an AVX2 SIMD vector.
const AVX2_X32_LANES: usize = 8;

/// Number of 32-bit lanes in an AVX-512 SIMD vector.
#[cfg(feature = "avx512")]
const AVX512_X32_LANES: usize = 16;

/// Wrapper for `pack_a_block` which enables AVX instructions.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn pack_a_block_avx<const MR: usize>(
    out: &mut [MaybeUninit<f32>],
    a: Matrix,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    pack_a_block::<f32, MR>(out, a, rows, cols);
}

/// Wrapper for `pack_b_block` which enables AVX instructions.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn pack_b_block_avx<const NR: usize>(
    out: &mut [MaybeUninit<f32>],
    b: Matrix,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    pack_b_block::<f32, NR>(out, b, rows, cols);
}

// Safety - The `new` fn tests for AVX-2 / FMA support.
unsafe impl Kernel<f32, f32, f32> for FmaKernel {
    fn new() -> Option<Self> {
        Avx2Isa::new().map(|isa| FmaKernel { isa })
    }

    fn name(&self) -> &'static str {
        "fma"
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

        // Safety: Kernel can only be constructed if AVX is supported.
        unsafe {
            pack_a_block_avx::<{ Self::MR }>(out, a, rows, cols);
        }
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

        // Safety: Kernel can only be constructed if AVX is supported.
        unsafe {
            pack_b_block_avx::<{ Self::NR }>(out, b, rows, cols);
        }
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<f32>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        const NR_REGS: usize = FmaKernel::NR / AVX2_X32_LANES;

        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        unsafe fn pack_im2col_avx<const NR_REGS: usize, const NR: usize>(
            isa: Avx2Isa,
            out: &mut [MaybeUninit<f32>],
            image: &Im2Col<f32>,
            rows: Range<usize>,
            cols: Range<usize>,
        ) {
            image.pack_block::<_, NR_REGS>(isa, out, NR, rows, cols);
        }

        // Safety: Kernel can only be constructed if AVX is supported
        let out = cast_pod_mut_slice(out).unwrap();
        unsafe {
            pack_im2col_avx::<NR_REGS, { Self::NR }>(self.isa, out, image, rows, cols);
        }
    }

    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
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
        const MR: usize = FmaKernel::MR;
        const NR: usize = FmaKernel::NR;
        const NR_REGS: usize = NR / AVX2_X32_LANES;

        let b = cast_pod_slice(b).unwrap();

        // TODO - Replace temporary tile with masked loads and stores.
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

        match used_rows {
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
        out: MatVecOutput<f32>,
        a: &[f32],
        b: Matrix,
        alpha: f32,
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        unsafe fn gemv_kernel_impl(
            isa: Avx2Isa,
            out: MatVecOutput<f32>,
            a: &[f32],
            b: Matrix,
            alpha: f32,
        ) {
            simd_gemv::<_, 4>(isa, out, a, b, alpha);
        }
        // Safety: Kernel can only be constructed if supported.
        unsafe {
            gemv_kernel_impl(self.isa, out, a, b, alpha);
        }
    }
}

/// Optimized kernel for x64 CPUs that support AVX 512 instructions.
#[cfg(feature = "avx512")]
pub struct Avx512Kernel {
    isa: Avx512Isa,
}

#[cfg(feature = "avx512")]
impl Avx512Kernel {
    // The optimal value of MR depends on how many AVX-512 FMA units the CPU has.
    // Client Intel CPUs have one, server CPUs have two. This smaller value is
    // tuned for single-FMA CPUs.
    //
    // See https://github.com/robertknight/rten/issues/17.
    const MR: usize = 6;

    // 2 x 16-f32-wide registers.
    const NR: usize = 32;
}

// Safety - The `new` fn checks for AVX-512 support.
#[cfg(feature = "avx512")]
unsafe impl Kernel<f32, f32, f32> for Avx512Kernel {
    fn new() -> Option<Self> {
        Avx512Isa::new().map(|isa| Avx512Kernel { isa })
    }

    fn name(&self) -> &'static str {
        "avx512"
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

        // Safety: AVX-512 implies availability of AVX 2.
        unsafe {
            pack_a_block_avx::<{ Self::MR }>(out, a, rows, cols);
        }
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

        // Safety: We assume AVX-512 implies availability of AVX 2.
        unsafe {
            pack_b_block_avx::<{ Self::NR }>(out, b, rows, cols);
        }
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<f32>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        #[cfg(feature = "avx512")]
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn pack_im2col_avx512<const NR_REGS: usize, const NR: usize>(
            isa: Avx512Isa,
            out: &mut [MaybeUninit<f32>],
            image: &Im2Col<f32>,
            rows: Range<usize>,
            cols: Range<usize>,
        ) {
            image.pack_block::<_, NR_REGS>(isa, out, NR, rows, cols);
        }

        const NR_REGS: usize = Avx512Kernel::NR / AVX512_X32_LANES;

        // Safety: Kernel can only be constructed if AVX-512 is supported.
        let out = cast_pod_mut_slice(out).unwrap();
        unsafe {
            pack_im2col_avx512::<NR_REGS, { Self::NR }>(self.isa, out, image, rows, cols);
        }
    }

    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
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
        const MR: usize = Avx512Kernel::MR;
        const NR: usize = Avx512Kernel::NR;
        const NR_REGS: usize = NR / AVX512_X32_LANES;

        let b = cast_pod_slice(b).unwrap();

        // TODO - Replace temporary tile with masked loads and stores.
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

        match used_rows {
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
        out: MatVecOutput<f32>,
        a: &[f32],
        b: Matrix,
        alpha: f32,
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn gemv_kernel_impl(
            isa: Avx512Isa,
            out: MatVecOutput<f32>,
            a: &[f32],
            b: Matrix,
            alpha: f32,
        ) {
            simd_gemv::<_, 2>(isa, out, a, b, alpha);
        }
        // Safety: Kernel can only be constructed if supported.
        unsafe {
            gemv_kernel_impl(self.isa, out, a, b, alpha);
        }
    }
}

pub struct Avx2Int8Kernel {
    isa: Avx2Isa,
}

impl Avx2Int8Kernel {
    const MR: usize = 6;
    const NR: usize = 16;
}

unsafe impl Kernel<u8, i8, i32> for Avx2Int8Kernel {
    fn new() -> Option<Self> {
        Avx2Isa::new().map(|isa| Avx2Int8Kernel { isa })
    }

    fn name(&self) -> &'static str {
        "avx2-int8"
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn may_saturate(&self) -> bool {
        true
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
        let out = cast_pod_mut_slice(out).unwrap();
        packing::int8::pack_b::<{ Self::NR }>(out, b.slice((rows, cols)))
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<i8>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        #[target_feature(enable = "avx2")]
        unsafe fn pack_im2col_avx(
            isa: Avx2Isa,
            out: &mut [MaybeUninit<u8>],
            image: &Im2Col<i8>,
            rows: Range<usize>,
            cols: Range<usize>,
        ) {
            const NR_REGS: usize = Avx2Int8Kernel::NR / AVX2_X32_LANES;

            let out = cast_pod_mut_slice(out).unwrap();
            image.pack_block_i8_dot::<_, NR_REGS>(isa, out, rows, cols);
        }

        unsafe {
            pack_im2col_avx(self.isa, out, image, rows, cols);
        }
    }

    #[target_feature(enable = "avx2")]
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
        let b_zero_points = extract_zero_points(b_quant, used_cols, |x| x);
        let (a_data, a_row_sums) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
        let (b, b_col_sums) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

        const NR_REGS: usize = Avx2Int8Kernel::NR / AVX2_X32_LANES;
        simd_int8_gemm::<_, _, { Self::MR }, { Self::NR }, NR_REGS>(
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
            self.isa,
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

        #[target_feature(enable = "avx2")]
        unsafe fn gemv_impl(
            isa: Avx2Isa,
            out: MatVecOutput<i32, bool>,
            a: &[u8],
            b: Matrix<i8>,
            a_zero: u8,
            b_zero: Option<&[i8]>,
        ) {
            simd_int8_gemv::<_, false /* CAST_B_U8 */>(isa, out, a, b, a_zero, b_zero, isa)
        }

        // Safety: AVX2 is supported if this kernel was constructed.
        unsafe { gemv_impl(self.isa, out, a, b, a_zero, b_zero) }
    }
}

type I8x32 = <Avx2Isa as Isa>::I8;
type I32x8 = <Avx2Isa as Isa>::I32;

/// `u8 x i8 -> i32` dot product for AVX2.
///
/// Safety: Avx2Isa can only be constructed if AVX2 is supported.
unsafe impl Int8DotProduct for Avx2Isa {
    type X8 = I8x32;
    type I32 = I32x8;

    /// Compute 8x dot products between `u8` values in `a`, `i8` values in `b` and
    /// add the `i32` results to `c`.
    #[inline]
    fn dot_product(self, a: Self::X8, b: Self::X8, c: Self::I32) -> Self::I32 {
        use core::arch::x86_64::{
            _mm256_add_epi32, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
        };

        unsafe {
            let tmp = _mm256_maddubs_epi16(a.0, b.0);
            let tmp = _mm256_madd_epi16(tmp, _mm256_set1_epi16(1));
            _mm256_add_epi32(c.0, tmp).into()
        }
    }
}

#[cfg(feature = "avx512")]
pub struct Avx512Int8Kernel {
    isa: Avx512Isa,
    /// VNNI ("DL Boost") int8 dot product, if supported.
    vnni_dot: Option<Avx512VnniDotProduct>,
}

#[cfg(feature = "avx512")]
impl Avx512Int8Kernel {
    const MR: usize = 8;
    const NR: usize = 32;
}

#[cfg(feature = "avx512")]
unsafe impl Kernel<u8, i8, i32> for Avx512Int8Kernel {
    fn new() -> Option<Self> {
        let isa = Avx512Isa::new()?;
        let vnni_dot = Avx512VnniDotProduct::new();
        Some(Avx512Int8Kernel { isa, vnni_dot })
    }

    fn name(&self) -> &'static str {
        "avx512-int8"
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn may_saturate(&self) -> bool {
        self.vnni_dot.is_none()
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
        let out = cast_pod_mut_slice(out).unwrap();
        packing::int8::pack_b::<{ Self::NR }>(out, b.slice((rows, cols)))
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<i8>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512bw")]
        unsafe fn pack_im2col_avx512(
            isa: Avx512Isa,
            out: &mut [MaybeUninit<u8>],
            image: &Im2Col<i8>,
            rows: Range<usize>,
            cols: Range<usize>,
        ) {
            const NR_REGS: usize = Avx512Int8Kernel::NR / AVX512_X32_LANES;

            let out = cast_pod_mut_slice(out).unwrap();
            image.pack_block_i8_dot::<_, NR_REGS>(isa, out, rows, cols);
        }

        unsafe {
            pack_im2col_avx512(self.isa, out, image, rows, cols);
        }
    }

    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
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
        let b_zero_points = extract_zero_points(b_quant, used_cols, |x| x);
        let (a_data, a_row_sums) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
        let (b, b_col_sums) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

        const NR_REGS: usize = Avx512Int8Kernel::NR / AVX512_X32_LANES;
        if let Some(vnni_dot) = self.vnni_dot {
            simd_int8_gemm::<_, _, { Self::MR }, { Self::NR }, NR_REGS>(
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
                vnni_dot,
            )
        } else {
            simd_int8_gemm::<_, _, { Self::MR }, { Self::NR }, NR_REGS>(
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
                self.isa, // Use non-VNNI dot product
            )
        }
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

        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512vl")]
        #[target_feature(enable = "avx512bw")]
        unsafe fn gemv_impl(
            isa: Avx512Isa,
            out: MatVecOutput<i32, bool>,
            a: &[u8],
            b: Matrix<i8>,
            a_zero: u8,
            b_zero: Option<&[i8]>,
        ) {
            simd_int8_gemv::<_, false /* CAST_B_U8 */>(
                isa, out, a, b, a_zero, b_zero, // TODO - Use VNNI here if available
                isa,
            )
        }

        // Safety: AVX512 is supported if this kernel was constructed.
        unsafe {
            gemv_impl(self.isa, out, a, b, a_zero, b_zero);
        }
    }
}

#[cfg(feature = "avx512")]
type I8x64 = <Avx512Isa as Isa>::I8;
#[cfg(feature = "avx512")]
type I32x16 = <Avx512Isa as Isa>::I32;

// Safety: Avx512Isa can only be constructed if AVX-512 is supported.
#[cfg(feature = "avx512")]
unsafe impl Int8DotProduct for Avx512Isa {
    type X8 = I8x64;
    type I32 = I32x16;

    /// Compute 16 dot products between `u8` values in `a`, `i8` values in `b` and
    /// add the `i32` results to `c`.
    #[inline]
    fn dot_product(self, a: I8x64, b: I8x64, c: I32x16) -> I32x16 {
        use core::arch::x86_64::{
            _mm512_add_epi32, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_set1_epi16,
        };

        unsafe {
            let tmp = _mm512_maddubs_epi16(a.0, b.0);
            let tmp = _mm512_madd_epi16(tmp, _mm512_set1_epi16(1));
            _mm512_add_epi32(c.0, tmp).into()
        }
    }
}

/// Eight-bit integer dot product using AVX512 VNNI instructions.
#[cfg(feature = "avx512")]
#[derive(Copy, Clone)]
struct Avx512VnniDotProduct {
    _private: (),
}

#[cfg(feature = "avx512")]
impl Avx512VnniDotProduct {
    pub fn new() -> Option<Self> {
        detect_avx512_vnni().then_some(Self { _private: () })
    }
}

// Safety: Avx512VnniDotProduct can only be constructed if AVX512-VNNI is
// supported.
#[cfg(feature = "avx512")]
unsafe impl Int8DotProduct for Avx512VnniDotProduct {
    type X8 = I8x64;
    type I32 = I32x16;

    /// Compute 16 dot products between `u8` values in `a`, `i8` values in `b` and
    /// add the `i32` results to `c`.
    ///
    /// This uses AVX-512 VNNI instructions for better performance and to avoid
    /// saturation issue that `VPMADDUBSW` has. See
    /// https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html.
    #[inline]
    fn dot_product(self, a: I8x64, b: I8x64, c: I32x16) -> I32x16 {
        unsafe { avx512_vnni_u8i8i32_dot_product(a, b, c) }
    }
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn avx512_vnni_u8i8i32_dot_product(a: I8x64, b: I8x64, mut c: I32x16) -> I32x16 {
    // Use inline asm rather than an intrinsic here to avoid needing to mark
    // this function as using the `avx512vnni` feature. If we did that, the
    // entire kernel function needs to have the same target feature statically
    // enabled in order for this function to be inlined into it, which is
    // critical for performance. This in turn means we would need to have
    // separate instantiations of the kernel for the VNNI and non-VNNI cases.
    //
    // By using asm we can use this instruction after a dynamic flag check
    // within the kernel.
    use std::arch::asm;
    asm! {
        "vpdpbusd {result}, {a}, {b}",
        result = inout(zmm_reg) c.0,
        a = in(zmm_reg) a.0,
        b = in(zmm_reg) b.0,
        options(nostack)
    }
    c.into()
}

/// Detect availability of AVX-512 VNNI instructions using cpuid.
///
/// This function only returns valid results if AVX-512 is supported.
///
/// See https://www.felixcloutier.com/x86/cpuid or the Intel Instruction Set
/// Reference for cpuid.
///
/// This function differs from `is_x86_feature_detected("avx512vnni")` as that
/// function can incorrectly return false on macOS if feature detection was
/// performed before an AVX512 instruction was used in a process. See
/// notes in [`is_avx512_supported`].
#[cfg(feature = "avx512")]
fn detect_avx512_vnni() -> bool {
    use core::arch::x86_64::__cpuid_count;
    let regs = unsafe { __cpuid_count(7, 0) };
    regs.ecx & (1 << 11) != 0
}

#[cfg(feature = "avx512")]
#[cfg(test)]
mod tests {
    use super::detect_avx512_vnni;

    #[test]
    fn test_vnni_detect() {
        // `detect_avx512_vnni` may return true in cases where
        // `is_x86_feature_detected` returns false, but the converse is not
        // true.
        let have_vnni = detect_avx512_vnni();
        if is_x86_feature_detected!("avx512vnni") {
            assert!(have_vnni);
        }
    }
}
