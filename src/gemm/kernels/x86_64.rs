use std::arch::x86_64::{__m256, __m256i};
use std::mem::MaybeUninit;
use std::ops::Range;

#[cfg(feature = "avx512")]
use std::arch::x86_64::{__m512, __m512i};

use rten_simd::vec_count;
use rten_tensor::{Matrix, MatrixLayout};

#[cfg(feature = "avx512")]
use rten_simd::isa_detection::is_avx512_supported;

use super::simd_generic::{simd_gemv, simd_int8_gemm, GemmDispatch};
use super::{Kernel, Lhs, PackedLayout, QuantParams, TempTile};
use crate::gemm::packing;
use crate::gemm::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::gemm::Im2Col;
use crate::slice_cast::{cast_pod_mut_slice, cast_pod_slice};

/// Optimized kernel for x64 CPUs that support AVX + FMA instructions.
pub struct FmaKernel {
    _private: (),
}

impl FmaKernel {
    const MR: usize = 6;

    // Chosen to fit 2 AVX registers and take advantage of the two FMA
    // execution ports.
    const NR: usize = 16;
}

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

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn pack_im2col_avx<const NR_REGS: usize, const NR: usize>(
    out: &mut [MaybeUninit<f32>],
    image: &Im2Col<f32>,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    image.pack_block::<__m256i, NR_REGS>(out, NR, rows, cols);
}

// Safety - The `new` fn tests for AVX-2 / FMA support.
unsafe impl Kernel<f32, f32, f32> for FmaKernel {
    fn new() -> Option<Self> {
        let supported = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        supported.then_some(FmaKernel { _private: () })
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
        const NR_REGS: usize = vec_count::<__m256i>(FmaKernel::NR);

        // Safety: Kernel can only be constructed if AVX is supported
        let out = cast_pod_mut_slice(out).unwrap();
        unsafe {
            pack_im2col_avx::<NR_REGS, { Self::NR }>(out, image, rows, cols);
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
        const NR_REGS: usize = vec_count::<__m256>(NR);

        let b = cast_pod_slice(b).unwrap();

        // TODO - Replace temporary tile with masked loads and stores.
        let mut tmp_tile = TempTile::<f32, MR, NR>::new();
        let (dest_ptr, dest_row_stride, dest_beta) = if used_cols == NR {
            (tile_ptr, tile_row_stride, beta)
        } else {
            (tmp_tile.as_mut_ptr() as *mut f32, NR, 0.)
        };

        let gemm = GemmDispatch::<__m256, MR, NR_REGS>::new(
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
        out: &mut [MaybeUninit<f32>],
        a: &[f32],
        b: Matrix,
        alpha: f32,
        beta: f32,
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        unsafe fn gemv_kernel_impl(
            out: &mut [MaybeUninit<f32>],
            a: &[f32],
            b: Matrix,
            alpha: f32,
            beta: f32,
        ) {
            simd_gemv::<__m256, 4>(out, a, b, alpha, beta);
        }
        // Safety: Kernel can only be constructed if supported.
        unsafe {
            gemv_kernel_impl(out, a, b, alpha, beta);
        }
    }
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vl")]
unsafe fn pack_im2col_avx512<const NR_REGS: usize, const NR: usize>(
    out: &mut [MaybeUninit<f32>],
    image: &Im2Col<f32>,
    rows: Range<usize>,
    cols: Range<usize>,
) {
    image.pack_block::<__m512i, NR_REGS>(out, NR, rows, cols);
}

/// Optimized kernel for x64 CPUs that support AVX 512 instructions.
#[cfg(feature = "avx512")]
pub struct Avx512Kernel {
    _private: (),
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
        is_avx512_supported().then_some(Avx512Kernel { _private: () })
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
        const NR_REGS: usize = vec_count::<__m512i>(Avx512Kernel::NR);

        // Safety: Kernel can only be constructed if AVX-512 is supported.
        let out = cast_pod_mut_slice(out).unwrap();
        unsafe {
            pack_im2col_avx512::<NR_REGS, { Self::NR }>(out, image, rows, cols);
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
        const NR_REGS: usize = vec_count::<__m512>(NR);

        let b = cast_pod_slice(b).unwrap();

        // TODO - Replace temporary tile with masked loads and stores.
        let mut tmp_tile = TempTile::<f32, MR, NR>::new();
        let (dest_ptr, dest_row_stride, dest_beta) = if used_cols == NR {
            (tile_ptr, tile_row_stride, beta)
        } else {
            (tmp_tile.as_mut_ptr() as *mut f32, NR, 0.)
        };

        let gemm = GemmDispatch::<__m512, MR, NR_REGS>::new(
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
        out: &mut [MaybeUninit<f32>],
        a: &[f32],
        b: Matrix,
        alpha: f32,
        beta: f32,
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn gemv_kernel_impl(
            out: &mut [MaybeUninit<f32>],
            a: &[f32],
            b: Matrix,
            alpha: f32,
            beta: f32,
        ) {
            simd_gemv::<__m512, 2>(out, a, b, alpha, beta);
        }
        // Safety: Kernel can only be constructed if supported.
        unsafe {
            gemv_kernel_impl(out, a, b, alpha, beta);
        }
    }
}

pub struct Avx2Int8Kernel {
    _private: (),
}

impl Avx2Int8Kernel {
    // Tile size matches AVX2 register
    const MR: usize = 8;
    const NR: usize = 8;
}

unsafe impl Kernel<u8, i8, i32> for Avx2Int8Kernel {
    fn new() -> Option<Self> {
        let supported = is_x86_feature_detected!("avx2");
        supported.then_some(Avx2Int8Kernel { _private: () })
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
        _out: &mut [MaybeUninit<u8>],
        _image: &Im2Col<i8>,
        _rows: Range<usize>,
        _cols: Range<usize>,
    ) {
        unimplemented!("pack_im2col not implemented");
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

        let a_zero_points = extract_zero_points(a_quant, used_rows);
        let b_zero_points = extract_zero_points(b_quant, used_cols);
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
            avx2_int8_dot_product,
        )
    }

    fn gemv_kernel(
        &self,
        out: &mut [MaybeUninit<i32>],
        a: &[u8],
        b: Matrix<i8>,
        alpha: f32,
        beta: i32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        // TODO - Optimize with AVX intrinsics.
        assert!(beta == 0 || beta == 1);
        assert_eq!(alpha, 1.);
        assert_eq!(b.rows(), a.len());
        assert_eq!(out.len(), b.cols());

        let a_zero = a_quant.map(|aq| aq.zero_point[0] as i32).unwrap_or(0);
        let depth = a.len();

        for (out, col) in out.iter_mut().zip(0..b.cols()) {
            let b_zero = b_quant.map(|bq| bq.zero_point[col] as i32).unwrap_or(0);
            let mut acc = 0;
            for k in 0..depth {
                let a_el = unsafe { *a.get_unchecked(k) } as i32 - a_zero;
                let b_el = unsafe { *b.get_unchecked([k, col]) } as i32 - b_zero;
                acc += a_el * b_el;
            }
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
}

/// Compute 8x dot products between `u8` values in `a`, `i8` values in `b` and
/// add the `i32` results to `c`.
#[inline(always)]
unsafe fn avx2_int8_dot_product(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    use core::arch::x86_64::{
        _mm256_add_epi32, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
    };
    let tmp = _mm256_maddubs_epi16(a, b);
    let tmp = _mm256_madd_epi16(tmp, _mm256_set1_epi16(1));
    _mm256_add_epi32(c, tmp)
}

/// Extract `len` zero points from `quant`, upconvert to i32 and pad unused
/// elements in the result with zero.
fn extract_zero_points<T: Copy + Into<i32>, const MAX_LEN: usize>(
    quant: Option<QuantParams<T>>,
    len: usize,
) -> [i32; MAX_LEN] {
    let mut zero_points = [0; MAX_LEN];
    if let Some(quant) = quant {
        #[allow(clippy::manual_memcpy)]
        for row in 0..len {
            zero_points[row] = quant.zero_point[row].into();
        }
    }
    zero_points
}

#[cfg(feature = "avx512")]
pub struct Avx512Int8Kernel {
    _private: (),
}

#[cfg(feature = "avx512")]
impl Avx512Int8Kernel {
    const MR: usize = 8;
    // Tile size matches AVX-512 register
    const NR: usize = 16;
}

#[cfg(feature = "avx512")]
unsafe impl Kernel<u8, i8, i32> for Avx512Int8Kernel {
    fn new() -> Option<Self> {
        is_avx512_supported().then_some(Avx512Int8Kernel { _private: () })
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
        _out: &mut [MaybeUninit<u8>],
        _image: &Im2Col<i8>,
        _rows: Range<usize>,
        _cols: Range<usize>,
    ) {
        unimplemented!("pack_im2col not implemented");
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

        let a_zero_points = extract_zero_points(a_quant, used_rows);
        let b_zero_points = extract_zero_points(b_quant, used_cols);
        let (a_data, a_row_sums) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
        let (b, b_col_sums) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

        unsafe {
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
                avx512_int8_dot_product,
            )
        }
    }

    fn gemv_kernel(
        &self,
        out: &mut [MaybeUninit<i32>],
        a: &[u8],
        b: Matrix<i8>,
        alpha: f32,
        beta: i32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        // TODO - Optimize with AVX intrinsics.
        assert!(beta == 0 || beta == 1);
        assert_eq!(alpha, 1.);
        assert_eq!(b.rows(), a.len());
        assert_eq!(out.len(), b.cols());

        let a_zero = a_quant.map(|aq| aq.zero_point[0] as i32).unwrap_or(0);
        let depth = a.len();

        for (out, col) in out.iter_mut().zip(0..b.cols()) {
            let b_zero = b_quant.map(|bq| bq.zero_point[col] as i32).unwrap_or(0);
            let mut acc = 0;
            for k in 0..depth {
                let a_el = unsafe { *a.get_unchecked(k) } as i32 - a_zero;
                let b_el = unsafe { *b.get_unchecked([k, col]) } as i32 - b_zero;
                acc += a_el * b_el;
            }
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
}

/// Compute 16 dot products between `u8` values in `a`, `i8` values in `b` and
/// add the `i32` results to `c`.
#[cfg(feature = "avx512")]
#[inline(always)]
unsafe fn avx512_int8_dot_product(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    use core::arch::x86_64::{
        _mm512_add_epi32, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_set1_epi16,
    };
    let tmp = _mm512_maddubs_epi16(a, b);
    let tmp = _mm512_madd_epi16(tmp, _mm512_set1_epi16(1));
    _mm512_add_epi32(c, tmp)
}
