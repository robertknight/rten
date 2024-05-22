use std::arch::x86_64::{
    __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256, _mm256_andnot_ps,
    _mm256_blendv_epi8, _mm256_blendv_ps, _mm256_castps256_ps128, _mm256_castsi256_ps,
    _mm256_cmp_ps, _mm256_cmpeq_epi32, _mm256_cmpgt_epi32, _mm256_cvttps_epi32, _mm256_div_ps,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_max_ps,
    _mm256_mul_ps, _mm256_or_si256, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256,
    _mm256_slli_epi32, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi32, _mm256_sub_ps,
    _mm_add_ps, _mm_cvtss_f32, _mm_movehl_ps, _mm_prefetch, _mm_shuffle_ps, _CMP_GE_OQ, _CMP_LE_OQ,
    _CMP_LT_OQ, _MM_HINT_ET0, _MM_HINT_T0,
};

use crate::{SimdFloat, SimdInt, SimdMask, SimdVal};

impl SimdMask for __m256i {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and(self, other: Self) -> Self {
        _mm256_and_si256(self, other)
    }
}

impl SimdVal for __m256i {
    type Mask = __m256i;
}

impl SimdInt for __m256i {
    type Float = __m256;

    const LEN: usize = 8;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn zero() -> Self {
        _mm256_setzero_si256()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn splat(val: i32) -> Self {
        _mm256_set1_epi32(val)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ge(self, other: Self) -> Self::Mask {
        _mm256_or_si256(self.gt(other), self.eq(other))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn le(self, other: Self) -> Self::Mask {
        _mm256_or_si256(self.lt(other), self.eq(other))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        other.gt(self)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        _mm256_cmpeq_epi32(self, other)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        _mm256_cmpgt_epi32(self, other)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
        _mm256_blendv_epi8(self, other, mask)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm256_add_epi32(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm256_sub_epi32(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shl<const COUNT: i32>(self) -> Self {
        _mm256_slli_epi32(self, COUNT)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn reinterpret_as_float(self) -> Self::Float {
        _mm256_castsi256_ps(self)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load(ptr: *const i32) -> Self {
        // Cast is OK because instruction does not require alignment.
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store(self, ptr: *mut i32) {
        // Cast is OK because instruction does not require alignment.
        _mm256_storeu_si256(ptr as *mut __m256i, self)
    }
}

impl SimdVal for __m256 {
    type Mask = __m256i;
}

impl SimdFloat for __m256 {
    type Int = __m256i;

    const LEN: usize = 8;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn splat(val: f32) -> Self {
        _mm256_set1_ps(val)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn abs(self) -> Self {
        // https://stackoverflow.com/q/63599391/434243
        let sign_bit = _mm256_set1_ps(-0.0);
        _mm256_andnot_ps(sign_bit, self)
    }

    #[inline]
    #[target_feature(enable = "fma")]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        _mm256_fmadd_ps(self, a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm256_sub_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm256_add_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn to_int_trunc(self) -> Self::Int {
        _mm256_cvttps_epi32(self)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul(self, rhs: Self) -> Self {
        _mm256_mul_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn div(self, rhs: Self) -> Self {
        _mm256_div_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ge(self, rhs: Self) -> Self::Mask {
        std::mem::transmute(_mm256_cmp_ps(self, rhs, _CMP_GE_OQ))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn le(self, rhs: Self) -> Self::Mask {
        std::mem::transmute(_mm256_cmp_ps(self, rhs, _CMP_LE_OQ))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lt(self, rhs: Self) -> Self::Mask {
        std::mem::transmute(_mm256_cmp_ps(self, rhs, _CMP_LT_OQ))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn max(self, rhs: Self) -> Self {
        _mm256_max_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn blend(self, rhs: Self, mask: Self::Mask) -> Self {
        _mm256_blendv_ps(self, rhs, std::mem::transmute(mask))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load(ptr: *const f32) -> Self {
        _mm256_loadu_ps(ptr)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn gather_mask(src: *const f32, offsets: Self::Int, mask: Self::Mask) -> Self {
        // AVX2 has a gather instruction, but we don't use it because on some
        // Intel CPUs it is slower than regular loads due to a mitigation for
        // the Gather Data Sampling (GDS) vulnerability.
        //
        // From initial testing it appears that AVX512 is not affected to the
        // same extent, so using an emulated gather may not pay off there.
        //
        // See https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/technical-documentation/gather-data-sampling.html
        super::simd_gather_mask::<Self, { Self::LEN }>(src, offsets, mask)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store(self, ptr: *mut f32) {
        _mm256_storeu_ps(ptr, self)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sum(self) -> f32 {
        // See https://stackoverflow.com/a/13222410/434243
        let hi_4 = _mm256_extractf128_ps(self, 1);
        let lo_4 = _mm256_castps256_ps128(self);
        let sum_4 = _mm_add_ps(lo_4, hi_4);
        let lo_2 = sum_4;
        let hi_2 = _mm_movehl_ps(sum_4, sum_4);
        let sum_2 = _mm_add_ps(lo_2, hi_2);
        let lo = sum_2;
        let hi = _mm_shuffle_ps(sum_2, sum_2, 0x1);
        let sum = _mm_add_ps(lo, hi);
        _mm_cvtss_f32(sum)
    }

    /// Prefetch the cache line containing `data`, for reading.
    #[inline]
    unsafe fn prefetch(data: *const f32) {
        _mm_prefetch(data as *const i8, _MM_HINT_T0);
    }

    /// Prefetch the cache line containing `data`, for writing.
    #[inline]
    unsafe fn prefetch_write(data: *mut f32) {
        _mm_prefetch(data as *const i8, _MM_HINT_ET0);
    }
}

#[cfg(feature = "avx512")]
use std::arch::x86_64::{
    __m512, __m512i, __mmask16, _mm512_abs_ps, _mm512_add_epi32, _mm512_add_ps,
    _mm512_castsi512_ps, _mm512_cmp_epi32_mask, _mm512_cmp_ps_mask, _mm512_cvttps_epi32,
    _mm512_div_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_loadu_si512, _mm512_mask_blend_epi32,
    _mm512_mask_blend_ps, _mm512_mask_i32gather_ps, _mm512_max_ps, _mm512_mul_ps,
    _mm512_reduce_add_ps, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_si512,
    _mm512_sllv_epi32, _mm512_storeu_ps, _mm512_storeu_si512, _mm512_sub_epi32, _mm512_sub_ps,
    _MM_CMPINT_EQ, _MM_CMPINT_LE, _MM_CMPINT_LT,
};

#[cfg(feature = "avx512")]
impl SimdMask for __mmask16 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn and(self, other: Self) -> Self {
        self & other
    }
}

#[cfg(feature = "avx512")]
impl SimdVal for __m512i {
    type Mask = __mmask16;
}

#[cfg(feature = "avx512")]
impl SimdInt for __m512i {
    type Float = __m512;

    const LEN: usize = 16;

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn zero() -> Self {
        _mm512_setzero_si512()
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn splat(val: i32) -> Self {
        _mm512_set1_epi32(val)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        _mm512_cmp_epi32_mask(self, other, _MM_CMPINT_EQ)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn le(self, other: Self) -> Self::Mask {
        _mm512_cmp_epi32_mask(self, other, _MM_CMPINT_LE)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        _mm512_cmp_epi32_mask(self, other, _MM_CMPINT_LT)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn ge(self, other: Self) -> Self::Mask {
        other.le(self)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        other.lt(self)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
        _mm512_mask_blend_epi32(mask, self, other)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm512_add_epi32(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm512_sub_epi32(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn shl<const COUNT: i32>(self) -> Self {
        let count = Self::splat(COUNT);
        _mm512_sllv_epi32(self, count)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn reinterpret_as_float(self) -> Self::Float {
        _mm512_castsi512_ps(self)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load(ptr: *const i32) -> Self {
        _mm512_loadu_si512(ptr)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store(self, ptr: *mut i32) {
        _mm512_storeu_si512(ptr, self)
    }
}

#[cfg(feature = "avx512")]
impl SimdVal for __m512 {
    type Mask = __mmask16;
}

#[cfg(feature = "avx512")]
impl SimdFloat for __m512 {
    type Int = __m512i;

    const LEN: usize = 16;

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn splat(val: f32) -> Self {
        _mm512_set1_ps(val)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn abs(self) -> Self {
        _mm512_abs_ps(self)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        _mm512_fmadd_ps(self, a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm512_sub_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm512_add_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn to_int_trunc(self) -> Self::Int {
        _mm512_cvttps_epi32(self)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul(self, rhs: Self) -> Self {
        _mm512_mul_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn div(self, rhs: Self) -> Self {
        _mm512_div_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn ge(self, rhs: Self) -> Self::Mask {
        _mm512_cmp_ps_mask(self, rhs, _CMP_GE_OQ)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn le(self, rhs: Self) -> Self::Mask {
        _mm512_cmp_ps_mask(self, rhs, _CMP_LE_OQ)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn lt(self, rhs: Self) -> Self::Mask {
        _mm512_cmp_ps_mask(self, rhs, _CMP_LT_OQ)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn max(self, rhs: Self) -> Self {
        _mm512_max_ps(self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn blend(self, rhs: Self, mask: Self::Mask) -> Self {
        _mm512_mask_blend_ps(mask, self, rhs)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load(ptr: *const f32) -> Self {
        _mm512_loadu_ps(ptr)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store(self, ptr: *mut f32) {
        _mm512_storeu_ps(ptr, self)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn gather_mask(ptr: *const f32, offsets: Self::Int, mask: Self::Mask) -> Self {
        _mm512_mask_i32gather_ps::<4>(Self::zero(), mask, offsets, ptr as *const u8)
    }

    #[inline]
    unsafe fn prefetch(data: *const f32) {
        _mm_prefetch(data as *const i8, _MM_HINT_T0);
    }

    #[inline]
    unsafe fn prefetch_write(data: *mut f32) {
        _mm_prefetch(data as *const i8, _MM_HINT_ET0);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sum(self) -> f32 {
        _mm512_reduce_add_ps(self)
    }
}
