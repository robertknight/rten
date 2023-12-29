use std::arch::x86_64::{
    __m512, __m512i, _mm512_add_epi32, _mm512_add_ps, _mm512_castsi512_ps, _mm512_cmp_ps_mask,
    _mm512_cmpgt_epi32_mask, _mm512_cvttps_epi32, _mm512_div_ps, _mm512_fmadd_ps, _mm512_loadu_ps,
    _mm512_loadu_si512, _mm512_mask_blend_epi8, _mm512_mask_blend_ps, _mm512_max_ps, _mm512_mul_ps,
    _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_si512, _mm512_slli_epi32, _mm512_storeu_ps,
    _mm512_storeu_si512, _mm512_sub_epi32, _mm512_sub_ps, _CMP_GE_OQ, _CMP_LE_OQ, _CMP_LT_OQ,
};

use crate::simd_vec::{SimdFloat, SimdInt};

impl SimdInt for __m512i {
    type Float = __m512;

    const LEN: usize = 8;

    #[target_feature(enable = "avx512f")]
    unsafe fn zero() -> Self {
        _mm512_setzero_si512()
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn splat(val: i32) -> Self {
        _mm512_set1_epi32(val)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn gt(self, other: Self) -> Self {
        _mm512_cmpgt_epi32(self, other)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn blend(self, other: Self, mask: Self) -> Self {
        _mm512_blendv_epi8(self, other, mask)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm512_add_epi32(self, rhs)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm512_sub_epi32(self, rhs)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn shl<const COUNT: i32>(self) -> Self {
        _mm512_slli_epi32(self, COUNT as u32)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn reinterpret_as_float(self) -> Self::Float {
        _mm512_castsi512_ps(self)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn load(ptr: *const i32) -> Self {
        // Cast is OK because instruction does not require alignment.
        _mm512_loadu_si512(ptr as *const __m512i)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn store(self, ptr: *mut i32) {
        // Cast is OK because instruction does not require alignment.
        _mm512_storeu_si512(ptr as *mut __m512i, self)
    }
}

impl SimdFloat for __m512 {
    type Int = __m512i;

    const LEN: usize = 8;

    #[target_feature(enable = "avx512f")]
    unsafe fn splat(val: f32) -> Self {
        _mm512_set1_ps(val)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn abs(self) -> Self {
        // https://stackoverflow.com/q/63599391/434243
        let sign_bit = _mm512_set1_ps(-0.0);
        _mm512_andnot_ps(sign_bit, self)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        _mm512_fmadd_ps(self, a, b)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm512_sub_ps(self, rhs)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm512_add_ps(self, rhs)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn to_int_trunc(self) -> Self::Int {
        _mm512_cvttps_epi32(self)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn mul(self, rhs: Self) -> Self {
        _mm512_mul_ps(self, rhs)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn div(self, rhs: Self) -> Self {
        _mm512_div_ps(self, rhs)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn ge(self, rhs: Self) -> Self {
        _mm512_cmp_ps_mask(self, rhs, _CMP_GE_OQ)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn le(self, rhs: Self) -> Self {
        _mm512_cmp_ps_mask(self, rhs, _CMP_LE_OQ)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn lt(self, rhs: Self) -> Self {
        _mm512_cmp_ps_mask(self, rhs, _CMP_LT_OQ)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn max(self, rhs: Self) -> Self {
        _mm512_max_ps(self, rhs)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn blend(self, rhs: Self, mask: Self) -> Self {
        _mm512_mask_blend_ps(self, rhs, mask)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn load(ptr: *const f32) -> Self {
        _mm512_loadu_ps(ptr)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn store(self, ptr: *mut f32) {
        _mm512_storeu_ps(ptr, self)
    }
}
