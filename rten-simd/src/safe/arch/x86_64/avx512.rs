use std::arch::x86_64::{
    __m512, __m512i, __mmask16, __mmask32, _mm512_add_epi16, _mm512_add_epi32, _mm512_add_ps,
    _mm512_andnot_ps, _mm512_cmp_epi16_mask, _mm512_cmp_epi32_mask, _mm512_cmp_ps_mask,
    _mm512_cvtps_epi32, _mm512_cvttps_epi32, _mm512_div_ps, _mm512_fmadd_ps, _mm512_loadu_ps,
    _mm512_loadu_si512, _mm512_mask_blend_epi16, _mm512_mask_blend_epi32, _mm512_mask_blend_ps,
    _mm512_mask_loadu_epi16, _mm512_mask_loadu_epi32, _mm512_mask_loadu_ps,
    _mm512_mask_storeu_epi16, _mm512_mask_storeu_epi32, _mm512_mask_storeu_ps, _mm512_max_ps,
    _mm512_min_ps, _mm512_mul_ps, _mm512_mullo_epi16, _mm512_mullo_epi32, _mm512_reduce_add_ps,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_si512, _mm512_sllv_epi16,
    _mm512_sllv_epi32, _mm512_storeu_ps, _mm512_storeu_si512, _mm512_sub_epi16, _mm512_sub_epi32,
    _mm512_sub_ps, _mm512_xor_ps, _mm_prefetch, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ,
    _CMP_LT_OQ, _MM_CMPINT_EQ, _MM_CMPINT_NLE, _MM_CMPINT_NLT, _MM_HINT_ET0, _MM_HINT_T0,
};
use std::mem::transmute;

use super::super::{lanes, simd_type};
use crate::safe::{Isa, Mask, MaskOps, Simd, SimdFloatOps, SimdIntOps, SimdOps};

simd_type!(F32x16, __m512, f32, __mmask16, Avx512Isa);
simd_type!(I32x16, __m512i, i32, __mmask16, Avx512Isa);
simd_type!(I16x32, __m512i, i16, __mmask32, Avx512Isa);

#[derive(Copy, Clone)]
pub struct Avx512Isa {
    _private: (),
}

impl Avx512Isa {
    pub fn new() -> Option<Self> {
        if crate::is_avx512_supported() {
            Some(Avx512Isa { _private: () })
        } else {
            None
        }
    }
}

// Safety: AVX-512 is supported as `Avx512Isa::new` checks this.
unsafe impl Isa for Avx512Isa {
    type F32 = F32x16;
    type I32 = I32x16;
    type I16 = I16x32;
    type Bits = I32x16;

    fn f32(self) -> impl SimdFloatOps<Self::F32, Int = Self::I32> {
        self
    }

    fn i32(self) -> impl SimdIntOps<Self::I32> {
        self
    }

    fn i16(self) -> impl SimdIntOps<Self::I16> {
        self
    }
}

macro_rules! simd_ops_common {
    ($simd:ty, $mask:ty) => {
        #[inline]
        fn len(self) -> usize {
            lanes::<$simd>()
        }

        #[inline]
        fn mask_ops(self) -> impl MaskOps<$mask> {
            self
        }

        #[inline]
        fn first_n_mask(self, n: usize) -> $mask {
            let mut mask = 0;
            for i in 0..n {
                mask |= 1 << i;
            }
            mask
        }

        #[inline]
        fn prefetch(self, ptr: *const <$simd as Simd>::Elem) {
            unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0) }
        }

        #[inline]
        fn prefetch_write(self, ptr: *mut <$simd as Simd>::Elem) {
            unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_ET0) }
        }
    };
}

unsafe impl SimdOps<F32x16> for Avx512Isa {
    simd_ops_common!(F32x16, __mmask16);

    #[inline]
    fn add(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_add_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_sub_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_mul_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn mul_add(self, a: F32x16, b: F32x16, c: F32x16) -> F32x16 {
        unsafe { _mm512_fmadd_ps(a.0, b.0, c.0) }.into()
    }

    #[inline]
    fn lt(self, x: F32x16, y: F32x16) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x.0, y.0, _CMP_LT_OQ) }
    }

    #[inline]
    fn le(self, x: F32x16, y: F32x16) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x.0, y.0, _CMP_LE_OQ) }
    }

    #[inline]
    fn eq(self, x: F32x16, y: F32x16) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x.0, y.0, _CMP_EQ_OQ) }
    }

    #[inline]
    fn ge(self, x: F32x16, y: F32x16) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x.0, y.0, _CMP_GE_OQ) }
    }

    #[inline]
    fn gt(self, x: F32x16, y: F32x16) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x.0, y.0, _CMP_GT_OQ) }
    }

    #[inline]
    fn min(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_min_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn max(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_max_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn splat(self, x: f32) -> F32x16 {
        unsafe { _mm512_set1_ps(x) }.into()
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const f32) -> F32x16 {
        unsafe { _mm512_loadu_ps(ptr) }.into()
    }

    #[inline]
    fn select(self, x: F32x16, y: F32x16, mask: <F32x16 as Simd>::Mask) -> F32x16 {
        unsafe { _mm512_mask_blend_ps(mask, y.0, x.0) }.into()
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const f32, mask: __mmask16) -> F32x16 {
        unsafe { _mm512_mask_loadu_ps(_mm512_set1_ps(0.), mask, ptr) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: F32x16, ptr: *mut f32, mask: __mmask16) {
        unsafe { _mm512_mask_storeu_ps(ptr, mask, x.0) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: F32x16, ptr: *mut f32) {
        unsafe { _mm512_storeu_ps(ptr, x.0) }
    }

    #[inline]
    fn sum(self, x: F32x16) -> f32 {
        unsafe { _mm512_reduce_add_ps(x.0) }
    }
}

impl SimdFloatOps<F32x16> for Avx512Isa {
    type Int = <Self as Isa>::I32;

    #[inline]
    fn div(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_div_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn abs(self, x: F32x16) -> F32x16 {
        unsafe { _mm512_andnot_ps(_mm512_set1_ps(-0.0), x.0) }.into()
    }

    #[inline]
    fn neg(self, x: F32x16) -> F32x16 {
        unsafe { _mm512_xor_ps(x.0, _mm512_set1_ps(-0.0)) }.into()
    }

    #[inline]
    fn to_int_trunc(self, x: F32x16) -> Self::Int {
        unsafe { _mm512_cvttps_epi32(x.0) }.into()
    }

    #[inline]
    fn to_int_round(self, x: F32x16) -> Self::Int {
        unsafe { _mm512_cvtps_epi32(x.0) }.into()
    }
}

unsafe impl SimdOps<I32x16> for Avx512Isa {
    simd_ops_common!(I32x16, __mmask16);

    #[inline]
    fn add(self, x: I32x16, y: I32x16) -> I32x16 {
        unsafe { _mm512_add_epi32(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: I32x16, y: I32x16) -> I32x16 {
        unsafe { _mm512_sub_epi32(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: I32x16, y: I32x16) -> I32x16 {
        unsafe { _mm512_mullo_epi32(x.0, y.0) }.into()
    }

    #[inline]
    fn splat(self, x: i32) -> I32x16 {
        unsafe { _mm512_set1_epi32(x) }.into()
    }

    #[inline]
    fn eq(self, x: I32x16, y: I32x16) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask(x.0, y.0, _MM_CMPINT_EQ) }
    }

    #[inline]
    fn ge(self, x: I32x16, y: I32x16) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask(x.0, y.0, _MM_CMPINT_NLT) }
    }

    #[inline]
    fn gt(self, x: I32x16, y: I32x16) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask(x.0, y.0, _MM_CMPINT_NLE) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i32) -> I32x16 {
        unsafe { _mm512_loadu_si512(ptr as *const i32) }.into()
    }

    #[inline]
    fn select(self, x: I32x16, y: I32x16, mask: <I32x16 as Simd>::Mask) -> I32x16 {
        unsafe { _mm512_mask_blend_epi32(mask, y.0, x.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: I32x16, ptr: *mut i32) {
        unsafe { _mm512_storeu_si512(ptr as *mut __m512i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i32, mask: __mmask16) -> I32x16 {
        unsafe { _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), mask, ptr) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: I32x16, ptr: *mut i32, mask: __mmask16) {
        unsafe { _mm512_mask_storeu_epi32(ptr, mask, x.0) }
    }
}

impl SimdIntOps<I32x16> for Avx512Isa {
    #[inline]
    fn neg(self, x: I32x16) -> I32x16 {
        unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), x.0) }.into()
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I32x16) -> I32x16 {
        let count: I32x16 = self.splat(SHIFT);
        unsafe { _mm512_sllv_epi32(x.0, count.0) }.into()
    }
}

unsafe impl SimdOps<I16x32> for Avx512Isa {
    simd_ops_common!(I16x32, __mmask32);

    #[inline]
    fn add(self, x: I16x32, y: I16x32) -> I16x32 {
        unsafe { _mm512_add_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: I16x32, y: I16x32) -> I16x32 {
        unsafe { _mm512_sub_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: I16x32, y: I16x32) -> I16x32 {
        unsafe { _mm512_mullo_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn splat(self, x: i16) -> I16x32 {
        unsafe { _mm512_set1_epi16(x) }.into()
    }

    #[inline]
    fn eq(self, x: I16x32, y: I16x32) -> __mmask32 {
        unsafe { _mm512_cmp_epi16_mask(x.0, y.0, _MM_CMPINT_EQ) }
    }

    #[inline]
    fn ge(self, x: I16x32, y: I16x32) -> __mmask32 {
        unsafe { _mm512_cmp_epi16_mask(x.0, y.0, _MM_CMPINT_NLT) }
    }

    #[inline]
    fn gt(self, x: I16x32, y: I16x32) -> __mmask32 {
        unsafe { _mm512_cmp_epi16_mask(x.0, y.0, _MM_CMPINT_NLE) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i16) -> I16x32 {
        unsafe { _mm512_loadu_si512(ptr as *const i32) }.into()
    }

    #[inline]
    fn select(self, x: I16x32, y: I16x32, mask: <I16x32 as Simd>::Mask) -> I16x32 {
        unsafe { _mm512_mask_blend_epi16(mask, y.0, x.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: I16x32, ptr: *mut i16) {
        unsafe { _mm512_storeu_si512(ptr as *mut __m512i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i16, mask: __mmask32) -> I16x32 {
        unsafe { _mm512_mask_loadu_epi16(_mm512_set1_epi16(0), mask, ptr) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: I16x32, ptr: *mut i16, mask: __mmask32) {
        unsafe { _mm512_mask_storeu_epi16(ptr, mask, x.0) }
    }
}

impl SimdIntOps<I16x32> for Avx512Isa {
    #[inline]
    fn neg(self, x: I16x32) -> I16x32 {
        unsafe { _mm512_sub_epi16(_mm512_setzero_si512(), x.0) }.into()
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I16x32) -> I16x32 {
        let count: I16x32 = self.splat(SHIFT as i16);
        unsafe { _mm512_sllv_epi16(x.0, count.0) }.into()
    }
}

macro_rules! impl_mask {
    ($mask:ty) => {
        impl Mask for $mask {
            type Array = [bool; size_of::<$mask>() * 8];

            #[inline]
            fn to_array(self) -> Self::Array {
                std::array::from_fn(|i| self & (1 << i) != 0)
            }
        }

        unsafe impl MaskOps<$mask> for Avx512Isa {
            #[inline]
            fn and(self, x: $mask, y: $mask) -> $mask {
                x & y
            }
        }
    };
}

impl_mask!(__mmask16);
impl_mask!(__mmask32);
