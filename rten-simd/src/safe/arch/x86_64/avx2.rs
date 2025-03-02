use std::arch::x86_64::{
    __m256, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps,
    _mm256_and_si256, _mm256_andnot_ps, _mm256_blendv_epi8, _mm256_blendv_ps,
    _mm256_castps256_ps128, _mm256_cmp_ps, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32,
    _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cvtps_epi32, _mm256_cvttps_epi32, _mm256_div_ps,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_maskload_epi32, _mm256_maskload_ps, _mm256_maskstore_epi32, _mm256_maskstore_ps,
    _mm256_max_ps, _mm256_min_ps, _mm256_movemask_epi8, _mm256_mul_ps, _mm256_mullo_epi16,
    _mm256_mullo_epi32, _mm256_or_si256, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_ps,
    _mm256_setzero_si256, _mm256_slli_epi16, _mm256_slli_epi32, _mm256_storeu_ps,
    _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_sub_ps, _mm256_xor_ps,
    _mm_add_ps, _mm_cvtss_f32, _mm_movehl_ps, _mm_prefetch, _mm_shuffle_ps, _CMP_EQ_OQ, _CMP_GE_OQ,
    _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _MM_HINT_ET0, _MM_HINT_T0,
};
use std::is_x86_feature_detected;
use std::mem::transmute;

use super::super::{lanes, simd_type};
use crate::safe::{Isa, Mask, MaskOps, Simd, SimdFloatOps, SimdIntOps, SimdOps};

simd_type!(F32x8, __m256, f32, F32x8, Avx2Isa);
simd_type!(I32x8, __m256i, i32, I32x8, Avx2Isa);
simd_type!(I16x16, __m256i, i16, I16x16, Avx2Isa);

#[derive(Copy, Clone)]
pub struct Avx2Isa {
    _private: (),
}

impl Avx2Isa {
    pub fn new() -> Option<Self> {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Some(Avx2Isa { _private: () })
        } else {
            None
        }
    }
}

// Safety: AVX2 is supported as `Avx2Isa::new` checks this.
unsafe impl Isa for Avx2Isa {
    type F32 = F32x8;
    type I32 = I32x8;
    type I16 = I16x16;
    type Bits = I32x8;

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
        fn prefetch(self, ptr: *const <$simd as Simd>::Elem) {
            unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0) }
        }

        #[inline]
        fn prefetch_write(self, ptr: *mut <$simd as Simd>::Elem) {
            unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_ET0) }
        }
    };
}

unsafe impl SimdOps<F32x8> for Avx2Isa {
    simd_ops_common!(F32x8, F32x8);

    #[inline]
    fn first_n_mask(self, n: usize) -> F32x8 {
        let mask: [i32; 8] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        unsafe { _mm256_loadu_ps(mask.as_ptr() as *const f32) }.into()
    }

    #[inline]
    fn add(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_add_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_sub_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_mul_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn mul_add(self, a: F32x8, b: F32x8, c: F32x8) -> F32x8 {
        unsafe { _mm256_fmadd_ps(a.0, b.0, c.0) }.into()
    }

    #[inline]
    fn lt(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_LT_OQ) }.into()
    }

    #[inline]
    fn le(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_LE_OQ) }.into()
    }

    #[inline]
    fn eq(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_EQ_OQ) }.into()
    }

    #[inline]
    fn ge(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_GE_OQ) }.into()
    }

    #[inline]
    fn gt(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_GT_OQ) }.into()
    }

    #[inline]
    fn min(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_min_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn max(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_max_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn splat(self, x: f32) -> F32x8 {
        unsafe { _mm256_set1_ps(x) }.into()
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const f32) -> F32x8 {
        unsafe { _mm256_loadu_ps(ptr) }.into()
    }

    #[inline]
    fn select(self, x: F32x8, y: F32x8, mask: <F32x8 as Simd>::Mask) -> F32x8 {
        unsafe { _mm256_blendv_ps(y.0, x.0, mask.0) }.into()
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const f32, mask: F32x8) -> F32x8 {
        unsafe { _mm256_maskload_ps(ptr, transmute::<F32x8, __m256i>(mask)) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: F32x8, ptr: *mut f32, mask: F32x8) {
        unsafe { _mm256_maskstore_ps(ptr, transmute::<F32x8, __m256i>(mask), x.0) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: F32x8, ptr: *mut f32) {
        unsafe { _mm256_storeu_ps(ptr, x.0) }
    }

    #[inline]
    fn sum(self, x: F32x8) -> f32 {
        // See https://stackoverflow.com/a/13222410/434243
        unsafe {
            let hi_4 = _mm256_extractf128_ps(x.0, 1);
            let lo_4 = _mm256_castps256_ps128(x.0);
            let sum_4 = _mm_add_ps(lo_4, hi_4);
            let lo_2 = sum_4;
            let hi_2 = _mm_movehl_ps(sum_4, sum_4);
            let sum_2 = _mm_add_ps(lo_2, hi_2);
            let lo = sum_2;
            let hi = _mm_shuffle_ps(sum_2, sum_2, 0x1);
            let sum = _mm_add_ps(lo, hi);
            _mm_cvtss_f32(sum)
        }
    }
}

impl SimdFloatOps<F32x8> for Avx2Isa {
    type Int = <Self as Isa>::I32;

    #[inline]
    fn div(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_div_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn abs(self, x: F32x8) -> F32x8 {
        unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), x.0) }.into()
    }

    #[inline]
    fn neg(self, x: F32x8) -> F32x8 {
        unsafe { _mm256_xor_ps(x.0, _mm256_set1_ps(-0.0)) }.into()
    }

    #[inline]
    fn to_int_trunc(self, x: F32x8) -> Self::Int {
        unsafe { _mm256_cvttps_epi32(x.0) }.into()
    }

    #[inline]
    fn to_int_round(self, x: F32x8) -> Self::Int {
        unsafe { _mm256_cvtps_epi32(x.0) }.into()
    }
}

unsafe impl SimdOps<I32x8> for Avx2Isa {
    simd_ops_common!(I32x8, I32x8);

    #[inline]
    fn first_n_mask(self, n: usize) -> I32x8 {
        let mask: [i32; 8] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        unsafe { _mm256_loadu_si256(mask.as_ptr() as *const __m256i) }.into()
    }

    #[inline]
    fn add(self, x: I32x8, y: I32x8) -> I32x8 {
        unsafe { _mm256_add_epi32(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: I32x8, y: I32x8) -> I32x8 {
        unsafe { _mm256_sub_epi32(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: I32x8, y: I32x8) -> I32x8 {
        unsafe { _mm256_mullo_epi32(x.0, y.0) }.into()
    }

    #[inline]
    fn splat(self, x: i32) -> I32x8 {
        unsafe { _mm256_set1_epi32(x) }.into()
    }

    #[inline]
    fn eq(self, x: I32x8, y: I32x8) -> I32x8 {
        unsafe { _mm256_cmpeq_epi32(x.0, y.0) }.into()
    }

    #[inline]
    fn ge(self, x: I32x8, y: I32x8) -> I32x8 {
        unsafe { _mm256_or_si256(_mm256_cmpgt_epi32(x.0, y.0), _mm256_cmpeq_epi32(x.0, y.0)) }
            .into()
    }

    #[inline]
    fn gt(self, x: I32x8, y: I32x8) -> I32x8 {
        unsafe { _mm256_cmpgt_epi32(x.0, y.0) }.into()
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i32) -> I32x8 {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }.into()
    }

    #[inline]
    fn select(self, x: I32x8, y: I32x8, mask: <I32x8 as Simd>::Mask) -> I32x8 {
        unsafe { _mm256_blendv_epi8(y.0, x.0, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: I32x8, ptr: *mut i32) {
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i32, mask: I32x8) -> I32x8 {
        unsafe { _mm256_maskload_epi32(ptr, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: I32x8, ptr: *mut i32, mask: I32x8) {
        unsafe { _mm256_maskstore_epi32(ptr, mask.0, x.0) }
    }
}

impl SimdIntOps<I32x8> for Avx2Isa {
    #[inline]
    fn neg(self, x: I32x8) -> I32x8 {
        unsafe { _mm256_sub_epi32(_mm256_setzero_si256(), x.0) }.into()
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I32x8) -> I32x8 {
        unsafe { _mm256_slli_epi32(x.0, SHIFT) }.into()
    }
}

unsafe impl SimdOps<I16x16> for Avx2Isa {
    simd_ops_common!(I16x16, I16x16);

    #[inline]
    fn first_n_mask(self, n: usize) -> I16x16 {
        let mask: [i16; 16] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        unsafe { _mm256_loadu_si256(mask.as_ptr() as *const __m256i) }.into()
    }

    #[inline]
    fn add(self, x: I16x16, y: I16x16) -> I16x16 {
        unsafe { _mm256_add_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: I16x16, y: I16x16) -> I16x16 {
        unsafe { _mm256_sub_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: I16x16, y: I16x16) -> I16x16 {
        unsafe { _mm256_mullo_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn splat(self, x: i16) -> I16x16 {
        unsafe { _mm256_set1_epi16(x) }.into()
    }

    #[inline]
    fn lt(self, x: I16x16, y: I16x16) -> I16x16 {
        unsafe { _mm256_cmpgt_epi16(y.0, x.0) }.into()
    }

    #[inline]
    fn le(self, x: I16x16, y: I16x16) -> I16x16 {
        unsafe { _mm256_or_si256(_mm256_cmpgt_epi16(y.0, x.0), _mm256_cmpeq_epi16(x.0, y.0)) }
            .into()
    }

    #[inline]
    fn eq(self, x: I16x16, y: I16x16) -> I16x16 {
        unsafe { _mm256_cmpeq_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn ge(self, x: I16x16, y: I16x16) -> I16x16 {
        unsafe { _mm256_or_si256(_mm256_cmpgt_epi16(x.0, y.0), _mm256_cmpeq_epi16(x.0, y.0)) }
            .into()
    }

    #[inline]
    fn gt(self, x: I16x16, y: I16x16) -> I16x16 {
        unsafe { _mm256_cmpgt_epi16(x.0, y.0) }.into()
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i16) -> I16x16 {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }.into()
    }

    #[inline]
    fn select(self, x: I16x16, y: I16x16, mask: <I16x16 as Simd>::Mask) -> I16x16 {
        unsafe { _mm256_blendv_epi8(y.0, x.0, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: I16x16, ptr: *mut i16) {
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i16, mask: I16x16) -> I16x16 {
        // There is no native masked-load instruction for i16, so fall back to
        // scalar loads.
        let mask = _mm256_movemask_epi8(mask.0) as u32;
        let xs: [i16; 16] = std::array::from_fn(|i| {
            let mask_bit = mask & (1 << (i * 2 + 1));
            if mask_bit != 0 {
                // Safety: Caller promises that `ptr.add(i)` is valid if mask[i] is set.
                unsafe { *ptr.add(i) }
            } else {
                0
            }
        });
        self.load_ptr(xs.as_ptr())
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: I16x16, ptr: *mut i16, mask: I16x16) {
        // There is no native masked-store instruction for i16, so fall back to
        // scalar store.
        let xs = Simd::to_array(x);
        let mask = _mm256_movemask_epi8(mask.0) as u32;
        for i in 0..16 {
            let mask_bit = mask & (1 << (i * 2 + 1));
            if mask_bit != 0 {
                // Safety: Caller promises that `ptr.add(i)` is valid if mask[i] is set.
                unsafe { *ptr.add(i) = xs[i] }
            }
        }
    }
}

impl SimdIntOps<I16x16> for Avx2Isa {
    #[inline]
    fn neg(self, x: I16x16) -> I16x16 {
        unsafe { _mm256_sub_epi16(_mm256_setzero_si256(), x.0) }.into()
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I16x16) -> I16x16 {
        unsafe { _mm256_slli_epi16(x.0, SHIFT) }.into()
    }
}

macro_rules! impl_mask {
    ($mask:ty, $elem:ty) => {
        impl Mask for $mask {
            type Array = [bool; lanes::<Self>()];

            #[inline]
            fn to_array(self) -> Self::Array {
                let array = unsafe { transmute::<Self, [$elem; lanes::<Self>()]>(self) };
                std::array::from_fn(|i| array[i] != <$elem>::default())
            }
        }
    };
}

impl_mask!(F32x8, f32);
impl_mask!(I32x8, i32);
impl_mask!(I16x16, i16);

macro_rules! impl_int_mask_ops {
    ($mask:ty) => {
        unsafe impl MaskOps<$mask> for Avx2Isa {
            #[inline]
            fn and(self, x: $mask, y: $mask) -> $mask {
                unsafe { _mm256_and_si256(x.0, y.0) }.into()
            }
        }
    };
}
impl_int_mask_ops!(I32x8);
impl_int_mask_ops!(I16x16);

unsafe impl MaskOps<F32x8> for Avx2Isa {
    #[inline]
    fn and(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_and_ps(x.0, y.0) }.into()
    }
}
