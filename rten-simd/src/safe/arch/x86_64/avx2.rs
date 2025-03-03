use std::arch::x86_64::{
    __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps, _mm256_and_si256,
    _mm256_andnot_ps, _mm256_blendv_epi8, _mm256_blendv_ps, _mm256_castps256_ps128, _mm256_cmp_ps,
    _mm256_cmpeq_epi32, _mm256_cmpgt_epi32, _mm256_cvttps_epi32, _mm256_div_ps,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_maskload_epi32, _mm256_maskload_ps, _mm256_maskstore_epi32, _mm256_maskstore_ps,
    _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_mullo_epi32, _mm256_or_si256,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_storeu_ps,
    _mm256_storeu_si256, _mm256_sub_epi32, _mm256_sub_ps, _mm256_xor_ps, _mm_add_ps, _mm_cvtss_f32,
    _mm_movehl_ps, _mm_prefetch, _mm_shuffle_ps, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ,
    _CMP_LT_OQ, _MM_HINT_ET0, _MM_HINT_T0,
};
use std::is_x86_feature_detected;
use std::mem::transmute;

use crate::safe::{Isa, Mask, MaskOps, Simd, SimdFloatOps, SimdIntOps, SimdOps};

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
    type F32 = __m256;
    type I32 = __m256i;
    type Bits = __m256i;

    fn f32(self) -> impl SimdFloatOps<Self::F32, Int = Self::I32> {
        self
    }

    fn i32(self) -> impl SimdIntOps<Self::I32> {
        self
    }
}

macro_rules! simd_ops_x32_common {
    ($simd:ty, $mask:ty) => {
        #[inline]
        fn len(self) -> usize {
            8
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

unsafe impl SimdOps<__m256> for Avx2Isa {
    simd_ops_x32_common!(__m256, __m256);

    #[inline]
    fn first_n_mask(self, n: usize) -> __m256 {
        let mask: [i32; 8] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        unsafe { _mm256_loadu_ps(mask.as_ptr() as *const f32) }
    }

    #[inline]
    fn add(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_add_ps(x, y) }
    }

    #[inline]
    fn sub(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_sub_ps(x, y) }
    }

    #[inline]
    fn mul(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_mul_ps(x, y) }
    }

    #[inline]
    fn mul_add(self, a: __m256, b: __m256, c: __m256) -> __m256 {
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }

    #[inline]
    fn lt(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps(x, y, _CMP_LT_OQ) }
    }

    #[inline]
    fn le(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps(x, y, _CMP_LE_OQ) }
    }

    #[inline]
    fn eq(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps(x, y, _CMP_EQ_OQ) }
    }

    #[inline]
    fn ge(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps(x, y, _CMP_GE_OQ) }
    }

    #[inline]
    fn gt(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps(x, y, _CMP_GT_OQ) }
    }

    #[inline]
    fn min(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_min_ps(x, y) }
    }

    #[inline]
    fn max(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_max_ps(x, y) }
    }

    #[inline]
    fn splat(self, x: f32) -> __m256 {
        unsafe { _mm256_set1_ps(x) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const f32) -> __m256 {
        unsafe { _mm256_loadu_ps(ptr) }
    }

    #[inline]
    fn select(self, x: __m256, y: __m256, mask: <__m256 as Simd>::Mask) -> __m256 {
        unsafe { _mm256_blendv_ps(y, x, mask) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const f32, mask: __m256) -> __m256 {
        unsafe { _mm256_maskload_ps(ptr, transmute::<__m256, __m256i>(mask)) }
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: __m256, ptr: *mut f32, mask: __m256) {
        unsafe { _mm256_maskstore_ps(ptr, transmute::<__m256, __m256i>(mask), x) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: __m256, ptr: *mut f32) {
        unsafe { _mm256_storeu_ps(ptr, x) }
    }

    #[inline]
    fn sum(self, x: __m256) -> f32 {
        // See https://stackoverflow.com/a/13222410/434243
        unsafe {
            let hi_4 = _mm256_extractf128_ps(x, 1);
            let lo_4 = _mm256_castps256_ps128(x);
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

impl SimdFloatOps<__m256> for Avx2Isa {
    type Int = <Self as Isa>::I32;

    #[inline]
    fn div(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_div_ps(x, y) }
    }

    #[inline]
    fn abs(self, x: __m256) -> __m256 {
        unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), x) }
    }

    #[inline]
    fn neg(self, x: __m256) -> __m256 {
        unsafe { _mm256_xor_ps(x, _mm256_set1_ps(-0.0)) }
    }

    #[inline]
    fn to_int_trunc(self, x: __m256) -> Self::Int {
        unsafe { _mm256_cvttps_epi32(x) }
    }
}

unsafe impl SimdOps<__m256i> for Avx2Isa {
    simd_ops_x32_common!(__m256i, __m256i);

    #[inline]
    fn first_n_mask(self, n: usize) -> __m256i {
        let mask: [i32; 8] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        unsafe { _mm256_loadu_si256(mask.as_ptr() as *const __m256i) }
    }

    #[inline]
    fn add(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(x, y) }
    }

    #[inline]
    fn sub(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi32(x, y) }
    }

    #[inline]
    fn mul(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_mullo_epi32(x, y) }
    }

    #[inline]
    fn splat(self, x: i32) -> __m256i {
        unsafe { _mm256_set1_epi32(x) }
    }

    #[inline]
    fn lt(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi32(y, x) }
    }

    #[inline]
    fn le(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_or_si256(_mm256_cmpgt_epi32(y, x), _mm256_cmpeq_epi32(x, y)) }
    }

    #[inline]
    fn eq(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_cmpeq_epi32(x, y) }
    }

    #[inline]
    fn ge(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_or_si256(_mm256_cmpgt_epi32(x, y), _mm256_cmpeq_epi32(x, y)) }
    }

    #[inline]
    fn gt(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi32(x, y) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i32) -> __m256i {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }
    }

    #[inline]
    fn select(self, x: __m256i, y: __m256i, mask: <__m256i as Simd>::Mask) -> __m256i {
        unsafe { _mm256_blendv_epi8(y, x, mask) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: __m256i, ptr: *mut i32) {
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, x) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i32, mask: __m256i) -> __m256i {
        unsafe { _mm256_maskload_epi32(ptr, mask) }
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: __m256i, ptr: *mut i32, mask: __m256i) {
        unsafe { _mm256_maskstore_epi32(ptr, mask, x) }
    }
}

impl SimdIntOps<__m256i> for Avx2Isa {
    #[inline]
    fn neg(self, x: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi32(_mm256_setzero_si256(), x) }
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: __m256i) -> __m256i {
        unsafe { _mm256_slli_epi32(x, SHIFT) }
    }
}

impl Mask for __m256i {
    type Array = [bool; 8];

    #[inline]
    fn to_array(self) -> Self::Array {
        let array = unsafe { transmute::<Self, [i32; 8]>(self) };
        std::array::from_fn(|i| array[i] != 0)
    }
}

unsafe impl MaskOps<__m256i> for Avx2Isa {
    #[inline]
    fn and(self, x: __m256i, y: __m256i) -> __m256i {
        unsafe { _mm256_and_si256(x, y) }
    }
}

impl Mask for __m256 {
    type Array = [bool; 8];

    #[inline]
    fn to_array(self) -> Self::Array {
        let array = unsafe { transmute::<Self, [f32; 8]>(self) };
        std::array::from_fn(|i| array[i] != 0.)
    }
}

unsafe impl MaskOps<__m256> for Avx2Isa {
    #[inline]
    fn and(self, x: __m256, y: __m256) -> __m256 {
        unsafe { _mm256_and_ps(x, y) }
    }
}

macro_rules! simd_x32_common {
    () => {
        type Array = [Self::Elem; 8];
        type Isa = Avx2Isa;

        #[inline]
        fn to_bits(self) -> <Self::Isa as Isa>::Bits {
            #[allow(clippy::useless_transmute)]
            unsafe {
                transmute::<Self, __m256i>(self)
            }
        }

        #[inline]
        fn from_bits(bits: <Self::Isa as Isa>::Bits) -> Self {
            #[allow(clippy::useless_transmute)]
            unsafe {
                transmute::<__m256i, Self>(bits)
            }
        }
    };
}

impl Simd for __m256 {
    type Elem = f32;
    type Mask = __m256;

    simd_x32_common!();

    #[inline]
    fn to_array(self) -> Self::Array {
        unsafe { transmute::<__m256, Self::Array>(self) }
    }
}

impl Simd for __m256i {
    type Elem = i32;
    type Mask = __m256i;

    simd_x32_common!();

    #[inline]
    fn to_array(self) -> Self::Array {
        unsafe { transmute::<__m256i, [i32; 8]>(self) }
    }
}
