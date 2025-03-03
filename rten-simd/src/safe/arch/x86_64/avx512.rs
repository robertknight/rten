use std::arch::x86_64::{
    __m512, __m512i, __mmask16, _mm512_add_epi32, _mm512_add_ps, _mm512_andnot_ps,
    _mm512_cmp_epi32_mask, _mm512_cmp_ps_mask, _mm512_cvttps_epi32, _mm512_div_ps, _mm512_fmadd_ps,
    _mm512_loadu_ps, _mm512_loadu_si512, _mm512_mask_blend_epi32, _mm512_mask_blend_ps,
    _mm512_mask_loadu_epi32, _mm512_mask_loadu_ps, _mm512_mask_storeu_epi32, _mm512_mask_storeu_ps,
    _mm512_max_ps, _mm512_min_ps, _mm512_mul_ps, _mm512_mullo_epi32, _mm512_reduce_add_ps,
    _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_si512, _mm512_sllv_epi32, _mm512_storeu_ps,
    _mm512_storeu_si512, _mm512_sub_epi32, _mm512_sub_ps, _mm512_xor_ps, _mm_prefetch, _CMP_EQ_OQ,
    _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _MM_CMPINT_EQ, _MM_CMPINT_LE, _MM_CMPINT_LT,
    _MM_HINT_ET0, _MM_HINT_T0,
};
use std::mem::transmute;

use crate::safe::{Isa, Mask, MaskOps, Simd, SimdFloatOps, SimdIntOps, SimdOps};

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
    type F32 = __m512;
    type I32 = __m512i;
    type Bits = __m512i;

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
            16
        }

        #[inline]
        fn mask_ops(self) -> impl MaskOps<$mask> {
            self
        }

        #[inline]
        fn first_n_mask(self, n: usize) -> __mmask16 {
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

unsafe impl SimdOps<__m512> for Avx512Isa {
    simd_ops_x32_common!(__m512, __mmask16);

    #[inline]
    fn add(self, x: __m512, y: __m512) -> __m512 {
        unsafe { _mm512_add_ps(x, y) }
    }

    #[inline]
    fn sub(self, x: __m512, y: __m512) -> __m512 {
        unsafe { _mm512_sub_ps(x, y) }
    }

    #[inline]
    fn mul(self, x: __m512, y: __m512) -> __m512 {
        unsafe { _mm512_mul_ps(x, y) }
    }

    #[inline]
    fn mul_add(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }

    #[inline]
    fn lt(self, x: __m512, y: __m512) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ) }
    }

    #[inline]
    fn le(self, x: __m512, y: __m512) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ) }
    }

    #[inline]
    fn eq(self, x: __m512, y: __m512) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ) }
    }

    #[inline]
    fn ge(self, x: __m512, y: __m512) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ) }
    }

    #[inline]
    fn gt(self, x: __m512, y: __m512) -> __mmask16 {
        unsafe { _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ) }
    }

    #[inline]
    fn min(self, x: __m512, y: __m512) -> __m512 {
        unsafe { _mm512_min_ps(x, y) }
    }

    #[inline]
    fn max(self, x: __m512, y: __m512) -> __m512 {
        unsafe { _mm512_max_ps(x, y) }
    }

    #[inline]
    fn splat(self, x: f32) -> __m512 {
        unsafe { _mm512_set1_ps(x) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const f32) -> __m512 {
        unsafe { _mm512_loadu_ps(ptr) }
    }

    #[inline]
    fn select(self, x: __m512, y: __m512, mask: <__m512 as Simd>::Mask) -> __m512 {
        unsafe { _mm512_mask_blend_ps(mask, y, x) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const f32, mask: __mmask16) -> __m512 {
        unsafe { _mm512_mask_loadu_ps(self.zero(), mask, ptr) }
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: __m512, ptr: *mut f32, mask: __mmask16) {
        unsafe { _mm512_mask_storeu_ps(ptr, mask, x) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: __m512, ptr: *mut f32) {
        unsafe { _mm512_storeu_ps(ptr, x) }
    }

    #[inline]
    fn sum(self, x: __m512) -> f32 {
        unsafe { _mm512_reduce_add_ps(x) }
    }
}

impl SimdFloatOps<__m512> for Avx512Isa {
    type Int = <Self as Isa>::I32;

    #[inline]
    fn div(self, x: __m512, y: __m512) -> __m512 {
        unsafe { _mm512_div_ps(x, y) }
    }

    #[inline]
    fn abs(self, x: __m512) -> __m512 {
        unsafe { _mm512_andnot_ps(_mm512_set1_ps(-0.0), x) }
    }

    #[inline]
    fn neg(self, x: __m512) -> __m512 {
        unsafe { _mm512_xor_ps(x, _mm512_set1_ps(-0.0)) }
    }

    #[inline]
    fn to_int_trunc(self, x: __m512) -> Self::Int {
        unsafe { _mm512_cvttps_epi32(x) }
    }
}

unsafe impl SimdOps<__m512i> for Avx512Isa {
    simd_ops_x32_common!(__m512i, __mmask16);

    #[inline]
    fn add(self, x: __m512i, y: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(x, y) }
    }

    #[inline]
    fn sub(self, x: __m512i, y: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(x, y) }
    }

    #[inline]
    fn mul(self, x: __m512i, y: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi32(x, y) }
    }

    #[inline]
    fn splat(self, x: i32) -> __m512i {
        unsafe { _mm512_set1_epi32(x) }
    }

    #[inline]
    fn lt(self, x: __m512i, y: __m512i) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_LT) }
    }

    #[inline]
    fn le(self, x: __m512i, y: __m512i) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_LE) }
    }

    #[inline]
    fn eq(self, x: __m512i, y: __m512i) -> __mmask16 {
        unsafe { _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_EQ) }
    }

    #[inline]
    fn ge(self, x: __m512i, y: __m512i) -> __mmask16 {
        self.le(y, x)
    }

    #[inline]
    fn gt(self, x: __m512i, y: __m512i) -> __mmask16 {
        self.lt(y, x)
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i32) -> __m512i {
        unsafe { _mm512_loadu_si512(ptr as *const i32) }
    }

    #[inline]
    fn select(self, x: __m512i, y: __m512i, mask: <__m512i as Simd>::Mask) -> __m512i {
        unsafe { _mm512_mask_blend_epi32(mask, y, x) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: __m512i, ptr: *mut i32) {
        unsafe { _mm512_storeu_si512(ptr as *mut __m512i, x) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i32, mask: __mmask16) -> __m512i {
        unsafe { _mm512_mask_loadu_epi32(self.zero(), mask, ptr) }
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: __m512i, ptr: *mut i32, mask: __mmask16) {
        unsafe { _mm512_mask_storeu_epi32(ptr, mask, x) }
    }
}

impl SimdIntOps<__m512i> for Avx512Isa {
    #[inline]
    fn neg(self, x: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), x) }
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: __m512i) -> __m512i {
        let count = self.splat(SHIFT);
        unsafe { _mm512_sllv_epi32(x, count) }
    }
}

impl Mask for __mmask16 {
    type Array = [bool; 16];

    #[inline]
    fn to_array(self) -> Self::Array {
        std::array::from_fn(|i| self & (1 << i) != 0)
    }
}

unsafe impl MaskOps<__mmask16> for Avx512Isa {
    #[inline]
    fn and(self, x: __mmask16, y: __mmask16) -> __mmask16 {
        x & y
    }
}

macro_rules! simd_x32_common {
    () => {
        type Array = [Self::Elem; 16];
        type Isa = Avx512Isa;
        type Mask = __mmask16;

        #[inline]
        fn to_bits(self) -> <Self::Isa as Isa>::Bits {
            #[allow(clippy::useless_transmute)]
            unsafe {
                transmute::<Self, __m512i>(self)
            }
        }

        #[inline]
        fn from_bits(bits: <Self::Isa as Isa>::Bits) -> Self {
            #[allow(clippy::useless_transmute)]
            unsafe {
                transmute::<__m512i, Self>(bits)
            }
        }
    };
}

impl Simd for __m512 {
    type Elem = f32;

    simd_x32_common!();

    #[inline]
    fn to_array(self) -> Self::Array {
        unsafe { transmute::<__m512, Self::Array>(self) }
    }
}

impl Simd for __m512i {
    type Elem = i32;

    simd_x32_common!();

    #[inline]
    fn to_array(self) -> Self::Array {
        unsafe { transmute::<__m512i, Self::Array>(self) }
    }
}
