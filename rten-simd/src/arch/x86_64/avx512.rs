use std::arch::x86_64::{
    __m512, __m512i, __mmask16, __mmask32, __mmask64, _mm512_add_epi16, _mm512_add_epi32,
    _mm512_add_epi8, _mm512_add_ps, _mm512_and_ps, _mm512_and_si512, _mm512_andnot_ps,
    _mm512_andnot_si512, _mm512_castsi256_si512, _mm512_cmp_epi16_mask, _mm512_cmp_epi32_mask,
    _mm512_cmp_epu16_mask, _mm512_cmp_ps_mask, _mm512_cmpeq_epi8_mask, _mm512_cmpeq_epu8_mask,
    _mm512_cmpge_epi8_mask, _mm512_cmpge_epu8_mask, _mm512_cmpgt_epi8_mask, _mm512_cmpgt_epu8_mask,
    _mm512_cvtepi16_epi32, _mm512_cvtepi16_epi8, _mm512_cvtepi8_epi16, _mm512_cvtepu8_epi16,
    _mm512_cvtps_epi32, _mm512_cvttps_epi32, _mm512_div_ps, _mm512_extracti64x4_epi64,
    _mm512_fmadd_ps, _mm512_inserti64x4, _mm512_loadu_ps, _mm512_loadu_si512,
    _mm512_mask_blend_epi16, _mm512_mask_blend_epi32, _mm512_mask_blend_epi8, _mm512_mask_blend_ps,
    _mm512_mask_loadu_epi16, _mm512_mask_loadu_epi32, _mm512_mask_loadu_epi8, _mm512_mask_loadu_ps,
    _mm512_mask_storeu_epi16, _mm512_mask_storeu_epi32, _mm512_mask_storeu_epi8,
    _mm512_mask_storeu_ps, _mm512_max_ps, _mm512_min_ps, _mm512_mul_ps, _mm512_mullo_epi16,
    _mm512_mullo_epi32, _mm512_or_ps, _mm512_or_si512, _mm512_packs_epi32, _mm512_packus_epi16,
    _mm512_permutex2var_epi32, _mm512_permutexvar_epi64, _mm512_reduce_add_ps, _mm512_set1_epi16,
    _mm512_set1_epi32, _mm512_set1_epi8, _mm512_set1_ps, _mm512_setr_epi32, _mm512_setr_epi64,
    _mm512_setzero_si512, _mm512_sllv_epi16, _mm512_sllv_epi32, _mm512_storeu_ps,
    _mm512_storeu_si512, _mm512_sub_epi16, _mm512_sub_epi32, _mm512_sub_epi8, _mm512_sub_ps,
    _mm512_unpackhi_epi16, _mm512_unpackhi_epi8, _mm512_unpacklo_epi16, _mm512_unpacklo_epi8,
    _mm512_xor_ps, _mm512_xor_si512, _mm_prefetch, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ,
    _CMP_LT_OQ, _MM_CMPINT_EQ, _MM_CMPINT_NLE, _MM_CMPINT_NLT, _MM_HINT_ET0, _MM_HINT_T0,
};
use std::mem::transmute;

use super::super::{lanes, simd_type};
use crate::ops::{
    Extend, FloatOps, IntOps, Interleave, MaskOps, Narrow, NarrowSaturate, NumOps, SignedIntOps,
};
use crate::{Isa, Mask, Simd};

simd_type!(F32x16, __m512, f32, __mmask16, Avx512Isa);
simd_type!(I32x16, __m512i, i32, __mmask16, Avx512Isa);
simd_type!(I16x32, __m512i, i16, __mmask32, Avx512Isa);
simd_type!(I8x64, __m512i, i8, __mmask64, Avx512Isa);
simd_type!(U8x64, __m512i, u8, __mmask64, Avx512Isa);
simd_type!(U16x32, __m512i, u16, __mmask32, Avx512Isa);

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
    type I8 = I8x64;
    type U8 = U8x64;
    type U16 = U16x32;
    type Bits = I32x16;

    fn f32(self) -> impl FloatOps<f32, Simd = Self::F32, Int = Self::I32> {
        self
    }

    fn i32(
        self,
    ) -> impl SignedIntOps<i32, Simd = Self::I32> + NarrowSaturate<i32, i16, Output = Self::I16>
    {
        self
    }

    fn i16(
        self,
    ) -> impl SignedIntOps<i16, Simd = Self::I16>
           + NarrowSaturate<i16, u8, Output = Self::U8>
           + Extend<i16, Output = Self::I32>
           + Interleave<i16> {
        self
    }

    fn i8(
        self,
    ) -> impl SignedIntOps<i8, Simd = Self::I8> + Extend<i8, Output = Self::I16> + Interleave<i8>
    {
        self
    }

    fn u8(self) -> impl NumOps<u8, Simd = Self::U8> {
        self
    }

    fn u16(self) -> impl IntOps<u16, Simd = Self::U16> {
        self
    }
}

macro_rules! simd_ops_common {
    ($simd:ty, $mask:ty) => {
        type Simd = $simd;

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

macro_rules! simd_int_ops_common {
    ($simd:ty) => {
        #[inline]
        fn and(self, x: $simd, y: $simd) -> $simd {
            unsafe { _mm512_and_si512(x.0, y.0) }.into()
        }

        #[inline]
        fn or(self, x: $simd, y: $simd) -> $simd {
            unsafe { _mm512_or_si512(x.0, y.0) }.into()
        }

        #[inline]
        fn xor(self, x: $simd, y: $simd) -> $simd {
            unsafe { _mm512_xor_si512(x.0, y.0) }.into()
        }

        #[inline]
        fn not(self, x: $simd) -> $simd {
            unsafe { _mm512_andnot_si512(x.0, _mm512_set1_epi8(-1)) }.into()
        }
    };
}

unsafe impl NumOps<f32> for Avx512Isa {
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
    fn and(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_and_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn not(self, x: F32x16) -> F32x16 {
        let all_ones: F32x16 = self.splat(f32::from_bits(0xFFFFFFFF));
        unsafe { _mm512_andnot_ps(x.0, all_ones.0) }.into()
    }

    #[inline]
    fn or(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_or_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn xor(self, x: F32x16, y: F32x16) -> F32x16 {
        unsafe { _mm512_xor_ps(x.0, y.0) }.into()
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

impl FloatOps<f32> for Avx512Isa {
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

unsafe impl NumOps<i32> for Avx512Isa {
    simd_ops_common!(I32x16, __mmask16);
    simd_int_ops_common!(I32x16);

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

impl IntOps<i32> for Avx512Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I32x16) -> I32x16 {
        let count: I32x16 = self.splat(SHIFT);
        unsafe { _mm512_sllv_epi32(x.0, count.0) }.into()
    }
}

impl SignedIntOps<i32> for Avx512Isa {
    #[inline]
    fn neg(self, x: I32x16) -> I32x16 {
        unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), x.0) }.into()
    }
}

impl NarrowSaturate<i32, i16> for Avx512Isa {
    type Output = I16x32;

    #[inline]
    fn narrow_saturate(self, low: I32x16, high: I32x16) -> I16x32 {
        unsafe {
            // _mm512_packs_epi32 treats each input as 4 128-bit lanes and
            // interleaves narrowed 64-bit blocks from each input. Shuffle the
            // output to get narrowed lanes from `low` followed by lanes from
            // `high`.
            let packed = _mm512_packs_epi32(low.0, high.0);
            let permutation = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
            _mm512_permutexvar_epi64(permutation, packed)
        }
        .into()
    }
}

unsafe impl NumOps<i16> for Avx512Isa {
    simd_ops_common!(I16x32, __mmask32);
    simd_int_ops_common!(I16x32);

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

impl IntOps<i16> for Avx512Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I16x32) -> I16x32 {
        let count: I16x32 = self.splat(SHIFT as i16);
        unsafe { _mm512_sllv_epi16(x.0, count.0) }.into()
    }
}

impl SignedIntOps<i16> for Avx512Isa {
    #[inline]
    fn neg(self, x: I16x32) -> I16x32 {
        unsafe { _mm512_sub_epi16(_mm512_setzero_si512(), x.0) }.into()
    }
}

impl NarrowSaturate<i16, u8> for Avx512Isa {
    type Output = U8x64;

    #[inline]
    fn narrow_saturate(self, low: I16x32, high: I16x32) -> U8x64 {
        unsafe {
            // _mm512_packus_epi16 treats each input as 4 128-bit lanes and
            // interleaves narrowed 64-bit blocks from each input. Shuffle the
            // output to get narrowed lanes from `low` followed by lanes from
            // `high`.
            let packed = _mm512_packus_epi16(low.0, high.0);
            let permutation = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
            _mm512_permutexvar_epi64(permutation, packed)
        }
        .into()
    }
}

impl Interleave<i16> for Avx512Isa {
    #[inline]
    fn interleave_low(self, a: I16x32, b: I16x32) -> I16x32 {
        unsafe {
            // AB{N} = Interleaved Nth 64-bit block.
            let lo = _mm512_unpacklo_epi16(a.0, b.0); // AB0 AB2 AB4 AB6
            let hi = _mm512_unpackhi_epi16(a.0, b.0); // AB1 AB3 AB5 AB7
            let idx = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
            _mm512_permutex2var_epi32(lo, idx, hi) // AB0 AB1 AB2 AB3
        }
        .into()
    }

    #[inline]
    fn interleave_high(self, a: I16x32, b: I16x32) -> I16x32 {
        unsafe {
            // AB{N} = Interleaved Nth 64-bit block.
            let lo = _mm512_unpacklo_epi16(a.0, b.0); // AB0 AB2 AB4 AB6
            let hi = _mm512_unpackhi_epi16(a.0, b.0); // AB1 AB3 AB5 AB7
            let idx =
                _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
            _mm512_permutex2var_epi32(lo, idx, hi) // AB4 AB5 AB6 AB7
        }
        .into()
    }
}

unsafe impl NumOps<i8> for Avx512Isa {
    simd_ops_common!(I8x64, __mmask64);
    simd_int_ops_common!(I8x64);

    #[inline]
    fn add(self, x: I8x64, y: I8x64) -> I8x64 {
        unsafe { _mm512_add_epi8(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: I8x64, y: I8x64) -> I8x64 {
        unsafe { _mm512_sub_epi8(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: I8x64, y: I8x64) -> I8x64 {
        let (x_lo, x_hi) = Extend::<i8>::extend(self, x);
        let (y_lo, y_hi) = Extend::<i8>::extend(self, y);

        let i16_ops = self.i16();
        let prod_lo = i16_ops.mul(x_lo, y_lo);
        let prod_hi = i16_ops.mul(x_hi, y_hi);

        self.narrow_truncate(prod_lo, prod_hi)
    }

    #[inline]
    fn splat(self, x: i8) -> I8x64 {
        unsafe { _mm512_set1_epi8(x) }.into()
    }

    #[inline]
    fn eq(self, x: I8x64, y: I8x64) -> __mmask64 {
        unsafe { _mm512_cmpeq_epi8_mask(x.0, y.0) }
    }

    #[inline]
    fn ge(self, x: I8x64, y: I8x64) -> __mmask64 {
        unsafe { _mm512_cmpge_epi8_mask(x.0, y.0) }
    }

    #[inline]
    fn gt(self, x: I8x64, y: I8x64) -> __mmask64 {
        unsafe { _mm512_cmpgt_epi8_mask(x.0, y.0) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i8) -> I8x64 {
        unsafe { _mm512_loadu_si512(ptr as *const i32) }.into()
    }

    #[inline]
    fn select(self, x: I8x64, y: I8x64, mask: <I8x64 as Simd>::Mask) -> I8x64 {
        unsafe { _mm512_mask_blend_epi8(mask, y.0, x.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: I8x64, ptr: *mut i8) {
        unsafe { _mm512_storeu_si512(ptr as *mut __m512i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i8, mask: __mmask64) -> I8x64 {
        unsafe { _mm512_mask_loadu_epi8(_mm512_set1_epi8(0), mask, ptr) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: I8x64, ptr: *mut i8, mask: __mmask64) {
        unsafe { _mm512_mask_storeu_epi8(ptr, mask, x.0) }
    }
}

impl IntOps<i8> for Avx512Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I8x64) -> I8x64 {
        let (x_lo, x_hi) = Extend::<i8>::extend(self, x);

        let i16_ops = self.i16();
        let (y_lo, y_hi) = (
            i16_ops.shift_left::<SHIFT>(x_lo),
            i16_ops.shift_left::<SHIFT>(x_hi),
        );

        self.narrow_truncate(y_lo, y_hi)
    }
}

impl SignedIntOps<i8> for Avx512Isa {
    #[inline]
    fn neg(self, x: I8x64) -> I8x64 {
        unsafe { _mm512_sub_epi8(_mm512_setzero_si512(), x.0) }.into()
    }
}

impl Interleave<i8> for Avx512Isa {
    #[inline]
    fn interleave_low(self, a: I8x64, b: I8x64) -> I8x64 {
        unsafe {
            // AB{N} = Interleaved Nth 64-bit block.
            let lo = _mm512_unpacklo_epi8(a.0, b.0); // AB0 AB2 AB4 AB6
            let hi = _mm512_unpackhi_epi8(a.0, b.0); // AB1 AB3 AB5 AB7
            let idx = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
            _mm512_permutex2var_epi32(lo, idx, hi) // AB0 AB1 AB2 AB3
        }
        .into()
    }

    #[inline]
    fn interleave_high(self, a: I8x64, b: I8x64) -> I8x64 {
        unsafe {
            // AB{N} = Interleaved Nth 64-bit block.
            let lo = _mm512_unpacklo_epi8(a.0, b.0); // AB0 AB2 AB4 AB6
            let hi = _mm512_unpackhi_epi8(a.0, b.0); // AB1 AB3 AB5 AB7
            let idx =
                _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
            _mm512_permutex2var_epi32(lo, idx, hi) // AB4 AB5 AB6 AB7
        }
        .into()
    }
}

unsafe impl NumOps<u8> for Avx512Isa {
    simd_ops_common!(U8x64, __mmask64);
    simd_int_ops_common!(U8x64);

    #[inline]
    fn add(self, x: U8x64, y: U8x64) -> U8x64 {
        unsafe { _mm512_add_epi8(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: U8x64, y: U8x64) -> U8x64 {
        unsafe { _mm512_sub_epi8(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: U8x64, y: U8x64) -> U8x64 {
        let (x_lo, x_hi) = Extend::<u8>::extend(self, x);
        let (y_lo, y_hi) = Extend::<u8>::extend(self, y);

        let u16_ops = self.u16();
        let prod_lo = u16_ops.mul(x_lo, y_lo);
        let prod_hi = u16_ops.mul(x_hi, y_hi);

        self.narrow_truncate(prod_lo, prod_hi)
    }

    #[inline]
    fn splat(self, x: u8) -> U8x64 {
        unsafe { _mm512_set1_epi8(x as i8) }.into()
    }

    #[inline]
    fn eq(self, x: U8x64, y: U8x64) -> __mmask64 {
        unsafe { _mm512_cmpeq_epu8_mask(x.0, y.0) }
    }

    #[inline]
    fn ge(self, x: U8x64, y: U8x64) -> __mmask64 {
        unsafe { _mm512_cmpge_epu8_mask(x.0, y.0) }
    }

    #[inline]
    fn gt(self, x: U8x64, y: U8x64) -> __mmask64 {
        unsafe { _mm512_cmpgt_epu8_mask(x.0, y.0) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const u8) -> U8x64 {
        unsafe { _mm512_loadu_si512(ptr as *const i32) }.into()
    }

    #[inline]
    fn select(self, x: U8x64, y: U8x64, mask: <U8x64 as Simd>::Mask) -> U8x64 {
        unsafe { _mm512_mask_blend_epi8(mask, y.0, x.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: U8x64, ptr: *mut u8) {
        unsafe { _mm512_storeu_si512(ptr as *mut __m512i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const u8, mask: __mmask64) -> U8x64 {
        unsafe { _mm512_mask_loadu_epi8(_mm512_set1_epi8(0), mask, ptr as *const i8) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: U8x64, ptr: *mut u8, mask: __mmask64) {
        unsafe { _mm512_mask_storeu_epi8(ptr as *mut i8, mask, x.0) }
    }
}

impl Extend<i16> for Avx512Isa {
    type Output = I32x16;

    #[inline]
    fn extend(self, x: I16x32) -> (Self::Output, Self::Output) {
        unsafe {
            let lo = _mm512_extracti64x4_epi64(x.0, 0);
            let lo = _mm512_cvtepi16_epi32(lo);

            let hi = _mm512_extracti64x4_epi64(x.0, 1);
            let hi = _mm512_cvtepi16_epi32(hi);
            (lo.into(), hi.into())
        }
    }
}

impl Extend<i8> for Avx512Isa {
    type Output = I16x32;

    #[inline]
    fn extend(self, x: I8x64) -> (I16x32, I16x32) {
        unsafe {
            let lo = _mm512_extracti64x4_epi64(x.0, 0);
            let lo = _mm512_cvtepi8_epi16(lo);

            let hi = _mm512_extracti64x4_epi64(x.0, 1);
            let hi = _mm512_cvtepi8_epi16(hi);
            (lo.into(), hi.into())
        }
    }
}

impl Extend<u8> for Avx512Isa {
    type Output = U16x32;

    #[inline]
    fn extend(self, x: U8x64) -> (U16x32, U16x32) {
        unsafe {
            let lo = _mm512_extracti64x4_epi64(x.0, 0);
            let lo = _mm512_cvtepu8_epi16(lo);

            let hi = _mm512_extracti64x4_epi64(x.0, 1);
            let hi = _mm512_cvtepu8_epi16(hi);
            (lo.into(), hi.into())
        }
    }
}

impl Narrow<I16x32> for Avx512Isa {
    type Output = I8x64;

    #[inline]
    fn narrow_truncate(self, a: I16x32, b: I16x32) -> I8x64 {
        let y = unsafe {
            let lo_i8 = _mm512_cvtepi16_epi8(a.0);
            let hi_i8 = _mm512_cvtepi16_epi8(b.0);
            _mm512_inserti64x4(_mm512_castsi256_si512(lo_i8), hi_i8, 1)
        };
        I8x64(y)
    }
}

impl Narrow<U16x32> for Avx512Isa {
    type Output = U8x64;

    #[inline]
    fn narrow_truncate(self, a: U16x32, b: U16x32) -> U8x64 {
        let y = unsafe {
            let lo_u8 = _mm512_cvtepi16_epi8(a.0);
            let hi_u8 = _mm512_cvtepi16_epi8(b.0);
            _mm512_inserti64x4(_mm512_castsi256_si512(lo_u8), hi_u8, 1)
        };
        U8x64(y)
    }
}

unsafe impl NumOps<u16> for Avx512Isa {
    simd_ops_common!(U16x32, __mmask32);
    simd_int_ops_common!(U16x32);

    #[inline]
    fn add(self, x: U16x32, y: U16x32) -> U16x32 {
        unsafe { _mm512_add_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: U16x32, y: U16x32) -> U16x32 {
        unsafe { _mm512_sub_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: U16x32, y: U16x32) -> U16x32 {
        unsafe { _mm512_mullo_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn splat(self, x: u16) -> U16x32 {
        unsafe { _mm512_set1_epi16(x as i16) }.into()
    }

    #[inline]
    fn eq(self, x: U16x32, y: U16x32) -> __mmask32 {
        unsafe { _mm512_cmp_epu16_mask(x.0, y.0, _MM_CMPINT_EQ) }
    }

    #[inline]
    fn ge(self, x: U16x32, y: U16x32) -> __mmask32 {
        unsafe { _mm512_cmp_epu16_mask(x.0, y.0, _MM_CMPINT_NLT) }
    }

    #[inline]
    fn gt(self, x: U16x32, y: U16x32) -> __mmask32 {
        unsafe { _mm512_cmp_epu16_mask(x.0, y.0, _MM_CMPINT_NLE) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const u16) -> U16x32 {
        unsafe { _mm512_loadu_si512(ptr as *const i32) }.into()
    }

    #[inline]
    fn select(self, x: U16x32, y: U16x32, mask: <U16x32 as Simd>::Mask) -> U16x32 {
        unsafe { _mm512_mask_blend_epi16(mask, y.0, x.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: U16x32, ptr: *mut u16) {
        unsafe { _mm512_storeu_si512(ptr as *mut __m512i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const u16, mask: __mmask32) -> U16x32 {
        unsafe { _mm512_mask_loadu_epi16(_mm512_set1_epi16(0), mask, ptr as *const i16) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: U16x32, ptr: *mut u16, mask: __mmask32) {
        unsafe { _mm512_mask_storeu_epi16(ptr as *mut i16, mask, x.0) }
    }
}

impl IntOps<u16> for Avx512Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: U16x32) -> U16x32 {
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
impl_mask!(__mmask64);
