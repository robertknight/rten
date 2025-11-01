use std::arch::x86_64::{
    __m128i, __m256, __m256i, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ,
    _MM_FROUND_TO_NEAREST_INT, _MM_HINT_ET0, _MM_HINT_T0, _mm_add_ps, _mm_cvtss_f32, _mm_movehl_ps,
    _mm_prefetch, _mm_setr_epi8, _mm_shuffle_epi8, _mm_shuffle_ps, _mm_unpacklo_epi64,
    _mm256_add_epi8, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps,
    _mm256_and_si256, _mm256_andnot_ps, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_blendv_ps,
    _mm256_castps256_ps128, _mm256_castsi256_si128, _mm256_cmp_ps, _mm256_cmpeq_epi8,
    _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpgt_epi8, _mm256_cmpgt_epi16,
    _mm256_cmpgt_epi32, _mm256_cvtepi8_epi16, _mm256_cvtepi16_epi32, _mm256_cvtepu8_epi16,
    _mm256_cvtps_epi32, _mm256_cvttps_epi32, _mm256_div_ps, _mm256_extractf128_ps,
    _mm256_extracti128_si256, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_insertf128_si256,
    _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskload_epi32, _mm256_maskload_ps,
    _mm256_maskstore_epi32, _mm256_maskstore_ps, _mm256_max_ps, _mm256_min_ps,
    _mm256_movemask_epi8, _mm256_mul_ps, _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_or_ps,
    _mm256_or_si256, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_round_ps, _mm256_set_m128i, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setr_m128i, _mm256_setzero_si256,
    _mm256_slli_epi16, _mm256_slli_epi32, _mm256_srai_epi16, _mm256_srai_epi32, _mm256_srli_epi16,
    _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_sub_epi16, _mm256_sub_epi32,
    _mm256_sub_ps, _mm256_unpackhi_epi8, _mm256_unpackhi_epi16, _mm256_unpacklo_epi8,
    _mm256_unpacklo_epi16, _mm256_xor_ps, _mm256_xor_si256,
};
use std::is_x86_feature_detected;
use std::mem::transmute;

use super::super::{lanes, simd_type};
use crate::ops::{
    Concat, Extend, FloatOps, IntOps, Interleave, MaskOps, Narrow, NarrowSaturate, NumOps,
    SignedIntOps,
};
use crate::{Isa, Mask, Simd};

simd_type!(F32x8, __m256, f32, M32, Avx2Isa);
simd_type!(I32x8, __m256i, i32, M32, Avx2Isa);
simd_type!(I16x16, __m256i, i16, M16, Avx2Isa);
simd_type!(I8x32, __m256i, i8, M8, Avx2Isa);
simd_type!(U8x32, __m256i, u8, M8, Avx2Isa);
simd_type!(U16x16, __m256i, u16, M16, Avx2Isa);
simd_type!(U32x8, __m256i, u32, M32, Avx2Isa);

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
    type M32 = M32;
    type M16 = M16;
    type M8 = M8;
    type F32 = F32x8;
    type I32 = I32x8;
    type I16 = I16x16;
    type I8 = I8x32;
    type U8 = U8x32;
    type U16 = U16x16;
    type U32 = U32x8;
    type Bits = I32x8;

    fn f32(self) -> impl FloatOps<f32, Simd = Self::F32, Int = Self::I32> {
        self
    }

    fn i32(
        self,
    ) -> impl SignedIntOps<i32, Simd = Self::I32>
    + NarrowSaturate<i32, i16, Output = Self::I16>
    + Concat<i32> {
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

    fn u8(
        self,
    ) -> impl IntOps<u8, Simd = Self::U8> + Extend<u8, Output = Self::U16> + Interleave<u8> {
        self
    }

    fn u16(self) -> impl IntOps<u16, Simd = Self::U16> {
        self
    }

    fn m32(self) -> impl MaskOps<Self::M32> {
        self
    }

    fn m16(self) -> impl MaskOps<Self::M16> {
        self
    }

    fn m8(self) -> impl MaskOps<Self::M8> {
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
            unsafe { _mm256_and_si256(x.0, y.0) }.into()
        }

        #[inline]
        fn or(self, x: $simd, y: $simd) -> $simd {
            unsafe { _mm256_or_si256(x.0, y.0) }.into()
        }

        #[inline]
        fn xor(self, x: $simd, y: $simd) -> $simd {
            unsafe { _mm256_xor_si256(x.0, y.0) }.into()
        }

        #[inline]
        fn not(self, x: $simd) -> $simd {
            unsafe { _mm256_andnot_si256(x.0, _mm256_set1_epi8(-1)) }.into()
        }
    };
}

unsafe impl NumOps<f32> for Avx2Isa {
    simd_ops_common!(F32x8, M32);

    #[inline]
    fn first_n_mask(self, n: usize) -> M32 {
        let mask: [i32; 8] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        M32::from_float(unsafe { _mm256_loadu_ps(mask.as_ptr() as *const f32) })
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
    fn lt(self, x: F32x8, y: F32x8) -> M32 {
        M32::from_float(unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_LT_OQ) })
    }

    #[inline]
    fn le(self, x: F32x8, y: F32x8) -> M32 {
        M32::from_float(unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_LE_OQ) })
    }

    #[inline]
    fn eq(self, x: F32x8, y: F32x8) -> M32 {
        M32::from_float(unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_EQ_OQ) })
    }

    #[inline]
    fn ge(self, x: F32x8, y: F32x8) -> M32 {
        M32::from_float(unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_GE_OQ) })
    }

    #[inline]
    fn gt(self, x: F32x8, y: F32x8) -> M32 {
        M32::from_float(unsafe { _mm256_cmp_ps(x.0, y.0, _CMP_GT_OQ) })
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
    fn and(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_and_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn not(self, x: F32x8) -> F32x8 {
        let all_ones: F32x8 = self.splat(f32::from_bits(0xFFFFFFFF));
        unsafe { _mm256_andnot_ps(x.0, all_ones.0) }.into()
    }

    #[inline]
    fn or(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_or_ps(x.0, y.0) }.into()
    }

    #[inline]
    fn xor(self, x: F32x8, y: F32x8) -> F32x8 {
        unsafe { _mm256_xor_ps(x.0, y.0) }.into()
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
    fn select(self, x: F32x8, y: F32x8, mask: M32) -> F32x8 {
        unsafe { _mm256_blendv_ps(y.0, x.0, mask.as_float()) }.into()
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const f32, mask: M32) -> F32x8 {
        unsafe { _mm256_maskload_ps(ptr, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: F32x8, ptr: *mut f32, mask: M32) {
        unsafe { _mm256_maskstore_ps(ptr, mask.0, x.0) }
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

impl FloatOps<f32> for Avx2Isa {
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
    fn mul_sub_from(self, a: F32x8, b: F32x8, c: F32x8) -> F32x8 {
        unsafe { _mm256_fnmadd_ps(a.0, b.0, c.0) }.into()
    }

    #[inline]
    fn round_ties_even(self, x: F32x8) -> F32x8 {
        unsafe { _mm256_round_ps(x.0, _MM_FROUND_TO_NEAREST_INT) }.into()
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

unsafe impl NumOps<i32> for Avx2Isa {
    simd_ops_common!(I32x8, M32);
    simd_int_ops_common!(I32x8);

    #[inline]
    fn first_n_mask(self, n: usize) -> M32 {
        let mask: [i32; 8] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        M32(unsafe { _mm256_loadu_si256(mask.as_ptr() as *const __m256i) })
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
    fn eq(self, x: I32x8, y: I32x8) -> M32 {
        M32(unsafe { _mm256_cmpeq_epi32(x.0, y.0) })
    }

    #[inline]
    fn ge(self, x: I32x8, y: I32x8) -> M32 {
        M32(unsafe { _mm256_or_si256(_mm256_cmpgt_epi32(x.0, y.0), _mm256_cmpeq_epi32(x.0, y.0)) })
    }

    #[inline]
    fn gt(self, x: I32x8, y: I32x8) -> M32 {
        M32(unsafe { _mm256_cmpgt_epi32(x.0, y.0) })
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i32) -> I32x8 {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }.into()
    }

    #[inline]
    fn select(self, x: I32x8, y: I32x8, mask: M32) -> I32x8 {
        unsafe { _mm256_blendv_epi8(y.0, x.0, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: I32x8, ptr: *mut i32) {
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i32, mask: M32) -> I32x8 {
        unsafe { _mm256_maskload_epi32(ptr, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr_mask(self, x: I32x8, ptr: *mut i32, mask: M32) {
        unsafe { _mm256_maskstore_epi32(ptr, mask.0, x.0) }
    }
}

impl IntOps<i32> for Avx2Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I32x8) -> I32x8 {
        unsafe { _mm256_slli_epi32(x.0, SHIFT) }.into()
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: I32x8) -> I32x8 {
        unsafe { _mm256_srai_epi32(x.0, SHIFT) }.into()
    }
}

impl SignedIntOps<i32> for Avx2Isa {
    #[inline]
    fn neg(self, x: I32x8) -> I32x8 {
        unsafe { _mm256_sub_epi32(_mm256_setzero_si256(), x.0) }.into()
    }
}

/// Copied from unstable `_MM_SHUFFLE` function in `core::arch::x86`.
const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

impl NarrowSaturate<i32, i16> for Avx2Isa {
    type Output = I16x16;

    #[inline]
    fn narrow_saturate(self, low: I32x8, high: I32x8) -> I16x16 {
        unsafe {
            // AVX2 pack functions treat each input as 2 128-bit lanes and
            // interleave narrowed 64-bit blocks from each input. Shuffle the
            // output to get narrowed lanes from `low` followed by lanes from
            // high.
            let packed = _mm256_packs_epi32(low.0, high.0);
            _mm256_permute4x64_epi64(packed, _mm_shuffle(3, 1, 2, 0))
        }
        .into()
    }
}

impl Concat<i32> for Avx2Isa {
    #[inline]
    fn concat_low(self, a: I32x8, b: I32x8) -> I32x8 {
        unsafe {
            let a_lo = _mm256_castsi256_si128(a.0);
            let b_lo = _mm256_castsi256_si128(b.0);
            _mm256_set_m128i(b_lo, a_lo)
        }
        .into()
    }

    #[inline]
    fn concat_high(self, a: I32x8, b: I32x8) -> I32x8 {
        unsafe {
            let a_hi = _mm256_extracti128_si256(a.0, 1);
            let b_hi = _mm256_extracti128_si256(b.0, 1);
            _mm256_set_m128i(b_hi, a_hi)
        }
        .into()
    }
}

unsafe impl NumOps<i16> for Avx2Isa {
    simd_ops_common!(I16x16, M16);
    simd_int_ops_common!(I16x16);

    #[inline]
    fn first_n_mask(self, n: usize) -> M16 {
        let mask: [i16; 16] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        M16(unsafe { _mm256_loadu_si256(mask.as_ptr() as *const __m256i) })
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
    fn eq(self, x: I16x16, y: I16x16) -> M16 {
        M16(unsafe { _mm256_cmpeq_epi16(x.0, y.0) })
    }

    #[inline]
    fn ge(self, x: I16x16, y: I16x16) -> M16 {
        M16(unsafe { _mm256_or_si256(_mm256_cmpgt_epi16(x.0, y.0), _mm256_cmpeq_epi16(x.0, y.0)) })
    }

    #[inline]
    fn gt(self, x: I16x16, y: I16x16) -> M16 {
        M16(unsafe { _mm256_cmpgt_epi16(x.0, y.0) })
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i16) -> I16x16 {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }.into()
    }

    #[inline]
    fn select(self, x: I16x16, y: I16x16, mask: M16) -> I16x16 {
        unsafe { _mm256_blendv_epi8(y.0, x.0, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: I16x16, ptr: *mut i16) {
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i16, mask: M16) -> I16x16 {
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
    unsafe fn store_ptr_mask(self, x: I16x16, ptr: *mut i16, mask: M16) {
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

impl IntOps<i16> for Avx2Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I16x16) -> I16x16 {
        unsafe { _mm256_slli_epi16(x.0, SHIFT) }.into()
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: I16x16) -> I16x16 {
        unsafe { _mm256_srai_epi16(x.0, SHIFT) }.into()
    }
}

impl SignedIntOps<i16> for Avx2Isa {
    #[inline]
    fn neg(self, x: I16x16) -> I16x16 {
        unsafe { _mm256_sub_epi16(_mm256_setzero_si256(), x.0) }.into()
    }
}

impl NarrowSaturate<i16, u8> for Avx2Isa {
    type Output = U8x32;

    #[inline]
    fn narrow_saturate(self, low: I16x16, high: I16x16) -> U8x32 {
        unsafe {
            // AVX2 pack functions treat each input as 2 128-bit lanes and
            // interleave narrowed 64-bit blocks from each input. Shuffle the
            // output to get narrowed lanes from `low` followed by lanes from
            // high.
            let packed = _mm256_packus_epi16(low.0, high.0);
            _mm256_permute4x64_epi64(packed, _mm_shuffle(3, 1, 2, 0))
        }
        .into()
    }
}

impl Interleave<i16> for Avx2Isa {
    #[inline]
    fn interleave_low(self, a: I16x16, b: I16x16) -> I16x16 {
        unsafe {
            // AB{N} = Interleaved Nth 64-bit block.
            let lo = _mm256_unpacklo_epi16(a.0, b.0); // AB0 AB2
            let hi = _mm256_unpackhi_epi16(a.0, b.0); // AB1 AB3
            _mm256_insertf128_si256(lo, _mm256_castsi256_si128(hi), 1) // AB0 AB1
        }
        .into()
    }

    #[inline]
    fn interleave_high(self, a: I16x16, b: I16x16) -> I16x16 {
        unsafe {
            // AB{N} = Interleaved Nth 64-bit block.
            let lo = _mm256_unpacklo_epi16(a.0, b.0); // AB0 AB2
            let hi = _mm256_unpackhi_epi16(a.0, b.0); // AB1 AB3
            _mm256_permute2x128_si256(lo, hi, 0x31) // AB2 AB3
        }
        .into()
    }
}

unsafe impl NumOps<i8> for Avx2Isa {
    simd_ops_common!(I8x32, M8);
    simd_int_ops_common!(I8x32);

    #[inline]
    fn first_n_mask(self, n: usize) -> M8 {
        let mask: [i8; 32] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        M8(unsafe { _mm256_loadu_si256(mask.as_ptr() as *const __m256i) })
    }

    #[inline]
    fn add(self, x: I8x32, y: I8x32) -> I8x32 {
        unsafe { _mm256_add_epi8(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: I8x32, y: I8x32) -> I8x32 {
        unsafe { _mm256_sub_epi8(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: I8x32, y: I8x32) -> I8x32 {
        let (x_lo, x_hi) = Extend::<i8>::extend(self, x);
        let (y_lo, y_hi) = Extend::<i8>::extend(self, y);

        let i16_ops = self.i16();
        let prod_lo = i16_ops.mul(x_lo, y_lo);
        let prod_hi = i16_ops.mul(x_hi, y_hi);

        self.narrow_truncate(prod_lo, prod_hi)
    }

    #[inline]
    fn splat(self, x: i8) -> I8x32 {
        unsafe { _mm256_set1_epi8(x) }.into()
    }

    #[inline]
    fn eq(self, x: I8x32, y: I8x32) -> M8 {
        M8(unsafe { _mm256_cmpeq_epi8(x.0, y.0) })
    }

    #[inline]
    fn ge(self, x: I8x32, y: I8x32) -> M8 {
        M8(unsafe { _mm256_or_si256(_mm256_cmpgt_epi8(x.0, y.0), _mm256_cmpeq_epi8(x.0, y.0)) })
    }

    #[inline]
    fn gt(self, x: I8x32, y: I8x32) -> M8 {
        M8(unsafe { _mm256_cmpgt_epi8(x.0, y.0) })
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i8) -> I8x32 {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }.into()
    }

    #[inline]
    fn select(self, x: I8x32, y: I8x32, mask: <I8x32 as Simd>::Mask) -> I8x32 {
        unsafe { _mm256_blendv_epi8(y.0, x.0, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: I8x32, ptr: *mut i8) {
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const i8, mask: M8) -> I8x32 {
        // There is no native masked-load instruction for i8, so fall back to
        // scalar loads.
        let mask = _mm256_movemask_epi8(mask.0) as u32;
        let xs: [i8; 32] = std::array::from_fn(|i| {
            let mask_bit = mask & (1 << i);
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
    unsafe fn store_ptr_mask(self, x: I8x32, ptr: *mut i8, mask: M8) {
        // There is no native masked-store instruction for i8, so fall back to
        // scalar store.
        let xs = Simd::to_array(x);
        let mask = _mm256_movemask_epi8(mask.0) as u32;
        for i in 0..32 {
            let mask_bit = mask & (1 << i);
            if mask_bit != 0 {
                // Safety: Caller promises that `ptr.add(i)` is valid if mask[i] is set.
                unsafe { *ptr.add(i) = xs[i] }
            }
        }
    }
}

impl IntOps<i8> for Avx2Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I8x32) -> I8x32 {
        let (x_lo, x_hi) = Extend::<i8>::extend(self, x);

        let i16_ops = self.i16();
        let y_lo = i16_ops.shift_left::<SHIFT>(x_lo);
        let y_hi = i16_ops.shift_left::<SHIFT>(x_hi);

        self.narrow_truncate(y_lo, y_hi)
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: I8x32) -> I8x32 {
        let (x_lo, x_hi) = Extend::<i8>::extend(self, x);

        let i16_ops = self.i16();
        let y_lo = i16_ops.shift_right::<SHIFT>(x_lo);
        let y_hi = i16_ops.shift_right::<SHIFT>(x_hi);

        self.narrow_truncate(y_lo, y_hi)
    }
}

impl SignedIntOps<i8> for Avx2Isa {
    #[inline]
    fn neg(self, x: I8x32) -> I8x32 {
        unsafe { _mm256_sub_epi8(_mm256_setzero_si256(), x.0) }.into()
    }
}

#[inline]
fn interleave_low_x8(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        // AB{N} = Interleaved Nth 64-bit block.
        let lo = _mm256_unpacklo_epi8(a, b); // AB0 AB2
        let hi = _mm256_unpackhi_epi8(a, b); // AB1 AB3
        _mm256_insertf128_si256(lo, _mm256_castsi256_si128(hi), 1) // AB0 AB1
    }
}

#[inline]
fn interleave_high_x8(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        // AB{N} = Interleaved Nth 64-bit block.
        let lo = _mm256_unpacklo_epi8(a, b); // AB0 AB2
        let hi = _mm256_unpackhi_epi8(a, b); // AB1 AB3
        _mm256_permute2x128_si256(lo, hi, 0x31) // AB2 AB3
    }
}

impl Interleave<i8> for Avx2Isa {
    #[inline]
    fn interleave_low(self, a: I8x32, b: I8x32) -> I8x32 {
        interleave_low_x8(a.0, b.0).into()
    }

    #[inline]
    fn interleave_high(self, a: I8x32, b: I8x32) -> I8x32 {
        interleave_high_x8(a.0, b.0).into()
    }
}

unsafe impl NumOps<u8> for Avx2Isa {
    simd_ops_common!(U8x32, M8);
    simd_int_ops_common!(U8x32);

    #[inline]
    fn first_n_mask(self, n: usize) -> M8 {
        let mask: [i8; 32] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        M8(unsafe { _mm256_loadu_si256(mask.as_ptr() as *const __m256i) })
    }

    #[inline]
    fn add(self, x: U8x32, y: U8x32) -> U8x32 {
        unsafe { _mm256_add_epi8(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: U8x32, y: U8x32) -> U8x32 {
        unsafe { _mm256_sub_epi8(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: U8x32, y: U8x32) -> U8x32 {
        let (x_lo, x_hi) = Extend::<u8>::extend(self, x);
        let (y_lo, y_hi) = Extend::<u8>::extend(self, y);

        let u16_ops = self.u16();
        let prod_lo = u16_ops.mul(x_lo, y_lo);
        let prod_hi = u16_ops.mul(x_hi, y_hi);

        self.narrow_truncate(prod_lo, prod_hi)
    }

    #[inline]
    fn splat(self, x: u8) -> U8x32 {
        unsafe { _mm256_set1_epi8(x as i8) }.into()
    }

    #[inline]
    fn eq(self, x: U8x32, y: U8x32) -> M8 {
        M8(unsafe { _mm256_cmpeq_epi8(x.0, y.0) })
    }

    #[inline]
    fn ge(self, x: U8x32, y: U8x32) -> M8 {
        let xy_eq = <Self as NumOps<u8>>::eq(self, x, y);
        let xy_gt = <Self as NumOps<u8>>::gt(self, x, y);
        M8(unsafe { _mm256_or_si256(xy_eq.0, xy_gt.0) })
    }

    #[inline]
    fn gt(self, x: U8x32, y: U8x32) -> M8 {
        // AVX2 lacks u8 comparison. Shift both values to i8 and use signed compare.
        M8(unsafe {
            let mask = _mm256_set1_epi8(0x80u8 as i8);
            let x_i8 = _mm256_xor_si256(x.0, mask);
            let y_i8 = _mm256_xor_si256(y.0, mask);
            _mm256_cmpgt_epi8(x_i8, y_i8)
        })
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const u8) -> U8x32 {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }.into()
    }

    #[inline]
    fn select(self, x: U8x32, y: U8x32, mask: M8) -> U8x32 {
        unsafe { _mm256_blendv_epi8(y.0, x.0, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: U8x32, ptr: *mut u8) {
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const u8, mask: M8) -> U8x32 {
        // There is no native masked-load instruction for u8, so fall back to
        // scalar loads.
        let mask = _mm256_movemask_epi8(mask.0) as u32;
        let xs: [u8; 32] = std::array::from_fn(|i| {
            let mask_bit = mask & (1 << i);
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
    unsafe fn store_ptr_mask(self, x: U8x32, ptr: *mut u8, mask: M8) {
        // There is no native masked-store instruction for u8, so fall back to
        // scalar store.
        let xs = Simd::to_array(x);
        let mask = _mm256_movemask_epi8(mask.0) as u32;
        for i in 0..32 {
            let mask_bit = mask & (1 << i);
            if mask_bit != 0 {
                // Safety: Caller promises that `ptr.add(i)` is valid if mask[i] is set.
                unsafe { *ptr.add(i) = xs[i] }
            }
        }
    }
}

unsafe impl NumOps<u16> for Avx2Isa {
    simd_ops_common!(U16x16, M16);
    simd_int_ops_common!(U16x16);

    #[inline]
    fn first_n_mask(self, n: usize) -> M16 {
        let mask: [i16; 16] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
        M16(unsafe { _mm256_loadu_si256(mask.as_ptr() as *const __m256i) })
    }

    #[inline]
    fn add(self, x: U16x16, y: U16x16) -> U16x16 {
        unsafe { _mm256_add_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn sub(self, x: U16x16, y: U16x16) -> U16x16 {
        unsafe { _mm256_sub_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn mul(self, x: U16x16, y: U16x16) -> U16x16 {
        unsafe { _mm256_mullo_epi16(x.0, y.0) }.into()
    }

    #[inline]
    fn splat(self, x: u16) -> U16x16 {
        unsafe { _mm256_set1_epi16(x as i16) }.into()
    }

    #[inline]
    fn eq(self, x: U16x16, y: U16x16) -> M16 {
        M16(unsafe { _mm256_cmpeq_epi16(x.0, y.0) })
    }

    #[inline]
    fn ge(self, x: U16x16, y: U16x16) -> M16 {
        let xy_eq = <Self as NumOps<u16>>::eq(self, x, y);
        let xy_gt = <Self as NumOps<u16>>::gt(self, x, y);
        M16(unsafe { _mm256_or_si256(xy_eq.0, xy_gt.0) })
    }

    #[inline]
    fn gt(self, x: U16x16, y: U16x16) -> M16 {
        // AVX2 lacks u16 comparison. Shift both values to i16 and use signed compare.
        M16(unsafe {
            let mask = _mm256_set1_epi16(0x8000u16 as i16);
            let x_i16 = _mm256_xor_si256(x.0, mask);
            let y_i16 = _mm256_xor_si256(y.0, mask);
            _mm256_cmpgt_epi16(x_i16, y_i16)
        })
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const u16) -> U16x16 {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }.into()
    }

    #[inline]
    fn select(self, x: U16x16, y: U16x16, mask: M16) -> U16x16 {
        unsafe { _mm256_blendv_epi8(y.0, x.0, mask.0) }.into()
    }

    #[inline]
    unsafe fn store_ptr(self, x: U16x16, ptr: *mut u16) {
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, x.0) }
    }

    #[inline]
    unsafe fn load_ptr_mask(self, ptr: *const u16, mask: M16) -> U16x16 {
        // There is no native masked-load instruction for i16, so fall back to
        // scalar loads.
        let mask = _mm256_movemask_epi8(mask.0) as u32;
        let xs: [u16; 16] = std::array::from_fn(|i| {
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
    unsafe fn store_ptr_mask(self, x: U16x16, ptr: *mut u16, mask: M16) {
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

impl IntOps<u16> for Avx2Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: U16x16) -> U16x16 {
        unsafe { _mm256_slli_epi16(x.0, SHIFT) }.into()
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: U16x16) -> U16x16 {
        unsafe { _mm256_srli_epi16(x.0, SHIFT) }.into()
    }
}

macro_rules! impl_mask {
    ($mask:ident, $elem:ty, $len:expr) => {
        #[derive(Copy, Clone, Debug)]
        #[repr(transparent)]
        pub struct $mask(__m256i);

        impl $mask {
            #[allow(unused)] // Not used for M16/M8
            #[inline]
            fn as_float(self) -> __m256 {
                unsafe { transmute::<__m256i, __m256>(self.0) }
            }

            #[allow(unused)] // Not used for M16/M8
            #[inline]
            fn from_float(m: __m256) -> Self {
                Self(unsafe { transmute::<__m256, __m256i>(m) })
            }
        }

        impl Mask for $mask {
            type Array = [bool; $len];

            #[inline]
            fn to_array(self) -> Self::Array {
                let array = unsafe { transmute::<Self, [$elem; $len]>(self) };
                std::array::from_fn(|i| array[i] != <$elem>::default())
            }
        }
    };
}

impl_mask!(M32, u32, 8);
impl_mask!(M16, u16, 16);
impl_mask!(M8, u8, 32);

macro_rules! impl_mask_ops {
    ($mask:ident) => {
        unsafe impl MaskOps<$mask> for Avx2Isa {
            #[inline]
            fn and(self, x: $mask, y: $mask) -> $mask {
                $mask(unsafe { _mm256_and_si256(x.0, y.0) })
            }

            #[inline]
            fn any(self, x: $mask) -> bool {
                unsafe { _mm256_movemask_epi8(x.0) != 0 }
            }

            #[inline]
            fn all(self, x: $mask) -> bool {
                unsafe { _mm256_movemask_epi8(x.0) == -1 }
            }
        }
    };
}
impl_mask_ops!(M32);
impl_mask_ops!(M16);
impl_mask_ops!(M8);

impl Extend<i16> for Avx2Isa {
    type Output = I32x8;

    #[inline]
    fn extend(self, x: I16x16) -> (Self::Output, Self::Output) {
        unsafe {
            let low = _mm256_castsi256_si128(x.0);
            let high = _mm256_extracti128_si256(x.0, 1);
            (
                _mm256_cvtepi16_epi32(low).into(),
                _mm256_cvtepi16_epi32(high).into(),
            )
        }
    }
}

impl Extend<i8> for Avx2Isa {
    type Output = I16x16;

    #[inline]
    fn extend(self, x: I8x32) -> (Self::Output, Self::Output) {
        unsafe {
            let low = _mm256_castsi256_si128(x.0);
            let high = _mm256_extracti128_si256(x.0, 1);
            (
                _mm256_cvtepi8_epi16(low).into(),
                _mm256_cvtepi8_epi16(high).into(),
            )
        }
    }
}

impl Extend<u8> for Avx2Isa {
    type Output = U16x16;

    #[inline]
    fn extend(self, x: U8x32) -> (Self::Output, Self::Output) {
        unsafe {
            let low = _mm256_castsi256_si128(x.0);
            let high = _mm256_extracti128_si256(x.0, 1);
            (
                _mm256_cvtepu8_epi16(low).into(),
                _mm256_cvtepu8_epi16(high).into(),
            )
        }
    }
}

impl IntOps<u8> for Avx2Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: U8x32) -> U8x32 {
        let (x_lo, x_hi) = Extend::<u8>::extend(self, x);

        let u16_ops = self.u16();
        let y_lo = u16_ops.shift_left::<SHIFT>(x_lo);
        let y_hi = u16_ops.shift_left::<SHIFT>(x_hi);

        self.narrow_truncate(y_lo, y_hi)
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: U8x32) -> U8x32 {
        let (x_lo, x_hi) = Extend::<u8>::extend(self, x);

        let u16_ops = self.u16();
        let y_lo = u16_ops.shift_right::<SHIFT>(x_lo);
        let y_hi = u16_ops.shift_right::<SHIFT>(x_hi);

        self.narrow_truncate(y_lo, y_hi)
    }
}

impl Interleave<u8> for Avx2Isa {
    #[inline]
    fn interleave_low(self, a: U8x32, b: U8x32) -> U8x32 {
        interleave_low_x8(a.0, b.0).into()
    }

    #[inline]
    fn interleave_high(self, a: U8x32, b: U8x32) -> U8x32 {
        interleave_high_x8(a.0, b.0).into()
    }
}

/// Extract bytes at even indices.
///
/// Given an input with 16-bit lanes, this extracts truncated 8-bit values.
#[inline]
unsafe fn extract_even_bytes(vec: __m256i) -> __m128i {
    let lo = _mm256_extracti128_si256(vec, 0);
    let hi = _mm256_extracti128_si256(vec, 1);

    // Shuffle mask that moves bytes at even indices into first half of output.
    // For the second half set the high bit to zero the bytes.
    let mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);

    // Extract even bytes from each half, then concatenate.
    let lo_even = _mm_shuffle_epi8(lo, mask);
    let hi_even = _mm_shuffle_epi8(hi, mask);
    _mm_unpacklo_epi64(lo_even, hi_even)
}

impl Narrow<I16x16> for Avx2Isa {
    type Output = I8x32;

    #[inline]
    fn narrow_truncate(self, low: I16x16, high: I16x16) -> Self::Output {
        let low_even = unsafe { extract_even_bytes(low.0) };
        let high_even = unsafe { extract_even_bytes(high.0) };
        let combined = unsafe { _mm256_setr_m128i(low_even, high_even) };
        I8x32(combined)
    }
}

impl Narrow<U16x16> for Avx2Isa {
    type Output = U8x32;

    #[inline]
    fn narrow_truncate(self, low: U16x16, high: U16x16) -> Self::Output {
        let low_even = unsafe { extract_even_bytes(low.0) };
        let high_even = unsafe { extract_even_bytes(high.0) };
        let combined = unsafe { _mm256_setr_m128i(low_even, high_even) };
        U8x32(combined)
    }
}
