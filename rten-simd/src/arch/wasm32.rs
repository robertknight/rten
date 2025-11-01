use std::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_div, f32x4_eq, f32x4_extract_lane, f32x4_ge, f32x4_gt, f32x4_le,
    f32x4_lt, f32x4_max, f32x4_min, f32x4_mul, f32x4_nearest, f32x4_neg, f32x4_splat, f32x4_sub,
    i8x16_add, i8x16_all_true, i8x16_eq, i8x16_ge, i8x16_gt, i8x16_neg, i8x16_shl, i8x16_shr_s,
    i8x16_shuffle, i8x16_splat, i8x16_sub, i16x8_add, i16x8_all_true, i16x8_eq,
    i16x8_extend_high_i8x16, i16x8_extend_low_i8x16, i16x8_extmul_high_i8x16,
    i16x8_extmul_low_i8x16, i16x8_ge, i16x8_gt, i16x8_mul, i16x8_narrow_i32x4, i16x8_neg,
    i16x8_shl, i16x8_shr_s, i16x8_shuffle, i16x8_splat, i16x8_sub, i32x4_add, i32x4_all_true,
    i32x4_eq, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, i32x4_ge, i32x4_gt, i32x4_mul,
    i32x4_neg, i32x4_shl, i32x4_shr_s, i32x4_shuffle, i32x4_splat, i32x4_sub,
    i32x4_trunc_sat_f32x4, u8x16_add, u8x16_eq, u8x16_ge, u8x16_gt, u8x16_narrow_i16x8,
    u8x16_shuffle, u8x16_splat, u8x16_sub, u16x8_add, u16x8_eq, u16x8_extend_high_u8x16,
    u16x8_extend_low_u8x16, u16x8_extmul_high_u8x16, u16x8_extmul_low_u8x16, u16x8_ge, u16x8_gt,
    u16x8_mul, u16x8_shl, u16x8_shr_u, u16x8_splat, u16x8_sub, v128, v128_and, v128_any_true,
    v128_bitselect, v128_load, v128_not, v128_or, v128_store, v128_xor,
};
use std::mem::transmute;

#[cfg(target_feature = "relaxed-simd")]
use std::arch::wasm32::f32x4_relaxed_madd;

use super::{lanes, simd_type};
use crate::ops::{
    Concat, Extend, FloatOps, IntOps, Interleave, MaskOps, NarrowSaturate, NumOps, SignedIntOps,
};
use crate::{Isa, Mask, Simd};

simd_type!(F32x4, v128, f32, M32, Wasm32Isa);
simd_type!(I32x4, v128, i32, M32, Wasm32Isa);
simd_type!(I16x8, v128, i16, M16, Wasm32Isa);
simd_type!(I8x16, v128, i8, M8, Wasm32Isa);
simd_type!(U8x16, v128, u8, M8, Wasm32Isa);
simd_type!(U16x8, v128, u16, M16, Wasm32Isa);
simd_type!(U32x4, v128, u32, M32, Wasm32Isa);

#[derive(Copy, Clone)]
pub struct Wasm32Isa {
    _private: (),
}

impl Wasm32Isa {
    pub fn new() -> Option<Self> {
        Some(Wasm32Isa { _private: () })
    }
}

// Safety: This module is only compiled if WASM SIMD is enabled at compile
// time, hence this module can treat SIMD as always-available.
unsafe impl Isa for Wasm32Isa {
    type M32 = M32;
    type M16 = M16;
    type M8 = M8;
    type F32 = F32x4;
    type I32 = I32x4;
    type I16 = I16x8;
    type I8 = I8x16;
    type U8 = U8x16;
    type U16 = U16x8;
    type U32 = U32x4;
    type Bits = I32x4;

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

    fn u8(self) -> impl IntOps<u8, Simd = Self::U8> + Extend<u8, Output = Self::U16> {
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
    ($simd:ident, $mask:ident, $mask_elem:ty) => {
        type Simd = $simd;

        #[inline]
        fn len(self) -> usize {
            lanes::<$simd>()
        }

        #[inline]
        fn first_n_mask(self, n: usize) -> $mask {
            let mask: [$mask_elem; lanes::<$simd>()] =
                std::array::from_fn(|i| if i < n { !0 } else { 0 });
            $mask(unsafe { v128_load(mask.as_ptr() as *const v128) })
        }

        #[inline]
        unsafe fn load_ptr(self, ptr: *const <$simd as Simd>::Elem) -> $simd {
            $simd(unsafe { v128_load(ptr as *const v128) })
        }

        #[inline]
        unsafe fn store_ptr(self, x: $simd, ptr: *mut <$simd as Simd>::Elem) {
            unsafe { v128_store(ptr as *mut v128, x.0) }
        }

        #[inline]
        unsafe fn load_ptr_mask(self, ptr: *const <$simd as Simd>::Elem, mask: $mask) -> $simd {
            type Elem = <$simd as Simd>::Elem;
            let mask_array = Mask::to_array(mask);
            let mut vec = Simd::to_array(NumOps::<Elem>::zero(self));
            for i in 0..mask_array.len() {
                if mask_array[i] {
                    vec[i] = *ptr.add(i);
                }
            }
            self.load_ptr(vec.as_ref().as_ptr())
        }

        #[inline]
        unsafe fn store_ptr_mask(
            self,
            x: $simd,
            ptr: *mut <$simd as Simd>::Elem,
            mask: <$simd as Simd>::Mask,
        ) {
            type Elem = <$simd as Simd>::Elem;

            let mask_array = Mask::to_array(mask);
            let x_array = Simd::to_array(x);
            for i in 0..NumOps::<Elem>::len(self) {
                if mask_array[i] {
                    *ptr.add(i) = x_array[i];
                }
            }
        }

        #[inline]
        fn select(self, x: $simd, y: $simd, mask: <$simd as Simd>::Mask) -> $simd {
            $simd(v128_bitselect(x.0, y.0, mask.0))
        }

        #[inline]
        fn and(self, x: $simd, y: $simd) -> $simd {
            v128_and(x.0, y.0).into()
        }

        #[inline]
        fn not(self, x: $simd) -> $simd {
            v128_not(x.0).into()
        }

        #[inline]
        fn or(self, x: $simd, y: $simd) -> $simd {
            v128_or(x.0, y.0).into()
        }

        #[inline]
        fn xor(self, x: $simd, y: $simd) -> $simd {
            v128_xor(x.0, y.0).into()
        }
    };
}

unsafe impl NumOps<f32> for Wasm32Isa {
    simd_ops_common!(F32x4, M32, i32);

    #[inline]
    fn add(self, x: F32x4, y: F32x4) -> F32x4 {
        F32x4(f32x4_add(x.0, y.0))
    }

    #[inline]
    fn sub(self, x: F32x4, y: F32x4) -> F32x4 {
        F32x4(f32x4_sub(x.0, y.0))
    }

    #[inline]
    fn mul(self, x: F32x4, y: F32x4) -> F32x4 {
        F32x4(f32x4_mul(x.0, y.0))
    }

    #[inline]
    fn mul_add(self, a: F32x4, b: F32x4, c: F32x4) -> F32x4 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            F32x4(f32x4_relaxed_madd(a.0, b.0, c.0))
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            F32x4(f32x4_add(f32x4_mul(a.0, b.0), c.0))
        }
    }

    #[inline]
    fn lt(self, x: F32x4, y: F32x4) -> M32 {
        M32(f32x4_lt(x.0, y.0))
    }

    #[inline]
    fn le(self, x: F32x4, y: F32x4) -> M32 {
        M32(f32x4_le(x.0, y.0))
    }

    #[inline]
    fn eq(self, x: F32x4, y: F32x4) -> M32 {
        M32(f32x4_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: F32x4, y: F32x4) -> M32 {
        M32(f32x4_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: F32x4, y: F32x4) -> M32 {
        M32(f32x4_gt(x.0, y.0))
    }

    #[inline]
    fn min(self, x: F32x4, y: F32x4) -> F32x4 {
        F32x4(f32x4_min(x.0, y.0))
    }

    #[inline]
    fn max(self, x: F32x4, y: F32x4) -> F32x4 {
        F32x4(f32x4_max(x.0, y.0))
    }

    #[inline]
    fn splat(self, x: f32) -> F32x4 {
        F32x4(f32x4_splat(x))
    }

    #[inline]
    fn sum(self, x: F32x4) -> f32 {
        // See https://github.com/WebAssembly/simd/issues/20.
        let lo_2 = x.0;
        let hi_2 = i32x4_shuffle::<2, 3, 0, 0>(x.0, x.0);
        let sum_2 = f32x4_add(lo_2, hi_2);
        let lo = sum_2;
        let hi = i32x4_shuffle::<1, 0, 0, 0>(sum_2, sum_2);
        let sum = f32x4_add(lo, hi);
        f32x4_extract_lane::<0>(sum)
    }
}

impl FloatOps<f32> for Wasm32Isa {
    type Int = <Self as Isa>::I32;

    #[inline]
    fn div(self, x: F32x4, y: F32x4) -> F32x4 {
        F32x4(f32x4_div(x.0, y.0))
    }

    #[inline]
    fn neg(self, x: F32x4) -> F32x4 {
        F32x4(f32x4_neg(x.0))
    }

    #[inline]
    fn abs(self, x: F32x4) -> F32x4 {
        F32x4(f32x4_abs(x.0))
    }

    #[inline]
    fn mul_sub_from(self, a: F32x4, b: F32x4, c: F32x4) -> F32x4 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            F32x4(f32x4_relaxed_nmadd(a.0, b.0, c.0))
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            F32x4(f32x4_sub(c.0, f32x4_mul(a.0, b.0)))
        }
    }

    #[inline]
    fn round_ties_even(self, x: F32x4) -> F32x4 {
        F32x4(f32x4_nearest(x.0))
    }

    #[inline]
    fn to_int_trunc(self, x: F32x4) -> Self::Int {
        I32x4(i32x4_trunc_sat_f32x4(x.0))
    }

    #[inline]
    fn to_int_round(self, x: F32x4) -> Self::Int {
        I32x4(i32x4_trunc_sat_f32x4(f32x4_nearest(x.0)))
    }
}

unsafe impl NumOps<i32> for Wasm32Isa {
    simd_ops_common!(I32x4, M32, i32);

    #[inline]
    fn add(self, x: I32x4, y: I32x4) -> I32x4 {
        I32x4(i32x4_add(x.0, y.0))
    }

    #[inline]
    fn sub(self, x: I32x4, y: I32x4) -> I32x4 {
        I32x4(i32x4_sub(x.0, y.0))
    }

    #[inline]
    fn mul(self, x: I32x4, y: I32x4) -> I32x4 {
        I32x4(i32x4_mul(x.0, y.0))
    }

    #[inline]
    fn splat(self, x: i32) -> I32x4 {
        I32x4(i32x4_splat(x))
    }

    #[inline]
    fn eq(self, x: I32x4, y: I32x4) -> M32 {
        M32(i32x4_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: I32x4, y: I32x4) -> M32 {
        M32(i32x4_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: I32x4, y: I32x4) -> M32 {
        M32(i32x4_gt(x.0, y.0))
    }
}

impl IntOps<i32> for Wasm32Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I32x4) -> I32x4 {
        I32x4(i32x4_shl(x.0, SHIFT as u32))
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: I32x4) -> I32x4 {
        I32x4(i32x4_shr_s(x.0, SHIFT as u32))
    }
}

impl SignedIntOps<i32> for Wasm32Isa {
    #[inline]
    fn neg(self, x: I32x4) -> I32x4 {
        I32x4(i32x4_neg(x.0))
    }
}

impl NarrowSaturate<i32, i16> for Wasm32Isa {
    type Output = I16x8;

    #[inline]
    fn narrow_saturate(self, low: I32x4, high: I32x4) -> I16x8 {
        I16x8(i16x8_narrow_i32x4(low.0, high.0))
    }
}

impl Concat<i32> for Wasm32Isa {
    #[inline]
    fn concat_low(self, a: I32x4, b: I32x4) -> I32x4 {
        I32x4(i32x4_shuffle::<0, 1, 4, 5>(a.0, b.0))
    }

    #[inline]
    fn concat_high(self, a: I32x4, b: I32x4) -> I32x4 {
        I32x4(i32x4_shuffle::<2, 3, 6, 7>(a.0, b.0))
    }
}

unsafe impl NumOps<i16> for Wasm32Isa {
    simd_ops_common!(I16x8, M16, i16);

    #[inline]
    fn add(self, x: I16x8, y: I16x8) -> I16x8 {
        I16x8(i16x8_add(x.0, y.0))
    }

    #[inline]
    fn sub(self, x: I16x8, y: I16x8) -> I16x8 {
        I16x8(i16x8_sub(x.0, y.0))
    }

    #[inline]
    fn mul(self, x: I16x8, y: I16x8) -> I16x8 {
        I16x8(i16x8_mul(x.0, y.0))
    }

    #[inline]
    fn splat(self, x: i16) -> I16x8 {
        I16x8(i16x8_splat(x))
    }

    #[inline]
    fn eq(self, x: I16x8, y: I16x8) -> M16 {
        M16(i16x8_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: I16x8, y: I16x8) -> M16 {
        M16(i16x8_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: I16x8, y: I16x8) -> M16 {
        M16(i16x8_gt(x.0, y.0))
    }
}

impl IntOps<i16> for Wasm32Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I16x8) -> I16x8 {
        I16x8(i16x8_shl(x.0, SHIFT as u32))
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: I16x8) -> I16x8 {
        I16x8(i16x8_shr_s(x.0, SHIFT as u32))
    }
}

impl SignedIntOps<i16> for Wasm32Isa {
    #[inline]
    fn neg(self, x: I16x8) -> I16x8 {
        I16x8(i16x8_neg(x.0))
    }
}

impl Extend<i16> for Wasm32Isa {
    type Output = I32x4;

    #[inline]
    fn extend(self, x: I16x8) -> (I32x4, I32x4) {
        let low = i32x4_extend_low_i16x8(x.0);
        let high = i32x4_extend_high_i16x8(x.0);
        (low.into(), high.into())
    }
}

impl Interleave<i16> for Wasm32Isa {
    #[inline]
    fn interleave_low(self, a: I16x8, b: I16x8) -> I16x8 {
        i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a.0, b.0).into()
    }

    #[inline]
    fn interleave_high(self, a: I16x8, b: I16x8) -> I16x8 {
        i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a.0, b.0).into()
    }
}

impl NarrowSaturate<i16, u8> for Wasm32Isa {
    type Output = U8x16;

    #[inline]
    fn narrow_saturate(self, low: I16x8, high: I16x8) -> U8x16 {
        U8x16(u8x16_narrow_i16x8(low.0, high.0))
    }
}

unsafe impl NumOps<i8> for Wasm32Isa {
    simd_ops_common!(I8x16, M8, i8);

    #[inline]
    fn add(self, x: I8x16, y: I8x16) -> I8x16 {
        I8x16(i8x16_add(x.0, y.0))
    }

    #[inline]
    fn sub(self, x: I8x16, y: I8x16) -> I8x16 {
        I8x16(i8x16_sub(x.0, y.0))
    }

    #[inline]
    fn mul(self, x: I8x16, y: I8x16) -> I8x16 {
        let prod_low = i16x8_extmul_low_i8x16(x.0, y.0);
        let prod_high = i16x8_extmul_high_i8x16(x.0, y.0);

        // Select even bytes from low and high products. This obtains the
        // i8 truncated product.
        let prod_i8 = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(
            prod_low, prod_high,
        );

        I8x16(prod_i8)
    }

    #[inline]
    fn splat(self, x: i8) -> I8x16 {
        I8x16(i8x16_splat(x))
    }

    #[inline]
    fn eq(self, x: I8x16, y: I8x16) -> M8 {
        M8(i8x16_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: I8x16, y: I8x16) -> M8 {
        M8(i8x16_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: I8x16, y: I8x16) -> M8 {
        M8(i8x16_gt(x.0, y.0))
    }
}

impl IntOps<i8> for Wasm32Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I8x16) -> I8x16 {
        I8x16(i8x16_shl(x.0, SHIFT as u32))
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: I8x16) -> I8x16 {
        I8x16(i8x16_shr_s(x.0, SHIFT as u32))
    }
}

impl SignedIntOps<i8> for Wasm32Isa {
    #[inline]
    fn neg(self, x: I8x16) -> I8x16 {
        I8x16(i8x16_neg(x.0))
    }
}

impl Extend<i8> for Wasm32Isa {
    type Output = I16x8;

    #[inline]
    fn extend(self, x: I8x16) -> (I16x8, I16x8) {
        let low = i16x8_extend_low_i8x16(x.0);
        let high = i16x8_extend_high_i8x16(x.0);
        (low.into(), high.into())
    }
}

impl Interleave<i8> for Wasm32Isa {
    #[inline]
    fn interleave_low(self, a: I8x16, b: I8x16) -> I8x16 {
        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a.0, b.0).into()
    }

    #[inline]
    fn interleave_high(self, a: I8x16, b: I8x16) -> I8x16 {
        i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a.0, b.0)
            .into()
    }
}

unsafe impl NumOps<u8> for Wasm32Isa {
    simd_ops_common!(U8x16, M8, i8);

    #[inline]
    fn add(self, x: U8x16, y: U8x16) -> U8x16 {
        U8x16(u8x16_add(x.0, y.0))
    }

    #[inline]
    fn sub(self, x: U8x16, y: U8x16) -> U8x16 {
        U8x16(u8x16_sub(x.0, y.0))
    }

    #[inline]
    fn mul(self, x: U8x16, y: U8x16) -> U8x16 {
        let prod_low = u16x8_extmul_low_u8x16(x.0, y.0);
        let prod_high = u16x8_extmul_high_u8x16(x.0, y.0);

        // Select even bytes from low and high products. This obtains the
        // u8 truncated product.
        let prod_u8 = u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(
            prod_low, prod_high,
        );

        U8x16(prod_u8)
    }

    #[inline]
    fn splat(self, x: u8) -> U8x16 {
        U8x16(u8x16_splat(x))
    }

    #[inline]
    fn eq(self, x: U8x16, y: U8x16) -> M8 {
        M8(u8x16_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: U8x16, y: U8x16) -> M8 {
        M8(u8x16_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: U8x16, y: U8x16) -> M8 {
        M8(u8x16_gt(x.0, y.0))
    }
}

unsafe impl NumOps<u16> for Wasm32Isa {
    simd_ops_common!(U16x8, M16, u16);

    #[inline]
    fn add(self, x: U16x8, y: U16x8) -> U16x8 {
        U16x8(u16x8_add(x.0, y.0))
    }

    #[inline]
    fn sub(self, x: U16x8, y: U16x8) -> U16x8 {
        U16x8(u16x8_sub(x.0, y.0))
    }

    #[inline]
    fn mul(self, x: U16x8, y: U16x8) -> U16x8 {
        U16x8(u16x8_mul(x.0, y.0))
    }

    #[inline]
    fn splat(self, x: u16) -> U16x8 {
        U16x8(u16x8_splat(x))
    }

    #[inline]
    fn eq(self, x: U16x8, y: U16x8) -> M16 {
        M16(u16x8_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: U16x8, y: U16x8) -> M16 {
        M16(u16x8_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: U16x8, y: U16x8) -> M16 {
        M16(u16x8_gt(x.0, y.0))
    }
}

impl IntOps<u16> for Wasm32Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: U16x8) -> U16x8 {
        U16x8(u16x8_shl(x.0, SHIFT as u32))
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: U16x8) -> U16x8 {
        U16x8(u16x8_shr_u(x.0, SHIFT as u32))
    }
}

impl Extend<u8> for Wasm32Isa {
    type Output = U16x8;

    #[inline]
    fn extend(self, x: U8x16) -> (U16x8, U16x8) {
        let low = u16x8_extend_low_u8x16(x.0);
        let high = u16x8_extend_high_u8x16(x.0);
        (low.into(), high.into())
    }
}

impl IntOps<u8> for Wasm32Isa {
    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: U8x16) -> U8x16 {
        U8x16(x.to_array().map(|x| x << SHIFT).into())
    }

    #[inline]
    fn shift_right<const SHIFT: i32>(self, x: U8x16) -> U8x16 {
        U8x16(x.to_array().map(|x| x >> SHIFT).into())
    }
}

macro_rules! mask_type {
    ($mask:ident, $elem:ty, $len: expr, $all_true_fn:ident) => {
        #[derive(Copy, Clone, Debug)]
        #[repr(transparent)]
        pub struct $mask(v128);

        impl Mask for $mask {
            type Array = [bool; $len];

            #[inline]
            fn to_array(self) -> Self::Array {
                let array = unsafe { transmute::<Self, [$elem; $len]>(self) };
                std::array::from_fn(|i| array[i] != 0)
            }
        }

        unsafe impl MaskOps<$mask> for Wasm32Isa {
            #[inline]
            fn and(self, x: $mask, y: $mask) -> $mask {
                $mask(v128_and(x.0, y.0))
            }

            #[inline]
            fn any(self, x: $mask) -> bool {
                unsafe { v128_any_true(x.0) }
            }

            #[inline]
            fn all(self, x: $mask) -> bool {
                unsafe { $all_true_fn(x.0) }
            }
        }
    };
}

// Define mask vector types. `Mn` is a mask for a vector with n-bit lanes.
mask_type!(M32, i32, 4, i32x4_all_true);
mask_type!(M16, i16, 8, i16x8_all_true);
mask_type!(M8, i8, 16, i8x16_all_true);
