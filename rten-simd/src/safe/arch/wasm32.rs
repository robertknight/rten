use std::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_div, f32x4_eq, f32x4_extract_lane, f32x4_ge, f32x4_gt, f32x4_le,
    f32x4_lt, f32x4_max, f32x4_min, f32x4_mul, f32x4_nearest, f32x4_neg, f32x4_splat, f32x4_sub,
    i16x8_add, i16x8_eq, i16x8_ge, i16x8_gt, i16x8_mul, i16x8_neg, i16x8_shl, i16x8_splat,
    i16x8_sub, i32x4_add, i32x4_eq, i32x4_ge, i32x4_gt, i32x4_mul, i32x4_neg, i32x4_shl,
    i32x4_shuffle, i32x4_splat, i32x4_sub, i32x4_trunc_sat_f32x4, v128, v128_and, v128_bitselect,
    v128_load, v128_store,
};
use std::mem::transmute;

use super::{lanes, simd_type};
use crate::safe::{Isa, Mask, MaskOps, Simd, SimdFloatOps, SimdIntOps, SimdOps};

simd_type!(F32x4, v128, f32, I32x4, Wasm32Isa);
simd_type!(I32x4, v128, i32, I32x4, Wasm32Isa);
simd_type!(I16x8, v128, i16, I16x8, Wasm32Isa);

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
    type F32 = F32x4;
    type I32 = I32x4;
    type I16 = I16x8;
    type Bits = I32x4;

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
    ($simd:ident, $mask:ident, $mask_elem:ty) => {
        #[inline]
        fn mask_ops(self) -> impl MaskOps<<$simd as Simd>::Mask> {
            self
        }

        #[inline]
        fn len(self) -> usize {
            lanes::<$simd>()
        }

        #[inline]
        fn first_n_mask(self, n: usize) -> $mask {
            let mask: [$mask_elem; lanes::<$simd>()] =
                std::array::from_fn(|i| if i < n { -1 } else { 0 });
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
            let mask_array = Mask::to_array(mask);
            let mut vec = Simd::to_array(<Self as SimdOps<$simd>>::zero(self));
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
            let mask_array = Mask::to_array(mask);
            let x_array = Simd::to_array(x);
            for i in 0..<Self as SimdOps<$simd>>::len(self) {
                if mask_array[i] {
                    *ptr.add(i) = x_array[i];
                }
            }
        }

        #[inline]
        fn select(self, x: $simd, y: $simd, mask: <$simd as Simd>::Mask) -> $simd {
            $simd(v128_bitselect(x.0, y.0, mask.0))
        }
    };
}

unsafe impl SimdOps<F32x4> for Wasm32Isa {
    simd_ops_common!(F32x4, I32x4, i32);

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
    fn lt(self, x: F32x4, y: F32x4) -> I32x4 {
        I32x4(f32x4_lt(x.0, y.0))
    }

    #[inline]
    fn le(self, x: F32x4, y: F32x4) -> I32x4 {
        I32x4(f32x4_le(x.0, y.0))
    }

    #[inline]
    fn eq(self, x: F32x4, y: F32x4) -> I32x4 {
        I32x4(f32x4_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: F32x4, y: F32x4) -> I32x4 {
        I32x4(f32x4_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: F32x4, y: F32x4) -> I32x4 {
        I32x4(f32x4_gt(x.0, y.0))
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

impl SimdFloatOps<F32x4> for Wasm32Isa {
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
    fn to_int_trunc(self, x: F32x4) -> Self::Int {
        I32x4(i32x4_trunc_sat_f32x4(x.0))
    }

    #[inline]
    fn to_int_round(self, x: F32x4) -> Self::Int {
        I32x4(i32x4_trunc_sat_f32x4(f32x4_nearest(x.0)))
    }
}

unsafe impl SimdOps<I32x4> for Wasm32Isa {
    simd_ops_common!(I32x4, I32x4, i32);

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
    fn eq(self, x: I32x4, y: I32x4) -> I32x4 {
        I32x4(i32x4_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: I32x4, y: I32x4) -> I32x4 {
        I32x4(i32x4_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: I32x4, y: I32x4) -> I32x4 {
        I32x4(i32x4_gt(x.0, y.0))
    }
}

impl SimdIntOps<I32x4> for Wasm32Isa {
    #[inline]
    fn neg(self, x: I32x4) -> I32x4 {
        I32x4(i32x4_neg(x.0))
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I32x4) -> I32x4 {
        I32x4(i32x4_shl(x.0, SHIFT as u32))
    }
}

unsafe impl SimdOps<I16x8> for Wasm32Isa {
    simd_ops_common!(I16x8, I16x8, i16);

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
    fn eq(self, x: I16x8, y: I16x8) -> I16x8 {
        I16x8(i16x8_eq(x.0, y.0))
    }

    #[inline]
    fn ge(self, x: I16x8, y: I16x8) -> I16x8 {
        I16x8(i16x8_ge(x.0, y.0))
    }

    #[inline]
    fn gt(self, x: I16x8, y: I16x8) -> I16x8 {
        I16x8(i16x8_gt(x.0, y.0))
    }
}

impl SimdIntOps<I16x8> for Wasm32Isa {
    #[inline]
    fn neg(self, x: I16x8) -> I16x8 {
        I16x8(i16x8_neg(x.0))
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I16x8) -> I16x8 {
        I16x8(i16x8_shl(x.0, SHIFT as u32))
    }
}

macro_rules! mask_type {
    ($mask:ident, $elem:ty, $len: expr) => {
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
        }
    };
}

mask_type!(I32x4, i32, 4);
mask_type!(I16x8, i16, 8);
