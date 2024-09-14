use std::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_div, f32x4_extract_lane, f32x4_ge, f32x4_le, f32x4_lt, f32x4_max,
    f32x4_mul, f32x4_splat, f32x4_sub, i32x4_add, i32x4_eq, i32x4_ge, i32x4_gt, i32x4_le, i32x4_lt,
    i32x4_shl, i32x4_shuffle, i32x4_splat, i32x4_sub, i32x4_trunc_sat_f32x4, v128, v128_and,
    v128_bitselect, v128_load, v128_store,
};

use crate::{SimdFloat, SimdInt, SimdMask, SimdVal};

/// Wrapper around a WASM v128 type that marks it as containing integers.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct v128i(v128);

/// Wrapper around a WASM v128 type that marks it as containing floats.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct v128f(v128);

impl SimdMask for v128i {
    type Array = [bool; 4];

    #[inline]
    unsafe fn and(self, other: Self) -> Self {
        Self(v128_and(self.0, other.0))
    }

    #[inline]
    unsafe fn to_array(self) -> Self::Array {
        let mut array = [0; Self::LEN];
        self.store(array.as_mut_ptr());
        std::array::from_fn(|i| array[i] != 0)
    }
}

impl SimdVal for v128i {
    const LEN: usize = 4;

    type Mask = v128i;
}

impl SimdInt for v128i {
    type Array = [i32; 4];
    type Float = v128f;

    #[inline]
    unsafe fn splat(val: i32) -> Self {
        Self(i32x4_splat(val))
    }

    #[inline]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        Self(i32x4_gt(self.0, other.0))
    }

    #[inline]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        Self(i32x4_lt(self.0, other.0))
    }

    #[inline]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        Self(i32x4_eq(self.0, other.0))
    }

    #[inline]
    unsafe fn le(self, other: Self) -> Self::Mask {
        Self(i32x4_le(self.0, other.0))
    }

    #[inline]
    unsafe fn ge(self, other: Self) -> Self::Mask {
        Self(i32x4_ge(self.0, other.0))
    }

    #[inline]
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
        Self(v128_bitselect(other.0, self.0, mask.0))
    }

    #[inline]
    unsafe fn add(self, rhs: Self) -> Self {
        Self(i32x4_add(self.0, rhs.0))
    }

    #[inline]
    unsafe fn sub(self, rhs: Self) -> Self {
        Self(i32x4_sub(self.0, rhs.0))
    }

    #[inline]
    unsafe fn shl<const COUNT: i32>(self) -> Self {
        Self(i32x4_shl(self.0, COUNT as u32))
    }

    #[inline]
    unsafe fn reinterpret_as_float(self) -> Self::Float {
        v128f(self.0)
    }

    #[inline]
    unsafe fn load(ptr: *const i32) -> Self {
        Self(v128_load(ptr as *const v128))
    }

    #[inline]
    unsafe fn store(self, ptr: *mut i32) {
        v128_store(ptr as *mut v128, self.0)
    }

    #[inline]
    unsafe fn to_array(self) -> Self::Array {
        let mut array = [0; Self::LEN];
        self.store(array.as_mut_ptr());
        array
    }
}

impl SimdVal for v128f {
    const LEN: usize = 4;

    type Mask = v128i;
}

impl SimdFloat for v128f {
    type Int = v128i;

    #[inline]
    unsafe fn splat(val: f32) -> Self {
        Self(f32x4_splat(val))
    }

    #[inline]
    unsafe fn abs(self) -> Self {
        Self(f32x4_abs(self.0))
    }

    #[inline]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        Self(f32x4_add(f32x4_mul(self.0, a.0), b.0))
    }

    #[inline]
    unsafe fn sub(self, rhs: Self) -> Self {
        Self(f32x4_sub(self.0, rhs.0))
    }

    #[inline]
    unsafe fn add(self, rhs: Self) -> Self {
        Self(f32x4_add(self.0, rhs.0))
    }

    #[inline]
    unsafe fn to_int_trunc(self) -> Self::Int {
        v128i(i32x4_trunc_sat_f32x4(self.0))
    }

    #[inline]
    unsafe fn mul(self, rhs: Self) -> Self {
        Self(f32x4_mul(self.0, rhs.0))
    }

    #[inline]
    unsafe fn div(self, rhs: Self) -> Self {
        Self(f32x4_div(self.0, rhs.0))
    }

    #[inline]
    unsafe fn ge(self, rhs: Self) -> Self::Mask {
        v128i(f32x4_ge(self.0, rhs.0))
    }

    #[inline]
    unsafe fn le(self, rhs: Self) -> Self::Mask {
        v128i(f32x4_le(self.0, rhs.0))
    }

    #[inline]
    unsafe fn lt(self, rhs: Self) -> Self::Mask {
        v128i(f32x4_lt(self.0, rhs.0))
    }

    #[inline]
    unsafe fn max(self, rhs: Self) -> Self {
        Self(f32x4_max(self.0, rhs.0))
    }

    #[inline]
    unsafe fn blend(self, rhs: Self, mask: Self::Mask) -> Self {
        Self(v128_bitselect(rhs.0, self.0, mask.0))
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        Self(v128_load(ptr as *const v128))
    }

    #[inline]
    unsafe fn store(self, ptr: *mut f32) {
        v128_store(ptr as *mut v128, self.0)
    }

    #[inline]
    unsafe fn gather_mask(src: *const f32, offsets: Self::Int, mask: Self::Mask) -> Self {
        super::simd_gather_mask::<Self, { Self::LEN }>(src, offsets, mask)
    }

    #[inline]
    unsafe fn sum(self) -> f32 {
        // See https://github.com/WebAssembly/simd/issues/20.
        let lo_2 = self.0;
        let hi_2 = i32x4_shuffle::<2, 3, 0, 0>(self.0, self.0);
        let sum_2 = f32x4_add(lo_2, hi_2);
        let lo = sum_2;
        let hi = i32x4_shuffle::<1, 0, 0, 0>(sum_2, sum_2);
        let sum = f32x4_add(lo, hi);
        f32x4_extract_lane::<0>(sum)
    }
}
