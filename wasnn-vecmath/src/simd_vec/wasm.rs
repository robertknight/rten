use std::arch::wasm32::{
    f32x4_add, f32x4_div, f32x4_ge, f32x4_le, f32x4_lt, f32x4_mul, f32x4_splat, f32x4_sub,
    i32x4_add, i32x4_gt, i32x4_shl, i32x4_splat, i32x4_sub, i32x4_trunc_sat_f32x4, v128,
    v128_bitselect, v128_load, v128_store,
};

use crate::simd_vec::{SimdFloat, SimdInt};

/// Wrapper around a WASM v128 type that marks it as containing integers.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct v128i(v128);

/// Wrapper around a WASM v128 type that marks it as containing floats.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct v128f(v128);

impl SimdInt for v128i {
    type Float = v128f;

    const LEN: usize = 4;

    unsafe fn splat(val: i32) -> Self {
        Self(i32x4_splat(val))
    }

    unsafe fn gt(self, other: Self) -> Self {
        Self(i32x4_gt(self.0, other.0))
    }

    unsafe fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(other.0, self.0, mask.0))
    }

    unsafe fn add(self, rhs: Self) -> Self {
        Self(i32x4_add(self.0, rhs.0))
    }

    unsafe fn sub(self, rhs: Self) -> Self {
        Self(i32x4_sub(self.0, rhs.0))
    }

    unsafe fn shl<const COUNT: i32>(self) -> Self {
        Self(i32x4_shl(self.0, COUNT as u32))
    }

    unsafe fn reinterpret_as_float(self) -> Self::Float {
        v128f(self.0)
    }

    unsafe fn load(ptr: *const i32) -> Self {
        Self(v128_load(ptr as *const v128))
    }

    unsafe fn store(self, ptr: *mut i32) {
        v128_store(ptr as *mut v128, self.0)
    }
}

impl SimdFloat for v128f {
    type Int = v128i;

    const LEN: usize = 4;

    unsafe fn splat(val: f32) -> Self {
        Self(f32x4_splat(val))
    }

    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        Self(f32x4_add(f32x4_mul(self.0, a.0), b.0))
    }

    unsafe fn sub(self, rhs: Self) -> Self {
        Self(f32x4_sub(self.0, rhs.0))
    }

    unsafe fn add(self, rhs: Self) -> Self {
        Self(f32x4_add(self.0, rhs.0))
    }

    unsafe fn to_int_trunc(self) -> Self::Int {
        v128i(i32x4_trunc_sat_f32x4(self.0))
    }

    unsafe fn mul(self, rhs: Self) -> Self {
        Self(f32x4_mul(self.0, rhs.0))
    }

    unsafe fn div(self, rhs: Self) -> Self {
        Self(f32x4_div(self.0, rhs.0))
    }

    unsafe fn ge(self, rhs: Self) -> Self {
        Self(f32x4_ge(self.0, rhs.0))
    }

    unsafe fn le(self, rhs: Self) -> Self {
        Self(f32x4_le(self.0, rhs.0))
    }

    unsafe fn lt(self, rhs: Self) -> Self {
        Self(f32x4_lt(self.0, rhs.0))
    }

    unsafe fn blend(self, rhs: Self, mask: Self) -> Self {
        Self(v128_bitselect(rhs.0, self.0, mask.0))
    }

    unsafe fn load(ptr: *const f32) -> Self {
        Self(v128_load(ptr as *const v128))
    }

    unsafe fn store(self, ptr: *mut f32) {
        v128_store(ptr as *mut v128, self.0)
    }
}
