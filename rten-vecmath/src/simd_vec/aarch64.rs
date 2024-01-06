use std::arch::aarch64::{
    float32x4_t, int32x4_t, uint32x4_t, vabsq_f32, vaddq_f32, vaddq_s32, vbslq_f32, vbslq_s32,
    vcgeq_f32, vcgtq_s32, vcleq_f32, vcltq_f32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s32,
    vfmaq_f32, vld1q_f32, vld1q_s32, vmaxq_f32, vmulq_f32, vreinterpretq_f32_s32, vshlq_n_s32,
    vst1q_f32, vst1q_s32, vsubq_f32, vsubq_s32,
};

use crate::simd_vec::{SimdFloat, SimdInt};

impl SimdInt for int32x4_t {
    type Float = float32x4_t;
    type Mask = uint32x4_t;

    const LEN: usize = 4;

    unsafe fn zero() -> Self {
        vdupq_n_s32(0)
    }

    unsafe fn splat(val: i32) -> Self {
        vdupq_n_s32(val)
    }

    unsafe fn gt(self, other: Self) -> Self::Mask {
        vcgtq_s32(self, other)
    }

    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
        vbslq_s32(mask, other, self)
    }

    unsafe fn add(self, rhs: Self) -> Self {
        vaddq_s32(self, rhs)
    }

    unsafe fn sub(self, rhs: Self) -> Self {
        vsubq_s32(self, rhs)
    }

    unsafe fn shl<const COUNT: i32>(self) -> Self {
        vshlq_n_s32(self, COUNT)
    }

    unsafe fn reinterpret_as_float(self) -> Self::Float {
        vreinterpretq_f32_s32(self)
    }

    unsafe fn load(ptr: *const i32) -> Self {
        vld1q_s32(ptr)
    }

    unsafe fn store(self, ptr: *mut i32) {
        vst1q_s32(ptr, self)
    }
}

impl SimdFloat for float32x4_t {
    type Int = int32x4_t;
    type Mask = uint32x4_t;

    const LEN: usize = 4;

    unsafe fn splat(val: f32) -> Self {
        vdupq_n_f32(val)
    }

    unsafe fn abs(self) -> Self {
        vabsq_f32(self)
    }

    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        vfmaq_f32(b, self, a)
    }

    unsafe fn sub(self, rhs: Self) -> Self {
        vsubq_f32(self, rhs)
    }

    unsafe fn add(self, rhs: Self) -> Self {
        vaddq_f32(self, rhs)
    }

    unsafe fn to_int_trunc(self) -> Self::Int {
        vcvtq_s32_f32(self)
    }

    unsafe fn mul(self, rhs: Self) -> Self {
        vmulq_f32(self, rhs)
    }

    unsafe fn div(self, rhs: Self) -> Self {
        vdivq_f32(self, rhs)
    }

    unsafe fn ge(self, rhs: Self) -> Self::Mask {
        vcgeq_f32(self, rhs)
    }

    unsafe fn le(self, rhs: Self) -> Self::Mask {
        vcleq_f32(self, rhs)
    }

    unsafe fn lt(self, rhs: Self) -> Self::Mask {
        vcltq_f32(self, rhs)
    }

    unsafe fn max(self, rhs: Self) -> Self {
        vmaxq_f32(self, rhs)
    }

    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
        vbslq_f32(mask, other, self)
    }

    unsafe fn load(ptr: *const f32) -> Self {
        vld1q_f32(ptr)
    }

    unsafe fn store(self, ptr: *mut f32) {
        vst1q_f32(ptr, self)
    }
}
