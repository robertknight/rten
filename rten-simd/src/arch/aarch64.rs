use std::arch::aarch64::{
    float32x4_t, int32x4_t, uint32x4_t, vabsq_f32, vaddq_f32, vaddq_s32, vaddvq_f32, vandq_u32,
    vbslq_f32, vbslq_s32, vceqq_s32, vcgeq_f32, vcgeq_s32, vcgtq_s32, vcleq_f32, vcleq_s32,
    vcltq_f32, vcltq_s32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s32, vfmaq_f32, vld1q_f32,
    vld1q_s32, vld1q_u32, vmaxq_f32, vmulq_f32, vreinterpretq_f32_s32, vshlq_n_s32, vst1q_f32,
    vst1q_s32, vst1q_u32, vsubq_f32, vsubq_s32,
};

use crate::{Simd, SimdFloat, SimdInt, SimdMask};

impl SimdMask for uint32x4_t {
    type Array = [bool; 4];

    #[inline]
    unsafe fn and(self, other: Self) -> Self {
        vandq_u32(self, other)
    }

    #[inline]
    unsafe fn from_array(array: [bool; 4]) -> Self {
        let u32_array = array.map(|b| if b { u32::MAX } else { 0 });
        vld1q_u32(u32_array.as_ptr())
    }

    #[inline]
    unsafe fn to_array(self) -> Self::Array {
        let mut array = [0; 4];
        vst1q_u32(array.as_mut_ptr(), self);
        std::array::from_fn(|i| array[i] != 0)
    }
}

impl Simd for int32x4_t {
    const LEN: usize = 4;

    type Array = [i32; 4];
    type Elem = i32;
    type Mask = uint32x4_t;

    #[inline]
    unsafe fn zero() -> Self {
        vdupq_n_s32(0)
    }

    #[inline]
    unsafe fn splat(val: i32) -> Self {
        vdupq_n_s32(val)
    }

    #[inline]
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
        vbslq_s32(mask, other, self)
    }

    #[inline]
    unsafe fn load(ptr: *const i32) -> Self {
        vld1q_s32(ptr)
    }

    #[inline]
    unsafe fn store(self, ptr: *mut i32) {
        vst1q_s32(ptr, self)
    }

    #[inline]
    unsafe fn to_array(self) -> Self::Array {
        let mut array = [0; Self::LEN];
        self.store(array.as_mut_ptr());
        array
    }
}

impl SimdInt for int32x4_t {
    type Float = float32x4_t;

    #[inline]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        vceqq_s32(self, other)
    }

    #[inline]
    unsafe fn le(self, other: Self) -> Self::Mask {
        vcleq_s32(self, other)
    }

    #[inline]
    unsafe fn ge(self, other: Self) -> Self::Mask {
        vcgeq_s32(self, other)
    }

    #[inline]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        vcgtq_s32(self, other)
    }

    #[inline]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        vcltq_s32(self, other)
    }

    #[inline]
    unsafe fn add(self, rhs: Self) -> Self {
        vaddq_s32(self, rhs)
    }

    #[inline]
    unsafe fn sub(self, rhs: Self) -> Self {
        vsubq_s32(self, rhs)
    }

    #[inline]
    unsafe fn shl<const COUNT: i32>(self) -> Self {
        vshlq_n_s32(self, COUNT)
    }

    #[inline]
    unsafe fn reinterpret_as_float(self) -> Self::Float {
        vreinterpretq_f32_s32(self)
    }
}

impl Simd for float32x4_t {
    const LEN: usize = 4;

    type Array = [f32; 4];
    type Elem = f32;
    type Mask = uint32x4_t;

    #[inline]
    unsafe fn splat(val: f32) -> Self {
        vdupq_n_f32(val)
    }

    #[inline]
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
        vbslq_f32(mask, other, self)
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        vld1q_f32(ptr)
    }

    #[inline]
    unsafe fn store(self, ptr: *mut f32) {
        vst1q_f32(ptr, self)
    }

    #[inline]
    unsafe fn to_array(self) -> Self::Array {
        let mut array = [0.; Self::LEN];
        self.store(array.as_mut_ptr());
        array
    }
}

impl SimdFloat for float32x4_t {
    type Int = int32x4_t;

    #[inline]
    unsafe fn abs(self) -> Self {
        vabsq_f32(self)
    }

    #[inline]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        vfmaq_f32(b, self, a)
    }

    #[inline]
    unsafe fn sub(self, rhs: Self) -> Self {
        vsubq_f32(self, rhs)
    }

    #[inline]
    unsafe fn add(self, rhs: Self) -> Self {
        vaddq_f32(self, rhs)
    }

    #[inline]
    unsafe fn to_int_trunc(self) -> Self::Int {
        vcvtq_s32_f32(self)
    }

    #[inline]
    unsafe fn mul(self, rhs: Self) -> Self {
        vmulq_f32(self, rhs)
    }

    #[inline]
    unsafe fn div(self, rhs: Self) -> Self {
        vdivq_f32(self, rhs)
    }

    #[inline]
    unsafe fn ge(self, rhs: Self) -> Self::Mask {
        vcgeq_f32(self, rhs)
    }

    #[inline]
    unsafe fn le(self, rhs: Self) -> Self::Mask {
        vcleq_f32(self, rhs)
    }

    #[inline]
    unsafe fn lt(self, rhs: Self) -> Self::Mask {
        vcltq_f32(self, rhs)
    }

    #[inline]
    unsafe fn max(self, rhs: Self) -> Self {
        vmaxq_f32(self, rhs)
    }

    #[inline]
    unsafe fn gather_mask(src: *const f32, offsets: Self::Int, mask: Self::Mask) -> Self {
        super::simd_gather_mask::<_, _, _, { Self::LEN }>(src, offsets, mask)
    }

    #[inline]
    unsafe fn sum(self) -> f32 {
        vaddvq_f32(self)
    }
}
