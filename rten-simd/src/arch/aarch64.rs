use std::arch::aarch64::{
    float32x4_t, int16x8_t, int32x4_t, int8x16_t, uint16x8_t, uint32x4_t, uint8x16_t, vabsq_f32,
    vaddq_f32, vaddq_s16, vaddq_s32, vaddq_s8, vaddq_u16, vaddq_u8, vaddvq_f32, vandq_u16,
    vandq_u32, vandq_u8, vbslq_f32, vbslq_s16, vbslq_s32, vbslq_s8, vbslq_u16, vbslq_u8, vceqq_f32,
    vceqq_s16, vceqq_s32, vceqq_s8, vceqq_u16, vceqq_u8, vcgeq_f32, vcgeq_s16, vcgeq_s32, vcgeq_s8,
    vcgeq_u16, vcgeq_u8, vcgtq_f32, vcgtq_s16, vcgtq_s32, vcgtq_s8, vcgtq_u16, vcgtq_u8, vcleq_f32,
    vcleq_s16, vcleq_s8, vcleq_u16, vcleq_u8, vcltq_f32, vcltq_s16, vcltq_s8, vcltq_u16, vcltq_u8,
    vcombine_s16, vcombine_u8, vcvtnq_s32_f32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s16,
    vdupq_n_s32, vdupq_n_s8, vdupq_n_u16, vdupq_n_u8, veorq_u32, vfmaq_f32, vget_low_s16,
    vget_low_s8, vld1q_f32, vld1q_s16, vld1q_s32, vld1q_s8, vld1q_u16, vld1q_u32, vld1q_u8,
    vmaxq_f32, vminq_f32, vmovl_high_s16, vmovl_high_s8, vmovl_s16, vmovl_s8, vmulq_f32, vmulq_s16,
    vmulq_s32, vmulq_s8, vmulq_u16, vmulq_u8, vmvnq_u32, vnegq_f32, vnegq_s16, vnegq_s32, vnegq_s8,
    vqmovn_s32, vqmovun_s16, vshlq_n_s16, vshlq_n_s32, vshlq_n_s8, vst1q_f32, vst1q_s16, vst1q_s32,
    vst1q_s8, vst1q_u16, vst1q_u8, vsubq_f32, vsubq_s16, vsubq_s32, vsubq_s8, vsubq_u16, vsubq_u8,
    vzip1q_s16, vzip1q_s8, vzip2q_s16, vzip2q_s8,
};
use std::mem::transmute;

use crate::{
    Extend, FloatOps, Interleave, Isa, Mask, MaskOps, NarrowSaturate, NumOps, SignedIntOps, Simd,
};

#[derive(Copy, Clone)]
pub struct ArmNeonIsa {
    _private: (),
}

impl ArmNeonIsa {
    pub fn new() -> Option<Self> {
        Some(ArmNeonIsa { _private: () })
    }
}

// Safety: Neon is supported, as it is a required feature of aarch64.
unsafe impl Isa for ArmNeonIsa {
    type F32 = float32x4_t;
    type I32 = int32x4_t;
    type I16 = int16x8_t;
    type I8 = int8x16_t;
    type U8 = uint8x16_t;
    type U16 = uint16x8_t;
    type Bits = int32x4_t;

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

    fn u16(self) -> impl NumOps<u16, Simd = Self::U16> {
        self
    }
}

macro_rules! simd_ops_common {
    ($simd:ty, $mask:ty) => {
        type Simd = $simd;

        #[inline]
        fn len(self) -> usize {
            super::lanes::<$simd>()
        }

        #[inline]
        unsafe fn load_ptr_mask(self, ptr: *const <$simd as Simd>::Elem, mask: $mask) -> $simd {
            type Elem = <$simd as Simd>::Elem;

            let mask_array = Mask::to_array(mask);
            let mut vec = Simd::to_array(<Self as NumOps<Elem>>::zero(self));
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
            for i in 0..<Self as NumOps<Elem>>::len(self) {
                if mask_array[i] {
                    *ptr.add(i) = x_array[i];
                }
            }
        }

        #[inline]
        fn mask_ops(self) -> impl MaskOps<$mask> {
            self
        }

        // Since bitwise ops work on individual bits, we can use the same
        // implementation regardless of numeric type.

        #[inline]
        fn xor(self, x: $simd, y: $simd) -> $simd {
            unsafe {
                let x = transmute::<$simd, uint32x4_t>(x);
                let y = transmute::<$simd, uint32x4_t>(y);
                let tmp = veorq_u32(x, y);
                transmute::<uint32x4_t, $simd>(tmp)
            }
        }

        #[inline]
        fn not(self, x: $simd) -> $simd {
            unsafe {
                let x = transmute::<$simd, uint32x4_t>(x);
                let tmp = vmvnq_u32(x);
                transmute::<uint32x4_t, $simd>(tmp)
            }
        }
    };
}

unsafe impl NumOps<f32> for ArmNeonIsa {
    simd_ops_common!(float32x4_t, uint32x4_t);

    #[inline]
    fn add(self, x: float32x4_t, y: float32x4_t) -> float32x4_t {
        unsafe { vaddq_f32(x, y) }
    }

    #[inline]
    fn sub(self, x: float32x4_t, y: float32x4_t) -> float32x4_t {
        unsafe { vsubq_f32(x, y) }
    }

    #[inline]
    fn mul(self, x: float32x4_t, y: float32x4_t) -> float32x4_t {
        unsafe { vmulq_f32(x, y) }
    }

    #[inline]
    fn mul_add(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        unsafe { vfmaq_f32(c, a, b) }
    }

    #[inline]
    fn lt(self, x: float32x4_t, y: float32x4_t) -> uint32x4_t {
        unsafe { vcltq_f32(x, y) }
    }

    #[inline]
    fn le(self, x: float32x4_t, y: float32x4_t) -> uint32x4_t {
        unsafe { vcleq_f32(x, y) }
    }

    #[inline]
    fn eq(self, x: float32x4_t, y: float32x4_t) -> uint32x4_t {
        unsafe { vceqq_f32(x, y) }
    }

    #[inline]
    fn ge(self, x: float32x4_t, y: float32x4_t) -> uint32x4_t {
        unsafe { vcgeq_f32(x, y) }
    }

    #[inline]
    fn gt(self, x: float32x4_t, y: float32x4_t) -> uint32x4_t {
        unsafe { vcgtq_f32(x, y) }
    }

    #[inline]
    fn min(self, x: float32x4_t, y: float32x4_t) -> float32x4_t {
        unsafe { vminq_f32(x, y) }
    }

    #[inline]
    fn max(self, x: float32x4_t, y: float32x4_t) -> float32x4_t {
        unsafe { vmaxq_f32(x, y) }
    }

    #[inline]
    fn splat(self, x: f32) -> float32x4_t {
        unsafe { vdupq_n_f32(x) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const f32) -> float32x4_t {
        unsafe { vld1q_f32(ptr) }
    }

    #[inline]
    fn first_n_mask(self, n: usize) -> uint32x4_t {
        let mask: [u32; 4] = std::array::from_fn(|i| if i < n { u32::MAX } else { 0 });
        unsafe { vld1q_u32(mask.as_ptr()) }
    }

    #[inline]
    fn select(
        self,
        x: float32x4_t,
        y: float32x4_t,
        mask: <float32x4_t as Simd>::Mask,
    ) -> float32x4_t {
        unsafe { vbslq_f32(mask, x, y) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: float32x4_t, ptr: *mut f32) {
        unsafe { vst1q_f32(ptr, x) }
    }

    #[inline]
    fn sum(self, x: float32x4_t) -> f32 {
        unsafe { vaddvq_f32(x) }
    }
}

impl FloatOps<f32> for ArmNeonIsa {
    type Int = <Self as Isa>::I32;

    #[inline]
    fn div(self, x: float32x4_t, y: float32x4_t) -> float32x4_t {
        unsafe { vdivq_f32(x, y) }
    }

    #[inline]
    fn neg(self, x: float32x4_t) -> float32x4_t {
        unsafe { vnegq_f32(x) }
    }

    #[inline]
    fn abs(self, x: float32x4_t) -> float32x4_t {
        unsafe { vabsq_f32(x) }
    }

    #[inline]
    fn to_int_trunc(self, x: float32x4_t) -> Self::Int {
        unsafe { vcvtq_s32_f32(x) }
    }

    #[inline]
    fn to_int_round(self, x: float32x4_t) -> Self::Int {
        unsafe { vcvtnq_s32_f32(x) }
    }
}

unsafe impl NumOps<i32> for ArmNeonIsa {
    simd_ops_common!(int32x4_t, uint32x4_t);

    #[inline]
    fn add(self, x: int32x4_t, y: int32x4_t) -> int32x4_t {
        unsafe { vaddq_s32(x, y) }
    }

    #[inline]
    fn sub(self, x: int32x4_t, y: int32x4_t) -> int32x4_t {
        unsafe { vsubq_s32(x, y) }
    }

    #[inline]
    fn mul(self, x: int32x4_t, y: int32x4_t) -> int32x4_t {
        unsafe { vmulq_s32(x, y) }
    }

    #[inline]
    fn splat(self, x: i32) -> int32x4_t {
        unsafe { vdupq_n_s32(x) }
    }

    #[inline]
    fn eq(self, x: int32x4_t, y: int32x4_t) -> uint32x4_t {
        unsafe { vceqq_s32(x, y) }
    }

    #[inline]
    fn ge(self, x: int32x4_t, y: int32x4_t) -> uint32x4_t {
        unsafe { vcgeq_s32(x, y) }
    }

    #[inline]
    fn gt(self, x: int32x4_t, y: int32x4_t) -> uint32x4_t {
        unsafe { vcgtq_s32(x, y) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i32) -> int32x4_t {
        unsafe { vld1q_s32(ptr) }
    }

    #[inline]
    fn first_n_mask(self, n: usize) -> uint32x4_t {
        let mask: [u32; 4] = std::array::from_fn(|i| if i < n { u32::MAX } else { 0 });
        unsafe { vld1q_u32(mask.as_ptr()) }
    }

    #[inline]
    fn select(self, x: int32x4_t, y: int32x4_t, mask: <int32x4_t as Simd>::Mask) -> int32x4_t {
        unsafe { vbslq_s32(mask, x, y) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: int32x4_t, ptr: *mut i32) {
        unsafe { vst1q_s32(ptr, x) }
    }
}

impl SignedIntOps<i32> for ArmNeonIsa {
    #[inline]
    fn neg(self, x: int32x4_t) -> int32x4_t {
        unsafe { vnegq_s32(x) }
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: int32x4_t) -> int32x4_t {
        unsafe { vshlq_n_s32::<SHIFT>(x) }
    }
}

impl NarrowSaturate<i32, i16> for ArmNeonIsa {
    type Output = int16x8_t;

    #[inline]
    fn narrow_saturate(self, low: int32x4_t, high: int32x4_t) -> int16x8_t {
        unsafe {
            let low = vqmovn_s32(low);
            let high = vqmovn_s32(high);
            vcombine_s16(low, high)
        }
    }
}

impl NarrowSaturate<i16, u8> for ArmNeonIsa {
    type Output = uint8x16_t;

    #[inline]
    fn narrow_saturate(self, low: int16x8_t, high: int16x8_t) -> uint8x16_t {
        unsafe {
            let low = vqmovun_s16(low);
            let high = vqmovun_s16(high);
            vcombine_u8(low, high)
        }
    }
}

unsafe impl NumOps<i16> for ArmNeonIsa {
    simd_ops_common!(int16x8_t, uint16x8_t);

    #[inline]
    fn add(self, x: int16x8_t, y: int16x8_t) -> int16x8_t {
        unsafe { vaddq_s16(x, y) }
    }

    #[inline]
    fn sub(self, x: int16x8_t, y: int16x8_t) -> int16x8_t {
        unsafe { vsubq_s16(x, y) }
    }

    #[inline]
    fn mul(self, x: int16x8_t, y: int16x8_t) -> int16x8_t {
        unsafe { vmulq_s16(x, y) }
    }

    #[inline]
    fn splat(self, x: i16) -> int16x8_t {
        unsafe { vdupq_n_s16(x) }
    }

    #[inline]
    fn lt(self, x: int16x8_t, y: int16x8_t) -> uint16x8_t {
        unsafe { vcltq_s16(x, y) }
    }

    #[inline]
    fn le(self, x: int16x8_t, y: int16x8_t) -> uint16x8_t {
        unsafe { vcleq_s16(x, y) }
    }

    #[inline]
    fn eq(self, x: int16x8_t, y: int16x8_t) -> uint16x8_t {
        unsafe { vceqq_s16(x, y) }
    }

    #[inline]
    fn ge(self, x: int16x8_t, y: int16x8_t) -> uint16x8_t {
        unsafe { vcgeq_s16(x, y) }
    }

    #[inline]
    fn gt(self, x: int16x8_t, y: int16x8_t) -> uint16x8_t {
        unsafe { vcgtq_s16(x, y) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i16) -> int16x8_t {
        unsafe { vld1q_s16(ptr) }
    }

    #[inline]
    fn first_n_mask(self, n: usize) -> uint16x8_t {
        let mask: [u16; 8] = std::array::from_fn(|i| if i < n { u16::MAX } else { 0 });
        unsafe { vld1q_u16(mask.as_ptr()) }
    }

    #[inline]
    fn select(self, x: int16x8_t, y: int16x8_t, mask: <int16x8_t as Simd>::Mask) -> int16x8_t {
        unsafe { vbslq_s16(mask, x, y) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: int16x8_t, ptr: *mut i16) {
        unsafe { vst1q_s16(ptr, x) }
    }
}

impl SignedIntOps<i16> for ArmNeonIsa {
    #[inline]
    fn neg(self, x: int16x8_t) -> int16x8_t {
        unsafe { vnegq_s16(x) }
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: int16x8_t) -> int16x8_t {
        unsafe { vshlq_n_s16::<SHIFT>(x) }
    }
}

impl Extend<i16> for ArmNeonIsa {
    type Output = int32x4_t;

    #[inline]
    fn extend(self, x: int16x8_t) -> (int32x4_t, int32x4_t) {
        unsafe {
            let low = vmovl_s16(vget_low_s16(x));
            let high = vmovl_high_s16(x);
            (low, high)
        }
    }
}

impl Interleave<i16> for ArmNeonIsa {
    #[inline]
    fn interleave_low(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vzip1q_s16(a, b) }
    }

    #[inline]
    fn interleave_high(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vzip2q_s16(a, b) }
    }
}

unsafe impl NumOps<i8> for ArmNeonIsa {
    simd_ops_common!(int8x16_t, uint8x16_t);

    #[inline]
    fn add(self, x: int8x16_t, y: int8x16_t) -> int8x16_t {
        unsafe { vaddq_s8(x, y) }
    }

    #[inline]
    fn sub(self, x: int8x16_t, y: int8x16_t) -> int8x16_t {
        unsafe { vsubq_s8(x, y) }
    }

    #[inline]
    fn mul(self, x: int8x16_t, y: int8x16_t) -> int8x16_t {
        unsafe { vmulq_s8(x, y) }
    }

    #[inline]
    fn splat(self, x: i8) -> int8x16_t {
        unsafe { vdupq_n_s8(x) }
    }

    #[inline]
    fn lt(self, x: int8x16_t, y: int8x16_t) -> uint8x16_t {
        unsafe { vcltq_s8(x, y) }
    }

    #[inline]
    fn le(self, x: int8x16_t, y: int8x16_t) -> uint8x16_t {
        unsafe { vcleq_s8(x, y) }
    }

    #[inline]
    fn eq(self, x: int8x16_t, y: int8x16_t) -> uint8x16_t {
        unsafe { vceqq_s8(x, y) }
    }

    #[inline]
    fn ge(self, x: int8x16_t, y: int8x16_t) -> uint8x16_t {
        unsafe { vcgeq_s8(x, y) }
    }

    #[inline]
    fn gt(self, x: int8x16_t, y: int8x16_t) -> uint8x16_t {
        unsafe { vcgtq_s8(x, y) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const i8) -> int8x16_t {
        unsafe { vld1q_s8(ptr) }
    }

    #[inline]
    fn first_n_mask(self, n: usize) -> uint8x16_t {
        let mask: [u8; 16] = std::array::from_fn(|i| if i < n { u8::MAX } else { 0 });
        unsafe { vld1q_u8(mask.as_ptr()) }
    }

    #[inline]
    fn select(self, x: int8x16_t, y: int8x16_t, mask: <int8x16_t as Simd>::Mask) -> int8x16_t {
        unsafe { vbslq_s8(mask, x, y) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: int8x16_t, ptr: *mut i8) {
        unsafe { vst1q_s8(ptr, x) }
    }
}

impl SignedIntOps<i8> for ArmNeonIsa {
    #[inline]
    fn neg(self, x: int8x16_t) -> int8x16_t {
        unsafe { vnegq_s8(x) }
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: int8x16_t) -> int8x16_t {
        unsafe { vshlq_n_s8::<SHIFT>(x) }
    }
}

impl Extend<i8> for ArmNeonIsa {
    type Output = int16x8_t;

    #[inline]
    fn extend(self, x: int8x16_t) -> (int16x8_t, int16x8_t) {
        unsafe {
            let low = vmovl_s8(vget_low_s8(x));
            let high = vmovl_high_s8(x);
            (low, high)
        }
    }
}

impl Interleave<i8> for ArmNeonIsa {
    #[inline]
    fn interleave_low(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vzip1q_s8(a, b) }
    }

    #[inline]
    fn interleave_high(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vzip2q_s8(a, b) }
    }
}

unsafe impl NumOps<u8> for ArmNeonIsa {
    simd_ops_common!(uint8x16_t, uint8x16_t);

    #[inline]
    fn add(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vaddq_u8(x, y) }
    }

    #[inline]
    fn sub(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vsubq_u8(x, y) }
    }

    #[inline]
    fn mul(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vmulq_u8(x, y) }
    }

    #[inline]
    fn splat(self, x: u8) -> uint8x16_t {
        unsafe { vdupq_n_u8(x) }
    }

    #[inline]
    fn lt(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vcltq_u8(x, y) }
    }

    #[inline]
    fn le(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vcleq_u8(x, y) }
    }

    #[inline]
    fn eq(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vceqq_u8(x, y) }
    }

    #[inline]
    fn ge(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vcgeq_u8(x, y) }
    }

    #[inline]
    fn gt(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vcgtq_u8(x, y) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const u8) -> uint8x16_t {
        unsafe { vld1q_u8(ptr) }
    }

    #[inline]
    fn first_n_mask(self, n: usize) -> uint8x16_t {
        let mask: [u8; 16] = std::array::from_fn(|i| if i < n { u8::MAX } else { 0 });
        unsafe { vld1q_u8(mask.as_ptr()) }
    }

    #[inline]
    fn select(self, x: uint8x16_t, y: uint8x16_t, mask: <uint8x16_t as Simd>::Mask) -> uint8x16_t {
        unsafe { vbslq_u8(mask, x, y) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: uint8x16_t, ptr: *mut u8) {
        unsafe { vst1q_u8(ptr, x) }
    }
}

unsafe impl NumOps<u16> for ArmNeonIsa {
    simd_ops_common!(uint16x8_t, uint16x8_t);

    #[inline]
    fn add(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vaddq_u16(x, y) }
    }

    #[inline]
    fn sub(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vsubq_u16(x, y) }
    }

    #[inline]
    fn mul(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vmulq_u16(x, y) }
    }

    #[inline]
    fn splat(self, x: u16) -> uint16x8_t {
        unsafe { vdupq_n_u16(x) }
    }

    #[inline]
    fn lt(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vcltq_u16(x, y) }
    }

    #[inline]
    fn le(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vcleq_u16(x, y) }
    }

    #[inline]
    fn eq(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vceqq_u16(x, y) }
    }

    #[inline]
    fn ge(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vcgeq_u16(x, y) }
    }

    #[inline]
    fn gt(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vcgtq_u16(x, y) }
    }

    #[inline]
    unsafe fn load_ptr(self, ptr: *const u16) -> uint16x8_t {
        unsafe { vld1q_u16(ptr) }
    }

    #[inline]
    fn first_n_mask(self, n: usize) -> uint16x8_t {
        let mask: [u16; 8] = std::array::from_fn(|i| if i < n { u16::MAX } else { 0 });
        unsafe { vld1q_u16(mask.as_ptr()) }
    }

    #[inline]
    fn select(self, x: uint16x8_t, y: uint16x8_t, mask: <uint16x8_t as Simd>::Mask) -> uint16x8_t {
        unsafe { vbslq_u16(mask, x, y) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: uint16x8_t, ptr: *mut u16) {
        unsafe { vst1q_u16(ptr, x) }
    }
}

macro_rules! impl_mask {
    ($mask:ty, $elem:ty, $len:expr) => {
        impl Mask for $mask {
            type Array = [bool; $len];

            #[inline]
            fn to_array(self) -> Self::Array {
                let array = unsafe { transmute::<Self, [$elem; $len]>(self) };
                std::array::from_fn(|i| array[i] != 0)
            }
        }
    };
}

impl_mask!(uint32x4_t, u32, 4);
impl_mask!(uint16x8_t, u16, 8);
impl_mask!(uint8x16_t, u8, 16);

unsafe impl MaskOps<uint32x4_t> for ArmNeonIsa {
    #[inline]
    fn and(self, x: uint32x4_t, y: uint32x4_t) -> uint32x4_t {
        unsafe { vandq_u32(x, y) }
    }
}

unsafe impl MaskOps<uint16x8_t> for ArmNeonIsa {
    #[inline]
    fn and(self, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe { vandq_u16(x, y) }
    }
}

unsafe impl MaskOps<uint8x16_t> for ArmNeonIsa {
    #[inline]
    fn and(self, x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        unsafe { vandq_u8(x, y) }
    }
}

macro_rules! simd_common {
    ($mask:ty, $len:expr) => {
        type Array = [Self::Elem; $len];
        type Mask = $mask;
        type Isa = ArmNeonIsa;

        #[inline]
        fn to_bits(self) -> <Self::Isa as Isa>::Bits {
            #[allow(clippy::useless_transmute)]
            unsafe {
                transmute::<Self, int32x4_t>(self)
            }
        }

        #[inline]
        fn from_bits(bits: <Self::Isa as Isa>::Bits) -> Self {
            #[allow(clippy::useless_transmute)]
            unsafe {
                transmute::<int32x4_t, Self>(bits)
            }
        }

        #[inline]
        fn to_array(self) -> Self::Array {
            unsafe { transmute::<Self, Self::Array>(self) }
        }
    };
}

macro_rules! impl_simd {
    ($simd:ident, $elem:ty, $len:expr, $mask:ty) => {
        impl Simd for $simd {
            type Elem = $elem;

            simd_common!($mask, $len);
        }
    };
}

impl_simd!(float32x4_t, f32, 4, uint32x4_t);
impl_simd!(int32x4_t, i32, 4, uint32x4_t);
impl_simd!(int16x8_t, i16, 8, uint16x8_t);
impl_simd!(int8x16_t, i8, 16, uint8x16_t);
impl_simd!(uint8x16_t, u8, 16, uint8x16_t);
impl_simd!(uint16x8_t, u16, 8, uint16x8_t);
