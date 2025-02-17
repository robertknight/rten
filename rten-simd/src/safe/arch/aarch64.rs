use std::arch::aarch64::{
    float32x4_t, int32x4_t, uint32x4_t, vaddq_f32, vaddq_s32, vbslq_f32, vbslq_s32, vceqq_f32,
    vceqq_s32, vcgeq_f32, vcgeq_s32, vcgtq_f32, vcgtq_s32, vcleq_f32, vcleq_s32, vcltq_f32,
    vcltq_s32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s32, vfmaq_f32, vld1q_f32, vld1q_s32,
    vld1q_u32, vmulq_f32, vmulq_s32, vnegq_f32, vnegq_s32, vshlq_n_s32, vst1q_f32, vst1q_s32,
    vsubq_f32, vsubq_s32,
};
use std::mem::transmute;

use crate::safe::{Isa, MakeSimd, Mask, Simd, SimdF32, SimdFloat, SimdInt};

#[derive(Copy, Clone)]
pub struct ArmNeonIsa {
    _private: (),
}

impl ArmNeonIsa {
    pub fn new() -> Option<Self> {
        Some(ArmNeonIsa { _private: () })
    }
}

unsafe impl Isa for ArmNeonIsa {
    type F32 = NeonF32;
    type I32 = NeonI32;
    type Bits = NeonI32;

    fn f32(self) -> impl MakeSimd<Self::F32> {
        self
    }

    fn i32(self) -> impl MakeSimd<Self::I32> {
        self
    }
}

macro_rules! simd_init_x32_common {
    ($simd:ty) => {
        fn len(self) -> usize {
            4
        }

        fn first_n_mask(self, n: usize) -> NeonX32Mask {
            let mask = std::array::from_fn(|i| if i < n { u32::MAX } else { 0 });
            NeonX32Mask::from_u32_array(mask)
        }

        unsafe fn load_ptr_mask(
            self,
            ptr: *const <$simd as Simd>::Elem,
            mask: NeonX32Mask,
        ) -> $simd {
            let mask_array = mask.to_array();
            let mut remainder = <Self as MakeSimd<$simd>>::zero(self).to_array();
            for i in 0..mask_array.len() {
                if mask_array[i] {
                    remainder[i] = *ptr.add(i);
                }
            }
            self.load_ptr(remainder.as_ref().as_ptr())
        }
    };
}

unsafe impl MakeSimd<NeonF32> for ArmNeonIsa {
    simd_init_x32_common!(NeonF32);

    fn splat(self, x: f32) -> NeonF32 {
        unsafe { NeonF32(vdupq_n_f32(x)) }
    }

    unsafe fn load_ptr(self, ptr: *const f32) -> NeonF32 {
        unsafe { NeonF32(vld1q_f32(ptr)) }
    }
}

unsafe impl MakeSimd<NeonI32> for ArmNeonIsa {
    simd_init_x32_common!(NeonI32);

    fn splat(self, x: i32) -> NeonI32 {
        unsafe { NeonI32(vdupq_n_s32(x)) }
    }

    unsafe fn load_ptr(self, ptr: *const i32) -> NeonI32 {
        unsafe { NeonI32(vld1q_s32(ptr)) }
    }
}

/// Mask for Neon vectors with 32-bit lanes.
#[derive(Copy, Clone)]
pub struct NeonX32Mask(uint32x4_t);

impl NeonX32Mask {
    fn from_u32_array(xs: [u32; 4]) -> Self {
        unsafe { Self(vld1q_u32(xs.as_ptr())) }
    }
}

unsafe impl Mask for NeonX32Mask {
    type Array = [bool; 4];

    fn to_array(self) -> Self::Array {
        let array = unsafe { transmute::<Self, [u32; 4]>(self) };
        std::array::from_fn(|i| array[i] != 0)
    }
}

macro_rules! simd_x32_common {
    () => {
        type Array = [Self::Elem; 4];
        type Mask = NeonX32Mask;
        type Isa = ArmNeonIsa;

        fn len(self) -> usize {
            4
        }

        fn init(self) -> impl MakeSimd<Self> {
            ArmNeonIsa { _private: () }
        }

        fn isa(self) -> Self::Isa {
            ArmNeonIsa { _private: () }
        }

        fn to_bits(self) -> <Self::Isa as Isa>::Bits {
            #[allow(clippy::useless_transmute)]
            #[allow(clippy::missing_transmute_annotations)]
            NeonI32(unsafe { transmute(self.0) })
        }

        fn from_bits(bits: <Self::Isa as Isa>::Bits) -> Self {
            #[allow(clippy::useless_transmute)]
            #[allow(clippy::missing_transmute_annotations)]
            Self(unsafe { transmute(bits.0) })
        }

        unsafe fn store_ptr_mask(self, ptr: *mut Self::Elem, mask: Self::Mask) {
            let mask_array = mask.to_array();
            let self_array = self.to_array();
            for i in 0..self.len() {
                if mask_array[i] {
                    *ptr.add(i) = self_array[i];
                }
            }
        }
    };
}

#[derive(Copy, Clone, Debug)]
pub struct NeonF32(float32x4_t);

unsafe impl Simd for NeonF32 {
    type Elem = f32;

    simd_x32_common!();

    fn lt(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vcltq_f32(self.0, rhs.0) })
    }

    fn le(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vcleq_f32(self.0, rhs.0) })
    }

    fn eq(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vceqq_f32(self.0, rhs.0) })
    }

    fn ge(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vcgeq_f32(self.0, rhs.0) })
    }

    fn gt(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vcgtq_f32(self.0, rhs.0) })
    }

    fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { vfmaq_f32(self.0, b.0, c.0) })
    }

    fn to_array(self) -> Self::Array {
        unsafe { transmute::<float32x4_t, Self::Array>(self.0) }
    }

    fn select(self, other: Self, mask: Self::Mask) -> Self {
        unsafe { Self(vbslq_f32(mask.0, self.0, other.0)) }
    }

    unsafe fn store_ptr(self, ptr: *mut f32) {
        unsafe { vst1q_f32(ptr, self.0) }
    }
}

impl SimdFloat for NeonF32 {}

impl SimdF32 for NeonF32 {
    fn to_i32_trunc(self) -> NeonI32 {
        NeonI32(unsafe { vcvtq_s32_f32(self.0) })
    }
}

impl PartialEq for NeonF32 {
    fn eq(&self, rhs: &NeonF32) -> bool {
        <Self as Simd>::eq(*self, *rhs).all_true()
    }
}

macro_rules! impl_bin_op {
    ($struct:ty, $op:ident, $op_method:ident, $intrinsic:ident) => {
        impl std::ops::$op for $struct {
            type Output = Self;

            fn $op_method(self, rhs: Self) -> Self {
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_unary_op {
    ($struct:ty, $op:ident, $op_method:ident, $intrinsic:ident) => {
        impl std::ops::$op for $struct {
            type Output = Self;

            fn $op_method(self) -> Self {
                Self(unsafe { $intrinsic(self.0) })
            }
        }
    };
}

impl_bin_op!(NeonF32, Add, add, vaddq_f32);
impl_bin_op!(NeonF32, Mul, mul, vmulq_f32);
impl_bin_op!(NeonF32, Sub, sub, vsubq_f32);
impl_bin_op!(NeonF32, Div, div, vdivq_f32);
impl_unary_op!(NeonF32, Neg, neg, vnegq_f32);

#[derive(Copy, Clone, Debug)]
pub struct NeonI32(int32x4_t);

unsafe impl Simd for NeonI32 {
    type Elem = i32;

    simd_x32_common!();

    fn lt(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vcltq_s32(self.0, rhs.0) })
    }

    fn le(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vcleq_s32(self.0, rhs.0) })
    }

    fn eq(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vceqq_s32(self.0, rhs.0) })
    }

    fn ge(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vcgeq_s32(self.0, rhs.0) })
    }

    fn gt(self, rhs: Self) -> Self::Mask {
        NeonX32Mask(unsafe { vcgtq_s32(self.0, rhs.0) })
    }

    fn to_array(self) -> Self::Array {
        unsafe { transmute::<int32x4_t, [i32; 4]>(self.0) }
    }

    fn select(self, other: Self, mask: Self::Mask) -> Self {
        unsafe { Self(vbslq_s32(mask.0, self.0, other.0)) }
    }

    unsafe fn store_ptr(self, ptr: *mut i32) {
        unsafe { vst1q_s32(ptr, self.0) }
    }
}

impl SimdInt for NeonI32 {
    fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_s32(self.0, N) })
    }
}

impl PartialEq for NeonI32 {
    fn eq(&self, rhs: &NeonI32) -> bool {
        <Self as Simd>::eq(*self, *rhs).all_true()
    }
}

macro_rules! impl_bin_op {
    ($struct:ty, $op:ident, $op_method:ident, $intrinsic:ident) => {
        impl std::ops::$op for $struct {
            type Output = Self;

            fn $op_method(self, rhs: Self) -> Self {
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_unary_op {
    ($struct:ty, $op:ident, $op_method:ident, $intrinsic:ident) => {
        impl std::ops::$op for $struct {
            type Output = Self;

            fn $op_method(self) -> Self {
                Self(unsafe { $intrinsic(self.0) })
            }
        }
    };
}

impl_bin_op!(NeonI32, Add, add, vaddq_s32);
impl_bin_op!(NeonI32, Mul, mul, vmulq_s32);
impl_bin_op!(NeonI32, Sub, sub, vsubq_s32);
impl_unary_op!(NeonI32, Neg, neg, vnegq_s32);
