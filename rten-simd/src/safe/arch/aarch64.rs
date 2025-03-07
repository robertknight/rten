use std::arch::aarch64::{
    float32x4_t, int32x4_t, uint32x4_t, vabsq_f32, vaddq_f32, vaddq_s32, vaddvq_f32, vandq_u32,
    vbslq_f32, vbslq_s32, vceqq_f32, vceqq_s32, vcgeq_f32, vcgeq_s32, vcgtq_f32, vcgtq_s32,
    vcleq_f32, vcltq_f32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s32, vfmaq_f32, vld1q_f32,
    vld1q_s32, vld1q_u32, vmaxq_f32, vminq_f32, vmulq_f32, vmulq_s32, vnegq_f32, vnegq_s32,
    vshlq_n_s32, vst1q_f32, vst1q_s32, vsubq_f32, vsubq_s32,
};
use std::mem::transmute;

use crate::safe::{Isa, Mask, MaskOps, Simd, SimdFloatOps, SimdIntOps, SimdOps};

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
    type Bits = int32x4_t;

    fn f32(self) -> impl SimdFloatOps<Self::F32, Int = Self::I32> {
        self
    }

    fn i32(self) -> impl SimdIntOps<Self::I32> {
        self
    }
}

macro_rules! simd_ops_x32_common {
    ($simd:ty) => {
        #[inline]
        fn len(self) -> usize {
            4
        }

        #[inline]
        fn first_n_mask(self, n: usize) -> uint32x4_t {
            let mask: [u32; 4] = std::array::from_fn(|i| if i < n { u32::MAX } else { 0 });
            unsafe { vld1q_u32(mask.as_ptr()) }
        }

        #[inline]
        unsafe fn load_ptr_mask(
            self,
            ptr: *const <$simd as Simd>::Elem,
            mask: uint32x4_t,
        ) -> $simd {
            let mask_array = mask.to_array();
            let mut vec = <Self as SimdOps<$simd>>::zero(self).to_array();
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
            let mask_array = mask.to_array();
            let x_array = x.to_array();
            for i in 0..<Self as SimdOps<$simd>>::len(self) {
                if mask_array[i] {
                    *ptr.add(i) = x_array[i];
                }
            }
        }

        #[inline]
        fn mask_ops(self) -> impl MaskOps<uint32x4_t> {
            self
        }
    };
}

unsafe impl SimdOps<float32x4_t> for ArmNeonIsa {
    simd_ops_x32_common!(float32x4_t);

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

impl SimdFloatOps<float32x4_t> for ArmNeonIsa {
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
}

unsafe impl SimdOps<int32x4_t> for ArmNeonIsa {
    simd_ops_x32_common!(int32x4_t);

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
    fn select(self, x: int32x4_t, y: int32x4_t, mask: <int32x4_t as Simd>::Mask) -> int32x4_t {
        unsafe { vbslq_s32(mask, x, y) }
    }

    #[inline]
    unsafe fn store_ptr(self, x: int32x4_t, ptr: *mut i32) {
        unsafe { vst1q_s32(ptr, x) }
    }
}

impl SimdIntOps<int32x4_t> for ArmNeonIsa {
    #[inline]
    fn neg(self, x: int32x4_t) -> int32x4_t {
        unsafe { vnegq_s32(x) }
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: int32x4_t) -> int32x4_t {
        unsafe { vshlq_n_s32::<SHIFT>(x) }
    }
}

impl Mask for uint32x4_t {
    type Array = [bool; 4];

    #[inline]
    fn to_array(self) -> Self::Array {
        let array = unsafe { transmute::<Self, [u32; 4]>(self) };
        std::array::from_fn(|i| array[i] != 0)
    }
}

macro_rules! simd_x32_common {
    () => {
        type Array = [Self::Elem; 4];
        type Mask = uint32x4_t;
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

unsafe impl MaskOps<uint32x4_t> for ArmNeonIsa {
    #[inline]
    fn and(self, x: uint32x4_t, y: uint32x4_t) -> uint32x4_t {
        unsafe { vandq_u32(x, y) }
    }
}

impl Simd for float32x4_t {
    type Elem = f32;

    simd_x32_common!();
}

impl Simd for int32x4_t {
    type Elem = i32;

    simd_x32_common!();
}
