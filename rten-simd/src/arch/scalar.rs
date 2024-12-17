use crate::{Simd, SimdFloat, SimdInt, SimdMask};

impl SimdMask for bool {
    type Array = [bool; 1];

    #[inline]
    unsafe fn and(self, rhs: Self) -> Self {
        self & rhs
    }

    #[inline]
    unsafe fn to_array(self) -> Self::Array {
        [self]
    }
}

macro_rules! impl_simd {
    ($type:ty) => {
        impl Simd for $type {
            const LEN: usize = 1;

            type Array = [$type; 1];
            type Elem = $type;
            type Mask = bool;

            #[inline]
            unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
                if !mask {
                    self
                } else {
                    other
                }
            }

            #[inline]
            unsafe fn splat(val: $type) -> Self {
                val
            }

            #[inline]
            unsafe fn load(ptr: *const $type) -> Self {
                *ptr
            }

            #[inline]
            unsafe fn store(self, ptr: *mut $type) {
                *ptr = self;
            }

            #[inline]
            unsafe fn to_array(self) -> Self::Array {
                [self]
            }
        }
    };
}

impl_simd!(i32);
impl_simd!(f32);

/// Treat an `i32` as a single-lane SIMD "vector".
impl SimdInt for i32 {
    type Float = f32;

    #[inline]
    unsafe fn ge(self, other: Self) -> Self::Mask {
        self >= other
    }

    #[inline]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    unsafe fn le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    unsafe fn lt(self, rhs: Self) -> Self::Mask {
        self < rhs
    }

    #[inline]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    unsafe fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline]
    unsafe fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline]
    unsafe fn shl<const COUNT: i32>(self) -> Self {
        self << COUNT
    }

    #[inline]
    unsafe fn reinterpret_as_float(self) -> Self::Float {
        f32::from_bits(self as u32)
    }
}

/// Treat an `f32` as a single-lane SIMD "vector".
impl SimdFloat for f32 {
    type Int = i32;

    #[inline]
    unsafe fn one() -> Self {
        1.
    }

    #[inline]
    unsafe fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    #[inline]
    unsafe fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline]
    unsafe fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline]
    unsafe fn to_int_trunc(self) -> Self::Int {
        self as i32
    }

    #[inline]
    unsafe fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    unsafe fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline]
    unsafe fn ge(self, rhs: Self) -> Self::Mask {
        self >= rhs
    }

    #[inline]
    unsafe fn le(self, rhs: Self) -> Self::Mask {
        self <= rhs
    }

    #[inline]
    unsafe fn lt(self, rhs: Self) -> Self::Mask {
        self < rhs
    }

    #[inline]
    unsafe fn max(self, rhs: Self) -> Self {
        f32::max(self, rhs)
    }

    #[inline]
    unsafe fn gather_mask(ptr: *const f32, offset: i32, mask: Self::Mask) -> Self {
        if mask {
            *ptr.add(offset as usize)
        } else {
            0.
        }
    }

    #[inline]
    unsafe fn sum(self) -> f32 {
        self
    }
}
