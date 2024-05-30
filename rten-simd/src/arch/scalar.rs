use crate::{SimdFloat, SimdInt, SimdMask, SimdVal};

impl SimdMask for bool {
    #[inline]
    unsafe fn and(self, rhs: Self) -> Self {
        self & rhs
    }
}

impl SimdVal for i32 {
    const LEN: usize = 1;

    type Mask = bool;
}

/// Treat an `i32` as a single-lane SIMD "vector".
impl SimdInt for i32 {
    type Float = f32;

    #[inline]
    unsafe fn zero() -> Self {
        0
    }

    #[inline]
    unsafe fn splat(val: i32) -> Self {
        val
    }

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
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self {
        if !mask {
            self
        } else {
            other
        }
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

    #[inline]
    unsafe fn load(ptr: *const i32) -> Self {
        *ptr
    }

    #[inline]
    unsafe fn store(self, ptr: *mut i32) {
        *ptr = self;
    }
}

impl SimdVal for f32 {
    const LEN: usize = 1;

    type Mask = bool;
}

/// Treat an `f32` as a single-lane SIMD "vector".
impl SimdFloat for f32 {
    type Int = i32;

    #[inline]
    unsafe fn one() -> Self {
        1.
    }

    #[inline]
    unsafe fn zero() -> Self {
        0.
    }

    #[inline]
    unsafe fn splat(val: f32) -> Self {
        val
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
    unsafe fn blend(self, rhs: Self, mask: Self::Mask) -> Self {
        if !mask {
            self
        } else {
            rhs
        }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        *ptr
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
    unsafe fn store(self, ptr: *mut f32) {
        *ptr = self;
    }

    #[inline]
    unsafe fn sum(self) -> f32 {
        self
    }
}
