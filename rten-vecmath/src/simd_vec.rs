#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

use crate::MAX_LEN;

/// Trait for SIMD vectors containing 32-bit integers.
///
/// All functions in this trait are unsafe due to limitations of Rust's
/// #[target_feature] macro. See
/// https://rust-lang.github.io/rfcs/2396-target-feature-1.1.html. Also as
/// a consequence of this, standard operations like add, multiply etc. are
/// implemented as functions in this trait rather than using the standard
/// trait from `std::ops`.
pub trait SimdInt: Copy + Sized {
    /// The number of elements in the SIMD vector.
    const LEN: usize;

    type Float: SimdFloat<Int = Self>;

    /// Return a new vector with all elements set to zero.
    unsafe fn zero() -> Self {
        Self::splat(0)
    }

    /// Broadcast `val` to all elements in a new vector.
    unsafe fn splat(val: i32) -> Self;

    /// Return a mask indicating whether `self > other`.
    unsafe fn gt(self, other: Self) -> Self;

    /// Select elements from this vector or `other` according to a mask.
    ///
    /// For each lane, if the mask value is zero, return the element from
    /// `self`, otherwise return the value from `other`.
    unsafe fn blend(self, other: Self, mask: Self) -> Self;

    /// Compute `self + rhs`.
    unsafe fn add(self, rhs: Self) -> Self;

    /// Compute `self - rhs`.
    unsafe fn sub(self, rhs: Self) -> Self;

    /// Shift the bits in each element left by `count`.
    unsafe fn shl<const COUNT: i32>(self) -> Self;

    /// Reinterpret the bits of each element as a float.
    unsafe fn reinterpret_as_float(self) -> Self::Float;

    /// Load `Self::LEN` values from the memory address at `ptr`.
    ///
    /// Implementations must not require `ptr` to be aligned.
    ///
    /// Safety: The caller must ensure `ptr` points to at least `Self::LEN`
    /// values.
    unsafe fn load(ptr: *const i32) -> Self;

    /// Store `Self::LEN` values to the memory address at `ptr`.
    ///
    /// Implementations must not require `ptr` to be aligned.
    ///
    /// Safety: The caller must ensure `ptr` points to a buffer with space for
    /// at least `Self::LEN` values.
    unsafe fn store(self, ptr: *mut i32);
}

/// Trait for SIMD vectors containing single-precision floats.
///
/// All functions in this trait are unsafe due to limitations of Rust's
/// #[target_feature] macro. See
/// https://rust-lang.github.io/rfcs/2396-target-feature-1.1.html. Also as
/// a consequence of this, standard operations like add, multiply etc. are
/// implemented as functions in this trait rather than using the standard
/// trait from `std::ops`.
pub trait SimdFloat: Copy + Sized {
    /// The number of elements in the SIMD vector.
    const LEN: usize;

    /// The type of vector produced by operations that convert this vector
    /// to a vector of ints.
    type Int: SimdInt<Float = Self>;

    /// Shorthand for `Self::splat(1.0)`.
    unsafe fn one() -> Self {
        Self::splat(1.0)
    }

    /// Shorthand for `Self::splat(0.0)`.
    unsafe fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Compute `-self`.
    unsafe fn neg(self) -> Self {
        Self::zero().sub(self)
    }

    /// Compute `1. / self`.
    unsafe fn reciprocal(self) -> Self {
        Self::one().div(self)
    }

    /// Return the absolute value of `self`.
    unsafe fn abs(self) -> Self;

    /// Broadcast `val` to all elements in a new vector.
    unsafe fn splat(val: f32) -> Self;

    /// Compute `self * a + b` as a single operation.
    unsafe fn mul_add(self, a: Self, b: Self) -> Self;

    /// Compute `self + rhs`.
    unsafe fn add(self, rhs: Self) -> Self;

    /// Compute `self - rhs`.
    unsafe fn sub(self, rhs: Self) -> Self;

    /// Convert this float to an int with truncation.
    unsafe fn to_int_trunc(self) -> Self::Int;

    /// Compute `self * rhs`.
    unsafe fn mul(self, rhs: Self) -> Self;

    /// Compute `self / rhs`.
    unsafe fn div(self, rhs: Self) -> Self;

    /// Compute a mask containing `self >= rhs`.
    unsafe fn ge(self, rhs: Self) -> Self;

    /// Compute a mask containing `self <= rhs`.
    unsafe fn le(self, rhs: Self) -> Self;

    /// Compute a mask containing `self < rhs`.
    unsafe fn lt(self, rhs: Self) -> Self;

    /// Compute the maximum of `self` and `rhs`.
    unsafe fn max(self, rhs: Self) -> Self;

    /// Combine elements of `self` and `rhs` according to a mask.
    ///
    /// For each lane, if the mask value is zero, return the element from
    /// `self`, otherwise return the value from `other`.
    unsafe fn blend(self, other: Self, mask: Self) -> Self;

    /// Evaluate a polynomial using Horner's method.
    ///
    /// Computes `self * coeffs[0] + self^2 * coeffs[1] ... self^n * coeffs[N]`
    unsafe fn poly_eval(self, coeffs: &[Self]) -> Self {
        let mut y = coeffs[coeffs.len() - 1];
        for i in (0..coeffs.len() - 1).rev() {
            y = y.mul_add(self, coeffs[i]);
        }
        y.mul(self)
    }

    /// Load `Self::LEN` floats from the memory address at `ptr`.
    ///
    /// Implementations must not require `ptr` to be aligned.
    ///
    /// Safety: The caller must ensure `ptr` points to at least `Self::LEN`
    /// floats.
    unsafe fn load(ptr: *const f32) -> Self;

    /// Store `Self::LEN` floats to the memory address at `ptr`.
    ///
    /// Implementations must not require `ptr` to be aligned.
    ///
    /// Safety: The caller must ensure `ptr` points to a buffer with space for
    /// at least `Self::LEN` floats.
    unsafe fn store(self, ptr: *mut f32);

    /// Reduce the elements in this vector to a single value using `f`, then
    /// return a new vector with the accumulated value broadcast to each lane.
    unsafe fn fold_splat<F: Fn(f32, f32) -> f32>(self, accum: f32, f: F) -> Self {
        let mut elements = [accum; MAX_LEN];
        self.store(elements.as_mut_ptr());
        let reduced = elements.into_iter().fold(accum, f);
        Self::splat(reduced)
    }
}

/// Treat an `i32` as a single-lane SIMD "vector".
impl SimdInt for i32 {
    const LEN: usize = 1;

    type Float = f32;

    unsafe fn zero() -> Self {
        0
    }

    unsafe fn splat(val: i32) -> Self {
        val
    }

    unsafe fn gt(self, other: Self) -> Self {
        (self > other) as i32
    }

    unsafe fn blend(self, other: Self, mask: Self) -> Self {
        if mask == 0 {
            self
        } else {
            other
        }
    }

    unsafe fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    unsafe fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    unsafe fn shl<const COUNT: i32>(self) -> Self {
        self << COUNT
    }

    unsafe fn reinterpret_as_float(self) -> Self::Float {
        f32::from_bits(self as u32)
    }

    unsafe fn load(ptr: *const i32) -> Self {
        *ptr
    }

    unsafe fn store(self, ptr: *mut i32) {
        *ptr = self;
    }
}

/// Treat an `f32` as a single-lane SIMD "vector".
impl SimdFloat for f32 {
    const LEN: usize = 1;

    type Int = i32;

    unsafe fn one() -> Self {
        1.
    }

    unsafe fn zero() -> Self {
        0.
    }

    unsafe fn splat(val: f32) -> Self {
        val
    }

    unsafe fn abs(self) -> Self {
        self.abs()
    }

    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    unsafe fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    unsafe fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    unsafe fn to_int_trunc(self) -> Self::Int {
        self as i32
    }

    unsafe fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    unsafe fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    unsafe fn ge(self, rhs: Self) -> Self {
        if self >= rhs {
            1.
        } else {
            0.
        }
    }

    unsafe fn le(self, rhs: Self) -> Self {
        if self <= rhs {
            1.
        } else {
            0.
        }
    }

    unsafe fn lt(self, rhs: Self) -> Self {
        if self < rhs {
            1.
        } else {
            0.
        }
    }

    unsafe fn max(self, rhs: Self) -> Self {
        f32::max(self, rhs)
    }

    unsafe fn blend(self, rhs: Self, mask: Self) -> Self {
        if mask == 0. {
            self
        } else {
            rhs
        }
    }

    unsafe fn load(ptr: *const f32) -> Self {
        *ptr
    }

    unsafe fn store(self, ptr: *mut f32) {
        *ptr = self;
    }
}
