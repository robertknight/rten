//! Traits to support writing portable SIMD code.
//!
//! Unlike [std::simd] (as of Rust v1.75.0), this compiles against stable Rust.
//!
//! This module provides traits to support writing generic SIMD code which
//! can be compiled to work with different instruction sets. These traits are
//! implemented for architecture-specific SIMD types (or wrappers around them,
//! in some cases). There is also a generic implementation for Rust primitives
//! (eg. f32, i32) which treats the primitive as a single-element SIMD type.
//! This is useful both as a fallback implementation and for debugging.

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

// The wasm module is exposed because it contains wrapper types which are needed
// to use the functionality outside of this crate.
#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

use crate::MAX_LEN;

/// Trait for SIMD vectors containing 32-bit integers.
///
/// # Safety
///
/// The caller must ensure that the SIMD instructions used by a type
/// implementing this trait are available on the current system.
///
/// All functions in this trait are unsafe due to limitations of Rust's
/// `#[target_feature]` macro. See
/// <https://rust-lang.github.io/rfcs/2396-target-feature-1.1.html>. Also as
/// a consequence of this, standard operations like add, multiply etc. are
/// implemented as functions in this trait rather than using the standard
/// trait from `std::ops`.
#[allow(clippy::missing_safety_doc)]
pub trait SimdInt: Copy + Sized {
    /// The number of elements in the SIMD vector.
    const LEN: usize;

    /// The type produced by an operation that converts each element in this
    /// vector to a float.
    type Float: SimdFloat<Int = Self>;

    /// The type used by operations that use or return masks.
    type Mask: Copy;

    /// Return a new vector with all elements set to zero.
    #[inline]
    unsafe fn zero() -> Self {
        Self::splat(0)
    }

    /// Broadcast `val` to all elements in a new vector.
    ///
    /// # Safety
    /// The caller must ensure SIMD operations on this type are supported.
    unsafe fn splat(val: i32) -> Self;

    /// Return a mask indicating whether `self > other`.
    unsafe fn gt(self, other: Self) -> Self::Mask;

    /// Return a mask indicating whether `self >= other`.
    unsafe fn ge(self, other: Self) -> Self::Mask;

    /// Return a mask indicating whether `self <= other`.
    unsafe fn le(self, other: Self) -> Self::Mask;

    /// Return a mask indicating whether `self < other`.
    unsafe fn lt(self, other: Self) -> Self::Mask;

    /// Return a mask indicating where `self == other`.
    unsafe fn eq(self, other: Self) -> Self::Mask;

    /// Select elements from this vector or `other` according to a mask.
    ///
    /// For each lane, if the mask value is zero, return the element from
    /// `self`, otherwise return the value from `other`.
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self;

    /// Compute `self + rhs`.
    unsafe fn add(self, rhs: Self) -> Self;

    /// Compute `self - rhs`.
    unsafe fn sub(self, rhs: Self) -> Self;

    /// Shift the bits in each element left by `count`.
    unsafe fn shl<const COUNT: i32>(self) -> Self;

    /// Reinterpret the bits of each element as a float.
    unsafe fn reinterpret_as_float(self) -> Self::Float;

    /// Reinterpret the bits of this value as a mask for use in float
    /// operations.
    ///
    /// Callers must set all bits in a lane to 1 for the mask to be "on" for
    /// that lane or all bits to 0 for the mask to be "off" for that lane.
    /// Hence `-1` means "all lanes on" and `0` means "all lanes off". Note that
    /// some architectures may only actually check a subset of bits within a
    /// lane.
    unsafe fn to_float_mask(self) -> <Self::Float as SimdFloat>::Mask;

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
/// # Safety
///
/// The caller must ensure that the SIMD instructions used by a type
/// implementing this trait are available on the current system.
///
/// All functions in this trait are unsafe due to limitations of Rust's
/// `#[target_feature]` macro. See
/// <https://rust-lang.github.io/rfcs/2396-target-feature-1.1.html>. Also as
/// a consequence of this, standard operations like add, multiply etc. are
/// implemented as functions in this trait rather than using the standard
/// trait from `std::ops`.
#[allow(clippy::missing_safety_doc)]
pub trait SimdFloat: Copy + Sized {
    /// The number of elements in the SIMD vector.
    const LEN: usize;

    /// The type of vector produced by operations that convert this vector
    /// to a vector of ints.
    type Int: SimdInt<Float = Self>;

    /// The type used by operations that use or return masks.
    type Mask: Copy;

    /// Shorthand for `Self::splat(1.0)`.
    #[inline]
    unsafe fn one() -> Self {
        Self::splat(1.0)
    }

    /// Shorthand for `Self::splat(0.0)`.
    #[inline]
    unsafe fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Compute `-self`.
    #[inline]
    unsafe fn neg(self) -> Self {
        Self::zero().sub(self)
    }

    /// Compute `1. / self`.
    #[inline]
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
    unsafe fn ge(self, rhs: Self) -> Self::Mask;

    /// Compute a mask containing `self <= rhs`.
    unsafe fn le(self, rhs: Self) -> Self::Mask;

    /// Compute a mask containing `self < rhs`.
    unsafe fn lt(self, rhs: Self) -> Self::Mask;

    /// Compute the maximum of `self` and `rhs`.
    unsafe fn max(self, rhs: Self) -> Self;

    /// Combine elements of `self` and `rhs` according to a mask.
    ///
    /// For each lane, if the mask value is zero, return the element from
    /// `self`, otherwise return the value from `other`.
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self;

    /// Evaluate a polynomial using Horner's method.
    ///
    /// Computes `self * coeffs[0] + self^2 * coeffs[1] ... self^n * coeffs[N]`
    #[inline]
    unsafe fn poly_eval(self, coeffs: &[Self]) -> Self {
        let mut y = coeffs[coeffs.len() - 1];
        for i in (0..coeffs.len() - 1).rev() {
            y = y.mul_add(self, coeffs[i]);
        }
        y.mul(self)
    }

    /// Sum all the lanes in this vector.
    ///
    /// The ordering of the summation is not specified. This can lead to small
    /// differences in results depending on the architecture.
    unsafe fn sum(self) -> f32;

    /// Load `Self::LEN` floats from the memory address at `ptr`.
    ///
    /// Implementations must not require `ptr` to be aligned.
    ///
    /// Safety: The caller must ensure `ptr` points to at least `Self::LEN`
    /// floats.
    unsafe fn load(ptr: *const f32) -> Self;

    /// Load `Self::LEN` values from the base memory address at `ptr` plus
    /// offsets in `offsets`, excluding elements where `mask` is off.
    ///
    /// Offsets are expressed in terms of elements, not bytes. Elements of the
    /// result are set to zero where the mask is off.
    ///
    /// # Safety
    ///
    /// All offsets in `offsets` and the offset zero must be valid for indexing
    /// into `ptr`. The requirement for offset zero to be valid is needed on
    /// architectures which do not have a gather instruction.
    unsafe fn gather_mask(ptr: *const f32, offsets: Self::Int, mask: Self::Mask) -> Self;

    /// Store `Self::LEN` floats to the memory address at `ptr`.
    ///
    /// Implementations must not require `ptr` to be aligned.
    ///
    /// Safety: The caller must ensure `ptr` points to a buffer with space for
    /// at least `Self::LEN` floats.
    unsafe fn store(self, ptr: *mut f32);

    /// Reduce the elements in this vector to a single value using `f`, then
    /// return a new vector with the accumulated value broadcast to each lane.
    #[inline]
    unsafe fn fold_splat<F: Fn(f32, f32) -> f32>(self, accum: f32, f: F) -> Self {
        let mut elements = [accum; MAX_LEN];
        self.store(elements.as_mut_ptr());
        let reduced = elements.into_iter().fold(accum, f);
        Self::splat(reduced)
    }

    /// Prefetch the cache line containing `data`, for reading.
    #[inline]
    unsafe fn prefetch(_data: *const f32) {
        // Noop
    }

    /// Prefetch the cache line containing `data`, for writing.
    #[inline]
    unsafe fn prefetch_write(_data: *mut f32) {
        // Noop
    }
}

/// Treat an `i32` as a single-lane SIMD "vector".
impl SimdInt for i32 {
    const LEN: usize = 1;

    type Float = f32;
    type Mask = bool;

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
    unsafe fn to_float_mask(self) -> <Self::Float as SimdFloat>::Mask {
        self != 0
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

/// Treat an `f32` as a single-lane SIMD "vector".
impl SimdFloat for f32 {
    const LEN: usize = 1;

    type Int = i32;
    type Mask = bool;

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
