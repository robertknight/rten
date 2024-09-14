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
//!
//! ## Inlining and target features
//!
//! Correct use of inlining and target features, in impls of these traits and
//! generic functions using them, are critical to getting the performance
//! benefits. If an intrinic is not inlined, the function call overhead can
//! negate the benefits of using SIMD in the first place.
//!
//! Implementations of SIMD traits should add `#[inline]` and
//! `#[target_feature(enable = "feature"]` attributes, where the feature names
//! match the wrapped architecture-specific intrinics. An exception is for
//! intrinsics which are always available in a given build configuration (eg.
//! we assume SSE is always available under x86_64 and Neon under Arm).
//!
//! Generic functions which use SIMD traits must have `#[inline(always)]`
//! annotations, including on any closures in their bodies. These generic
//! functions must then be wrapped in target feature-specific wrappers which
//! have `#[target_feature]`s that are a union of all those used in the
//! implementation.
//!
//! # Safety
//!
//! The caller must ensure that the SIMD instructions used by a type
//! implementing a SIMD trait are available on the current system.
//!
//! All architecture-specific functions in SIMD traits are unsafe due to
//! limitations of Rust's `#[target_feature]` macro. See
//! <https://rust-lang.github.io/rfcs/2396-target-feature-1.1.html>. Also as a
//! consequence of this, standard operations like add, multiply etc. are
//! implemented as functions in this trait rather than using the standard trait
//! from `std::ops`.

/// Maximum number of 32-bit lanes in a vector register across all supported
/// architectures.
pub const MAX_LEN: usize = 16;

/// Base trait for SIMD vectors.
///
/// This provides common associated types and methods that are applicable for
/// all SIMD vectors.
#[allow(clippy::missing_safety_doc)]
pub trait SimdVal: Copy + Sized {
    /// The number of elements in the SIMD vector.
    const LEN: usize;

    /// The type used by operations that use or return masks.
    ///
    /// This should be the same for all vector types with a given number of
    /// lanes in a particular architecture.
    type Mask: SimdMask;
}

/// Return the number of SIMD vectors required to hold `count` elements.
pub const fn vec_count<S: SimdVal>(count: usize) -> usize {
    count.div_ceil(S::LEN)
}

/// Trait implemented by SIMD masks.
#[allow(clippy::missing_safety_doc)]
pub trait SimdMask: Copy {
    /// A representation of this mask as a bool array.
    type Array: std::ops::Index<usize, Output = bool>;

    /// Return a bitwise AND of self and `rhs`.
    unsafe fn and(self, rhs: Self) -> Self;

    /// Convert this SIMD mask to a boolean array.
    unsafe fn to_array(self) -> Self::Array;
}

/// Trait for SIMD vectors containing 32-bit integers.
#[allow(clippy::missing_safety_doc)]
pub trait SimdInt: SimdVal {
    /// The type produced by an operation that converts each element in this
    /// vector to a float.
    type Float: SimdFloat<Int = Self, Mask = Self::Mask>;

    /// The contents of a vector as an array.
    type Array: std::ops::Index<usize, Output = i32>;

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

    /// Return the contents of this vector as an array.
    unsafe fn to_array(self) -> Self::Array;
}

/// Trait for SIMD vectors containing single-precision floats.
#[allow(clippy::missing_safety_doc)]
pub trait SimdFloat: SimdVal {
    /// The type of vector produced by operations that convert this vector
    /// to a vector of ints.
    type Int: SimdInt<Float = Self, Mask = Self::Mask>;

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
