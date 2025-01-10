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

use std::mem::MaybeUninit;

/// Maximum number of 32-bit lanes in a vector register across all supported
/// architectures.
pub const MAX_LEN: usize = 16;

/// Base trait for SIMD vectors.
///
/// This provides common associated types and methods that are applicable for
/// all SIMD vectors.
#[allow(clippy::missing_safety_doc)]
pub trait Simd: Copy + Sized {
    type Elem: Copy + Default;

    /// The number of elements in the SIMD vector.
    const LEN: usize;

    /// The type used by operations that use or return masks.
    ///
    /// This should be the same for all vector types with a given number of
    /// lanes in a particular architecture.
    type Mask: SimdMask;

    /// The contents of a vector as an array.
    ///
    /// This type should always be `[Self::ELEM; Self::LEN]`. The `to_array`
    /// method returns this associated type rather than a concrete array due to
    /// const generics limitations.
    type Array: Copy
        + std::fmt::Debug
        + std::ops::Index<usize, Output = Self::Elem>
        + AsRef<[Self::Elem]>;

    /// Combine elements of `self` and `rhs` according to a mask.
    ///
    /// For each lane, if the mask value is zero, return the element from
    /// `self`, otherwise return the value from `other`.
    unsafe fn blend(self, other: Self, mask: Self::Mask) -> Self;

    /// Broadcast `val` to all elements in a new vector.
    ///
    /// # Safety
    /// The caller must ensure SIMD operations on this type are supported.
    unsafe fn splat(val: Self::Elem) -> Self;

    /// Load `Self::LEN` values from the memory address at `ptr`.
    ///
    /// Implementations must not require `ptr` to be aligned.
    ///
    /// Safety: The caller must ensure `ptr` points to at least `Self::LEN`
    /// values.
    unsafe fn load(ptr: *const Self::Elem) -> Self;

    /// Load `len` values from `ptr` into a vector and zero the unused lanes.
    ///
    /// Panics if `len > Self::LEN`.
    #[inline]
    unsafe fn load_partial(ptr: *const Self::Elem, len: usize) -> Self {
        assert!(len <= Self::LEN);
        let mut remainder = [Self::Elem::default(); MAX_LEN];
        for i in 0..len {
            remainder[i] = *ptr.add(i);
        }
        Self::load(remainder.as_ptr())
    }

    /// Prefetch the cache line containing `data`, for reading.
    #[inline]
    unsafe fn prefetch(_data: *const Self::Elem) {
        // Noop
    }

    /// Prefetch the cache line containing `data`, for writing.
    #[inline]
    unsafe fn prefetch_write(_data: *mut Self::Elem) {
        // Noop
    }

    /// Store `Self::LEN` values to the memory address at `ptr`.
    ///
    /// Implementations must not require `ptr` to be aligned.
    ///
    /// Safety: The caller must ensure `ptr` points to a buffer with space for
    /// at least `Self::LEN` values.
    unsafe fn store(self, ptr: *mut Self::Elem);

    /// Store the first `len` lanes from `self` into `dest`.
    ///
    /// Panics if `len > Self::LEN`.
    #[inline]
    unsafe fn store_partial(self, dest: *mut Self::Elem, len: usize) {
        assert!(len <= Self::LEN);
        let mut remainder = [MaybeUninit::uninit(); MAX_LEN];
        self.store(remainder.as_mut_ptr() as *mut Self::Elem);
        for i in 0..len {
            dest.add(i).write(remainder[i].assume_init());
        }
    }

    /// Return the contents of this vector as an array.
    unsafe fn to_array(self) -> Self::Array;

    /// Return a new vector with all elements set to zero.
    #[inline]
    unsafe fn zero() -> Self
    where
        Self::Elem: Default,
    {
        Self::splat(Self::Elem::default())
    }
}

/// Return the number of SIMD vectors required to hold `count` elements.
pub const fn vec_count<S: Simd>(count: usize) -> usize {
    count.div_ceil(S::LEN)
}

/// Trait implemented by SIMD masks.
#[allow(clippy::missing_safety_doc)]
pub trait SimdMask: Copy {
    /// A representation of this mask as a bool array.
    type Array: Copy + Default + std::ops::Index<usize, Output = bool> + std::ops::IndexMut<usize>;

    /// Return a bitwise AND of self and `rhs`.
    unsafe fn and(self, rhs: Self) -> Self;

    /// Return a mask with the first `n` lanes set.
    unsafe fn first_n(n: usize) -> Self {
        let mut array = Self::Array::default();
        for i in 0..n {
            array[i] = true;
        }
        Self::from_array(array)
    }

    /// Convert this SIMD mask to a boolean array.
    unsafe fn to_array(self) -> Self::Array;

    /// Create a SIMD mask from a boolean array.
    unsafe fn from_array(mask: Self::Array) -> Self;
}

/// Trait for SIMD vectors containing 32-bit integers.
#[allow(clippy::missing_safety_doc)]
pub trait SimdInt: Simd<Elem = i32> {
    /// The type produced by an operation that converts each element in this
    /// vector to a float.
    type Float: SimdFloat<Int = Self, Mask = Self::Mask>;

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

    /// Compute `self + rhs`.
    unsafe fn add(self, rhs: Self) -> Self;

    /// Compute `self - rhs`.
    unsafe fn sub(self, rhs: Self) -> Self;

    /// Compute minimum of self and `rhs`.
    unsafe fn min(self, rhs: Self) -> Self;

    /// Compute maximum of self and `rhs`.
    unsafe fn max(self, rhs: Self) -> Self;

    /// Shift the bits in each element left by `count`.
    unsafe fn shl<const COUNT: i32>(self) -> Self;

    /// Reinterpret the bits of each element as a float.
    unsafe fn reinterpret_as_float(self) -> Self::Float;
}

/// Trait for SIMD vectors containing single-precision floats.
#[allow(clippy::missing_safety_doc)]
pub trait SimdFloat: Simd<Elem = f32> {
    /// The type of vector produced by operations that convert this vector
    /// to a vector of ints.
    type Int: SimdInt<Float = Self, Mask = Self::Mask>;

    /// Shorthand for `Self::splat(1.0)`.
    #[inline]
    unsafe fn one() -> Self {
        Self::splat(1.0)
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

    /// Compute the minimum of `self` and `rhs`.
    unsafe fn min(self, rhs: Self) -> Self;

    /// Compute the maximum of `self` and `rhs`.
    unsafe fn max(self, rhs: Self) -> Self;

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

    /// Reduce the elements in this vector to a single value using `f`, then
    /// return a new vector with the accumulated value broadcast to each lane.
    #[inline]
    unsafe fn fold_splat<F: Fn(f32, f32) -> f32>(self, accum: f32, f: F) -> Self {
        let mut elements = [accum; MAX_LEN];
        self.store(elements.as_mut_ptr());
        let reduced = elements.into_iter().fold(accum, f);
        Self::splat(reduced)
    }
}
