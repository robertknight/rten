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

/// Base trait for SIMD vectors.
///
/// This provides common associated types and methods that are applicable for
/// all SIMD vectors.
#[allow(clippy::missing_safety_doc)]
pub trait Simd: Copy {
    type Elem: Copy + Default;

    /// The number of elements in the SIMD vector, if known at compile time.
    const LEN: Option<usize>;

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
        + std::ops::IndexMut<usize, Output = Self::Elem>
        + AsRef<[Self::Elem]>
        + IntoIterator<Item = Self::Elem>;

    /// Return the number of elements in the SIMD vector.
    ///
    /// This value is a constant which may be known either at compile time
    /// (eg. AVX2, Arm Neon) or only at runtime (eg. Arm SVE).
    unsafe fn len() -> usize {
        Self::LEN.unwrap()
    }

    /// Combine elements of `self` and `other` according to a mask.
    ///
    /// For each lane, if the mask value is one, return the element from
    /// `self`, otherwise return the value from `other`.
    unsafe fn select(self, other: Self, mask: Self::Mask) -> Self;

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
        assert!(len <= Self::len());
        let mut remainder = Self::zero().to_array();
        for i in 0..len {
            remainder[i] = *ptr.add(i);
        }
        Self::load(remainder.as_ref().as_ptr())
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
        assert!(len <= Self::len());
        let remainder = self.to_array();
        for i in 0..len {
            dest.add(i).write(remainder[i]);
        }
    }

    /// Return the contents of this vector as an array.
    ///
    /// This is a cheap transmute for most implementations because the SIMD
    /// type and the array have the same layout. The converse is not true
    /// because the SIMD type may have greater alignment.
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
pub const fn vec_count<S: Simd>(count: usize) -> Option<usize> {
    if let Some(len) = S::LEN {
        Some(count.div_ceil(len))
    } else {
        None
    }
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
    ///
    /// Unlike [`Simd::to_array`] this is not a simple transmute because
    /// the elements need to be converted from the architecture-specific
    /// representation of a mask to a `bool` array.
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

    /// Compute `self * rhs`, keeping the low 32-bits of each result.
    unsafe fn mul(self, rhs: Self) -> Self;

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

    /// Convert each lane in `self` to a `u8` value with saturation.
    unsafe fn saturating_cast_u8(self) -> impl Simd<Elem = u8>;

    /// Load `S::LEN` i8 values from `ptr` and sign-extend to i32.
    unsafe fn load_extend_i8(ptr: *const i8) -> Self;

    /// Interleave i8 values from the low half of `self` and `rhs`.
    unsafe fn zip_lo_i8(self, rhs: Self) -> Self;

    /// Interleave i8 values from the high half of `self` and `rhs`.
    unsafe fn zip_hi_i8(self, rhs: Self) -> Self;

    /// Interleave i16 values from the low half of `self` and `rhs`.
    unsafe fn zip_lo_i16(self, rhs: Self) -> Self;

    /// Interleave i16 values from the high half of `self` and `rhs`.
    unsafe fn zip_hi_i16(self, rhs: Self) -> Self;

    /// Horizontally sum the lanes in this vector.
    unsafe fn sum(self) -> i32 {
        let mut acc = 0;
        for x in self.to_array().as_ref() {
            acc += x;
        }
        acc
    }

    /// Bitwise XOR this value with `rhs`.
    unsafe fn xor(self, rhs: Self) -> Self;
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

    /// Convert each f32 lane to an i32 with truncation.
    unsafe fn to_int_trunc(self) -> Self::Int;

    /// Convert each f32 lane to an i32 with rounding.
    unsafe fn to_int_round(self) -> Self::Int;

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
        let reduced = self.to_array().into_iter().fold(accum, f);
        Self::splat(reduced)
    }
}

#[cfg(test)]
pub mod tests {
    /// Generate tests for a `SimdInt` implementation.
    macro_rules! test_simdint {
        ($modname:ident, $type_import_path:ty) => {
            mod $modname {
                use crate::vec::{Simd, SimdInt};
                use $type_import_path as SimdVec;

                const LEN: usize = <SimdVec as Simd>::LEN.unwrap();

                #[test]
                fn test_load_extend_i8() {
                    let src: Vec<i8> = (0..).take(LEN).collect();
                    let vec = unsafe { <SimdVec as SimdInt>::load_extend_i8(src.as_ptr()) };
                    let actual = unsafe { vec.to_array() };
                    let expected: Vec<_> = src.iter().map(|x| *x as i32).collect();
                    assert_eq!(actual.as_ref(), expected);
                }

                #[test]
                fn test_zip_lo_i8() {
                    let a_start = 0i8;
                    // `bstart` is not i8 to avoid overflow when `LEN` is 64.
                    let b_start = LEN * 4;
                    let a: Vec<_> = (a_start..).take(LEN * 4).collect();
                    let b: Vec<_> = (b_start..).map(|x| x as i8).take(LEN * 4).collect();

                    let a_vec = unsafe { SimdVec::load(a.as_ptr() as *const i32) };
                    let b_vec = unsafe { SimdVec::load(b.as_ptr() as *const i32) };

                    let i8_lo = unsafe { a_vec.zip_lo_i8(b_vec) };

                    let mut actual_i8_lo = [0i8; LEN * 4];
                    unsafe { i8_lo.store(actual_i8_lo.as_mut_ptr() as *mut i32) }

                    let expected_i8_lo: Vec<_> = (a_start..)
                        .zip(b_start..)
                        .flat_map(|(a, b)| [a, b as i8])
                        .take(LEN * 4)
                        .collect();
                    assert_eq!(actual_i8_lo, expected_i8_lo.as_slice());
                }

                #[test]
                fn test_zip_hi_i8() {
                    let a_start = 0i8;
                    // `bstart` is not i8 to avoid overflow when `LEN` is 64.
                    let b_start = LEN * 4;
                    let a: Vec<_> = (a_start..).take(LEN * 4).collect();
                    let b: Vec<_> = (b_start..).map(|x| x as i8).take(LEN * 4).collect();

                    let a_vec = unsafe { SimdVec::load(a.as_ptr() as *const i32) };
                    let b_vec = unsafe { SimdVec::load(b.as_ptr() as *const i32) };

                    let i8_hi = unsafe { a_vec.zip_hi_i8(b_vec) };

                    let mut actual_i8_hi = [0i8; LEN * 4];
                    unsafe { i8_hi.store(actual_i8_hi.as_mut_ptr() as *mut i32) }

                    let expected_i8_hi: Vec<_> = (a_start + LEN as i8 * 2..)
                        .zip(b_start + LEN * 2..)
                        .flat_map(|(a, b)| [a, b as i8])
                        .take(LEN * 4)
                        .collect();
                    assert_eq!(actual_i8_hi, expected_i8_hi.as_slice());
                }

                #[test]
                fn test_zip_lo_i16() {
                    let a_start = 0i16;
                    let b_start = LEN as i16 * 2;
                    let a: Vec<_> = (a_start..).take(LEN * 2).collect();
                    let b: Vec<_> = (b_start..).take(LEN * 2).collect();

                    let a_vec = unsafe { SimdVec::load(a.as_ptr() as *const i32) };
                    let b_vec = unsafe { SimdVec::load(b.as_ptr() as *const i32) };

                    let i16_lo = unsafe { a_vec.zip_lo_i16(b_vec) };

                    let mut actual_i16_lo = [0i16; LEN * 2];
                    unsafe { i16_lo.store(actual_i16_lo.as_mut_ptr() as *mut i32) }

                    let expected_i16_lo: Vec<_> = (a_start..)
                        .zip(b_start..)
                        .flat_map(|(a, b)| [a, b])
                        .take(LEN * 2)
                        .collect();
                    assert_eq!(actual_i16_lo, expected_i16_lo.as_slice());
                }

                #[test]
                fn test_zip_hi_i16() {
                    let a_start = 0i16;
                    let b_start = LEN as i16 * 2;
                    let a: Vec<_> = (a_start..).take(LEN * 2).collect();
                    let b: Vec<_> = (b_start..).take(LEN * 2).collect();

                    let a_vec = unsafe { SimdVec::load(a.as_ptr() as *const i32) };
                    let b_vec = unsafe { SimdVec::load(b.as_ptr() as *const i32) };

                    let i16_hi = unsafe { a_vec.zip_hi_i16(b_vec) };

                    let mut actual_i16_hi = [0i16; LEN * 2];
                    unsafe { i16_hi.store(actual_i16_hi.as_mut_ptr() as *mut i32) }

                    let expected_i16_hi: Vec<_> = (a_start + LEN as i16..)
                        .zip(b_start + LEN as i16..)
                        .flat_map(|(a, b)| [a, b])
                        .take(LEN * 2)
                        .collect();
                    assert_eq!(actual_i16_hi, expected_i16_hi.as_slice());
                }

                #[test]
                fn test_saturating_cast_u8() {
                    let src: Vec<i32> = [0, 1, -1, 256].iter().cycle().take(LEN).copied().collect();
                    let expected: Vec<u8> = src
                        .iter()
                        .map(|&x| x.clamp(0, u8::MAX as i32) as u8)
                        .collect();
                    let vec = unsafe { <SimdVec as Simd>::load(src.as_ptr()) };
                    let vec_u8 = unsafe { vec.saturating_cast_u8().to_array() };

                    assert_eq!(vec_u8.as_ref(), expected);
                }

                #[test]
                fn test_sum() {
                    let src: Vec<i32> = (0..).take(LEN).collect();
                    let vec = unsafe { <SimdVec as Simd>::load(src.as_ptr()) };
                    let actual = unsafe { vec.sum() };
                    let expected: i32 = src.iter().sum();
                    assert_eq!(actual, expected);
                }
            }
        };
    }

    pub(crate) use test_simdint;
}
