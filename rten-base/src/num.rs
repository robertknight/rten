//! Numeric traits and functions.

/// Trait for int -> bool conversions.
///
/// The conversion matches how these conversions work in most popular languages
/// where zero is treated as false and other values coerce to true.
pub trait AsBool {
    fn as_bool(&self) -> bool;
}

impl AsBool for bool {
    fn as_bool(&self) -> bool {
        *self
    }
}

impl AsBool for i32 {
    fn as_bool(&self) -> bool {
        *self != 0
    }
}

/// Trait for value-preserving, non-fallible conversion to usize.
///
/// This is like `x as usize` but only implemented for types where conversion
/// will preserve the value, under the assumption that the pointer width is
/// 32 or 64 bits. This differs from Rust's `Into<usize>` impls which do not
/// support u32 -> usize conversion because pointers are 16-bits on some
/// platforms. We assume that such platforms are not relevant for consumers
/// of this crate.
///
/// This trait should be used instead of `as usize` where possible, as it
/// ensures the conversion is value-preserving.
pub trait AsUsize {
    fn as_usize(self) -> usize;
}

macro_rules! impl_as_usize {
    ($type:ty) => {
        impl AsUsize for $type {
            fn as_usize(self) -> usize {
                self as usize
            }
        }
    };
}

impl_as_usize!(u8);
impl_as_usize!(u16);
impl_as_usize!(u32);
impl_as_usize!(usize);

/// Trait indicating whether type is an integer or float.
pub trait IsInt {
    fn is_int() -> bool;
}

impl IsInt for f32 {
    fn is_int() -> bool {
        false
    }
}

impl IsInt for i32 {
    fn is_int() -> bool {
        true
    }
}

/// Trait providing additive and multiplicative identities.
pub trait Identities {
    fn one() -> Self;
    fn zero() -> Self;
}

macro_rules! impl_float_identities {
    ($type:ty) => {
        impl Identities for $type {
            fn one() -> Self {
                1.
            }

            fn zero() -> Self {
                0.
            }
        }
    };
}

macro_rules! impl_int_identities {
    ($type:ty) => {
        impl Identities for $type {
            fn one() -> Self {
                1
            }

            fn zero() -> Self {
                0
            }
        }
    };
}

impl_float_identities!(f32);
impl_int_identities!(isize);
impl_int_identities!(i32);
impl_int_identities!(i8);
impl_int_identities!(u8);

/// Test if a number is a float NaN ("Not a number") value.
pub trait IsNaN {
    /// Return true if the current value is a NaN. See [`f32::is_nan`].
    ///
    /// This is always false for integer types.
    #[allow(clippy::wrong_self_convention)] // Match `f32::is_nan` etc.
    fn is_nan(self) -> bool;
}

macro_rules! impl_isnan_float {
    ($type:ty) => {
        impl IsNaN for $type {
            fn is_nan(self) -> bool {
                <$type>::is_nan(self)
            }
        }
    };
}
macro_rules! impl_isnan_int {
    ($type:ty) => {
        impl IsNaN for $type {
            fn is_nan(self) -> bool {
                false
            }
        }
    };
}

impl_isnan_float!(f32);
impl_isnan_int!(i32);
impl_isnan_int!(i8);
impl_isnan_int!(u8);

/// Convert between a primitive type and an array of bytes in little-endian
/// order.
pub trait LeBytes {
    /// The `[u8; N]` array type holding the serialized bytes for this value.
    type Bytes: AsRef<[u8]> + for<'a> TryFrom<&'a [u8], Error = std::array::TryFromSliceError>;

    fn from_le_bytes(bytes: Self::Bytes) -> Self;
    fn to_le_bytes(self) -> Self::Bytes;
}

macro_rules! impl_le_bytes {
    ($type:ty, $size:literal) => {
        impl LeBytes for $type {
            type Bytes = [u8; $size];

            fn from_le_bytes(bytes: Self::Bytes) -> Self {
                <$type>::from_le_bytes(bytes)
            }

            fn to_le_bytes(self) -> Self::Bytes {
                <$type>::to_le_bytes(self)
            }
        }
    };
}

impl_le_bytes!(i8, 1);
impl_le_bytes!(u8, 1);
impl_le_bytes!(i32, 4);
impl_le_bytes!(f32, 4);
impl_le_bytes!(u32, 4);
impl_le_bytes!(i64, 8);
impl_le_bytes!(u64, 8);

pub trait MinMax {
    /// Return the maximum value for this type.
    #[allow(unused)] // Not used yet, but included for completeness
    fn max_val() -> Self;

    /// Return the minimum value for this type.
    fn min_val() -> Self;

    /// Return the minimum of `self` and `other`.
    #[allow(unused)] // Not used yet, but included for completeness
    fn min(self, other: Self) -> Self;

    /// Return the maximum of `self` and `other`.
    fn max(self, other: Self) -> Self;
}

impl MinMax for f32 {
    #[inline]
    fn max_val() -> Self {
        f32::INFINITY
    }

    #[inline]
    fn min_val() -> Self {
        f32::NEG_INFINITY
    }

    #[inline]
    fn max(self, other: f32) -> f32 {
        self.max(other)
    }

    #[inline]
    fn min(self, other: f32) -> f32 {
        self.min(other)
    }
}

macro_rules! impl_minmax_int {
    ($type:ty) => {
        impl MinMax for $type {
            #[inline]
            fn max_val() -> Self {
                Self::MAX
            }

            #[inline]
            fn min_val() -> Self {
                Self::MIN
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                Ord::max(self, other)
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                Ord::min(self, other)
            }
        }
    };
}
impl_minmax_int!(i32);

/// Compute `x / y` rounding up.
///
/// Replace with standard library method when stabilized. See
/// <https://github.com/rust-lang/rust/issues/88581>.
pub fn div_ceil(x: isize, y: isize) -> isize {
    let d = x / y;
    let r = x % y;

    // Int division rounds towards zero. This rounds up if the result is
    // negative or down if the result is positive. We always want to round up,
    // hence we need to adjust only if the result is positive, which occurs
    // when the operands have the same sign. See https://stackoverflow.com/a/924160.
    if r != 0 && x.signum() == y.signum() {
        d + 1
    } else {
        d
    }
}

/// Provides Rust's `as` conversions as a trait.
///
/// See <https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions>
/// for details.
pub trait Cast<T> {
    /// Convert `self` to type T using `self as T`.
    fn cast(self) -> T;
}

macro_rules! impl_cast {
    ($src:ty, $dest:ty) => {
        impl Cast<$dest> for $src {
            fn cast(self) -> $dest {
                self as $dest
            }
        }
    };
}
impl_cast!(i32, f32);
impl_cast!(f32, i32);
impl_cast!(i8, u8);
impl_cast!(u8, i8);

#[cfg(test)]
mod tests {
    use super::div_ceil;

    #[test]
    fn test_div_ceil() {
        // Same-sign operands, no rounding required.
        assert_eq!(div_ceil(10, 5), 2);
        // Same-sign operands, rounded up.
        assert_eq!(div_ceil(11, 5), 3);
        // Opposite-sign operands, no rounding required.
        assert_eq!(div_ceil(10, -5), -2);
        // Opposite-sign operands, rounded up.
        assert_eq!(div_ceil(11, -5), -2);
    }
}
