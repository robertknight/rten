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

impl Identities for f32 {
    fn one() -> f32 {
        1.
    }

    fn zero() -> f32 {
        0.
    }
}

impl Identities for i32 {
    fn one() -> i32 {
        1
    }
    fn zero() -> i32 {
        0
    }
}

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
    fn max_val() -> Self {
        f32::INFINITY
    }

    fn min_val() -> Self {
        f32::NEG_INFINITY
    }

    fn max(self, other: f32) -> f32 {
        self.max(other)
    }

    fn min(self, other: f32) -> f32 {
        self.min(other)
    }
}

/// FastDiv optimizes repeated integer division or modulus by the same divisor
/// in the case where the divisor is a power of 2.
///
/// This is useful because integer division is a slow operation. See
/// https://stackoverflow.com/q/70132913/434243. In the power-of-2 case, this
/// can be replaced with simple shifts and masks.
#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)] // No longer used currently, but likely useful in future.
pub enum FastDiv<T> {
    /// Divisor is a power of 2. Payload is `divisor.ilog2()`.
    PowerOf2(u32),
    /// General case. Payload is the divisor.
    Fallback(T),
}

macro_rules! impl_fastdiv {
    ($int_type:ident) => {
        #[allow(dead_code)] // No longer used currently, but likely useful in future.
        impl FastDiv<$int_type> {
            /// Create a new `FastDiv` which can compute `lhs / divisor` or
            /// `lhs % divisor`. Panics if divisor is zero.
            pub fn divide_by(divisor: $int_type) -> FastDiv<$int_type> {
                let log = divisor.ilog2();
                if 1 << log == divisor {
                    FastDiv::PowerOf2(log)
                } else {
                    FastDiv::Fallback(divisor)
                }
            }

            /// Compute `lhs / self`.
            #[inline]
            pub fn divide(self, lhs: $int_type) -> $int_type {
                match self {
                    FastDiv::PowerOf2(divisor_log2) => lhs >> divisor_log2,
                    FastDiv::Fallback(divisor) => lhs / divisor,
                }
            }

            /// Compute `lhs % self`.
            #[inline]
            pub fn rem(self, lhs: $int_type) -> $int_type {
                match self {
                    FastDiv::PowerOf2(divisor_log2) => {
                        let mask = (1 << divisor_log2) - 1;
                        lhs & mask
                    }
                    FastDiv::Fallback(divisor) => lhs % divisor,
                }
            }
        }
    };
}

// Add more types as needed. The `FastDiv::divide_by` impl currently assumes an
// unsigned type.
impl_fastdiv!(usize);

/// Marker trait for "plain old data".
///
/// POD types which are simple value types that impl `Copy`, have no padding,
/// and for which any bit pattern is valid.
///
/// This means an arbitrary byte sequence can be converted to this type, as
/// long as the byte sequence length is a multiple of the type's size.
pub trait Pod: Copy {}
impl Pod for i8 {}
impl Pod for i32 {}
impl Pod for f32 {}
impl Pod for u8 {}

#[cfg(test)]
mod tests {
    use super::FastDiv;

    #[test]
    fn test_fast_div_divide() {
        let test = |divisor| {
            let div = FastDiv::divide_by(divisor);
            for lhs in 0..20 {
                assert_eq!(
                    div.divide(lhs),
                    lhs / divisor,
                    "mismatch with lhs = {}, divisor = {}",
                    lhs,
                    divisor
                );
            }
        };

        test(1);

        // Non-power of two.
        test(3);

        // Powers of two.
        test(2);
        test(8);
    }

    #[test]
    fn test_fast_div_rem() {
        let test = |divisor| {
            let div = FastDiv::divide_by(divisor);
            for lhs in 0..20 {
                assert_eq!(
                    div.rem(lhs),
                    lhs % divisor,
                    "mismatch with lhs = {}, divisor = {}",
                    lhs,
                    divisor
                );
            }
        };

        test(1);

        // Non-power of two.
        test(3);

        // Powers of two.
        test(2);
        test(8);
    }
}
