//! A 16-bit floating point ("half precision") type.
//!
//! Rust does not yet have a stable built-in `f16` type, so this module defines
//! one as a wrapper around [`u16`].
//!
//! This can be replaced with f16 from the Rust standard library when that is
//! stabilized. See <https://github.com/rust-lang/rust/issues/116909>.

use crate::elem::{Elem, WrappingAdd};

/// A 16-bit floating point number, stored in IEEE 754 half-precision format.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default, PartialEq)]
#[repr(transparent)]
pub struct f16(u16);

impl f16 {
    /// Create an `f16` from its raw bit pattern.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        f16(bits)
    }

    /// Return the raw bit pattern of this value.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert an `f32` to the nearest `f16`, rounding ties to even.
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        f16(f32_to_f16(x))
    }

    /// Convert this value to an `f32`.
    ///
    /// This conversion is always exact, since every `f16` value is
    /// representable as an `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        f16_to_f32(self.0)
    }
}

impl std::fmt::Debug for f16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for f16 {
    #[inline]
    fn from(x: f32) -> f16 {
        f16::from_f32(x)
    }
}

impl From<f16> for f32 {
    #[inline]
    fn from(x: f16) -> f32 {
        x.to_f32()
    }
}

// This is implemented only because the `Elem` trait requires it.
impl WrappingAdd for f16 {
    type Output = Self;

    fn wrapping_add(self, x: Self) -> Self {
        f16::from_f32(self.to_f32() + x.to_f32())
    }
}

impl Elem for f16 {
    fn one() -> Self {
        // 1.0 in IEEE 754 half precision.
        f16(0x3C00)
    }
}

/// Convert an f16 value to f32.
///
/// The implementation is copied from the `half` crate -
/// <https://github.com/VoidStarKat/half-rs>.
pub fn f16_to_f32(i: u16) -> f32 {
    // Check for signed zero
    if i & 0x7FFFu16 == 0 {
        return f32::from_bits((i as u32) << 16);
    }

    let half_sign = (i & 0x8000u16) as u32;
    let half_exp = (i & 0x7C00u16) as u32;
    let half_man = (i & 0x03FFu16) as u32;

    // Check for an infinity or NaN when all exponent bits set
    if half_exp == 0x7C00u32 {
        // Check for signed infinity if mantissa is zero
        if half_man == 0 {
            return f32::from_bits((half_sign << 16) | 0x7F80_0000u32);
        } else {
            // NaN, keep current mantissa but also set most significiant mantissa bit
            return f32::from_bits((half_sign << 16) | 0x7FC0_0000u32 | (half_man << 13));
        }
    }

    // Calculate single-precision components with adjusted exponent
    let sign = half_sign << 16;
    // Unbias exponent
    let unbiased_exp = ((half_exp as i32) >> 10) - 15;

    // Check for subnormals, which will be normalized by adjusting exponent
    if half_exp == 0 {
        // Calculate how much to adjust the exponent by
        let e = (half_man as u16).leading_zeros() - 6;

        // Rebias and adjust exponent
        let exp = (127 - 15 - e) << 23;
        let man = (half_man << (14 + e)) & 0x7F_FF_FFu32;
        return f32::from_bits(sign | exp | man);
    }

    // Rebias exponent for a normalized normal
    let exp = ((unbiased_exp + 127) as u32) << 23;
    let man = (half_man & 0x03FFu32) << 13;
    f32::from_bits(sign | exp | man)
}

/// Convert an f16 value to f32.
///
/// The implementation is copied from the `half` crate -
/// <https://github.com/VoidStarKat/half-rs>.
pub fn f32_to_f16(value: f32) -> u16 {
    let x: u32 = value.to_bits();

    // Extract IEEE754 components
    let sign = x & 0x8000_0000u32;
    let exp = x & 0x7F80_0000u32;
    let man = x & 0x007F_FFFFu32;

    // Check for all exponent bits being set, which is Infinity or NaN
    if exp == 0x7F80_0000u32 {
        // Set mantissa MSB for NaN (and also keep shifted mantissa bits)
        let nan_bit = if man == 0 { 0 } else { 0x0200u32 };
        return ((sign >> 16) | 0x7C00u32 | nan_bit | (man >> 13)) as u16;
    }

    // The number is normalized, start assembling half precision version
    let half_sign = sign >> 16;
    // Unbias the exponent, then bias for half precision
    let unbiased_exp = ((exp >> 23) as i32) - 127;
    let half_exp = unbiased_exp + 15;

    // Check for exponent overflow, return +infinity
    if half_exp >= 0x1F {
        return (half_sign | 0x7C00u32) as u16;
    }

    // Check for underflow
    if half_exp <= 0 {
        // Check mantissa for what we can do
        if 14 - half_exp > 24 {
            // No rounding possibility, so this is a full underflow, return signed zero
            return half_sign as u16;
        }
        // Don't forget about hidden leading mantissa bit when assembling mantissa
        let man = man | 0x0080_0000u32;
        let mut half_man = man >> (14 - half_exp);
        // Check for rounding (see comment above functions)
        let round_bit = 1 << (13 - half_exp);
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            half_man += 1;
        }
        // No exponent for subnormals
        return (half_sign | half_man) as u16;
    }

    // Rebias the exponent
    let half_exp = (half_exp as u32) << 10;
    let half_man = man >> 13;
    // Check for rounding (see comment above functions)
    let round_bit = 0x0000_1000u32;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        // Round it
        ((half_sign | half_exp | half_man) + 1) as u16
    } else {
        (half_sign | half_exp | half_man) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::{f16, f16_to_f32, f32_to_f16};

    #[test]
    fn test_known_values() {
        // (f32 value, f16 bit pattern)
        let cases = [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3C00),
            (-1.0, 0xBC00),
            (2.0, 0x4000),
            (0.5, 0x3800),
            (-2.0, 0xC000),
            (65504.0, 0x7BFF), // Largest normal f16.
            (f32::INFINITY, 0x7C00),
            (f32::NEG_INFINITY, 0xFC00),
        ];

        for (f, bits) in cases {
            assert_eq!(f32_to_f16(f), bits, "f32_to_f16({f})");
            assert_eq!(f16_to_f32(bits), f, "f16_to_f32({bits:#06x})");
        }
    }

    #[test]
    fn test_overflow_to_inf() {
        assert_eq!(f32_to_f16(1e30), 0x7C00);
        assert_eq!(f32_to_f16(-1e30), 0xFC00);
    }

    #[test]
    fn test_nan() {
        let nan = f32_to_f16(f32::NAN);
        assert!(f16_to_f32(nan).is_nan());
    }

    #[test]
    fn test_subnormal() {
        // Smallest positive subnormal f16 is 2^-24.
        let smallest = 2f32.powi(-24);
        assert_eq!(f32_to_f16(smallest), 0x0001);
        assert_eq!(f16_to_f32(0x0001), smallest);

        // Values below half the smallest subnormal round to zero.
        assert_eq!(f32_to_f16(2f32.powi(-26)), 0x0000);
    }

    #[test]
    fn test_round_to_even() {
        // 2^-25 is exactly halfway between 0 and the smallest subnormal
        // (2^-24). Ties round to even, so it rounds down to zero.
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        // Just above the halfway point rounds up.
        assert_eq!(f32_to_f16(2f32.powi(-25) * 1.001), 0x0001);
    }

    #[test]
    fn test_roundtrip_exact() {
        // Every f16 -> f32 -> f16 round-trip is exact.
        for bits in 0..=u16::MAX {
            // Skip NaNs, whose bit pattern is not preserved exactly.
            let exp = (bits >> 10) & 0x1F;
            let mant = bits & 0x3FF;
            if exp == 0x1F && mant != 0 {
                continue;
            }
            let f = f16_to_f32(bits);
            assert_eq!(f32_to_f16(f), bits, "roundtrip {bits:#06x}");
        }
    }

    #[test]
    fn test_f16_wrapper() {
        assert_eq!(f16::from_f32(1.0).to_bits(), 0x3C00);
        assert_eq!(f16::from_bits(0x4000).to_f32(), 2.0);
        assert_eq!(f32::from(f16::from(3.5f32)), 3.5);
        assert_eq!(format!("{:?}", f16::from_f32(1.5)), "1.5");
    }
}
