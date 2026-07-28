//! A 16-bit "brain floating point" type.
//!
//! `bf16` has the same exponent range as [`f32`], but only 8 bits of
//! significand (7 stored plus the implicit leading bit). This makes conversion
//! to and from `f32` a matter of appending or removing 16 low mantissa bits,
//! which is why it is often used in place of [`f16`](crate::f16) for machine
//! learning workloads.
//!
//! Unlike `f16`, CPUs rarely provide instructions to convert between `bf16` and
//! `f32`, so the conversions here are implemented using shifts and additions.

use crate::elem::{Elem, WrappingAdd};

/// A 16-bit floating point number, stored in "brain float" format.
///
/// This is the top 16 bits of the IEEE 754 single-precision representation of
/// the value. Hence it has the same exponent range as `f32` but a much smaller
/// significand.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default, PartialEq)]
#[repr(transparent)]
pub struct bf16(u16);

impl bf16 {
    /// Create a `bf16` from its raw bit pattern.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        bf16(bits)
    }

    /// Return the raw bit pattern of this value.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert an `f32` to the nearest `bf16`, rounding ties to even.
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        bf16(f32_to_bf16(x))
    }

    /// Convert this value to an `f32`.
    ///
    /// This conversion is always exact, since every `bf16` value is
    /// representable as an `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        bf16_to_f32(self.0)
    }
}

impl std::fmt::Debug for bf16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for bf16 {
    #[inline]
    fn from(x: f32) -> bf16 {
        bf16::from_f32(x)
    }
}

impl From<bf16> for f32 {
    #[inline]
    fn from(x: bf16) -> f32 {
        x.to_f32()
    }
}

// This is implemented only because the `Elem` trait requires it.
impl WrappingAdd for bf16 {
    type Output = Self;

    fn wrapping_add(self, x: Self) -> Self {
        bf16::from_f32(self.to_f32() + x.to_f32())
    }
}

impl Elem for bf16 {
    fn one() -> Self {
        // 1.0 in brain float format.
        bf16(0x3F80)
    }
}

/// Convert a `bf16` value to `f32`.
///
/// A `bf16` is the most significant 16 bits of the `f32` with the same value,
/// so this just needs to append zeros for the missing mantissa bits.
#[inline]
pub fn bf16_to_f32(i: u16) -> f32 {
    f32::from_bits((i as u32) << 16)
}

/// Convert an `f32` value to `bf16`, rounding to nearest with ties to even.
///
/// Values whose magnitude exceeds the `bf16` range are rounded to infinity.
/// Since `bf16` has the same exponent range as `f32` this can only happen for
/// values very close to `f32::MAX`.
#[inline]
pub fn f32_to_bf16(value: f32) -> u16 {
    let x = value.to_bits();

    // NaNs are handled separately because the rounding below can turn a NaN
    // whose mantissa has only low-order bits set into an infinity.
    if x & 0x7FFF_FFFFu32 > 0x7F80_0000u32 {
        // Set the MSB of the mantissa so the result is a quiet NaN.
        return ((x >> 16) as u16) | 0x0040u16;
    }

    // Round to nearest, ties to even. Adding half of the discarded bits'
    // range rounds up when the remainder is more than half, and adding the
    // low bit of the retained mantissa breaks ties towards even.
    //
    // This cannot overflow because the largest non-NaN input is `0xFF80_0000`
    // (negative infinity).
    let round_bias = 0x7FFFu32 + ((x >> 16) & 1);
    ((x + round_bias) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{bf16, bf16_to_f32, f32_to_bf16};

    #[test]
    fn test_known_values() {
        // (f32 value, bf16 bit pattern)
        let cases = [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3F80),
            (-1.0, 0xBF80),
            (2.0, 0x4000),
            (0.5, 0x3F00),
            (-2.0, 0xC000),
            (f32::INFINITY, 0x7F80),
            (f32::NEG_INFINITY, 0xFF80),
        ];

        for (f, bits) in cases {
            assert_eq!(f32_to_bf16(f), bits, "f32_to_bf16({f})");
            assert_eq!(bf16_to_f32(bits), f, "bf16_to_f32({bits:#06x})");
        }
    }

    #[test]
    fn test_overflow_to_inf() {
        // Unlike f16, bf16 has the same exponent range as f32, so only values
        // very close to `f32::MAX` overflow.
        assert_eq!(f32_to_bf16(f32::MAX), 0x7F80);
        assert_eq!(f32_to_bf16(-f32::MAX), 0xFF80);

        // The largest normal bf16 is just under `f32::MAX` and is preserved.
        assert_eq!(f32_to_bf16(bf16_to_f32(0x7F7F)), 0x7F7F);
    }

    #[test]
    fn test_nan() {
        let nan = f32_to_bf16(f32::NAN);
        assert!(bf16_to_f32(nan).is_nan());

        // A NaN whose set mantissa bits are all discarded by the conversion
        // must still convert to a NaN rather than an infinity.
        let low_bit_nan = f32::from_bits(0x7F80_0001);
        assert!(bf16_to_f32(f32_to_bf16(low_bit_nan)).is_nan());
    }

    #[test]
    fn test_subnormal() {
        // The smallest positive subnormal bf16 is 2^-133, which is itself a
        // subnormal `f32`. `powi` can't be used to construct these values as
        // it overflows computing the positive power first.
        let smallest = f32::from_bits(0x0001_0000);
        assert_eq!(f32_to_bf16(smallest), 0x0001);
        assert_eq!(bf16_to_f32(0x0001), smallest);

        // Values below half the smallest subnormal round to zero. This is
        // 2^-135, ie. a quarter of the smallest subnormal.
        assert_eq!(f32_to_bf16(f32::from_bits(0x0000_4000)), 0x0000);
    }

    #[test]
    fn test_round_to_even() {
        // The step between consecutive bf16 values just above 1.0 is 2^-7, so
        // 1.0 + 2^-8 is exactly halfway between 1.0 (0x3F80) and the next
        // bf16 (0x3F81). Ties round to even, so it rounds down.
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3F80);
        // Just above the halfway point rounds up.
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8) * 1.001), 0x3F81);
        // The tie above 0x3F81 rounds up, to the even value 0x3F82.
        assert_eq!(f32_to_bf16(bf16_to_f32(0x3F81) + 2f32.powi(-8)), 0x3F82);
    }

    #[test]
    fn test_roundtrip_exact() {
        // Every bf16 -> f32 -> bf16 round-trip is exact.
        for bits in 0..=u16::MAX {
            // Skip NaNs, whose bit pattern is not preserved exactly.
            let exp = (bits >> 7) & 0xFF;
            let mant = bits & 0x7F;
            if exp == 0xFF && mant != 0 {
                continue;
            }
            let f = bf16_to_f32(bits);
            assert_eq!(f32_to_bf16(f), bits, "roundtrip {bits:#06x}");
        }
    }

    #[test]
    fn test_bf16_wrapper() {
        assert_eq!(bf16::from_f32(1.0).to_bits(), 0x3F80);
        assert_eq!(bf16::from_bits(0x4000).to_f32(), 2.0);
        assert_eq!(f32::from(bf16::from(3.5f32)), 3.5);
        assert_eq!(format!("{:?}", bf16::from_f32(1.5)), "1.5");
    }
}
