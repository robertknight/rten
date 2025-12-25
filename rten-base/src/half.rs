//! Operations on 16-bit floating point types.
//!
//! This can be replaced with f16 from the Rust standard library when that is
//! stabilized. See <https://github.com/rust-lang/rust/issues/116909>.

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
    use rten_testing::TestCases;

    use crate::half::{f16_to_f32, f32_to_f16};

    #[test]
    fn test_f16_to_f32() {
        #[derive(Clone, Debug)]
        struct Case {
            input: f32,
            output: f32,
        }
        let cases = [
            Case {
                input: 0.,
                output: 0.,
            },
            Case {
                input: 1.,
                output: 1.,
            },
            Case {
                input: -1.,
                output: -1.,
            },
            Case {
                input: 2.,
                output: 2.,
            },
            Case {
                input: -2.,
                output: -2.,
            },
            Case {
                input: 0.333,
                output: 0.3330078,
            },
            Case {
                input: f32::INFINITY,
                output: f32::INFINITY,
            },
            Case {
                input: f32::NEG_INFINITY,
                output: f32::NEG_INFINITY,
            },
            Case {
                input: f32::MIN,
                output: f32::NEG_INFINITY,
            },
            Case {
                input: f32::MAX,
                output: f32::INFINITY,
            },
            Case {
                input: f32::MIN_POSITIVE,
                output: 0.,
            },
        ];

        cases.test_each_clone(|Case { input, output }| {
            let half = f32_to_f16(input);
            let float = f16_to_f32(half);
            assert_eq!(float, output);
        });
    }
}
