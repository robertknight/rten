/// Trait for obtaining the size of the Unit in the Last Place (ULP) for floats.
pub trait Ulp {
    /// Return the size of the ULP for a given value.
    fn ulp(self) -> Self;

    /// Return the difference between this value and `other` in units of
    /// `other.ulp()`.
    fn diff_ulps(self, other: Self) -> Self;
}

impl Ulp for f32 {
    /// Return the size of the Unit in the Last Place for a given value.
    ///
    /// Handling of special cases (NaN, infinity, zero, min/max) follows `Math.ulp`
    /// in Java [1].
    ///
    /// [1] https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/Math.html#ulp(float)
    fn ulp(self: f32) -> f32 {
        if self.is_nan() {
            self
        } else if self.is_infinite() {
            f32::INFINITY
        } else if self == 0. {
            f32::MIN
        } else if self == f32::MIN || self == f32::MAX {
            f32::from_bits((127 + 104) << 23) // 2^104
        } else {
            let bits = self.to_bits();
            let next_up = f32::from_bits(bits + 1);
            (next_up - self).abs()
        }
    }

    fn diff_ulps(self: f32, other: f32) -> f32 {
        (self - other).abs() / other.ulp()
    }
}

/// Assert that the difference between two values is less than or equal to a
/// given number of [ULPs](Ulp).
macro_rules! assert_ulp_diff_le {
    ($actual:expr, $expected:expr, $max_diff:expr) => {{
        use crate::ulp::Ulp;

        let ulp_diff = ($actual).diff_ulps($expected);
        assert!(
            ulp_diff <= $max_diff,
            "difference between {} and {} is {} ULPs which exceeds {}",
            $actual,
            $expected,
            ulp_diff,
            $max_diff
        );
    }};
}

pub(crate) use assert_ulp_diff_le;

#[cfg(test)]
mod tests {
    use super::Ulp;

    #[test]
    fn test_f32_ulp() {
        assert_eq!((1.0f32).ulp(), f32::EPSILON);

        // Special cases. See the Java `Math.ulp` docs.
        assert!(f32::NAN.ulp().is_nan());
        assert_eq!(f32::INFINITY.ulp(), f32::INFINITY);
        assert_eq!(f32::NEG_INFINITY.ulp(), f32::INFINITY);
        assert_eq!((0.0f32).ulp(), f32::MIN);
        assert_eq!((-0.0f32).ulp(), f32::MIN);
        assert_eq!(f32::MAX.ulp(), (104f32).exp2());
        assert_eq!(f32::MIN.ulp(), (104f32).exp2());
    }

    #[test]
    fn test_f32_diff_ulps() {
        let x = 1.0f32;
        let y = 1.001f32;

        let diff_bits = y.to_bits() - x.to_bits();
        assert_eq!(y.diff_ulps(x), diff_bits as f32);
    }
}
