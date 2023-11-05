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

pub trait MinMax {
    /// Return the maximum value for this type.
    fn max_val() -> Self;

    /// Return the minimum value for this type.
    fn min_val() -> Self;

    /// Return the minimum of `self` and `other`.
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
