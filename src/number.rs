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
