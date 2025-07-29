//! Traits for elements of SIMD vectors.

/// Types used as elements (or _lanes_) of SIMD vectors.
pub trait Elem: Copy + Default + WrappingAdd<Output = Self> {
    /// Return the 1 value of this type.
    fn one() -> Self;
}

impl Elem for f32 {
    fn one() -> Self {
        1.
    }
}

macro_rules! impl_elem_for_int {
    ($int:ty) => {
        impl Elem for $int {
            fn one() -> Self {
                1
            }
        }
    };
}

impl_elem_for_int!(i32);
impl_elem_for_int!(i16);
impl_elem_for_int!(i8);
impl_elem_for_int!(u8);
impl_elem_for_int!(u16);
impl_elem_for_int!(u32);

/// Wrapping addition of numbers.
///
/// For float types, this is the same as [`std::ops::Add`]. For integer types,
/// this is the same as the type's inherent `wrapping_add` method.
pub trait WrappingAdd: Sized {
    type Output;

    fn wrapping_add(self, x: Self) -> Self;
}

macro_rules! impl_wrapping_add {
    ($type:ty) => {
        impl WrappingAdd for $type {
            type Output = Self;

            fn wrapping_add(self, x: Self) -> Self {
                Self::wrapping_add(self, x)
            }
        }
    };
}

impl_wrapping_add!(i32);
impl_wrapping_add!(i16);
impl_wrapping_add!(i8);
impl_wrapping_add!(u8);
impl_wrapping_add!(u16);
impl_wrapping_add!(u32);

impl WrappingAdd for f32 {
    type Output = Self;

    fn wrapping_add(self, x: f32) -> f32 {
        self + x
    }
}
