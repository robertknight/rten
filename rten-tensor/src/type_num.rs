//! Traits and types for compile-time arithmetic.
//!
//! These types are used in various tensor methods, such as
//! [`slice`](crate::TensorBase::slice), as part of computing the layout of the
//! result at compile time.

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// Type representing an integer whose value is unknown at compile time.
pub struct Unknown {}

/// Type representing the integer value 0.
pub struct U0 {}

/// Type representing the integer value 1.
pub struct U1 {}

/// Type representing the integer value 2.
pub struct U2 {}

/// Type representing the integer value 3.
pub struct U3 {}

/// Type representing the integer value 4.
pub struct U4 {}

/// Type representing the integer value 5.
pub struct U5 {}

/// Trait providing the integer value of a `U<N>` type (eg. [`U0`]).
///
/// The value can be unknown to represent numbers that are known only at
/// runtime.
pub trait OptionalUInt {
    const VALUE: Option<usize>;
}

macro_rules! impl_const_int {
    ($type:ty, $val:expr) => {
        impl OptionalUInt for $type {
            const VALUE: Option<usize> = $val;
        }
    };
}

impl_const_int!(Unknown, None);
impl_const_int!(U0, Some(0));
impl_const_int!(U1, Some(1));
impl_const_int!(U2, Some(2));
impl_const_int!(U3, Some(3));
impl_const_int!(U4, Some(4));
impl_const_int!(U5, Some(5));

/// Trait that computes the sum of [`OptionalUInt`] types.
///
/// It is implemented for 2-tuples, as well as arrays of either `U0` or `U1`.
pub trait Add {
    type Result: OptionalUInt;
}

macro_rules! impl_add {
    ($lhs:ty, $rhs:ty, $result:ty) => {
        impl Add for ($lhs, $rhs) {
            type Result = $result;
        }

        impl Add for ($rhs, $lhs) {
            type Result = $result;
        }
    };

    ($lhs:ty, $result:ty) => {
        impl Add for ($lhs, $lhs) {
            type Result = $result;
        }
    };
}

// Implement addition of 2-tuples up to a value of 5.
impl_add!(U0, U0);
impl_add!(U0, U1, U1);
impl_add!(U0, U2, U2);
impl_add!(U0, U3, U3);
impl_add!(U0, U4, U4);
impl_add!(U0, U5, U5);
impl_add!(U1, U2);
impl_add!(U1, U2, U3);
impl_add!(U1, U3, U4);
impl_add!(U1, U4, U5);
impl_add!(U2, U4);
impl_add!(U2, U3, U5);

// Implement addition of bits for values up to 5.
impl<const N: usize> Add for [U0; N] {
    type Result = U0;
}

macro_rules! impl_add_ones {
    ($count:literal, $type:ty) => {
        impl Add for [U1; $count] {
            type Result = $type;
        }
    };
}
impl_add_ones!(0, U0);
impl_add_ones!(1, U1);
impl_add_ones!(2, U2);
impl_add_ones!(3, U3);
impl_add_ones!(4, U4);
impl_add_ones!(5, U5);

/// Marker trait indicating whether a value is an index or a range.
pub trait IsIndex {
    /// Associated type that is either [`U0`] or [`U1`] indicating whether this
    /// type is an index.
    ///
    /// The value can also be [`Unknown`] to indicate a value that may be either
    /// an index or a range.
    type IsIndex: OptionalUInt;
}

macro_rules! impl_is_index {
    ($type:ty, true) => {
        impl IsIndex for $type {
            type IsIndex = U1;
        }
    };
}

impl_is_index!(usize, true);
impl_is_index!(isize, true);
impl_is_index!(i32, true);

impl<T> IsIndex for Range<T> {
    type IsIndex = U0;
}

impl IsIndex for RangeFull {
    type IsIndex = U0;
}

impl<T> IsIndex for RangeTo<T> {
    type IsIndex = U0;
}

impl<T> IsIndex for RangeFrom<T> {
    type IsIndex = U0;
}

/// Trait that counts the number of items in a sequence which are indices, as
/// opposed to ranges.
///
/// This trait is a helper used by slicing methods in
/// [`TensorBase`](crate::TensorBase) to infer the rank of a view after slicing
/// with a range.
///
/// The trait is implemented for scalar values, ranges and tuples up to length 5.
///
/// ```
/// use rten_tensor::type_num::IndexCount;
/// assert_eq!((.., 1..2).index_count(), Some(0));
/// assert_eq!((0, 1..2).index_count(), Some(1));
/// assert_eq!((0, 1).index_count(), Some(2));
/// assert_eq!([0, 1].as_slice().index_count(), None);
/// ```
pub trait IndexCount {
    /// Type representing the count value.
    type Count: OptionalUInt;

    /// Returns [`Count`](IndexCount::Count) as a numeric value.
    fn index_count(&self) -> Option<usize> {
        Self::Count::VALUE
    }
}

impl IndexCount for usize {
    type Count = U1;
}

impl<T> IndexCount for Range<T> {
    type Count = U0;
}

impl<T> IndexCount for RangeTo<T> {
    type Count = U0;
}

impl<T> IndexCount for RangeFrom<T> {
    type Count = U0;
}

impl IndexCount for RangeFull {
    type Count = U0;
}

impl<T: IsIndex> IndexCount for (T,) {
    type Count = T::IsIndex;
}

impl<T1: IsIndex, T2: IsIndex> IndexCount for (T1, T2)
where
    (T1::IsIndex, T2::IsIndex): Add,
{
    type Count = <(T1::IsIndex, T2::IsIndex) as Add>::Result;
}

impl<T1: IsIndex, T2: IsIndex, T3: IsIndex> IndexCount for (T1, T2, T3)
where
    (T1, T2): IndexCount,
    (<(T1, T2) as IndexCount>::Count, T3::IsIndex): Add,
{
    type Count = <(<(T1, T2) as IndexCount>::Count, T3::IsIndex) as Add>::Result;
}

impl<T1: IsIndex, T2: IsIndex, T3: IsIndex, T4: IsIndex> IndexCount for (T1, T2, T3, T4)
where
    (T1, T2, T3): IndexCount,
    (<(T1, T2, T3) as IndexCount>::Count, T4::IsIndex): Add,
{
    type Count = <(<(T1, T2, T3) as IndexCount>::Count, T4::IsIndex) as Add>::Result;
}

impl<T1: IsIndex, T2: IsIndex, T3: IsIndex, T4: IsIndex, T5: IsIndex> IndexCount
    for (T1, T2, T3, T4, T5)
where
    (T1, T2, T3, T4): IndexCount,
    (<(T1, T2, T3, T4) as IndexCount>::Count, T5::IsIndex): Add,
{
    type Count = <(<(T1, T2, T3, T4) as IndexCount>::Count, T5::IsIndex) as Add>::Result;
}

impl<T: IsIndex, const N: usize> IndexCount for [T; N]
where
    [T::IsIndex; N]: Add,
{
    type Count = <[T::IsIndex; N] as Add>::Result;
}

impl<'a, T> IndexCount for &'a [T] {
    type Count = Unknown;
}

#[cfg(test)]
mod tests {
    use super::IndexCount;

    #[test]
    fn test_index_count() {
        // Single values
        assert_eq!((0).index_count(), Some(1));
        assert_eq!((..).index_count(), Some(0));
        assert_eq!((..1).index_count(), Some(0));
        assert_eq!((1..).index_count(), Some(0));
        assert_eq!((1..2).index_count(), Some(0));

        // Tuples
        assert_eq!((0,).index_count(), Some(1));
        assert_eq!((0, ..).index_count(), Some(1));
        assert_eq!((0, .., 2).index_count(), Some(2));
        assert_eq!((0, .., 2, ..).index_count(), Some(2));
        assert_eq!((0, .., 2, .., 3).index_count(), Some(3));

        // Arrays
        assert_eq!([1, 2, 3].index_count(), Some(3));

        // Slices
        assert_eq!([1, 2, 3].as_slice().index_count(), None);
    }
}
