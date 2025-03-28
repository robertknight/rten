//! Traits for SIMD vectors and masks.

use std::fmt::Debug;

use crate::elem::Elem;
use crate::ops::Isa;

/// Masks used or returned by SIMD operations.
///
/// Most operations on masks are available via the
/// [`MaskOps`](crate::ops::MaskOps) trait. Implementations are obtained via
/// [`NumOps::mask_ops`](crate::ops::NumOps::mask_ops).
pub trait Mask: Copy + Debug {
    type Array: AsRef<[bool]>
        + Copy
        + Debug
        + IntoIterator<Item = bool>
        + PartialEq<Self::Array>
        + std::ops::Index<usize, Output = bool>;

    /// Convert this mask to a bool array.
    fn to_array(self) -> Self::Array;

    /// Return true if all lanes in the mask are one.
    fn all_true(self) -> bool {
        self.to_array().as_ref().iter().all(|&x| x)
    }

    /// Return true if all lanes in the mask are false.
    fn all_false(self) -> bool {
        self.to_array().as_ref().iter().all(|&x| !x)
    }
}

/// SIMD vector type.
#[allow(clippy::len_without_is_empty)]
pub trait Simd: Copy + Debug {
    /// Representation of this vector as a `[Self::Elem; N]` array.
    type Array: AsRef<[Self::Elem]>
        + Copy
        + Debug
        + IntoIterator<Item = Self::Elem>
        + PartialEq<Self::Array>
        + std::ops::Index<usize, Output = Self::Elem>
        + std::ops::IndexMut<usize, Output = Self::Elem>;

    /// Type of data held in each SIMD lane.
    type Elem: Elem;

    /// Mask with the same number of elements as this vector.
    type Mask: Mask;

    /// The ISA associated with this SIMD vector.
    type Isa: Isa;

    /// Convert this SIMD vector to the common "bits" type used by all vectors
    /// in this family.
    fn to_bits(self) -> <Self::Isa as Isa>::Bits;

    /// Convert this SIMD vector from the common "bits" type used by all vectors
    /// in this family.
    fn from_bits(bits: <Self::Isa as Isa>::Bits) -> Self;

    /// Reinterpret the bits of this vector as another vector from the same
    /// family.
    fn reinterpret_cast<T>(self) -> T
    where
        T: Simd<Isa = Self::Isa>,
    {
        T::from_bits(self.to_bits())
    }

    /// Cast this vector to another with the same ISA and element type.
    ///
    /// This cast is a no-op which doesn't generate any code. It is needed in
    /// some cases to downcast a `Simd` type to one of an `Isa`s associated
    /// types, or vice-versa.
    fn same_cast<T>(self) -> T
    where
        T: Simd<Elem = Self::Elem, Isa = Self::Isa>,
    {
        T::from_bits(self.to_bits())
    }

    /// Convert `self` to a SIMD array.
    ///
    /// This is a cheap transmute in most cases, since SIMD vectors usually
    /// have the same layout as `[S::Elem; N]` but a greater alignment.
    fn to_array(self) -> Self::Array;
}
