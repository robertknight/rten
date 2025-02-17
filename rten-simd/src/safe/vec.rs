use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Types used as elements (or _lanes_) of SIMD vectors.
pub trait Elem: Copy + Default {
    /// Return the 1 value of this type.
    fn one() -> Self;
}

impl Elem for f32 {
    fn one() -> Self {
        1.
    }
}

impl Elem for i32 {
    fn one() -> Self {
        1
    }
}

/// Masks used or returned by SIMD operations.
///
/// # Safety
///
/// It must only be possible to construct the type if the associated SIMD ISA
/// is supported.
pub unsafe trait Mask: Copy {
    type Array: AsRef<[bool]>;

    /// Convert this mask to a bool array.
    fn to_array(self) -> Self::Array;

    /// Return true if all lanes in the mask are set to one.
    fn all_true(self) -> bool {
        self.to_array().as_ref().iter().all(|x| *x)
    }
}

/// A SIMD vector with associated instruction set.
///
/// Instances are constructed using methods of a [`Isa`]. For example
/// `isa.f32().one()` returns a SIMD vector of f32 lanes which is each set to
/// `1.0`. The number of lanes depends on the ISA.
///
/// # Safety
///
/// This trait is unsafe because implementations must ensure that instances
/// can only be constructed if the associated SIMD ISA is supported.
#[allow(clippy::len_without_is_empty)]
pub unsafe trait Simd:
    Copy
    + Debug
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + PartialEq
{
    /// Representation of this vector as a `[Self::Elem; N]` array.
    type Array: AsRef<[Self::Elem]> + IntoIterator<Item = Self::Elem>;

    /// Type of data held in each SIMD lane.
    type Elem: Elem;

    /// Mask with the same number of elements as this vector.
    type Mask: Mask;

    /// The ISA associated with this SIMD vector.
    type Isa: Isa;

    /// Return the [`MakeSimd`] impl for creating new vectors of the same type.
    fn init(self) -> impl MakeSimd<Self>;

    /// Return the [`Isa`] impl for creating new vectors from the same ISA
    /// family.
    fn isa(self) -> Self::Isa;

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

    /// Return the number of elements (or _lanes_) in the SIMD vector.
    fn len(self) -> usize;

    /// Return a mask indicating whether elements in `self` are less than rhs.
    fn lt(self, rhs: Self) -> Self::Mask;

    /// Return a mask indicating whether elements in `self` are less or equal to rhs.
    fn le(self, rhs: Self) -> Self::Mask;

    /// Return a mask indicating whether elements in `self` are equal to rhs.
    fn eq(self, rhs: Self) -> Self::Mask;

    /// Return a mask indicating whether elements in `self` are greater or equal to rhs.
    fn ge(self, rhs: Self) -> Self::Mask;

    /// Return a mask indicating whether elements in `self` are greater than rhs.
    fn gt(self, rhs: Self) -> Self::Mask;

    /// Return the minimum of `self` and `rhs` for each lane.
    fn min(self, rhs: Self) -> Self {
        self.select(rhs, self.lt(rhs))
    }

    /// Return the maximum of `self` and `rhs` for each lane.
    fn max(self, rhs: Self) -> Self {
        self.select(rhs, self.gt(rhs))
    }

    /// Select elements from `self` or `other` according to a mask.
    ///
    /// Elements are selected from `self` where the corresponding mask element
    /// is one or `other` if zero.
    fn select(self, other: Self, mask: Self::Mask) -> Self;

    /// Compute `self * b + c`.
    ///
    /// This will use fused multiply-add instructions if available. For float
    /// element types, this may use one or two roundings.
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    /// Convert `self` to a SIMD array.
    ///
    /// This is a cheap transmute in most cases, since SIMD vectors usually
    /// have the same layout as `[S::Elem; N]` but a greater alignment.
    fn to_array(self) -> Self::Array;

    /// Store the values in this vector to a memory location.
    ///
    /// # Safety
    ///
    /// `ptr` must point to `self.len()` elements.
    unsafe fn store_ptr(self, ptr: *mut Self::Elem);

    /// Store the values in this vector to a memory location, where the
    /// corresponding mask element is set.
    ///
    /// # Safety
    ///
    /// For each position `i` in the mask which is true, `ptr.add(i)` must point
    /// to a valid element of type `Self::Elem`.
    unsafe fn store_ptr_mask(self, ptr: *mut Self::Elem, mask: Self::Mask);
}

/// Extends the [`Simd`] trait with operations supported by float values.
pub trait SimdFloat: Simd + Neg<Output = Self> + Div<Output = Self> {
    /// Compute the absolute value of each element.
    fn abs(self) -> Self {
        let neg_mask = self.lt(self.init().zero());
        self.neg().select(self, neg_mask)
    }
}

/// Extends the [`Simd`] trait with operations supported by signed integers.
///
/// This excludes some common operations (eg. division) which are available
/// on scalars where common architectures don't have SIMD instructions for
/// those operations.
pub trait SimdInt: Simd + Neg<Output = Self> {
    /// Compute the absolute value of each element.
    fn abs(self) -> Self {
        let neg_mask = self.lt(self.init().zero());
        self.neg().select(self, neg_mask)
    }

    /// Shift the bits of this element left by a constant.
    fn shl<const N: i32>(self) -> Self;
}

/// Operations available only on SIMD vectors with `f32` elements.
pub trait SimdF32: Simd<Elem = f32> + SimdFloat {
    fn to_i32_trunc(self) -> <Self::Isa as Isa>::I32;
}

/// Entry point for performing SIMD operations using a particular instruction
/// set.
///
/// Implementations of this trait are types which can only be instantiated
/// if the instruction set is available. They are usually zero-sized and thus
/// free to copy.
///
/// # Safety
///
/// Implementations must ensure they can only be constructed if the
/// instruction set is supported on the current system.
pub unsafe trait Isa: Copy {
    /// A SIMD vector of unspecified type which all vectors in this family can
    /// be cast to/from.
    type Bits: Simd;

    /// SIMD vector type for this ISA with `f32` elements.
    type F32: Simd<Isa = Self> + SimdF32;

    /// SIMD vector type for this ISA with `i32` elements.
    type I32: Simd<Elem = i32, Isa = Self> + SimdInt;

    /// Return initializer for creating SIMD vectors with `f32` elements.
    fn f32(self) -> impl MakeSimd<Self::F32>;

    /// Return initializer for creating SIMD vectors with `i32` elements.
    fn i32(self) -> impl MakeSimd<Self::I32>;
}

/// Trait for creating SIMD vectors of a particular type.
///
/// # Safety
///
/// Implementations must ensure they can only be constructed if the
/// instruction set is supported on the current system.
#[allow(clippy::len_without_is_empty)]
pub unsafe trait MakeSimd<S: Simd>: Copy {
    /// Return the number of elements in the vector.
    fn len(self) -> usize;

    /// Create a new vector with all lanes set to zero.
    fn zero(self) -> S {
        self.splat(S::Elem::default())
    }

    /// Create a new vector with all lanes set to one.
    fn one(self) -> S {
        self.splat(S::Elem::one())
    }

    /// Create a new vector with all lanes set to `x`.
    fn splat(self, x: S::Elem) -> S;

    /// Return a mask with the first `n` lanes set to true.
    fn first_n_mask(self, n: usize) -> S::Mask;

    /// Load vector of elements from `ptr`.
    ///
    /// `ptr` is not required to have any particular alignment.
    ///
    /// # Safety
    ///
    /// `ptr` must point to `self.len()` initialized elements of type `S::Elem`.
    unsafe fn load_ptr(self, ptr: *const S::Elem) -> S;

    /// Load vector elements from `ptr` using a mask.
    ///
    /// `ptr` is not required to have any particular alignment.
    ///
    /// # Safety
    ///
    /// For each mask position `i` which is true, `ptr.add(i)` must point to
    /// an initialized element of type `S::Elem`.
    unsafe fn load_ptr_mask(self, ptr: *const S::Elem, mask: S::Mask) -> S;
}

#[cfg(test)]
mod test {
    use crate::safe::{Isa, MakeSimd, Simd, SimdF32, SimdInt, SimdOp};

    macro_rules! test_simd_op {
        ($isa:ident, $op:block) => {{
            struct TestOp {}

            impl SimdOp for TestOp {
                type Output = ();

                fn eval<I: Isa>(self, $isa: I) {
                    $op
                }
            }

            TestOp {}.dispatch()
        }};
    }

    #[test]
    fn test_bin_ops_f32() {
        test_simd_op!(isa, {
            let x = isa.f32().splat(1.);
            let y = isa.f32().splat(2.);

            // Add
            let expected = isa.f32().splat(3.);
            let actual = x + y;
            assert_eq!(actual, expected);

            // Sub
            let expected = isa.f32().splat(-1.);
            let actual = x - y;
            assert_eq!(actual, expected);

            // Div
            let expected = isa.f32().splat(0.5);
            let actual = x / y;
            assert_eq!(actual, expected);

            // Mul
            let expected = isa.f32().splat(2.);
            let actual = x * y;
            assert_eq!(actual, expected);
        })
    }

    #[test]
    fn test_unary_ops_f32() {
        test_simd_op!(isa, {
            let x = isa.f32().splat(3.);

            // Neg
            let expected = isa.f32().splat(-3.);
            let actual = -x;
            assert_eq!(actual, expected);
        })
    }

    #[test]
    fn test_bin_ops_i32() {
        test_simd_op!(isa, {
            let x = isa.i32().splat(1);
            let y = isa.i32().splat(2);

            // Add
            let expected = isa.i32().splat(3);
            let actual = x + y;
            assert_eq!(actual, expected);

            // Sub
            let expected = isa.i32().splat(-1);
            let actual = x - y;
            assert_eq!(actual, expected);

            // Mul
            let expected = isa.i32().splat(2);
            let actual = x * y;
            assert_eq!(actual, expected);
        })
    }

    #[test]
    fn test_unary_ops_i32() {
        test_simd_op!(isa, {
            let x = isa.i32().splat(3);

            // Add
            let expected = isa.i32().splat(-3);
            let actual = -x;
            assert_eq!(actual, expected);
        })
    }

    #[test]
    fn test_reinterpret_cast() {
        test_simd_op!(isa, {
            let x = 1.456f32;
            let x_i32 = x.to_bits() as i32;

            let x_vec = isa.f32().splat(x);
            let y_vec: I::I32 = x_vec.reinterpret_cast();

            let expected = isa.i32().splat(x_i32);
            assert_eq!(y_vec, expected);
        })
    }

    #[test]
    fn test_shl() {
        test_simd_op!(isa, {
            let x = isa.i32().splat(42);
            let y = x.shl::<1>();
            let expected = x.init().splat(42 << 1);
            assert_eq!(y, expected);
        })
    }

    #[test]
    fn test_f32_to_i32_trunc() {
        test_simd_op!(isa, {
            let x = isa.f32().splat(12.345);
            let y = x.to_i32_trunc();
            let expected = isa.i32().splat(12);
            assert_eq!(y, expected);
        })
    }
}
