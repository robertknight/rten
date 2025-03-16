use std::fmt::Debug;
use std::mem::MaybeUninit;

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

impl WrappingAdd for f32 {
    type Output = Self;

    fn wrapping_add(self, x: f32) -> f32 {
        self + x
    }
}

/// Masks used or returned by SIMD operations.
///
/// Most operations on masks are available via the [`MaskOps`] trait.
/// Implementations are obtained via [`NumOps::mask_ops`].
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

/// Entry point for performing SIMD operations using a particular Instruction
/// Set Architecture (ISA).
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
    /// SIMD vector with an unspecified element type. This is used for
    /// bitwise casting between different vector types.
    type Bits: Simd;

    /// SIMD vector with `f32` elements.
    type F32: Simd<Elem = f32, Isa = Self>;

    /// SIMD vector with `i32` elements.
    type I32: Simd<Elem = i32, Isa = Self>;

    /// SIMD vector with `i16` elements.
    type I16: Simd<Elem = i16, Isa = Self>;

    /// SIMD vector with `i8` elements.
    type I8: Simd<Elem = i8, Isa = Self>;

    /// SIMD vector with `u8` elements.
    type U8: Simd<Elem = u8, Isa = Self>;

    /// SIMD vector with `u16` elements.
    type U16: Simd<Elem = u16, Isa = Self>;

    /// Operations on SIMD vectors with `f32` elements.
    fn f32(self) -> impl FloatOps<Self::F32, Int = Self::I32>;

    /// Operations on SIMD vectors with `i32` elements.
    fn i32(self) -> impl SignedIntOps<Self::I32> + NarrowSaturate<Self::I32, Self::I16>;

    /// Operations on SIMD vectors with `i16` elements.
    fn i16(self) -> impl SignedIntOps<Self::I16> + NarrowSaturate<Self::I16, Self::U8>;

    /// Operations on SIMD vectors with `i8` elements.
    fn i8(self) -> impl SignedIntOps<Self::I8>;

    /// Operations on SIMD vectors with `u8` elements.
    fn u8(self) -> impl NumOps<Self::U8>;

    /// Operations on SIMD vectors with `u16` elements.
    fn u16(self) -> impl NumOps<Self::U16>;
}

/// SIMD operations on a [`Mask`] vector.
///
/// # Safety
///
/// Implementations must ensure they can only be constructed if the
/// instruction set is supported on the current system.
pub unsafe trait MaskOps<M: Mask>: Copy {
    /// Compute `x & y`.
    fn and(self, x: M, y: M) -> M;
}

/// Operations available on all SIMD vector types.
///
/// This trait provides core operations available on all SIMD vector types:
///
/// - Load from and store into memory
/// - Creating a new vector filled with zeros or a specific value
/// - Combining elements from two vectors according to a mask
/// - Add, subtract and multiply
/// - Comparison (equality, less than, greater than etc.)
///
/// # Safety
///
/// Implementations must ensure they can only be constructed if the
/// instruction set is supported on the current system.
#[allow(clippy::len_without_is_empty)]
pub unsafe trait NumOps<S: Simd>: Copy {
    /// Convert `x` to an untyped vector of the same width.
    #[allow(clippy::wrong_self_convention)]
    fn from_bits(self, x: <S::Isa as Isa>::Bits) -> S {
        S::from_bits(x)
    }

    /// Return the implementation of mask operations for the mask vector used
    /// by this SIMD type.
    fn mask_ops(self) -> impl MaskOps<S::Mask>;

    /// Return the number of elements in the vector.
    fn len(self) -> usize;

    /// Compute `x + y`.
    fn add(self, x: S, y: S) -> S;

    /// Compute `x - y`.
    fn sub(self, x: S, y: S) -> S;

    /// Compute `x * y`.
    fn mul(self, x: S, y: S) -> S;

    /// Create a new vector with all lanes set to zero.
    fn zero(self) -> S {
        self.splat(S::Elem::default())
    }

    /// Create a new vector with all lanes set to one.
    fn one(self) -> S {
        self.splat(S::Elem::one())
    }

    /// Compute `a * b + c`.
    ///
    /// This will use fused multiply-add instructions if available. For float
    /// element types, this may use one or two roundings.
    fn mul_add(self, a: S, b: S, c: S) -> S {
        self.add(self.mul(a, b), c)
    }

    /// Evaluate a polynomial using Horner's method.
    ///
    /// Computes `x * coeffs[0] + x^2 * coeffs[1] ... x^n * coeffs[N]`
    #[inline]
    fn poly_eval(self, x: S, coeffs: &[S]) -> S {
        let mut y = coeffs[coeffs.len() - 1];
        for i in (0..coeffs.len() - 1).rev() {
            y = self.mul_add(y, x, coeffs[i]);
        }
        self.mul(y, x)
    }

    /// Return a mask indicating whether elements in `x` are less than `y`.
    #[inline]
    fn lt(self, x: S, y: S) -> S::Mask {
        self.gt(y, x)
    }

    /// Return a mask indicating whether elements in `x` are less or equal to `y`.
    #[inline]
    fn le(self, x: S, y: S) -> S::Mask {
        self.ge(y, x)
    }

    /// Return a mask indicating whether elements in `x` are equal to `y`.
    fn eq(self, x: S, y: S) -> S::Mask;

    /// Return a mask indicating whether elements in `x` are greater or equal to `y`.
    fn ge(self, x: S, y: S) -> S::Mask;

    /// Return a mask indicating whether elements in `x` are greater than `y`.
    fn gt(self, x: S, y: S) -> S::Mask;

    /// Return the minimum of `x` and `y` for each lane.
    fn min(self, x: S, y: S) -> S {
        self.select(x, y, self.le(x, y))
    }

    /// Return the maximum of `x` and `y` for each lane.
    fn max(self, x: S, y: S) -> S {
        self.select(x, y, self.ge(x, y))
    }

    /// Clamp values in `x` to minimum and maximum values from corresponding
    /// lanes in `min` and `max`.
    fn clamp(self, x: S, min: S, max: S) -> S {
        self.min(self.max(x, min), max)
    }

    /// Create a new vector with all lanes set to `x`.
    fn splat(self, x: S::Elem) -> S;

    /// Reduce the elements in `x` to a single value using `f`, then
    /// return a new vector with the accumulated value broadcast to each lane.
    #[inline]
    fn fold_splat<F: Fn(S::Elem, S::Elem) -> S::Elem>(self, x: S, accum: S::Elem, f: F) -> S {
        let reduced = x.to_array().into_iter().fold(accum, f);
        self.splat(reduced)
    }

    /// Return a mask with the first `n` lanes set to true.
    fn first_n_mask(self, n: usize) -> S::Mask;

    /// Load the first `self.len()` elements from a slice into a vector.
    ///
    /// Panics if `xs.len() < self.len()`.
    #[inline]
    fn load(self, xs: &[S::Elem]) -> S {
        assert!(xs.len() >= self.len());
        unsafe { self.load_ptr(xs.as_ptr()) }
    }

    /// Load `N` vectors from consecutive sub-slices of `xs`.
    ///
    /// Panics if `xs.len() < self.len() * N`.
    #[inline]
    fn load_many<const N: usize>(self, xs: &[S::Elem]) -> [S; N] {
        let v_len = self.len();
        assert!(xs.len() >= v_len * N);

        // Safety: `xs.add(i * v_len)` points to at least `v_len` elements.
        std::array::from_fn(|i| unsafe { self.load_ptr(xs.as_ptr().add(i * v_len)) })
    }

    /// Load elements from `xs` into a vector.
    ///
    /// If the vector length exceeds `xs.len()`, the tail is padded with zeros.
    ///
    /// Returns the padded vector and a mask of the lanes which were set.
    #[inline]
    fn load_pad(self, xs: &[S::Elem]) -> (S, S::Mask) {
        let n = xs.len().min(self.len());
        let mask = self.first_n_mask(n);

        // Safety: `xs.add(i)` is valid for all positions where mask is set
        let vec = unsafe { self.load_ptr_mask(xs.as_ptr(), mask) };

        (vec, mask)
    }

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

    /// Select elements from `x` or `y` according to a mask.
    ///
    /// Elements are selected from `x` where the corresponding mask element
    /// is one or `y` if zero.
    fn select(self, x: S, y: S, mask: S::Mask) -> S;

    /// Store the values in this vector to a memory location.
    ///
    /// # Safety
    ///
    /// `ptr` must point to `self.len()` elements.
    unsafe fn store_ptr(self, x: S, ptr: *mut S::Elem);

    /// Store `x` into the first `self.len()` elements of `xs`.
    #[inline]
    fn store(self, x: S, xs: &mut [S::Elem]) {
        assert!(xs.len() >= self.len());
        unsafe { self.store_ptr(x, xs.as_mut_ptr()) }
    }

    /// Store `x` into the first `self.len()` elements of `xs`.
    ///
    /// This is a variant of [`store`](NumOps::store) which takes an
    /// uninitialized slice as input and returns the initialized portion of the
    /// slice.
    #[inline]
    fn store_uninit(self, x: S, xs: &mut [MaybeUninit<S::Elem>]) -> &mut [S::Elem] {
        let len = self.len();
        let xs_ptr = xs.as_mut_ptr() as *mut S::Elem;
        assert!(xs.len() >= len);
        unsafe {
            self.store_ptr(x, xs_ptr);

            // Safety: `store_ptr` initialized `len` elements of `xs`.
            std::slice::from_raw_parts_mut(xs_ptr, len)
        }
    }

    /// Store the values in this vector to a memory location, where the
    /// corresponding mask element is set.
    ///
    /// # Safety
    ///
    /// For each position `i` in the mask which is true, `ptr.add(i)` must point
    /// to a valid element of type `Self::Elem`.
    unsafe fn store_ptr_mask(self, x: S, ptr: *mut S::Elem, mask: S::Mask);

    fn prefetch(self, ptr: *const S::Elem) {
        // Default implementation does nothing
        let _ = ptr;
    }

    fn prefetch_write(self, ptr: *mut S::Elem) {
        // Default implementation does nothing
        let _ = ptr;
    }

    /// Horizontally sum the elements in a vector.
    ///
    /// If the sum overflows, it will wrap. This choice was made to enable
    /// consistency between native intrinsics for horizontal addition and the
    /// generic implementation.
    fn sum(self, x: S) -> S::Elem {
        let mut sum = S::Elem::default();
        for elem in x.to_array() {
            sum = sum.wrapping_add(elem);
        }
        sum
    }
}

/// Operations available on SIMD vectors with float elements.
pub trait FloatOps<S: Simd>: NumOps<S> {
    /// Integer SIMD vector of the same bit-width as this vector.
    type Int: Simd;

    /// Compute x / y
    fn div(self, x: S, y: S) -> S;

    /// Compute 1. / x
    fn reciprocal(self, x: S) -> S {
        self.div(self.one(), x)
    }

    /// Compute `-x`
    fn neg(self, x: S) -> S {
        self.sub(self.zero(), x)
    }

    /// Compute the absolute value of `x`
    fn abs(self, x: S) -> S {
        self.select(self.neg(x), x, self.lt(x, self.zero()))
    }

    /// Convert each lane to an integer of the same width, rounding towards zero.
    fn to_int_trunc(self, x: S) -> Self::Int;

    /// Convert each lane to an integer of the same width, rounding to nearest
    /// with ties to even.
    fn to_int_round(self, x: S) -> Self::Int;
}

/// Operations on SIMD vectors with signed integer elements.
pub trait SignedIntOps<S: Simd>: NumOps<S> {
    /// Shift each lane in `x` left by `SHIFT` bits.
    fn shift_left<const SHIFT: i32>(self, x: S) -> S;

    /// Compute the absolute value of `x`
    fn abs(self, x: S) -> S {
        self.select(self.neg(x), x, self.lt(x, self.zero()))
    }

    /// Return `-x`.
    fn neg(self, x: S) -> S {
        self.sub(self.zero(), x)
    }
}

/// Widen lanes to a type with twice the width.
///
/// For integer types, the extended type has the same signed-ness.
#[cfg(target_arch = "x86_64")]
pub(crate) trait Extend<S: Simd> {
    type Output;

    /// Extend each lane to a type with twice the width.
    ///
    /// Returns a tuple containing the extended low and high half of the input.
    fn extend(self, x: S) -> (Self::Output, Self::Output);
}

/// Narrow lanes to one with half the bit-width, using truncation.
///
/// For integer types, the narrowed type has the same signed-ness.
#[cfg(target_arch = "x86_64")]
pub(crate) trait Narrow<S: Simd> {
    type Output;

    /// Truncate each lane in a pair of vectors to one with half the bit-width.
    ///
    /// Returns a vector containing the concatenation of the narrowed lanes
    /// from `low` followed by the narrowed lanes from `high`.
    fn narrow_truncate(self, low: S, high: S) -> Self::Output;
}

/// Narrow lanes to one with half the bit-width, using saturation.
///
/// Conceptually, this converts each element from `S1::Elem` to `S2::Elem` using
/// `x.clamp(S2::Elem::MIN as S1::Elem, S2::Elem::MAX as S1::Elem) as S2::Elem`.
pub trait NarrowSaturate<S1: Simd, S2: Simd> {
    /// Narrow each lane in a pair of vectors to one with half the bit-width.
    ///
    /// Returns a vector containing the concatenation of the narrowed lanes
    /// from `low` followed by the narrowed lanes from `high`.
    fn narrow_saturate(self, low: S1, high: S1) -> S2;
}

#[cfg(test)]
mod tests {
    use super::WrappingAdd;
    use crate::safe::{
        assert_simd_eq, test_simd_op, FloatOps, Isa, Mask, MaskOps, NarrowSaturate, NumOps,
        SignedIntOps, Simd, SimdOp,
    };

    // Generate tests for operations available on all numeric types.
    macro_rules! test_num_ops {
        ($modname:ident, $elem:ident) => {
            mod $modname {
                use super::{
                    assert_simd_eq, test_simd_op, Isa, Mask, NumOps, Simd, SimdOp, WrappingAdd,
                };

                #[test]
                fn test_load_store() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let src: Vec<_> = (0..ops.len() * 4).map(|x| x as $elem).collect();
                        let mut dst = vec![0 as $elem; src.len()];

                        for (src_chunk, dst_chunk) in
                            src.chunks(ops.len()).zip(dst.chunks_mut(ops.len()))
                        {
                            let x = ops.load(src_chunk);
                            ops.store(x, dst_chunk);
                        }

                        assert_eq!(dst, src);
                    })
                }

                #[test]
                fn test_store_uninit() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let src: Vec<_> = (0..ops.len() + 3).map(|x| x as $elem).collect();
                        let mut dest = Vec::with_capacity(src.len());

                        let x = ops.load(&src);

                        let init = ops.store_uninit(x, dest.spare_capacity_mut());
                        assert_eq!(init, &src[0..ops.len()]);
                    })
                }

                #[test]
                fn test_load_many() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let src: Vec<_> = (0..ops.len() * 2).map(|x| x as $elem).collect();

                        let xs = ops.load_many::<2>(&src);
                        assert_simd_eq!(xs[0], ops.load(&src));
                        assert_simd_eq!(xs[1], ops.load(&src[ops.len()..]));
                    })
                }

                #[test]
                fn test_load_pad() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        // Array which is shorter than vector length for all ISAs.
                        let src = [0, 1, 2].map(|x| x as $elem);

                        let (vec, _mask) = ops.load_pad(&src);
                        let vec_array = vec.to_array();
                        let vec_slice = vec_array.as_ref();

                        assert_eq!(&vec_slice[..src.len()], &src);
                        for i in ops.len()..vec_slice.len() {
                            assert_eq!(vec_array[i], 0 as $elem);
                        }
                    })
                }

                #[test]
                fn test_bin_ops() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let a = 2 as $elem;
                        let b = 3 as $elem;

                        let x = ops.splat(a);
                        let y = ops.splat(b);

                        // Add
                        let expected = ops.splat(a + b);
                        let actual = ops.add(x, y);
                        assert_simd_eq!(actual, expected);

                        // Sub
                        let expected = ops.splat(b - a);
                        let actual = ops.sub(y, x);
                        assert_simd_eq!(actual, expected);

                        // Mul
                        let expected = ops.splat(a * b);
                        let actual = ops.mul(x, y);
                        assert_simd_eq!(actual, expected);
                    })
                }

                #[test]
                fn test_cmp_ops() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let x = ops.splat(1 as $elem);
                        let y = ops.splat(2 as $elem);

                        assert!(ops.eq(x, x).all_true());
                        assert!(ops.eq(x, y).all_false());
                        assert!(ops.le(x, x).all_true());
                        assert!(ops.le(x, y).all_true());
                        assert!(ops.le(y, x).all_false());
                        assert!(ops.ge(x, x).all_true());
                        assert!(ops.ge(x, y).all_false());
                        assert!(ops.gt(x, y).all_false());
                        assert!(ops.gt(y, x).all_true());
                    })
                }

                #[test]
                fn test_mul_add() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let a = ops.splat(2 as $elem);
                        let b = ops.splat(3 as $elem);
                        let c = ops.splat(4 as $elem);

                        let actual = ops.mul_add(a, b, c);
                        let expected = ops.splat(((2. * 3.) + 4.) as $elem);

                        assert_simd_eq!(actual, expected);
                    })
                }

                #[test]
                fn test_min_max() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let x = ops.splat(3 as $elem);

                        // Min
                        let y_min = ops.min(x, ops.splat(2 as $elem));
                        let y_min_2 = ops.min(ops.splat(2 as $elem), x);
                        assert_simd_eq!(y_min, y_min_2);
                        assert_simd_eq!(y_min, ops.splat(2 as $elem));

                        // Max
                        let y_max = ops.max(x, ops.splat(4 as $elem));
                        let y_max_2 = ops.max(ops.splat(4 as $elem), x);
                        assert_simd_eq!(y_max, y_max_2);
                        assert_simd_eq!(y_max, ops.splat(4 as $elem));

                        // Clamp
                        let y_clamped = ops.clamp(x, ops.splat(0 as $elem), ops.splat(4 as $elem));
                        assert_simd_eq!(y_clamped, ops.splat(3 as $elem));
                    })
                }

                #[test]
                fn test_sum() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let vec: Vec<_> = (0..ops.len()).map(|x| x as $elem).collect();
                        let expected = vec
                            .iter()
                            .fold(0 as $elem, |sum, x| WrappingAdd::wrapping_add(sum, *x));

                        let x = ops.load(&vec);
                        let y = ops.sum(x);

                        assert_eq!(y, expected);
                    })
                }

                #[test]
                fn test_poly_eval() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let coeffs = [2, 3, 4].map(|x| x as $elem);
                        let x = 2 as $elem;
                        let y = ops.poly_eval(ops.splat(x), &coeffs.map(|c| ops.splat(c)));

                        let expected =
                            (x * coeffs[0]) + (x * x * coeffs[1]) + (x * x * x * coeffs[2]);
                        assert_simd_eq!(y, ops.splat(expected));
                    })
                }
            }
        };
    }

    test_num_ops!(num_ops_f32, f32);
    test_num_ops!(num_ops_i32, i32);
    test_num_ops!(num_ops_i16, i16);
    test_num_ops!(num_ops_i8, i8);
    test_num_ops!(num_ops_u8, u8);
    test_num_ops!(num_ops_u16, u16);

    // Test that x8 multiply truncates result as expected.
    #[test]
    fn test_i8_mul_truncate() {
        test_simd_op!(isa, {
            let ops = isa.i8();

            let x = 17i8;
            let y = 19i8;

            let x_vec = ops.splat(x);
            let y_vec = ops.splat(y);
            let expected = ops.splat(x.wrapping_mul(y));
            let actual = ops.mul(x_vec, y_vec);

            assert_simd_eq!(actual, expected);
        })
    }

    #[test]
    fn test_u8_mul_truncate() {
        test_simd_op!(isa, {
            let ops = isa.u8();

            let x = 17u8;
            let y = 19u8;

            let x_vec = ops.splat(x);
            let y_vec = ops.splat(y);
            let expected = ops.splat(x.wrapping_mul(y));
            let actual = ops.mul(x_vec, y_vec);

            assert_simd_eq!(actual, expected);
        })
    }

    // Generate tests for operations available on all float types.
    macro_rules! test_float_ops {
        ($modname:ident, $elem:ident, $int_elem:ident) => {
            mod $modname {
                use super::{assert_simd_eq, test_simd_op, FloatOps, Isa, NumOps, Simd, SimdOp};

                #[test]
                fn test_div() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let x = ops.splat(1.);
                        let y = ops.splat(2.);
                        let expected = ops.splat(0.5);
                        let actual = ops.div(x, y);
                        assert_simd_eq!(actual, expected);
                    })
                }

                #[test]
                fn test_reciprocal() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let vals = [-5., -2., 2., 5.];
                        for v in vals {
                            let x = ops.splat(v);
                            let y = ops.reciprocal(x);
                            let expected = ops.splat(1. / v);
                            assert_simd_eq!(y, expected);
                        }
                    })
                }

                #[test]
                fn test_abs() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let vals = [-1., 0., 1.];
                        for v in vals {
                            let x = ops.splat(v);
                            let y = ops.abs(x);
                            let expected = ops.splat(v.abs());
                            assert_simd_eq!(y, expected);
                        }
                    })
                }

                #[test]
                fn test_neg() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let x = ops.splat(3 as $elem);

                        let expected = ops.splat(-3 as $elem);
                        let actual = ops.neg(x);
                        assert_simd_eq!(actual, expected);
                    })
                }

                #[test]
                fn test_to_int_trunc() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let x = ops.splat(12.345);
                        let y = ops.to_int_trunc(x);
                        let expected = isa.$int_elem().splat(12);
                        assert_simd_eq!(y, expected);
                    })
                }
            }
        };
    }

    test_float_ops!(float_ops_f32, f32, i32);

    // Generate tests for operations available on signed integer types.
    macro_rules! test_signed_int_ops {
        ($modname:ident, $elem:ident) => {
            mod $modname {
                use super::{
                    assert_simd_eq, test_simd_op, Isa, NumOps, SignedIntOps, Simd, SimdOp,
                };

                #[test]
                fn test_abs() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let vals = [-1, 0, 1];
                        for v in vals {
                            let x = ops.splat(v);
                            let y = ops.abs(x);
                            let expected = ops.splat(v.abs());
                            assert_simd_eq!(y, expected);
                        }
                    })
                }

                // Add / Sub / Mul with a negative argument.
                #[test]
                fn test_bin_ops_neg() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let a = -2 as $elem;
                        let b = 3 as $elem;

                        let x = ops.splat(a);
                        let y = ops.splat(b);

                        // Add
                        let expected = ops.splat(a + b);
                        let actual = ops.add(x, y);
                        assert_simd_eq!(actual, expected);

                        // Sub
                        let expected = ops.splat(b - a);
                        let actual = ops.sub(y, x);
                        assert_simd_eq!(actual, expected);

                        // Mul
                        let expected = ops.splat(a * b);
                        let actual = ops.mul(x, y);
                        assert_simd_eq!(actual, expected);
                    })
                }

                #[test]
                fn test_shl() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let x = ops.splat(42);
                        let y = ops.shift_left::<1>(x);
                        let expected = ops.splat(42 << 1);
                        assert_simd_eq!(y, expected);
                    })
                }

                #[test]
                fn test_neg() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let x = ops.splat(3 as $elem);

                        let expected = ops.splat(-3 as $elem);
                        let actual = ops.neg(x);
                        assert_simd_eq!(actual, expected);
                    })
                }
            }
        };
    }

    test_signed_int_ops!(int_ops_i32, i32);
    test_signed_int_ops!(int_ops_i16, i16);
    test_signed_int_ops!(int_ops_i8, i8);

    // For small positive values, signed comparison ops will work on unsigned
    // values. Make sure we really are using unsigned comparison.
    #[test]
    fn test_cmp_gt_ge_u16() {
        test_simd_op!(isa, {
            let ops = isa.u16();
            let x = ops.splat(i16::MAX as u16);
            let y = ops.splat(i16::MAX as u16 + 1);
            assert!(ops.gt(y, x).all_true());
            assert!(ops.ge(y, x).all_true());
        });
    }

    #[test]
    fn test_cmp_gt_ge_u8() {
        test_simd_op!(isa, {
            let ops = isa.u8();
            let x = ops.splat(i8::MAX as u8);
            let y = ops.splat(i8::MAX as u8 + 1);
            assert!(ops.gt(y, x).all_true());
            assert!(ops.ge(y, x).all_true());
        });
    }

    macro_rules! test_mask_ops {
        ($type:ident) => {
            test_simd_op!(isa, {
                let ops = isa.$type();
                let mask_ops = ops.mask_ops();

                // First-n mask
                let ones = ops.first_n_mask(ops.len());
                let zeros = ops.first_n_mask(0);
                let first = ops.first_n_mask(1);

                assert!(ones.all_true());
                assert!(zeros.all_false());

                // Bitwise and
                assert_simd_eq!(mask_ops.and(ones, ones), ones);
                assert_simd_eq!(mask_ops.and(first, ones), first);
                assert_simd_eq!(mask_ops.and(first, zeros), zeros);
            });
        };
    }

    #[test]
    fn test_mask_ops_f32() {
        test_mask_ops!(f32);
    }

    #[test]
    fn test_mask_ops_i32() {
        test_mask_ops!(i32);
    }

    #[test]
    fn test_mask_ops_i16() {
        test_mask_ops!(i16);
    }

    #[test]
    fn test_mask_ops_i8() {
        test_mask_ops!(i8);
    }

    macro_rules! test_narrow_saturate {
        ($test_name:ident, $src:ident, $dest:ident) => {
            #[test]
            fn $test_name() {
                test_simd_op!(isa, {
                    let ops = isa.$src();

                    let src: Vec<$src> = (0..ops.len() * 2).map(|x| x as $src).collect();
                    let expected: Vec<$dest> = src
                        .iter()
                        .map(|&x| x.clamp($dest::MIN as $src, $dest::MAX as $src) as $dest)
                        .collect();

                    let x_low = ops.load(&src[..ops.len()]);
                    let x_high = ops.load(&src[ops.len()..]);
                    let y = ops.narrow_saturate(x_low, x_high);

                    assert_eq!(y.to_array().as_ref(), expected);
                });
            }
        };
    }

    test_narrow_saturate!(test_narrow_i32_i16, i32, i16);
    test_narrow_saturate!(test_narrow_u16_u8, i16, u8);

    #[test]
    fn test_reinterpret_cast() {
        test_simd_op!(isa, {
            let x = 1.456f32;
            let x_i32 = x.to_bits() as i32;

            let x_vec = isa.f32().splat(x);
            let y_vec: I::I32 = x_vec.reinterpret_cast();

            let expected = isa.i32().splat(x_i32);
            assert_simd_eq!(y_vec, expected);
        })
    }
}
