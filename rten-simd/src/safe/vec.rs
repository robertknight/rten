use std::fmt::Debug;

/// Types used as elements (or _lanes_) of SIMD vectors.
pub trait Elem: Copy + Default + std::ops::Add<Output = Self> {
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

/// Masks used or returned by SIMD operations.
///
/// Most operations on masks are available via the [`MaskOps`] trait.
/// Implementations are obtained via [`SimdOps::mask_ops`].
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

    /// Operations on SIMD vectors with `f32` elements.
    fn f32(self) -> impl SimdFloatOps<Self::F32, Int = Self::I32>;

    /// Operations on SIMD vectors with `i32` elements.
    fn i32(self) -> impl SimdIntOps<Self::I32>;

    /// Operations on SIMD vectors with `i16` elements.
    fn i16(self) -> impl SimdIntOps<Self::I16>;
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

/// Trait for SIMD operations on a particular vector type.
///
/// # Safety
///
/// Implementations must ensure they can only be constructed if the
/// instruction set is supported on the current system.
#[allow(clippy::len_without_is_empty)]
pub unsafe trait SimdOps<S: Simd>: Copy {
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
    fn sum(self, x: S) -> S::Elem {
        let mut sum = S::Elem::default();
        for elem in x.to_array() {
            sum = sum + elem;
        }
        sum
    }
}

/// Extends [`SimdOps`] with operations available on SIMD vectors with float
/// elements.
pub trait SimdFloatOps<S: Simd>: SimdOps<S> {
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

/// Extends [`SimdOps`] with operations available on SIMD vectors with signed
/// integer elements.
pub trait SimdIntOps<S: Simd>: SimdOps<S> {
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

#[cfg(test)]
mod tests {
    use crate::safe::{
        assert_simd_eq, test_simd_op, Isa, Mask, MaskOps, Simd, SimdFloatOps, SimdIntOps, SimdOp,
        SimdOps,
    };

    // Generate tests for operations available on all numeric types.
    macro_rules! test_num_ops {
        ($modname:ident, $elem:ident) => {
            mod $modname {
                use super::{assert_simd_eq, test_simd_op, Isa, Mask, Simd, SimdOp, SimdOps};

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

                        let a: $elem = (1u8).into();
                        let b: $elem = (2u8).into();

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
                fn test_sum() {
                    test_simd_op!(isa, {
                        let ops = isa.$elem();

                        let vec: Vec<_> = (0..ops.len()).map(|x| x as $elem).collect();
                        let expected = vec.iter().sum::<$elem>();

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

    // Generate tests for operations available on all float types.
    macro_rules! test_float_ops {
        ($modname:ident, $elem:ident, $int_elem:ident) => {
            mod $modname {
                use super::{
                    assert_simd_eq, test_simd_op, Isa, Simd, SimdFloatOps, SimdOp, SimdOps,
                };

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
    macro_rules! test_int_ops {
        ($modname:ident, $elem:ident) => {
            mod $modname {
                use super::{assert_simd_eq, test_simd_op, Isa, Simd, SimdIntOps, SimdOp, SimdOps};

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

    test_int_ops!(int_ops_i32, i32);
    test_int_ops!(int_ops_i16, i16);

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
