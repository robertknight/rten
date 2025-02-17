use std::mem::MaybeUninit;

use super::functional::simd_map;
use super::{Elem, Isa, Simd};
use crate::span::SrcDest;

/// A vectorized operation which can be instantiated for different instruction
/// sets.
pub trait SimdOp {
    type Output;

    /// Evaluate the operation using the given instruction set.
    fn eval<I: Isa>(self, isa: I) -> Self::Output;

    /// Dispatch this operation using the preferred ISA for the current platform.
    fn dispatch(self) -> Self::Output
    where
        Self: Sized,
    {
        dispatch(self)
    }
}

/// Invoke a SIMD operation using the preferred ISA for the current system.
///
/// This function will check the available SIMD instruction sets and then
/// dispatch to [`SimdOp::eval`], passing the selected [`Isa`].
pub fn dispatch<Op: SimdOp>(op: Op) -> Op::Output {
    #[cfg(target_arch = "aarch64")]
    if let Some(isa) = super::arch::aarch64::ArmNeonIsa::new() {
        return op.eval(isa);
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "avx")]
        #[target_feature(enable = "fma")]
        unsafe fn dispatch_avx2<Op: SimdOp>(isa: impl Isa, op: Op) -> Op::Output {
            op.eval(isa)
        }

        if let Some(isa) = super::arch::x86_64::Avx2Isa::new() {
            // Safety: AVX2 is supported
            unsafe {
                return dispatch_avx2(isa, op);
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[cfg(target_feature = "simd128")]
    {
        if let Some(isa) = super::arch::wasm32::Wasm32Isa::new() {
            return op.eval(isa);
        }
    }

    let isa = super::arch::generic::GenericIsa::new();
    op.eval(isa)
}

/// Convenience trait for defining vectorized unary operations.
pub trait SimdUnaryOp<T: Elem> {
    /// Evaluate the unary function on the elements in `x`.
    ///
    /// `eval` is passed an untyped SIMD vector. This can be cast to the
    /// specific type expected by the operation.
    ///
    /// ```
    /// use rten_simd::safe::{Isa, Simd, SimdFloatOps, SimdOps, SimdUnaryOp};
    ///
    /// struct Reciprocal {}
    ///
    /// impl SimdUnaryOp<f32> for Reciprocal {
    ///     fn eval<I: Isa>(&self, isa: I, x: I::Bits) -> I::Bits {
    ///         let ops = isa.f32();
    ///         let x = ops.from_bits(x);
    ///         let reciprocal = ops.div(ops.one(), x);
    ///         reciprocal.to_bits()
    ///     }
    /// }
    /// ```
    fn eval<I: Isa>(&self, isa: I, x: I::Bits) -> I::Bits;

    /// Evaluate the unary function on elements in `x`.
    ///
    /// This is a shorthand for `Self::default().eval(x)`. It is mainly useful
    /// when one vectorized operation needs to call another as part of its
    /// implementation.
    #[inline(always)]
    fn apply<I: Isa, S: Simd<Elem = T, Isa = I>>(isa: I, x: S) -> S
    where
        Self: Default,
    {
        S::from_bits(Self::default().eval(isa, x.to_bits()))
    }

    /// Apply this function to a slice.
    ///
    /// This reads elements from `input` in SIMD vector-sized chunks, applies
    /// the operation and writes the results to `output`.
    #[allow(private_bounds)]
    fn map(&self, input: &[T], output: &mut [MaybeUninit<T>])
    where
        Self: Sized,
        for<'a> SimdMapOp<'a, T, Self>: SimdOp,
    {
        let wrapped_op = SimdMapOp::wrap((input, output).into(), self);
        dispatch(wrapped_op);
    }

    /// Apply a vectorized unary function to a mutable slice.
    ///
    /// This is similar to [`map`](SimdUnaryOp::map) but reads and writes
    /// to the same slice.
    #[allow(private_bounds)]
    fn map_mut(&self, input: &mut [T])
    where
        Self: Sized,
        for<'a> SimdMapOp<'a, T, Self>: SimdOp,
    {
        let wrapped_op = SimdMapOp::wrap(input.into(), self);
        dispatch(wrapped_op);
    }
}

/// SIMD operation which applies a unary operator `Op` to all elements in
/// an input buffer using [`simd_map`].
struct SimdMapOp<'a, T: Elem, Op: SimdUnaryOp<T>> {
    src_dest: SrcDest<'a, T>,
    op: &'a Op,
}

impl<'a, T: Elem, Op: SimdUnaryOp<T>> SimdMapOp<'a, T, Op> {
    pub fn wrap(src_dest: SrcDest<'a, T>, op: &'a Op) -> Self {
        SimdMapOp { src_dest, op }
    }
}

macro_rules! impl_simd_map_op {
    ($type:ident, $cap_type:ident) => {
        impl<'a, Op: SimdUnaryOp<$type>> SimdOp for SimdMapOp<'a, $type, Op> {
            type Output = &'a mut [$type];

            #[inline(always)]
            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                simd_map(
                    isa.$type(),
                    self.src_dest,
                    #[inline(always)]
                    |x| I::$cap_type::from_bits(self.op.eval(isa, x.to_bits())),
                )
            }
        }
    };
}

impl_simd_map_op!(f32, F32);
impl_simd_map_op!(i32, I32);

/// Convenience macro for defining and evaluating a SIMD operation.
#[cfg(test)]
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

#[cfg(test)]
pub(crate) use test_simd_op;

#[cfg(test)]
mod tests {
    use super::SimdUnaryOp;
    use crate::safe::{Isa, Simd, SimdFloatOps, SimdOps};

    #[test]
    fn test_unary_float_op() {
        struct Reciprocal {}

        impl SimdUnaryOp<f32> for Reciprocal {
            fn eval<I: Isa>(&self, isa: I, x: I::Bits) -> I::Bits {
                let ops = isa.f32();
                let x = ops.from_bits(x);
                let y = ops.div(ops.one(), x);
                y.to_bits()
            }
        }

        let mut buf = [1., 2., 3., 4.];
        Reciprocal {}.map_mut(&mut buf);

        assert_eq!(buf, [1., 1. / 2., 1. / 3., 1. / 4.]);
    }

    #[test]
    fn test_unary_int_op() {
        struct Double {}

        impl SimdUnaryOp<i32> for Double {
            fn eval<I: Isa>(&self, isa: I, x: I::Bits) -> I::Bits {
                let ops = isa.i32();
                let x = ops.from_bits(x);
                ops.add(x, x).to_bits()
            }
        }

        let mut buf = [1, 2, 3, 4];
        Double {}.map_mut(&mut buf);

        assert_eq!(buf, [2, 4, 6, 8]);
    }
}
