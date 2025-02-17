use std::mem::MaybeUninit;

use super::functional::simd_map;
use super::{Elem, Isa, Simd, SimdFloat};
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
/// dispatch to [`SimdOp::eval`], passing the selected [`SimdIsa`].
pub fn dispatch<Op: SimdOp>(op: Op) -> Op::Output {
    #[cfg(target_arch = "aarch64")]
    if let Some(isa) = super::arch::aarch64::ArmNeonIsa::new() {
        return op.eval(isa);
    }
    panic!("no ISA found")
}

/// Convenience trait for defining vectorized unary operations.
pub trait SimdUnaryFloatOp {
    type Elem: Elem;

    /// Evaluate the unary function on the elements in `x`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `S` is a supported SIMD vector type
    /// on the current system.
    fn eval<S: Simd<Elem = Self::Elem> + SimdFloat>(&self, x: S) -> S;

    /// Evaluate the unary function on elements in `x`.
    ///
    /// This is a shorthand for `Self::default().eval(x)`. It is mainly useful
    /// when one vectorized operation needs to call another as part of its
    /// implementation.
    #[inline(always)]
    fn apply<S: Simd<Elem = Self::Elem> + SimdFloat>(x: S) -> S
    where
        Self: Default,
    {
        Self::default().eval(x)
    }

    /// Apply this function to a slice.
    ///
    /// This reads elements from `input` in SIMD vector-sized chunks, applies
    /// `op` and writes the results to `output`.
    #[allow(private_bounds)]
    fn map(&self, input: &[Self::Elem], output: &mut [MaybeUninit<Self::Elem>])
    where
        Self: Sized,
        for<'a> SimdMapOp<'a, Self>: SimdOp,
    {
        let wrapped_op = SimdMapOp::wrap((input, output).into(), self);
        dispatch(wrapped_op);
    }

    /// Apply a vectorized unary function to a mutable slice.
    ///
    /// This is similar to [`map`](SimdUnaryFloatOp::map) but reads and writes
    /// to the same slice.
    #[allow(private_bounds)]
    fn map_mut(&self, input: &mut [Self::Elem])
    where
        Self: Sized,
        for<'a> SimdMapOp<'a, Self>: SimdOp,
    {
        let wrapped_op = SimdMapOp::wrap(input.into(), self);
        dispatch(wrapped_op);
    }
}

/// SIMD operation which applies a unary operator `Op` to all elements in
/// an input buffer using [`simd_map`].
struct SimdMapOp<'a, Op: SimdUnaryFloatOp> {
    src_dest: SrcDest<'a, Op::Elem>,
    op: &'a Op,
}

impl<'a, Op: SimdUnaryFloatOp> SimdMapOp<'a, Op> {
    pub fn wrap(src_dest: SrcDest<'a, Op::Elem>, op: &'a Op) -> SimdMapOp<'a, Op> {
        SimdMapOp { src_dest, op }
    }
}

impl<'a, Op: SimdUnaryFloatOp<Elem = f32>> SimdOp for SimdMapOp<'a, Op> {
    type Output = &'a mut [Op::Elem];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        simd_map(
            isa.f32(),
            self.src_dest,
            #[inline(always)]
            |x| self.op.eval(x),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::SimdUnaryFloatOp;
    use crate::safe::{MakeSimd, Simd, SimdFloat};

    #[test]
    fn test_unary_float_op() {
        struct Reciprocal {}

        impl SimdUnaryFloatOp for Reciprocal {
            type Elem = f32;

            fn eval<S: Simd<Elem = f32> + SimdFloat>(&self, x: S) -> S {
                x.init().one() / x
            }
        }

        let mut buf = [1., 2., 3., 4.];
        Reciprocal {}.map_mut(&mut buf);

        assert_eq!(buf, [1., 1. / 2., 1. / 3., 1. / 4.]);
    }
}
