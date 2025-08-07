use std::mem::MaybeUninit;

use crate::functional::simd_map;
use crate::ops::GetNumOps;
use crate::span::SrcDest;
use crate::{Elem, Isa, Simd};

/// A vectorized operation which can be instantiated for different instruction
/// sets.
pub trait SimdOp {
    /// The type of the operation's result.
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
        {
            // The target features enabled here must match those tested for by `Avx512Isa::new`.
            #[target_feature(enable = "avx512f")]
            #[target_feature(enable = "avx512vl")]
            #[target_feature(enable = "avx512bw")]
            #[target_feature(enable = "avx512dq")]
            unsafe fn dispatch_avx512<Op: SimdOp>(isa: impl Isa, op: Op) -> Op::Output {
                op.eval(isa)
            }

            if let Some(isa) = super::arch::x86_64::Avx512Isa::new() {
                // Safety: AVX-512 is supported
                unsafe {
                    return dispatch_avx512(isa, op);
                }
            }
        }

        // The target features enabled here must match those tested for by `Avx2Isa::new`.
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
    /// In order to perform operations on `x`, it will need to be cast to
    /// the specific type used by the ISA:
    ///
    /// ```
    /// use rten_simd::{Isa, Simd, SimdUnaryOp};
    /// use rten_simd::ops::{FloatOps, NumOps};
    ///
    /// struct Reciprocal {}
    ///
    /// impl SimdUnaryOp<f32> for Reciprocal {
    ///     fn eval<I: Isa, S: Simd<Elem=f32, Isa=I>>(&self, isa: I, x: S) -> S {
    ///         let ops = isa.f32();
    ///         let x = x.same_cast();
    ///         let reciprocal = ops.div(ops.one(), x);
    ///         reciprocal.same_cast()
    ///     }
    /// }
    /// ```
    fn eval<I: Isa, S: Simd<Elem = T, Isa = I>>(&self, isa: I, x: S) -> S;

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
        Self::default().eval(isa, x)
    }

    /// Apply this function to a slice.
    ///
    /// This reads elements from `input` in SIMD vector-sized chunks, applies
    /// the operation and writes the results to `output`.
    #[allow(private_bounds)]
    fn map(&self, input: &[T], output: &mut [MaybeUninit<T>])
    where
        Self: Sized,
        T: GetNumOps,
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
        T: GetNumOps,
    {
        let wrapped_op = SimdMapOp::wrap(input.into(), self);
        dispatch(wrapped_op);
    }

    /// Apply this operation to a single element.
    #[allow(private_bounds)]
    fn scalar_eval(&self, x: T) -> T
    where
        Self: Sized,
        T: GetNumOps,
    {
        let mut array = [x];
        self.map_mut(&mut array);
        array[0]
    }
}

/// SIMD operation which applies a unary operator `Op` to all elements in
/// an input buffer using [`simd_map`].
struct SimdMapOp<'src, 'dst, 'op, T: Elem, Op: SimdUnaryOp<T>> {
    src_dest: SrcDest<'src, 'dst, T>,
    op: &'op Op,
}

impl<'src, 'dst, 'op, T: Elem, Op: SimdUnaryOp<T>> SimdMapOp<'src, 'dst, 'op, T, Op> {
    pub fn wrap(src_dest: SrcDest<'src, 'dst, T>, op: &'op Op) -> Self {
        SimdMapOp { src_dest, op }
    }
}

impl<'dst, T: GetNumOps, Op: SimdUnaryOp<T>> SimdOp for SimdMapOp<'_, 'dst, '_, T, Op> {
    type Output = &'dst mut [T];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        simd_map(
            T::num_ops(isa),
            self.src_dest,
            #[inline(always)]
            |x| self.op.eval(isa, x),
        )
    }
}

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
    use crate::ops::{FloatOps, GetNumOps, NumOps};
    use crate::{Isa, Simd};

    #[test]
    fn test_unary_float_op() {
        struct Reciprocal {}

        impl SimdUnaryOp<f32> for Reciprocal {
            fn eval<I: Isa, S: Simd<Elem = f32, Isa = I>>(&self, isa: I, x: S) -> S {
                let ops = isa.f32();
                let x = x.same_cast();
                let y = ops.div(ops.one(), x);
                y.same_cast()
            }
        }

        let mut buf = [1., 2., 3., 4.];
        Reciprocal {}.map_mut(&mut buf);

        assert_eq!(buf, [1., 1. / 2., 1. / 3., 1. / 4.]);
    }

    #[test]
    fn test_unary_generic_op() {
        struct Double {}

        impl<T: GetNumOps> SimdUnaryOp<T> for Double {
            fn eval<I: Isa, S: Simd<Elem = T, Isa = I>>(&self, isa: I, x: S) -> S {
                let ops = T::num_ops(isa);
                let x = x.same_cast();
                ops.add(x, x).same_cast()
            }
        }

        let mut buf = [1i32, 2, 3, 4];
        Double {}.map_mut(&mut buf);
        assert_eq!(buf, [2, 4, 6, 8]);

        let mut buf = [1.0f32, 2., 3., 4.];
        Double {}.map_mut(&mut buf);
        assert_eq!(buf, [2., 4., 6., 8.]);
    }
}
