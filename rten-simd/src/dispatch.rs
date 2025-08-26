use std::mem::MaybeUninit;

use crate::Isa;
use crate::functional::simd_map;
use crate::ops::{GetNumOps, GetSimd};
use crate::span::SrcDest;

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
pub trait SimdUnaryOp<T: GetSimd> {
    /// Evaluate the unary function on the elements in `x`.
    ///
    /// ```
    /// use rten_simd::{Isa, Simd, SimdUnaryOp};
    /// use rten_simd::ops::{FloatOps, NumOps};
    ///
    /// struct Reciprocal {}
    ///
    /// impl SimdUnaryOp<f32> for Reciprocal {
    ///     fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
    ///         let ops = isa.f32();
    ///         ops.div(ops.one(), x)
    ///     }
    /// }
    /// ```
    fn eval<I: Isa>(&self, isa: I, x: T::Simd<I>) -> T::Simd<I>;

    /// Evaluate the unary function on elements in `x`.
    ///
    /// This is a shorthand for `Self::default().eval(x)`. It is mainly useful
    /// when one vectorized operation needs to call another as part of its
    /// implementation.
    #[inline(always)]
    fn apply<I: Isa>(isa: I, x: T::Simd<I>) -> T::Simd<I>
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
struct SimdMapOp<'src, 'dst, 'op, T: GetSimd, Op: SimdUnaryOp<T>> {
    src_dest: SrcDest<'src, 'dst, T>,
    op: &'op Op,
}

impl<'src, 'dst, 'op, T: GetSimd, Op: SimdUnaryOp<T>> SimdMapOp<'src, 'dst, 'op, T, Op> {
    pub fn wrap(src_dest: SrcDest<'src, 'dst, T>, op: &'op Op) -> Self {
        SimdMapOp { src_dest, op }
    }
}

impl<'dst, T: GetNumOps + GetSimd, Op: SimdUnaryOp<T>> SimdOp for SimdMapOp<'_, 'dst, '_, T, Op> {
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
    use crate::Isa;
    use crate::ops::{FloatOps, GetNumOps, GetSimd, NumOps};

    #[test]
    fn test_unary_float_op() {
        struct Reciprocal {}

        impl SimdUnaryOp<f32> for Reciprocal {
            fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
                let ops = isa.f32();
                ops.div(ops.one(), x)
            }
        }

        let mut buf = [1., 2., 3., 4.];
        Reciprocal {}.map_mut(&mut buf);

        assert_eq!(buf, [1., 1. / 2., 1. / 3., 1. / 4.]);
    }

    #[test]
    fn test_unary_generic_op() {
        struct Double {}

        impl<T> SimdUnaryOp<T> for Double
        where
            T: GetSimd + GetNumOps,
        {
            fn eval<I: Isa>(&self, isa: I, x: T::Simd<I>) -> T::Simd<I> {
                let ops = T::num_ops(isa);
                ops.add(x, x)
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
