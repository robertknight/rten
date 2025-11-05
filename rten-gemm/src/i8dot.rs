//! Portable SIMD extensions for int8 dot products.

use rten_simd::isa::GenericIsa;
use rten_simd::{Isa, Simd};

/// An extended [`Isa`] which adds int8 dot product operations.
///
/// # Safety
///
/// This has the safety requirements of [`Isa`], plus constructors must ensure
/// that additional operations provided must be supported on the system.
pub unsafe trait Int8DotIsa {
    /// The base SIMD instruction set.
    type Isa: Isa;

    fn isa(&self) -> Self::Isa;

    /// Compute the i32 dot product of each group of 4 elements in `a` with
    /// the corresponding group of 4 elements in `b` and add to `acc`.
    fn dot(
        &self,
        a: <Self::Isa as Isa>::I8,
        b: <Self::Isa as Isa>::I8,
        acc: <Self::Isa as Isa>::I32,
    ) -> <Self::Isa as Isa>::I32;
}

/// An extended [`SimdOp`](rten_simd::SimdOp) which enables the use of int8
/// dot product operations.
pub trait SimdInt8DotOp {
    type Output;

    fn eval<I: Int8DotIsa>(self, isa: I) -> Self::Output;

    /// Evaluate the operation using the preferred instruction set on the
    /// current platform.
    fn dispatch(self) -> Self::Output
    where
        Self: Sized,
    {
        #[cfg(target_arch = "aarch64")]
        {
            #[target_feature(enable = "dotprod")]
            unsafe fn dispatch_dotprod<Op: SimdInt8DotOp>(
                isa: impl Int8DotIsa,
                op: Op,
            ) -> Op::Output {
                op.eval(isa)
            }

            if let Some(isa) = aarch64::ArmInt8DotIsa::new() {
                // Safety: dotprod feature is supported
                return unsafe { dispatch_dotprod(isa, self) };
            }
        }

        self.eval(GenericInt8Dot::new())
    }
}

struct GenericInt8Dot {
    isa: GenericIsa,
}

impl GenericInt8Dot {
    fn new() -> Self {
        Self {
            isa: GenericIsa::new(),
        }
    }
}

unsafe impl Int8DotIsa for GenericInt8Dot {
    type Isa = GenericIsa;

    fn isa(&self) -> Self::Isa {
        self.isa
    }

    #[inline]
    fn dot(
        &self,
        a: <Self::Isa as Isa>::I8,
        b: <Self::Isa as Isa>::I8,
        acc: <Self::Isa as Isa>::I32,
    ) -> <Self::Isa as Isa>::I32 {
        let a = a.to_array();
        let b = b.to_array();
        let mut acc = acc.to_array();
        for group in 0..acc.len() {
            for i in 0..4 {
                acc[group] += a[group * 4 + i] as i32 * b[group * 4 + i] as i32;
            }
        }
        acc.into()
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use rten_simd::Isa;
    use rten_simd::isa::ArmNeonIsa;

    use super::Int8DotIsa;

    pub struct ArmInt8DotIsa {
        isa: ArmNeonIsa,
    }

    impl ArmInt8DotIsa {
        pub fn new() -> Option<Self> {
            let isa = rten_simd::isa::ArmNeonIsa::new()?;
            if !std::arch::is_aarch64_feature_detected!("dotprod") {
                return None;
            }
            Some(Self { isa })
        }
    }

    unsafe impl Int8DotIsa for ArmInt8DotIsa {
        type Isa = ArmNeonIsa;

        fn isa(&self) -> Self::Isa {
            self.isa
        }

        #[inline]
        fn dot(
            &self,
            a: <Self::Isa as Isa>::I8,
            b: <Self::Isa as Isa>::I8,
            acc: <Self::Isa as Isa>::I32,
        ) -> <Self::Isa as Isa>::I32 {
            #[target_feature(enable = "dotprod")]
            #[inline]
            unsafe fn dot(
                a: <ArmNeonIsa as Isa>::I8,
                b: <ArmNeonIsa as Isa>::I8,
                mut acc: <ArmNeonIsa as Isa>::I32,
            ) -> <ArmNeonIsa as Isa>::I32 {
                use core::arch::asm;
                unsafe {
                    // Use inline asm here because the `vdotq_s32` intrinsic is not
                    // stabilized yet.
                    asm! {
                        "sdot {result:v}.4s, {a:v}.16b, {b:v}.16b",
                        result = inout(vreg) acc,
                        a = in(vreg) a,
                        b = in(vreg) b,
                        options(nostack)
                    }
                }
                acc
            }
            // Safety: Constructor checks "dotprod" feature is supported.
            unsafe { dot(a, b, acc) }
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::ops::NumOps;
    use rten_simd::{Isa, SimdIterable};

    use super::{Int8DotIsa, SimdInt8DotOp};

    struct VecDot<'a> {
        a: &'a [i8],
        b: &'a [i8],
    }

    impl<'a> VecDot<'a> {
        fn new(a: &'a [i8], b: &'a [i8]) -> Self {
            Self { a, b }
        }
    }

    impl<'a> SimdInt8DotOp for VecDot<'a> {
        type Output = i32;

        fn eval<I: Int8DotIsa>(self, isa: I) -> Self::Output {
            let i8_ops = isa.isa().i8();
            let i32_ops = isa.isa().i32();

            let mut acc = i32_ops.zero();

            for (a, b) in self
                .a
                .simd_iter_pad(i8_ops)
                .zip(self.b.simd_iter_pad(i8_ops))
            {
                acc = isa.dot(a, b, acc)
            }

            i32_ops.sum(acc)
        }
    }

    fn reference_dot(a: &[i8], b: &[i8]) -> i32 {
        let mut acc = 0;
        for (x, y) in a.iter().zip(b) {
            acc += (*x as i32) * (*y as i32);
        }
        acc
    }

    #[test]
    fn test_simd_int8_dot_op() {
        // Input range chosen to include both negative and positive values and
        // to have a length that is not a multiple of any SIMD vector width.
        let a: Vec<i8> = (-16..17).collect();
        let b: Vec<i8> = (-1..32).collect();
        let expected = reference_dot(&a, &b);
        let dotprod = VecDot::new(&a, &b).dispatch();
        assert_eq!(dotprod, expected);

        let rev_dotprod = VecDot::new(&b, &a).dispatch();
        assert_eq!(rev_dotprod, expected);
    }
}
