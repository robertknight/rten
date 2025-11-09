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
    /// Is this a SIMD-accelerated (ie. non-generic) ISA?
    const SIMD: bool;

    /// True if [`dot`](Int8DotIsa::dot)'s LHS argument is unsigned.
    const LHS_UNSIGNED: bool;

    /// The base SIMD instruction set.
    type Isa: Isa;

    fn isa(&self) -> Self::Isa;

    /// Compute the i32 dot product of each group of 4 elements in `a` with
    /// the corresponding group of 4 elements in `b` and add to `acc`.
    ///
    /// The LHS argument `a` may be interpreted as either signed or unsigned
    /// depending on the architecture. It is signed on Arm and unsigned on x86.
    /// The [`LHS_UNSIGNED`](Int8DotIsa::LHS_UNSIGNED) associated constant
    /// indicates which.
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
        // The target features enabled for each dispatch function should be
        // a superset of those used for the base ISA by the `SimdOp::dispatch`
        // impl in rten-simd.

        #[cfg(target_arch = "aarch64")]
        {
            // ISA extensions. There are no base ISA features as the "neon"
            // feature is always enabled.
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

        #[cfg(target_arch = "x86_64")]
        {
            // Base ISA features
            #[target_feature(enable = "avx512f")]
            #[target_feature(enable = "avx512vl")]
            #[target_feature(enable = "avx512bw")]
            #[target_feature(enable = "avx512dq")]
            // ISA extensions
            #[target_feature(enable = "avx512vnni")]
            unsafe fn dispatch_avx512_vnni<Op: SimdInt8DotOp>(
                isa: impl Int8DotIsa,
                op: Op,
            ) -> Op::Output {
                op.eval(isa)
            }

            if let Some(isa) = x86_64::Avx512VnniIsa::new() {
                return unsafe { dispatch_avx512_vnni(isa, self) };
            }

            // Base ISA features (no extensions required)
            #[target_feature(enable = "avx2")]
            #[target_feature(enable = "avx")]
            #[target_feature(enable = "fma")]
            unsafe fn dispatch_avx2<Op: SimdInt8DotOp>(isa: impl Int8DotIsa, op: Op) -> Op::Output {
                op.eval(isa)
            }

            if let Some(isa) = x86_64::Avx2Int8DotIsa::new() {
                return unsafe { dispatch_avx2(isa, self) };
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
    const SIMD: bool = false;
    const LHS_UNSIGNED: bool = false;

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
        const SIMD: bool = true;
        const LHS_UNSIGNED: bool = false;
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

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use rten_simd::Isa;
    use rten_simd::isa::{Avx2Isa, Avx512Isa};

    use super::Int8DotIsa;

    pub struct Avx512VnniIsa {
        isa: Avx512Isa,
    }

    impl Avx512VnniIsa {
        pub fn new() -> Option<Self> {
            let isa = Avx512Isa::new()?;
            if !is_x86_feature_detected!("avx512vnni") {
                return None;
            }
            Some(Self { isa })
        }
    }

    unsafe impl Int8DotIsa for Avx512VnniIsa {
        const SIMD: bool = true;
        const LHS_UNSIGNED: bool = true;

        type Isa = Avx512Isa;

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
            use std::arch::x86_64::_mm512_dpbusd_epi32;

            #[target_feature(enable = "avx512vnni")]
            #[inline]
            unsafe fn dot(
                a: <Avx512Isa as Isa>::I8,
                b: <Avx512Isa as Isa>::I8,
                acc: <Avx512Isa as Isa>::I32,
            ) -> <Avx512Isa as Isa>::I32 {
                _mm512_dpbusd_epi32(acc.0, a.0, b.0).into()
            }
            // Safety: Constructor checks "avx512vnni" feature is supported.
            unsafe { dot(a, b, acc) }
        }
    }

    pub struct Avx2Int8DotIsa {
        isa: Avx2Isa,
    }

    impl Avx2Int8DotIsa {
        pub fn new() -> Option<Self> {
            let isa = Avx2Isa::new()?;
            Some(Self { isa })
        }
    }

    unsafe impl Int8DotIsa for Avx2Int8DotIsa {
        const SIMD: bool = true;
        const LHS_UNSIGNED: bool = true;

        type Isa = Avx2Isa;

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
            use std::arch::x86_64::{
                _mm256_add_epi32, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
            };

            #[target_feature(enable = "avx2")]
            #[target_feature(enable = "avx")]
            #[target_feature(enable = "fma")]
            #[inline]
            unsafe fn dot(
                a: <Avx2Isa as Isa>::I8,
                b: <Avx2Isa as Isa>::I8,
                acc: <Avx2Isa as Isa>::I32,
            ) -> <Avx2Isa as Isa>::I32 {
                let tmp = _mm256_maddubs_epi16(a.0, b.0);
                let tmp = _mm256_madd_epi16(tmp, _mm256_set1_epi16(1));
                _mm256_add_epi32(acc.0, tmp).into()
            }
            // Safety: Constructor checks "avx2" feature is supported.
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
        // Input ranges chosen such that:
        //  - LHS is a positive value which produces the same result whether the
        //    input is treated as signed or unsigned.
        //  - RHS includes both negative and positive values
        //  - Length is at least max vector width (512 bits / 64 bytes)
        //  - Length is not a multiple of any SIMD vector width (so tail handling
        //    is exercised).
        let a: Vec<i8> = (0..65).collect();
        let b: Vec<i8> = (-1..64).collect();
        let expected = reference_dot(&a, &b);
        let dotprod = VecDot::new(&a, &b).dispatch();
        assert_eq!(dotprod, expected);

        let rev_dotprod = VecDot::new(&b, &a).dispatch();
        assert_eq!(rev_dotprod, expected);
    }
}
