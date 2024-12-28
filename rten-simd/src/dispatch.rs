//! Dispatch SIMD operations using the preferred SIMD instruction set for the
//! current system, as determined at runtime.

use std::mem::MaybeUninit;

use crate::functional::simd_map;
use crate::span::{MutPtrLen, PtrLen};
use crate::SimdFloat;

/// Dispatches SIMD operations using the preferred SIMD types for the current
/// platform.
#[derive(Default)]
pub struct SimdDispatcher {}

impl SimdDispatcher {
    /// Evaluate `op` using the preferred SIMD instruction set for the current
    /// system.
    #[allow(unused_imports)]
    #[allow(unreachable_code)] // Ignore fallback, if unused
    pub fn dispatch<Op: SimdOp>(&self, op: Op) -> Op::Output {
        #[cfg(feature = "avx512")]
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn simd_op_avx512<Op: SimdOp>(op: Op) -> Op::Output {
            use std::arch::x86_64::__m512;
            op.eval::<__m512>()
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        unsafe fn simd_op_avx<Op: SimdOp>(op: Op) -> Op::Output {
            use std::arch::x86_64::__m256;
            op.eval::<__m256>()
        }

        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            if crate::is_avx512_supported() {
                return unsafe { simd_op_avx512(op) };
            }

            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                // Safety: We've checked that AVX2 + FMA are available.
                return unsafe { simd_op_avx(op) };
            }
        }

        #[cfg(target_arch = "wasm32")]
        #[cfg(target_feature = "simd128")]
        {
            use crate::arch::wasm::v128f;

            // Safety: The WASM runtime will have verified SIMD instructions
            // are accepted when loading the binary.
            return unsafe { op.eval::<v128f>() };
        }

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::float32x4_t;
            return unsafe { op.eval::<float32x4_t>() };
        }

        // Generic fallback.
        unsafe { op.eval::<f32>() }
    }
}

/// Run `op` using the default SIMD dispatch configuration.
pub fn dispatch<Op: SimdOp>(op: Op) -> Op::Output {
    SimdDispatcher::default().dispatch(op)
}

/// Trait for SIMD operations which can be evaluated using different SIMD
/// vector types.
///
/// To dispatch the operation, create a [`SimdDispatcher`] and call
/// [`dispatch(op)`](SimdDispatcher::dispatch).
pub trait SimdOp {
    /// Output type returned by `eval`.
    type Output;

    /// Evaluate the operator using a given SIMD vector type.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `S` is a supported SIMD vector type
    /// on the current system.
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output;
}

/// Trait for evaluating a unary function on a SIMD vector.
pub trait SimdUnaryOp {
    /// Evaluate the unary function on the elements in `x`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `S` is a supported SIMD vector type
    /// on the current system.
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S;

    /// Evaluate the unary function on `x`.
    fn scalar_eval(&self, x: f32) -> f32 {
        // Safety: `f32` is a supported "SIMD" type on all platforms.
        unsafe { self.eval(x) }
    }

    /// Apply this function to a slice.
    ///
    /// This reads elements from `input` in SIMD vector-sized chunks, applies
    /// `op` and writes the results to `output`.
    fn map(&self, input: &[f32], output: &mut [MaybeUninit<f32>])
    where
        Self: Sized,
    {
        let wrapped_op = SimdMapOp::wrap(input.into(), output.into(), self);
        dispatch(wrapped_op)
    }

    /// Apply a vectorized unary function to a mutable slice.
    ///
    /// This is similar to [`map`](SimdUnaryOp::map) but reads and writes to the
    /// same slice.
    fn map_mut(&self, input: &mut [f32])
    where
        Self: Sized,
    {
        let out: MutPtrLen<f32> = input.into();
        let wrapped_op = SimdMapOp::wrap(input.into(), out.as_uninit(), self);
        dispatch(wrapped_op)
    }
}

/// SIMD operation which applies a unary operator `Op` to all elements in
/// an input buffer using [`simd_map`].
pub struct SimdMapOp<'a, Op: SimdUnaryOp> {
    input: PtrLen<f32>,
    output: MutPtrLen<MaybeUninit<f32>>,
    op: &'a Op,
}

impl<'a, Op: SimdUnaryOp> SimdMapOp<'a, Op> {
    pub fn wrap(
        input: PtrLen<f32>,
        output: MutPtrLen<MaybeUninit<f32>>,
        op: &'a Op,
    ) -> SimdMapOp<'a, Op> {
        SimdMapOp { input, output, op }
    }
}

impl<Op: SimdUnaryOp> SimdOp for SimdMapOp<'_, Op> {
    type Output = ();

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) {
        simd_map(
            self.input,
            self.output,
            #[inline(always)]
            |x: S| self.op.eval(x),
        );
    }
}
