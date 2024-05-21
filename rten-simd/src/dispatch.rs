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
    pub fn dispatch<Op: SimdOp>(&self, op: Op) {
        #[cfg(feature = "avx512")]
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn simd_op_avx512<Op: SimdOp>(op: Op) {
            use std::arch::x86_64::__m512;
            op.eval::<__m512>();
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        unsafe fn simd_op_avx<Op: SimdOp>(op: Op) {
            use std::arch::x86_64::__m256;
            op.eval::<__m256>();
        }

        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            if crate::is_avx512_supported() {
                unsafe { simd_op_avx512(op) };
                return;
            }

            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                // Safety: We've checked that AVX2 + FMA are available.
                unsafe { simd_op_avx(op) };
                return;
            }
        }

        #[cfg(target_arch = "wasm32")]
        #[cfg(target_feature = "simd128")]
        {
            use crate::arch::wasm::v128f;

            // Safety: The WASM runtime will have verified SIMD instructions
            // are accepted when loading the binary.
            unsafe { op.eval::<v128f>() };
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::float32x4_t;
            unsafe { op.eval::<float32x4_t>() };
            return;
        }

        // Generic fallback.
        unsafe { op.eval::<f32>() };
    }
}

/// Trait for SIMD operations which can be evaluated using different SIMD
/// vector types.
///
/// To dispatch the operation, create a [SimdDispatcher] and call
/// [`dispatch(op)`](SimdDispatcher::dispatch).
pub trait SimdOp {
    /// Evaluate the operator using a given SIMD vector type.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `S` is a supported SIMD vector type
    /// on the current system.
    unsafe fn eval<S: SimdFloat>(&self);
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
}

/// Apply a vectorized unary function to elements of `input` using [simd_map].
pub fn dispatch_map_op<Op: SimdUnaryOp>(input: &[f32], out: &mut [MaybeUninit<f32>], op: Op) {
    let wrapped_op = SimdMapOp::wrap(input.into(), out.into(), op, 0. /* pad */);
    let dispatcher = SimdDispatcher::default();
    dispatcher.dispatch(wrapped_op);
}

/// Apply a vectorized unary function in-place to elements of `input`.
pub fn dispatch_map_op_in_place<Op: SimdUnaryOp>(input: &mut [f32], op: Op) {
    let out: MutPtrLen<f32> = input.into();
    let wrapped_op = SimdMapOp::wrap(input.into(), out.as_uninit(), op, 0. /* pad */);
    let dispatcher = SimdDispatcher::default();
    dispatcher.dispatch(wrapped_op);
}

/// SIMD operation which applies a unary operator `Op` to all elements in
/// an input buffer using [simd_map].
pub struct SimdMapOp<Op: SimdUnaryOp> {
    input: PtrLen<f32>,
    output: MutPtrLen<MaybeUninit<f32>>,
    op: Op,
    pad: f32,
}

impl<Op: SimdUnaryOp> SimdMapOp<Op> {
    pub fn wrap(
        input: PtrLen<f32>,
        output: MutPtrLen<MaybeUninit<f32>>,
        op: Op,
        pad: f32,
    ) -> SimdMapOp<Op> {
        SimdMapOp {
            input,
            output,
            op,
            pad,
        }
    }
}

impl<Op: SimdUnaryOp> SimdOp for SimdMapOp<Op> {
    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(&self) {
        simd_map(
            self.input,
            self.output,
            #[inline(always)]
            |x: S| self.op.eval(x),
            self.pad,
        );
    }
}
