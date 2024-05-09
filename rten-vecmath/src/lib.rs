//! SIMD-vectorized implementations of various math functions that are commonly
//! used in neural networks.
//!
//! For each function in this library there are multiple variants, which
//! typically include:
//!
//!  - A version that operates on scalars
//!  - A version that reads values from an input slice and writes to the
//!    corresponding position in an equal-length output slice. These have a
//!    `vec_` prefix.
//!  - A version that reads values from a mutable input slice and writes
//!    the computed values back in-place. These have a `vec_` prefix and
//!    `_in_place` suffix.
//!
//! All variants use the same underlying implementation and should have the
//! same accuracy.
//!
//! See the source code for comments on accuracy.

#![cfg_attr(
    feature = "avx512",
    feature(stdarch_x86_avx512),
    feature(avx512_target_feature)
)]

use std::mem::MaybeUninit;

mod erf;
mod exp;
pub mod simd_vec;
mod softmax;
mod tanh;
mod ulp;

#[cfg(test)]
mod testing;

pub use erf::{erf, vec_erf, vec_erf_in_place};
pub use exp::{exp, sigmoid, vec_exp, vec_exp_in_place, vec_sigmoid, vec_sigmoid_in_place};
use simd_vec::SimdFloat;
pub use softmax::{vec_softmax, vec_softmax_in_place};
pub use tanh::{tanh, vec_tanh, vec_tanh_in_place};

/// Detect availability of AVX-512 on macOS, where `is_x86_feature_detected`
/// can return false even if AVX-512 is available.
///
/// See https://github.com/golang/go/issues/43089. Go chose to use the
/// `commpage` to get the info. We use `sysctlbyname` instead since it is
/// a documented API.
#[cfg(feature = "avx512")]
#[cfg(target_os = "macos")]
fn test_for_avx512_on_macos() -> bool {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    #[link(name = "c")]
    extern "C" {
        /// See https://developer.apple.com/documentation/kernel/1387446-sysctlbyname.
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> c_int;
    }

    static AVX512_AVAILABLE: OnceLock<bool> = OnceLock::new();

    *AVX512_AVAILABLE.get_or_init(|| {
        unsafe {
            let mut ret = 0u64;
            let mut size = std::mem::size_of::<u64>();

            // We test only for avx512vl, as this implies avx512f.
            let sysctl_ret = sysctlbyname(
                CStr::from_bytes_with_nul(b"hw.optional.avx512vl\0")
                    .unwrap()
                    .as_ptr(),
                std::mem::transmute(&mut ret),
                &mut size,
                std::ptr::null(),
                0,
            );
            sysctl_ret == 0 && ret == 1
        }
    })
}

/// Test if the current system has basic AVX-512 support (AVX-512 F, AVX-512 VL).
///
/// This is unfortunately not as simple as using `is_x86_feature_detected`
/// because that can return incorrect results on macOS.
#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
pub fn is_avx512_supported() -> bool {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
        true
    } else {
        #[cfg(target_os = "macos")]
        {
            test_for_avx512_on_macos()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

/// Maximum SIMD vector size supported by this library, in units of 32-bit lanes.
///
/// Chosen as 16 to match AVX-512.
const MAX_LEN: usize = 16;

/// Const pointer to a range of `T`s.
///
/// This is like an `&[T]`, but without the guarantee that no mutable aliases
/// exist. This is useful as it enables re-using the same unsafe code for
/// mutating and non-mutating variants of a function.
#[derive(Copy, Clone)]
struct PtrLen<T> {
    ptr: *const T,
    len: usize,
}

impl<'a, T> From<&'a [T]> for PtrLen<T> {
    fn from(val: &'a [T]) -> PtrLen<T> {
        PtrLen {
            ptr: val.as_ptr(),
            len: val.len(),
        }
    }
}

impl<'a, T> From<&'a mut [T]> for PtrLen<T> {
    fn from(val: &'a mut [T]) -> PtrLen<T> {
        PtrLen {
            ptr: val.as_ptr(),
            len: val.len(),
        }
    }
}

impl<T> From<MutPtrLen<T>> for PtrLen<T> {
    fn from(val: MutPtrLen<T>) -> PtrLen<T> {
        PtrLen {
            ptr: val.ptr,
            len: val.len,
        }
    }
}

/// Mutable pointer to a range of `T`s.
///
/// This is like an `&mut [T]`, but without the guarantee that no aliases exist.
#[derive(Copy, Clone)]
struct MutPtrLen<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> MutPtrLen<MaybeUninit<T>> {
    /// Promise that the span of `T`s that are pointed to have been initialized.
    unsafe fn assume_init(self) -> MutPtrLen<T> {
        MutPtrLen {
            ptr: unsafe { std::mem::transmute(self.ptr) },
            len: self.len,
        }
    }
}

impl<T> MutPtrLen<T> {
    /// Transmute a span of initialized `T`s to uninitialized `T`s.
    fn as_uninit(self) -> MutPtrLen<MaybeUninit<T>>
    where
        T: Copy,
    {
        MutPtrLen {
            ptr: unsafe { std::mem::transmute(self.ptr) },
            len: self.len,
        }
    }
}

impl<'a, T> From<&'a mut [T]> for MutPtrLen<T> {
    fn from(val: &'a mut [T]) -> MutPtrLen<T> {
        MutPtrLen {
            ptr: val.as_mut_ptr(),
            len: val.len(),
        }
    }
}

/// Apply a unary operation to each element in `xs` and store the results in
/// `out`.
///
/// When this function returns, all elements in `out` will have been
/// initialized.
///
/// The operation is applied to SIMD vector-sized groups of elements at
/// a time using `simd_op`. If the final group has a size that is smaller than
/// the SIMD vector width, `simd_op` will be called with a SIMD vector that
/// is padded with `pad` on the right.
///
/// Safety: The caller must ensure that `xs` and `out` are valid pointers
/// to buffers of the expected lengths.
#[inline(always)]
unsafe fn vec_unary_op<S: SimdFloat, Op: FnMut(S) -> S>(
    xs: PtrLen<f32>,
    out: MutPtrLen<MaybeUninit<f32>>,
    mut simd_op: Op,
    pad: f32,
) {
    assert!(xs.len == out.len);

    let mut n = xs.len;
    let mut x_ptr = xs.ptr;
    let mut out_ptr = out.ptr;

    // S::LEN can't be used as the array size due to const generics limitations.
    const MAX_LEN: usize = 16;
    assert!(S::LEN <= MAX_LEN);
    let mut remainder = [pad; MAX_LEN];

    // Main loop over full vectors.
    while n >= S::LEN {
        let x = S::load(x_ptr);
        let y = simd_op(x);
        y.store(out_ptr as *mut f32);

        n -= S::LEN;
        x_ptr = x_ptr.add(S::LEN);
        out_ptr = out_ptr.add(S::LEN);
    }

    // Handler remainder with a padded vector.
    if n > 0 {
        for i in 0..n {
            remainder[i] = *x_ptr.add(i);
        }

        let x = S::load(remainder.as_ptr());
        let y = simd_op(x);
        y.store(remainder.as_mut_ptr());

        for i in 0..n {
            out_ptr.add(i).write(MaybeUninit::new(remainder[i]));
        }
    }
}

#[inline(always)]
unsafe fn vec_fold<S: SimdFloat, Op: Fn(S, S) -> S>(
    xs: PtrLen<f32>,
    mut accum: S,
    simd_op: Op,
    pad: f32,
) -> S {
    let mut n = xs.len;
    let mut x_ptr = xs.ptr;

    // S::LEN can't be used as the array size due to const generics limitations.
    assert!(S::LEN <= MAX_LEN);
    let mut remainder = [pad; MAX_LEN];

    // Main loop over full vectors.
    while n >= S::LEN {
        let x = S::load(x_ptr);
        accum = simd_op(accum, x);
        n -= S::LEN;
        x_ptr = x_ptr.add(S::LEN);
    }

    // Handler remainder with a padded vector.
    if n > 0 {
        for i in 0..n {
            remainder[i] = *x_ptr.add(i);
        }
        let x = S::load(remainder.as_ptr());
        accum = simd_op(accum, x);
    }

    accum
}

/// Trait for evaluating a unary function on a SIMD vector.
trait SimdUnaryOp {
    /// Evaluate the unary function on the elements in `x`.
    unsafe fn eval<S: SimdFloat>(&self, x: S) -> S;
}

/// Apply a vectorized unary function to elements of `xs`, and write the
/// results to `out`.
///
/// This function will dispatch to the best SIMD implementation for the current
/// platform.
#[allow(unused_imports)]
#[allow(unreachable_code)] // Ignore fallback, if unused
fn dispatch_unary_op<Op: SimdUnaryOp>(xs: &[f32], out: &mut [MaybeUninit<f32>], op: Op) {
    assert!(xs.len() == out.len());

    #[cfg(feature = "avx512")]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn vec_unary_op_avx512<Op: SimdUnaryOp>(
        xs: PtrLen<f32>,
        out: MutPtrLen<MaybeUninit<f32>>,
        op: Op,
    ) {
        use std::arch::x86_64::__m512;
        vec_unary_op(
            xs,
            out,
            #[inline(always)]
            |x: __m512| op.eval(x),
            0., /* pad */
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn vec_unary_op_avx<Op: SimdUnaryOp>(
        xs: PtrLen<f32>,
        out: MutPtrLen<MaybeUninit<f32>>,
        op: Op,
    ) {
        use std::arch::x86_64::__m256;
        vec_unary_op(
            xs,
            out,
            #[inline(always)]
            |x: __m256| op.eval(x),
            0., /* pad */
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if crate::is_avx512_supported() {
            unsafe {
                vec_unary_op_avx512(xs.into(), out.into(), op);
            }
            return;
        }

        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            // Safety: We've checked that AVX2 + FMA are available.
            unsafe {
                vec_unary_op_avx(xs.into(), out.into(), op);
            }
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use crate::simd_vec::wasm::v128f;

        // Safety: The WASM runtime will have verified SIMD instructions
        // are accepted when loading the binary.
        unsafe {
            vec_unary_op(
                xs.into(),
                out.into(),
                #[inline(always)]
                |x: v128f| op.eval(x),
                0., /* pad */
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::float32x4_t;

        unsafe {
            vec_unary_op(
                xs.into(),
                out.into(),
                #[inline(always)]
                |x: float32x4_t| op.eval(x),
                0., /* pad */
            );
        }
        return;
    }

    unsafe {
        vec_unary_op(
            xs.into(),
            out.into(),
            #[inline(always)]
            |x: f32| op.eval(x),
            0., /* pad */
        );
    }
}

/// Apply a vectorized unary function in-place to elements of `xs`.
#[allow(unused_imports)]
#[allow(unreachable_code)] // Ignore fallback, if unused
fn dispatch_unary_op_in_place<Op: SimdUnaryOp>(xs: &mut [f32], op: Op) {
    let out: MutPtrLen<f32> = xs.into();

    #[cfg(feature = "avx512")]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn vec_unary_op_avx512<Op: SimdUnaryOp>(
        xs: PtrLen<f32>,
        out: MutPtrLen<MaybeUninit<f32>>,
        op: Op,
    ) {
        use std::arch::x86_64::__m512;
        vec_unary_op(
            xs,
            out,
            #[inline(always)]
            |x: __m512| op.eval(x),
            0., /* pad */
        );
    }

    // Non-generic wrapper for `vec_unary_op` which instantiates the
    // AVX + FMA version.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn vec_unary_op_avx<Op: SimdUnaryOp>(
        xs: PtrLen<f32>,
        out: MutPtrLen<MaybeUninit<f32>>,
        op: Op,
    ) {
        use std::arch::x86_64::__m256;
        vec_unary_op(
            xs,
            out,
            #[inline(always)]
            |x: __m256| op.eval(x),
            0., /* pad */
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if crate::is_avx512_supported() {
            unsafe {
                vec_unary_op_avx512(xs.into(), out.as_uninit(), op);
            }
            return;
        }

        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            // Safety: We've checked that AVX2 + FMA are available.
            unsafe {
                vec_unary_op_avx(xs.into(), out.as_uninit(), op);
            }
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use crate::simd_vec::wasm::v128f;

        // Safety: The WASM runtime will have verified SIMD instructions
        // are accepted when loading the binary.
        unsafe {
            vec_unary_op(
                xs.into(),
                out.as_uninit(),
                #[inline(always)]
                |x: v128f| op.eval(x),
                0., /* pad */
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::float32x4_t;

        unsafe {
            vec_unary_op(
                xs.into(),
                out.as_uninit(),
                #[inline(always)]
                |x: float32x4_t| op.eval(x),
                0., /* pad */
            );
        }
        return;
    }

    unsafe {
        vec_unary_op(
            xs.into(),
            out.as_uninit(),
            #[inline(always)]
            |x: f32| op.eval(x),
            0., /* pad */
        );
    }
}

/// Trait for evaluating a SIMD reduction or normalization operator.
trait SimdOp {
    /// Evaluate the operator on `input` and write the results to `out`.
    unsafe fn eval<S: SimdFloat>(&self, input: PtrLen<f32>, out: MutPtrLen<MaybeUninit<f32>>);
}

/// Apply a vectorized normalization or reduction function to `input`, writing
/// the results to`out`.
///
/// This function will dispatch to the best SIMD implementation for the current
/// platform.
#[allow(unused_imports)]
#[allow(unreachable_code)] // Ignore fallback, if unused
fn dispatch_simd_op<Op: SimdOp>(input: PtrLen<f32>, out: MutPtrLen<MaybeUninit<f32>>, op: Op) {
    #[cfg(feature = "avx512")]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn simd_op_avx512<Op: SimdOp>(
        xs: PtrLen<f32>,
        out: MutPtrLen<MaybeUninit<f32>>,
        op: Op,
    ) {
        use std::arch::x86_64::__m512;
        op.eval::<__m512>(xs, out);
    }

    // Non-generic wrapper for `$func` which instantiates the AVX + FMA version.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn simd_op_avx<Op: SimdOp>(xs: PtrLen<f32>, out: MutPtrLen<MaybeUninit<f32>>, op: Op) {
        use std::arch::x86_64::__m256;
        op.eval::<__m256>(xs, out);
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if crate::is_avx512_supported() {
            unsafe { simd_op_avx512(input, out, op) };
            return;
        }

        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            // Safety: We've checked that AVX2 + FMA are available.
            unsafe { simd_op_avx(input, out, op) };
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use crate::simd_vec::wasm::v128f;

        // Safety: The WASM runtime will have verified SIMD instructions
        // are accepted when loading the binary.
        unsafe { op.eval::<v128f>(input, out) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::float32x4_t;

        unsafe { op.eval::<float32x4_t>(input, out) };
        return;
    }

    // Generic fallback.
    unsafe { op.eval::<f32>(input, out) };
}
