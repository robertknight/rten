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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
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

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
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

/// Invoke the best available implementation of a unary operator on the current
/// platform.
///
/// After this macro executes, the contents of `out` will have been initialized.
///
/// This generates a call to [vec_unary_op] for each of the supported
/// instruction sets, passing in a version of `$op_func` that is parametrized by
/// the corresponding SIMD vector type. At runtime the appropriate
/// `vec_unary_op` call will be invoked.
macro_rules! dispatch_unary_op {
    ($in:ident, $out:ident, $op_func:ident, $fallback_func:ident) => {
        assert!($in.len() == $out.len());

        #[allow(unused_imports)]
        #[allow(unreachable_code)] // Ignore fallback, if unused
        {
            use std::mem::MaybeUninit;

            use crate::{vec_unary_op, MutPtrLen, PtrLen};

            // Non-generic wrapper for `vec_unary_op` which instantiates the
            // AVX + FMA version.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            #[target_feature(enable = "fma")]
            unsafe fn vec_unary_op_avx(xs: PtrLen<f32>, out: MutPtrLen<MaybeUninit<f32>>) {
                use std::arch::x86_64::__m256;
                vec_unary_op(xs, out, |x: __m256| $op_func(x), 0. /* pad */);
            }

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                // Safety: We've checked that AVX2 + FMA are available.
                unsafe {
                    vec_unary_op_avx($in.into(), $out.into());
                }

                return;
            }

            #[cfg(target_arch = "wasm32")]
            {
                use crate::simd_vec::wasm::v128f;

                // Safety: The WASM runtime will have verified SIMD instructions
                // are accepted when loading the binary.
                unsafe {
                    vec_unary_op(
                        $in.into(),
                        $out.into(),
                        |x: v128f| $op_func(x),
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
                        $in.into(),
                        $out.into(),
                        |x: float32x4_t| $op_func(x),
                        0., /* pad */
                    );
                }
                return;
            }

            // Generic fallback.
            for (x, y) in $in.iter().zip($out.iter_mut()) {
                y.write($fallback_func(*x));
            }
        }
    };

    ($out:ident, $op_func:ident, $fallback_func:ident) => {
        #[allow(unused_imports)]
        use crate::{vec_unary_op, MutPtrLen, PtrLen};

        #[allow(unreachable_code)] // Ignore fallback, if unused
        {
            // Non-generic wrapper for `vec_unary_op` which instantiates the
            // AVX + FMA version.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            #[target_feature(enable = "fma")]
            unsafe fn vec_unary_op_avx(xs: PtrLen<f32>, out: MutPtrLen<f32>) {
                use std::arch::x86_64::__m256;
                vec_unary_op(
                    xs,
                    out.as_uninit(),
                    |x: __m256| $op_func(x),
                    0., /* pad */
                );
            }

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                // Safety: We've checked that AVX2 + FMA are available.
                unsafe {
                    vec_unary_op_avx($out.into(), $out.into());
                }
                return;
            }

            #[cfg(target_arch = "wasm32")]
            {
                use crate::simd_vec::wasm::v128f;

                // Safety: The WASM runtime will have verified SIMD instructions
                // are accepted when loading the binary.
                let out: MutPtrLen<f32> = $out.into();
                unsafe {
                    vec_unary_op(
                        $out.into(),
                        out.as_uninit(),
                        |x: v128f| $op_func(x),
                        0., /* pad */
                    );
                }
                return;
            }

            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::float32x4_t;

                let out: MutPtrLen<f32> = $out.into();
                unsafe {
                    vec_unary_op(
                        $out.into(),
                        out.as_uninit(),
                        |x: float32x4_t| $op_func(x),
                        0., /* pad */
                    );
                }
                return;
            }

            // Generic fallback.
            for x in $out.iter_mut() {
                *x = $fallback_func(*x);
            }
        }
    };
}

pub(crate) use dispatch_unary_op;

/// Dispatch a SIMD function using the best available `SimdFloat` implementation
/// on the current system.
///
/// `$func` should be a function with a generic argument `S: SimdFloat`. `$in`
/// and `$out` are the function arguments.
macro_rules! dispatch_simd {
    ($func:ident, $in:expr, $out:expr) => {
        #[allow(unused_imports)]
        #[allow(unreachable_code)] // Ignore fallback, if unused
        {
            use crate::{MutPtrLen, PtrLen};

            // Non-generic wrapper for `$func` which instantiates the AVX + FMA version.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            #[target_feature(enable = "fma")]
            unsafe fn simd_op_avx(xs: PtrLen<f32>, out: MutPtrLen<MaybeUninit<f32>>) {
                use std::arch::x86_64::__m256;
                $func::<__m256>(xs, out);
            }

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                // Safety: We've checked that AVX2 + FMA are available.
                unsafe { simd_op_avx($in, $out) };
                return;
            }

            #[cfg(target_arch = "wasm32")]
            {
                use crate::simd_vec::wasm::v128f;

                // Safety: The WASM runtime will have verified SIMD instructions
                // are accepted when loading the binary.
                unsafe { $func::<v128f>($in, $out) };
                return;
            }

            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::float32x4_t;

                unsafe { $func::<float32x4_t>($in, $out) };
                return;
            }

            // Generic fallback.
            unsafe { $func::<f32>($in, $out) };
        }
    };
}

pub(crate) use dispatch_simd;
