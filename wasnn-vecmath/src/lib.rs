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

mod erf;
mod exp;
mod simd_vec;
mod ulp;

#[cfg(test)]
mod testing;

pub use erf::{erf, vec_erf, vec_erf_in_place};
pub use exp::{exp, sigmoid, vec_exp, vec_exp_in_place, vec_sigmoid, vec_sigmoid_in_place};
use simd_vec::SimdFloat;

/// Const pointer to a range of `T`s.
///
/// This is like an `&[T]`, but without the guarantee that no mutable aliases
/// exist. This is useful as it enables re-using the same unsafe code for
/// mutating and non-mutating variants of a function.
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

/// Mutable pointer to a range of `T`s.
///
/// This is like an `&mut [T]`, but without the guarantee that no aliases exist.
struct MutPtrLen<T> {
    ptr: *mut T,
    len: usize,
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
/// The operation is applied to SIMD vector-sized groups of elements at
/// a time using `simd_op`. If the final group has a size that is smaller than
/// the SIMD vector width, `simd_op` will be called with a SIMD vector that
/// is padded with `pad` on the right.
///
/// Safety: The caller must ensure that `xs` and `out` are valid pointers
/// to buffers of the expected lengths.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn vec_unary_op<S: SimdFloat, Op: Fn(S) -> S>(
    xs: PtrLen<f32>,
    out: MutPtrLen<f32>,
    simd_op: Op,
    pad: f32,
) {
    assert!(xs.len == out.len);

    let mut n = xs.len;
    let mut x_ptr = xs.ptr;
    let mut out_ptr = out.ptr;

    // We use an allocation because const generics aren't allowed as sizes
    // for static arrays.
    //
    // TODO - Rework this to avoid allocation.
    let mut remainder = vec![pad; S::LEN];

    // Main loop over full vectors.
    while n >= S::LEN {
        let x = S::load(x_ptr);
        let y = simd_op(x);
        y.store(out_ptr);

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
            *out_ptr.add(i) = remainder[i];
        }
    }
}

/// Invoke the best available implementation of a unary operator on the current
/// platform.
///
/// This generates a call to [vec_unary_op] for each of the supported
/// instruction sets, passing in a version of `$op_func` that is parametrized by
/// the corresponding SIMD vector type. At runtime the appropriate
/// `vec_unary_op` call will be invoked.
macro_rules! dispatch_unary_op {
    ($in:ident, $out:ident, $op_func:ident, $fallback_func:ident) => {
        use crate::vec_unary_op;

        assert!($in.len() == $out.len());

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::__m256;

            // Safety: We've checked that AVX2 + FMA are available.
            unsafe {
                vec_unary_op(
                    $in.into(),
                    $out.into(),
                    |x: __m256| $op_func(x),
                    0., /* pad */
                );
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

        // Fallback for platforms where optimized implementation is used
        // conditionally.
        #[cfg(target_arch = "x86_64")]
        for (x, y) in $in.iter().zip($out.iter_mut()) {
            *y = $fallback_func(*x);
        }
    };

    ($out:ident, $op_func:ident, $fallback_func:ident) => {
        use crate::vec_unary_op;

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::__m256;

            // Safety: We've checked that AVX2 + FMA are available.
            unsafe {
                vec_unary_op(
                    $out.into(),
                    $out.into(),
                    |x: __m256| $op_func(x),
                    0., /* pad */
                );
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
                    $out.into(),
                    $out.into(),
                    |x: v128f| $op_func(x),
                    0., /* pad */
                );
            }
            return;
        }

        // Fallback for platforms where optimized implementation is used
        // conditionally.
        #[cfg(target_arch = "x86_64")]
        for x in $out.iter_mut() {
            *x = $fallback_func(*x);
        }
    };
}

pub(crate) use dispatch_unary_op;
