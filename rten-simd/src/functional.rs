//! Higher order functions (map, fold etc.) that use vectorized operations.

use crate::span::SrcDest;
use crate::{Simd, SimdMask};

/// Apply a unary operation to each element in `input` and store the results
/// in `output`.
///
/// When evaluated, all elements in `output` will be initialized.
///
/// The operation is applied to SIMD vector-sized groups of elements at a
/// time using `op`. If the final group has a size that is smaller than the
/// SIMD vector width, `op` will be called with a SIMD vector that is
/// padded.
///
/// # Safety
///
/// The caller must ensure that `S` is a supported SIMD vector type on the
/// current system.
#[inline(always)]
pub unsafe fn simd_map<'dst, S: Simd, Op: FnMut(S) -> S>(
    mut src_dest: SrcDest<'_, 'dst, S::Elem>,
    mut op: Op,
) -> &'dst mut [S::Elem] {
    let (mut in_ptr, mut out_ptr, mut n) = src_dest.src_dest_ptr();

    let v_len = S::len();
    while n >= v_len {
        let x = S::load(in_ptr);
        let y = op(x);
        y.store(out_ptr as *mut S::Elem);

        n -= v_len;
        in_ptr = in_ptr.add(v_len);
        out_ptr = out_ptr.add(v_len);
    }

    if n > 0 {
        let x = S::load_partial(in_ptr, n);
        let y = op(x);
        y.store_partial(out_ptr as *mut S::Elem, n);
    }

    src_dest.dest_assume_init()
}

/// Apply a vectorized fold operation over `xs`. If the length of `xs` is not a
/// multiple of the vector length then the accumulator is left unchanged for
/// unused lanes in the final update.
///
/// # Safety
///
/// The caller must ensure that `S` is a supported SIMD vector type on the
/// current system.
#[inline(always)]
pub unsafe fn simd_fold<S: Simd, Op: Fn(S, S) -> S>(
    xs: &[S::Elem],
    mut accum: S,
    simd_op: Op,
) -> S {
    let mut n = xs.len();
    let mut x_ptr = xs.as_ptr();
    let v_len = S::len();

    while n >= v_len {
        let x = S::load(x_ptr);
        accum = simd_op(accum, x);
        n -= v_len;
        x_ptr = x_ptr.add(v_len);
    }

    let n_mask = S::Mask::first_n(n);
    if n > 0 {
        let x = S::load_partial(x_ptr, n);
        let prev_accum = accum;
        let new_accum = simd_op(accum, x);
        accum = new_accum.select(prev_accum, n_mask);
    }

    accum
}

/// A variant of [`simd_fold`] where the accumulator is an array of values
/// instead of just one.
///
/// # Safety
///
/// The caller must ensure that `S` is a supported SIMD vector type on the
/// current system.
#[inline(always)]
pub unsafe fn simd_fold_array<S: Simd, const N: usize, Op: Fn([S; N], S) -> [S; N]>(
    xs: &[S::Elem],
    mut accum: [S; N],
    simd_op: Op,
) -> [S; N] {
    let mut n = xs.len();
    let mut x_ptr = xs.as_ptr();
    let v_len = S::len();

    while n >= v_len {
        let x = S::load(x_ptr);
        accum = simd_op(accum, x);
        n -= v_len;
        x_ptr = x_ptr.add(v_len);
    }

    let n_mask = S::Mask::first_n(n);
    if n > 0 {
        let x = S::load_partial(x_ptr, n);
        let prev_accum = accum;
        let new_accum = simd_op(accum, x);

        for i in 0..N {
            accum[i] = new_accum[i].select(prev_accum[i], n_mask);
        }
    }

    accum
}
