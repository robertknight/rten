//! Vectorized higher-order operations (map etc.)

use crate::ops::NumOps;
use crate::span::SrcDest;
use crate::Elem;

/// Transform a slice by applying a vectorized map function to its elements.
///
/// This function can be applied both in-place (mutable destination) and
/// with separate source/destination buffers.
///
/// If the slice is not a multiple of the vector length, the final call to
/// `op` will use a vector padded with zeros.
///
/// The map function must have the same input and output type.
#[inline(always)]
pub fn simd_map<'src, 'dst, T: Elem + 'static, O: NumOps<T>, Op: FnMut(O::Simd) -> O::Simd>(
    ops: O,
    src_dest: impl Into<SrcDest<'src, 'dst, T>>,
    mut op: Op,
) -> &'dst mut [T] {
    let mut src_dest = src_dest.into();
    let (mut in_ptr, mut out_ptr, mut n) = src_dest.src_dest_ptr();

    let v_len = ops.len();
    while n >= v_len {
        // Safety: `in_ptr` and `out_ptr` point to a buffer with at least `v_len`
        // elements.
        let x = unsafe { ops.load_ptr(in_ptr) };
        let y = op(x);
        unsafe { ops.store_ptr(y, out_ptr as *mut T) };

        // Safety: `in_ptr` and `out_ptr` are pointers into buffers of the same
        // length, with at least `v_len` elements.
        n -= v_len;
        unsafe {
            in_ptr = in_ptr.add(v_len);
            out_ptr = out_ptr.add(v_len);
        }
    }

    if n > 0 {
        let mask = ops.first_n_mask(n);

        // Safety: Mask bit `i` is only set if `in_ptr.add(i)` and
        // `out_ptr.add(i)` are valid.
        let x = unsafe { ops.load_ptr_mask(in_ptr, mask) };
        let y = op(x);
        unsafe {
            ops.store_ptr_mask(y, out_ptr as *mut T, mask);
        }
    }

    // Safety: All elements in `src_dest` have been initialized.
    unsafe { src_dest.dest_assume_init() }
}

/// Transform a slice in-place by applying a vectorized map function to its
/// elements.
///
/// If the slice is not a multiple of the vector length, the final call to
/// `op` will use a vector padded with zeros.
///
/// `UNROLL` specifies a loop unrolling factor. When the operation is very
/// cheap, explicit unrolling can improve instruction level parallelism.
#[inline(always)]
pub fn simd_apply<
    T: Elem + 'static,
    O: NumOps<T>,
    Op: FnMut(O::Simd) -> O::Simd,
    const UNROLL: usize,
>(
    ops: O,
    dest: &mut [T],
    mut op: Op,
) -> &mut [T] {
    let v_len = ops.len();
    let mut chunks = dest.chunks_exact_mut(v_len * UNROLL);
    for chunk in &mut chunks {
        for i in 0..UNROLL {
            // Safety: Sliced chunk points to `v_len` elements.
            let x = unsafe { ops.load_ptr(chunk.as_ptr().add(i * v_len)) };
            let y = op(x);
            unsafe {
                ops.store_ptr(y, chunk.as_mut_ptr().add(i * v_len));
            }
        }
    }

    let mut tail_chunks = chunks.into_remainder().chunks_exact_mut(v_len);
    for chunk in &mut tail_chunks {
        let x = ops.load(chunk);
        let y = op(x);
        ops.store(y, chunk);
    }

    let tail = tail_chunks.into_remainder();
    if !tail.is_empty() {
        let mask = ops.first_n_mask(tail.len());

        // Safety: `mask[i]` is true where `tail.add(i)` is valid.
        let x = unsafe { ops.load_ptr_mask(tail.as_ptr(), mask) };
        let y = op(x);
        unsafe {
            ops.store_ptr_mask(y, tail.as_mut_ptr(), mask);
        }
    }

    dest
}

#[cfg(test)]
mod tests {
    use crate::ops::NumOps;
    use crate::{Isa, SimdOp};

    use super::{simd_apply, simd_map};

    // f32 vector length, chosen to exercise main and tail loops for all ISAs.
    const TEST_LEN: usize = 18;

    #[test]
    fn test_simd_map() {
        struct Square<'a> {
            xs: &'a mut [f32],
        }

        impl<'a> SimdOp for Square<'a> {
            type Output = &'a mut [f32];

            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                let ops = isa.f32();
                simd_map(ops, self.xs, |x| ops.mul(x, x))
            }
        }

        let mut buf: Vec<_> = (0..TEST_LEN).map(|x| x as f32).collect();
        let expected: Vec<_> = buf.iter().map(|x| *x * *x).collect();

        let squared = Square { xs: &mut buf }.dispatch();

        assert_eq!(squared, &expected);
    }

    #[test]
    fn test_simd_apply() {
        struct Square<'a> {
            xs: &'a mut [f32],
        }

        impl<'a> SimdOp for Square<'a> {
            type Output = &'a mut [f32];

            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                let ops = isa.f32();
                const UNROLL: usize = 2;
                simd_apply::<_, _, _, UNROLL>(ops, self.xs, |x| ops.mul(x, x))
            }
        }

        // Extend `TEST_LEN` to test the unrolled loops in `simd_apply`.
        let test_len = TEST_LEN * 4;
        let mut buf: Vec<_> = (0..test_len).map(|x| x as f32).collect();
        let expected: Vec<_> = buf.iter().map(|x| *x * *x).collect();

        let squared = Square { xs: &mut buf }.dispatch();

        assert_eq!(squared, &expected);
    }
}
