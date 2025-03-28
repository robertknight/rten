//! Vectorized higher-order operations (map etc.)

use crate::ops::NumOps;
use crate::span::SrcDest;
use crate::Elem;

/// Transform a slice by applying a vectorized map function to its elements.
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

#[cfg(test)]
mod tests {
    use crate::ops::NumOps;
    use crate::{Isa, SimdOp};

    use super::simd_map;

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
}
