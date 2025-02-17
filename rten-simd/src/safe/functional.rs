//! Vectorized higher-order operations (map, fold etc.)
use super::{MakeSimd, Simd};
use crate::span::SrcDest;

/// Transform a slice by applying a vectorized map function to its elements.
///
/// If the slice is not a multiple of the vector length, the final call to
/// `op` will use a vector padded with zeros.
///
/// The map function must have the same input and output type.
pub fn simd_map<S: Simd, Op: FnMut(S) -> S>(
    init: impl MakeSimd<S>,
    mut src_dest: SrcDest<S::Elem>,
    mut op: Op,
) -> &mut [S::Elem] {
    let (mut in_ptr, mut out_ptr, mut n) = src_dest.src_dest_ptr();

    let v_len = init.len();
    while n >= v_len {
        let x = unsafe { init.load_ptr(in_ptr) };
        let y = op(x);
        unsafe { y.store_ptr(out_ptr as *mut S::Elem) };

        n -= v_len;
        unsafe {
            in_ptr = in_ptr.add(v_len);
            out_ptr = out_ptr.add(v_len);
        }
    }

    if n > 0 {
        let mask = init.first_n_mask(n);
        let x = unsafe { init.load_ptr_mask(in_ptr, mask) };
        let y = op(x);
        unsafe {
            y.store_ptr_mask(out_ptr as *mut S::Elem, mask);
        }
    }

    unsafe { src_dest.dest_assume_init() }
}

/// Reduce a slice to a single SIMD vector by applying a vectorized fold
/// function to its elements.
///
/// If the slice is not a multiple of the vector length, the final call to
/// `op` will use a vector padded with zeros and only the lanes of the accumulator
/// corresponding to used entries will be updated with the result.
///
/// The accumulator must have the same type as the elements.
pub fn simd_fold<S: Simd, Op: Fn(S, S) -> S>(
    init: impl MakeSimd<S>,
    xs: &[S::Elem],
    mut accum: S,
    simd_op: Op,
) -> S {
    let mut n = xs.len();
    let mut x_ptr = xs.as_ptr();
    let v_len = init.len();

    while n >= v_len {
        let x = unsafe { init.load_ptr(x_ptr) };
        accum = simd_op(accum, x);
        n -= v_len;
        x_ptr = unsafe { x_ptr.add(v_len) };
    }

    let mask = init.first_n_mask(n);
    if n > 0 {
        let x = unsafe { init.load_ptr_mask(x_ptr, mask) };
        let prev_accum = accum;
        let new_accum = simd_op(accum, x);
        accum = new_accum.select(prev_accum, mask);
    }

    accum
}

#[cfg(test)]
mod tests {
    use crate::safe::{Isa, MakeSimd, Simd, SimdOp};

    use super::{simd_fold, simd_map};

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
                simd_map(isa.f32(), self.xs.into(), |x| x * x)
            }
        }

        let mut buf: Vec<_> = (0..TEST_LEN).map(|x| x as f32).collect();
        let expected: Vec<_> = buf.iter().map(|x| *x * *x).collect();

        let squared = Square { xs: &mut buf }.dispatch();

        assert_eq!(squared, &expected);
    }

    #[test]
    fn test_simd_fold() {
        struct Sum<'a> {
            xs: &'a [f32],
        }

        impl<'a> SimdOp for Sum<'a> {
            type Output = f32;

            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                let vec_sum = simd_fold(isa.f32(), self.xs, isa.f32().zero(), |sum, x| sum + x);
                vec_sum.to_array().into_iter().fold(0., |sum, x| sum + x)
            }
        }

        let buf: Vec<_> = (0..TEST_LEN).map(|x| x as f32).collect();
        let expected = (buf.len() as f32 * buf[buf.len() - 1]) / 2.;

        let sum = Sum { xs: &buf }.dispatch();
        assert_eq!(sum, expected);
    }
}
