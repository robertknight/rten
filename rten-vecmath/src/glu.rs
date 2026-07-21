use rten_simd::ops::{BitOps, NumOps};
use rten_simd::span::SrcDest;
use rten_simd::{Isa, SimdOp, SimdUnaryOp};

/// Computes a GLU (Gated Linear Unit) function over two slices.
///
/// This evaluates `out[i] = act(a[i]) * b[i]` where `a` and `b` are slices of
/// equal length and `act` is a vectorized activation such as [`Silu`](crate::Silu)
/// or [`Gelu`](crate::Gelu). Compared to applying the activation and
/// multiplication as separate passes, this performs the multiplication while
/// the activated value is still in registers.
pub struct Glu<'src, 'dst, Act: SimdUnaryOp<f32>> {
    act: Act,
    a_dest: SrcDest<'src, 'dst, f32>,
    b: &'src [f32],
}

impl<'src, 'dst, Act: SimdUnaryOp<f32>> Glu<'src, 'dst, Act> {
    /// Create a GLU operation which computes `act(a) * b`.
    ///
    /// `a_dest` is either an `(input, uninit_output)` pair of equal-length
    /// slices, or a single mutable slice to apply the operation in-place to
    /// `a`'s buffer. `b` must have the same length as `a`.
    pub fn new(act: Act, a_dest: impl Into<SrcDest<'src, 'dst, f32>>, b: &'src [f32]) -> Self {
        let a_dest = a_dest.into();
        assert!(
            a_dest.len() == b.len(),
            "a len {} != b len {}",
            a_dest.len(),
            b.len(),
        );
        Self { act, a_dest, b }
    }
}

impl<'dst, Act: SimdUnaryOp<f32>> SimdOp for Glu<'_, 'dst, Act> {
    type Output = &'dst mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(mut self, isa: I) -> Self::Output {
        let ops = isa.f32();
        let (mut a_ptr, mut out_ptr, mut n) = self.a_dest.src_dest_ptr();
        let mut b_ptr = self.b.as_ptr();

        let v_len = ops.len();
        while n >= v_len {
            // Safety: `a_ptr` and `b_ptr` point to buffers with at least
            // `v_len` elements.
            let a = unsafe { ops.load_ptr(a_ptr) };
            let b = unsafe { ops.load_ptr(b_ptr) };
            let y = ops.mul(self.act.eval(isa, a), b);

            // Safety: `out_ptr` points to a buffer with at least `v_len`
            // elements.
            unsafe { ops.store_ptr(y, out_ptr as *mut f32) };

            // Safety: `a_ptr`, `b_ptr` and `out_ptr` are pointers into buffers
            // of the same length, with at least `v_len` elements.
            n -= v_len;
            unsafe {
                a_ptr = a_ptr.add(v_len);
                b_ptr = b_ptr.add(v_len);
                out_ptr = out_ptr.add(v_len);
            }
        }

        if n > 0 {
            let mask = ops.first_n_mask(n);

            // Safety: Mask bit `i` is only set if `a_ptr.add(i)`, `b_ptr.add(i)`
            // and `out_ptr.add(i)` are valid.
            let a = unsafe { ops.load_ptr_mask(a_ptr, mask) };
            let b = unsafe { ops.load_ptr_mask(b_ptr, mask) };
            let y = ops.mul(self.act.eval(isa, a), b);
            unsafe {
                ops.store_ptr_mask(y, out_ptr as *mut f32, mask);
            }
        }

        // Safety: All output elements have been initialized.
        unsafe { self.a_dest.dest_assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use rten_simd::{SimdOp, SimdUnaryOp};

    use super::Glu;
    use crate::Silu;

    // Test length chosen to exercise main and tail loops for all ISAs.
    const TEST_LEN: usize = 18;

    fn test_input() -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.5 - 4.).collect();
        let b: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.25 - 2.).collect();
        (a, b)
    }

    // Expected result computed using the same activation kernel as a separate
    // pass. Results should be identical since the fused op evaluates the
    // activation in the same way.
    fn reference_glu(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut act = a.to_vec();
        Silu {}.map_mut(&mut act);
        act.iter().zip(b).map(|(x, y)| x * y).collect()
    }

    #[test]
    fn test_glu() {
        let (a, b) = test_input();
        let expected = reference_glu(&a, &b);

        let mut out = vec![MaybeUninit::uninit(); TEST_LEN];
        let result = Glu::new(Silu {}, (a.as_slice(), out.as_mut_slice()), &b).dispatch();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_glu_in_place() {
        let (mut a, b) = test_input();
        let expected = reference_glu(&a, &b);

        let result = Glu::new(Silu {}, a.as_mut_slice(), &b).dispatch();

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "a len 18 != b len 4")]
    fn test_glu_length_mismatch() {
        let (a, _) = test_input();
        let b = vec![0.; 4];
        let mut out = vec![MaybeUninit::uninit(); TEST_LEN];
        Glu::new(Silu {}, (a.as_slice(), out.as_mut_slice()), &b).dispatch();
    }
}
