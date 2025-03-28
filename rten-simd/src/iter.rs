//! Tools for vectorized iteration over slices.

use crate::ops::NumOps;
use crate::{Elem, Simd};

/// Methods for creating vectorized iterators.
pub trait SimdIterable {
    /// Element type in the slice.
    type Elem: Elem;

    /// Iterate over SIMD-sized chunks of the input.
    ///
    /// If the input length is not divisble by the SIMD vector width, the
    /// iterator yields only the full chunks. The tail is accessible via the
    /// iterator's [`tail`](Iter::tail) method.
    fn simd_iter<O: NumOps<Self::Elem>>(&self, ops: O) -> Iter<Self::Elem, O>;

    /// Iterate over SIMD-sized chunks of the input.
    ///
    /// If the input length is not divisble by the SIMD vector width, the final
    /// chunk will be padded with zeros.
    fn simd_iter_pad<O: NumOps<Self::Elem>>(
        &self,
        ops: O,
    ) -> impl ExactSizeIterator<Item = O::Simd>;
}

impl<T: Elem> SimdIterable for [T] {
    type Elem = T;

    #[inline]
    fn simd_iter<O: NumOps<T>>(&self, ops: O) -> Iter<T, O> {
        Iter::new(ops, self)
    }

    #[inline]
    fn simd_iter_pad<O: NumOps<T>>(&self, ops: O) -> impl ExactSizeIterator<Item = O::Simd> {
        IterPad::new(ops, self)
    }
}

/// Iterator which yields chunks of a slice as a SIMD vector.
///
/// This type is created by [`SimdIterable::simd_iter`].
pub struct Iter<'a, T: Elem, O: NumOps<T>> {
    ops: O,
    xs: &'a [T],
    n_full_chunks: usize,
}

impl<'a, T: Elem, O: NumOps<T>> Iter<'a, T, O> {
    #[inline]
    fn new(ops: O, xs: &'a [T]) -> Self {
        let n_full_chunks = xs.len() / ops.len();
        Iter {
            ops,
            xs,
            n_full_chunks,
        }
    }

    /// Reduce an iterator to a single SIMD vector.
    ///
    /// This is like [`Iterator::fold`] but the `fold` function receives SIMD
    /// vectors instead of single elements. If the iterator length is not a
    /// multiple of the SIMD vector length, the final vector will be padded with
    /// zeros.
    #[inline]
    pub fn fold<F: FnMut(O::Simd, O::Simd) -> O::Simd>(
        mut self,
        mut accum: O::Simd,
        mut fold: F,
    ) -> O::Simd {
        for chunk in &mut self {
            accum = fold(accum, chunk);
        }

        if let Some((tail, mask)) = self.tail() {
            let new_accum = fold(accum, tail);
            accum = self.ops.select(new_accum, accum, mask);
        }

        accum
    }

    /// Variant of [`fold`](Self::fold) that computes multiple accumulator
    /// values in a single pass.
    #[inline]
    pub fn fold_n<const N: usize>(
        mut self,
        mut accum: [O::Simd; N],
        mut fold: impl FnMut([O::Simd; N], O::Simd) -> [O::Simd; N],
    ) -> [O::Simd; N] {
        for chunk in &mut self {
            accum = fold(accum, chunk);
        }

        if let Some((tail, mask)) = self.tail() {
            let new_accum = fold(accum, tail);
            for i in 0..N {
                accum[i] = self.ops.select(new_accum[i], accum[i], mask);
            }
        }

        accum
    }

    /// Return a SIMD vector and mask for the left-over elements in the
    /// slice after iterating over all full SIMD chunks.
    ///
    /// Elements of the SIMD vector that correspond to positions where the mask
    /// is false will be set to zero.
    #[inline]
    pub fn tail(&self) -> Option<(O::Simd, <O::Simd as Simd>::Mask)> {
        let n = self.xs.len();
        if n > 0 {
            Some(self.ops.load_pad(self.xs))
        } else {
            None
        }
    }
}

impl<T: Elem, O: NumOps<T>> Iterator for Iter<'_, T, O> {
    type Item = O::Simd;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let v_len = self.ops.len();
        if let Some((chunk, tail)) = self.xs.split_at_checked(v_len) {
            self.xs = tail;

            // Safety: `chunk.as_ptr()` points to `v_len` elements.
            let x = unsafe { self.ops.load_ptr(chunk.as_ptr()) };

            Some(x)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.n_full_chunks, Some(self.n_full_chunks))
    }
}

impl<T: Elem, O: NumOps<T>> ExactSizeIterator for Iter<'_, T, O> {}

impl<T: Elem, O: NumOps<T>> std::iter::FusedIterator for Iter<'_, T, O> {}

/// Iterator which yields chunks of a slice as a SIMD vector.
///
/// This type is created by [`SimdIterable::simd_iter_pad`].
pub struct IterPad<'a, T: Elem, O: NumOps<T>> {
    iter: Iter<'a, T, O>,
    has_tail: bool,
}

impl<'a, T: Elem, O: NumOps<T>> IterPad<'a, T, O> {
    #[inline]
    fn new(ops: O, xs: &'a [T]) -> Self {
        let iter = Iter::new(ops, xs);
        let has_tail = xs.len() % ops.len() != 0;
        Self { iter, has_tail }
    }
}

impl<T: Elem, O: NumOps<T>> Iterator for IterPad<'_, T, O> {
    type Item = O::Simd;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(chunk) = self.iter.next() {
            Some(chunk)
        } else if self.has_tail {
            let (tail, _mask) = self.iter.tail().unwrap();
            self.has_tail = false;
            Some(tail)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n_tail = if self.has_tail { 1 } else { 0 };
        let n_chunks = self.iter.len() + n_tail;
        (n_chunks, Some(n_chunks))
    }
}

impl<T: Elem, O: NumOps<T>> ExactSizeIterator for IterPad<'_, T, O> {}

impl<T: Elem, O: NumOps<T>> std::iter::FusedIterator for IterPad<'_, T, O> {}

#[cfg(test)]
mod tests {
    use super::SimdIterable;
    use crate::dispatch::test_simd_op;
    use crate::ops::NumOps;
    use crate::{Isa, Simd, SimdOp};

    // f32 vector length, chosen to exercise main and tail loops for all ISAs.
    const TEST_LEN: usize = 18;

    #[test]
    fn test_iter() {
        test_simd_op!(isa, {
            let buf: Vec<_> = (0..TEST_LEN).map(|x| x as f32).collect();
            let chunks = buf.chunks_exact(isa.f32().len());

            let iter = buf.simd_iter(isa.f32());
            assert_eq!(iter.len(), chunks.len());

            for (scalar_chunk, simd_chunk) in chunks.zip(iter) {
                assert_eq!(simd_chunk.to_array().as_ref(), scalar_chunk);
            }
        });
    }

    #[test]
    fn test_iter_pad() {
        test_simd_op!(isa, {
            let buf: Vec<_> = (0..TEST_LEN).map(|x| x as f32).collect();
            let chunks = buf.chunks(isa.f32().len());

            let iter = buf.simd_iter_pad(isa.f32());
            assert_eq!(iter.len(), chunks.len());

            for (scalar_chunk, simd_chunk) in chunks.zip(iter) {
                let simd_elts = simd_chunk.to_array();
                let simd_elts = simd_elts.as_ref();
                assert_eq!(&simd_elts[..scalar_chunk.len()], scalar_chunk);
                if simd_elts.len() > scalar_chunk.len() {
                    assert!(&simd_elts[scalar_chunk.len()..].iter().all(|x| *x == 0.));
                }
            }
        });
    }

    #[test]
    fn test_fold() {
        struct Sum<'a> {
            xs: &'a [f32],
        }

        impl<'a> SimdOp for Sum<'a> {
            type Output = f32;

            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                let ops = isa.f32();
                let vec_sum = self
                    .xs
                    .simd_iter(ops)
                    .fold(ops.zero(), |sum, x| ops.add(sum, x));
                vec_sum.to_array().into_iter().fold(0., |sum, x| sum + x)
            }
        }

        let buf: Vec<_> = (0..TEST_LEN).map(|x| x as f32).collect();
        let expected = (buf.len() as f32 * buf[buf.len() - 1]) / 2.;

        let sum = Sum { xs: &buf }.dispatch();
        assert_eq!(sum, expected);
    }

    #[test]
    fn test_fold_n() {
        struct MinMax<'a> {
            xs: &'a [f32],
        }

        impl<'a> SimdOp for MinMax<'a> {
            type Output = (f32, f32);

            fn eval<I: Isa>(self, isa: I) -> Self::Output {
                let ops = isa.f32();
                let [vec_min, vec_max] = self.xs.simd_iter(ops).fold_n(
                    [ops.splat(f32::MAX), ops.splat(f32::MIN)],
                    |[min, max], x| [ops.min(min, x), ops.max(max, x)],
                );
                let min = vec_min
                    .to_array()
                    .into_iter()
                    .reduce(|min, x| min.min(x))
                    .unwrap();
                let max = vec_max
                    .to_array()
                    .into_iter()
                    .reduce(|max, x| max.max(x))
                    .unwrap();
                (min, max)
            }
        }

        let buf: Vec<_> = (0..TEST_LEN).map(|x| x as f32).collect();

        let (min, max) = MinMax { xs: &buf }.dispatch();
        assert_eq!(min, 0. as f32);
        assert_eq!(max, (TEST_LEN - 1) as f32);
    }
}
