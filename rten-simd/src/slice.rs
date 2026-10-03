//! Slices which support loading and storing SIMD vectors at arbitrary offsets.

use std::marker::PhantomData;
use std::ops::Range;

use crate::Elem;
use crate::ops::BitOps;

/// Wrapper around a slice for loading SIMD vectors at arbitrary offsets.
///
/// The wrapper checks once when it is created that the slice holds at least
/// one vector. Loads then take an element offset which is clamped into range.
/// This makes random-access vector loads safe without a branch per access: in
/// release builds an out-of-range offset loads the final full vector instead of
/// panicking or causing undefined behavior. Debug builds panic on an
/// out-of-range offset.
///
/// Compared to [`BitOps::load`], which checks the slice length on every call,
/// the clamp compiles to a compare and conditional select with no branch. This
/// keeps a kernel's inner loop as a single basic block, which matters for
/// kernels with strided access patterns where the compiler cannot prove the
/// offsets in bounds itself.
// The slice always holds at least one vector, so there is no `is_empty`.
#[allow(clippy::len_without_is_empty)]
pub struct SimdSlice<'a, T: Elem, O: BitOps<T>> {
    ops: O,
    ptr: *const T,
    len: usize,
    /// Largest offset at which a full vector can be loaded, `len - lanes`.
    max_offset: usize,
    _marker: PhantomData<&'a [T]>,
}

impl<T: Elem, O: BitOps<T>> Clone for SimdSlice<'_, T, O> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Elem, O: BitOps<T>> Copy for SimdSlice<'_, T, O> {}

impl<'a, T: Elem, O: BitOps<T>> SimdSlice<'a, T, O> {
    /// Wrap `xs`.
    ///
    /// Returns `None` if `xs` is shorter than one vector.
    pub fn new(ops: O, xs: &'a [T]) -> Option<Self> {
        let max_offset = xs.len().checked_sub(ops.len())?;
        Some(SimdSlice {
            ops,
            ptr: xs.as_ptr(),
            len: xs.len(),
            max_offset,
            _marker: PhantomData,
        })
    }

    /// Return the number of elements in the slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return the largest offset at which a full vector can be loaded.
    pub fn max_offset(&self) -> usize {
        self.max_offset
    }

    /// Return the underlying slice.
    pub fn as_slice(&self) -> &'a [T] {
        // Safety: `ptr` and `len` came from a slice with lifetime `'a`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Narrow the slice to `range`.
    ///
    /// Returns `None` if `range` is out of bounds or shorter than one vector.
    pub fn slice(&self, range: Range<usize>) -> Option<SimdSlice<'a, T, O>> {
        let xs = self.as_slice().get(range)?;
        SimdSlice::new(self.ops, xs)
    }

    /// Load the vector starting at element `offset`.
    ///
    /// `offset` is clamped to [`max_offset`](Self::max_offset), so the result
    /// is always a vector of elements from the slice. Debug builds assert
    /// that the offset is in range.
    #[inline(always)]
    pub fn load(&self, offset: usize) -> O::Simd {
        debug_assert!(offset <= self.max_offset, "vector load out of range");
        let offset = offset.min(self.max_offset);
        // Safety: `offset + lanes <= len` by construction of `max_offset`.
        unsafe { self.ops.load_ptr(self.ptr.add(offset)) }
    }
}

/// Wrapper around a mutable slice for loading and storing SIMD vectors at
/// arbitrary offsets.
///
/// This has the same semantics as [`SimdSlice`]. A store at an out-of-range
/// offset overwrites the final full vector, which is sound but silently
/// wrong, so debug builds assert that offsets are in range.
#[allow(clippy::len_without_is_empty)]
pub struct SimdSliceMut<'a, T: Elem, O: BitOps<T>> {
    ops: O,
    ptr: *mut T,
    len: usize,
    max_offset: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T: Elem, O: BitOps<T>> SimdSliceMut<'a, T, O> {
    /// Wrap `xs`.
    ///
    /// Returns `None` if `xs` is shorter than one vector.
    pub fn new(ops: O, xs: &'a mut [T]) -> Option<Self> {
        let max_offset = xs.len().checked_sub(ops.len())?;
        Some(SimdSliceMut {
            ops,
            ptr: xs.as_mut_ptr(),
            len: xs.len(),
            max_offset,
            _marker: PhantomData,
        })
    }

    /// Return the number of elements in the slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return the largest offset at which a full vector can be loaded or
    /// stored.
    pub fn max_offset(&self) -> usize {
        self.max_offset
    }

    /// Return a read-only wrapper around the same slice.
    pub fn as_simd_slice(&self) -> SimdSlice<'_, T, O> {
        SimdSlice {
            ops: self.ops,
            ptr: self.ptr,
            len: self.len,
            max_offset: self.max_offset,
            _marker: PhantomData,
        }
    }

    /// Reborrow with a shorter lifetime.
    pub fn reborrow(&mut self) -> SimdSliceMut<'_, T, O> {
        SimdSliceMut {
            ops: self.ops,
            ptr: self.ptr,
            len: self.len,
            max_offset: self.max_offset,
            _marker: PhantomData,
        }
    }

    /// Return the underlying slice.
    pub fn into_mut_slice(self) -> &'a mut [T] {
        // Safety: `ptr` and `len` came from a mutable slice with lifetime
        // `'a`, which this wrapper holds exclusively.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Narrow the slice to `range`.
    ///
    /// Returns `None` if `range` is out of bounds or shorter than one vector.
    pub fn slice_mut(self, range: Range<usize>) -> Option<SimdSliceMut<'a, T, O>> {
        let ops = self.ops;
        let xs = self.into_mut_slice().get_mut(range)?;
        SimdSliceMut::new(ops, xs)
    }

    /// Split the slice into two at `mid`.
    ///
    /// Returns `None` if either half is shorter than one vector.
    #[allow(clippy::type_complexity)]
    pub fn split_at_mut(
        self,
        mid: usize,
    ) -> Option<(SimdSliceMut<'a, T, O>, SimdSliceMut<'a, T, O>)> {
        let ops = self.ops;
        let (left, right) = self.into_mut_slice().split_at_mut_checked(mid)?;
        Some((
            SimdSliceMut::new(ops, left)?,
            SimdSliceMut::new(ops, right)?,
        ))
    }

    /// Load the vector starting at element `offset`.
    ///
    /// See [`SimdSlice::load`].
    #[inline(always)]
    pub fn load(&self, offset: usize) -> O::Simd {
        self.as_simd_slice().load(offset)
    }

    /// Store `x` starting at element `offset`.
    ///
    /// `offset` is clamped to [`max_offset`](Self::max_offset), so the store
    /// always writes within the slice. Debug builds assert that the offset is
    /// in range.
    #[inline(always)]
    pub fn store(&mut self, offset: usize, x: O::Simd) {
        debug_assert!(offset <= self.max_offset, "vector store out of range");
        let offset = offset.min(self.max_offset);
        // Safety: `offset + lanes <= len` by construction of `max_offset`.
        unsafe { self.ops.store_ptr(x, self.ptr.add(offset)) }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::BitOps;
    use crate::{Isa, Simd, SimdOp, test_simd_op};

    use super::{SimdSlice, SimdSliceMut};

    fn lanes<T: Simd>(x: T) -> Vec<T::Elem> {
        x.to_array().as_ref().to_vec()
    }

    #[test]
    fn test_simd_slice_load() {
        test_simd_op!(isa, {
            let ops = isa.f32();
            let n = ops.len();
            let xs: Vec<f32> = (0..3 * n).map(|x| x as f32).collect();

            let slice = SimdSlice::new(ops, &xs).unwrap();
            assert_eq!(slice.len(), xs.len());
            assert_eq!(slice.max_offset(), 2 * n);
            assert_eq!(slice.as_slice(), &xs);

            // Aligned and unaligned offsets.
            for offset in [0, 1, n, 2 * n] {
                assert_eq!(lanes(slice.load(offset)), &xs[offset..offset + n]);
            }

            // Narrowed slice.
            let sub = slice.slice(1..1 + n).unwrap();
            assert_eq!(lanes(sub.load(0)), &xs[1..1 + n]);
            assert!(slice.slice(0..n - 1).is_none());
            assert!(slice.slice(0..xs.len() + 1).is_none());

            // Too short.
            assert!(SimdSlice::new(ops, &xs[..n - 1]).is_none());
        });
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "vector load out of range"))]
    fn test_simd_slice_load_clamped() {
        test_simd_op!(isa, {
            let ops = isa.f32();
            let n = ops.len();
            let xs: Vec<f32> = (0..2 * n).map(|x| x as f32).collect();

            // In release builds an out-of-range offset loads the final vector.
            let slice = SimdSlice::new(ops, &xs).unwrap();
            assert_eq!(lanes(slice.load(usize::MAX)), &xs[n..]);
        });
    }

    #[test]
    fn test_simd_slice_mut_store() {
        test_simd_op!(isa, {
            let ops = isa.f32();
            let n = ops.len();
            let mut xs = vec![0.; 3 * n];

            let mut slice = SimdSliceMut::new(ops, &mut xs).unwrap();
            assert_eq!(slice.len(), 3 * n);
            assert_eq!(slice.max_offset(), 2 * n);

            slice.store(0, ops.splat(1.));
            slice.store(1, ops.splat(2.));
            slice.store(2 * n, ops.splat(3.));
            assert_eq!(lanes(slice.load(1)), vec![2.; n]);
            assert_eq!(lanes(slice.as_simd_slice().load(2 * n)), vec![3.; n]);

            // Split and narrowed slices.
            let (mut left, mut right) = slice.reborrow().split_at_mut(n).unwrap();
            left.store(0, ops.splat(4.));
            right.store(0, ops.splat(5.));
            let mut sub = slice.reborrow().slice_mut(2 * n..3 * n).unwrap();
            sub.store(0, ops.splat(6.));
            assert!(slice.reborrow().split_at_mut(n - 1).is_none());
            assert!(slice.reborrow().slice_mut(0..n - 1).is_none());

            let xs = slice.into_mut_slice();
            let mut expected = vec![4.; n];
            expected.extend(vec![5.; n]);
            expected.extend(vec![6.; n]);
            assert_eq!(xs, expected);

            // Too short.
            assert!(SimdSliceMut::new(ops, &mut xs[..n - 1]).is_none());
        });
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "vector store out of range"))]
    fn test_simd_slice_mut_store_clamped() {
        test_simd_op!(isa, {
            let ops = isa.f32();
            let n = ops.len();
            let mut xs = vec![0.; 2 * n];

            // In release builds an out-of-range offset stores to the final
            // vector.
            let mut slice = SimdSliceMut::new(ops, &mut xs).unwrap();
            slice.store(usize::MAX, ops.splat(1.));
            let mut expected = vec![0.; n];
            expected.extend(vec![1.; n]);
            assert_eq!(slice.into_mut_slice(), expected);
        });
    }
}
