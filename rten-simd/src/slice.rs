//! Slices which support loading and storing SIMD vectors at arbitrary offsets.

use crate::Elem;
use crate::ops::BitOps;

/// Wrapper around a slice for loading SIMD vectors at arbitrary offsets.
///
/// To load a SIMD vector directly from an arbitrary offset in a slice you could
/// use [`BitOps::load`] via `ops.load(&slice[offset..])`. This has two panic
/// branches, one for indexing the slice and one in `ops.load` if the sub-slice
/// is too short. These panic branches may inhibit compiler optimization. This
/// wrapper provides an alternative which clamps offsets to a valid range in
/// release builds. Debug builds will panic if the offset is invalid. For this
/// behavior to work, the wrapped slice must be at least one SIMD vector long.
/// This is checked when the wrapper is constructed.
#[derive(Copy, Clone)]
pub struct SimdSlice<'a, T: Elem, O: BitOps<T>> {
    ops: O,
    xs: &'a [T],
    /// Largest offset at which a full vector can be loaded, `len - lanes`.
    max_offset: usize,
}

impl<'a, T: Elem, O: BitOps<T>> SimdSlice<'a, T, O> {
    /// Wrap `xs`.
    ///
    /// Returns `None` if `xs` is shorter than one vector.
    pub fn new(ops: O, xs: &'a [T]) -> Option<Self> {
        let max_offset = xs.len().checked_sub(ops.len())?;
        Some(SimdSlice {
            ops,
            xs,
            max_offset,
        })
    }

    /// Load the vector starting at element `offset`.
    ///
    /// In debug builds, this panics if the offset is out of bounds. In release
    /// builds, `offset` is clamped to the last valid offset.
    #[inline(always)]
    pub fn load(&self, offset: usize) -> O::Simd {
        debug_assert!(
            offset <= self.max_offset,
            "load offset {} exceeds maximum {}",
            offset,
            self.max_offset
        );
        let offset = offset.min(self.max_offset);
        // Safety: `offset + lanes <= xs.len()` by construction of `max_offset`.
        unsafe { self.ops.load_ptr(self.xs.as_ptr().add(offset)) }
    }
}

/// Wrapper around a mutable slice for loading and storing SIMD vectors at
/// arbitrary offsets.
///
/// This has the same semantics as [`SimdSlice`]. A store at an out-of-range
/// offset overwrites the final full vector, which is sound but silently
/// wrong. Debug builds panic if the offset is out of bounds.
pub struct SimdSliceMut<'a, T: Elem, O: BitOps<T>> {
    ops: O,
    xs: &'a mut [T],
    max_offset: usize,
}

impl<'a, T: Elem, O: BitOps<T>> SimdSliceMut<'a, T, O> {
    /// Wrap `xs`.
    ///
    /// Returns `None` if `xs` is shorter than one vector.
    pub fn new(ops: O, xs: &'a mut [T]) -> Option<Self> {
        let max_offset = xs.len().checked_sub(ops.len())?;
        Some(SimdSliceMut {
            ops,
            xs,
            max_offset,
        })
    }

    /// Load the vector starting at element `offset`.
    ///
    /// See [`SimdSlice::load`].
    #[inline(always)]
    pub fn load(&self, offset: usize) -> O::Simd {
        debug_assert!(
            offset <= self.max_offset,
            "load offset {} exceeds maximum {}",
            offset,
            self.max_offset
        );
        let offset = offset.min(self.max_offset);
        // Safety: `offset + lanes <= xs.len()` by construction of `max_offset`.
        unsafe { self.ops.load_ptr(self.xs.as_ptr().add(offset)) }
    }

    /// Store `x` starting at element `offset`.
    ///
    /// In release builds `offset` is clamped to the last valid offset. In
    /// debug builds this will panic if the offset is out of range.
    #[inline(always)]
    pub fn store(&mut self, offset: usize, x: O::Simd) {
        debug_assert!(
            offset <= self.max_offset,
            "store offset {} exceeds maximum {}",
            offset,
            self.max_offset
        );
        let offset = offset.min(self.max_offset);
        // Safety: `offset + lanes <= xs.len()` by construction of `max_offset`.
        unsafe { self.ops.store_ptr(x, self.xs.as_mut_ptr().add(offset)) }
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
    fn test_simd_slice_new() {
        test_simd_op!(isa, {
            let ops = isa.f32();
            let n = ops.len();
            let xs: Vec<f32> = (0..3 * n).map(|x| x as f32).collect();

            assert!(SimdSlice::new(ops, &xs[..n]).is_some());
            assert!(SimdSlice::new(ops, &xs[..n - 1]).is_none());
        });
    }

    #[test]
    fn test_simd_slice_load() {
        test_simd_op!(isa, {
            let ops = isa.f32();
            let n = ops.len();
            let xs: Vec<f32> = (0..3 * n).map(|x| x as f32).collect();

            let slice = SimdSlice::new(ops, &xs).unwrap();

            // Aligned and unaligned offsets, up to the last full vector.
            for offset in [0, 1, n, 2 * n] {
                assert_eq!(lanes(slice.load(offset)), &xs[offset..offset + n]);
            }
        });
    }

    #[test]
    fn test_simd_slice_mut_new() {
        test_simd_op!(isa, {
            let ops = isa.f32();
            let n = ops.len();
            let mut xs = vec![0.; 3 * n];

            assert!(SimdSliceMut::new(ops, &mut xs[..n]).is_some());
            assert!(SimdSliceMut::new(ops, &mut xs[..n - 1]).is_none());
        });
    }

    #[test]
    fn test_simd_slice_mut_store() {
        test_simd_op!(isa, {
            let ops = isa.f32();
            let n = ops.len();
            let mut xs = vec![0.; 3 * n];

            let mut slice = SimdSliceMut::new(ops, &mut xs).unwrap();

            // Aligned and unaligned offsets, up to the last full vector.
            slice.store(0, ops.splat(1.));
            slice.store(1, ops.splat(2.));
            slice.store(2 * n, ops.splat(3.));
            assert_eq!(lanes(slice.load(1)), vec![2.; n]);
            assert_eq!(lanes(slice.load(2 * n)), vec![3.; n]);

            let mut expected = vec![1.];
            expected.extend(vec![2.; n]);
            expected.extend(vec![0.; n - 1]);
            expected.extend(vec![3.; n]);
            assert_eq!(xs, expected);
        });
    }
}
