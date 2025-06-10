use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;

use super::{
    AxisChunks, AxisChunksMut, AxisIter, AxisIterMut, IndexingIter, IndexingIterMut, InnerIter,
    InnerIterBase, InnerIterMut, Iter, IterKind, IterMut, IterMutKind, LaneRanges, Lanes, LanesMut,
    Offsets,
};
use crate::layout::RemoveDim;
use crate::{Layout, MutLayout, Storage};

/// A trait to simplify adding Rayon support to iterators.
///
/// This is used in combination with [`ParIter`] and assumes your iterator has
/// a known size and can be split at an arbitrary position.
///
/// Adding Rayon support to an iterator using this trait requires implementing
/// the following traits for the iterator:
///
/// 1. [`DoubleEndedIterator`] and [`ExactSizeIterator`]. These are requirements
///    for Rayon's [`IndexedParallelIterator`].
/// 2. `SplitIterator` to define how Rayon should split the iterator
/// 3. [`IntoParallelIterator`] using [`ParIter<I>`] as the parallel iterator
///    type.
pub trait SplitIterator: DoubleEndedIterator + ExactSizeIterator {
    /// Split the iterator in two at a given index.
    ///
    /// The left result will yield the first `index` items and the right result
    /// will yield items starting from `index`.
    ///
    /// Panics if `index` is greater than the iterator's length, as reported by
    /// [`ExactSizeIterator::len`].
    fn split_at(self, index: usize) -> (Self, Self)
    where
        Self: Sized;
}

/// A parallel wrapper around a serial iterator.
///
/// This type should be used as the [`IntoParallelIterator::Iter`] associated
/// type in an implementation of [`IntoParallelIterator`] for `I`.
pub struct ParIter<I: SplitIterator>(I);

impl<I: SplitIterator> From<I> for ParIter<I> {
    fn from(val: I) -> Self {
        ParIter(val)
    }
}

/// Generate the body of an [`IntoParallelIterator`] impl which uses [`ParIter`]
/// as the iterator type.
macro_rules! impl_parallel_iterator {
    () => {
        type Iter = ParIter<Self>;
        type Item = <Self as Iterator>::Item;

        fn into_par_iter(self) -> Self::Iter {
            self.into()
        }
    };
}

impl<I: SplitIterator + Send> ParallelIterator for ParIter<I>
where
    <I as Iterator>::Item: Send,
{
    type Item = I::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(ExactSizeIterator::len(&self.0))
    }
}

impl<I: SplitIterator + Send> IndexedParallelIterator for ParIter<I>
where
    <I as Iterator>::Item: Send,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        ExactSizeIterator::len(&self.0)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

impl<I: SplitIterator + Send> Producer for ParIter<I> {
    type Item = I::Item;

    type IntoIter = I;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_inner, right_inner) = SplitIterator::split_at(self.0, index);
        (Self(left_inner), Self(right_inner))
    }
}

impl SplitIterator for Offsets {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_base, right_base) = self.base.split_at(index);
        (Offsets { base: left_base }, Offsets { base: right_base })
    }
}

impl<L: Layout + Clone> SplitIterator for InnerIterBase<L> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_offsets, right_offsets) = self.outer_offsets.split_at(index);
        let left = Self {
            outer_offsets: left_offsets,
            inner_layout: self.inner_layout.clone(),
            inner_data_len: self.inner_data_len,
        };
        let right = Self {
            outer_offsets: right_offsets,
            inner_layout: self.inner_layout,
            inner_data_len: self.inner_data_len,
        };
        (left, right)
    }
}

impl<'a, T, L: MutLayout + Send + Sync> SplitIterator for InnerIter<'a, T, L> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_base, right_base) = self.base.split_at(index);
        let left = Self {
            base: left_base,
            data: self.data,
        };
        let right = Self {
            base: right_base,
            data: self.data,
        };
        (left, right)
    }
}

impl<'a, T, L: MutLayout + Send + Sync> IntoParallelIterator for InnerIter<'a, T, L> {
    impl_parallel_iterator!();
}

impl<'a, T, L: MutLayout + Send + Sync> SplitIterator for InnerIterMut<'a, T, L> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_base, right_base) = self.base.split_at(index);
        let len = self.data.len();

        // The left/right splits use the same storage. We rely on the left/right
        // layouts being logically disjoint to ensure we don't create multiple
        // mutable references to the same elements.
        let (left_data, right_data) = self.data.split_mut(0..len, 0..len);

        let left = Self {
            base: left_base,
            data: left_data,
        };
        let right = Self {
            base: right_base,
            data: right_data,
        };
        (left, right)
    }
}

impl<'a, T, L: MutLayout + Send + Sync> IntoParallelIterator for InnerIterMut<'a, T, L> {
    impl_parallel_iterator!();
}

impl<'a, T, L: MutLayout + RemoveDim> SplitIterator for AxisIter<'a, T, L> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_view, right_view) = self.view.split_at(self.axis, index);
        let left = AxisIter::new(&left_view, self.axis);
        let right = AxisIter::new(&right_view, self.axis);
        (left, right)
    }
}

impl<'a, T, L: MutLayout + RemoveDim + Send> IntoParallelIterator for AxisIter<'a, T, L>
where
    <L as RemoveDim>::Output: Send,
{
    impl_parallel_iterator!();
}

impl<'a, T, L: MutLayout + RemoveDim> SplitIterator for AxisIterMut<'a, T, L> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_view, right_view) = self.view.split_at_mut(self.axis, index);
        let left = AxisIterMut::new(left_view, self.axis);
        let right = AxisIterMut::new(right_view, self.axis);
        (left, right)
    }
}

impl<'a, T, L: MutLayout + RemoveDim + Send> IntoParallelIterator for AxisIterMut<'a, T, L>
where
    <L as RemoveDim>::Output: Send,
{
    impl_parallel_iterator!();
}

impl<'a, T, L: MutLayout> SplitIterator for AxisChunks<'a, T, L> {
    fn split_at(mut self, index: usize) -> (Self, Self) {
        let (left_remainder, right_remainder) = if let Some(remainder) = self.remainder.take() {
            let (l, r) = remainder.split_at(self.axis, self.chunk_size * index);
            (Some(l), Some(r))
        } else {
            (None, None)
        };

        let left = AxisChunks {
            remainder: left_remainder,
            axis: self.axis,
            chunk_size: self.chunk_size,
        };
        let right = AxisChunks {
            remainder: right_remainder,
            axis: self.axis,
            chunk_size: self.chunk_size,
        };

        (left, right)
    }
}

impl<'a, T, L: MutLayout + Send> IntoParallelIterator for AxisChunks<'a, T, L> {
    impl_parallel_iterator!();
}

impl<'a, T, L: MutLayout> SplitIterator for AxisChunksMut<'a, T, L> {
    fn split_at(mut self, index: usize) -> (Self, Self) {
        let (left_remainder, right_remainder) = if let Some(remainder) = self.remainder.take() {
            let (l, r) = remainder.split_at_mut(self.axis, self.chunk_size * index);
            (Some(l), Some(r))
        } else {
            (None, None)
        };

        let left = Self {
            remainder: left_remainder,
            axis: self.axis,
            chunk_size: self.chunk_size,
        };
        let right = Self {
            remainder: right_remainder,
            axis: self.axis,
            chunk_size: self.chunk_size,
        };

        (left, right)
    }
}

impl<'a, T, L: MutLayout + Send> IntoParallelIterator for AxisChunksMut<'a, T, L> {
    impl_parallel_iterator!();
}

impl<'a, T> SplitIterator for Iter<'a, T> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = match self.iter {
            IterKind::Direct(iter) => {
                let (left_slice, right_slice) = iter.as_slice().split_at(index);
                (
                    IterKind::Direct(left_slice.iter()),
                    IterKind::Direct(right_slice.iter()),
                )
            }
            IterKind::Indexing(iter) => {
                let (left_base, right_base) = iter.base.split_at(index);
                let left = IndexingIter {
                    base: left_base,
                    data: iter.data,
                };
                let right = IndexingIter {
                    base: right_base,
                    data: iter.data,
                };
                (IterKind::Indexing(left), IterKind::Indexing(right))
            }
        };

        (Self { iter: left }, Self { iter: right })
    }
}

impl<'a, T: Sync> IntoParallelIterator for Iter<'a, T> {
    impl_parallel_iterator!();
}

impl<'a, T> SplitIterator for IterMut<'a, T> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = match self.iter {
            IterMutKind::Direct(iter) => {
                let (left_slice, right_slice) = iter.into_slice().split_at_mut(index);
                (
                    IterMutKind::Direct(left_slice.iter_mut()),
                    IterMutKind::Direct(right_slice.iter_mut()),
                )
            }
            IterMutKind::Indexing(iter) => {
                let (left_base, right_base) = iter.base.split_at(index);
                let len = iter.data.len();

                // Safety note: `split_mut` relies on the caller to ensure that
                // associated layouts do not overlap.
                let (left_data, right_data) = iter.data.split_mut(0..len, 0..len);

                let left = IndexingIterMut {
                    base: left_base,
                    data: left_data,
                };
                let right = IndexingIterMut {
                    base: right_base,
                    data: right_data,
                };
                (IterMutKind::Indexing(left), IterMutKind::Indexing(right))
            }
        };

        (Self { iter: left }, Self { iter: right })
    }
}

impl<'a, T: Sync + Send> IntoParallelIterator for IterMut<'a, T> {
    impl_parallel_iterator!();
}

impl SplitIterator for LaneRanges {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_offsets, right_offsets) = self.offsets.split_at(index);
        let left = LaneRanges {
            offsets: left_offsets,
            dim_size: self.dim_size,
            dim_stride: self.dim_stride,
        };
        let right = LaneRanges {
            offsets: right_offsets,
            dim_size: self.dim_size,
            dim_stride: self.dim_stride,
        };
        (left, right)
    }
}

impl<'a, T> SplitIterator for Lanes<'a, T> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_range, right_range) = self.ranges.split_at(index);

        let left = Lanes {
            data: self.data,
            ranges: left_range,
            size: self.size,
            stride: self.stride,
        };
        let right = Lanes {
            data: self.data,
            ranges: right_range,
            size: self.size,
            stride: self.stride,
        };

        (left, right)
    }
}

impl<'a, T: Sync + Send> IntoParallelIterator for Lanes<'a, T> {
    impl_parallel_iterator!();
}

impl<'a, T> SplitIterator for LanesMut<'a, T> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_range, right_range) = self.ranges.split_at(index);
        let len = self.data.len();

        // Safety note: `split_mut` relies on the caller to ensure that
        // associated layouts do not overlap.
        let (left_data, right_data) = self.data.split_mut(0..len, 0..len);

        let left = Self {
            data: left_data,
            ranges: left_range,
            size: self.size,
            stride: self.stride,
        };
        let right = Self {
            data: right_data,
            ranges: right_range,
            size: self.size,
            stride: self.stride,
        };

        (left, right)
    }
}

impl<T: Sync + Send> IntoParallelIterator for LanesMut<'_, T> {
    impl_parallel_iterator!();
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;

    use crate::rng::XorShiftRng;
    use crate::{AsView, Tensor};

    // These helpers use macros to work around difficulties expressing lifetime
    // relationships between input and output in closures that take an `&Tensor`
    // and return an `impl Iterator + IntoParallelIterator`.

    // Test that the parallel version of an iterator yields the same items as
    // the serial version.
    macro_rules! test_parallel_iterator {
        ($x:ident, $iter:expr) => {
            let mut rng = XorShiftRng::new(1234);
            let $x = Tensor::<f32>::rand(&[4, 8, 16, 32], &mut rng);
            let serial: Vec<_> = $iter.collect();
            let parallel: Vec<_> = $iter.into_par_iter().collect();
            assert_eq!(serial, parallel);
        };
    }

    // Test that the parallel version of a mutable iterator yields the same
    // items as the serial version.
    macro_rules! test_parallel_iterator_mut {
        ($x:ident, $iter:expr, $item_sum:expr) => {
            let mut rng = XorShiftRng::new(1234);

            // Use ints rather than floats here to avoid mismatches due to
            // parallel iteration visiting items in a different order to serial
            // iteration.
            let mut $x =
                Tensor::<i32>::from_simple_fn(&[4, 8, 16, 32], || (rng.next_f32() * 100.) as i32);
            let serial: i32 = $iter.map($item_sum).sum();
            let parallel: i32 = $iter.into_par_iter().map($item_sum).sum();

            assert_eq!(serial, parallel);
        };
    }

    // Test that the parallel version of an iterator yields the same items as
    // the serial version.
    //
    // This is a variant for the case where the items are themselves iterators.
    macro_rules! test_parallel_iterator_flatten {
        ($x:ident, $iter:expr) => {
            let mut rng = XorShiftRng::new(1234);
            let $x = Tensor::<f32>::rand(&[4, 8, 16, 32], &mut rng);

            let serial: Vec<_> = $iter.collect();
            let parallel: Vec<_> = $iter.into_par_iter().collect();

            let serial_items: Vec<f32> = serial.into_iter().flatten().copied().collect();
            let parallel_items: Vec<f32> = parallel.into_iter().flatten().copied().collect();
            assert_eq!(serial_items, parallel_items);
        };
    }

    // Parallel tests are skipped under Miri due to
    // https://github.com/crossbeam-rs/crossbeam/issues/1181.

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_inner_iter_parallel() {
        test_parallel_iterator!(x, x.inner_iter::<2>());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_inner_iter_mut_parallel() {
        test_parallel_iterator_mut!(x, x.inner_iter_mut::<2>(), |x| x.iter().sum::<i32>());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_iter_parallel() {
        test_parallel_iterator!(x, x.iter());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_iter_mut_parallel() {
        test_parallel_iterator_mut!(x, x.iter_mut(), |x| *x);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_axis_chunks_parallel() {
        test_parallel_iterator!(x, x.axis_chunks(0, 2));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_axis_chunks_mut_parallel() {
        test_parallel_iterator_mut!(x, x.axis_chunks_mut(0, 2), |x| x.iter().sum::<i32>());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_axis_iter_parallel() {
        test_parallel_iterator!(x, x.axis_iter(0));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_axis_iter_mut_parallel() {
        test_parallel_iterator_mut!(x, x.axis_iter_mut(0), |x| x.iter().sum::<i32>());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_lanes_parallel() {
        test_parallel_iterator_flatten!(x, x.lanes(0));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_lanes_mut_parallel() {
        test_parallel_iterator_mut!(x, x.lanes_mut(0), |x| x.map(|x| *x).sum::<i32>());
    }
}
