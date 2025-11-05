use rayon::prelude::*;
use rten_base::iter::{ParIter, SplitIterator};

use super::{
    AxisChunks, AxisChunksMut, AxisIter, AxisIterMut, InnerIter, InnerIterBase, InnerIterMut, Iter,
    IterMut, LaneRanges, Lanes, LanesMut, Offsets, OffsetsKind,
};
use crate::Storage;
use crate::layout::{Layout, MutLayout, RemoveDim};

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

impl SplitIterator for Offsets {
    fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index <= self.len());
        let (left_kind, right_kind) = match self.base {
            OffsetsKind::Range(r) => {
                let left = r.start..r.start + index;
                let right = r.start + index..r.end;
                (OffsetsKind::Range(left), OffsetsKind::Range(right))
            }
            OffsetsKind::Indexing(base) => {
                let (left, right) = base.split_at(index);
                (OffsetsKind::Indexing(left), OffsetsKind::Indexing(right))
            }
        };
        (Offsets { base: left_kind }, Offsets { base: right_kind })
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
        let (left_offsets, right_offsets) = self.offsets.split_at(index);
        let left = Self {
            offsets: left_offsets,
            data: self.data,
        };
        let right = Self {
            offsets: right_offsets,
            data: self.data,
        };
        (left, right)
    }
}

impl<'a, T: Sync> IntoParallelIterator for Iter<'a, T> {
    impl_parallel_iterator!();
}

impl<'a, T> SplitIterator for IterMut<'a, T> {
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_offsets, right_offsets) = self.offsets.split_at(index);
        let len = self.data.len();
        let (left_data, right_data) = self.data.split_mut(0..len, 0..len);
        let left = Self {
            offsets: left_offsets,
            data: left_data,
        };
        let right = Self {
            offsets: right_offsets,
            data: right_data,
        };
        (left, right)
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
            lane_layout: self.lane_layout,
        };
        let right = Lanes {
            data: self.data,
            ranges: right_range,
            lane_layout: self.lane_layout,
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
            lane_layout: self.lane_layout,
        };
        let right = Self {
            data: right_data,
            ranges: right_range,
            lane_layout: self.lane_layout,
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
