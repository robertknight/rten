//! Iterators and iterator-related traits.

use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::prelude::*;

mod range;
pub use range::{RangeChunks, RangeChunksExact, range_chunks, range_chunks_exact};

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
