//! [`ParIter`] utility to simplify implementing Rayon's parallel iterator traits.
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::prelude::*;

use rten_base::iter::SplitIterator;

/// Wraps a splittable iterator to implement Rayon's parallel iterator traits.
///
/// This type makes it easy to implement Rayon's parallel iterator traits
/// ([`ParallelIterator`], [`IndexedParallelIterator`]) for custom iterators.
/// Adding Rayon support to an iterator using this type requires implementing
/// the following traits for the iterator:
///
/// 1. [`DoubleEndedIterator`] and [`ExactSizeIterator`]. These are requirements
///    for Rayon's [`IndexedParallelIterator`]. The [`DoubleEndedIterator`]
///    implementation is not actually needed for most uses, so a simple but
///    slow implementation will suffice.
///    See <https://github.com/rayon-rs/rayon/issues/1053>.
/// 2. [`SplitIterator`] to define how Rayon should split the iterator
///
/// With these traits implemented, a parallel iterator can be created using
/// `ParIter::from(iter)`. For improved ergonomics, you can also implement
/// Rayon's [`IntoParallelIterator`] trait using [`ParIter<I>`] as the parallel
/// iterator associated type.
///
/// # Example
///
/// This is a minimal example showing how to add Rayon support to a custom
/// iterator type.
///
/// ```
/// use rayon::iter::{ParallelIterator, IntoParallelIterator};
///
/// use rten_base::iter::SplitIterator;
/// use rten_parallel::par_iter::ParIter;
///
/// #[derive(Clone)]
/// struct CustomRange {
///     start: u32,
///     end: u32,
/// }
///
/// impl CustomRange {
///     fn new(start: u32, end: u32) -> Self {
///         assert!(start <= end);
///         Self { start, end }
///     }
/// }
///
/// impl Iterator for CustomRange {
///     type Item = u32;
///
///     fn next(&mut self) -> Option<u32> {
///         if self.start < self.end {
///             let item = self.start;
///             self.start += 1;
///             Some(item)
///         } else {
///             None
///         }
///     }
///
///     fn size_hint(&self) -> (usize, Option<usize>) {
///         let len = self.end as usize - self.start as usize;
///         (len, Some(len))
///     }
/// }
///
/// impl ExactSizeIterator for CustomRange {}
///
/// // DoubleEndedIterator is currently necessary, but usually unused. A crude
/// // but simple implementation will suffice.
/// impl DoubleEndedIterator for CustomRange {
///     fn next_back(&mut self) -> Option<Self::Item> {
///         if self.start < self.end {
///             let item = self.end - 1;
///             self.end -= 1;
///             Some(item)
///         } else {
///             None
///         }
///     }
/// }
///
/// impl SplitIterator for CustomRange {
///     fn split_at(self, index: usize) -> (Self, Self) {
///         assert!(index < self.len());
///         let left = CustomRange { start: self.start, end: self.start + index as u32 };
///         let right = CustomRange { start: self.start + index as u32, end: self.end };
///         debug_assert_eq!(left.len() + right.len(), self.len());
///         (left, right)
///     }
/// }
///
/// impl IntoParallelIterator for CustomRange {
///     type Iter = ParIter<Self>;
///     type Item = u32;
///
///     fn into_par_iter(self) -> Self::Iter {
///         ParIter::from(self)
///     }
/// }
///
/// let range = CustomRange::new(0, 100);
///
/// // Process items in serial.
/// let serial_nums: Vec<_> = range.clone().map(|x| x * x).collect();
///
/// // Process items in parallel.
/// let par_nums: Vec<_> = range.into_par_iter().map(|x| x * x).collect();
/// assert_eq!(par_nums, serial_nums);
/// ```
pub struct ParIter<I: SplitIterator>(I);

impl<I: SplitIterator> From<I> for ParIter<I> {
    fn from(val: I) -> Self {
        ParIter(val)
    }
}

impl<I: SplitIterator + DoubleEndedIterator + Send> ParallelIterator for ParIter<I>
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

impl<I: SplitIterator + DoubleEndedIterator + Send> IndexedParallelIterator for ParIter<I>
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

impl<I: SplitIterator + DoubleEndedIterator + Send> Producer for ParIter<I> {
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

/// Wrapper around either a serial or parallel iterator, returned by
/// [`MaybeParIter::maybe_par_iter`].
pub enum MaybeParallel<PI: ParallelIterator, SI: Iterator<Item = PI::Item>> {
    Serial(SI),
    Parallel(PI),
}

impl<PI: ParallelIterator, SI: Iterator<Item = PI::Item>> MaybeParallel<PI, SI> {
    pub fn for_each<F: Fn(PI::Item) + Send + Sync>(self, f: F) {
        match self {
            MaybeParallel::Serial(iter) => iter.for_each(f),
            MaybeParallel::Parallel(iter) => iter.for_each(f),
        }
    }
}

/// Trait which allows use of Rayon parallelism to be conditionally enabled.
///
/// See <https://crates.io/crates/rayon-cond> for a more full-featured alternative.
pub trait MaybeParIter {
    type Item;
    type ParIter: ParallelIterator<Item = Self::Item>;
    type Iter: Iterator<Item = Self::Item>;

    /// Return an iterator which executes either in serial on the current
    /// thread, or in parallel in a Rayon thread pool if `parallel` is true.
    fn maybe_par_iter(self, parallel: bool) -> MaybeParallel<Self::ParIter, Self::Iter>;
}

impl<Item, I: rayon::iter::IntoParallelIterator<Item = Item> + IntoIterator<Item = Item>>
    MaybeParIter for I
{
    type Item = Item;
    type ParIter = I::Iter;
    type Iter = I::IntoIter;

    fn maybe_par_iter(self, parallel: bool) -> MaybeParallel<Self::ParIter, Self::Iter> {
        if parallel {
            MaybeParallel::Parallel(self.into_par_iter())
        } else {
            MaybeParallel::Serial(self.into_iter())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::MaybeParIter;

    #[test]
    fn test_maybe_par_iter() {
        let count = AtomicU32::new(0);
        (0..1000).maybe_par_iter(false).for_each(|_| {
            count.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(count.load(Ordering::SeqCst), 1000);

        let count = AtomicU32::new(0);
        (0..1000).maybe_par_iter(true).for_each(|_| {
            count.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(count.load(Ordering::SeqCst), 1000);
    }
}
