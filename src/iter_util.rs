use std::ops::Range;

use rayon::prelude::*;

/// Iterator returned by [range_chunks].
pub struct RangeChunks {
    remainder: Range<usize>,
    chunk_size: usize,
}

impl Iterator for RangeChunks {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.remainder.is_empty() {
            let start = self.remainder.start;
            let end = (start + self.chunk_size).min(self.remainder.end);
            self.remainder.start += self.chunk_size;
            Some(start..end)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remainder.len().div_ceil(self.chunk_size);
        (len, Some(len))
    }
}

impl ExactSizeIterator for RangeChunks {}

impl std::iter::FusedIterator for RangeChunks {}

/// Return an iterator over sub-ranges of `range`. If `range.len()` is not a
/// multiple of `chunk_size` then the final chunk will be shorter.
pub fn range_chunks(range: Range<usize>, chunk_size: usize) -> RangeChunks {
    RangeChunks {
        remainder: range,
        chunk_size,
    }
}

pub struct RangeChunksExact {
    remainder: Range<usize>,
    chunk_size: usize,
}

impl RangeChunksExact {
    /// Return the part of the range that has not yet been visited.
    #[allow(dead_code)]
    pub fn remainder(&self) -> Range<usize> {
        self.remainder.clone()
    }
}

impl Iterator for RangeChunksExact {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remainder.len() >= self.chunk_size {
            let start = self.remainder.start;
            let end = start + self.chunk_size;
            self.remainder.start += self.chunk_size;
            Some(start..end)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remainder.len() / self.chunk_size;
        (len, Some(len))
    }
}

impl ExactSizeIterator for RangeChunksExact {}

impl std::iter::FusedIterator for RangeChunksExact {}

/// Return an iterator over sub-ranges of `range`. If `range.len()` is not a
/// multiple of `chunk_size` then there will be a remainder after iteration
/// completes, available via [RangeChunksExact::remainder].
#[allow(dead_code)]
pub fn range_chunks_exact(range: Range<usize>, chunk_size: usize) -> RangeChunksExact {
    RangeChunksExact {
        remainder: range,
        chunk_size,
    }
}

/// Wrapper around either a serial or parallel iterator, returned by
/// [MaybeParIter::maybe_par_iter].
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
/// See https://crates.io/crates/rayon-cond for a more full-featured alternative.
pub trait MaybeParIter {
    type Item;
    type ParIter: ParallelIterator<Item = Self::Item>;
    type Iter: Iterator<Item = Self::Item>;

    /// Return an iterator which executes either in serial on the current
    /// thread, or in parallel in a Rayon thread pool if `parallel` is true.
    fn maybe_par_iter(self, parallel: bool) -> MaybeParallel<Self::ParIter, Self::Iter>;
}

impl MaybeParIter for Range<usize> {
    type Item = usize;
    type ParIter = rayon::range::Iter<usize>;
    type Iter = Range<usize>;

    fn maybe_par_iter(self, parallel: bool) -> MaybeParallel<Self::ParIter, Self::Iter> {
        if parallel {
            MaybeParallel::Parallel(self.into_par_iter())
        } else {
            MaybeParallel::Serial(self)
        }
    }
}

#[macro_export]
macro_rules! unroll_loop {
    ($count:expr, $loop_var:ident, $factor: literal, $block:tt) => {
        let mut n = $count;
        let mut $loop_var = 0;
        while n >= $factor {
            for _i in 0..$factor {
                $block;
                $loop_var += 1;
            }
            n -= $factor;
        }
        while n > 0 {
            $block;

            $loop_var += 1;
            n -= 1;
        }
    };
}

#[allow(unused_imports)]
pub use unroll_loop;

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::{range_chunks, range_chunks_exact, unroll_loop, MaybeParIter};

    #[test]
    fn test_range_chunks() {
        // All chunks full.
        let mut chunks = range_chunks(0..15, 5);
        assert_eq!(chunks.size_hint(), (3, Some(3)));
        assert_eq!(chunks.next(), Some(0..5));
        assert_eq!(chunks.next(), Some(5..10));
        assert_eq!(chunks.next(), Some(10..15));
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.next(), None);

        // Smaller last chunk.
        let mut chunks = range_chunks(0..13, 5);
        assert_eq!(chunks.size_hint(), (3, Some(3)));
        assert_eq!(chunks.next(), Some(0..5));
        assert_eq!(chunks.next(), Some(5..10));
        assert_eq!(chunks.next(), Some(10..13));
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn test_range_chunks_exact() {
        // All chunks full (empty remainder).
        let mut chunks = range_chunks_exact(0..15, 5);
        assert_eq!(chunks.size_hint(), (3, Some(3)));
        assert_eq!(chunks.next(), Some(0..5));
        assert_eq!(chunks.next(), Some(5..10));
        assert_eq!(chunks.next(), Some(10..15));
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.remainder(), 15..15);

        // Non-empty remainder
        let mut chunks = range_chunks_exact(0..13, 5);
        assert_eq!(chunks.size_hint(), (2, Some(2)));
        assert_eq!(chunks.next(), Some(0..5));
        assert_eq!(chunks.next(), Some(5..10));
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.remainder(), 10..13);
    }

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

    #[test]
    fn test_unroll_loop() {
        let mut items: Vec<i32> = Vec::new();
        unroll_loop!(10, i, 4, {
            items.push(i);
        });
        assert_eq!(items, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
