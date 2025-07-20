use std::ops::Range;

use rayon::prelude::*;

use super::{ParIter, SplitIterator};

/// Iterator returned by [`range_chunks`].
pub struct RangeChunks {
    remainder: Range<usize>,
    chunk_size: usize,
}

impl Iterator for RangeChunks {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.remainder.is_empty() {
            let start = self.remainder.start;
            let end = (start + self.chunk_size).min(self.remainder.end);
            self.remainder.start += end - start;
            Some(start..end)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remainder.len().div_ceil(self.chunk_size);
        (len, Some(len))
    }
}

impl ExactSizeIterator for RangeChunks {}

impl DoubleEndedIterator for RangeChunks {
    fn next_back(&mut self) -> Option<Self::Item> {
        if !self.remainder.is_empty() {
            let end = self.remainder.end;
            let start = end
                .saturating_sub(self.chunk_size)
                .max(self.remainder.start);
            self.remainder.end = start;
            Some(start..end)
        } else {
            None
        }
    }
}

impl SplitIterator for RangeChunks {
    fn split_at(self, index: usize) -> (Self, Self) {
        let len = self.len();
        assert!(
            index <= len,
            "split index {} out of bounds for iterator of length {}",
            index,
            len
        );

        let offset = self.chunk_size * index;
        let split_point = self.remainder.start + offset.min(self.remainder.len());

        let left = RangeChunks {
            remainder: self.remainder.start..split_point,
            chunk_size: self.chunk_size,
        };
        let right = RangeChunks {
            remainder: split_point..self.remainder.end,
            chunk_size: self.chunk_size,
        };
        (left, right)
    }
}

impl IntoParallelIterator for RangeChunks {
    type Iter = ParIter<Self>;
    type Item = <Self as Iterator>::Item;

    fn into_par_iter(self) -> Self::Iter {
        self.into()
    }
}

impl std::iter::FusedIterator for RangeChunks {}

/// Return an iterator over sub-ranges of `range`. If `range.len()` is not a
/// multiple of `chunk_size` then the final chunk will be shorter.
#[inline]
pub fn range_chunks(range: Range<usize>, chunk_size: usize) -> RangeChunks {
    RangeChunks {
        remainder: range,
        chunk_size,
    }
}

/// Iterator returned by [`range_chunks_exact`].
pub struct RangeChunksExact {
    remainder: Range<usize>,
    chunk_size: usize,
}

impl RangeChunksExact {
    /// Return the part of the range that has not yet been visited.
    #[inline]
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
/// completes, available via [`RangeChunksExact::remainder`].
#[inline]
pub fn range_chunks_exact(range: Range<usize>, chunk_size: usize) -> RangeChunksExact {
    RangeChunksExact {
        remainder: range,
        chunk_size,
    }
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;

    use super::{range_chunks, range_chunks_exact};

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

        // Reversed
        let mut chunks = range_chunks(0..13, 5).rev();
        assert_eq!(chunks.next(), Some(8..13));
        assert_eq!(chunks.next(), Some(3..8));
        assert_eq!(chunks.next(), Some(0..3));
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.next(), None);

        // Parallel
        let chunks = range_chunks(0..100, 5);
        let sum = chunks.into_par_iter().map(|r| r.len()).sum::<usize>();
        assert_eq!(sum, 100);
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
}
