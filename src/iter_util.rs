use std::ops::Range;

/// Iterator returned by [range_chunks].
pub struct RangeChunks {
    /// Remaining range.
    range: Range<usize>,

    chunk_size: usize,
}

impl Iterator for RangeChunks {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.range.is_empty() {
            let start = self.range.start;
            let end = (start + self.chunk_size).min(self.range.end);
            self.range.start += self.chunk_size;
            Some(start..end)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.range.len().div_ceil(self.chunk_size);
        (len, Some(len))
    }
}

impl ExactSizeIterator for RangeChunks {}

impl std::iter::FusedIterator for RangeChunks {}

/// Return an iterator over sub-ranges of `range`. If `range.len()` is not a
/// multiple of `chunk_size` then the final chunk will be shorter.
pub fn range_chunks(range: Range<usize>, chunk_size: usize) -> RangeChunks {
    RangeChunks { range, chunk_size }
}

#[cfg(test)]
mod tests {
    use super::range_chunks;

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
}
