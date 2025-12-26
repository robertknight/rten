//! Iterators and iterator-related traits.
mod range;
pub use range::{RangeChunks, RangeChunksExact, range_chunks, range_chunks_exact};

/// Split an iterator at a given position.
pub trait SplitIterator: ExactSizeIterator {
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
