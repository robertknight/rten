use std::ops::{Index, IndexMut, Range};

/// Wrapper around Vec which allows for unused elements at the start. Indexing
/// and slicing operate on the used portion of the Vec.
#[derive(Clone, Debug)]
pub struct VecWithOffset<T> {
    data: Vec<T>,

    /// Offset of the first used element in `data`.
    base: usize,
}

impl<T> VecWithOffset<T> {
    pub fn new(data: Vec<T>) -> VecWithOffset<T> {
        VecWithOffset { data, base: 0 }
    }

    /// Return a slice of the used portion of the wrapped Vec.
    pub fn as_slice(&self) -> &[T] {
        &self.data[self.base..]
    }

    /// Return a mutable slice of the used portion of the wrapped Vec.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[self.base..]
    }

    /// Set the indices within the vec that are used. Subsequent indexing and
    /// slicing will operator on `previous_data[range.start..range.end]`.
    pub fn set_used_range(&mut self, range: Range<usize>) {
        self.base += range.start;
        self.data.truncate(self.base + (range.end - range.start));
    }
}

impl<T> Index<usize> for VecWithOffset<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.data[self.base + index]
    }
}

impl<T> IndexMut<usize> for VecWithOffset<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[self.base + index]
    }
}
