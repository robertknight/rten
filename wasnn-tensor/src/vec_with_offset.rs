use std::ops::{Deref, DerefMut, Range};

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

    /// Set the indices within the vec that are used. Subsequent indexing and
    /// slicing will operator on `previous_data[range.start..range.end]`.
    pub fn set_used_range(&mut self, range: Range<usize>) {
        self.base += range.start;
        self.data.truncate(self.base + (range.end - range.start));
    }
}

impl<T> From<Vec<T>> for VecWithOffset<T> {
    fn from(value: Vec<T>) -> Self {
        VecWithOffset::new(value)
    }
}

impl<T> From<VecWithOffset<T>> for Vec<T> {
    fn from(value: VecWithOffset<T>) -> Vec<T> {
        value.data
    }
}

impl<T> Deref for VecWithOffset<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_ref()
    }
}

impl<T> DerefMut for VecWithOffset<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut()
    }
}

impl<T> AsRef<[T]> for VecWithOffset<T> {
    /// Return a slice of the used portion of the wrapped Vec.
    fn as_ref(&self) -> &[T] {
        &self.data[self.base..]
    }
}

impl<T> AsMut<[T]> for VecWithOffset<T> {
    /// Return a mutable slice of the used portion of the wrapped Vec.
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data[self.base..]
    }
}
