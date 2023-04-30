use std::fmt::Debug;
use std::ops::{Range, RangeFull, RangeTo};

/// Specifies a subset of a dimension to include when slicing a tensor or view.
///
/// Can be constructed from an index or range using `index_or_range.into()`.
#[derive(Clone)]
pub enum SliceItem {
    /// View a specific index from a dimension. The number of dimensions in the
    /// sliced view will be one minus the number of dimensions sliced with an
    /// index.
    Index(usize),

    /// Include a subset of the range of the dimension.
    Range(Range<usize>),

    /// Include the full range of the dimension.
    RangeFull,
}

impl From<usize> for SliceItem {
    fn from(value: usize) -> Self {
        SliceItem::Index(value)
    }
}

impl From<Range<usize>> for SliceItem {
    fn from(value: Range<usize>) -> Self {
        SliceItem::Range(value)
    }
}

impl From<RangeFull> for SliceItem {
    fn from(_: RangeFull) -> Self {
        SliceItem::RangeFull
    }
}

/// Trait for types that can be converted into a fixed-sized sequence of
/// items (ranges or indices) for slicing a tensor.
///
/// This trait is implemented for arrays of indices and ranges (types satisfying
/// `Into<SliceItem>`) as well as tuples of heterogenous types satisfying
/// `Into<SliceItem>`.
pub trait IntoSliceItems<const N: usize> {
    fn into_slice_items(self) -> [SliceItem; N];
}

impl<const N: usize, T: Into<SliceItem>> IntoSliceItems<N> for [T; N] {
    fn into_slice_items(self) -> [SliceItem; N] {
        self.map(|x| x.into())
    }
}

impl<T1: Into<SliceItem>> IntoSliceItems<1> for (T1,) {
    fn into_slice_items(self) -> [SliceItem; 1] {
        [self.0.into()]
    }
}

impl<T1: Into<SliceItem>, T2: Into<SliceItem>> IntoSliceItems<2> for (T1, T2) {
    fn into_slice_items(self) -> [SliceItem; 2] {
        [self.0.into(), self.1.into()]
    }
}

impl<T1: Into<SliceItem>, T2: Into<SliceItem>, T3: Into<SliceItem>> IntoSliceItems<3>
    for (T1, T2, T3)
{
    fn into_slice_items(self) -> [SliceItem; 3] {
        [self.0.into(), self.1.into(), self.2.into()]
    }
}

impl<T1: Into<SliceItem>, T2: Into<SliceItem>, T3: Into<SliceItem>, T4: Into<SliceItem>>
    IntoSliceItems<4> for (T1, T2, T3, T4)
{
    fn into_slice_items(self) -> [SliceItem; 4] {
        [self.0.into(), self.1.into(), self.2.into(), self.3.into()]
    }
}

/// A range for slicing a Tensor.
///
/// This has two main differences from [Range].
///
/// - A non-zero step between indices can be specified. The step can be negative,
///   which means that the dimension should be traversed in reverse order.
/// - The `start` and `end` indexes can also be negative, in which case they
///   count backwards from the end of the array.
///
/// This system for specifying slicing and indexing follows NumPy, which in
/// turn strongly influenced slicing in ONNX.
#[derive(Clone, Copy, Debug)]
pub struct SliceRange {
    pub start: isize,
    pub end: isize,

    /// The steps between adjacent elements selected by this range. This
    /// is private so this module can enforce the invariant that it is non-zero.
    step: isize,
}

impl SliceRange {
    /// Create a new range from `start` to `end`. The `start` index is inclusive
    /// and the `end` value is exclusive.
    ///
    /// Panics if the `step` size is 0.
    pub fn new(start: isize, end: isize, step: isize) -> SliceRange {
        assert!(step != 0, "Slice step cannot be 0");
        SliceRange { start, end, step }
    }

    /// Return the number of elements that would be retained if using this range
    /// to slice a dimension of size `dim_size`.
    pub fn steps(&self, dim_size: usize) -> usize {
        let clamped = self.clamp(dim_size);

        let start_idx = Self::offset_from_start(clamped.start, dim_size);
        let end_idx = Self::offset_from_start(clamped.end, dim_size);

        if (clamped.step > 0 && end_idx <= start_idx) || (clamped.step < 0 && end_idx >= start_idx)
        {
            return 0;
        }

        let steps = if clamped.step > 0 {
            1 + (end_idx - start_idx - 1) / clamped.step
        } else {
            1 + (start_idx - end_idx - 1) / -clamped.step
        };

        steps.max(0) as usize
    }

    /// Return a copy of this range with indexes adjusted so that they are valid
    /// for a tensor dimension of size `dim_size`.
    ///
    /// Valid indexes depend on direction that the dimension is traversed
    /// (forwards if `self.step` is positive or backwards if negative). They
    /// start at the first element going in that direction and end after the
    /// last element.
    pub fn clamp(&self, dim_size: usize) -> SliceRange {
        let len = dim_size as isize;

        let min_idx;
        let max_idx;

        if self.step > 0 {
            // When traversing forwards, the range of valid +ve indexes is `[0,
            // len]` and for -ve indexes `[-len, -1]`.
            min_idx = -len;
            max_idx = len;
        } else {
            // When traversing backwards, the range of valid +ve indexes are
            // `[0, len-1]` and for -ve indexes `[-len-1, -1]`.
            min_idx = -len - 1;
            max_idx = len - 1;
        }

        SliceRange::new(
            self.start.clamp(min_idx, max_idx),
            self.end.clamp(min_idx, max_idx),
            self.step,
        )
    }

    pub fn step(&self) -> isize {
        self.step
    }

    /// Resolve the range endpoints to positive indices in `[0, dim_size)`.
    ///
    /// If `self.step` is positive, the returned range counts forwards from
    /// the first index of the dimension, otherwise it counts backwards from
    /// the last index.
    pub fn resolve(&self, dim_size: usize) -> Range<usize> {
        let clamped = self.clamp(dim_size);

        let offset_fn = if self.step > 0 {
            Self::offset_from_start
        } else {
            Self::offset_from_end
        };

        let start = offset_fn(clamped.start, dim_size) as usize;
        let end = offset_fn(clamped.end, dim_size) as usize;

        start..end
    }

    /// Resolve an index to an offset from the first index of the dimension.
    fn offset_from_start(index: isize, dim_size: usize) -> isize {
        if index >= 0 {
            index
        } else {
            dim_size as isize + index
        }
    }

    /// Resolve an index to an offset from the last index of the dimension.
    fn offset_from_end(index: isize, dim_size: usize) -> isize {
        if index >= 0 {
            dim_size as isize - 1 - index
        } else {
            dim_size as isize + index
        }
    }
}

impl<T> From<Range<T>> for SliceRange
where
    T: TryInto<isize>,
    <T as TryInto<isize>>::Error: Debug,
{
    fn from(r: Range<T>) -> SliceRange {
        let start = r.start.try_into().unwrap();
        let end = r.end.try_into().unwrap();
        SliceRange::new(start, end, 1)
    }
}

impl<T> From<RangeTo<T>> for SliceRange
where
    T: TryInto<isize>,
    <T as TryInto<isize>>::Error: Debug,
{
    fn from(r: RangeTo<T>) -> SliceRange {
        let end = r.end.try_into().unwrap();
        SliceRange::new(0, end, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::SliceRange;

    #[test]
    fn test_slice_range_resolve() {
        // +ve endpoints, +ve step
        assert_eq!(SliceRange::new(0, 5, 1).resolve(10), 0..5);
        assert_eq!(SliceRange::new(15, 20, 1).resolve(10), 10..10);

        // -ve endpoints, +ve step
        assert_eq!(SliceRange::new(-5, -1, 1).resolve(10), 5..9);
        assert_eq!(SliceRange::new(-20, -1, 1).resolve(10), 0..9);

        // +ve endpoints, -ve step
        assert_eq!(SliceRange::new(5, 0, -1).resolve(10), 4..9);
    }
}
