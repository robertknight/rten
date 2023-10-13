use std::fmt::Debug;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// Specifies a subset of a dimension to include when slicing a tensor or view.
///
/// Can be constructed from an index or range using `index_or_range.into()`.
#[derive(Clone, Debug, PartialEq)]
pub enum SliceItem {
    /// Extract a specific index from a dimension. The number of dimensions in
    /// the sliced view will be one minus the number of dimensions sliced with
    /// an index.
    Index(usize),

    /// Include a subset of the range of the dimension.
    Range(SliceRange),
}

impl SliceItem {
    /// Return a SliceItem that extracts the full range of a dimension.
    pub fn full_range() -> Self {
        (..).into()
    }

    /// Return a SliceItem that extracts part of an axis.
    pub fn range(start: isize, end: Option<isize>, step: isize) -> SliceItem {
        SliceItem::Range(SliceRange::new(start, end, step))
    }
}

impl From<usize> for SliceItem {
    fn from(value: usize) -> Self {
        SliceItem::Index(value)
    }
}

impl<R> From<R> for SliceItem
where
    R: Into<SliceRange>,
{
    fn from(value: R) -> Self {
        SliceItem::Range(value.into())
    }
}

/// Used to convert sequences of indices and/or ranges into a uniform
/// `[SliceItem; N]` array that can be used to slice a tensor.
///
/// This trait is implemented for:
///
///  - Individual indices and ranges (types satisfying `Into<SliceItem>`)
///  - Arrays of indices or ranges
///  - Tuples of indices and/or ranges
pub trait IntoSliceItems<const N: usize> {
    fn into_slice_items(self) -> [SliceItem; N];
}

impl<const N: usize, T: Into<SliceItem>> IntoSliceItems<N> for [T; N] {
    fn into_slice_items(self) -> [SliceItem; N] {
        self.map(|x| x.into())
    }
}

impl<T: Into<SliceItem>> IntoSliceItems<1> for T {
    fn into_slice_items(self) -> [SliceItem; 1] {
        [self.into()]
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

/// A range for slicing a [Tensor] or [NdTensor].
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceRange {
    pub start: isize,
    pub end: Option<isize>,

    /// The steps between adjacent elements selected by this range. This
    /// is private so this module can enforce the invariant that it is non-zero.
    step: isize,
}

impl SliceRange {
    /// Create a new range from `start` to `end`. The `start` index is inclusive
    /// and the `end` value is exclusive. If `end` is None, the range spans
    /// to the end of the dimension.
    ///
    /// Panics if the `step` size is 0.
    pub fn new(start: isize, end: Option<isize>, step: isize) -> SliceRange {
        assert!(step != 0, "Slice step cannot be 0");
        SliceRange { start, end, step }
    }

    /// Return the number of elements that would be retained if using this range
    /// to slice a dimension of size `dim_size`.
    pub fn steps(&self, dim_size: usize) -> usize {
        let clamped = self.clamp(dim_size);

        let start_idx = Self::offset_from_start(clamped.start, dim_size);
        let end_idx = clamped
            .end
            .map(|index| Self::offset_from_start(index, dim_size))
            .unwrap_or(dim_size as isize);

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
            self.end.map(|e| e.clamp(min_idx, max_idx)),
            self.step,
        )
    }

    pub fn step(&self) -> isize {
        self.step
    }

    /// Clamp this range so that it is valid for a dimension of size `dim_size`
    /// and resolve it to a positive range.
    ///
    /// This method is useful for implementing Python/NumPy-style slicing where
    /// range endpoints can be out of bounds.
    pub fn resolve_clamped(&self, dim_size: usize) -> Range<usize> {
        self.clamp(dim_size).resolve(dim_size).unwrap()
    }

    /// Resolve the range endpoints to a positive range in `[0, dim_size)`.
    ///
    /// Returns the range if resolved or None if out of bounds.
    ///
    /// If `self.step` is positive, the returned range counts forwards from
    /// the first index of the dimension, otherwise it counts backwards from
    /// the last index.
    pub fn resolve(&self, dim_size: usize) -> Option<Range<usize>> {
        let offset_fn = if self.step > 0 {
            Self::offset_from_start
        } else {
            Self::offset_from_end
        };

        let start = offset_fn(self.start, dim_size);
        let end = self
            .end
            .map(|end| offset_fn(end, dim_size))
            .unwrap_or(dim_size as isize);

        if start >= 0 && end <= dim_size as isize && start <= end {
            Some(start as usize..end as usize)
        } else {
            None
        }
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
        SliceRange::new(start, Some(end), 1)
    }
}

impl<T> From<RangeTo<T>> for SliceRange
where
    T: TryInto<isize>,
    <T as TryInto<isize>>::Error: Debug,
{
    fn from(r: RangeTo<T>) -> SliceRange {
        let end = r.end.try_into().unwrap();
        SliceRange::new(0, Some(end), 1)
    }
}

impl<T> From<RangeFrom<T>> for SliceRange
where
    T: TryInto<isize>,
    <T as TryInto<isize>>::Error: Debug,
{
    fn from(r: RangeFrom<T>) -> SliceRange {
        let start = r.start.try_into().unwrap();
        SliceRange::new(start, None, 1)
    }
}

impl From<RangeFull> for SliceRange {
    fn from(_: RangeFull) -> SliceRange {
        SliceRange::new(0, None, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::{IntoSliceItems, SliceItem, SliceRange};

    #[test]
    fn test_into_slice_items() {
        let x = (42).into_slice_items();
        assert_eq!(x, [SliceItem::Index(42)]);

        let x = (2..5).into_slice_items();
        assert_eq!(x, [SliceItem::Range((2..5).into())]);

        let x = (..5).into_slice_items();
        assert_eq!(x, [SliceItem::Range((0..5).into())]);

        let x = (3..).into_slice_items();
        assert_eq!(x, [SliceItem::Range((3..).into())]);

        let x = [1].into_slice_items();
        assert_eq!(x, [SliceItem::Index(1)]);
        let x = [1, 2].into_slice_items();
        assert_eq!(x, [SliceItem::Index(1), SliceItem::Index(2)]);

        let x = (0, 1..2, ..).into_slice_items();
        assert_eq!(
            x,
            [
                SliceItem::Index(0),
                SliceItem::Range((1..2).into()),
                SliceItem::full_range()
            ]
        );
    }

    #[test]
    fn test_slice_range_resolve() {
        // +ve endpoints, +ve step
        assert_eq!(SliceRange::new(0, Some(5), 1).resolve_clamped(10), 0..5);
        assert_eq!(SliceRange::new(0, None, 1).resolve_clamped(10), 0..10);
        assert_eq!(SliceRange::new(15, Some(20), 1).resolve_clamped(10), 10..10);
        assert_eq!(SliceRange::new(15, Some(20), 1).resolve(10), None);

        // -ve endpoints, +ve step
        assert_eq!(SliceRange::new(-5, Some(-1), 1).resolve_clamped(10), 5..9);
        assert_eq!(SliceRange::new(-20, Some(-1), 1).resolve_clamped(10), 0..9);
        assert_eq!(SliceRange::new(-20, Some(-1), 1).resolve(10), None);
        assert_eq!(SliceRange::new(-5, None, 1).resolve_clamped(10), 5..10);

        // +ve endpoints, -ve step.
        //
        // Note the returned ranges count backwards from the end of the
        // dimension.
        assert_eq!(SliceRange::new(5, Some(0), -1).resolve_clamped(10), 4..9);
        assert_eq!(SliceRange::new(5, None, -1).resolve_clamped(10), 4..10);
        assert_eq!(SliceRange::new(9, None, -1).resolve_clamped(10), 0..10);
    }
}
