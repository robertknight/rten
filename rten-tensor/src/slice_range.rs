use smallvec::SmallVec;

use std::fmt::Debug;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// Specifies a subset of a dimension to include when slicing a tensor or view.
///
/// Can be constructed from an index or range using `index_or_range.into()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SliceItem {
    /// Extract a specific index from a dimension.
    ///
    /// The number of dimensions in the sliced view will be one minus the number
    /// of dimensions sliced with an index. If the index is negative, it counts
    /// back from the end of the dimension.
    Index(isize),

    /// Include a subset of the range of the dimension.
    Range(SliceRange),
}

impl SliceItem {
    /// Return a SliceItem that extracts the full range of a dimension.
    #[inline]
    pub fn full_range() -> Self {
        (..).into()
    }

    /// Return a SliceItem that extracts part of an axis.
    #[inline]
    pub fn range(start: isize, end: Option<isize>, step: isize) -> SliceItem {
        SliceItem::Range(SliceRange::new(start, end, step))
    }

    /// Return stepped index range selected by this item from an axis with a
    /// given size.
    pub(crate) fn index_range(&self, dim_size: usize) -> IndexRange {
        let range = match *self {
            SliceItem::Range(range) => range,
            SliceItem::Index(idx) => SliceRange::new(idx, Some(idx + 1), 1),
        };
        range.index_range(dim_size)
    }
}

// This conversion exists to avoid ambiguity when slicing a tensor with a
// numeric literal of unspecified type (eg. `tensor.slice((0, 0))`). In this
// case it is ambiguous which `SliceItem::from` should be used, but the i32
// case is used if it exists.
impl From<i32> for SliceItem {
    #[inline]
    fn from(value: i32) -> Self {
        SliceItem::Index(value as isize)
    }
}

impl From<isize> for SliceItem {
    #[inline]
    fn from(value: isize) -> Self {
        SliceItem::Index(value)
    }
}

impl From<usize> for SliceItem {
    #[inline]
    fn from(value: usize) -> Self {
        SliceItem::Index(value as isize)
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
/// `[SliceItem]` array that can be used to slice a tensor.
///
/// This trait is implemented for:
///
///  - Individual indices and ranges (types satisfying `Into<SliceItem>`)
///  - Arrays of indices or ranges
///  - Tuples of indices and/or ranges
///  - `[SliceItem]` slices
///
/// Ranges can be specified using regular Rust ranges (eg. `start..end`,
/// `start..`, `..end`, `..`) or a [`SliceRange`], which extends regular Rust
/// ranges with support for steps and specifying endpoints using negative
/// values, which behaves similarly to using negative values in NumPy.
pub trait IntoSliceItems {
    type Array: AsRef<[SliceItem]>;

    fn into_slice_items(self) -> Self::Array;
}

impl<'a> IntoSliceItems for &'a [SliceItem] {
    type Array = &'a [SliceItem];

    fn into_slice_items(self) -> &'a [SliceItem] {
        self
    }
}

impl<const N: usize, T: Into<SliceItem>> IntoSliceItems for [T; N] {
    type Array = [SliceItem; N];

    fn into_slice_items(self) -> [SliceItem; N] {
        self.map(|x| x.into())
    }
}

impl<T: Into<SliceItem>> IntoSliceItems for T {
    type Array = [SliceItem; 1];

    fn into_slice_items(self) -> [SliceItem; 1] {
        [self.into()]
    }
}

impl<T1: Into<SliceItem>> IntoSliceItems for (T1,) {
    type Array = [SliceItem; 1];

    fn into_slice_items(self) -> [SliceItem; 1] {
        [self.0.into()]
    }
}

impl<T1: Into<SliceItem>, T2: Into<SliceItem>> IntoSliceItems for (T1, T2) {
    type Array = [SliceItem; 2];

    fn into_slice_items(self) -> [SliceItem; 2] {
        [self.0.into(), self.1.into()]
    }
}

impl<T1: Into<SliceItem>, T2: Into<SliceItem>, T3: Into<SliceItem>> IntoSliceItems
    for (T1, T2, T3)
{
    type Array = [SliceItem; 3];

    fn into_slice_items(self) -> [SliceItem; 3] {
        [self.0.into(), self.1.into(), self.2.into()]
    }
}

impl<T1: Into<SliceItem>, T2: Into<SliceItem>, T3: Into<SliceItem>, T4: Into<SliceItem>>
    IntoSliceItems for (T1, T2, T3, T4)
{
    type Array = [SliceItem; 4];

    fn into_slice_items(self) -> [SliceItem; 4] {
        [self.0.into(), self.1.into(), self.2.into(), self.3.into()]
    }
}

/// Dynamically sized array of [`SliceItem`]s, which avoids allocating in the
/// common case where the length is small.
pub type DynSliceItems = SmallVec<[SliceItem; 5]>;

/// Convert a slice of indices into [`SliceItem`]s.
///
/// To convert indices of a statically known length to [`SliceItem`]s, use
/// [`IntoSliceItems`] instead. This function is for the case when the length
/// is not statically known, but is assumed to likely be small.
pub fn to_slice_items<T: Clone + Into<SliceItem>>(index: &[T]) -> DynSliceItems {
    index.iter().map(|x| x.clone().into()).collect()
}

/// A range for slicing a [`Tensor`](crate::Tensor) or [`NdTensor`](crate::NdTensor).
///
/// This has two main differences from [`Range`].
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
    /// First index in range.
    pub start: isize,

    /// Last index (exclusive) in range, or None if the range extends to the
    /// end of a dimension.
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
    #[inline]
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
            .unwrap_or(if self.step > 0 { dim_size as isize } else { -1 });

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
    #[inline]
    pub fn resolve(&self, dim_size: usize) -> Option<Range<usize>> {
        let (start, end) = if self.step > 0 {
            let start = Self::offset_from_start(self.start, dim_size);
            let end = self
                .end
                .map(|end| Self::offset_from_start(end, dim_size))
                .unwrap_or(dim_size as isize);
            (start, end)
        } else {
            let start = Self::offset_from_end(self.start, dim_size);
            let end = self
                .end
                .map(|end| Self::offset_from_end(end, dim_size))
                .unwrap_or(dim_size as isize);
            (start, end)
        };

        if start >= 0 && start <= dim_size as isize && end >= 0 && end <= dim_size as isize {
            // If `end < start` this means the range is empty. Set `end ==
            // start` to have a canonical representation for this case.
            let end = end.max(start);

            Some(start as usize..end as usize)
        } else {
            None
        }
    }

    /// Return stepped index range selected by this range from an axis with a
    /// given size.
    pub(crate) fn index_range(&self, dim_size: usize) -> IndexRange {
        // Resolve range endpoints to `[0, N]`, counting forwards from the
        // start if step > 0 or backwards from the end otherwise.
        let resolved = self.resolve_clamped(dim_size);

        if self.step > 0 {
            IndexRange::new(resolved.start, resolved.end as isize, self.step)
        } else {
            IndexRange::new(
                dim_size - 1 - resolved.start,
                dim_size as isize - 1 - resolved.end as isize,
                self.step,
            )
        }
    }

    /// Resolve an index to an offset from the first index of the dimension.
    #[inline]
    fn offset_from_start(index: isize, dim_size: usize) -> isize {
        if index >= 0 {
            index
        } else {
            dim_size as isize + index
        }
    }

    /// Resolve an index to an offset from the last index of the dimension.
    #[inline]
    fn offset_from_end(index: isize, dim_size: usize) -> isize {
        if index >= 0 {
            dim_size as isize - 1 - index
        } else {
            -index - 1
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
    #[inline]
    fn from(_: RangeFull) -> SliceRange {
        SliceRange::new(0, None, 1)
    }
}

/// A range of indices with a step, which may be positive or negative.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IndexRange {
    /// Start index in [0, (dim_size - 1).max(0)]
    start: usize,

    /// End index in [-1, dim_size]
    end: isize,
    step: isize,
}

impl IndexRange {
    /// Create a new range which steps from `start` (inclusive) to `end`
    /// (exclusive) with a given step.
    ///
    /// The `step` value must not be zero.
    ///
    /// The `end` argument is signed to allow for a range which yields index 0
    /// when `step` is negative. eg. `SteppedIndexRange::new(4, -1, -1)` will
    /// yield indices `[4, 3, 2, 1, 0]`.
    fn new(start: usize, end: isize, step: isize) -> Self {
        assert!(step != 0);
        assert!(start <= isize::MAX as usize);

        IndexRange {
            start,
            end: end.max(-1),
            step,
        }
    }

    /// Return the start index.
    #[allow(unused)]
    pub fn start(&self) -> usize {
        self.start
    }

    /// Return the index that is one past the end. This is signed since this
    /// index can be -1 when `self.step() < 0`.
    #[allow(unused)]
    pub fn end(&self) -> isize {
        self.end
    }

    /// Return the increment between indices.
    #[allow(unused)]
    pub fn step(&self) -> isize {
        self.step
    }

    /// Return the number of steps along this dimension.
    pub fn steps(&self) -> usize {
        let len = if self.step > 0 {
            (self.end - self.start as isize).max(0).unsigned_abs()
        } else {
            (self.end - self.start as isize).min(0).unsigned_abs()
        };
        len.div_ceil(self.step.unsigned_abs())
    }
}

impl IntoIterator for IndexRange {
    type Item = usize;
    type IntoIter = IndexRangeIter;

    #[inline]
    fn into_iter(self) -> IndexRangeIter {
        IndexRangeIter {
            step: self.step,
            index: self.start as isize,
            remaining: self.steps(),
        }
    }
}

/// An iterator over the indices in an [`IndexRange`].
#[derive(Clone, Debug, PartialEq)]
pub struct IndexRangeIter {
    /// Next index. This is in the range [-1, N] where `N` is the size of
    /// the dimension. The values yielded by `next` are always in [0, N).
    index: isize,

    /// Remaining indices to yield.
    remaining: usize,

    step: isize,
}

impl Iterator for IndexRangeIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        let idx = self.index;
        self.index += self.step;
        self.remaining -= 1;
        Some(idx as usize)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for IndexRangeIter {}
impl std::iter::FusedIterator for IndexRangeIter {}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

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
    fn test_index_range() {
        #[derive(Debug)]
        struct Case {
            range: SliceItem,
            dim_size: usize,
            indices: Vec<usize>,
        }

        let cases = [
            // +ve step, +ve endpoints
            Case {
                range: SliceItem::range(0, Some(4), 1),
                dim_size: 6,
                indices: (0..4).collect(),
            },
            Case {
                range: SliceItem::range(2, Some(4), 1),
                dim_size: 6,
                indices: vec![2, 3],
            },
            Case {
                range: SliceItem::range(2, Some(128), 1),
                dim_size: 5,
                indices: vec![2, 3, 4],
            },
            // +ve step > 1, +ve endpoints
            Case {
                range: SliceItem::range(0, Some(5), 2),
                dim_size: 5,
                indices: vec![0, 2, 4],
            },
            // +ve step, no end
            Case {
                range: SliceItem::range(0, None, 1),
                dim_size: 6,
                indices: (0..6).collect(),
            },
            // +ve step, -ve endpoints
            Case {
                range: SliceItem::range(-1, Some(-6), 2),
                dim_size: 5,
                indices: vec![],
            },
            // -ve step, -ve endpoints
            Case {
                range: SliceItem::range(-1, Some(-128), -1),
                dim_size: 5,
                indices: vec![4, 3, 2, 1, 0],
            },
            // -ve step, no end
            Case {
                range: SliceItem::range(-1, None, -1),
                dim_size: 5,
                indices: vec![4, 3, 2, 1, 0],
            },
            // -ve step < -1, -ve endpoints
            Case {
                range: SliceItem::range(-1, Some(-6), -2),
                dim_size: 5,
                indices: vec![4, 2, 0],
            },
            // -ve step, +ve endpoints
            Case {
                range: SliceItem::range(1, Some(5), -2),
                dim_size: 5,
                indices: vec![],
            },
            // Empty range, +ve step
            Case {
                range: SliceItem::range(0, Some(0), 1),
                dim_size: 4,
                indices: vec![],
            },
            // Empty range, -ve step
            Case {
                range: SliceItem::range(0, Some(0), -1),
                dim_size: 4,
                indices: vec![],
            },
            // Single index
            Case {
                range: SliceItem::Index(2),
                dim_size: 4,
                indices: vec![2],
            },
            // Single index, out of range
            Case {
                range: SliceItem::Index(2),
                dim_size: 0,
                indices: vec![],
            },
        ];

        cases.test_each(|case| {
            let Case {
                range,
                dim_size,
                indices,
            } = case;

            let mut index_iter = range.index_range(*dim_size).into_iter();
            let size_hint = index_iter.size_hint();
            let index_vec: Vec<_> = index_iter.by_ref().collect();

            assert_eq!(size_hint, (index_vec.len(), Some(index_vec.len())));
            assert_eq!(index_vec, *indices);
            assert_eq!(index_iter.size_hint(), (0, Some(0)));
        })
    }

    #[test]
    fn test_index_range_steps() {
        #[derive(Debug)]
        struct Case {
            range: SliceRange,
            dim_size: usize,
            steps: usize,
        }

        let cases = [
            // Positive step, no end.
            Case {
                range: SliceRange::new(0, None, 1),
                dim_size: 4,
                steps: 4,
            },
            // Positive step size exceeds range length.
            Case {
                range: SliceRange::new(0, None, 5),
                dim_size: 4,
                steps: 1,
            },
            // Negative step, no end.
            Case {
                range: SliceRange::new(-1, None, -1),
                dim_size: 3,
                steps: 3,
            },
            // Negative step size exceeds range length.
            Case {
                range: SliceRange::new(1, Some(0), -2),
                dim_size: 2,
                steps: 1,
            },
        ];

        cases.test_each(|case| {
            assert_eq!(case.range.index_range(case.dim_size).steps(), case.steps);
        })
    }

    #[test]
    #[should_panic(expected = "Slice step cannot be 0")]
    fn test_slice_range_zero_step() {
        SliceRange::new(0, None, 0);
    }

    #[test]
    fn test_slice_range_resolve() {
        // +ve endpoints, +ve step
        assert_eq!(SliceRange::new(0, Some(5), 1).resolve_clamped(10), 0..5);
        assert_eq!(SliceRange::new(0, None, 1).resolve_clamped(10), 0..10);
        assert_eq!(SliceRange::new(15, Some(20), 1).resolve_clamped(10), 10..10);
        assert_eq!(SliceRange::new(15, Some(20), 1).resolve(10), None);
        assert_eq!(SliceRange::new(4, None, 1).resolve(3), None);
        assert_eq!(SliceRange::new(0, Some(10), 1).resolve(3), None);

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

        // -ve endpoints, -ve step.
        assert_eq!(SliceRange::new(-1, Some(-4), -1).resolve_clamped(3), 0..3);
        assert_eq!(SliceRange::new(-1, None, -1).resolve_clamped(2), 0..2);
    }
}
