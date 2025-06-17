use std::iter::FusedIterator;
use std::mem::transmute;
use std::ops::Range;
use std::slice;

use crate::layout::{Layout, NdLayout, OverlapPolicy, RemoveDim, merge_axes};
use crate::slice_range::SliceItem;
use crate::storage::{StorageMut, ViewData, ViewMutData};

use super::{AsView, DynLayout, MutLayout, TensorBase, TensorViewMut};

mod parallel;
pub use parallel::{ParIter, SplitIterator};

/// Borrowed reference to a tensor's data and layout. This differs from
/// [`TensorView`] in that it borrows the layout rather than having its own.
///
/// `'d` is the lifetime of the data and `'l` the lifetime of the layout.
pub(crate) struct ViewRef<'d, 'l, T, L: Layout> {
    data: ViewData<'d, T>,
    layout: &'l L,
}

impl<'d, 'l, T, L: Layout> ViewRef<'d, 'l, T, L> {
    pub(crate) fn new(data: ViewData<'d, T>, layout: &'l L) -> ViewRef<'d, 'l, T, L> {
        ViewRef { data, layout }
    }

    fn contiguous_data(&self) -> Option<&'d [T]> {
        self.layout.is_contiguous().then_some(unsafe {
            // Safety: We verified the layout is contigous
            self.data.as_slice()
        })
    }
}

impl<'d, 'l, T, L: Layout> Clone for ViewRef<'d, 'l, T, L> {
    fn clone(&self) -> ViewRef<'d, 'l, T, L> {
        ViewRef {
            data: self.data,
            layout: self.layout,
        }
    }
}

/// Mutably borrowed reference to a tensor's data and layout. This differs from
/// [`TensorViewMut`] in that it borrows the layout rather than having its own.
pub(crate) struct MutViewRef<'d, 'l, T, L: Layout> {
    data: ViewMutData<'d, T>,
    layout: &'l L,
}

impl<'d, 'l, T, L: Layout> MutViewRef<'d, 'l, T, L> {
    pub(crate) fn new(data: ViewMutData<'d, T>, layout: &'l L) -> MutViewRef<'d, 'l, T, L> {
        MutViewRef { data, layout }
    }
}

/// Tracks the iteration position within a single dimension.
#[derive(Copy, Clone, Debug, Default)]
struct IterPos {
    /// Remaining steps along this dimension before it needs to be reset.
    remaining: usize,

    /// Current index in this dimension pre-multiplied by stride.
    offset: usize,

    /// Update to `offset` for each step.
    stride: usize,

    /// Maximum value of `self.remaining`. Used when resetting position.
    max_remaining: usize,
}

impl IterPos {
    fn from_size_stride(size: usize, stride: usize) -> Self {
        let remaining = size.saturating_sub(1);
        IterPos {
            remaining,
            offset: 0,
            stride,
            max_remaining: remaining,
        }
    }

    #[inline(always)]
    fn step(&mut self) -> bool {
        if self.remaining != 0 {
            self.remaining -= 1;
            self.offset += self.stride;
            true
        } else {
            self.remaining = self.max_remaining;
            self.offset = 0;
            false
        }
    }

    /// Return the size of this dimension.
    fn size(&self) -> usize {
        // nb. The size is always > 0 since if any dim has zero size, the
        // iterator will have a length of zero.
        self.max_remaining + 1
    }

    /// Return the current index along this dimension.
    fn index(&self) -> usize {
        self.max_remaining - self.remaining
    }

    /// Set the current index along this dimension.
    fn set_index(&mut self, index: usize) {
        self.remaining = self.max_remaining - index;
        self.offset = index * self.stride;
    }
}

const INNER_NDIM: usize = 2;

/// Helper for iterating over offsets of elements in a tensor.
#[derive(Clone, Debug)]
struct IndexingIterBase {
    /// Remaining number of elements this iterator will yield.
    ///
    /// The offsets and positions in other fields are only valid if this is
    /// non-zero.
    len: usize,

    /// Component of next element offset from innermost (fastest-changing) dims.
    inner_offset: usize,

    /// Current position in innermost dims.
    inner_pos: [IterPos; INNER_NDIM],

    /// Component of next element offset from outermost (slowest-changing) dims.
    outer_offset: usize,

    /// Current position in outermost dims.
    ///
    /// Optimization note: The number of outermost dims will usually be small,
    /// so you might be tempted to use `SmallVec`. However this resulted in
    /// worse performance for `IndexingIterBase::step`, as the compiler was
    /// less likely/able to unroll iteration loops.
    outer_pos: Vec<IterPos>,
}

impl IndexingIterBase {
    /// Create an iterator over element offsets in `tensor`.
    fn new<L: Layout>(layout: &L) -> IndexingIterBase {
        // Merge axes to maximize the number of iterations that use the fast
        // path for stepping over the inner dimensions.
        let merged = merge_axes(layout.shape().as_ref(), layout.strides().as_ref());

        let inner_pos_pad = INNER_NDIM.saturating_sub(merged.len());
        let n_outer = merged.len().saturating_sub(INNER_NDIM);

        let inner_pos = std::array::from_fn(|dim| {
            let (size, stride) = if dim < inner_pos_pad {
                (1, 0)
            } else {
                merged[n_outer + dim - inner_pos_pad]
            };
            IterPos::from_size_stride(size, stride)
        });

        let outer_pos = (0..n_outer)
            .map(|i| {
                let (size, stride) = merged[i];
                IterPos::from_size_stride(size, stride)
            })
            .collect();

        IndexingIterBase {
            len: merged.iter().map(|dim| dim.0).product(),
            inner_pos,
            inner_offset: 0,
            outer_pos,
            outer_offset: 0,
        }
    }

    /// Return the offset of the next element in the storage.
    fn offset(&self) -> Option<usize> {
        if self.len > 0 {
            Some(self.outer_offset + self.inner_offset)
        } else {
            None
        }
    }

    /// Advance to the next element offset.
    ///
    /// After calling this `self.offset()` will return the offset of the next
    /// element in storage.
    #[inline(always)]
    fn step(&mut self) {
        self.len -= 1;

        // Optimistically update offset, assuming we haven't reached the
        // end of the last dimension.
        self.inner_offset += self.inner_pos[1].stride;

        // Use a fast path to step inner dimensions and fall back to the slower
        // path to step the outer dimensions only when we reach the end.
        if !self.inner_pos[1].step() {
            if !self.inner_pos[0].step() {
                self.step_outer_pos();
            }

            // `inner_offset` is the sum of `inner_pos[i].offset`. It only
            // contains two entries, and we know `inner_pos[1].offset` is zero
            // since `inner_pos[1].step()` returned false. Hence we can use
            // an assignment.
            self.inner_offset = self.inner_pos[0].offset;
        }
    }

    fn step_outer_pos(&mut self) {
        let n_outer = self.outer_pos.len();
        if n_outer > 0 {
            let mut dim = n_outer - 1;
            while !self.outer_pos[dim].step() && dim > 0 {
                dim -= 1;
            }
            self.outer_offset = self.outer_pos.iter().map(|p| p.offset).sum();
        }
    }

    fn pos(&self, dim: usize) -> IterPos {
        let outer_ndim = self.outer_pos.len();
        if dim >= outer_ndim {
            self.inner_pos[dim - outer_ndim]
        } else {
            self.outer_pos[dim]
        }
    }

    fn pos_mut(&mut self, dim: usize) -> &mut IterPos {
        let outer_ndim = self.outer_pos.len();
        if dim >= outer_ndim {
            &mut self.inner_pos[dim - outer_ndim]
        } else {
            &mut self.outer_pos[dim]
        }
    }

    /// Advance iterator by up to `n` indices.
    fn step_by(&mut self, n: usize) {
        let mut remaining = n.min(self.len);
        self.len -= remaining;

        for dim in (0..self.ndim()).rev() {
            if remaining == 0 {
                break;
            }

            let pos = self.pos_mut(dim);
            let size = pos.size();
            let new_index = pos.index() + remaining;
            pos.set_index(new_index % size);
            remaining = new_index / size;
        }

        // Update offset of next element.
        self.inner_offset = self.inner_pos.iter().map(|p| p.offset).sum();
        self.outer_offset = self.outer_pos.iter().map(|p| p.offset).sum();
    }

    fn ndim(&self) -> usize {
        self.outer_pos.len() + self.inner_pos.len()
    }

    /// Compute the storage offset of an element given a linear index into a
    /// tensor's element sequence.
    fn offset_from_linear_index(&self, index: usize) -> usize {
        let mut offset = 0;
        let mut shape_product = 1;
        for dim in (0..self.ndim()).rev() {
            let pos = self.pos(dim);
            let dim_index = (index / shape_product) % pos.size();
            shape_product *= pos.size();
            offset += dim_index * pos.stride;
        }
        offset
    }

    /// Return the offset of the last item in this iterator, or `None` if there
    /// are no more items left.
    fn step_back(&mut self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }

        // This is inefficient compared to forward iteration, but that's OK
        // because reverse iteration is not performance critical.
        let index = self.len - 1;
        let offset = self.offset_from_linear_index(index);
        self.len -= 1;

        Some(offset)
    }

    /// Split this iterator into two. The left result visits indices before
    /// `index`, the right result visits indices from `index` onwards.
    fn split_at(mut self, index: usize) -> (Self, Self) {
        assert!(self.len >= index);

        let mut right = self.clone();
        right.step_by(index);

        self.len = index;

        (self, right)
    }
}

/// Alternate implementations of [`Iter`].
///
/// When the tensor has a contiguous layout, this iterator is just a thin
/// wrapper around a slice iterator.
enum IterKind<'a, T> {
    Direct(slice::Iter<'a, T>),
    Indexing(IndexingIter<'a, T>),
}

impl<T> Clone for IterKind<'_, T> {
    fn clone(&self) -> Self {
        match self {
            IterKind::Direct(slice_iter) => IterKind::Direct(slice_iter.clone()),
            IterKind::Indexing(iter) => IterKind::Indexing((*iter).clone()),
        }
    }
}

/// Iterator over elements of a tensor, in their logical order.
pub struct Iter<'a, T> {
    iter: IterKind<'a, T>,
}

impl<'a, T> Iter<'a, T> {
    pub(super) fn new<L: Layout>(view: ViewRef<'a, '_, T, L>) -> Iter<'a, T> {
        if let Some(data) = view.contiguous_data() {
            Iter {
                iter: IterKind::Direct(data.iter()),
            }
        } else {
            Iter {
                iter: IterKind::Indexing(IndexingIter::new(view)),
            }
        }
    }
}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter {
            IterKind::Direct(ref mut iter) => iter.next(),
            IterKind::Indexing(ref mut iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.iter {
            IterKind::Direct(iter) => iter.size_hint(),
            IterKind::Indexing(iter) => iter.size_hint(),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.iter {
            IterKind::Direct(ref mut iter) => iter.nth(n),
            IterKind::Indexing(ref mut iter) => {
                iter.base.step_by(n);
                iter.next()
            }
        }
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.iter {
            IterKind::Direct(ref mut iter) => iter.next_back(),
            IterKind::Indexing(ref mut iter) => iter.next_back(),
        }
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> FusedIterator for Iter<'_, T> {}

struct IndexingIter<'a, T> {
    base: IndexingIterBase,

    /// Data buffer of the tensor
    data: ViewData<'a, T>,
}

impl<'a, T> IndexingIter<'a, T> {
    fn new<L: Layout>(view: ViewRef<'a, '_, T, L>) -> IndexingIter<'a, T> {
        IndexingIter {
            base: IndexingIterBase::new(view.layout),
            data: view.data,
        }
    }
}

impl<T> Clone for IndexingIter<'_, T> {
    fn clone(&self) -> Self {
        IndexingIter {
            base: self.base.clone(),
            data: self.data,
        }
    }
}

impl<'a, T> Iterator for IndexingIter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.base.offset()?;
        let element = unsafe {
            // Safety: See comments in Storage trait.
            self.data.get(offset).unwrap()
        };
        self.base.step();

        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }
}

impl<'a, T> DoubleEndedIterator for IndexingIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let offset = self.base.step_back()?;
        let element = unsafe {
            // Safety: See comments in Storage trait.
            self.data.get(offset).unwrap()
        };
        Some(element)
    }
}

impl<T> ExactSizeIterator for IndexingIter<'_, T> {}

impl<T> FusedIterator for IndexingIter<'_, T> {}

/// Mutable iterator over elements of a tensor.
pub struct IterMut<'a, T> {
    iter: IterMutKind<'a, T>,
}

/// Alternate implementations of `ElementsMut`.
///
/// When the tensor has a contiguous layout, this iterator is just a thin
/// wrapper around a slice iterator.
enum IterMutKind<'a, T> {
    Direct(slice::IterMut<'a, T>),
    Indexing(IndexingIterMut<'a, T>),
}

impl<'a, T> IterMut<'a, T> {
    pub(super) fn new<L: Layout>(view: MutViewRef<'a, '_, T, L>) -> IterMut<'a, T> {
        if view.layout.is_contiguous() {
            // Safety: The data is contiguous.
            let data = unsafe { view.data.to_slice_mut() };
            IterMut {
                iter: IterMutKind::Direct(data.iter_mut()),
            }
        } else {
            IterMut {
                iter: IterMutKind::Indexing(IndexingIterMut::new(view)),
            }
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter {
            IterMutKind::Direct(ref mut iter) => iter.next(),
            IterMutKind::Indexing(ref mut iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.iter {
            IterMutKind::Direct(iter) => iter.size_hint(),
            IterMutKind::Indexing(iter) => iter.size_hint(),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.iter {
            IterMutKind::Direct(ref mut iter) => iter.nth(n),
            IterMutKind::Indexing(ref mut iter) => {
                iter.base.step_by(n);
                iter.next()
            }
        }
    }
}

impl<T> DoubleEndedIterator for IterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.iter {
            IterMutKind::Direct(ref mut iter) => iter.next_back(),
            IterMutKind::Indexing(ref mut iter) => iter.next_back(),
        }
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

impl<T> FusedIterator for IterMut<'_, T> {}

struct IndexingIterMut<'a, T> {
    base: IndexingIterBase,

    /// Data buffer of the tensor
    data: ViewMutData<'a, T>,
}

impl<'a, T> IndexingIterMut<'a, T> {
    fn new<L: Layout>(view: MutViewRef<'a, '_, T, L>) -> IndexingIterMut<'a, T> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !view.layout.is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );
        IndexingIterMut {
            base: IndexingIterBase::new(view.layout),
            data: view.data,
        }
    }
}

impl<'a, T> Iterator for IndexingIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.base.offset()?;
        let element = unsafe {
            // Safety: See comments in Storage trait.
            let el = self.data.get_mut(offset).unwrap();

            // Safety: IndexingIterBase never yields the same offset more than
            // once as long as we're not broadcasting, which was checked in the
            // constructor.
            std::mem::transmute::<&'_ mut T, &'a mut T>(el)
        };
        self.base.step();
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }
}

impl<'a, T> DoubleEndedIterator for IndexingIterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let offset = self.base.step_back()?;
        let element = unsafe {
            // Safety: See comments in Storage trait.
            let el = self.data.get_mut(offset).unwrap();

            // Safety: IndexingIterBase never yields the same offset more than
            // once as long as we're not broadcasting, which was checked in the
            // constructor.
            std::mem::transmute::<&'_ mut T, &'a mut T>(el)
        };
        Some(element)
    }
}

impl<T> ExactSizeIterator for IndexingIterMut<'_, T> {}

impl<T> FusedIterator for IndexingIterMut<'_, T> {}

/// Iterator over element offsets of a tensor.
///
/// `Offsets` does not hold a reference to the tensor, allowing the tensor to
/// be modified during iteration. It is the caller's responsibilty not to modify
/// the tensor in ways that invalidate the offset sequence returned by this
/// iterator.
struct Offsets {
    base: IndexingIterBase,
}

impl Offsets {
    pub fn new<L: Layout>(layout: &L) -> Offsets {
        Offsets {
            base: IndexingIterBase::new(layout),
        }
    }
}

impl Iterator for Offsets {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.base.offset()?;
        self.base.step();
        Some(offset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.base.step_by(n);
        self.next()
    }
}

impl DoubleEndedIterator for Offsets {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.base.step_back()
    }
}

impl ExactSizeIterator for Offsets {}

impl FusedIterator for Offsets {}

/// Iterator over the ranges of a tensor's data that correspond to 1D lanes
/// along a particular dimension.
struct LaneRanges {
    /// Start offsets of each lane.
    offsets: Offsets,

    // Number of elements in each lane and gap between them.
    dim_size: usize,
    dim_stride: usize,
}

impl LaneRanges {
    fn new<L: MutLayout>(layout: &L, dim: usize) -> LaneRanges {
        let slice_starts: Vec<SliceItem> = (0..layout.ndim())
            .map(|i| {
                let end = if i == dim {
                    1.min(layout.size(i) as isize)
                } else {
                    layout.size(i) as isize
                };
                (0..end).into()
            })
            .collect();
        let (_range, sliced) = layout.slice_dyn(&slice_starts).unwrap();
        let offsets = Offsets::new(&sliced);
        LaneRanges {
            offsets,
            dim_size: layout.size(dim),
            dim_stride: layout.stride(dim),
        }
    }

    /// Return the range of storage offsets for a 1D lane where the first
    /// element is at `start_offset`.
    fn lane_offset_range(&self, start_offset: usize) -> Range<usize> {
        // nb. `dim_size` should be >= 1 here as otherwise the tensor
        // has no elements and `self.offsets` should therefore yield no
        // items.
        start_offset..start_offset + (self.dim_size - 1) * self.dim_stride + 1
    }
}

impl Iterator for LaneRanges {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Range<usize>> {
        self.offsets
            .next()
            .map(|offset| self.lane_offset_range(offset))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl DoubleEndedIterator for LaneRanges {
    fn next_back(&mut self) -> Option<Range<usize>> {
        self.offsets
            .next_back()
            .map(|offset| self.lane_offset_range(offset))
    }
}

impl ExactSizeIterator for LaneRanges {}

impl FusedIterator for LaneRanges {}

/// Iterator over 1D slices of a tensor along a target dimension of size N.
///
/// Conceptually this iterator steps through every distinct slice of a tensor
/// where a target dim is varied from 0..N and other indices are held fixed.
pub struct Lanes<'a, T> {
    data: ViewData<'a, T>,
    ranges: LaneRanges,
    size: usize,
    stride: usize,
}

/// Iterator over items in a 1D slice of a tensor.
#[derive(Clone)]
pub struct Lane<'a, T> {
    data: ViewData<'a, T>,
    index: usize,
    stride: usize,
    size: usize,
}

impl<'a, T> Lane<'a, T> {
    /// Return the remaining part of the lane as a slice, if it is contiguous.
    pub fn as_slice(&self) -> Option<&'a [T]> {
        match self.stride {
            1 => {
                let remainder = self.data.slice(self.index..self.size);
                // Safety: The stride is 1, so we know the lane is contiguous.
                Some(unsafe { remainder.as_slice() })
            }
            _ => None,
        }
    }

    /// Return the item at a given index in this lane.
    pub fn get(&self, idx: usize) -> Option<&'a T> {
        if idx < self.size {
            // Safety: `idx * self.stride` is a valid offset since `idx < self.size`.
            Some(unsafe { self.data.get_unchecked(idx * self.stride) })
        } else {
            None
        }
    }
}

impl<'a, T> Iterator for Lane<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.size {
            let index = self.index;
            self.index += 1;

            // Safety: See comments in Storage trait.
            unsafe { self.data.get(index * self.stride) }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<T> ExactSizeIterator for Lane<'_, T> {}

impl<T> FusedIterator for Lane<'_, T> {}

impl<'a, T> Lanes<'a, T> {
    /// Create an iterator which yields all possible slices over the `dim`
    /// dimension of `tensor`.
    pub(crate) fn new<L: MutLayout>(view: ViewRef<'a, '_, T, L>, dim: usize) -> Lanes<'a, T> {
        Lanes {
            data: view.data,
            ranges: LaneRanges::new(view.layout, dim),
            size: view.layout.size(dim),
            stride: view.layout.stride(dim),
        }
    }

    fn lane_for_offset_range(&self, offsets: Range<usize>) -> Lane<'a, T> {
        Lane {
            data: self.data.slice(offsets),
            index: 0,
            stride: self.stride,
            size: self.size,
        }
    }
}

impl<'a, T> Iterator for Lanes<'a, T> {
    type Item = Lane<'a, T>;

    /// Yield the next slice over the target dimension.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.ranges
            .next()
            .map(|range| self.lane_for_offset_range(range))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
    }
}

impl<T> DoubleEndedIterator for Lanes<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.ranges
            .next_back()
            .map(|range| self.lane_for_offset_range(range))
    }
}

impl<T> ExactSizeIterator for Lanes<'_, T> {}

impl<T> FusedIterator for Lanes<'_, T> {}

/// Mutable version of [`Lanes`].
///
/// Unlike [`Lanes`], this does not implement [`Iterator`] due to complications
/// in implementing this for an iterator that returns mutable references, but
/// it has a similar interface.
pub struct LanesMut<'a, T> {
    data: ViewMutData<'a, T>,
    ranges: LaneRanges,
    size: usize,
    stride: usize,
}

impl<'a, T> LanesMut<'a, T> {
    /// Create an iterator which yields all possible slices over the `dim`
    /// dimension of `view`.
    pub(crate) fn new<L: MutLayout>(view: MutViewRef<'a, '_, T, L>, dim: usize) -> LanesMut<'a, T> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !view.layout.is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );
        LanesMut {
            ranges: LaneRanges::new(view.layout, dim),
            data: view.data,
            size: view.layout.size(dim),
            stride: view.layout.stride(dim),
        }
    }

    /// Safety: Caller must ensure that this function is never called with
    /// the same offset ranges more than once, so that each lane contains an
    /// independent set of elements.
    unsafe fn lane_for_offset_range(&mut self, offsets: Range<usize>) -> LaneMut<'a, T> {
        let data = self.data.slice_mut(offsets);
        LaneMut {
            data: unsafe { transmute::<ViewMutData<'_, T>, ViewMutData<'a, T>>(data) },
            size: self.size,
            stride: self.stride,
            index: 0,
        }
    }
}

impl<'a, T> Iterator for LanesMut<'a, T> {
    type Item = LaneMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<LaneMut<'a, T>> {
        self.ranges.next().map(|range| {
            // Safety: Each iteration yields a lane that does not overlap with
            // any previous lane.
            unsafe { self.lane_for_offset_range(range) }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for LanesMut<'a, T> {}

impl<'a, T> DoubleEndedIterator for LanesMut<'a, T> {
    fn next_back(&mut self) -> Option<LaneMut<'a, T>> {
        self.ranges.next_back().map(|range| {
            // Safety: Each iteration yields a lane that does not overlap with
            // any previous lane.
            unsafe { self.lane_for_offset_range(range) }
        })
    }
}

/// Iterator over items in a 1D slice of a tensor.
pub struct LaneMut<'a, T> {
    data: ViewMutData<'a, T>,
    index: usize,
    stride: usize,
    size: usize,
}

impl<'a, T> LaneMut<'a, T> {
    /// Return the remaining part of the lane as a slice, if it is contiguous.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        match self.stride {
            1 => {
                let remainder = self.data.slice_mut(self.index..self.size);
                // Safety: The stride is 1, so we know the lane is contiguous.
                Some(unsafe { remainder.to_slice_mut() })
            }
            _ => None,
        }
    }
}

impl<'a, T> Iterator for LaneMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.size {
            let index = self.index;
            self.index += 1;
            unsafe {
                // Safety: See comments in Storage trait.
                let item = self.data.get_mut(index * self.stride);

                // Transmute to preserve lifetime of data. This is safe as we
                // yield each element only once.
                transmute::<Option<&mut T>, Option<Self::Item>>(item)
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<T> ExactSizeIterator for LaneMut<'_, T> {}

/// Base for iterators over views of the inner dimensions of a tensor, where
/// the inner dimensions have layout `L`.
struct InnerIterBase<L: Layout> {
    // Iterator over storage start offsets for each inner view. The storage
    // range for each view is `offset..offset + inner_data_len`.
    outer_offsets: Offsets,
    inner_layout: L,
    inner_data_len: usize,
}

impl<L: Layout + Clone> InnerIterBase<L> {
    fn new_impl<PL: Layout, F: Fn(&[usize], &[usize]) -> L>(
        parent_layout: &PL,
        inner_dims: usize,
        make_inner_layout: F,
    ) -> InnerIterBase<L> {
        assert!(parent_layout.ndim() >= inner_dims);
        let outer_dims = parent_layout.ndim() - inner_dims;
        let parent_shape = parent_layout.shape();
        let parent_strides = parent_layout.strides();
        let (outer_shape, inner_shape) = parent_shape.as_ref().split_at(outer_dims);
        let (outer_strides, inner_strides) = parent_strides.as_ref().split_at(outer_dims);

        let outer_layout = DynLayout::from_shape_and_strides(
            outer_shape,
            outer_strides,
            OverlapPolicy::AllowOverlap,
        )
        .unwrap();

        let inner_layout = make_inner_layout(inner_shape, inner_strides);

        InnerIterBase {
            outer_offsets: Offsets::new(&outer_layout),
            inner_data_len: inner_layout.min_data_len(),
            inner_layout,
        }
    }
}

impl<const N: usize> InnerIterBase<NdLayout<N>> {
    pub fn new<L: Layout>(parent_layout: &L) -> Self {
        Self::new_impl(parent_layout, N, |inner_shape, inner_strides| {
            let inner_shape: [usize; N] = inner_shape.try_into().unwrap();
            let inner_strides: [usize; N] = inner_strides.try_into().unwrap();
            NdLayout::from_shape_and_strides(
                inner_shape,
                inner_strides,
                // We allow overlap here, but the view that owns `parent_layout`
                // will enforce there is no overlap if it is a mutable view.
                OverlapPolicy::AllowOverlap,
            )
            .expect("failed to create layout")
        })
    }
}

impl InnerIterBase<DynLayout> {
    pub fn new_dyn<L: Layout>(parent_layout: &L, inner_dims: usize) -> Self {
        Self::new_impl(parent_layout, inner_dims, |inner_shape, inner_strides| {
            DynLayout::from_shape_and_strides(
                inner_shape,
                inner_strides,
                // We allow overlap here, but the view that owns `parent_layout`
                // will enforce there is no overlap if it is a mutable view.
                OverlapPolicy::AllowOverlap,
            )
            .expect("failed to create layout")
        })
    }
}

impl<L: Layout> Iterator for InnerIterBase<L> {
    /// Storage offset range for next view
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        self.outer_offsets
            .next()
            .map(|offset| offset..offset + self.inner_data_len)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.outer_offsets.size_hint()
    }
}

impl<L: Layout> ExactSizeIterator for InnerIterBase<L> {}

impl<L: Layout> DoubleEndedIterator for InnerIterBase<L> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.outer_offsets
            .next_back()
            .map(|offset| offset..offset + self.inner_data_len)
    }
}

/// Iterator over views of the innermost dimensions of a tensor, where the
/// tensor has element type T and the inner dimensions have layout L.
pub struct InnerIter<'a, T, L: MutLayout> {
    base: InnerIterBase<L>,
    data: ViewData<'a, T>,
}

impl<'a, T, const N: usize> InnerIter<'a, T, NdLayout<N>> {
    pub fn new<L: MutLayout>(view: TensorBase<ViewData<'a, T>, L>) -> Self {
        let base = InnerIterBase::new(&view);
        InnerIter {
            base,
            data: view.storage(),
        }
    }
}

impl<'a, T> InnerIter<'a, T, DynLayout> {
    pub fn new_dyn<L: MutLayout>(view: TensorBase<ViewData<'a, T>, L>, inner_dims: usize) -> Self {
        let base = InnerIterBase::new_dyn(&view, inner_dims);
        InnerIter {
            base,
            data: view.storage(),
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for InnerIter<'a, T, L> {
    type Item = TensorBase<ViewData<'a, T>, L>;

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|offset_range| {
            TensorBase::from_storage_and_layout(
                self.data.slice(offset_range),
                self.base.inner_layout.clone(),
            )
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}

impl<T, L: MutLayout> ExactSizeIterator for InnerIter<'_, T, L> {}

impl<T, L: MutLayout> DoubleEndedIterator for InnerIter<'_, T, L> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.base.next_back().map(|offset_range| {
            TensorBase::from_storage_and_layout(
                self.data.slice(offset_range),
                self.base.inner_layout.clone(),
            )
        })
    }
}

/// Iterator over mutable views of the innermost dimensions of a tensor, where
/// the tensor has element type T and the inner dimensions have layout L.
pub struct InnerIterMut<'a, T, L: MutLayout> {
    base: InnerIterBase<L>,
    data: ViewMutData<'a, T>,
}

impl<'a, T, const N: usize> InnerIterMut<'a, T, NdLayout<N>> {
    pub fn new<L: MutLayout>(view: TensorBase<ViewMutData<'a, T>, L>) -> Self {
        let base = InnerIterBase::new(&view);
        InnerIterMut {
            base,
            data: view.into_storage(),
        }
    }
}

impl<'a, T> InnerIterMut<'a, T, DynLayout> {
    pub fn new_dyn<L: MutLayout>(
        view: TensorBase<ViewMutData<'a, T>, L>,
        inner_dims: usize,
    ) -> Self {
        let base = InnerIterBase::new_dyn(&view, inner_dims);
        InnerIterMut {
            base,
            data: view.into_storage(),
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for InnerIterMut<'a, T, L> {
    type Item = TensorBase<ViewMutData<'a, T>, L>;

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|offset_range| {
            let storage = self.data.slice_mut(offset_range);
            let storage = unsafe {
                // Safety: The iterator was constructed from a tensor with a
                // non-overlapping layout, and no two views yielded by this
                // iterator overlap. Hence we can transmute the lifetime without
                // creating multiple mutable references to the same elements.
                std::mem::transmute::<ViewMutData<'_, T>, ViewMutData<'a, T>>(storage)
            };
            TensorBase::from_storage_and_layout(storage, self.base.inner_layout.clone())
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}

impl<T, L: MutLayout> ExactSizeIterator for InnerIterMut<'_, T, L> {}

impl<'a, T, L: MutLayout> DoubleEndedIterator for InnerIterMut<'a, T, L> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.base.next_back().map(|offset_range| {
            let storage = self.data.slice_mut(offset_range);
            let storage = unsafe {
                // Safety: Outer view is non-broadcasting, and we increment the
                // outer index each time, so returned views will not overlap.
                std::mem::transmute::<ViewMutData<'_, T>, ViewMutData<'a, T>>(storage)
            };
            TensorBase::from_storage_and_layout(storage, self.base.inner_layout.clone())
        })
    }
}

/// Iterator over slices of a tensor along an axis. See
/// [`TensorView::axis_iter`](crate::TensorView::axis_iter).
pub struct AxisIter<'a, T, L: MutLayout + RemoveDim> {
    view: TensorBase<ViewData<'a, T>, L>,
    axis: usize,
    index: usize,
    end: usize,
}

impl<'a, T, L: MutLayout + RemoveDim> AxisIter<'a, T, L> {
    pub fn new(view: &TensorBase<ViewData<'a, T>, L>, axis: usize) -> AxisIter<'a, T, L> {
        assert!(axis < view.ndim());
        AxisIter {
            view: view.clone(),
            axis,
            index: 0,
            end: view.size(axis),
        }
    }
}

impl<'a, T, L: MutLayout + RemoveDim> Iterator for AxisIter<'a, T, L> {
    type Item = TensorBase<ViewData<'a, T>, <L as RemoveDim>::Output>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let slice = self.view.index_axis(self.axis, self.index);
            self.index += 1;
            Some(slice)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.index;
        (len, Some(len))
    }
}

impl<'a, T, L: MutLayout + RemoveDim> ExactSizeIterator for AxisIter<'a, T, L> {}

impl<'a, T, L: MutLayout + RemoveDim> DoubleEndedIterator for AxisIter<'a, T, L> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let slice = self.view.index_axis(self.axis, self.end - 1);
            self.end -= 1;
            Some(slice)
        }
    }
}

/// Iterator over mutable slices of a tensor along an axis. See [`TensorViewMut::axis_iter_mut`].
pub struct AxisIterMut<'a, T, L: MutLayout + RemoveDim> {
    view: TensorBase<ViewMutData<'a, T>, L>,
    axis: usize,
    index: usize,
    end: usize,
}

impl<'a, T, L: MutLayout + RemoveDim> AxisIterMut<'a, T, L> {
    pub fn new(view: TensorBase<ViewMutData<'a, T>, L>, axis: usize) -> AxisIterMut<'a, T, L> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !view.layout().is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );
        assert!(axis < view.ndim());
        AxisIterMut {
            axis,
            index: 0,
            end: view.size(axis),
            view,
        }
    }
}

/// Mutable tensor view with one less dimension than `L`.
type SmallerMutView<'b, T, L> = TensorBase<ViewMutData<'b, T>, <L as RemoveDim>::Output>;

impl<'a, T, L: MutLayout + RemoveDim> Iterator for AxisIterMut<'a, T, L> {
    type Item = TensorBase<ViewMutData<'a, T>, <L as RemoveDim>::Output>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let index = self.index;
            self.index += 1;

            let slice = self.view.index_axis_mut(self.axis, index);

            // Promote lifetime from self -> 'a.
            //
            // Safety: This is non-broadcasting view, and we increment the index
            // each time, so returned views will not overlap.
            let view = unsafe { transmute::<SmallerMutView<'_, T, L>, Self::Item>(slice) };

            Some(view)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.index;
        (len, Some(len))
    }
}

impl<'a, T, L: MutLayout + RemoveDim> ExactSizeIterator for AxisIterMut<'a, T, L> {}

impl<'a, T, L: MutLayout + RemoveDim> DoubleEndedIterator for AxisIterMut<'a, T, L> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let index = self.end - 1;
            self.end -= 1;

            let slice = self.view.index_axis_mut(self.axis, index);

            // Promote lifetime from self -> 'a.
            //
            // Safety: This is non-broadcasting view, and we increment the index
            // each time, so returned views will not overlap.
            let view = unsafe { transmute::<SmallerMutView<'_, T, L>, Self::Item>(slice) };

            Some(view)
        }
    }
}

/// Iterator over slices of a tensor along an axis. See
/// [`TensorView::axis_chunks`](crate::TensorView::axis_chunks).
pub struct AxisChunks<'a, T, L: MutLayout> {
    remainder: Option<TensorBase<ViewData<'a, T>, L>>,
    axis: usize,
    chunk_size: usize,
}

impl<'a, T, L: MutLayout> AxisChunks<'a, T, L> {
    pub fn new(
        view: &TensorBase<ViewData<'a, T>, L>,
        axis: usize,
        chunk_size: usize,
    ) -> AxisChunks<'a, T, L> {
        assert!(chunk_size > 0, "chunk size must be > 0");
        AxisChunks {
            remainder: if view.size(axis) > 0 {
                Some(view.view())
            } else {
                None
            },
            axis,
            chunk_size,
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for AxisChunks<'a, T, L> {
    type Item = TensorBase<ViewData<'a, T>, L>;

    fn next(&mut self) -> Option<Self::Item> {
        let remainder = self.remainder.take()?;
        let chunk_len = self.chunk_size.min(remainder.size(self.axis));
        let (current, next_remainder) = remainder.split_at(self.axis, chunk_len);
        self.remainder = if next_remainder.size(self.axis) > 0 {
            Some(next_remainder)
        } else {
            None
        };
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self
            .remainder
            .as_ref()
            .map(|r| r.size(self.axis))
            .unwrap_or(0)
            .div_ceil(self.chunk_size);
        (len, Some(len))
    }
}

impl<'a, T, L: MutLayout> ExactSizeIterator for AxisChunks<'a, T, L> {}

impl<'a, T, L: MutLayout> DoubleEndedIterator for AxisChunks<'a, T, L> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let remainder = self.remainder.take()?;
        let chunk_len = self.chunk_size.min(remainder.size(self.axis));
        let (prev_remainder, current) =
            remainder.split_at(self.axis, remainder.size(self.axis) - chunk_len);
        self.remainder = if prev_remainder.size(self.axis) > 0 {
            Some(prev_remainder)
        } else {
            None
        };
        Some(current)
    }
}

/// Iterator over mutable slices of a tensor along an axis. See [`TensorViewMut::axis_chunks_mut`].
pub struct AxisChunksMut<'a, T, L: MutLayout> {
    remainder: Option<TensorBase<ViewMutData<'a, T>, L>>,
    axis: usize,
    chunk_size: usize,
}

impl<'a, T, L: MutLayout> AxisChunksMut<'a, T, L> {
    pub fn new(
        view: TensorBase<ViewMutData<'a, T>, L>,
        axis: usize,
        chunk_size: usize,
    ) -> AxisChunksMut<'a, T, L> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !view.layout().is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );
        assert!(chunk_size > 0, "chunk size must be > 0");
        AxisChunksMut {
            remainder: if view.size(axis) > 0 {
                Some(view)
            } else {
                None
            },
            axis,
            chunk_size,
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for AxisChunksMut<'a, T, L> {
    type Item = TensorBase<ViewMutData<'a, T>, L>;

    fn next(&mut self) -> Option<Self::Item> {
        let remainder = self.remainder.take()?;
        let chunk_len = self.chunk_size.min(remainder.size(self.axis));
        let (current, next_remainder) = remainder.split_at_mut(self.axis, chunk_len);
        self.remainder = if next_remainder.size(self.axis) > 0 {
            Some(next_remainder)
        } else {
            None
        };
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self
            .remainder
            .as_ref()
            .map(|r| r.size(self.axis))
            .unwrap_or(0)
            .div_ceil(self.chunk_size);
        (len, Some(len))
    }
}

impl<'a, T, L: MutLayout> ExactSizeIterator for AxisChunksMut<'a, T, L> {}

impl<'a, T, L: MutLayout> DoubleEndedIterator for AxisChunksMut<'a, T, L> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let remainder = self.remainder.take()?;
        let remainder_size = remainder.size(self.axis);
        let chunk_len = self.chunk_size.min(remainder_size);
        let (prev_remainder, current) =
            remainder.split_at_mut(self.axis, remainder_size - chunk_len);
        self.remainder = if prev_remainder.size(self.axis) > 0 {
            Some(prev_remainder)
        } else {
            None
        };
        Some(current)
    }
}

/// Call `f` on each element of `view`.
pub fn for_each_mut<T, F: Fn(&mut T)>(mut view: TensorViewMut<T>, f: F) {
    while view.ndim() < 4 {
        view.insert_axis(0);
    }

    // This could be improved by sorting dimensions of `view` in order of
    // decreasing stride. If the resulting view is contiguous, `f` can be
    // applied to the underlying data directly. Even if it isn't, this will
    // still make memory access as contiguous as possible.

    view.inner_iter_mut::<4>().for_each(|mut src| {
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                for i2 in 0..src.size(2) {
                    for i3 in 0..src.size(3) {
                        // Safety: i0..i3 are in `[0, src.size(i))`.
                        let x = unsafe { src.get_unchecked_mut([i0, i1, i2, i3]) };
                        f(x);
                    }
                }
            }
        }
    });
}

// Tests for iterator internals. Most tests of iterators are currently done via
// tests on tensor methods.
#[cfg(test)]
mod tests {
    use crate::{AsView, AxisChunks, AxisChunksMut, Lanes, LanesMut, Layout, NdTensor, Tensor};

    fn compare_reversed<T: PartialEq + std::fmt::Debug>(fwd: &[T], rev: &[T]) {
        assert_eq!(fwd.len(), rev.len());
        for (x, y) in fwd.iter().zip(rev.iter().rev()) {
            assert_eq!(x, y);
        }
    }

    fn test_double_ended_iter<I: DoubleEndedIterator>(create_iter: impl Fn() -> I)
    where
        I::Item: PartialEq + std::fmt::Debug,
    {
        let items: Vec<_> = create_iter().collect();
        let rev_items: Vec<_> = create_iter().rev().collect();
        compare_reversed(&items, &rev_items);
    }

    #[test]
    fn test_axis_chunks_rev() {
        let tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        test_double_ended_iter(|| tensor.axis_chunks(0, 1));
    }

    #[test]
    fn test_axis_chunks_empty() {
        let x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(AxisChunks::new(&x.view(), 1, 1).next().is_none());
    }

    #[test]
    #[should_panic(expected = "chunk size must be > 0")]
    fn test_axis_chunks_zero_size() {
        let x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(AxisChunks::new(&x.view(), 1, 0).next().is_none());
    }

    #[test]
    fn test_axis_chunks_mut_empty() {
        let mut x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(AxisChunksMut::new(x.view_mut(), 1, 1).next().is_none());
    }

    #[test]
    fn test_axis_chunks_mut_rev() {
        let mut tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let fwd: Vec<_> = tensor
            .axis_chunks_mut(0, 1)
            .map(|view| view.to_vec())
            .collect();
        let mut tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let rev: Vec<_> = tensor
            .axis_chunks_mut(0, 1)
            .rev()
            .map(|view| view.to_vec())
            .collect();
        compare_reversed(&fwd, &rev);
    }

    #[test]
    #[should_panic(expected = "chunk size must be > 0")]
    fn test_axis_chunks_mut_zero_size() {
        let mut x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(AxisChunksMut::new(x.view_mut(), 1, 0).next().is_none());
    }

    #[test]
    fn test_axis_iter_rev() {
        let tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        test_double_ended_iter(|| tensor.axis_iter(0));
    }

    #[test]
    fn test_axis_iter_mut_rev() {
        let mut tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let fwd: Vec<_> = tensor.axis_iter_mut(0).map(|view| view.to_vec()).collect();
        let mut tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let rev: Vec<_> = tensor
            .axis_iter_mut(0)
            .rev()
            .map(|view| view.to_vec())
            .collect();
        compare_reversed(&fwd, &rev);
    }

    #[test]
    fn test_inner_iter_rev() {
        let tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        test_double_ended_iter(|| tensor.inner_iter::<2>());
    }

    #[test]
    fn test_inner_iter_mut_rev() {
        let mut tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let fwd: Vec<_> = tensor
            .inner_iter_mut::<2>()
            .map(|view| view.to_vec())
            .collect();
        let mut tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let rev: Vec<_> = tensor
            .inner_iter_mut::<2>()
            .rev()
            .map(|view| view.to_vec())
            .collect();
        compare_reversed(&fwd, &rev);
    }

    #[test]
    fn test_lanes_empty() {
        let x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(Lanes::new(x.view().view_ref(), 0).next().is_none());
        assert!(Lanes::new(x.view().view_ref(), 1).next().is_none());
    }

    #[test]
    fn test_lanes_rev() {
        let x = NdTensor::from([[1, 2], [3, 4]]);
        let mut lanes = x.lanes(0).rev();
        assert_eq!(lanes.next().unwrap().copied().collect::<Vec<_>>(), &[2, 4]);
        assert_eq!(lanes.next().unwrap().copied().collect::<Vec<_>>(), &[1, 3]);
        assert!(lanes.next().is_none());
    }

    #[test]
    fn test_lanes_mut_rev() {
        let mut x = NdTensor::from([[1, 2], [3, 4]]);
        let mut lanes = x.lanes_mut(0).rev();
        assert_eq!(
            lanes.next().unwrap().map(|x| *x).collect::<Vec<_>>(),
            &[2, 4]
        );
        assert_eq!(
            lanes.next().unwrap().map(|x| *x).collect::<Vec<_>>(),
            &[1, 3]
        );
        assert!(lanes.next().is_none());
    }

    #[test]
    fn test_lane_as_slice() {
        // Contiguous lane
        let x = NdTensor::from([0, 1, 2]);
        let mut lane = x.lanes(0).next().unwrap();
        assert_eq!(lane.as_slice(), Some([0, 1, 2].as_slice()));
        lane.next();
        assert_eq!(lane.as_slice(), Some([1, 2].as_slice()));
        lane.next();
        lane.next();
        assert_eq!(lane.as_slice(), Some([0i32; 0].as_slice()));
        lane.next();
        assert_eq!(lane.as_slice(), Some([0i32; 0].as_slice()));

        // Non-contiguous lane
        let x = NdTensor::from([[1i32, 2], [3, 4]]);
        let lane = x.lanes(0).next().unwrap();
        assert_eq!(lane.as_slice(), None);
    }

    #[test]
    fn test_lanes_mut_empty() {
        let mut x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(LanesMut::new(x.mut_view_ref(), 0).next().is_none());
        assert!(LanesMut::new(x.mut_view_ref(), 1).next().is_none());
    }

    #[test]
    fn test_iter_step_by() {
        let tensor = Tensor::<f32>::full(&[1, 3, 16, 8], 1.);

        // Take a non-contiguous slice so we don't use the fast path for
        // contiguous tensors.
        let tensor = tensor.slice((.., .., 1.., ..));

        let sum = tensor.iter().sum::<f32>();
        for n_skip in 0..tensor.len() {
            let sum_skip = tensor.iter().skip(n_skip).sum::<f32>();
            assert_eq!(
                sum_skip,
                sum - n_skip as f32,
                "wrong sum for n_skip={}",
                n_skip
            );
        }
    }

    #[test]
    fn test_iter_broadcast() {
        let tensor = Tensor::<f32>::full(&[1], 1.);
        let broadcast = tensor.broadcast([1, 3, 16, 8]);
        assert_eq!(broadcast.iter().len(), broadcast.len());
        let count = broadcast.iter().count();
        assert_eq!(count, broadcast.len());
        let sum = broadcast.iter().sum::<f32>();
        assert_eq!(sum, broadcast.len() as f32);
    }

    #[test]
    fn test_iter_rev() {
        let tensor = NdTensor::from([[[1, 2], [3, 4]]]);

        // Reverse iteration using non-indexed iterator.
        test_double_ended_iter(|| tensor.iter());

        // Reverse iteration using non-indexed iterator.
        test_double_ended_iter(|| tensor.transposed().iter());

        // Reverse iteration using mutable indexed iterator.
        let mut tensor = NdTensor::from([[[1, 2], [3, 4]]]);
        let fwd: Vec<_> = tensor
            .permuted_mut([2, 1, 0])
            .iter_mut()
            .map(|x| *x)
            .collect();
        let rev: Vec<_> = tensor
            .permuted_mut([2, 1, 0])
            .iter_mut()
            .rev()
            .map(|x| *x)
            .collect();
        compare_reversed(&fwd, &rev);
    }

    #[test]
    #[ignore]
    fn bench_iter() {
        use crate::Layout;
        use rten_bench::run_bench;

        let tensor = std::hint::black_box(Tensor::<f32>::full(&[1, 6, 768, 64], 1.));
        let n_trials = 1000;
        let mut result = 0.;

        fn reduce<'a>(iter: impl Iterator<Item = &'a f32>) -> f32 {
            iter.sum()
        }

        // Iterate directly over data slice.
        run_bench(n_trials, Some("slice iter"), || {
            result = reduce(tensor.data().unwrap().iter());
        });
        println!("sum {}", result);

        // Use tensor iterator with contiguous tensor. This will use the fast
        // path which wraps a slice iterator.
        run_bench(n_trials, Some("contiguous iter"), || {
            result = reduce(tensor.iter());
        });
        println!("sum {}", result);

        run_bench(n_trials, Some("contiguous reverse iter"), || {
            result = reduce(tensor.iter().rev());
        });
        println!("sum {}", result);

        // Use tensor iterator with non-contiguous slice. This will fall back
        // to indexed iteration.
        let slice = tensor.slice((.., .., 1.., ..));
        assert!(!slice.is_contiguous());
        let n_trials = 1000;
        run_bench(n_trials, Some("non-contiguous iter"), || {
            result = reduce(slice.iter());
        });
        println!("sum {}", result);

        // Reverse iteration with non-contiguous slice. This is much slower
        // because it translates linear indexes into offsets using division.
        let n_trials = 100;
        run_bench(n_trials, Some("non-contiguous reverse iter"), || {
            result = reduce(slice.iter().rev());
        });
        println!("sum {}", result);
    }

    #[test]
    #[ignore]
    fn bench_inner_iter() {
        use crate::rng::XorShiftRng;
        use rten_bench::run_bench;

        let n_trials = 100;
        let mut rng = XorShiftRng::new(1234);

        // Tensor with many steps along the outer two dimensions relative to the
        // steps along the inner two dimensions. This emphasizes the overhead of
        // stepping `inner_iter`.
        let tensor = Tensor::<f32>::rand(&[512, 512, 12, 1], &mut rng);

        let mut sum = 0.;
        run_bench(n_trials, Some("inner iter"), || {
            for inner in tensor.inner_iter::<2>() {
                for i0 in 0..inner.size(0) {
                    for i1 in 0..inner.size(1) {
                        sum += inner[[i0, i1]];
                    }
                }
            }
        });
        println!("sum {}", sum);
    }
}
