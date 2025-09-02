use std::iter::FusedIterator;
use std::mem::transmute;
use std::ops::Range;

use rten_base::iter::SplitIterator;

use super::{
    AsView, DynLayout, MutLayout, NdTensorView, NdTensorViewMut, TensorBase, TensorViewMut,
};
use crate::layout::{Layout, NdLayout, OverlapPolicy, RemoveDim, merge_axes};
use crate::storage::{StorageMut, ViewData, ViewMutData};

mod parallel;

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

/// Iterator over offsets of a tensor's elements.
#[derive(Clone, Debug)]
struct OffsetsBase {
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

impl OffsetsBase {
    /// Create an iterator over element offsets in `tensor`.
    fn new<L: Layout>(layout: &L) -> OffsetsBase {
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

        OffsetsBase {
            len: merged.iter().map(|dim| dim.0).product(),
            inner_pos,
            inner_offset: 0,
            outer_pos,
            outer_offset: 0,
        }
    }

    /// Step in the outer dimensions.
    ///
    /// Returns `true` if the position was advanced or `false` if the end was
    /// reached.
    fn step_outer_pos(&mut self) -> bool {
        let mut done = self.outer_pos.is_empty();
        for (i, dim) in self.outer_pos.iter_mut().enumerate().rev() {
            if dim.step() {
                break;
            } else if i == 0 {
                done = true;
            }
        }
        self.outer_offset = self.outer_pos.iter().map(|p| p.offset).sum();
        !done
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

    /// Truncate this iterator so that it yields at most `len` elements.
    fn truncate(&mut self, len: usize) {
        // We adjust `self.len` here but not any of the iteration positions.
        // This means that methods like `next` and `fold` must always check
        // `self.len` before each step.
        self.len = self.len.min(len);
    }
}

impl Iterator for OffsetsBase {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        let offset = self.outer_offset + self.inner_offset;

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

        Some(offset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, usize) -> B,
    {
        // Iter positions are only valid if `self.len > 0`.
        if self.len == 0 {
            return init;
        }

        let mut accum = init;
        'outer: loop {
            for i0 in self.inner_pos[0].index()..self.inner_pos[0].size() {
                for i1 in self.inner_pos[1].index()..self.inner_pos[1].size() {
                    let inner_offset =
                        i0 * self.inner_pos[0].stride + i1 * self.inner_pos[1].stride;
                    accum = f(accum, self.outer_offset + inner_offset);

                    self.len -= 1;
                    if self.len == 0 {
                        break 'outer;
                    }
                }
                self.inner_pos[1].set_index(0);
            }
            self.inner_pos[0].set_index(0);

            if !self.step_outer_pos() {
                break;
            }
        }

        accum
    }
}

impl ExactSizeIterator for OffsetsBase {}

impl DoubleEndedIterator for OffsetsBase {
    fn next_back(&mut self) -> Option<usize> {
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
}

impl SplitIterator for OffsetsBase {
    /// Split this iterator into two. The left result visits indices before
    /// `index`, the right result visits indices from `index` onwards.
    fn split_at(mut self, index: usize) -> (Self, Self) {
        assert!(self.len >= index);

        let mut right = self.clone();
        OffsetsBase::step_by(&mut right, index);

        self.truncate(index);

        (self, right)
    }
}

/// Iterator over elements of a tensor, in their logical order.
pub struct Iter<'a, T> {
    offsets: Offsets,
    data: ViewData<'a, T>,
}

impl<'a, T> Iter<'a, T> {
    pub(super) fn new<L: Layout + Clone>(view: TensorBase<ViewData<'a, T>, L>) -> Iter<'a, T> {
        Iter {
            offsets: Offsets::new(view.layout()),
            data: view.storage(),
        }
    }
}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter {
            offsets: self.offsets.clone(),
            data: self.data,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next()?;

        // Safety: Offset is valid for data length.
        Some(unsafe { self.data.get_unchecked(offset) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let offset = self.offsets.nth(n)?;

        // Safety: Offset is valid for data length.
        Some(unsafe { self.data.get_unchecked(offset) })
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.offsets.fold(init, |acc, offset| {
            // Safety: Offset is valid for data length.
            let item = unsafe { self.data.get_unchecked(offset) };
            f(acc, item)
        })
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next_back()?;

        // Safety: Offset is valid for data length.
        Some(unsafe { self.data.get_unchecked(offset) })
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> FusedIterator for Iter<'_, T> {}

/// Wrapper around [`transmute`] which allows transmuting only the lifetime,
/// not the type, of a reference.
unsafe fn transmute_lifetime_mut<'a, 'b, T>(x: &'a mut T) -> &'b mut T {
    unsafe { transmute::<&'a mut T, &'b mut T>(x) }
}

/// Mutable iterator over elements of a tensor.
pub struct IterMut<'a, T> {
    offsets: Offsets,
    data: ViewMutData<'a, T>,
}

impl<'a, T> IterMut<'a, T> {
    pub(super) fn new<L: Layout + Clone>(
        view: TensorBase<ViewMutData<'a, T>, L>,
    ) -> IterMut<'a, T> {
        IterMut {
            offsets: Offsets::new(view.layout()),
            data: view.into_storage(),
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next()?;

        // Safety: Offset is valid for data length, `offsets.next` yields each
        // offset only once.
        Some(unsafe { transmute_lifetime_mut(self.data.get_unchecked_mut(offset)) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let offset = self.offsets.nth(n)?;

        // Safety: Offset is valid for data length, `offsets.next` yields each
        // offset only once.
        Some(unsafe { transmute_lifetime_mut(self.data.get_unchecked_mut(offset)) })
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.offsets.fold(init, |acc, offset| {
            // Safety: Offset is valid for data length, `offsets.fold` yields
            // each offset only once.
            let item = unsafe { transmute_lifetime_mut(self.data.get_unchecked_mut(offset)) };
            f(acc, item)
        })
    }
}

impl<T> DoubleEndedIterator for IterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next_back()?;

        // Safety: Offset is valid for data length, `offsets.next` yields each
        // offset only once.
        Some(unsafe { transmute_lifetime_mut(self.data.get_unchecked_mut(offset)) })
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

impl<T> FusedIterator for IterMut<'_, T> {}

#[derive(Clone)]
enum OffsetsKind {
    Range(Range<usize>),
    Indexing(OffsetsBase),
}

/// Iterator over element offsets of a tensor.
///
/// `Offsets` does not hold a reference to the tensor, allowing the tensor to
/// be modified during iteration. It is the caller's responsibilty not to modify
/// the tensor in ways that invalidate the offset sequence returned by this
/// iterator.
#[derive(Clone)]
struct Offsets {
    base: OffsetsKind,
}

impl Offsets {
    pub fn new<L: Layout>(layout: &L) -> Offsets {
        Offsets {
            base: if layout.is_contiguous() {
                OffsetsKind::Range(0..layout.min_data_len())
            } else {
                OffsetsKind::Indexing(OffsetsBase::new(layout))
            },
        }
    }
}

impl Iterator for Offsets {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.base {
            OffsetsKind::Range(r) => r.next(),
            OffsetsKind::Indexing(base) => base.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.base {
            OffsetsKind::Range(r) => r.size_hint(),
            OffsetsKind::Indexing(base) => (base.len, Some(base.len)),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.base {
            OffsetsKind::Range(r) => r.nth(n),
            OffsetsKind::Indexing(base) => {
                base.step_by(n);
                self.next()
            }
        }
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        match self.base {
            OffsetsKind::Range(r) => r.fold(init, f),
            OffsetsKind::Indexing(base) => base.fold(init, f),
        }
    }
}

impl DoubleEndedIterator for Offsets {
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.base {
            OffsetsKind::Range(r) => r.next_back(),
            OffsetsKind::Indexing(base) => base.next_back(),
        }
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
    fn new<L: Layout + RemoveDim>(layout: &L, dim: usize) -> LaneRanges {
        // If the layout is empty (has any zero-sized dims), we need to make
        // sure that `offsets` is as well.
        let offsets = if layout.is_empty() {
            Offsets::new(layout)
        } else {
            let other_dims = layout.remove_dim(dim);
            Offsets::new(&other_dims)
        };

        LaneRanges {
            offsets,
            dim_size: layout.size(dim),
            dim_stride: layout.stride(dim),
        }
    }

    /// Return the range of storage offsets for a 1D lane where the first
    /// element is at `start_offset`.
    fn lane_offset_range(&self, start_offset: usize) -> Range<usize> {
        lane_offsets(start_offset, self.dim_size, self.dim_stride)
    }
}

fn lane_offsets(start_offset: usize, size: usize, stride: usize) -> Range<usize> {
    start_offset..start_offset + (size - 1) * stride + 1
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

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let Self {
            offsets,
            dim_size,
            dim_stride,
        } = self;

        offsets.fold(init, |acc, offset| {
            f(acc, lane_offsets(offset, dim_size, dim_stride))
        })
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
    lane_layout: NdLayout<1>,
}

/// Iterator over items in a 1D slice of a tensor.
#[derive(Clone, Debug)]
pub struct Lane<'a, T> {
    view: NdTensorView<'a, T, 1>,
    index: usize,
}

impl<'a, T> Lane<'a, T> {
    /// Return the remaining part of the lane as a slice, if it is contiguous.
    pub fn as_slice(&self) -> Option<&'a [T]> {
        self.view.data().map(|data| &data[self.index..])
    }

    /// Return the item at a given index in this lane.
    pub fn get(&self, idx: usize) -> Option<&'a T> {
        self.view.get([idx])
    }

    /// Return the entire lane as a 1D tensor view.
    pub fn as_view(&self) -> NdTensorView<'a, T, 1> {
        self.view
    }
}

impl<'a, T> From<NdTensorView<'a, T, 1>> for Lane<'a, T> {
    fn from(val: NdTensorView<'a, T, 1>) -> Self {
        Lane {
            view: val,
            index: 0,
        }
    }
}

impl<'a, T> Iterator for Lane<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.view.len() {
            let index = self.index;
            self.index += 1;

            // Safety: Index is in bounds for axis 0.
            Some(unsafe { self.view.get_unchecked([index]) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.view.size(0);
        (size, Some(size))
    }
}

impl<T> ExactSizeIterator for Lane<'_, T> {}

impl<T> FusedIterator for Lane<'_, T> {}

impl<T: PartialEq> PartialEq<Lane<'_, T>> for Lane<'_, T> {
    fn eq(&self, other: &Lane<'_, T>) -> bool {
        self.view.slice(self.index..) == other.view.slice(other.index..)
    }
}

impl<T: PartialEq> PartialEq<Lane<'_, T>> for LaneMut<'_, T> {
    fn eq(&self, other: &Lane<'_, T>) -> bool {
        self.view.slice(self.index..) == other.view.slice(other.index..)
    }
}

impl<'a, T> Lanes<'a, T> {
    /// Create an iterator which yields all possible slices over the `dim`
    /// dimension of `tensor`.
    pub(crate) fn new<L: Layout + RemoveDim + Clone>(
        view: TensorBase<ViewData<'a, T>, L>,
        dim: usize,
    ) -> Lanes<'a, T> {
        let size = view.size(dim);
        let stride = view.stride(dim);
        let lane_layout =
            NdLayout::from_shape_and_strides([size], [stride], OverlapPolicy::AllowOverlap)
                .unwrap();
        Lanes {
            data: view.storage(),
            ranges: LaneRanges::new(view.layout(), dim),
            lane_layout,
        }
    }
}

fn lane_for_offset_range<T>(
    data: ViewData<T>,
    layout: NdLayout<1>,
    offsets: Range<usize>,
) -> Lane<T> {
    let view = NdTensorView::from_storage_and_layout(data.slice(offsets), layout);
    Lane { view, index: 0 }
}

impl<'a, T> Iterator for Lanes<'a, T> {
    type Item = Lane<'a, T>;

    /// Yield the next slice over the target dimension.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.ranges
            .next()
            .map(|range| lane_for_offset_range(self.data, self.lane_layout, range))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.ranges.fold(init, |acc, offsets| {
            let lane = lane_for_offset_range(self.data, self.lane_layout, offsets);
            f(acc, lane)
        })
    }
}

impl<T> DoubleEndedIterator for Lanes<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.ranges
            .next_back()
            .map(|range| lane_for_offset_range(self.data, self.lane_layout, range))
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
    lane_layout: NdLayout<1>,
}

impl<'a, T> LanesMut<'a, T> {
    /// Create an iterator which yields all possible slices over the `dim`
    /// dimension of `view`.
    pub(crate) fn new<L: Layout + RemoveDim + Clone>(
        view: TensorBase<ViewMutData<'a, T>, L>,
        dim: usize,
    ) -> LanesMut<'a, T> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !view.is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );

        let size = view.size(dim);
        let stride = view.stride(dim);

        // We allow overlap here to handle the case where the stride is zero,
        // but the tensor is empty. If the tensor was not empty, the assert above
        // would have caught this.
        let lane_layout =
            NdLayout::from_shape_and_strides([size], [stride], OverlapPolicy::AllowOverlap)
                .unwrap();

        LanesMut {
            ranges: LaneRanges::new(view.layout(), dim),
            data: view.into_storage(),
            lane_layout,
        }
    }
}

impl<'a, T> Iterator for LanesMut<'a, T> {
    type Item = LaneMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<LaneMut<'a, T>> {
        self.ranges.next().map(|offsets| {
            // Safety: Offsets range length is sufficient for layout, elements
            // in each lane do not overlap.
            unsafe {
                LaneMut::from_storage_layout(self.data.to_view_slice_mut(offsets), self.lane_layout)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.ranges.fold(init, |acc, offsets| {
            // Safety: Offsets range length is sufficient for layout, elements
            // in each lane do not overlap.
            let lane = unsafe {
                LaneMut::from_storage_layout(self.data.to_view_slice_mut(offsets), self.lane_layout)
            };
            f(acc, lane)
        })
    }
}

impl<'a, T> ExactSizeIterator for LanesMut<'a, T> {}

impl<'a, T> DoubleEndedIterator for LanesMut<'a, T> {
    fn next_back(&mut self) -> Option<LaneMut<'a, T>> {
        self.ranges.next_back().map(|offsets| {
            // Safety: Offsets range length is sufficient for layout, elements
            // in each lane do not overlap.
            unsafe {
                LaneMut::from_storage_layout(self.data.to_view_slice_mut(offsets), self.lane_layout)
            }
        })
    }
}

/// Iterator over items in a 1D slice of a tensor.
#[derive(Debug)]
pub struct LaneMut<'a, T> {
    view: NdTensorViewMut<'a, T, 1>,
    index: usize,
}

impl<'a, T> LaneMut<'a, T> {
    /// Create a new lane given the storage and layout.
    ///
    /// # Safety
    ///
    /// - Caller must ensure that no two lanes are created which overlap.
    /// - Storage length must exceed `layout.min_data_len()`.
    unsafe fn from_storage_layout(data: ViewMutData<'a, T>, layout: NdLayout<1>) -> Self {
        let view = unsafe {
            // Safety: Caller promises that each call uses the offset ranges for
            // a different lane and that the range length is sufficient for the
            // lane's size and stride.
            NdTensorViewMut::from_storage_and_layout_unchecked(data, layout)
        };
        LaneMut { view, index: 0 }
    }

    /// Return the remaining part of the lane as a slice, if it is contiguous.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.view.data_mut().map(|data| &mut data[self.index..])
    }

    /// Return the entire lane as a mutable 1D tensor view.
    pub fn into_view(self) -> NdTensorViewMut<'a, T, 1> {
        self.view
    }
}

impl<'a, T> Iterator for LaneMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.view.size(0) {
            let index = self.index;
            self.index += 1;
            let item = unsafe { self.view.get_unchecked_mut([index]) };

            // Transmute to preserve lifetime of data. This is safe as we
            // yield each element only once.
            Some(unsafe { transmute::<&mut T, Self::Item>(item) })
        } else {
            None
        }
    }

    #[inline]
    fn nth(&mut self, nth: usize) -> Option<Self::Item> {
        self.index = (self.index + nth).min(self.view.size(0));
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.view.size(0);
        (size, Some(size))
    }
}

impl<T> ExactSizeIterator for LaneMut<'_, T> {}

impl<T: PartialEq> PartialEq<LaneMut<'_, T>> for LaneMut<'_, T> {
    fn eq(&self, other: &LaneMut<'_, T>) -> bool {
        self.view.slice(self.index..) == other.view.slice(other.index..)
    }
}

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

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.outer_offsets.fold(init, |acc, offset| {
            f(acc, offset..offset + self.inner_data_len)
        })
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
pub struct InnerIter<'a, T, L: Layout> {
    base: InnerIterBase<L>,
    data: ViewData<'a, T>,
}

impl<'a, T, const N: usize> InnerIter<'a, T, NdLayout<N>> {
    pub fn new<L: Layout + Clone>(view: TensorBase<ViewData<'a, T>, L>) -> Self {
        let base = InnerIterBase::new(&view);
        InnerIter {
            base,
            data: view.storage(),
        }
    }
}

impl<'a, T> InnerIter<'a, T, DynLayout> {
    pub fn new_dyn<L: Layout + Clone>(
        view: TensorBase<ViewData<'a, T>, L>,
        inner_dims: usize,
    ) -> Self {
        let base = InnerIterBase::new_dyn(&view, inner_dims);
        InnerIter {
            base,
            data: view.storage(),
        }
    }
}

impl<'a, T, L: Layout + Clone> Iterator for InnerIter<'a, T, L> {
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

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let inner_layout = self.base.inner_layout.clone();
        self.base.fold(init, |acc, offset_range| {
            let item = TensorBase::from_storage_and_layout(
                self.data.slice(offset_range),
                inner_layout.clone(),
            );
            f(acc, item)
        })
    }
}

impl<T, L: Layout + Clone> ExactSizeIterator for InnerIter<'_, T, L> {}

impl<T, L: Layout + Clone> DoubleEndedIterator for InnerIter<'_, T, L> {
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
pub struct InnerIterMut<'a, T, L: Layout> {
    base: InnerIterBase<L>,
    data: ViewMutData<'a, T>,
}

impl<'a, T, const N: usize> InnerIterMut<'a, T, NdLayout<N>> {
    pub fn new<L: Layout>(view: TensorBase<ViewMutData<'a, T>, L>) -> Self {
        let base = InnerIterBase::new(&view);
        InnerIterMut {
            base,
            data: view.into_storage(),
        }
    }
}

impl<'a, T> InnerIterMut<'a, T, DynLayout> {
    pub fn new_dyn<L: Layout>(view: TensorBase<ViewMutData<'a, T>, L>, inner_dims: usize) -> Self {
        let base = InnerIterBase::new_dyn(&view, inner_dims);
        InnerIterMut {
            base,
            data: view.into_storage(),
        }
    }
}

impl<'a, T, L: Layout + Clone> Iterator for InnerIterMut<'a, T, L> {
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

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let inner_layout = self.base.inner_layout.clone();
        self.base.fold(init, |acc, offset_range| {
            let storage = self.data.slice_mut(offset_range);
            let storage = unsafe {
                // Safety: The iterator was constructed from a tensor with a
                // non-overlapping layout, and no two views yielded by this
                // iterator overlap. Hence we can transmute the lifetime without
                // creating multiple mutable references to the same elements.
                std::mem::transmute::<ViewMutData<'_, T>, ViewMutData<'a, T>>(storage)
            };
            let item = TensorBase::from_storage_and_layout(storage, inner_layout.clone());
            f(acc, item)
        })
    }
}

impl<T, L: Layout + Clone> ExactSizeIterator for InnerIterMut<'_, T, L> {}

impl<'a, T, L: Layout + Clone> DoubleEndedIterator for InnerIterMut<'a, T, L> {
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
pub struct AxisIter<'a, T, L: Layout + RemoveDim> {
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
pub struct AxisIterMut<'a, T, L: Layout + RemoveDim> {
    view: TensorBase<ViewMutData<'a, T>, L>,
    axis: usize,
    index: usize,
    end: usize,
}

impl<'a, T, L: Layout + RemoveDim + Clone> AxisIterMut<'a, T, L> {
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
    use crate::{
        AsView, AxisChunks, AxisChunksMut, Lanes, LanesMut, Layout, NdLayout, NdTensor, Tensor,
    };

    fn compare_reversed<T: PartialEq + std::fmt::Debug>(fwd: &[T], rev: &[T]) {
        assert_eq!(fwd.len(), rev.len());
        for (x, y) in fwd.iter().zip(rev.iter().rev()) {
            assert_eq!(x, y);
        }
    }

    /// Apply a standard set of tests to an iterator.
    fn test_iterator<I: Iterator + ExactSizeIterator + DoubleEndedIterator>(
        create_iter: impl Fn() -> I,
        expected: &[I::Item],
    ) where
        I::Item: PartialEq + std::fmt::Debug,
    {
        let iter = create_iter();

        let (min_len, max_len) = iter.size_hint();
        let items: Vec<_> = iter.collect();

        assert_eq!(&items, expected);

        // Test ExactSizeIterator via `size_hint`.
        assert_eq!(min_len, items.len(), "incorrect size lower bound");
        assert_eq!(max_len, Some(items.len()), "incorrect size upper bound");

        // Test DoubleEndedIterator via `rev`.
        let rev_items: Vec<_> = create_iter().rev().collect();
        compare_reversed(&items, &rev_items);

        // Test FusedIterator.
        let mut iter = create_iter();
        for _x in &mut iter { /* noop */ }
        assert_eq!(iter.next(), None);

        // Test fold.
        let mut fold_items = Vec::new();
        let mut idx = 0;
        create_iter().fold(0, |acc, item| {
            assert_eq!(acc, idx);
            fold_items.push(item);
            idx += 1;
            idx
        });
        assert_eq!(items, fold_items);
    }

    /// A collection that can be mutably iterated over multiple times.
    ///
    /// We use a different pattern for testing mutable iterators to avoid
    /// restrictions on values returned from `FnMut` closures.
    trait MutIterable {
        type Iter<'a>: Iterator + ExactSizeIterator + DoubleEndedIterator
        where
            Self: 'a;

        fn iter_mut(&mut self) -> Self::Iter<'_>;
    }

    /// Apply a standard set of tests to a mutable iterator.
    fn test_mut_iterator<M, T>(mut iterable: M, expected: &[T])
    where
        M: MutIterable,
        T: std::fmt::Debug,
        for<'a> <M::Iter<'a> as Iterator>::Item: std::fmt::Debug + PartialEq + PartialEq<T>,
    {
        // Test Iterator and ExactSizeIterator.
        {
            let iter = iterable.iter_mut();
            let (min_len, max_len) = iter.size_hint();
            let items: Vec<_> = iter.collect();

            // Test `next`
            assert_eq!(items, expected);

            // Test `size_hint`
            assert_eq!(min_len, items.len(), "incorrect size lower bound");
            assert_eq!(max_len, Some(items.len()), "incorrect size upper bound");
        }

        // Test FusedIterator.
        {
            let mut iter = iterable.iter_mut();
            for _x in &mut iter { /* noop */ }
            assert!(iter.next().is_none());
        }

        // Test DoubleEndedIterator via `rev`.
        //
        // We use `format!` here to convert mutable references into comparable
        // items that have no connection to the mutable references yielded by
        // the iterator. Ideally this should be replaced by a clone or something.
        {
            let items: Vec<_> = iterable.iter_mut().map(|x| format!("{:?}", x)).collect();
            let rev_items: Vec<_> = iterable
                .iter_mut()
                .rev()
                .map(|x| format!("{:?}", x))
                .collect();
            compare_reversed(&items, &rev_items);
        }

        // Test fold.
        {
            let items: Vec<_> = iterable.iter_mut().map(|x| format!("{:?}", x)).collect();
            let mut fold_items = Vec::new();
            let mut idx = 0;
            iterable.iter_mut().fold(0, |acc, item| {
                assert_eq!(acc, idx);
                fold_items.push(format!("{:?}", item));
                idx += 1;
                idx
            });
            assert_eq!(items, fold_items);
        }
    }

    #[test]
    fn test_axis_chunks() {
        let tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        test_iterator(
            || tensor.axis_chunks(0, 1),
            &[tensor.slice(0..1), tensor.slice(1..2)],
        );
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
    fn test_axis_iter() {
        let tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        test_iterator(|| tensor.axis_iter(0), &[tensor.slice(0), tensor.slice(1)]);
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
    fn test_inner_iter() {
        let tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        test_iterator(
            || tensor.inner_iter::<2>(),
            &[tensor.slice(0), tensor.slice(1)],
        );
    }

    #[test]
    fn test_inner_iter_mut() {
        struct InnerIterMutTest(NdTensor<i32, 3>);

        impl MutIterable for InnerIterMutTest {
            type Iter<'a> = super::InnerIterMut<'a, i32, NdLayout<2>>;

            fn iter_mut(&mut self) -> Self::Iter<'_> {
                self.0.inner_iter_mut::<2>()
            }
        }

        let tensor = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        test_mut_iterator(
            InnerIterMutTest(tensor.clone()),
            &[tensor.slice(0), tensor.slice(1)],
        );
    }

    #[test]
    fn test_lanes() {
        let x = NdTensor::from([[1, 2], [3, 4]]);
        test_iterator(
            || x.lanes(0),
            &[x.slice((.., 0)).into(), x.slice((.., 1)).into()],
        );
        test_iterator(|| x.lanes(1), &[x.slice(0).into(), x.slice(1).into()]);
    }

    #[test]
    fn test_lanes_empty() {
        let x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(Lanes::new(x.view().view_ref(), 0).next().is_none());
        assert!(Lanes::new(x.view().view_ref(), 1).next().is_none());
    }

    #[test]
    fn test_lanes_mut() {
        use super::Lane;

        struct LanesMutTest(NdTensor<i32, 2>);

        impl MutIterable for LanesMutTest {
            type Iter<'a> = super::LanesMut<'a, i32>;

            fn iter_mut(&mut self) -> Self::Iter<'_> {
                self.0.lanes_mut(0)
            }
        }

        let tensor = NdTensor::from([[1, 2], [3, 4]]);
        test_mut_iterator::<_, Lane<i32>>(
            LanesMutTest(tensor.clone()),
            &[
                Lane::from(tensor.slice((.., 0))),
                Lane::from(tensor.slice((.., 1))),
            ],
        );
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
    fn test_iter() {
        let tensor = NdTensor::from([[[1, 2], [3, 4]]]);

        // Test iterator over contiguous tensor.
        test_iterator(|| tensor.iter().copied(), &[1, 2, 3, 4]);

        // Test iterator over non-contiguous tensor.
        test_iterator(|| tensor.transposed().iter().copied(), &[1, 3, 2, 4]);
    }

    #[test]
    fn test_iter_mut() {
        struct IterTest(NdTensor<i32, 3>);

        impl MutIterable for IterTest {
            type Iter<'a> = super::IterMut<'a, i32>;

            fn iter_mut(&mut self) -> Self::Iter<'_> {
                self.0.iter_mut()
            }
        }

        let tensor = NdTensor::from([[[1, 2], [3, 4]]]);
        test_mut_iterator(IterTest(tensor), &[&1, &2, &3, &4]);
    }

    #[test]
    #[ignore]
    fn bench_iter() {
        use crate::Layout;
        use rten_bench::run_bench;

        type Elem = i32;

        let tensor = std::hint::black_box(Tensor::<Elem>::full(&[1, 6, 768, 64], 1));
        let n_trials = 1000;
        let mut result = Elem::default();

        fn reduce<'a>(iter: impl Iterator<Item = &'a Elem>) -> Elem {
            iter.fold(Elem::default(), |acc, x| acc.wrapping_add(*x))
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
