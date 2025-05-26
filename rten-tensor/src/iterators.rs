use std::iter::FusedIterator;
use std::mem::transmute;
use std::ops::Range;
use std::slice;

use smallvec::SmallVec;

use crate::index_iterator::DynIndices;
use crate::layout::{Layout, NdLayout, OverlapPolicy, RemoveDim};
use crate::slice_range::{to_slice_items, SliceItem};
use crate::storage::{StorageMut, ViewData, ViewMutData};

use super::{
    AsView, MutLayout, NdTensorView, NdTensorViewMut, TensorBase, TensorView, TensorViewMut,
};

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
        let inner_pos_pad = INNER_NDIM.saturating_sub(layout.ndim());
        let mut inner_pos = [IterPos::default(); INNER_NDIM];

        let mut dim = 0;
        while dim < inner_pos_pad {
            inner_pos[dim] = IterPos {
                offset: 0,
                stride: 0,
                max_remaining: 0,
                remaining: 0,
            };
            dim += 1;
        }

        let n_outer = layout.ndim().saturating_sub(INNER_NDIM);

        while dim < inner_pos.len() {
            let stride = layout.stride(n_outer + dim - inner_pos_pad);
            let remaining = layout.size(n_outer + dim - inner_pos_pad).saturating_sub(1);
            inner_pos[dim] = IterPos {
                offset: 0,
                remaining,
                max_remaining: remaining,
                stride,
            };
            dim += 1;
        }

        let outer_pos = (0..n_outer)
            .map(|i| {
                let stride = layout.stride(i);
                let remaining = layout.size(i).saturating_sub(1);

                IterPos {
                    offset: 0,
                    remaining,
                    stride,
                    max_remaining: remaining,
                }
            })
            .collect();

        IndexingIterBase {
            len: layout.len(),
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
        let ndim = self.outer_pos.len() + self.inner_pos.len();

        // Advance positions in each dimension, equivalent to calling `step`
        // `n` times.
        let mut n = n.min(self.len);
        while n > 0 {
            // Find the outermost dimension that we can step along which will
            // advance the iterator by <= N elements.
            let mut dim = ndim - 1;
            let mut stride = 1;
            while dim > 0 {
                let size = self.pos(dim).max_remaining + 1;
                let next_stride = stride * size;
                if next_stride >= n {
                    break;
                }
                dim -= 1;
                stride = next_stride;
            }

            // Step along the selected dimension.
            let n_steps = n / stride;
            n -= n_steps * stride;
            self.len -= n_steps * stride;

            for _ in 0..n_steps {
                let mut pos = self.pos_mut(dim);
                while !pos.step() && dim > 0 {
                    dim -= 1;
                    pos = self.pos_mut(dim);
                }
            }
        }

        // Update offset of next element.
        self.inner_offset = self.inner_pos.iter().map(|p| p.offset).sum();
        self.outer_offset = self.outer_pos.iter().map(|p| p.offset).sum();
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
}

impl Iterator for LaneRanges {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        self.offsets.next().map(|offset| {
            // nb. `dim_size` should be >= 1 here as otherwise the tensor
            // has no elements and `self.offsets` should therefore yield no
            // items.
            offset..offset + (self.dim_size - 1) * self.dim_stride + 1
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
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
}

impl<'a, T> Iterator for Lanes<'a, T> {
    type Item = Lane<'a, T>;

    /// Yield the next slice over the target dimension.
    fn next(&mut self) -> Option<Self::Item> {
        self.ranges.next().map(|range| Lane {
            data: self.data.slice(range),
            index: 0,
            stride: self.stride,
            size: self.size,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
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
}

/// Iterator over items in a 1D slice of a tensor.
pub struct LaneMut<'a, T> {
    data: ViewMutData<'a, T>,
    index: usize,
    stride: usize,
    size: usize,
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

impl<'a, T> Iterator for LanesMut<'a, T> {
    type Item = LaneMut<'a, T>;

    fn next(&mut self) -> Option<LaneMut<'a, T>> {
        self.ranges.next().map(|range| {
            let data = self.data.slice_mut(range);
            LaneMut {
                data: unsafe { transmute::<ViewMutData<'_, T>, ViewMutData<'a, T>>(data) },
                size: self.size,
                stride: self.stride,
                index: 0,
            }
        })
    }
}

/// Base for iterators over views of the inner N dimensions of a tensor.
struct InnerIterBase<const N: usize> {
    outer_indices: DynIndices,
    outer_strides: SmallVec<[usize; 4]>,
    inner_layout: NdLayout<N>,
}

impl<const N: usize> InnerIterBase<N> {
    pub fn new<L: Layout>(parent_layout: &L) -> Self {
        assert!(parent_layout.ndim() >= N);
        let outer_dims = parent_layout.ndim() - N;
        let parent_shape = parent_layout.shape();
        let parent_strides = parent_layout.strides();
        let (outer_shape, inner_shape) = parent_shape.as_ref().split_at(outer_dims);
        let (outer_strides, inner_strides) = parent_strides.as_ref().split_at(outer_dims);

        let inner_shape: [usize; N] = inner_shape.try_into().unwrap();
        let inner_strides: [usize; N] = inner_strides.try_into().unwrap();
        let inner_layout = NdLayout::from_shape_and_strides(
            inner_shape,
            inner_strides,
            // We allow overlap here, but the view that owns `parent_layout`
            // will enforce there is no overlap if it is a mutable view.
            OverlapPolicy::AllowOverlap,
        )
        .expect("failed to create layout");

        let outer_indices = DynIndices::from_shape(outer_shape);
        InnerIterBase {
            outer_indices,
            outer_strides: SmallVec::from_slice(outer_strides),
            inner_layout,
        }
    }
}

impl<const N: usize> Iterator for InnerIterBase<N> {
    /// Storage offset range for next view
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        self.outer_indices.next().map(|idx| {
            let offset: usize = idx
                .iter()
                .zip(self.outer_strides.as_ref())
                .map(|(idx, stride)| idx * stride)
                .sum();
            offset..(offset + self.inner_layout.min_data_len())
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.outer_indices.size_hint()
    }
}

/// Iterator over views of the N innermost dimensions of a tensor with element
/// type `T` and layout `L`.
pub struct InnerIter<'a, T, const N: usize> {
    base: InnerIterBase<N>,
    data: ViewData<'a, T>,
}

impl<'a, T, const N: usize> InnerIter<'a, T, N> {
    pub fn new<L: MutLayout>(view: TensorBase<ViewData<'a, T>, L>) -> Self {
        let base = InnerIterBase::new(&view);
        InnerIter {
            base,
            data: view.storage(),
        }
    }
}

impl<'a, T, const N: usize> Iterator for InnerIter<'a, T, N> {
    type Item = NdTensorView<'a, T, N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|offset_range| {
            NdTensorView::from_storage_and_layout(
                self.data.slice(offset_range),
                self.base.inner_layout,
            )
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}

impl<T, const N: usize> ExactSizeIterator for InnerIter<'_, T, N> {}

/// Iterator over views of the N innermost dimensions of a tensor with element
/// type `T` and layout `L`, where `N` is determined at runtime.
pub struct InnerIterDyn<'a, T, L: MutLayout> {
    outer_indices: DynIndices,
    view: TensorBase<ViewData<'a, T>, L>,
}

impl<'a, T, L: MutLayout> InnerIterDyn<'a, T, L> {
    pub fn new(view: TensorBase<ViewData<'a, T>, L>, inner_dims: usize) -> Self {
        assert!(view.ndim() >= inner_dims);
        let outer_dims = view.ndim() - inner_dims;
        let outer_indices = DynIndices::from_shape(&view.shape().as_ref()[..outer_dims]);
        InnerIterDyn {
            outer_indices,
            view,
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for InnerIterDyn<'a, T, L> {
    type Item = TensorView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.outer_indices.next().map(|idx| {
            let slice_items = to_slice_items(&idx);
            self.view.slice(slice_items.as_slice())
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.outer_indices.size_hint()
    }
}

impl<T, L: MutLayout> ExactSizeIterator for InnerIterDyn<'_, T, L> {}

/// Iterator over mutable views of the N innermost dimensions of a tensor.
pub struct InnerIterMut<'a, T, const N: usize> {
    base: InnerIterBase<N>,
    data: ViewMutData<'a, T>,
}

impl<'a, T, const N: usize> InnerIterMut<'a, T, N> {
    pub fn new<L: MutLayout>(view: TensorBase<ViewMutData<'a, T>, L>) -> InnerIterMut<'a, T, N> {
        let base = InnerIterBase::new(&view);
        InnerIterMut {
            base,
            data: view.into_storage(),
        }
    }
}

impl<'a, T, const N: usize> Iterator for InnerIterMut<'a, T, N> {
    type Item = NdTensorViewMut<'a, T, N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|offset_range| {
            let view: NdTensorViewMut<'_, T, N> = NdTensorViewMut::from_storage_and_layout(
                self.data.slice_mut(offset_range),
                self.base.inner_layout,
            );
            unsafe {
                // Safety: Outer view is non-broadcasting, and we increment the
                // outer index each time, so returned views will not overlap.
                std::mem::transmute::<NdTensorViewMut<'_, T, N>, NdTensorViewMut<'a, T, N>>(view)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}

impl<T, const N: usize> ExactSizeIterator for InnerIterMut<'_, T, N> {}

/// Iterator over mutable views of the N innermost dimensions of a tensor,
/// where N is determined at runtime.
pub struct InnerIterDynMut<'a, T, L: MutLayout> {
    outer_indices: DynIndices,
    view: TensorBase<ViewMutData<'a, T>, L>,
}

impl<'a, T, L: MutLayout> InnerIterDynMut<'a, T, L> {
    pub fn new(view: TensorBase<ViewMutData<'a, T>, L>, inner_dims: usize) -> Self {
        assert!(view.ndim() >= inner_dims);
        let outer_dims = view.ndim() - inner_dims;
        let outer_indices = DynIndices::from_shape(&view.shape().as_ref()[..outer_dims]);
        InnerIterDynMut {
            outer_indices,
            view,
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for InnerIterDynMut<'a, T, L> {
    type Item = TensorViewMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.outer_indices.next().map(|idx| {
            let slice_items = to_slice_items(&idx);
            let view: TensorViewMut<'_, T> = self.view.slice_mut(slice_items.as_slice());
            unsafe {
                // Safety: Outer view is non-broadcasting, and we increment the
                // outer index each time, so returned views will not overlap.
                std::mem::transmute::<TensorViewMut<'_, T>, TensorViewMut<'a, T>>(view)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.outer_indices.size_hint()
    }
}

impl<T, L: MutLayout> ExactSizeIterator for InnerIterDynMut<'_, T, L> {}

/// Iterator over slices of a tensor along an axis. See [`TensorView::axis_iter`].
pub struct AxisIter<'a, T, L: MutLayout + RemoveDim> {
    view: TensorBase<ViewData<'a, T>, L>,
    axis: usize,
    index: usize,
}

impl<'a, T, L: MutLayout + RemoveDim> AxisIter<'a, T, L> {
    pub fn new(view: &TensorBase<ViewData<'a, T>, L>, axis: usize) -> AxisIter<'a, T, L> {
        assert!(axis < view.ndim());
        AxisIter {
            view: view.clone(),
            axis,
            index: 0,
        }
    }
}

impl<'a, T, L: MutLayout + RemoveDim> Iterator for AxisIter<'a, T, L> {
    type Item = TensorBase<ViewData<'a, T>, <L as RemoveDim>::Output>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.size(self.axis) {
            None
        } else {
            let slice = self.view.index_axis(self.axis, self.index);
            self.index += 1;
            Some(slice)
        }
    }
}

/// Iterator over mutable slices of a tensor along an axis. See [`TensorViewMut::axis_iter_mut`].
pub struct AxisIterMut<'a, T, L: MutLayout + RemoveDim> {
    view: TensorBase<ViewMutData<'a, T>, L>,
    axis: usize,
    index: usize,
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
            view,
            axis,
            index: 0,
        }
    }
}

/// Mutable tensor view with one less dimension than `L`.
type SmallerMutView<'b, T, L> = TensorBase<ViewMutData<'b, T>, <L as RemoveDim>::Output>;

impl<'a, T, L: MutLayout + RemoveDim> Iterator for AxisIterMut<'a, T, L> {
    type Item = TensorBase<ViewMutData<'a, T>, <L as RemoveDim>::Output>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.size(0) {
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
}

/// Iterator over slices of a tensor along an axis. See [`TensorView::axis_chunks`].
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
    #[should_panic(expected = "chunk size must be > 0")]
    fn test_axis_chunks_mut_zero_size() {
        let mut x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(AxisChunksMut::new(x.view_mut(), 1, 0).next().is_none());
    }

    #[test]
    fn test_lanes_empty() {
        let x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(Lanes::new(x.view().view_ref(), 0).next().is_none());
        assert!(Lanes::new(x.view().view_ref(), 1).next().is_none());
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

        // Use tensor iterator with non-contiguous slice. This will fall back
        // to indexed iteration.
        let slice = tensor.slice((.., .., 1.., ..));
        assert!(!slice.is_contiguous());
        let n_trials = 1000;
        run_bench(n_trials, Some("non-contiguous iter"), || {
            result = reduce(slice.iter());
        });
        println!("sum {}", result);
    }
}
