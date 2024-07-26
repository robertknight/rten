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
/// [TensorView] in that it borrows the layout rather than having its own.
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
/// [TensorViewMut] in that it borrows the layout rather than having its own.
pub(crate) struct MutViewRef<'d, 'l, T, L: Layout> {
    data: ViewMutData<'d, T>,
    layout: &'l L,
}

impl<'d, 'l, T, L: Layout> MutViewRef<'d, 'l, T, L> {
    pub(crate) fn new(data: ViewMutData<'d, T>, layout: &'l L) -> MutViewRef<'d, 'l, T, L> {
        MutViewRef { data, layout }
    }
}

/// IterPos tracks the position within a single dimension of an IndexingIter.
#[derive(Copy, Clone, Debug)]
struct IterPos {
    /// Steps remaining along this dimension before we reset. Each step
    /// corresponds to advancing one or more indexes either forwards or backwards.
    ///
    /// This starts at `steps - 1` in each iteration, since the first step is
    /// effectively taken when we reset.
    steps_remaining: usize,

    /// Number of steps in each iteration over this dimension.
    steps: usize,

    /// Data offset adjustment for each step along this dimension.
    offset_step: isize,
}

impl IterPos {
    fn new(steps: usize, offset_step: isize) -> IterPos {
        IterPos {
            steps_remaining: steps.saturating_sub(1),
            steps,
            offset_step,
        }
    }

    /// Take one step along this dimension or reset if we reached the end.
    #[inline(always)]
    fn step(&mut self) -> bool {
        if self.steps_remaining != 0 {
            self.steps_remaining -= 1;
            true
        } else {
            self.steps_remaining = self.steps.saturating_sub(1);
            false
        }
    }
}

/// Helper for iterating over offsets of elements in a tensor.
#[derive(Clone, Debug)]
struct IndexingIterBase {
    /// Remaining elements to visit
    len: usize,

    /// Offset of the next element to return from the tensor's element buffer.
    offset: isize,

    /// Current position within each dimension.
    pos: Vec<IterPos>,
}

impl IndexingIterBase {
    /// Create an iterator over element offsets in `tensor`.
    fn new<L: Layout>(layout: &L) -> IndexingIterBase {
        let dims = layout
            .shape()
            .as_ref()
            .iter()
            .enumerate()
            .map(|(dim, &len)| IterPos::new(len, layout.stride(dim) as isize))
            .collect();

        IndexingIterBase {
            len: layout.len(),
            offset: 0,
            pos: dims,
        }
    }

    /// Advance the iterator by stepping along dimension `dim`.
    ///
    /// The caller must calculate `stride`, the number of indices being stepped
    /// over.
    #[inline(always)]
    fn step_dim(&mut self, mut dim: usize, stride: usize) {
        self.len -= stride;
        let mut pos = &mut self.pos[dim];
        while !pos.step() {
            // End of range reached for dimension `dim`. Rewind offset by
            // amount it moved since iterating from the start of this dimension.
            self.offset -= pos.offset_step * (pos.steps as isize - 1);

            if dim == 0 {
                break;
            }

            dim -= 1;
            pos = &mut self.pos[dim];
        }
        self.offset += pos.offset_step;
    }

    /// Advance iterator by one index.
    #[inline(always)]
    fn step(&mut self) {
        self.step_dim(self.pos.len() - 1, 1);
    }

    /// Advance iterator by up to `n` indices.
    fn step_by(&mut self, n: usize) {
        let mut n = n.min(self.len);
        while n > 0 {
            // Find the outermost dimension that we can step along which will
            // advance the iterator by <= N elements.
            let mut dim = self.pos.len() - 1;
            let mut stride = 1;
            while dim > 0 {
                let next_stride = stride * self.pos[dim].steps;
                if next_stride >= n {
                    break;
                }
                dim -= 1;
                stride = next_stride;
            }

            // Step along the selected dimension.
            let n_steps = n / stride;
            for _ in 0..n_steps {
                n -= stride;
                self.step_dim(dim, stride);
            }
        }
    }
}

/// Iterator over elements of a tensor, in their logical order.
#[derive(Clone)]
pub struct Iter<'a, T> {
    iter: IterKind<'a, T>,
}

/// Alternate implementations of `Elements`.
///
/// When the tensor has a contiguous layout, this iterator is just a thin
/// wrapper around a slice iterator.
#[derive(Clone)]
enum IterKind<'a, T> {
    Direct(slice::Iter<'a, T>),
    Indexing(IndexingIter<'a, T>),
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

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

impl<'a, T> FusedIterator for Iter<'a, T> {}

#[derive(Clone)]
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

impl<'a, T> Iterator for IndexingIter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.base.len == 0 {
            return None;
        }
        let element = unsafe {
            // Safety: See comments in Storage trait.
            self.data.get(self.base.offset as usize).unwrap()
        };
        self.base.step();
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }
}

impl<'a, T> ExactSizeIterator for IndexingIter<'a, T> {}

impl<'a, T> FusedIterator for IndexingIter<'a, T> {}

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

impl<'a, T> ExactSizeIterator for IterMut<'a, T> {}

impl<'a, T> FusedIterator for IterMut<'a, T> {}

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
        if self.base.len == 0 {
            return None;
        }
        let element = unsafe {
            // Safety: See comments in Storage trait.
            let el = self.data.get_mut(self.base.offset as usize).unwrap();

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

impl<'a, T> ExactSizeIterator for IndexingIterMut<'a, T> {}

impl<'a, T> FusedIterator for IndexingIterMut<'a, T> {}

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
        if self.base.len == 0 {
            return None;
        }
        let offset = self.base.offset;
        self.base.step();
        Some(offset as usize)
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
        let (_range, sliced) = layout.slice_dyn(&slice_starts);
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

impl<'a, T> ExactSizeIterator for Lane<'a, T> {}

impl<'a, T> FusedIterator for Lane<'a, T> {}

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

impl<'a, T> ExactSizeIterator for Lanes<'a, T> {}

impl<'a, T> FusedIterator for Lanes<'a, T> {}

/// Mutable version of [Lanes].
///
/// Unlike [Lanes], this does not implement [Iterator] due to complications
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

impl<'a, T> ExactSizeIterator for LaneMut<'a, T> {}

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

impl<'a, T, const N: usize> ExactSizeIterator for InnerIter<'a, T, N> {}

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
            self.view.slice_dyn(slice_items.as_slice())
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.outer_indices.size_hint()
    }
}

impl<'a, T, L: MutLayout> ExactSizeIterator for InnerIterDyn<'a, T, L> {}

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

impl<'a, T, const N: usize> ExactSizeIterator for InnerIterMut<'a, T, N> {}

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
            let view: TensorViewMut<'_, T> = self.view.slice_mut_dyn(slice_items.as_slice());
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

impl<'a, T, L: MutLayout> ExactSizeIterator for InnerIterDynMut<'a, T, L> {}

/// Iterator over slices of a tensor along an axis. See [TensorView::axis_iter].
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

/// Iterator over mutable slices of a tensor along an axis. See [TensorViewMut::axis_iter_mut].
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

/// Iterator over slices of a tensor along an axis. See [TensorView::axis_chunks].
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

/// Iterator over mutable slices of a tensor along an axis. See [TensorViewMut::axis_chunks_mut].
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
    use crate::{AsView, AxisChunks, AxisChunksMut, Lanes, LanesMut, Tensor};

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
    fn test_lanes_mut_empty() {
        let mut x = Tensor::<i32>::zeros(&[5, 0]);
        assert!(LanesMut::new(x.mut_view_ref(), 0).next().is_none());
        assert!(LanesMut::new(x.mut_view_ref(), 1).next().is_none());
    }
}
