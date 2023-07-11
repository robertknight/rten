use std::iter::{repeat, zip, Cycle, Take};
use std::slice;

use super::layout::Layout;
use super::range::SliceRange;
use super::{TensorBase, TensorLayout, TensorView, TensorViewMut};

/// IterPos tracks the position within a single dimension of an IndexingIter.
#[derive(Debug)]
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
#[derive(Debug)]
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
    fn new(layout: &Layout) -> IndexingIterBase {
        let dims = layout
            .shape()
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

    /// Create an iterator over offsets of elements in `tensor`, as if it had
    /// a given `shape`. This will repeat offsets as necessary.
    fn broadcast(layout: &Layout, shape: &[usize]) -> IndexingIterBase {
        // nb. We require that the broadcast shape has a length >= the actual
        // shape.
        let added_dims = shape.len() - layout.shape().len();
        let padded_tensor_shape = repeat(&0).take(added_dims).chain(layout.shape().iter());
        let dims = zip(padded_tensor_shape, shape.iter())
            .enumerate()
            .map(|(dim, (&actual_len, &broadcast_len))| {
                // If the dimension is being broadcast, set its stride to 0 so
                // that when we increment in this dimension, we just repeat
                // elements. Otherwise, use the real stride.
                let offset_step = if actual_len == broadcast_len {
                    layout.stride(dim - added_dims) as isize
                } else {
                    0
                };

                IterPos::new(broadcast_len, offset_step)
            })
            .collect();

        IndexingIterBase {
            len: shape.iter().product(),
            offset: 0,
            pos: dims,
        }
    }

    /// Create an iterator over offsets of a subset of elements in `tensor`.
    fn slice(layout: &Layout, ranges: &[SliceRange]) -> IndexingIterBase {
        assert!(
            ranges.len() == layout.ndim(),
            "slice dimensions {} do not match tensor dimensions {}",
            ranges.len(),
            layout.ndim()
        );
        let mut offset = 0;
        let dims: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(dim, range)| {
                let len = layout.size(dim);
                let range = range.clamp(len);
                let stride = layout.stride(dim);

                let start_index = if range.start >= 0 {
                    range.start
                } else {
                    (len as isize) + range.start
                };

                // Clamped ranges either have a start index that is valid, or
                // that is one before/after the first/last valid index
                // (depending on step direction). If invalid, the slice is
                // empty.
                if start_index >= 0 && start_index < (len as isize) {
                    offset += stride * start_index as usize;
                } else {
                    assert!(range.steps(len) == 0);
                }

                IterPos::new(range.steps(len), (stride as isize) * range.step())
            })
            .collect();

        IndexingIterBase {
            len: dims.iter().map(|dim| dim.steps).product(),
            offset: offset as isize,
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
pub struct Iter<'a, T> {
    iter: IterKind<'a, T>,
}

/// Alternate implementations of `Elements`.
///
/// When the tensor has a contiguous layout, this iterator is just a thin
/// wrapper around a slice iterator.
enum IterKind<'a, T> {
    Direct(slice::Iter<'a, T>),
    Indexing(IndexingIter<'a, T>),
}

impl<T> Iter<'_, T> {
    pub(super) fn new<S: AsRef<[T]>>(view: &TensorBase<T, S>) -> Iter<T> {
        if view.layout().is_contiguous() {
            Iter {
                iter: IterKind::Direct(view.data().as_ref().iter()),
            }
        } else {
            Iter {
                iter: IterKind::Indexing(IndexingIter::new(view)),
            }
        }
    }

    /// Create a new iterator for elements of a given view. Unlike
    /// [Elements::new], the lifetime is that of the element storage rather than
    /// the view.
    pub(super) fn from_view<'a>(view: &TensorBase<T, &'a [T]>) -> Iter<'a, T> {
        if view.layout().is_contiguous() {
            Iter {
                iter: IterKind::Direct(view.to_data().iter()),
            }
        } else {
            Iter {
                iter: IterKind::Indexing(IndexingIter::from_view(view)),
            }
        }
    }

    pub(super) fn slice<'a, S: AsRef<[T]>>(
        view: &'a TensorBase<T, S>,
        ranges: &[SliceRange],
    ) -> Iter<'a, T> {
        let iter = IndexingIter {
            base: IndexingIterBase::slice(view.layout(), ranges),
            data: view.data(),
        };
        Iter {
            iter: IterKind::Indexing(iter),
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

struct IndexingIter<'a, T> {
    base: IndexingIterBase,

    /// Data buffer of the tensor
    data: &'a [T],
}

impl<'a, T> IndexingIter<'a, T> {
    fn new<S: AsRef<[T]>>(view: &TensorBase<T, S>) -> IndexingIter<T> {
        IndexingIter {
            base: IndexingIterBase::new(view.layout()),
            data: view.data(),
        }
    }

    fn from_view(view: &TensorBase<T, &'a [T]>) -> IndexingIter<'a, T> {
        IndexingIter {
            base: IndexingIterBase::new(view.layout()),
            data: view.to_data(),
        }
    }

    fn broadcast<S: AsRef<[T]>>(
        view: &'a TensorBase<T, S>,
        shape: &[usize],
    ) -> IndexingIter<'a, T> {
        IndexingIter {
            base: IndexingIterBase::broadcast(view.layout(), shape),
            data: view.data(),
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
        let element = &self.data[self.base.offset as usize];
        self.base.step();
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }
}

impl<'a, T> ExactSizeIterator for IndexingIter<'a, T> {}

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
    pub(super) fn new(data: &'a mut [T], layout: &Layout) -> IterMut<'a, T> {
        if layout.is_contiguous() {
            IterMut {
                iter: IterMutKind::Direct(data.iter_mut()),
            }
        } else {
            IterMut {
                iter: IterMutKind::Indexing(IndexingIterMut::new(data, layout)),
            }
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter {
            IterMutKind::Direct(ref mut iter) => iter.next(),
            IterMutKind::Indexing(ref mut iter) => iter.next(),
        }
    }

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

struct IndexingIterMut<'a, T> {
    base: IndexingIterBase,

    /// Data buffer of the tensor
    data: &'a mut [T],
}

impl<'a, T> IndexingIterMut<'a, T> {
    fn new(data: &'a mut [T], layout: &Layout) -> IndexingIterMut<'a, T> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !layout.is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );
        IndexingIterMut {
            base: IndexingIterBase::new(layout),
            data,
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
            let el = &mut self.data[self.base.offset as usize];

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

/// Iterator over element offsets of a tensor.
///
/// `Offsets` does not hold a reference to the tensor, allowing the tensor to
/// be modified during iteration. It is the caller's responsibilty not to modify
/// the tensor in ways that invalidate the offset sequence returned by this
/// iterator.
pub struct Offsets {
    base: IndexingIterBase,
}

impl Offsets {
    pub fn new(layout: &Layout) -> Offsets {
        Offsets {
            base: IndexingIterBase::new(layout),
        }
    }

    pub fn broadcast(layout: &Layout, shape: &[usize]) -> Offsets {
        Offsets {
            base: IndexingIterBase::broadcast(layout, shape),
        }
    }

    pub fn slice(layout: &Layout, ranges: &[SliceRange]) -> Offsets {
        Offsets {
            base: IndexingIterBase::slice(layout, ranges),
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

/// Iterator over elements of a tensor which broadcasts to a different shape.
///
/// This iterator will repeat elements of the underlying tensor until the total
/// number yielded matches the product of the shape being broadcast to.
pub struct BroadcastIter<'a, T> {
    iter: BroadcastIterKind<'a, T>,
}

/// Alternate implementations for broadcasting. See notes in
/// `BroadcastElements::can_broadcast_by_cycling`.
enum BroadcastIterKind<'a, T> {
    Direct(Take<Cycle<slice::Iter<'a, T>>>),
    Indexing(IndexingIter<'a, T>),
}

/// Return true if a tensor with shape `from_shape` can be broadcast to shape
/// `to_shape` by cycling through all of its elements repeatedly.
///
/// This requires that, after left-padding `from_shape` with 1s to match the
/// length of `to_shape`, all non-1 dimensions in `from_shape` are contiguous
/// at the end. For example, `[1, 5, 10]` can be broadcast to `[3, 4, 5, 10]`
/// by cycling, but `[5, 1, 10]` cannot be broadcast to `[5, 4, 10]` this way,
/// as the inner (`[1, 10]`) dimensions will need to be repeated 4 times before
/// moving to the next index in the outermost dimension.
///
/// If the tensor can be broadcast via cycling, and is also contiguous, it can
/// be broadcast efficiently using `tensor.data().iter().cycle()`.
fn can_broadcast_by_cycling(from_shape: &[usize], to_shape: &[usize]) -> bool {
    assert!(to_shape.len() >= from_shape.len());

    let excess_dims = to_shape.len() - from_shape.len();
    let mut dims_to_check = to_shape.len() - excess_dims;

    while dims_to_check > 0 {
        if from_shape[dims_to_check - 1] != to_shape[excess_dims + dims_to_check - 1] {
            break;
        }
        dims_to_check -= 1;
    }

    while dims_to_check > 0 {
        if from_shape[dims_to_check - 1] != 1 {
            return false;
        }
        dims_to_check -= 1;
    }

    true
}

impl<'a, T> BroadcastIter<'a, T> {
    pub fn new<S: AsRef<[T]>>(
        view: &'a TensorBase<T, S>,
        to_shape: &[usize],
    ) -> BroadcastIter<'a, T> {
        let iter = if view.is_contiguous() && can_broadcast_by_cycling(view.shape(), to_shape) {
            let iter_len = to_shape.iter().product();
            BroadcastIterKind::Direct(view.data().iter().cycle().take(iter_len))
        } else {
            BroadcastIterKind::Indexing(IndexingIter::broadcast(view, to_shape))
        };
        BroadcastIter { iter }
    }
}

impl<'a, T> Iterator for BroadcastIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter {
            BroadcastIterKind::Direct(ref mut iter) => iter.next(),
            BroadcastIterKind::Indexing(ref mut iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.iter {
            BroadcastIterKind::Direct(iter) => iter.size_hint(),
            BroadcastIterKind::Indexing(iter) => iter.size_hint(),
        }
    }
}

impl<'a, T> ExactSizeIterator for BroadcastIter<'a, T> {}

/// Iterator over slices of a tensor along an axis. See [TensorBase::axis_iter].
pub struct AxisIter<'a, T> {
    view: TensorView<'a, T>,
    index: usize,
}

impl<'a, T> AxisIter<'a, T> {
    pub fn new(mut view: TensorView<'a, T>, dim: usize) -> AxisIter<'a, T> {
        view.move_axis(dim, 0);
        AxisIter { view, index: 0 }
    }
}

impl<'a, T> Iterator for AxisIter<'a, T> {
    type Item = TensorView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.size(0) {
            None
        } else {
            let view = self.view.to_slice([self.index]);
            self.index += 1;
            Some(view)
        }
    }
}

/// Iterator over mutable slices of a tensor along an axis. See [TensorBase::axis_iter_mut].
pub struct AxisIterMut<'a, T> {
    view: TensorViewMut<'a, T>,
    index: usize,
}

impl<'a, T> AxisIterMut<'a, T> {
    pub fn new(mut view: TensorViewMut<'a, T>, dim: usize) -> AxisIterMut<'a, T> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !view.layout().is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );
        view.move_axis(dim, 0);
        AxisIterMut { view, index: 0 }
    }
}

impl<'a, T> Iterator for AxisIterMut<'a, T> {
    type Item = TensorViewMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.size(0) {
            None
        } else {
            let index = self.index;
            self.index += 1;

            // Safety: This is non-broadcasting view, and we increment the index
            // each time, so returned views will not overlap.
            let view = unsafe {
                let view = self.view.slice_mut([index]);
                std::mem::transmute::<TensorViewMut<'_, T>, TensorViewMut<'a, T>>(view)
            };
            Some(view)
        }
    }
}
