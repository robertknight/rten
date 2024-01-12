use std::ops::Add;

use crate::index_iterator::DynIndices;
use crate::layout::Layout;
use crate::range::to_slice_items;

use super::{
    AsView, MutLayout, NdTensorView, NdTensorViewMut, TensorBase, TensorView, TensorViewMut,
};

/// Iterator over views of the N innermost dimensions of a tensor with element
/// type `T` and layout `L`.
pub struct InnerIter<'a, T, L: MutLayout, const N: usize> {
    outer_indices: DynIndices,
    view: TensorBase<T, &'a [T], L>,
}

impl<'a, T, L: MutLayout, const N: usize> InnerIter<'a, T, L, N> {
    pub fn new(view: TensorBase<T, &'a [T], L>) -> Self {
        assert!(view.ndim() >= N);
        let outer_dims = view.ndim() - N;
        let outer_indices = DynIndices::from_shape(&view.shape().as_ref()[..outer_dims]);
        InnerIter {
            outer_indices,
            view,
        }
    }
}

impl<'a, T, L: MutLayout, const N: usize> Iterator for InnerIter<'a, T, L, N> {
    type Item = NdTensorView<'a, T, N>;

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

impl<'a, T, L: MutLayout, const N: usize> ExactSizeIterator for InnerIter<'a, T, L, N> {}

/// Iterator over mutable views of the N innermost dimensions of a tensor.
pub struct InnerIterMut<'a, T, L: MutLayout, const N: usize> {
    outer_indices: DynIndices,
    view: TensorBase<T, &'a mut [T], L>,
}

impl<'a, T, L: MutLayout, const N: usize> InnerIterMut<'a, T, L, N> {
    pub fn new(view: TensorBase<T, &'a mut [T], L>) -> Self {
        assert!(view.ndim() >= N);
        let outer_dims = view.ndim() - N;
        let outer_indices = DynIndices::from_shape(&view.shape().as_ref()[..outer_dims]);
        InnerIterMut {
            outer_indices,
            view,
        }
    }
}

impl<'a, T, L: MutLayout, const N: usize> Iterator for InnerIterMut<'a, T, L, N> {
    type Item = NdTensorViewMut<'a, T, N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.outer_indices.next().map(|idx| {
            let slice_items = to_slice_items(&idx);
            let view: NdTensorViewMut<'_, T, N> = self.view.slice_mut(slice_items.as_slice());
            unsafe {
                // Safety: Outer view is non-broadcasting, and we increment the
                // outer index each time, so returned views will not overlap.
                std::mem::transmute::<NdTensorViewMut<'_, T, N>, NdTensorViewMut<'a, T, N>>(view)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.outer_indices.size_hint()
    }
}

impl<'a, T, L: MutLayout, const N: usize> ExactSizeIterator for InnerIterMut<'a, T, L, N> {}

/// Iterator over slices of a tensor along an axis. See [TensorView::axis_iter].
pub struct AxisIter<'a, T, L: MutLayout> {
    view: TensorBase<T, &'a [T], L>,
    index: usize,
}

impl<'a, T, L: MutLayout> AxisIter<'a, T, L> {
    pub fn new(view: &TensorBase<T, &'a [T], L>, dim: usize) -> AxisIter<'a, T, L> {
        let mut permuted = view.clone();
        permuted.move_axis(dim, 0);
        AxisIter {
            view: permuted,
            index: 0,
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for AxisIter<'a, T, L> {
    type Item = TensorView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.size(0) {
            None
        } else {
            let view = self.view.slice_dyn([self.index]);
            self.index += 1;
            Some(view)
        }
    }
}

/// Iterator over mutable slices of a tensor along an axis. See [TensorViewMut::axis_iter_mut].
pub struct AxisIterMut<'a, T, L: MutLayout> {
    view: TensorBase<T, &'a mut [T], L>,
    index: usize,
}

impl<'a, T, L: MutLayout> AxisIterMut<'a, T, L> {
    pub fn new(mut view: TensorBase<T, &'a mut [T], L>, dim: usize) -> AxisIterMut<'a, T, L> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !view.layout().is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );
        view.move_axis(dim, 0);
        AxisIterMut { view, index: 0 }
    }
}

impl<'a, T, L: MutLayout> Iterator for AxisIterMut<'a, T, L> {
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
                let view = self.view.slice_mut_dyn([index]);
                std::mem::transmute::<TensorViewMut<'_, T>, TensorViewMut<'a, T>>(view)
            };
            Some(view)
        }
    }
}

/// Iterator over slices of a tensor along an axis. See [TensorView::axis_chunks].
pub struct AxisChunks<'a, T, L: MutLayout> {
    view: TensorBase<T, &'a [T], L>,
    index: usize,
    chunk_size: usize,
}

impl<'a, T, L: MutLayout> AxisChunks<'a, T, L> {
    pub fn new(
        view: &TensorBase<T, &'a [T], L>,
        dim: usize,
        chunk_size: usize,
    ) -> AxisChunks<'a, T, L> {
        let mut permuted = view.clone();
        permuted.move_axis(dim, 0);
        AxisChunks {
            view: permuted,
            index: 0,
            chunk_size,
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for AxisChunks<'a, T, L> {
    type Item = TensorView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let size = self.view.size(0);
        if self.index >= self.view.size(0) {
            None
        } else {
            let view = self
                .view
                .slice_dyn(self.index..self.index.add(self.chunk_size).min(size));
            self.index += self.chunk_size;
            Some(view)
        }
    }
}

/// Iterator over mutable slices of a tensor along an axis. See [TensorViewMut::axis_chunks_mut].
pub struct AxisChunksMut<'a, T, L: MutLayout> {
    view: TensorBase<T, &'a mut [T], L>,
    index: usize,
    chunk_size: usize,
}

impl<'a, T, L: MutLayout> AxisChunksMut<'a, T, L> {
    pub fn new(
        mut view: TensorBase<T, &'a mut [T], L>,
        dim: usize,
        chunk_size: usize,
    ) -> AxisChunksMut<'a, T, L> {
        // See notes in `Layout` about internal overlap.
        assert!(
            !view.layout().is_broadcast(),
            "Cannot mutably iterate over broadcasting view"
        );
        view.move_axis(dim, 0);
        AxisChunksMut {
            view,
            chunk_size,
            index: 0,
        }
    }
}

impl<'a, T, L: MutLayout> Iterator for AxisChunksMut<'a, T, L> {
    type Item = TensorViewMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let size = self.view.size(0);

        if self.index >= size {
            None
        } else {
            let index = self.index;
            self.index += self.chunk_size;

            // Safety: This is non-broadcasting view, and we increment the index
            // each time, so returned views will not overlap.
            let view = unsafe {
                let view = self
                    .view
                    .slice_mut_dyn(index..index.add(self.chunk_size).min(size));
                std::mem::transmute::<TensorViewMut<'_, T>, TensorViewMut<'a, T>>(view)
            };
            Some(view)
        }
    }
}
