use crate::index_iterator::DynIndices;
use crate::layout::Layout;
use crate::range::to_slice_items;

use super::{MutLayout, NdTensorView, NdTensorViewMut, TensorBase};

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
            self.view.slice(slice_items.as_slice()).try_into().unwrap()
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
            let view: NdTensorViewMut<'_, T, N> = self
                .view
                .slice_mut(slice_items.as_slice())
                .try_into()
                .unwrap();

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
