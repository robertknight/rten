use std::ops::Deref;

use crate::storage::{CowData, ViewData};
use crate::{AsView, Layout, Storage, TensorBase};

/// A tensor wrapper which guarantees that the tensor has a contiguous layout.
///
/// A contiguous layout means that the order of elements in memory matches the
/// logical row-major ordering of elements with no gaps.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Contiguous<T>(T);

impl<T> Deref for Contiguous<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Contiguous<T> {
    /// Extract the tensor from the wrapper.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<S: Storage, L: Layout> Contiguous<TensorBase<S, L>> {
    /// Wrap a tensor if it is contiguous, or return `None` if the tensor has
    /// a non-contiguous layout.
    pub fn new(inner: TensorBase<S, L>) -> Option<Self> {
        if inner.is_contiguous() {
            Some(Self(inner))
        } else {
            None
        }
    }

    /// Return the tensor's underlying data as a slice.
    ///
    /// Unlike [`TensorBase::data`] this returns a slice instead of an option
    /// because the tensor is known to be contiguous.
    pub fn data(&self) -> &[S::Elem] {
        let len = self.0.len();
        let ptr = self.0.data_ptr();

        // Safety: Constructor verified that tensor is contiguous.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Return a contiguous view of this tensor.
    pub fn view(&self) -> Contiguous<TensorBase<ViewData<'_, S::Elem>, L>>
    where
        TensorBase<S, L>: AsView<Elem = S::Elem, Layout = L>,
    {
        Contiguous(self.0.view())
    }
}

impl<T, L: Clone + Layout> Contiguous<TensorBase<Vec<T>, L>> {
    /// Extract the owned, contiguous data from this tensor.
    pub fn into_data(self) -> Vec<T> {
        self.0.into_non_contiguous_data()
    }
}

impl<'a, T, L: Clone + Layout> Contiguous<TensorBase<CowData<'a, T>, L>> {
    /// Extract the owned data from this tensor, if the data is owned.
    pub fn into_data(self) -> Option<Vec<T>> {
        self.0.into_non_contiguous_data()
    }
}

impl<S: Storage, L: Layout> From<Contiguous<TensorBase<S, L>>> for TensorBase<S, L> {
    fn from(val: Contiguous<TensorBase<S, L>>) -> Self {
        val.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{AsView, Contiguous, Layout, NdTensor};

    #[test]
    fn test_contiguous() {
        let tensor = NdTensor::<f32, 2>::zeros([3, 3]);
        let wrapped = Contiguous::new(tensor);
        assert!(wrapped.is_some());

        let mut tensor: NdTensor<f32, 2> = wrapped.unwrap().into();
        tensor.transpose();
        let wrapped = Contiguous::new(tensor);
        assert!(wrapped.is_none());
    }

    #[test]
    fn test_contiguous_view() {
        let tensor = NdTensor::<f32, 2>::zeros([3, 4]);
        let wrapped = Contiguous::new(tensor).unwrap();
        assert_eq!(wrapped.view().shape(), [3, 4]);
    }
}
