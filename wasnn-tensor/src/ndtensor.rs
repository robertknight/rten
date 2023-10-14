use std::borrow::Cow;
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::errors::{DimensionError, FromDataError};
use crate::index_iterator::NdIndices;
use crate::iterators::{Iter, IterMut};
use crate::layout::{Layout, MatrixLayout, NdLayout, OverlapPolicy};
use crate::range::SliceItem;
use crate::{IntoSliceItems, TensorBase, TensorView, TensorViewMut};

/// Multi-dimensional array view with a static dimension count. This trait
/// includes operations that are available on tensors that own their data
/// ([NdTensor]), as well as views ([NdTensorView], [NdTensorViewMut]).
///
/// `N` is the static rank of this tensor.
///
/// [NdTensorView] implements specialized versions of these methods as inherent
/// methods, which preserve lifetiems on the result.
pub trait NdView<const N: usize>: Layout {
    /// The data type of elements in this tensor.
    type Elem;

    /// Return a view of this tensor with a dynamic dimension count.
    fn as_dyn(&self) -> TensorView<Self::Elem> {
        self.view().as_dyn()
    }

    /// Return a reference to the underlying data of this tensor.
    fn data(&self) -> &[Self::Elem];

    /// Return the element at a given index, or `None` if the index is out of
    /// bounds in any dimension.
    fn get(&self, index: [usize; N]) -> Option<&Self::Elem> {
        self.view().get(index)
    }

    /// Return an iterator over elements of this tensor, in their logical order.
    fn iter(&self) -> Iter<Self::Elem> {
        self.view().iter()
    }

    /// Return a copy of this tensor with each element replaced by `f(element)`.
    ///
    /// The order in which elements are visited is unspecified and may not
    /// correspond to the logical order.
    fn map<F, U>(&self, f: F) -> NdTensor<U, N>
    where
        F: Fn(&Self::Elem) -> U,
    {
        self.view().map(f)
    }

    /// Return a new view with a given shape.
    ///
    /// The current view must be contiguous and the new shape must have the
    /// same product as the current shape.
    fn reshaped<const M: usize>(&self, shape: [usize; M]) -> NdTensorView<Self::Elem, M> {
        self.view().reshaped(shape)
    }

    /// Return a new view with the dimensions re-ordered according to `dims`.
    fn permuted(&self, dims: [usize; N]) -> NdTensorView<Self::Elem, N> {
        self.view().permuted(dims)
    }

    /// Return an immutable view of part of this tensor.
    ///
    /// `M` specifies the number of dimensions that the layout must have after
    /// slicing with `range`. Panics if the sliced layout has a different number
    /// of dims.
    ///
    /// `K` is the number of items in the array or tuple being used to slice
    /// the tensor. If it must be <= N. If it is less than N, it refers to the
    /// leading dimensions of the tensor and is padded to extract the full
    /// range of the remaining dimensions.
    fn slice<const M: usize, const K: usize, R: IntoSliceItems<K>>(
        &self,
        range: R,
    ) -> NdTensorView<Self::Elem, M> {
        self.view().slice(range)
    }

    /// Return a tensor with data laid out in contiguous order. This will
    /// be a view if the data is already contiguous, or a copy otherwise.
    fn to_contiguous(&self) -> NdTensorBase<Self::Elem, Cow<[Self::Elem]>, N>
    where
        Self::Elem: Clone,
    {
        self.view().to_contiguous()
    }

    /// Return a new contiguous tensor with the same shape and elements as this
    /// view.
    fn to_tensor(&self) -> NdTensor<Self::Elem, N>
    where
        Self::Elem: Clone,
    {
        self.view().to_tensor()
    }

    /// Return an immutable view of this tensor.
    fn view(&self) -> NdTensorView<Self::Elem, N>;
}

/// N-dimensional array, where `N` is specified as generic argument.
///
/// `T` is the element type, `S` is the element storage and `N` is the number
/// of dimensions.
///
/// Most code will not use `NdTensorBase` directly but instead use the type
/// aliases [NdTensor], [NdTensorView] and [NdTensorViewMut]. [NdTensor] owns
/// its elements, and the other two types are views of slices.
///
/// All [NdTensorBase] variants implement the [Layout] trait which provide
/// operations related to the shape and strides of the tensor, and the
/// [NdView] trait which provides common methods applicable to all variants.
#[derive(Clone, Copy, Debug)]
pub struct NdTensorBase<T, S: AsRef<[T]>, const N: usize> {
    data: S,
    layout: NdLayout<N>,

    /// Avoids compiler complaining `T` is unused.
    element_type: PhantomData<T>,
}

impl<T, S: AsRef<[T]>, const N: usize> NdTensorBase<T, S, N> {
    /// Constructs a tensor from the associated storage type and optional
    /// strides.
    ///
    /// If creating an immutable view with strides, prefer
    /// [NdTensorBase::from_slice]. This method enforces that every index in the
    /// tensor maps to a unique element in the data. This upholds Rust's rules
    /// for mutable aliasing. [NdTensorBase::from_slice] does not have this
    /// restriction.
    pub fn from_data(
        data: S,
        shape: [usize; N],
        strides: Option<[usize; N]>,
    ) -> Result<NdTensorBase<T, S, N>, FromDataError> {
        NdLayout::try_from_shape_and_strides(
            shape,
            strides,
            data.as_ref().len(),
            OverlapPolicy::DisallowOverlap,
        )
        .map(|layout| NdTensorBase {
            data,
            layout,
            element_type: PhantomData,
        })
    }

    /// Consume self and return the underlying element storage.
    pub fn into_data(self) -> S {
        self.data
    }

    /// Return the layout which maps indices to offsets in the data.
    pub fn layout(&self) -> &NdLayout<N> {
        &self.layout
    }

    /// Return a new tensor by applying `f` to each element of this tensor.
    pub fn map<F, U>(&self, f: F) -> NdTensor<U, N>
    where
        F: Fn(&T) -> U,
    {
        let data = self.iter().map(f).collect();
        NdTensor::from_data(data, self.shape(), None).unwrap()
    }

    /// Change the layout to put dimensions in the order specified by `dims`.
    ///
    /// This does not modify the order of elements in the data buffer, it just
    /// updates the strides used by indexing.
    pub fn permute(&mut self, dims: [usize; N]) {
        self.layout = self.layout.permuted(dims);
    }

    /// Return a copy of this view that owns its data. For [NdTensorView] this
    /// is different than cloning the view, as that returns a view which has
    /// its own layout, but the same underlying data buffer.
    pub fn to_tensor(&self) -> NdTensor<T, N>
    where
        T: Clone,
    {
        NdTensor {
            data: self.data.as_ref().to_vec(),
            layout: self.layout,
            element_type: PhantomData,
        }
    }

    /// Return an immutable view of this tensor.
    pub fn view(&self) -> NdTensorView<T, N> {
        NdTensorView {
            data: self.data.as_ref(),
            layout: self.layout,
            element_type: PhantomData,
        }
    }
}

impl<T, S: AsRef<[T]>, const N: usize> NdView<N> for NdTensorBase<T, S, N> {
    type Elem = T;

    fn data(&self) -> &[T] {
        self.data.as_ref()
    }

    fn view(&self) -> NdTensorView<T, N> {
        NdTensorBase {
            data: self.data.as_ref(),
            layout: self.layout,
            element_type: PhantomData,
        }
    }
}

/// Convert a slice into a contiguous 1D tensor view.
impl<'a, T, S: AsRef<[T]>> From<&'a S> for NdTensorBase<T, &'a [T], 1> {
    fn from(data: &'a S) -> Self {
        Self::from_slice(data.as_ref(), [data.as_ref().len()], None).unwrap()
    }
}

impl<'a, T, const N: usize> NdTensorView<'a, T, N> {
    /// Constructs a view from a slice and optional strides.
    ///
    /// Unlike [NdTensorBase::from_data], combinations of strides which cause
    /// multiple indices in the tensor to refer to the same data element are
    /// allowed. Since the returned view is immutable, this will not enable
    /// violation of Rust's aliasing rules.
    pub fn from_slice(
        data: &'a [T],
        shape: [usize; N],
        strides: Option<[usize; N]>,
    ) -> Result<Self, FromDataError> {
        NdLayout::try_from_shape_and_strides(
            shape,
            strides,
            data.len(),
            OverlapPolicy::AllowOverlap,
        )
        .map(|layout| NdTensorBase {
            data,
            layout,
            element_type: PhantomData,
        })
    }

    /// Return the element at a given index, without performing any bounds-
    /// checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is valid for the tensor's shape.
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &'a T {
        self.data.get_unchecked(self.layout.offset_unchecked(index))
    }

    /// Return a view of this tensor where indexing checks the bounds of offsets
    /// into the data buffer, but not individual dimensions. This is faster, but
    /// can hide errors.
    pub fn unchecked(&self) -> UncheckedNdTensor<T, &'a [T], N> {
        let base = NdTensorBase {
            data: self.data,
            layout: self.layout,
            element_type: PhantomData,
        };
        UncheckedNdTensor { base }
    }
}

/// Specialized versions of the [NdView] methods for immutable views.
/// These preserve the underlying lifetime of the view in results, allowing for
/// method calls to be chained.
impl<'a, T, const N: usize> NdTensorView<'a, T, N> {
    pub fn as_dyn(&self) -> TensorView<'a, T> {
        TensorView::new(self.data, &self.layout.as_dyn())
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn get(&self, index: [usize; N]) -> Option<&'a T> {
        self.layout
            .try_offset(index)
            .and_then(|offset| self.data.get(offset))
    }

    pub fn iter(&self) -> Iter<'a, T> {
        Iter::new(&self.as_dyn())
    }

    pub fn permuted(&self, dims: [usize; N]) -> NdTensorView<'a, T, N> {
        NdTensorBase {
            data: self.data,
            layout: self.layout.permuted(dims),
            element_type: PhantomData,
        }
    }

    pub fn reshaped<const M: usize>(&self, shape: [usize; M]) -> NdTensorView<'a, T, M> {
        NdTensorBase {
            data: self.data,
            layout: self.layout.reshaped(shape),
            element_type: PhantomData,
        }
    }

    pub fn to_contiguous(&self) -> NdTensorBase<T, Cow<'a, [T]>, N>
    where
        T: Clone,
    {
        if self.is_contiguous() {
            NdTensorBase {
                data: Cow::Borrowed(self.data),
                layout: self.layout,
                element_type: PhantomData,
            }
        } else {
            let data: Vec<T> = self.iter().cloned().collect();
            NdTensorBase {
                data: Cow::Owned(data),
                layout: NdLayout::from_shape(self.layout.shape()),
                element_type: PhantomData,
            }
        }
    }

    pub fn slice<const M: usize, const K: usize, R: IntoSliceItems<K>>(
        &self,
        range: R,
    ) -> NdTensorView<'a, T, M> {
        let range = range.into_slice_items();
        let (offset_range, sliced_layout) = self.layout.slice(&range);
        NdTensorView {
            data: &self.data[offset_range],
            layout: sliced_layout,
            element_type: PhantomData,
        }
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, const N: usize> NdTensorBase<T, S, N> {
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }

    /// Return a mutable reference to the element at a given index.
    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        self.layout
            .try_offset(index)
            .and_then(|offset| self.data_mut().get_mut(offset))
    }

    /// Return the element at a given index, without performing any bounds-
    /// checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is valid for the tensor's shape.
    pub unsafe fn get_unchecked_mut(&mut self, index: [usize; N]) -> &mut T {
        let offset = self.layout.offset_unchecked(index);
        self.data_mut().get_unchecked_mut(offset)
    }

    /// Return a mutable view of this tensor.
    pub fn view_mut(&mut self) -> NdTensorViewMut<T, N> {
        NdTensorViewMut {
            data: self.data.as_mut(),
            layout: self.layout,
            element_type: PhantomData,
        }
    }

    /// Return a mutable view of part of this tensor.
    ///
    /// `M` specifies the number of dimensions that the layout must have after
    /// slicing with `range`. Panics if the sliced layout has a different number
    /// of dims. `K` is the number of elements in the slice range. Must be <= N.
    pub fn slice_mut<const M: usize, const K: usize, R: IntoSliceItems<K>>(
        &mut self,
        range: R,
    ) -> NdTensorViewMut<T, M> {
        self.slice_mut_dyn(&range.into_slice_items())
    }

    /// Return a mutable view of part of this tensor.
    ///
    /// `M` specifies the number of dimensions that the layout must have after
    /// slicing with `range`. Panics if the sliced layout has a different number
    /// of dims.
    pub fn slice_mut_dyn<const M: usize>(&mut self, range: &[SliceItem]) -> NdTensorViewMut<T, M> {
        let (offset_range, sliced_layout) = self.layout.slice(range);
        NdTensorViewMut {
            data: &mut self.data.as_mut()[offset_range],
            layout: sliced_layout,
            element_type: PhantomData,
        }
    }

    /// Return a mutable view of this tensor which uses unchecked indexing.
    ///
    /// See [NdTensorView::unchecked] for more details.
    pub fn unchecked_mut(&mut self) -> UncheckedNdTensor<T, &mut [T], N> {
        let base = NdTensorBase {
            data: self.data.as_mut(),
            layout: self.layout,
            element_type: PhantomData,
        };
        UncheckedNdTensor { base }
    }

    /// Return a view of this tensor with a dynamic dimension count.
    pub fn as_dyn_mut(&mut self) -> TensorViewMut<T> {
        TensorViewMut::new(self.data.as_mut(), &self.layout.as_dyn())
    }

    /// Return a mutable iterator over elements of this tensor.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self.data.as_mut(), &self.layout.as_dyn())
    }

    /// Replace elements of this tensor with `f(element)`.
    ///
    /// This is the in-place version of `map`.
    ///
    /// The order in which elements are visited is unspecified and may not
    /// correspond to the logical order.
    pub fn apply<F: Fn(&T) -> T>(&mut self, f: F) {
        // TODO: Skip unused elements when tensor is not contiguous.
        for val in self.data.as_mut().iter_mut() {
            *val = f(val);
        }
    }

    /// Copy elements from another tensor into this tensor.
    ///
    /// This tensor and `other` must have the same shape.
    pub fn copy_from(&mut self, other: &NdTensorView<T, N>)
    where
        T: Clone,
    {
        assert!(self.shape() == other.shape());
        for (out, x) in zip(self.iter_mut(), other.iter()) {
            *out = x.clone();
        }
    }
}

impl<T: Clone + Default, const N: usize> NdTensorBase<T, Vec<T>, N> {
    /// Create a new tensor with a given shape, contigous layout and all
    /// elements set to zero (or whatever `T::default()` returns).
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::from_element(shape, T::default())
    }

    /// Create a new tensor with a given shape, contiguous layout and all
    /// elements initialized to `element`.
    pub fn from_element(shape: [usize; N], element: T) -> Self {
        let layout = NdLayout::from_shape(shape);
        NdTensorBase {
            data: vec![element; layout.len()],
            layout,
            element_type: PhantomData,
        }
    }
}

impl<T, S1: AsRef<[T]>, S2: AsRef<[T]>, const N: usize> TryFrom<TensorBase<T, S1>>
    for NdTensorBase<T, S2, N>
where
    S1: Into<S2>,
{
    type Error = DimensionError;

    /// Convert a dynamic-dimensional tensor or view into a static-dimensional one.
    ///
    /// Fails if `value` does not have `N` dimensions.
    fn try_from(value: TensorBase<T, S1>) -> Result<Self, Self::Error> {
        let layout: NdLayout<N> = value.layout().try_into()?;
        Ok(NdTensorBase {
            data: value.into_data().into(),
            layout,
            element_type: PhantomData,
        })
    }
}

impl<T, S: AsRef<[T]>, const N: usize> Index<[usize; N]> for NdTensorBase<T, S, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.data.as_ref()[self.layout.offset(index)]
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, const N: usize> IndexMut<[usize; N]> for NdTensorBase<T, S, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let offset = self.layout.offset(index);
        &mut self.data.as_mut()[offset]
    }
}

impl<T, S: AsRef<[T]>, const N: usize> Layout for NdTensorBase<T, S, N> {
    type Index<'a> = [usize; N] where S: 'a, T: 'a;
    type Indices = NdIndices<N>;

    fn ndim(&self) -> usize {
        N
    }

    fn len(&self) -> usize {
        self.layout.len()
    }

    fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    fn shape(&self) -> Self::Index<'_> {
        self.layout.shape()
    }

    fn size(&self, dim: usize) -> usize {
        self.layout.size(dim)
    }

    fn strides(&self) -> Self::Index<'_> {
        self.layout.strides()
    }

    fn stride(&self, dim: usize) -> usize {
        self.layout.stride(dim)
    }

    fn indices(&self) -> Self::Indices {
        self.layout.indices()
    }
}

impl<T, S: AsRef<[T]>> MatrixLayout for NdTensorBase<T, S, 2> {
    fn rows(&self) -> usize {
        self.layout.rows()
    }

    fn cols(&self) -> usize {
        self.layout.cols()
    }

    fn row_stride(&self) -> usize {
        self.layout.row_stride()
    }

    fn col_stride(&self) -> usize {
        self.layout.col_stride()
    }
}

/// Provides methods specific to 2D tensors (matrices).
impl<T, S: AsRef<[T]>> NdTensorBase<T, S, 2> {
    /// Return a new view which transposes the columns and rows.
    pub fn transposed(self) -> Self {
        NdTensorBase {
            data: self.data,
            layout: self.layout.transposed(),
            element_type: PhantomData,
        }
    }
}

/// Variant of [NdTensorBase] which owns its elements, using a `Vec<T>` as
/// the backing storage.
pub type NdTensor<T, const N: usize> = NdTensorBase<T, Vec<T>, N>;

/// Variant of [NdTensorBase] which borrows its elements from an [NdTensor].
///
/// Conceptually the relationship between [NdTensorView] and [NdTensor] is
/// similar to that between `[T]` and `Vec<T>`. They share the same element
/// buffer, but views can have distinct layouts, with some limitations.
pub type NdTensorView<'a, T, const N: usize> = NdTensorBase<T, &'a [T], N>;

/// Variant of [NdTensorBase] which mutably borrows its elements from an
/// [NdTensor].
///
/// This is similar to [NdTensorView], except elements in the underyling
/// Tensor can be modified through it.
pub type NdTensorViewMut<'a, T, const N: usize> = NdTensorBase<T, &'a mut [T], N>;

/// Alias for viewing a slice as a 2D matrix.
pub type Matrix<'a, T = f32> = NdTensorBase<T, &'a [T], 2>;

/// Alias for viewing a mutable slice as a 2D matrix.
pub type MatrixMut<'a, T = f32> = NdTensorBase<T, &'a mut [T], 2>;

/// A variant of NdTensor which does not bounds-check individual dimensions
/// when indexing, although the computed offset into the underlying storage
/// is still bounds-checked.
///
/// Using unchecked indexing is faster, at the cost of not catching errors
/// in specific indices.
pub struct UncheckedNdTensor<T, S: AsRef<[T]>, const N: usize> {
    base: NdTensorBase<T, S, N>,
}

impl<T, S: AsRef<[T]>, const N: usize> Index<[usize; N]> for UncheckedNdTensor<T, S, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.base.data.as_ref()[self.base.layout.offset_unchecked(index)]
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, const N: usize> IndexMut<[usize; N]>
    for UncheckedNdTensor<T, S, N>
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let offset = self.base.layout.offset_unchecked(index);
        &mut self.base.data.as_mut()[offset]
    }
}

impl<T> FromIterator<T> for NdTensor<T, 1> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<_> = FromIterator::from_iter(iter);
        let len = data.len();
        NdTensor::from_data(data, [len], None).unwrap()
    }
}

impl<T: PartialEq, S1: AsRef<[T]>, S2: AsRef<[T]>, const N: usize> PartialEq<NdTensorBase<T, S2, N>>
    for NdTensorBase<T, S1, N>
{
    fn eq(&self, other: &NdTensorBase<T, S2, N>) -> bool {
        self.shape() == other.shape() && self.iter().eq(other.iter())
    }
}

#[cfg(test)]
mod tests {
    use crate::errors::{DimensionError, FromDataError};
    use crate::{
        Layout, MatrixLayout, NdTensor, NdTensorView, NdTensorViewMut, NdView, SliceItem, Tensor,
        View,
    };

    /// Return elements of `tensor` in their logical order.
    fn tensor_elements<T: Clone, const N: usize>(tensor: NdTensorView<T, N>) -> Vec<T> {
        tensor.iter().cloned().collect()
    }

    #[test]
    fn test_ndtensor_apply() {
        let mut tensor = NdTensor::from_data(vec![1, 2, 3, 4], [2, 2], None).unwrap();
        tensor.apply(|x| x * 2);
        assert_eq!(tensor_elements(tensor.view()), &[2, 4, 6, 8]);
    }

    // Test conversion of a static-dim tensor with default strides, to a
    // dynamic dim tensor.
    #[test]
    fn test_ndtensor_as_dyn() {
        let tensor = NdTensor::from_data(vec![1, 2, 3, 4], [2, 2], None).unwrap();
        let dyn_tensor = tensor.as_dyn();
        assert_eq!(tensor.shape(), dyn_tensor.shape());
        assert_eq!(tensor.data(), dyn_tensor.data());
    }

    #[test]
    fn test_ndtensor_as_dyn_mut() {
        let mut tensor = NdTensor::from_data(vec![1, 2, 3, 4], [2, 2], None).unwrap();
        let mut dyn_tensor = tensor.as_dyn_mut();
        assert_eq!(dyn_tensor.shape(), [2, 2]);
        assert_eq!(dyn_tensor.data_mut(), &[1, 2, 3, 4]);
    }

    // Test conversion of a static-dim tensor with broadcasting strides (ie.
    // some strides are 0), to a dynamic dim tensor.
    #[test]
    fn test_ndtensor_as_dyn_broadcast() {
        let data = [1, 2, 3, 4];
        let view = NdTensorView::from_slice(&data, [4, 4], Some([0, 1])).unwrap();
        let dyn_view = view.as_dyn();
        let elements: Vec<_> = dyn_view.iter().copied().collect();
        assert_eq!(elements, &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn test_ndtensor_copy_from() {
        let x = NdTensor::from_data(vec![1, 2, 3, 4], [2, 2], None).unwrap();
        let mut y = NdTensor::zeros(x.shape());

        y.copy_from(&x.view());

        assert_eq!(y, x);
    }

    #[test]
    fn test_ndtensor_from_data() {
        let data = vec![1., 2., 3., 4.];
        let view = NdTensorView::<f32, 2>::from_data(&data, [2, 2], None).unwrap();
        assert_eq!(view.data(), data);
        assert_eq!(view.shape(), [2, 2]);
        assert_eq!(view.strides(), [2, 1]);
    }

    #[test]
    fn test_ndtensor_from_data_custom_strides() {
        struct Case {
            data: Vec<f32>,
            shape: [usize; 2],
            strides: [usize; 2],
        }

        let cases = [
            // Contiguous view (no gaps, shortest stride last)
            Case {
                data: vec![1., 2., 3., 4.],
                shape: [2, 2],
                strides: [2, 1],
            },
            // Transposed view (reversed strides)
            Case {
                data: vec![1., 2., 3., 4.],
                shape: [2, 2],
                strides: [1, 2],
            },
            // Sliced view (gaps between elements)
            Case {
                data: vec![1.; 10],
                shape: [2, 2],
                strides: [4, 2],
            },
            // Sliced view (gaps between rows)
            Case {
                data: vec![1.; 10],
                shape: [2, 2],
                strides: [4, 1],
            },
        ];

        for case in cases {
            let result =
                NdTensorView::<f32, 2>::from_data(&case.data, case.shape, Some(case.strides))
                    .unwrap();
            assert_eq!(result.data(), case.data);
            assert_eq!(result.shape(), case.shape);
            assert_eq!(result.strides(), case.strides);
        }
    }

    #[test]
    fn test_ndtensor_from_iterator() {
        let tensor: NdTensor<f32, 1> = [1., 2., 3., 4.].into_iter().collect();
        assert_eq!(tensor_elements(tensor.view()), [1., 2., 3., 4.]);
    }

    #[test]
    fn test_slice_into_1d_ndtensor() {
        let data = &[1., 2., 3., 4.];
        let view: NdTensorView<f32, 1> = data.into();
        assert_eq!(view.data(), data);
        assert_eq!(view.shape(), [4]);
        assert_eq!(view.strides(), [1]);
    }

    #[test]
    fn test_ndtensor_from_slice() {
        let data = vec![1., 2., 3., 4.];
        let view = NdTensorView::<f32, 2>::from_slice(&data, [2, 2], None).unwrap();
        assert_eq!(view.data(), data);
        assert_eq!(view.shape(), [2, 2]);
        assert_eq!(view.strides(), [2, 1]);
    }

    #[test]
    fn test_ndtensor_from_slice_fails_if_too_short() {
        let data = vec![1., 2., 3., 4.];
        let result = NdTensorView::<f32, 2>::from_slice(&data, [3, 3], Some([2, 1]));
        assert_eq!(result.err(), Some(FromDataError::StorageTooShort));
    }

    #[test]
    fn test_ndtensor_from_data_fails_if_overlap() {
        struct Case {
            data: Vec<f32>,
            shape: [usize; 3],
            strides: [usize; 3],
        }

        let cases = [
            // Broadcasting view (zero strides)
            Case {
                data: vec![1., 2., 3., 4.],
                shape: [10, 2, 2],
                strides: [0, 2, 1],
            },
            // Case where there is actually no overlap, but `from_data` fails
            // with a `MayOverlap` error due to the conservative logic it uses.
            Case {
                data: vec![1.; (3 * 3) + (3 * 4) + 1],
                shape: [1, 4, 4],
                strides: [20, 3, 4],
            },
        ];

        for case in cases {
            let result =
                NdTensorView::<f32, 3>::from_data(&case.data, case.shape, Some(case.strides));
            assert_eq!(result.err(), Some(FromDataError::MayOverlap));
        }
    }

    #[test]
    fn test_ndtensor_from_slice_allows_overlap() {
        let data = vec![1., 2., 3., 4.];
        let result = NdTensorView::<f32, 3>::from_slice(&data, [10, 2, 2], Some([0, 2, 1]));
        assert!(result.is_ok());
    }

    #[test]
    fn test_ndtensor_try_from_tensor() {
        // Tensor -> NdTensor
        let tensor = Tensor::zeros(&[1, 10, 20]);
        let ndtensor: NdTensor<i32, 3> = tensor.clone().try_into().unwrap();
        assert_eq!(ndtensor.data(), tensor.data());
        assert_eq!(ndtensor.shape(), tensor.shape());
        assert_eq!(ndtensor.strides(), tensor.strides());

        // Failed Tensor -> NdTensor
        let matrix: Result<NdTensor<i32, 2>, _> = tensor.clone().try_into();
        assert_eq!(matrix, Err(DimensionError {}));

        // TensorView -> NdTensorView
        let ndview: NdTensorView<i32, 3> = tensor.view().try_into().unwrap();
        assert_eq!(ndview.data(), tensor.data());
        assert_eq!(ndview.shape(), tensor.shape());
        assert_eq!(ndview.strides(), tensor.strides());

        // TensorViewMut -> NdTensorViewMut
        let mut tensor = Tensor::zeros(&[1, 10, 20]);
        let mut ndview: NdTensorViewMut<i32, 3> = tensor.view_mut().try_into().unwrap();
        ndview[[0, 0, 0]] = 1;
        assert_eq!(tensor[[0, 0, 0]], 1);
    }

    #[test]
    fn test_ndtensor_get() {
        let tensor = NdTensor::<i32, 3>::zeros([5, 10, 15]);

        assert_eq!(tensor.get([0, 0, 0]), Some(&0));
        assert_eq!(tensor.get([4, 9, 14]), Some(&0));
        assert_eq!(tensor.get([5, 9, 14]), None);
        assert_eq!(tensor.get([4, 10, 14]), None);
        assert_eq!(tensor.get([4, 9, 15]), None);
    }

    #[test]
    fn test_ndtensor_get_mut() {
        let mut tensor = NdTensor::<i32, 3>::zeros([5, 10, 15]);

        assert_eq!(tensor.get_mut([0, 0, 0]), Some(&mut 0));
        assert_eq!(tensor.get_mut([4, 9, 14]), Some(&mut 0));
        assert_eq!(tensor.get_mut([5, 9, 14]), None);
        assert_eq!(tensor.get_mut([4, 10, 14]), None);
        assert_eq!(tensor.get_mut([4, 9, 15]), None);
    }

    #[test]
    fn test_ndtensor_get_unchecked() {
        let tensor = NdTensor::<i32, 3>::zeros([5, 10, 15]);
        let tensor = tensor.view();
        unsafe {
            assert_eq!(tensor.get_unchecked([0, 0, 0]), &0);
            assert_eq!(tensor.get_unchecked([4, 9, 14]), &0);
        }
    }

    #[test]
    fn test_ndtensor_get_unchecked_mut() {
        let mut tensor = NdTensor::<i32, 3>::zeros([5, 10, 15]);
        unsafe {
            assert_eq!(tensor.get_unchecked_mut([0, 0, 0]), &0);
            assert_eq!(tensor.get_unchecked_mut([4, 9, 14]), &0);
        }
    }

    #[test]
    fn test_ndtensor_iter() {
        let tensor = NdTensor::<i32, 2>::from_data(vec![1, 2, 3, 4], [2, 2], None).unwrap();
        let elements: Vec<_> = tensor.iter().copied().collect();
        assert_eq!(elements, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_ndtensor_iter_mut() {
        let mut tensor = NdTensor::<i32, 2>::zeros([2, 2]);
        tensor
            .iter_mut()
            .enumerate()
            .for_each(|(i, el)| *el = i as i32);
        let elements: Vec<_> = tensor.iter().copied().collect();
        assert_eq!(elements, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_ndtensor_map() {
        let tensor = NdTensor::<i32, 2>::from_data(vec![1, 2, 3, 4], [2, 2], None).unwrap();
        let doubled = tensor.map(|x| x * 2);
        assert_eq!(tensor_elements(doubled.view()), &[2, 4, 6, 8]);
    }

    #[test]
    fn test_ndtensor_to_owned() {
        let data = vec![1., 2., 3., 4.];
        let view = NdTensorView::<f32, 2>::from_slice(&data, [2, 2], None).unwrap();
        let owned = view.to_tensor();
        assert_eq!(owned.shape(), view.shape());
        assert_eq!(owned.strides(), view.strides());
        assert_eq!(owned.data(), view.data());
    }

    #[test]
    fn test_ndtensor_partial_eq() {
        let a = NdTensor::from_data(vec![1, 2, 3, 4], [2, 2], None).unwrap();
        let b = NdTensor::from_data(vec![1, 2, 3, 4], [2, 2], None).unwrap();
        let c = NdTensor::from_data(vec![1, 2, 3, 4], [1, 4], None).unwrap();
        let d = NdTensor::from_data(vec![1, 2, 3, 5], [2, 2], None).unwrap();

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn test_ndtensor_permuted() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::from(&data).reshaped([2, 2]);
        let transposed = view.permuted([1, 0]);
        assert_eq!(tensor_elements(transposed), &[1, 3, 2, 4]);

        let transposed = transposed.permuted([1, 0]);
        assert_eq!(tensor_elements(transposed), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_ndtensor_permute() {
        let data = vec![1, 2, 3, 4];
        let mut view = NdTensorView::from(&data).reshaped([2, 2]);
        view.permute([1, 0]);
        assert_eq!(tensor_elements(view), &[1, 3, 2, 4]);
        view.permute([1, 0]);
        assert_eq!(tensor_elements(view), &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "permutation is invalid")]
    fn test_ndtensor_permuted_panics_if_dims_invalid() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::from(&data).reshaped([2, 2]);
        view.permuted([2, 0]);
    }

    #[test]
    fn test_ndtensor_reshaped() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::from(&data);
        let matrix = view.reshaped([2, 2]);
        assert_eq!(matrix.shape(), [2, 2]);
        assert_eq!(tensor_elements(matrix), &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "new shape must have same number of elements as current shape")]
    fn test_ndtensor_reshaped_panics_if_product_not_equal() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::from(&data);
        view.reshaped([2, 3]);
    }

    #[test]
    #[should_panic(expected = "can only reshape a contiguous tensor")]
    fn test_ndtensor_reshaped_panics_if_not_contiguous() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::from(&data).reshaped([2, 2]);
        let transposed = view.transposed();
        transposed.reshaped([4]);
    }

    #[test]
    fn test_ndtensor_to_contiguous() {
        let x = NdTensor::from_data(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], None).unwrap();
        let y = x.to_contiguous();
        assert!(y.is_contiguous());
        assert_eq!(y.data().as_ptr(), x.data().as_ptr());

        let x = x.permuted([1, 0]);
        assert!(!x.is_contiguous());

        let y = x.to_contiguous();
        assert!(y.is_contiguous());
        assert_ne!(y.data().as_ptr(), x.data().as_ptr());
        assert_eq!(y.data(), x.iter().copied().collect::<Vec<_>>());
    }

    #[test]
    fn test_ndtensor_transposed() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::from(&data).reshaped([2, 2]);
        assert_eq!(tensor_elements(view), &[1, 2, 3, 4]);
        let view = view.transposed();
        assert_eq!(tensor_elements(view), &[1, 3, 2, 4]);
    }

    #[test]
    fn test_ndtensor_slice() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let view = NdTensorView::from(&data).reshaped([4, 4]);
        let slice: NdTensorView<_, 2> = view.slice([1..3, 1..3]);
        assert_eq!(tensor_elements(slice), &[6, 7, 10, 11]);
    }

    #[test]
    fn test_ndtensor_slice_step() {
        let data: Vec<i32> = (0..25).collect();
        let view = NdTensorView::from(&data).reshaped([5, 5]);
        let slice: NdTensorView<_, 2> =
            view.slice((SliceItem::range(0, None, 2), SliceItem::range(0, None, 2)));
        assert_eq!(slice.shape(), [3, 3]);
        assert_eq!(
            slice.iter().copied().collect::<Vec<_>>(),
            [0, 2, 4, 10, 12, 14, 20, 22, 24]
        );
    }

    #[test]
    #[should_panic(expected = "sliced dims != 3")]
    fn test_ndtensor_slice_wrong_dims() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::from(&data).reshaped([2, 2]);
        view.slice::<3, 2, _>([0..2, 0..2]);
    }

    #[test]
    fn test_ndtensor_slice_mut() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut view = NdTensorViewMut::<i32, 2>::from_data(&mut data, [4, 4], None).unwrap();
        let mut slice = view.slice_mut([1..3, 1..3]);
        slice[[0, 0]] = -1;
        slice[[0, 1]] = -2;
        slice[[1, 0]] = -3;
        slice[[1, 1]] = -4;
        assert_eq!(
            tensor_elements(view.view()),
            &[1, 2, 3, 4, 5, -1, -2, 8, 9, -3, -4, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    #[should_panic(expected = "sliced dims != 3")]
    fn test_ndtensor_slice_mut_wrong_dims() {
        let mut data = vec![1, 2, 3, 4];
        let mut view = NdTensorViewMut::<i32, 2>::from_data(&mut data, [2, 2], None).unwrap();
        view.slice_mut::<3, 2, _>([0..2, 0..2]);
    }

    #[test]
    fn test_matrix_layout() {
        let data = vec![1., 2., 3., 4.];
        let mat = NdTensorView::from(&data).reshaped([2, 2]);
        assert_eq!(mat.data(), data);
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.row_stride(), 2);
        assert_eq!(mat.col_stride(), 1);
    }
}
