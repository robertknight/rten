use std::borrow::Cow;
use std::fmt::Debug;
use std::io;
use std::io::Write;
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};

use crate::errors::SliceError;
use crate::iterators::{
    AxisChunks, AxisChunksMut, AxisIter, AxisIterMut, BroadcastIter, InnerIter, InnerIterMut, Iter,
    IterMut, Lanes, LanesMut, Offsets,
};
use crate::layout::{DynLayout, Layout};
use crate::ndtensor::{NdTensorBase, NdTensorView, NdTensorViewMut};
use crate::range::{IntoSliceItems, SliceItem};
use crate::rng::XorShiftRng;

/// Multi-dimensional array view with a dynamic dimension count. This trait
/// includes operations that are available on tensors that own their data
/// ([Tensor]) as well as views ([TensorView], [TensorViewMut]).
///
/// [TensorView] implements specialized versions of these methods as
/// inherent methods, which preserve lifetimes on the result.
pub trait View: Layout {
    /// The data type of elements in this tensor.
    type Elem;

    /// Return an iterator over slices of this tensor along a given axis.
    fn axis_iter(&self, dim: usize) -> AxisIter<Self::Elem> {
        self.view().axis_iter(dim)
    }

    /// Return an iterator over slices of this tensor along a given axis.
    fn axis_chunks(&self, dim: usize, chunk_size: usize) -> AxisChunks<Self::Elem> {
        self.view().axis_chunks(dim, chunk_size)
    }

    /// Create a view of this tensor which is broadcasted to `shape`.
    ///
    /// A broadcasted view behaves as if the underlying tensor had the broadcasted
    /// shape, repeating elements as necessary to fill the given dimensions.
    /// Broadcasting is only possible if the actual and broadcast shapes are
    /// compatible according to ONNX's rules. See
    /// <https://github.com/onnx/onnx/blob/main/docs/Operators.md>.
    ///
    /// See also <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>
    /// for worked examples of how broadcasting works.
    ///
    /// Panics if `shape` is not broadcast-compatible with the current shape.
    fn broadcast(&self, shape: &[usize]) -> TensorView<Self::Elem> {
        self.view().broadcast(shape)
    }

    /// Return an iterator over the elements of this view broadcasted to
    /// `shape`.
    ///
    /// This is functionally the same as `self.broadcast(shape).iter()` but
    /// has some additional optimizations for common broadcasting uses - eg.
    /// when the sequence of elements in the broadcasted view is equivalent to
    /// cycling the original view using `tensor.iter().cycle()`.
    fn broadcast_iter(&self, shape: &[usize]) -> BroadcastIter<Self::Elem> {
        self.view().broadcast_iter(shape)
    }

    /// Return the underlying data of the tensor as a slice, if it is contiguous.
    fn data(&self) -> Option<&[Self::Elem]>;

    /// Return an iterator over views of the innermost N dimensions of this
    /// tensor.
    fn inner_iter<const N: usize>(&self) -> InnerIter<Self::Elem, N> {
        self.view().inner_iter()
    }

    /// Returns the single item if this tensor is a 0-dimensional tensor
    /// (ie. a scalar)
    fn item(&self) -> Option<&Self::Elem> {
        self.view().item()
    }

    /// Return the element at a given index, or `None` if the index is out of
    /// bounds in any dimension.
    #[inline]
    fn get<I: AsRef<[usize]>>(&self, index: I) -> Option<&Self::Elem> {
        self.view().get(index)
    }

    /// Return an iterator over elements of this tensor, in their logical order.
    fn iter(&self) -> Iter<Self::Elem> {
        Iter::new(&self.view())
    }

    /// Return an iterator over all 1D slices ("lanes") along a given axis.
    ///
    /// Each slice is an iterator over the elements in that lane.
    fn lanes(&self, dim: usize) -> Lanes<Self::Elem> {
        Lanes::new(self.view(), dim)
    }

    /// Return a copy of this tensor with each element replaced by `f(element)`.
    ///
    /// The order in which elements are visited is unspecified and may not
    /// correspond to the logical order.
    fn map<F, U>(&self, f: F) -> Tensor<U>
    where
        F: Fn(&Self::Elem) -> U,
    {
        let data = if let Some(data) = self.data() {
            data.iter().map(f).collect()
        } else {
            self.iter().map(f).collect()
        };
        Tensor {
            data,
            layout: DynLayout::new(self.shape().as_ref()),
            element_type: PhantomData,
        }
    }

    /// Return a view with a static rank.
    ///
    /// Panics if the rank of this tensor is not `N`.
    fn nd_view<const N: usize>(&self) -> NdTensorView<Self::Elem, N> {
        self.view().nd_view()
    }

    /// Return a new view with the given shape.
    ///
    /// The current view must be contiguous and the new shape must have the
    /// same product as the current shape.
    fn reshaped(&self, shape: &[usize]) -> TensorView<Self::Elem> {
        self.view().reshaped(shape)
    }

    /// Return a new view with the dimensions re-ordered according to `dims`.
    fn permuted(&self, dims: &[usize]) -> TensorView<Self::Elem> {
        self.view().permuted(dims)
    }

    /// Return a view of part of this tensor.
    ///
    /// `range` specifies the indices or ranges of this tensor to include in the
    /// returned view. If the range has fewer dimensions than the tensor, they
    /// refer to the leading dimensions.
    ///
    /// See [IntoSliceItems] for a description of how slices can be specified.
    /// Slice ranges are currently restricted to use positive steps. In other
    /// words, NumPy-style slicing with negative steps is not supported.
    fn slice<R: IntoSliceItems>(&self, range: R) -> TensorView<Self::Elem> {
        self.view().slice(range)
    }

    /// Variant of [View::slice] which returns an error if the slice spec
    /// is invalid, instead of panicking.
    fn try_slice<R: IntoSliceItems>(&self, range: R) -> Result<TensorView<Self::Elem>, SliceError> {
        self.view().try_slice(range)
    }

    /// Return an iterator over a slice of this tensor.
    ///
    /// This is similar to `self.slice(range).iter()` except that it
    /// returns an iterator directly instead of creating an intermediate view.
    /// Also slicing with this method is more flexible as negative steps are
    /// supported for items in `range`.
    fn slice_iter(&self, range: &[SliceItem]) -> Iter<Self::Elem> {
        self.view().slice_iter(range)
    }

    /// Return a view of this tensor with all dimensions of size 1 removed.
    fn squeezed(&self) -> TensorView<Self::Elem> {
        self.view().squeezed()
    }

    /// Return a tensor with data laid out in contiguous order. This will
    /// be a view if the data is already contiguous, or a copy otherwise.
    fn to_contiguous(&self) -> TensorBase<Self::Elem, Cow<[Self::Elem]>>
    where
        Self::Elem: Clone,
    {
        self.view().to_contiguous()
    }

    /// Return a new contiguous tensor with the same shape and elements as this
    /// view.
    fn to_tensor(&self) -> Tensor<Self::Elem>
    where
        Self::Elem: Clone,
    {
        self.view().to_tensor()
    }

    /// Return a copy of the elements of this tensor in their logical order
    /// as a vector.
    ///
    /// This is equivalent to `self.iter().cloned().collect()` but faster
    /// when the tensor is already contiguous or has a small number (<= 4)
    /// dimensions.
    fn to_vec(&self) -> Vec<Self::Elem>
    where
        Self::Elem: Clone,
    {
        self.view().to_vec()
    }

    /// Return a new view with the order of dimensions reversed.
    fn transposed(&self) -> TensorView<Self::Elem> {
        self.view().transposed()
    }

    /// Return an immutable view of this tensor.
    fn view(&self) -> TensorView<Self::Elem>;
}

/// N-dimensional array, where `N` is determined at runtime based on the shape
/// that is specified when the tensor is constructed.
///
/// `T` is the element type and `S` is the element storage.
///
/// Most code will not use `TensorBase` directly but instead use the type
/// aliases [Tensor], [TensorView] and [TensorViewMut]. [Tensor] owns
/// its elements, and the other two types are views of slices.
///
/// All [TensorBase] variants implement the [Layout] trait which provide
/// operations related to the shape and strides of the tensor, and the
/// [View] trait which provides common methods applicable to all variants.
#[derive(Debug)]
pub struct TensorBase<T, S: AsRef<[T]>> {
    data: S,
    layout: DynLayout,
    element_type: PhantomData<T>,
}

/// Variant of [TensorBase] which borrows its elements from a [Tensor].
///
/// Conceptually the relationship between [TensorView] and [Tensor] is similar
/// to that between `[T]` and `Vec<T>`. They share the same element buffer, but
/// views can have distinct layouts, with some limitations.
pub type TensorView<'a, T = f32> = TensorBase<T, &'a [T]>;

/// Variant of [TensorBase] which mutably borrows its elements from a [Tensor].
///
/// This is similar to [TensorView], except elements in the underyling
/// Tensor can be modified through it.
pub type TensorViewMut<'a, T = f32> = TensorBase<T, &'a mut [T]>;

/// Trait for sources of random data for tensors, for use with [Tensor::rand].
pub trait RandomSource<T> {
    /// Generate the next random value.
    fn next(&mut self) -> T;
}

impl<T, S: AsRef<[T]>> TensorBase<T, S> {
    /// Create a new tensor with a given layout and storage.
    pub(crate) fn new(data: S, layout: &DynLayout) -> Self {
        TensorBase {
            data,
            layout: layout.clone(),
            element_type: PhantomData,
        }
    }

    /// Create a new tensor from a given shape and set of elements. No copying
    /// is required.
    pub fn from_data<D: Into<S>>(shape: &[usize], data: D) -> Self {
        let data = data.into();
        assert!(
            shape[..].iter().product::<usize>() == data.as_ref().len(),
            "Number of elements given by shape {:?} does not match data length {}",
            shape,
            data.as_ref().len()
        );
        TensorBase {
            data,
            layout: DynLayout::new(shape),
            element_type: PhantomData,
        }
    }

    /// Consume self and return the underlying element buffer.
    ///
    /// As with [TensorBase::data], there is no guarantee about the ordering of
    /// elements.
    pub fn into_data(self) -> S {
        self.data
    }

    /// Return an immutable view of this tensor.
    ///
    /// Views share the same element array, but can have an independent layout,
    /// with some limitations.
    pub fn view(&self) -> TensorView<T> {
        TensorView::new(self.data.as_ref(), &self.layout)
    }

    /// Change the layout to put dimensions in the order specified by `dims`.
    ///
    /// This does not modify the order of elements in the data buffer, it just
    /// updates the strides used by indexing.
    pub fn permute(&mut self, dims: &[usize]) {
        self.layout.permute(dims);
    }

    /// Move the index at axis `from` to `to`, keeping the relative order of
    /// other dimensions the same. This is like NumPy's `moveaxis` function.
    ///
    /// Panics if the `from` or `to` axes are >= `self.ndim()`.
    pub fn move_axis(&mut self, from: usize, to: usize) {
        self.layout.move_axis(from, to);
    }

    /// Reverse the order of dimensions.
    ///
    /// This does not modify the order of elements in the data buffer, it just
    /// changes the strides used by indexing.
    pub fn transpose(&mut self) {
        self.layout.transpose();
    }

    /// Insert a dimension of size one at index `dim`.
    pub fn insert_dim(&mut self, dim: usize) {
        self.layout.insert_dim(dim);
    }

    /// Return the layout which maps indices to offsets in the data.
    pub fn layout(&self) -> &DynLayout {
        &self.layout
    }

    /// Return an iterator over offsets of elements in this tensor, in their
    /// logical order.
    ///
    /// See also the notes for `slice_offsets`.
    #[cfg(test)]
    fn offsets(&self) -> Offsets {
        Offsets::new(self.layout())
    }

    /// Return an iterator over offsets of elements in this tensor.
    ///
    /// Note that the offset order of the returned iterator will become incorrect
    /// if the tensor's layout is modified during iteration.
    pub(crate) fn slice_offsets(&self, range: &[SliceItem]) -> Offsets {
        Offsets::slice(self.layout(), range)
    }
}

/// Specialized versions of the [View] methods for immutable views.
/// These preserve the underlying lifetime of the view in results, allowing for
/// method calls to be chained.
impl<'a, T> TensorView<'a, T> {
    pub fn axis_iter(&self, dim: usize) -> AxisIter<'a, T> {
        AxisIter::new(self, dim)
    }

    pub fn axis_chunks(&self, dim: usize, chunk_size: usize) -> AxisChunks<'a, T> {
        AxisChunks::new(self, dim, chunk_size)
    }

    pub fn broadcast(&self, shape: &[usize]) -> TensorView<'a, T> {
        Self {
            data: self.data,
            layout: self.layout.broadcast(shape),
            element_type: PhantomData,
        }
    }

    pub fn broadcast_iter(&self, shape: &[usize]) -> BroadcastIter<'a, T> {
        assert!(
            self.can_broadcast_to(shape),
            "Cannot broadcast to specified shape"
        );
        BroadcastIter::new(self, shape)
    }

    pub fn data(&self) -> Option<&'a [T]> {
        self.is_contiguous().then_some(self.data)
    }

    /// Return the view's underlying data as a slice, without checking whether
    /// it is contiguous.
    ///
    /// # Safety
    ///
    /// It is the caller's responsibility not to access elements in the slice
    /// which are not part of this view.
    pub unsafe fn data_unchecked(&self) -> &'a [T] {
        self.data
    }

    #[inline]
    fn get<I: AsRef<[usize]>>(&self, index: I) -> Option<&'a T> {
        let offset = self.layout.try_offset(index)?;
        Some(&self.data[offset])
    }

    pub fn inner_iter<const N: usize>(&self) -> InnerIter<'a, T, N> {
        InnerIter::new(self.clone())
    }

    pub fn iter(&self) -> Iter<'a, T> {
        Iter::new(self)
    }

    pub fn item(&self) -> Option<&'a T> {
        match self.ndim() {
            0 => Some(&self.data[0]),
            _ if self.len() == 1 => self.iter().next(),
            _ => None,
        }
    }

    pub fn lanes(&self, dim: usize) -> Lanes<'a, T> {
        Lanes::new(self.clone(), dim)
    }

    pub fn nd_view<const N: usize>(&self) -> NdTensorView<'a, T, N> {
        assert!(self.ndim() == N);
        let shape: [usize; N] = self.shape().try_into().unwrap();
        let strides: [usize; N] = self.strides().try_into().unwrap();
        NdTensorView::from_slice(self.data, shape, Some(strides)).unwrap()
    }

    pub fn permuted(&self, dims: &[usize]) -> TensorView<'a, T> {
        Self {
            data: self.data,
            layout: self.layout.permuted(dims),
            element_type: PhantomData,
        }
    }

    pub fn reshaped(&self, shape: &[usize]) -> TensorView<'a, T> {
        Self {
            data: self.data,
            layout: self.layout.reshaped(shape),
            element_type: PhantomData,
        }
    }

    /// Change the layout of this view to have the given shape.
    ///
    /// The current view must be contiguous and the new shape must have the
    /// same product as the current shape.
    pub fn reshape(&mut self, shape: &[usize]) {
        self.layout.reshape(shape);
    }

    pub fn slice<R: IntoSliceItems>(&self, range: R) -> TensorView<'a, T> {
        self.try_slice(range.into_slice_items().as_ref()).unwrap()
    }

    pub fn try_slice<R: IntoSliceItems>(&self, range: R) -> Result<TensorView<'a, T>, SliceError> {
        let (offset_range, layout) = self.layout.try_slice(range.into_slice_items().as_ref())?;
        Ok(TensorBase {
            data: &self.data[offset_range],
            layout,
            element_type: PhantomData,
        })
    }

    pub fn slice_iter(&self, range: &[SliceItem]) -> Iter<'a, T> {
        Iter::slice(self, range)
    }

    pub fn squeezed(&self) -> TensorView<'a, T> {
        TensorBase {
            data: self.data,
            layout: self.layout.squeezed(),
            element_type: PhantomData,
        }
    }

    pub fn to_contiguous(&self) -> TensorBase<T, Cow<'a, [T]>>
    where
        T: Clone,
    {
        if self.is_contiguous() {
            TensorBase {
                data: Cow::Borrowed(self.data),
                layout: self.layout.clone(),
                element_type: PhantomData,
            }
        } else {
            let data = self.to_vec();
            TensorBase {
                data: Cow::Owned(data),
                layout: DynLayout::new(self.layout().shape()),
                element_type: PhantomData,
            }
        }
    }

    pub fn transposed(&self) -> TensorView<'a, T> {
        Self {
            data: self.data,
            layout: self.layout.transposed(),
            element_type: PhantomData,
        }
    }
}

impl<T, S: AsRef<[T]>> Layout for TensorBase<T, S> {
    type Index<'a> = <DynLayout as Layout>::Index<'a>;
    type Indices = <DynLayout as Layout>::Indices;

    /// Return the number of dimensions.
    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn offset(&self, index: &[usize]) -> usize {
        self.layout.offset(index)
    }

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.layout.len()
    }

    /// Returns true if the array has no elements.
    fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    /// Returns an array of the sizes of each dimension.
    fn shape(&self) -> Self::Index<'_> {
        self.layout.shape()
    }

    /// Returns the size of the dimension `dim`.
    fn size(&self, dim: usize) -> usize {
        self.layout.size(dim)
    }

    /// Returns an array of the strides of each dimension.
    fn strides(&self) -> Self::Index<'_> {
        self.layout.strides()
    }

    /// Returns the offset between adjacent indices along dimension `dim`.
    fn stride(&self, dim: usize) -> usize {
        self.layout.stride(dim)
    }

    /// Return an iterator over all valid indices in this tensor.
    fn indices(&self) -> Self::Indices {
        self.layout.indices()
    }
}

impl<T, S: AsRef<[T]>> View for TensorBase<T, S> {
    type Elem = T;

    fn data(&self) -> Option<&[T]> {
        self.is_contiguous().then_some(self.data.as_ref())
    }

    fn to_tensor(&self) -> Tensor<T>
    where
        T: Clone,
    {
        Tensor::from_data(self.shape(), self.to_vec())
    }

    fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        if let Some(data) = self.data() {
            data.to_vec()
        } else {
            // This branch is equivalent to
            // `x.iter().cloned().collect::<Vec<_>>()` but uses a faster
            // iteration method that is optimized for tensors with few (<= 4)
            // dimensions.
            let mut data = Vec::with_capacity(self.len());
            let ptr: *mut T = data.as_mut_ptr();

            let mut offset = 0;
            fast_for_each_element(self.view(), |elt| {
                // Safety: `fast_for_each_element` calls fn `self.len()` times,
                // matching the buffer capacity.
                unsafe { *ptr.add(offset) = elt.clone() };
                offset += 1;
            });

            // Safety: Length here matches capacity passed to `Vec::with_capacity`.
            unsafe { data.set_len(self.len()) }

            data
        }
    }

    fn view(&self) -> TensorView<Self::Elem> {
        TensorView::new(self.data.as_ref(), &self.layout)
    }
}

impl<I: AsRef<[usize]>, T, S: AsRef<[T]>> Index<I> for TensorBase<T, S> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        let offset = self.layout.offset(index.as_ref());
        &self.data.as_ref()[offset]
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>> TensorBase<T, S> {
    /// Copy elements from another tensor into this tensor.
    ///
    /// This tensor and `other` must have the same shape.
    pub fn copy_from(&mut self, other: &TensorView<T>)
    where
        T: Clone,
    {
        assert!(self.shape() == other.shape());
        for (out, x) in zip(self.iter_mut(), other.iter()) {
            *out = x.clone();
        }
    }

    /// Return the underlying data of the tensor as a mutable slice, if it is
    /// contiguous.
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        self.is_contiguous().then_some(self.data.as_mut())
    }

    /// Return the element buffer for this tensor as a mutable slice.
    ///
    /// Unlike [TensorBase::data_mut] this does not check if the data is
    /// contigous. If multiple mutable views of a non-contiguous tensor exist,
    /// the returned slice may overlap with other mutable slices.
    pub(crate) unsafe fn data_mut_unchecked(&mut self) -> &mut [T] {
        self.data.as_mut()
    }

    /// Return a mutable iterator over elements of this view.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        let layout = &self.layout;
        IterMut::new(self.data.as_mut(), layout)
    }

    /// Return an iterator over mutable slices of this tensor along a given
    /// axis.
    pub fn axis_iter_mut(&mut self, dim: usize) -> AxisIterMut<T> {
        AxisIterMut::new(self.view_mut(), dim)
    }

    pub fn axis_chunks_mut(&mut self, dim: usize, chunk_size: usize) -> AxisChunksMut<T> {
        AxisChunksMut::new(self.view_mut(), dim, chunk_size)
    }

    /// Return the element at a given index, or `None` if the index is out of
    /// bounds in any dimension.
    #[inline]
    pub fn get_mut<I: AsRef<[usize]>>(&mut self, index: I) -> Option<&mut T> {
        let offset = self.layout.try_offset(index)?;
        Some(&mut self.data.as_mut()[offset])
    }

    /// Return an iterator over views of the innermost N dimensions of this
    /// tensor.
    pub fn inner_iter_mut<const N: usize>(&mut self) -> InnerIterMut<T, N> {
        InnerIterMut::new(self.view_mut())
    }

    /// Return a mutable iterator over all 1D slices of this tensor along a
    /// given axis.
    pub fn lanes_mut(&mut self, dim: usize) -> LanesMut<T> {
        LanesMut::new(self.view_mut(), dim)
    }

    /// Replace elements of this tensor with `f(element)`.
    ///
    /// This is the in-place version of `map`.
    ///
    /// The order in which elements are visited is unspecified and may not
    /// correspond to the logical order.
    pub fn apply<F: Fn(&T) -> T>(&mut self, f: F) {
        if self.is_contiguous() {
            self.data.as_mut().iter_mut().for_each(|x| *x = f(x));
        } else {
            self.iter_mut().for_each(|x| *x = f(x));
        }
    }

    /// Replace all elements of this tensor with `value`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.apply(|_| value.clone());
    }

    /// Return a new view with the dimensions re-ordered according to `dims`.
    pub fn permuted_mut(&mut self, dims: &[usize]) -> TensorViewMut<T> {
        TensorBase {
            data: self.data.as_mut(),
            layout: self.layout.permuted(dims),
            element_type: PhantomData,
        }
    }

    /// Return a new view with a given shape. This has the same requirements
    /// as `reshape`.
    pub fn reshaped_mut(&mut self, shape: &[usize]) -> TensorViewMut<T> {
        TensorBase {
            data: self.data.as_mut(),
            layout: self.layout.reshaped(shape),
            element_type: PhantomData,
        }
    }

    /// Return a new mutable slice of this tensor.
    ///
    /// Slices are specified in the same way as for [TensorBase::slice].
    pub fn slice_mut<R: IntoSliceItems>(&mut self, range: R) -> TensorViewMut<T> {
        self.try_slice_mut(range.into_slice_items().as_ref())
            .unwrap()
    }

    /// Variant of [TensorViewMut::slice_mut] which returns an error instead of
    /// panicking if the slice range is invalid.
    pub fn try_slice_mut<R: IntoSliceItems>(
        &mut self,
        range: R,
    ) -> Result<TensorViewMut<T>, SliceError> {
        let (offset_range, layout) = self.layout.try_slice(range.into_slice_items().as_ref())?;
        let data = &mut self.data.as_mut()[offset_range];
        Ok(TensorViewMut {
            data,
            layout,
            element_type: PhantomData,
        })
    }

    /// Return a new view with the order of dimensions reversed.
    pub fn transposed_mut(&mut self) -> TensorViewMut<T> {
        TensorBase {
            data: self.data.as_mut(),
            layout: self.layout.transposed(),
            element_type: PhantomData,
        }
    }

    /// Return a mutable view of this tensor.
    ///
    /// Views share the same element array, but can have an independent layout,
    /// with some limitations.
    pub fn view_mut(&mut self) -> TensorViewMut<T> {
        TensorViewMut::new(self.data.as_mut(), &self.layout)
    }

    /// Return a mutable view with a static rank.
    ///
    /// Panics if the rank of this tensor is not `N`.
    pub fn nd_view_mut<const N: usize>(&mut self) -> NdTensorViewMut<T, N> {
        assert!(self.ndim() == N);
        let shape: [usize; N] = self.shape().try_into().unwrap();
        let strides: [usize; N] = self.strides().try_into().unwrap();
        NdTensorViewMut::from_data(self.data.as_mut(), shape, Some(strides)).unwrap()
    }
}

impl<'a, T> TensorViewMut<'a, T> {
    /// Consume this view and return the underlying data slice.
    ///
    /// This differs from [Self::data_mut] as the lifetime of the returned slice
    /// is tied to the underlying tensor, rather than the view.
    pub fn into_data_mut(self) -> &'a mut [T] {
        self.data
    }
}

impl<I: AsRef<[usize]>, T, S: AsRef<[T]> + AsMut<[T]>> IndexMut<I> for TensorBase<T, S> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let offset = self.layout.offset(index.as_ref());
        &mut self.data.as_mut()[offset]
    }
}

/// Variant of [TensorBase] which owns its elements, using a `Vec<T>` as
/// the backing storage.
pub type Tensor<T = f32> = TensorBase<T, Vec<T>>;

impl<T> Tensor<T> {
    /// Create a new zero-filled tensor with a given shape.
    pub fn zeros(shape: &[usize]) -> Tensor<T>
    where
        T: Clone + Default,
    {
        let n_elts = shape.iter().product();
        let data = vec![T::default(); n_elts];
        Tensor {
            data,
            layout: DynLayout::new(shape),
            element_type: PhantomData,
        }
    }

    /// Create a new tensor filled with a given value.
    pub fn full(shape: &[usize], value: T) -> Tensor<T>
    where
        T: Clone,
    {
        let n_elts = shape.iter().product();
        let data = vec![value; n_elts];
        Tensor {
            data,
            layout: DynLayout::new(shape),
            element_type: PhantomData,
        }
    }

    /// Create a new tensor filled with random numbers from a given source.
    pub fn rand<R: RandomSource<T>>(shape: &[usize], rand_src: &mut R) -> Tensor<T>
    where
        T: Clone + Default,
    {
        let mut tensor = Tensor::zeros(shape);
        tensor.data.fill_with(|| rand_src.next());
        tensor
    }

    /// Create a new 1D tensor filled with an arithmetic sequence of values
    /// in the range `[start, end)` separated by `step`. If `step` is omitted,
    /// it defaults to 1.
    pub fn arange(start: T, end: T, step: Option<T>) -> Tensor<T>
    where
        T: Copy + PartialOrd + From<bool> + std::ops::Add<Output = T>,
    {
        let step = step.unwrap_or((true).into());
        let mut data = Vec::new();
        let mut curr = start;
        while curr < end {
            data.push(curr);
            curr = curr + step;
        }
        Tensor::from_vec(data)
    }

    /// Create a new 0-dimensional (scalar) tensor from a single value.
    pub fn from_scalar(value: T) -> Tensor<T> {
        Self::from_data(&[], vec![value])
    }

    /// Create a new 1-dimensional tensor from a vector. No copying is required.
    pub fn from_vec(data: Vec<T>) -> Tensor<T> {
        Self::from_data(&[data.len()], data)
    }

    /// Clone this tensor with a new shape. The new shape must have the same
    /// total number of elements as the existing shape. See `reshape`.
    pub fn to_shape(&self, shape: &[usize]) -> Tensor<T>
    where
        T: Clone,
    {
        Self::from_data(shape, self.to_vec())
    }

    /// Clip dimension `dim` to `[range.start, range.end)`. The new size for
    /// the dimension must be <= the old size.
    ///
    /// This currently requires `T: Copy` to support efficiently moving data
    /// from the new start offset to the beginning of the element buffer.
    pub fn clip_dim(&mut self, dim: usize, range: Range<usize>)
    where
        T: Copy,
    {
        let (start, end) = (range.start, range.end);

        assert!(start <= end, "start must be <= end");
        assert!(end <= self.size(dim), "end must be <= dim size");

        let start_offset = self.layout.stride(dim) * start;
        self.layout.resize_dim(dim, end - start);

        let range = start_offset..start_offset + self.layout.end_offset();
        self.data.copy_within(range.clone(), 0);
        self.data.truncate(range.end - range.start);
    }

    /// Convert the internal layout of elements to be contiguous, as reported
    /// by `is_contiguous`.
    ///
    /// This is a no-op if the tensor is already contiguous.
    pub fn make_contiguous(&mut self)
    where
        T: Clone,
    {
        if self.is_contiguous() {
            return;
        }
        self.data = self.to_vec();
        self.layout.make_contiguous();
    }

    /// Update the shape of the tensor.
    ///
    /// The total number of elements for the new shape must be the same as the
    /// existing shape.
    ///
    /// This is a cheap operation if the tensor is contiguous, but requires
    /// copying data if the tensor has a non-contiguous layout.
    pub fn reshape(&mut self, shape: &[usize])
    where
        T: Clone,
    {
        let len: usize = shape.iter().product();
        let current_len = self.len();
        assert!(
            len == current_len,
            "New shape must have same total elements as current shape"
        );

        // We currently always copy data whenever the input is non-contiguous.
        // However there are cases of custom strides where copies could be
        // avoided. See https://pytorch.org/docs/stable/generated/torch.Tensor.view.html.
        self.make_contiguous();
        self.layout = DynLayout::new(shape);
    }

    /// Like [Tensor::reshape] but consumes self.
    pub fn into_shape(mut self, new_shape: &[usize]) -> Tensor<T>
    where
        T: Clone,
    {
        self.reshape(new_shape);
        self
    }
}

impl<S: AsRef<[f32]>> TensorBase<f32, S> {
    /// Serialize the tensor to a simple binary format.
    ///
    /// The serialized data is in little-endian order and has the structure:
    ///
    /// ```text
    /// [ndim: u32][dims: u32 * rank][elements: T * product(dims)]
    /// ```
    ///
    /// Where `T` is the tensor's element type.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let mut buf_writer = io::BufWriter::new(writer);
        let ndim: u32 = self.ndim() as u32;
        buf_writer.write_all(&ndim.to_le_bytes())?;
        for &dim in self.shape() {
            buf_writer.write_all(&(dim as u32).to_le_bytes())?;
        }
        for el in self.iter() {
            buf_writer.write_all(&el.to_le_bytes())?;
        }
        buf_writer.flush()?;
        Ok(())
    }
}

impl<T: PartialEq, S: AsRef<[T]>, V: View<Elem = T>> PartialEq<V> for TensorBase<T, S> {
    fn eq(&self, other: &V) -> bool {
        self.shape() == other.shape().as_ref() && self.iter().eq(other.iter())
    }
}

impl<T, S: AsRef<[T]> + Clone> Clone for TensorBase<T, S> {
    fn clone(&self) -> TensorBase<T, S> {
        let data = self.data.clone();
        TensorBase {
            data,
            layout: self.layout.clone(),
            element_type: PhantomData,
        }
    }
}

impl<T, S1: AsRef<[T]>, S2: AsRef<[T]>, const N: usize> From<NdTensorBase<T, S1, N>>
    for TensorBase<T, S2>
where
    S1: Into<S2>,
{
    fn from(value: NdTensorBase<T, S1, N>) -> TensorBase<T, S2> {
        let layout: DynLayout = value.layout().into();
        TensorBase {
            data: value.into_data().into(),
            layout,
            element_type: PhantomData,
        }
    }
}

impl<T> FromIterator<T> for Tensor<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<_> = FromIterator::from_iter(iter);
        Tensor::from_vec(data)
    }
}

impl RandomSource<f32> for XorShiftRng {
    fn next(&mut self) -> f32 {
        self.next_f32()
    }
}

// Trait for scalar (ie. non-array) values.
//
// This is used as a bound in contexts where we don't want a generic type
// `T` to be inferred as an array type.
pub trait Scalar {}

impl Scalar for i32 {}
impl Scalar for f32 {}

// The `T: Scalar` bound avoids ambiguity when choosing a `Tensor::from`
// impl for a nested array literal, as it prevents `T` from matching an array
// type.

impl<const N: usize, T: Clone + Scalar> From<[T; N]> for Tensor<T> {
    /// Construct a 1D tensor from a 1D array.
    fn from(value: [T; N]) -> Tensor<T> {
        Tensor::from_vec(value.iter().cloned().collect())
    }
}

impl<const N: usize, const M: usize, T: Clone + Scalar> From<[[T; N]; M]> for Tensor<T> {
    /// Construct a 2D tensor from a nested array.
    fn from(value: [[T; N]; M]) -> Tensor<T> {
        let data: Vec<_> = value.iter().flat_map(|y| y.iter()).cloned().collect();
        Tensor::from_data(&[M, N], data)
    }
}

impl<const N: usize, const M: usize, const K: usize, T: Clone + Scalar> From<[[[T; K]; N]; M]>
    for Tensor<T>
{
    /// Construct a 3D tensor from a nested array.
    fn from(value: [[[T; K]; N]; M]) -> Tensor<T> {
        let data: Vec<_> = value
            .iter()
            .flat_map(|y| y.iter().flat_map(|z| z.iter()))
            .cloned()
            .collect();
        Tensor::from_data(&[M, N, K], data)
    }
}

/// Call `f` with every element in `x` in logical order.
///
/// This is equivalent to `x.iter().for_each(f)` but is faster that Rust's
/// standard iteration protocol when `x` is non-contiguous and has <= 4
/// dimensions.
fn fast_for_each_element<T, F: FnMut(&T)>(mut x: TensorView<T>, mut f: F) {
    if x.ndim() > 4 {
        x.iter().for_each(f)
    } else {
        while x.ndim() < 4 {
            x.insert_dim(0);
        }

        // Safety: We only access valid offsets according to the shape and
        // strides in the loop below.
        let x_data = unsafe { x.data_unchecked() };
        let x: NdTensorView<T, 4> = x.nd_view();
        let shape = x.shape();
        let strides = x.strides();

        assert!(x_data.len() >= x.layout().min_data_len());

        for i0 in 0..shape[0] {
            for i1 in 0..shape[1] {
                for i2 in 0..shape[2] {
                    for i3 in 0..shape[3] {
                        let offset =
                            i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];

                        // Safety: We checked data length > max offset produced
                        // by layout.
                        let elt = unsafe { x_data.get_unchecked(offset) };
                        f(elt)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::IndexMut;

    use crate::rng::XorShiftRng;
    use crate::tensor;
    use crate::{
        Lanes, LanesMut, Layout, NdTensor, NdView, SliceItem, SliceRange, Tensor, TensorView,
        TensorViewMut, View,
    };

    /// Create a tensor where the value of each element is its logical index
    /// plus one.
    fn steps(shape: &[usize]) -> Tensor<i32> {
        let steps: usize = shape.iter().product();
        Tensor::arange(1, steps as i32 + 1, None).into_shape(shape)
    }

    #[test]
    fn test_apply() {
        let mut x = steps(&[3, 3]);
        x.apply(|el| el * el);
        let expected = Tensor::from_data(&[3, 3], vec![1, 4, 9, 16, 25, 36, 49, 64, 81]);
        assert_eq!(x, expected);
    }

    #[test]
    fn test_fill() {
        let mut x = Tensor::zeros(&[2, 2]);
        x.fill(1i32);
        assert_eq!(x.to_vec(), &[1, 1, 1, 1]);

        x.slice_mut(0).fill(2);
        x.slice_mut(1).fill(3);

        assert_eq!(x.to_vec(), &[2, 2, 3, 3]);
    }

    #[test]
    fn test_arange() {
        let x = Tensor::arange(1, 5, None);
        assert_eq!(x.to_vec(), [1, 2, 3, 4]);

        let x = Tensor::arange(1, 10, Some(2));
        assert_eq!(x.to_vec(), [1, 3, 5, 7, 9]);

        let x = Tensor::arange(1., 5., None);
        assert_eq!(x.to_vec(), [1., 2., 3., 4.]);

        let x = Tensor::arange(1., 5., Some(0.5));
        assert_eq!(x.to_vec(), [1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]);
    }

    #[test]
    fn test_axis_iter() {
        let x = steps(&[2, 3, 4]);

        // First dimension.
        let views: Vec<_> = x.axis_iter(0).collect();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0], x.slice([0]));
        assert_eq!(views[1], x.slice([1]));

        // Second dimension.
        let views: Vec<_> = x.axis_iter(1).collect();
        assert_eq!(views.len(), 3);
        assert_eq!(views[0], x.slice((.., 0)));
        assert_eq!(views[1], x.slice((.., 1)));
    }

    #[test]
    fn test_axis_iter_mut() {
        let mut x = steps(&[2, 3]);
        let y0 = x.slice([0]).to_tensor();
        let y1 = x.slice([1]).to_tensor();

        // First dimension.
        let mut views: Vec<_> = x.axis_iter_mut(0).collect();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0], y0);
        assert_eq!(views[1], y1);
        views[0].iter_mut().for_each(|x| *x += 1);
        views[1].iter_mut().for_each(|x| *x += 2);
        assert_eq!(x.to_vec(), &[2, 3, 4, 6, 7, 8]);

        let z0 = x.slice((.., 0)).to_tensor();
        let z1 = x.slice((.., 1)).to_tensor();

        // Second dimension.
        let views: Vec<_> = x.axis_iter_mut(1).collect();
        assert_eq!(views.len(), 3);
        assert_eq!(views[0], z0);
        assert_eq!(views[1], z1);
    }

    #[test]
    fn test_axis_chunks() {
        let x = steps(&[4, 2, 2]);

        let mut chunks = x.axis_chunks(0, 2);

        let chunk = chunks.next().expect("expected chunk");
        assert_eq!(chunk.shape(), &[2, 2, 2]);
        assert_eq!(
            chunk.iter().copied().collect::<Vec<_>>(),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );

        let chunk = chunks.next().expect("expected chunk");
        assert_eq!(chunk.shape(), &[2, 2, 2]);
        assert_eq!(
            chunk.iter().copied().collect::<Vec<_>>(),
            &[9, 10, 11, 12, 13, 14, 15, 16]
        );

        assert!(chunks.next().is_none());
    }

    #[test]
    fn test_axis_chunks_mut() {
        let mut x = steps(&[4, 2, 2]);

        let mut chunks = x.axis_chunks_mut(0, 2);

        let mut chunk = chunks.next().expect("expected chunk");
        assert_eq!(chunk.shape(), &[2, 2, 2]);
        chunk.iter_mut().for_each(|x| *x *= 10);
        assert_eq!(
            chunk.iter().copied().collect::<Vec<_>>(),
            &[10, 20, 30, 40, 50, 60, 70, 80]
        );

        let mut chunk = chunks.next().expect("expected chunk");
        assert_eq!(chunk.shape(), &[2, 2, 2]);
        chunk.iter_mut().for_each(|x| *x *= 10);
        assert_eq!(
            chunk.iter().copied().collect::<Vec<_>>(),
            &[90, 100, 110, 120, 130, 140, 150, 160]
        );

        assert!(chunks.next().is_none());
    }

    #[test]
    fn test_clip_dim() {
        let mut x = steps(&[3, 3]);
        x.clip_dim(0, 1..2);
        x.clip_dim(1, 1..2);
        assert_eq!(x.to_vec(), vec![5]);
    }

    #[test]
    fn test_clip_dim_start() {
        let mut x = steps(&[3, 3]);

        // Clip the start of the tensor, adjusting the `base` offset.
        x.clip_dim(0, 1..3);

        // Indexing should reflect the slice.
        assert_eq!(x.to_vec(), &[4, 5, 6, 7, 8, 9]);
        assert_eq!(x[[0, 0]], 4);
        assert_eq!(*x.index_mut([0, 0]), 4);

        // Slices returned by `data` should reflect the slice.
        assert_eq!(x.data(), Some([4, 5, 6, 7, 8, 9].as_slice()));
        assert_eq!(x.data_mut().as_deref(), Some([4, 5, 6, 7, 8, 9].as_slice()));

        // Offsets should be relative to the sliced returned by `data`,
        // `data_mut`.
        assert_eq!(x.offsets().collect::<Vec<usize>>(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(x.layout().offset(&[0, 0]), 0);
    }

    #[test]
    fn test_copy_from() {
        let x = steps(&[3, 3]);
        let mut y = Tensor::zeros(x.shape());

        y.copy_from(&x.view());

        assert_eq!(y, x);
    }

    #[test]
    fn test_from_arrays() {
        // 2D
        let x = Tensor::from([[2, 3], [4, 5], [6, 7]]);
        assert_eq!(x.shape(), &[3, 2]);
        assert_eq!(x.data(), Some([2, 3, 4, 5, 6, 7].as_slice()));

        // 3D
        let x = Tensor::from([[[2, 3], [4, 5], [6, 7]]]);
        assert_eq!(x.shape(), &[1, 3, 2]);
        assert_eq!(x.data(), Some([2, 3, 4, 5, 6, 7].as_slice()));
    }

    #[test]
    fn test_from_scalar() {
        let x = Tensor::from_scalar(5);
        assert_eq!(x.shape().len(), 0);
        assert_eq!(x.data(), Some([5].as_slice()));
    }

    #[test]
    fn test_from_vec() {
        let x = tensor!([1, 2, 3]);
        assert_eq!(x.shape(), &[3]);
        assert_eq!(x.data(), Some([1, 2, 3].as_slice()));
    }

    #[test]
    fn test_full() {
        let x = Tensor::full(&[2, 2], 1.0);
        assert_eq!(x.shape(), &[2, 2]);
        assert_eq!(x.data(), Some([1., 1., 1., 1.].as_slice()));
    }

    #[test]
    fn test_from_iterator() {
        let x: Tensor<i32> = FromIterator::from_iter(0..10);
        assert_eq!(x.shape(), &[10]);
        assert_eq!(x.data(), Some([0, 1, 2, 3, 4, 5, 6, 7, 8, 9].as_slice()));
    }

    #[test]
    fn test_stride() {
        let x = Tensor::<f32>::zeros(&[2, 5, 7, 3]);
        assert_eq!(x.stride(3), 1);
        assert_eq!(x.stride(2), 3);
        assert_eq!(x.stride(1), 7 * 3);
        assert_eq!(x.stride(0), 5 * 7 * 3);
    }

    #[test]
    fn test_strides() {
        let x = Tensor::<f32>::zeros(&[2, 5, 7, 3]);
        assert_eq!(x.strides(), [5 * 7 * 3, 7 * 3, 3, 1]);
    }

    #[test]
    fn test_get() {
        let mut x = Tensor::<f32>::zeros(&[2, 2]);

        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        x.data[3] = 4.0;

        // Index with fixed-sized array.
        assert_eq!(x.get([0, 0]), Some(&1.0));
        assert_eq!(x.get([0, 1]), Some(&2.0));
        assert_eq!(x.get([1, 0]), Some(&3.0));
        assert_eq!(x.get([1, 1]), Some(&4.0));

        // Invalid indices
        assert_eq!(x.get([0, 2]), None);
        assert_eq!(x.get([2, 0]), None);
        assert_eq!(x.get([1, 0, 0]), None);

        // Index with slice.
        assert_eq!(x.get([0, 0].as_slice()), Some(&1.0));
        assert_eq!(x.get([0, 1].as_slice()), Some(&2.0));
        assert_eq!(x.get([1, 0].as_slice()), Some(&3.0));
        assert_eq!(x.get([1, 1].as_slice()), Some(&4.0));
    }

    #[test]
    fn test_index() {
        let mut x = Tensor::<f32>::zeros(&[2, 2]);

        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        x.data[3] = 4.0;

        // Index with fixed-sized array.
        assert_eq!(x[[0, 0]], 1.0);
        assert_eq!(x[[0, 1]], 2.0);
        assert_eq!(x[[1, 0]], 3.0);
        assert_eq!(x[[1, 1]], 4.0);

        // Index with slice.
        assert_eq!(x[[0, 0].as_slice()], 1.0);
        assert_eq!(x[[0, 1].as_slice()], 2.0);
        assert_eq!(x[[1, 0].as_slice()], 3.0);
        assert_eq!(x[[1, 1].as_slice()], 4.0);
    }

    #[test]
    fn test_index_scalar() {
        let x = Tensor::from_scalar(5.0);
        assert_eq!(x[[]], 5.0);
    }

    #[test]
    fn test_index_mut() {
        let mut x = Tensor::<f32>::zeros(&[2, 2]);

        x[[0, 0]] = 1.0;
        x[[0, 1]] = 2.0;
        x[[1, 0]] = 3.0;
        x[[1, 1]] = 4.0;

        assert_eq!(x.data[0], 1.0);
        assert_eq!(x.data[1], 2.0);
        assert_eq!(x.data[2], 3.0);
        assert_eq!(x.data[3], 4.0);
    }

    #[test]
    fn test_get_mut() {
        let mut x = Tensor::<f32>::zeros(&[2, 2]);

        *x.get_mut([0, 0]).unwrap() = 1.0;
        *x.get_mut([0, 1]).unwrap() = 2.0;
        *x.get_mut([1, 0]).unwrap() = 3.0;
        *x.get_mut([1, 1]).unwrap() = 4.0;

        assert_eq!(x.data[0], 1.0);
        assert_eq!(x.data[1], 2.0);
        assert_eq!(x.data[2], 3.0);
        assert_eq!(x.data[3], 4.0);

        assert_eq!(x.get_mut([1, 2]), None);
    }

    #[test]
    #[should_panic]
    fn test_index_panics_if_invalid() {
        let x = Tensor::<f32>::zeros(&[2, 2]);
        x[[2, 0]];
    }

    #[test]
    #[should_panic]
    fn test_index_panics_if_wrong_dim_count() {
        let x = Tensor::<f32>::zeros(&[2, 2]);
        x[[0, 0, 0]];
    }

    #[test]
    fn test_indices() {
        let x = Tensor::<f32>::zeros(&[2, 2]);
        let x_indices = {
            let mut indices = Vec::new();
            let mut iter = x.indices();
            while let Some(index) = iter.next() {
                indices.push(index.to_vec());
            }
            indices
        };
        assert_eq!(
            x_indices,
            &[vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1],]
        );
    }

    #[test]
    fn test_item() {
        let scalar = Tensor::from_scalar(5.0);
        assert_eq!(scalar.item(), Some(&5.0));

        let vec_one_item = tensor!([5.0]);
        assert_eq!(vec_one_item.item(), Some(&5.0));

        let vec_many_items = tensor!([1.0, 2.0]);
        assert_eq!(vec_many_items.item(), None);

        let matrix_one_item = Tensor::from_data(&[1, 1], vec![5.0]);
        assert_eq!(matrix_one_item.item(), Some(&5.0));
    }

    #[test]
    fn test_map() {
        // Contiguous tensor.
        let x = steps(&[2, 3]).map(|val| val * 2);
        assert_eq!(x.to_vec(), &[2, 4, 6, 8, 10, 12]);

        // Non-contiguous view.
        let x = steps(&[2, 3]);
        let x = x.transposed();
        assert!(!x.is_contiguous());
        assert_eq!(x.to_vec(), &[1, 4, 2, 5, 3, 6]);
        let x = x.map(|val| val * 2);
        assert_eq!(x.to_vec(), &[2, 8, 4, 10, 6, 12]);
    }

    #[test]
    fn test_move_axis() {
        let mut x = steps(&[2, 3]);
        x.move_axis(1, 0);
        assert_eq!(x.shape(), [3, 2]);
    }

    #[test]
    fn test_ndim() {
        let scalar = Tensor::from_scalar(5.0);
        let vec = tensor!([5.0]);
        let matrix = Tensor::from_data(&[1, 1], vec![5.0]);

        assert_eq!(scalar.ndim(), 0);
        assert_eq!(vec.ndim(), 1);
        assert_eq!(matrix.ndim(), 2);
    }

    #[test]
    fn test_partial_eq() {
        let x = tensor!([1, 2, 3, 4, 5]);
        let y = x.clone();
        let z = x.to_shape(&[1, 5]);

        // Int tensors are equal if they have the same shape and elements.
        assert_eq!(&x, &y);
        assert_ne!(&x, &z);
    }

    #[test]
    fn test_len() {
        let scalar = Tensor::from_scalar(5);
        let vec = tensor!([1, 2, 3]);
        let matrix = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);

        assert_eq!(scalar.len(), 1);
        assert_eq!(vec.len(), 3);
        assert_eq!(matrix.len(), 4);
    }

    #[test]
    fn test_is_empty() {
        assert!(Tensor::<f32>::from_vec(vec![]).is_empty());
        assert!(!tensor!([1]).is_empty());
        assert!(!Tensor::from_scalar(5.0).is_empty());
    }

    #[test]
    fn test_reshape() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = Tensor::rand(&[10, 5, 3, 7], &mut rng);
        let x_data: Vec<f32> = x.data().unwrap().to_vec();

        assert_eq!(x.shape(), &[10, 5, 3, 7]);

        x.reshape(&[10, 5, 3 * 7]);

        assert_eq!(x.shape(), &[10, 5, 3 * 7]);
        assert_eq!(x.data(), Some(x_data.as_slice()));
    }

    #[test]
    fn test_reshape_non_contiguous() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = Tensor::rand(&[10, 10], &mut rng);

        // Set the input up so that it is non-contiguous and has a non-zero
        // `base` offset.
        x.permute(&[1, 0]);
        x.clip_dim(0, 2..8);

        // Reshape the tensor. This should copy the data and reset the `base`
        // offset.
        x.reshape(&[x.shape().iter().product()]);

        // After reshaping, we should be able to successfully read all the elements.
        // Note this test doesn't check that the correct elements were read.
        let elts: Vec<_> = x.iter().collect();
        assert_eq!(elts.len(), 60);

        // Set up another input so it is non-contiguous and has a non-zero `base` offset.
        let mut x = steps(&[3, 3]);
        x.clip_dim(0, 1..3);
        x.clip_dim(1, 1..3);

        // Flatten the input with reshape.
        x.reshape(&[4]);

        // Check that the correct elements were read.
        assert_eq!(x.to_vec(), &[5, 6, 8, 9]);
    }

    #[test]
    fn test_reshape_copies_with_custom_strides() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = Tensor::rand(&[10, 10], &mut rng);

        // Give the tensor a non-default stride
        x.clip_dim(1, 0..8);
        assert!(!x.is_contiguous());
        let x_elements = x.to_vec();

        x.reshape(&[80]);

        // Since the tensor had a non-default stride, `reshape` will have copied
        // data.
        assert_eq!(x.shape(), &[80]);
        assert!(x.is_contiguous());
        assert_eq!(x.data(), Some(x_elements.as_slice()));
    }

    #[test]
    #[should_panic(expected = "New shape must have same total elements as current shape")]
    fn test_reshape_with_wrong_size() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = Tensor::rand(&[10, 5, 3, 7], &mut rng);
        x.reshape(&[10, 5]);
    }

    #[test]
    fn test_permute() {
        // Test with a vector (this is a no-op)
        let mut input = steps(&[5]);
        assert!(input.iter().eq([1, 2, 3, 4, 5].iter()));
        input.permute(&[0]);
        assert!(input.iter().eq([1, 2, 3, 4, 5].iter()));

        // Test with a matrix (ie. transpose the matrix)
        let mut input = steps(&[2, 3]);
        assert!(input.iter().eq([1, 2, 3, 4, 5, 6].iter()));
        input.permute(&[1, 0]);
        assert_eq!(input.shape(), &[3, 2]);
        assert!(input.iter().eq([1, 4, 2, 5, 3, 6].iter()));

        // Test with a higher-rank tensor. For this test we don't list out the
        // full permuted element sequence, but just check the shape and strides
        // were updated.
        let mut input = steps(&[3, 4, 5]);
        let (stride_0, stride_1, stride_2) = (input.stride(0), input.stride(1), input.stride(2));
        input.permute(&[2, 0, 1]);
        assert_eq!(input.shape(), &[5, 3, 4]);
        assert_eq!(
            (input.stride(0), input.stride(1), input.stride(2)),
            (stride_2, stride_0, stride_1)
        );
    }

    #[test]
    #[should_panic(expected = "permutation is invalid")]
    fn test_permute_wrong_dim_count() {
        let mut input = steps(&[2, 3]);
        input.permute(&[1, 2, 3]);
    }

    #[test]
    fn test_transpose() {
        // Test with a vector (this is a no-op)
        let mut input = steps(&[5]);
        input.transpose();
        assert_eq!(input.shape(), &[5]);

        // Test with a matrix
        let mut input = steps(&[2, 3]);
        assert!(input.iter().eq([1, 2, 3, 4, 5, 6].iter()));
        input.transpose();
        assert_eq!(input.shape(), &[3, 2]);
        assert!(input.iter().eq([1, 4, 2, 5, 3, 6].iter()));

        // Test with a higher-rank tensor
        let mut input = steps(&[1, 3, 7]);
        input.transpose();
        assert_eq!(input.shape(), [7, 3, 1]);
    }

    #[test]
    fn test_insert_dim() {
        // Insert dims in contiguous tensor.
        let mut input = steps(&[2, 3]);
        input.insert_dim(1);
        assert_eq!(input.shape(), &[2, 1, 3]);
        assert_eq!(input.strides(), &[3, 6, 1]);

        input.insert_dim(1);
        assert_eq!(input.shape(), &[2, 1, 1, 3]);
        assert_eq!(input.strides(), &[3, 6, 6, 1]);

        input.insert_dim(0);
        assert_eq!(input.shape(), &[1, 2, 1, 1, 3]);
        assert_eq!(input.strides(), &[6, 3, 6, 6, 1]);

        // Insert dims in non-contiguous tensor.
        let mut input = steps(&[2, 3]);
        input.transpose();
        input.insert_dim(0);
        assert_eq!(input.shape(), &[1, 3, 2]);
        assert_eq!(input.strides(), &[6, 1, 3]);

        input.insert_dim(3);
        assert_eq!(input.shape(), &[1, 3, 2, 1]);
        assert_eq!(input.strides(), &[6, 1, 3, 6]);

        // Insert dims in a tensor where the smallest stride is > 1.
        let input = steps(&[2, 4]);
        let mut view = input.slice((.., SliceRange::new(0, None, 2)));
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[4, 2]);

        view.insert_dim(0);

        assert_eq!(view.shape(), &[1, 2, 2]);
        assert_eq!(view.strides(), &[8, 4, 2]);
    }

    #[test]
    fn test_to_shape() {
        let mut rng = XorShiftRng::new(1234);
        let x = Tensor::rand(&[10, 5, 3, 7], &mut rng);
        let y = x.to_shape(&[10, 5, 3 * 7]);

        assert_eq!(y.shape(), &[10, 5, 3 * 7]);
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_nd_view() {
        let mut rng = XorShiftRng::new(1234);
        let x = Tensor::rand(&[10, 5, 3, 7], &mut rng);
        let x_view = x.nd_view::<4>();
        assert_eq!(x_view.shape(), x.shape());
        assert_eq!(x_view.strides(), x.strides());
        assert_eq!(x_view.data(), x.data());
    }

    #[test]
    fn test_nd_view_mut() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = Tensor::rand(&[10, 5, 3, 7], &mut rng);
        let layout = x.layout().clone();
        let x_view = x.nd_view_mut::<4>();
        assert_eq!(x_view.shape(), layout.shape());
        assert_eq!(x_view.strides(), layout.strides());
    }

    #[test]
    fn test_iter_for_contiguous_array() {
        for dims in 1..7 {
            let mut shape = Vec::new();
            for d in 0..dims {
                shape.push(d + 1);
            }
            let mut rng = XorShiftRng::new(1234);
            let x = Tensor::rand(&shape, &mut rng);

            let elts: Vec<f32> = x.iter().copied().collect();

            assert_eq!(x.data(), Some(elts.as_slice()));
        }
    }

    #[test]
    fn test_iter_for_empty_array() {
        let empty = Tensor::<f32>::zeros(&[3, 0, 5]);
        assert!(empty.iter().next().is_none());
    }

    #[test]
    fn test_iter_for_non_contiguous_array() {
        let mut x = Tensor::zeros(&[3, 3]);
        for (index, elt) in x.iter_mut().enumerate() {
            *elt = index + 1;
        }

        // Initially tensor is contiguous, so data buffer and element sequence
        // match.
        assert_eq!(
            x.data(),
            Some(x.iter().copied().collect::<Vec<_>>().as_slice())
        );

        // Slice the tensor along an outer dimension. This will leave the tensor
        // contiguous, and hence `data` and `elements` should return the same
        // elements.
        x.clip_dim(0, 0..2);
        assert_eq!(x.data(), Some([1, 2, 3, 4, 5, 6].as_slice()));
        assert_eq!(x.iter().copied().collect::<Vec<_>>(), &[1, 2, 3, 4, 5, 6]);
        // Test with step > 1 to exercise `Elements::nth`.
        assert_eq!(x.iter().step_by(2).copied().collect::<Vec<_>>(), &[1, 3, 5]);

        // Slice the tensor along an inner dimension. The tensor will no longer
        // be contiguous and hence `elements` will return different results than
        // `data`.
        x.clip_dim(1, 0..2);
        assert_eq!(x.data(), None);
        assert_eq!(x.iter().copied().collect::<Vec<_>>(), &[1, 2, 4, 5]);
        // Test with step > 1 to exercise `Elements::nth`.
        assert_eq!(x.iter().step_by(2).copied().collect::<Vec<_>>(), &[1, 4]);
    }

    // PyTorch and numpy do not allow iteration over a scalar, but it seems
    // consistent for `Tensor::iter` to always yield `Tensor::len` elements,
    // and `len` returns 1 for a scalar.
    #[test]
    fn test_iter_for_scalar() {
        let x = Tensor::from_scalar(5.0);
        let elements = x.iter().copied().collect::<Vec<_>>();
        assert_eq!(&elements, &[5.0]);
    }

    #[test]
    fn test_iter_mut_for_contiguous_array() {
        for dims in 1..7 {
            let mut shape = Vec::new();
            for d in 0..dims {
                shape.push(d + 1);
            }
            let mut rng = XorShiftRng::new(1234);
            let mut x = Tensor::rand(&shape, &mut rng);

            let elts: Vec<f32> = x.iter().map(|x| x * 2.).collect();

            for elt in x.iter_mut() {
                *elt *= 2.;
            }

            assert_eq!(x.data(), Some(elts.as_slice()));
        }
    }

    #[test]
    fn test_iter_mut_for_non_contiguous_array() {
        let mut x = Tensor::zeros(&[3, 3]);
        for (index, elt) in x.iter_mut().enumerate() {
            *elt = index + 1;
        }
        x.permute(&[1, 0]);

        let x_doubled: Vec<usize> = x.iter().map(|x| x * 2).collect();
        for elt in x.iter_mut() {
            *elt *= 2;
        }
        assert_eq!(x.to_vec(), x_doubled);
    }

    #[test]
    fn test_lanes() {
        let x = steps(&[3, 3]);

        let collect_lane =
            |lanes: &mut Lanes<'_, i32>| lanes.next().map(|lane| lane.copied().collect::<Vec<_>>());

        let mut rows = x.lanes(1);
        assert_eq!(collect_lane(&mut rows), Some([1, 2, 3].to_vec()));
        assert_eq!(collect_lane(&mut rows), Some([4, 5, 6].to_vec()));
        assert_eq!(collect_lane(&mut rows), Some([7, 8, 9].to_vec()));

        let mut cols = x.lanes(0);
        assert_eq!(collect_lane(&mut cols), Some([1, 4, 7].to_vec()));
        assert_eq!(collect_lane(&mut cols), Some([2, 5, 8].to_vec()));
        assert_eq!(collect_lane(&mut cols), Some([3, 6, 9].to_vec()));
    }

    #[test]
    fn test_lanes_mut() {
        let update_lanes = |lanes: LanesMut<'_, i32>| {
            let mut lane_idx = 0;
            for lane in lanes {
                for el in lane {
                    *el = lane_idx;
                }
                lane_idx += 1;
            }
        };

        let mut x = Tensor::zeros(&[3, 3]);
        let rows = x.lanes_mut(1);
        update_lanes(rows);
        assert_eq!(x.to_vec(), &[0, 0, 0, 1, 1, 1, 2, 2, 2]);

        let mut x = Tensor::zeros(&[3, 3]);
        let cols = x.lanes_mut(0);
        update_lanes(cols);
        assert_eq!(x.to_vec(), &[0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_to_vec() {
        let mut x = steps(&[3, 3]);

        // Contiguous case. This should use the fast-path.
        assert_eq!(x.to_vec(), x.iter().copied().collect::<Vec<_>>());

        // Non-contiguous case.
        x.clip_dim(1, 0..2);
        assert!(!x.is_contiguous());
        assert_eq!(x.to_vec(), x.iter().copied().collect::<Vec<_>>());
    }

    #[test]
    fn test_offsets() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = Tensor::rand(&[10, 10], &mut rng);

        let x_elts: Vec<_> = x.to_vec();

        let x_offsets = x.offsets();
        let x_data = x.data_mut().unwrap();
        let x_elts_from_offset: Vec<_> = x_offsets.map(|off| x_data[off]).collect();

        assert_eq!(x_elts, x_elts_from_offset);
    }

    #[test]
    fn test_offsets_nth() {
        let x = steps(&[3]);
        let mut iter = x.offsets();
        assert_eq!(iter.nth(0), Some(0));
        assert_eq!(iter.nth(0), Some(1));
        assert_eq!(iter.nth(0), Some(2));
        assert_eq!(iter.nth(0), None);

        let x = steps(&[10]);
        let mut iter = x.offsets();
        assert_eq!(iter.nth(1), Some(1));
        assert_eq!(iter.nth(5), Some(7));
        assert_eq!(iter.nth(1), Some(9));
        assert_eq!(iter.nth(0), None);
    }

    #[test]
    fn test_from_data() {
        let scalar = Tensor::from_data(&[], vec![1.0]);
        assert_eq!(scalar.len(), 1);

        let matrix = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        assert_eq!(matrix.shape(), &[2, 2]);
        assert_eq!(matrix.data(), Some([1, 2, 3, 4].as_slice()));
    }

    #[test]
    fn test_from_data_with_slice() {
        let matrix = TensorView::from_data(&[2, 2], [1, 2, 3, 4].as_slice());
        assert_eq!(matrix.shape(), &[2, 2]);
        assert_eq!(matrix.data(), Some([1, 2, 3, 4].as_slice()));
    }

    #[test]
    fn test_from_data_with_mut_slice() {
        let mut data = vec![1, 2, 3, 4];
        let mut matrix = TensorViewMut::from_data(&[2, 2], &mut data[..]);
        matrix[[0, 1]] = 5;
        matrix[[1, 0]] = 6;
        assert_eq!(data, &[1, 5, 6, 4]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_with_wrong_len() {
        Tensor::from_data(&[1], vec![1, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_if_scalar_data_empty() {
        Tensor::<i32>::from_data(&[], vec![]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_if_scalar_data_has_many_elements() {
        Tensor::from_data(&[], vec![1, 2, 3]);
    }

    #[test]
    fn test_from_ndtensor() {
        // NdTensor -> Tensor
        let ndtensor = NdTensor::zeros([1, 10, 20]);
        let tensor: Tensor<i32> = ndtensor.clone().into();
        assert_eq!(tensor.data(), ndtensor.data());
        assert_eq!(tensor.shape(), ndtensor.shape());
        assert_eq!(tensor.strides(), ndtensor.strides());

        // NdTensorView -> TensorView
        let view: TensorView<i32> = ndtensor.view().into();
        assert_eq!(view.shape(), ndtensor.shape());

        // NdTensorViewMut -> TensorViewMut
        let mut ndtensor = NdTensor::zeros([1, 10, 20]);
        let mut view: TensorViewMut<i32> = ndtensor.view_mut().into();
        view[[0, 0, 0]] = 1;
        assert_eq!(ndtensor[[0, 0, 0]], 1);
    }

    #[test]
    fn test_is_contiguous() {
        let mut x = Tensor::zeros(&[3, 3]);
        for (index, elt) in x.iter_mut().enumerate() {
            *elt = index + 1;
        }

        // Freshly-allocated tensor
        assert!(x.is_contiguous());

        // Tensor where outermost dimension has been clipped at the end.
        let mut y = x.clone();
        y.clip_dim(0, 0..2);
        assert!(y.is_contiguous());
        assert_eq!(y.data(), Some([1, 2, 3, 4, 5, 6].as_slice()));

        // Tensor where outermost dimension has been clipped at the start.
        let mut y = x.clone();
        y.clip_dim(0, 1..3);
        assert!(y.is_contiguous());
        assert_eq!(y.data(), Some([4, 5, 6, 7, 8, 9].as_slice()));

        // Tensor where inner dimension has been clipped at the start.
        let mut y = x.clone();
        y.clip_dim(1, 1..3);
        assert!(!y.is_contiguous());

        // Tensor where inner dimension has been clipped at the end.
        let mut y = x.clone();
        y.clip_dim(1, 0..2);
        assert!(!y.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_1d() {
        let mut x = Tensor::zeros(&[10]);
        for (index, elt) in x.iter_mut().enumerate() {
            *elt = index + 1;
        }

        assert!(x.is_contiguous());
        x.clip_dim(0, 0..5);
        assert!(x.is_contiguous());
    }

    #[test]
    fn test_make_contiguous() {
        let mut x = steps(&[3, 3]);
        assert!(x.is_contiguous());

        // Clip outer dimension at start. This will modify the base offset.
        x.clip_dim(0, 1..3);

        // Clip inner dimension at start. This will modify the strides.
        x.clip_dim(1, 1..3);
        assert!(!x.is_contiguous());

        x.make_contiguous();
        assert!(x.is_contiguous());
        assert_eq!(x.to_vec(), &[5, 6, 8, 9]);
    }

    #[test]
    fn test_to_contiguous() {
        let x = steps(&[3, 3]);
        let y = x.to_contiguous();
        assert!(y.is_contiguous());
        assert_eq!(y.data().unwrap().as_ptr(), x.data().unwrap().as_ptr());

        let x = x.permuted(&[1, 0]);
        let y = x.to_contiguous();
        assert!(y.is_contiguous());
        assert_eq!(x.data(), None);
        assert_eq!(y.data().unwrap(), x.to_vec());
    }

    #[test]
    fn test_broadcast_iter() {
        let x = steps(&[1, 2, 1, 2]);
        assert_eq!(x.to_vec(), &[1, 2, 3, 4]);

        // Broadcast a 1-size dimension to size 2
        let bx = x.broadcast_iter(&[2, 2, 1, 2]);
        assert_eq!(bx.copied().collect::<Vec<i32>>(), &[1, 2, 3, 4, 1, 2, 3, 4]);

        // Broadcast a different 1-size dimension to size 2
        let bx = x.broadcast_iter(&[1, 2, 2, 2]);
        assert_eq!(bx.copied().collect::<Vec<i32>>(), &[1, 2, 1, 2, 3, 4, 3, 4]);

        // Broadcast to a larger number of dimensions
        let x = steps(&[5]);
        let bx = x.broadcast_iter(&[1, 5]);
        assert_eq!(bx.copied().collect::<Vec<i32>>(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_broadcast_iter_with_scalar() {
        let scalar = Tensor::from_scalar(7);
        let bx = scalar.broadcast_iter(&[3, 3]);
        assert_eq!(
            bx.copied().collect::<Vec<i32>>(),
            &[7, 7, 7, 7, 7, 7, 7, 7, 7]
        );
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast to specified shape")]
    fn test_broadcast_iter_with_invalid_shape() {
        let x = steps(&[2, 2]);
        x.broadcast_iter(&[3, 2]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast to specified shape")]
    fn test_broadcast_iter_with_shorter_shape() {
        let x = steps(&[2, 2]);
        x.broadcast_iter(&[4]);
    }

    #[test]
    fn test_broadcast() {
        let x = steps(&[1, 2, 1, 2]);
        assert_eq!(x.to_vec(), &[1, 2, 3, 4]);

        // Broadcast a 1-size dimension to size 2
        let bx = x.broadcast(&[2, 2, 1, 2]);
        assert_eq!(bx.shape(), &[2, 2, 1, 2]);
        assert_eq!(bx.strides(), &[0, x.stride(1), x.stride(2), x.stride(3)]);
        assert_eq!(bx.to_vec(), &[1, 2, 3, 4, 1, 2, 3, 4]);

        // Broadcast a different 1-size dimension to size 2
        let bx = x.broadcast(&[1, 2, 2, 2]);
        assert_eq!(bx.shape(), &[1, 2, 2, 2]);
        assert_eq!(bx.strides(), &[x.stride(0), x.stride(1), 0, x.stride(3)]);
        assert_eq!(bx.to_vec(), &[1, 2, 1, 2, 3, 4, 3, 4]);

        // Broadcast to a larger number of dimensions
        let x = steps(&[5]);
        let bx = x.broadcast(&[1, 5]);
        assert_eq!(bx.shape(), &[1, 5]);
        assert_eq!(bx.strides(), &[0, x.stride(0)]);
        assert_eq!(bx.to_vec(), &[1, 2, 3, 4, 5]);

        // Broadcast a scalar
        let scalar = Tensor::from_scalar(7);
        let bx = scalar.broadcast(&[3, 3]);
        assert_eq!(bx.shape(), &[3, 3]);
        assert_eq!(bx.strides(), &[0, 0]);
        assert_eq!(bx.to_vec(), &[7, 7, 7, 7, 7, 7, 7, 7, 7]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast to specified shape")]
    fn test_broadcast_invalid() {
        let x = steps(&[2, 2]);
        x.broadcast(&[4]);
    }

    #[test]
    fn test_can_broadcast_to() {
        let x = steps(&[1, 5, 10]);
        assert!(x.can_broadcast_to(&[2, 5, 10]));
        assert!(x.can_broadcast_to(&[1, 5, 10]));
        assert!(!x.can_broadcast_to(&[1, 1, 10]));
    }

    #[test]
    fn test_can_broadcast_with() {
        let x = steps(&[1, 5, 10]);
        assert!(x.can_broadcast_with(&[2, 5, 10]));
        assert!(x.can_broadcast_with(&[1, 5, 10]));
        assert!(x.can_broadcast_with(&[1, 1, 10]));
    }

    #[test]
    fn test_inner_iter() {
        let x = steps(&[2, 2, 2]);
        let mut iter = x.inner_iter::<2>();

        let mat = iter.next().unwrap();
        assert_eq!(mat.shape(), [2, 2]);
        assert_eq!(mat.iter().copied().collect::<Vec<_>>(), [1, 2, 3, 4]);

        let mat = iter.next().unwrap();
        assert_eq!(mat.shape(), [2, 2]);
        assert_eq!(mat.iter().copied().collect::<Vec<_>>(), [5, 6, 7, 8]);

        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_inner_iter_mut() {
        let mut x = steps(&[2, 2, 2]);
        let mut iter = x.inner_iter_mut::<2>();

        let mat = iter.next().unwrap();
        assert_eq!(mat.shape(), [2, 2]);
        assert_eq!(mat.iter().copied().collect::<Vec<_>>(), [1, 2, 3, 4]);

        let mat = iter.next().unwrap();
        assert_eq!(mat.shape(), [2, 2]);
        assert_eq!(mat.iter().copied().collect::<Vec<_>>(), [5, 6, 7, 8]);

        assert_eq!(iter.next(), None);
    }

    // Common slice tests for all slicing functions.
    macro_rules! slice_tests {
        ($x:expr, $method:ident) => {
            assert_eq!($x.shape(), &[2, 3, 4]);

            // 1D index
            let y = $x.$method([0]);
            assert_eq!(y.shape(), [3, 4]);
            assert_eq!(y.to_vec(), (1..=(3 * 4)).into_iter().collect::<Vec<i32>>());

            // Negative 1D index
            let y = $x.$method([-1]);
            assert_eq!(y.shape(), [3, 4]);
            assert_eq!(
                y.to_vec(),
                ((3 * 4 + 1)..=(2 * 3 * 4))
                    .into_iter()
                    .collect::<Vec<i32>>()
            );

            // 2D index
            let y = $x.$method([0, 1]);
            assert_eq!(y.shape(), [4]);
            assert_eq!(y.to_vec(), (5..=8).into_iter().collect::<Vec<i32>>());

            // 3D index
            let y = $x.$method([0, 1, 2]);
            assert_eq!(y.shape(), []);
            assert_eq!(y.item(), Some(&7));

            // Full range
            let y = $x.$method([..]);
            assert_eq!(y.shape(), [2, 3, 4]);
            assert_eq!(y.to_vec(), $x.to_vec());

            // Partial ranges
            let y = $x.$method((.., ..2, 1..));
            assert_eq!(y.shape(), [2, 2, 3]);

            // Stepped range
            let y = $x.$method((.., .., SliceItem::range(0, None, 2)));
            assert_eq!(
                y.to_vec(),
                $x.iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(i, x)| (i % 2 == 0).then_some(x))
                    .collect::<Vec<_>>()
            );

            // Mixed indices and ranges
            let y = $x.$method((.., 0, ..));
            assert_eq!(y.shape(), [2, 4]);

            let y = $x.$method((.., .., 0));
            assert_eq!(y.shape(), [2, 3]);
            assert_eq!(y.to_vec(), &[1, 5, 9, 13, 17, 21]);
        };
    }

    #[test]
    fn test_slice() {
        let x = steps(&[2, 3, 4]);
        slice_tests!(x.view(), slice);
    }

    #[test]
    fn test_slice_mut() {
        let mut x = steps(&[2, 3, 4]);
        slice_tests!(x, slice_mut);
    }

    #[test]
    fn test_slice_iter() {
        let sr = |start, end| SliceItem::range(start, Some(end), 1);
        let x = steps(&[3, 3]);

        // Slice that extracts a specific index
        let slice: Vec<_> = x
            .slice_iter(&[SliceItem::Index(0), SliceItem::full_range()])
            .copied()
            .collect();
        assert_eq!(slice, &[1, 2, 3]);

        // Slice that removes start of each dimension
        let slice: Vec<_> = x.slice_iter(&[sr(1, 3), sr(1, 3)]).copied().collect();
        assert_eq!(slice, &[5, 6, 8, 9]);

        // Slice that removes end of each dimension
        let slice: Vec<_> = x.slice_iter(&[sr(0, 2), sr(0, 2)]).copied().collect();
        assert_eq!(slice, &[1, 2, 4, 5]);

        // Slice that removes start and end of first dimension
        let slice: Vec<_> = x.slice_iter(&[sr(1, 2), sr(0, 3)]).copied().collect();
        assert_eq!(slice, &[4, 5, 6]);

        // Slice that removes start and end of second dimension
        let slice: Vec<_> = x.slice_iter(&[sr(0, 3), sr(1, 2)]).copied().collect();
        assert_eq!(slice, &[2, 5, 8]);
    }

    #[test]
    fn test_slice_iter_with_step() {
        let sr = |start, end, step| SliceItem::range(start, Some(end), step);
        let x = steps(&[10]);

        // Positive steps > 1.
        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, 2)]).copied().collect();
        assert_eq!(slice, &[1, 3, 5, 7, 9]);

        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, 3)]).copied().collect();
        assert_eq!(slice, &[1, 4, 7, 10]);

        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, 10)]).copied().collect();
        assert_eq!(slice, &[1]);

        // Negative steps.
        let slice: Vec<_> = x.slice_iter(&[sr(10, -11, -1)]).copied().collect();
        assert_eq!(slice, &[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);

        let slice: Vec<_> = x.slice_iter(&[sr(8, 0, -1)]).copied().collect();
        assert_eq!(slice, &[9, 8, 7, 6, 5, 4, 3, 2]);

        let slice: Vec<_> = x.slice_iter(&[sr(10, 0, -2)]).copied().collect();
        assert_eq!(slice, &[10, 8, 6, 4, 2]);

        let slice: Vec<_> = x.slice_iter(&[sr(10, 0, -10)]).copied().collect();
        assert_eq!(slice, &[10]);
    }

    #[test]
    fn test_slice_iter_negative_indices() {
        let sr = |start, end| SliceItem::range(start, Some(end), 1);
        let x = steps(&[10]);

        // Negative start
        let slice: Vec<_> = x.slice_iter(&[sr(-2, 10)]).copied().collect();
        assert_eq!(slice, &[9, 10]);

        // Negative end
        let slice: Vec<_> = x.slice_iter(&[sr(7, -1)]).copied().collect();
        assert_eq!(slice, &[8, 9]);

        // Negative start and end
        let slice: Vec<_> = x.slice_iter(&[sr(-3, -1)]).copied().collect();
        assert_eq!(slice, &[8, 9]);
    }

    #[test]
    fn test_slice_iter_clamps_indices() {
        let sr = |start, end, step| SliceItem::range(start, Some(end), step);
        let x = steps(&[5]);

        // Test cases for positive steps (ie. traversing forwards).

        // Positive start out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(10, 11, 1)]).collect();
        assert_eq!(slice.len(), 0);

        // Positive end out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, 1)]).copied().collect();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        // Negative start out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(-10, 5, 1)]).copied().collect();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        // Negative end out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(-10, -5, 1)]).collect();
        assert_eq!(slice.len(), 0);

        // Test cases for negative steps (ie. traversing backwards).

        // Positive start out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(10, -6, -1)]).copied().collect();
        assert_eq!(slice, &[5, 4, 3, 2, 1]);

        // Positive end out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, -1)]).collect();
        assert_eq!(slice.len(), 0);

        // Negative start out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(-10, 5, -1)]).collect();
        assert_eq!(slice.len(), 0);

        // Negative end out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(-1, -10, -1)]).copied().collect();
        assert_eq!(slice, &[5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_slice_iter_start_end_step_combinations() {
        let sr = |start, end, step| SliceItem::range(start, Some(end), step);
        let x = steps(&[3]);

        // Test various combinations of slice starts, ends and steps that are
        // positive and negative, in-bounds and out-of-bounds, and ensure they
        // don't cause a panic.
        for start in -5..5 {
            for end in -5..5 {
                for step in -5..5 {
                    if step == 0 {
                        continue;
                    }
                    x.slice_iter(&[sr(start, end, step)]).for_each(drop);
                }
            }
        }
    }

    // These tests assume the correctness of `slice_iter`, given the tests
    // above, and check for consistency between the results of `slice_offsets`
    // and `slice_iter`.
    #[test]
    fn test_slice_offsets() {
        let x = steps(&[5, 5]);

        // Range that removes the start and end of each dimension.
        let range = &[
            SliceItem::range(1, Some(4), 1),
            SliceItem::range(1, Some(4), 1),
        ];
        let expected: Vec<_> = x.slice_iter(range).copied().collect();
        let x_data = x.data().unwrap();
        let result: Vec<_> = x
            .slice_offsets(range)
            .map(|offset| x_data[offset])
            .collect();

        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_squeezed() {
        let mut rng = XorShiftRng::new(1234);
        let x = Tensor::rand(&[1, 1, 10, 20], &mut rng);
        let y = x.squeezed();
        assert_eq!(y.data(), x.data());
        assert_eq!(y.shape(), &[10, 20]);
        assert_eq!(y.stride(0), 20);
        assert_eq!(y.stride(1), 1);
    }

    #[test]
    fn test_write() -> std::io::Result<()> {
        use std::io::{Cursor, Read};
        let x = Tensor::from_data(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let mut buf: Vec<u8> = Vec::new();

        x.write(&mut buf)?;

        assert_eq!(buf.len(), 4 + x.ndim() * 4 + x.len() * 4);

        let mut cursor = Cursor::new(buf);
        let mut tmp = [0u8; 4];

        cursor.read(&mut tmp)?;
        let ndim = u32::from_le_bytes(tmp);
        assert_eq!(ndim, x.ndim() as u32);

        for &size in x.shape().iter() {
            cursor.read(&mut tmp)?;
            let written_size = u32::from_le_bytes(tmp);
            assert_eq!(written_size, size as u32);
        }

        for el in x.iter().copied() {
            cursor.read(&mut tmp)?;
            let written_el = f32::from_le_bytes(tmp);
            assert_eq!(written_el, el);
        }

        Ok(())
    }
}
