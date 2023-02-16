use std::borrow::Cow;
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};

use crate::linalg::Matrix;

#[cfg(test)]
use crate::rng::XorShiftRng;

mod index_iterator;
mod iterators;
mod layout;
mod macros;
mod range;
mod vec_with_offset;

pub use self::index_iterator::IndexIterator;
pub use self::iterators::{BroadcastElements, Elements, ElementsMut, Offsets};
pub(crate) use self::layout::Layout;
pub use self::range::{SliceItem, SliceRange};
pub use layout::is_valid_permutation;
use vec_with_offset::VecWithOffset;

/// Provides methods for querying the shape and data layout of a [Tensor]
/// or [TensorView].
pub trait TensorLayout {
    /// Returns the internal struct that contains layout information for the tensor.
    #[doc(hidden)]
    fn layout(&self) -> &Layout;

    /// Return a slice of the sizes of each dimension.
    fn shape(&self) -> &[usize] {
        self.layout().shape()
    }

    /// Return the number of elements between successive entries in the `dim`
    /// dimension.
    fn stride(&self, dim: usize) -> usize {
        self.layout().stride(dim)
    }

    /// Return the total number of elements in this tensor.
    fn len(&self) -> usize {
        self.layout().len()
    }

    /// Return true if this tensor has no elements.
    fn is_empty(&self) -> bool {
        self.layout().is_empty()
    }

    /// Return the number of dimensions the tensor has, aka. the rank of the
    /// tensor.
    fn ndim(&self) -> usize {
        self.layout().ndim()
    }

    /// Return an iterator over all valid indices in this tensor.
    ///
    /// The returned iterator does not implement the `Iterator` trait but has
    /// a similar API. See `IndexIterator` docs.
    fn indices(&self) -> IndexIterator {
        IndexIterator::from_shape(self.shape())
    }

    /// Return true if the logical order of elements in this tensor matches the
    /// order in which elements are stored in the underlying array.
    fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Return the offset of an element in the array.
    ///
    /// The length of `index` must match the tensor's dimension count.
    ///
    /// Panics if the index length is incorrect or the value of an index
    /// exceeds the size of the corresponding dimension.
    fn offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        self.layout().offset(index)
    }

    /// Return the offset of the first element in a slice of the array.
    ///
    /// This is the same as `slice`, except that `index` can have fewer
    /// dimensions than the tensor, in which case the index is implicitly
    /// zero-padded on the right.
    fn slice_offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        self.layout().slice_offset(index)
    }

    /// Return an iterator over offsets of elements in this tensor, in their
    /// logical order.
    ///
    /// See also the notes for `slice_offsets`.
    fn offsets(&self) -> Offsets {
        Offsets::new(self.layout())
    }

    /// Return an iterator over offsets of this tensor, broadcasted to `shape`.
    ///
    /// This is very similar to `broadcast_iter`, except that the iterator
    /// yields offsets into rather than elements of the data buffer.
    fn broadcast_offsets(&self, shape: &[usize]) -> Offsets {
        assert!(
            self.can_broadcast_to(shape),
            "Cannot broadcast to specified shape"
        );
        Offsets::broadcast(self.layout(), shape)
    }

    /// Return an iterator over offsets of elements in this tensor.
    ///
    /// Note that the offset order of the returned iterator will become incorrect
    /// if the tensor's layout is modified during iteration.
    fn slice_offsets(&self, ranges: &[SliceRange]) -> Offsets {
        Offsets::slice(self.layout(), ranges)
    }

    /// Return true if the tensor/view can be broadcast with another tensor or
    /// view with a given `shape` as part of a binary operation.
    ///
    /// The shape of the result may be larger than either the current shape
    /// or `shape`. eg. If a tensor of shape `[1, 5]` is broadcast with one
    /// of size `[2, 1, 1]` the result has shape `[2, 1, 5]`.
    ///
    /// See <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md> for
    /// conditions in which broadcasting is allowed.
    fn can_broadcast_with(&self, shape: &[usize]) -> bool {
        self.layout().can_broadcast_with(shape)
    }

    /// Return true if the tensor/view can be broadcast to a given `shape`.
    ///
    /// See <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md> for
    /// conditions in which broadcasting is allowed.
    fn can_broadcast_to(&self, shape: &[usize]) -> bool {
        self.layout().can_broadcast_to(shape)
    }

    /// Return the shape of this tensor/view as a fixed-sized array.
    ///
    /// Panics if the tensor's dimension count does not match `N`.
    fn dims<const N: usize>(&self) -> [usize; N] {
        self.layout().dims()
    }
}

/// Trait for indexing a `Tensor`
pub trait TensorIndex {
    /// Return the number of dimensions in the index.
    fn len(&self) -> usize;

    /// Return the index for dimension `dim`
    fn index(&self, dim: usize) -> usize;
}

impl<const N: usize> TensorIndex for [usize; N] {
    fn len(&self) -> usize {
        N
    }

    fn index(&self, dim: usize) -> usize {
        self[dim]
    }
}

impl TensorIndex for &[usize] {
    fn len(&self) -> usize {
        (self as &[usize]).len()
    }

    fn index(&self, dim: usize) -> usize {
        self[dim]
    }
}

/// TensorBase is the base tensor type used by [Tensor], [TensorView]
/// and [TensorViewMut].
///
/// It is parametrized by an element type T and the data storage type S.
#[derive(Debug)]
pub struct TensorBase<T: Copy, S: AsRef<[T]>> {
    data: S,
    layout: Layout,
    element_type: PhantomData<T>,
}

/// TensorView provides a view onto data owned by a [Tensor].
///
/// Conceptually the relationship between TensorView and Tensor is similar to
/// that between slice and Vec. They share the same element buffer, but views
/// can have distinct layouts, with some limitations.
pub type TensorView<'a, T = f32> = TensorBase<T, &'a [T]>;

/// TensorViewMut provides a mutable view onto data owned by a [Tensor].
///
/// This is similar to [TensorView], except elements in the underyling
/// Tensor can be modified through it.
pub type TensorViewMut<'a, T = f32> = TensorBase<T, &'a mut [T]>;

impl<T: Copy, S: AsRef<[T]>> TensorBase<T, S> {
    fn new(data: S, layout: &Layout) -> Self {
        TensorBase {
            data,
            layout: layout.clone(),
            element_type: PhantomData,
        }
    }

    /// Return the underlying element buffer for this tensor or view.
    ///
    /// If the tensor is contiguous, the buffer will contain the same elements
    /// in the same order as yielded by [Tensor::iter]. In other cases the buffer
    /// may have unused indexes or a different ordering.
    ///
    /// The lifetime of the returned data is tied to self. See [TensorBase::to_data]
    /// for getting data with lifetime tied to the underlying storage.
    pub fn data(&self) -> &[T] {
        self.data.as_ref()
    }

    /// Return a copy of this tensor with each element replaced by `f(element)`.
    ///
    /// The order in which elements are visited is unspecified and may not
    /// correspond to the logical order.
    pub fn map<F, U: Copy>(&self, f: F) -> Tensor<U>
    where
        F: Fn(T) -> U,
    {
        let data = self.iter().map(f).collect();
        Tensor {
            data: VecWithOffset::new(data),
            layout: self.layout.clone(),
            element_type: PhantomData,
        }
    }

    /// Return a new contiguous tensor with the same shape and elements as this
    /// view.
    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor::from_data(self.shape(), self.iter().collect())
    }

    /// Return a copy of the elements of this tensor as a contiguous vector
    /// in row-major order.
    ///
    /// This is slightly more efficient than `iter().collect()` in the case
    /// where the tensor is already contiguous.
    pub fn to_vec(&self) -> Vec<T> {
        if self.is_contiguous() {
            self.data().to_vec()
        } else {
            self.iter().collect()
        }
    }

    /// Returns the single item if this tensor is a 0-dimensional tensor
    /// (ie. a scalar)
    pub fn item(&self) -> Option<T> {
        match self.ndim() {
            0 => Some(self.data.as_ref()[0]),
            _ if self.len() == 1 => self.iter().next(),
            _ => None,
        }
    }

    /// Return an immutable view of this tensor.
    ///
    /// Views share the same element array, but can have an independent layout,
    /// with some limitations.
    pub fn view(&self) -> TensorView<T> {
        TensorView::new(self.data.as_ref(), &self.layout)
    }

    /// Return an iterator over elements of this tensor, in their logical order.
    pub fn iter(&self) -> Elements<T> {
        Elements::new(self)
    }

    /// Return a new view which views a subset of the elements accessible in
    /// this view.
    pub fn slice(&self, range: &[SliceItem]) -> TensorView<T> {
        let (offset, layout) = self.layout.slice(range);
        TensorBase {
            data: &self.data.as_ref()[offset..offset + layout.end_offset()],
            layout,
            element_type: PhantomData,
        }
    }

    /// Return an iterator over elements of this tensor, broadcasted to `shape`.
    ///
    /// A broadcasted iterator behaves as if the tensor had the broadcasted
    /// shape, repeating elements as necessary to fill the given dimensions.
    /// Broadcasting is only possible if the actual and broadcast shapes are
    /// compatible according to ONNX's rules. See
    /// <https://github.com/onnx/onnx/blob/main/docs/Operators.md>.
    ///
    /// See also <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>
    /// for worked examples of how broadcasting works.
    pub fn broadcast_iter(&self, shape: &[usize]) -> BroadcastElements<T> {
        assert!(
            self.can_broadcast_to(shape),
            "Cannot broadcast to specified shape"
        );
        BroadcastElements::new(&self, shape)
    }

    /// Return an iterator over a subset of elements in this tensor.
    pub fn slice_iter(&self, ranges: &[SliceRange]) -> Elements<T> {
        Elements::slice(&self, ranges)
    }

    /// Change the layout to put dimensions in the order specified by `dims`.
    ///
    /// This does not modify the order of elements in the data buffer, it just
    /// updates the strides used by indexing.
    pub fn permute(&mut self, dims: &[usize]) {
        self.layout.permute(dims);
    }

    /// Reverse the order of dimensions.
    ///
    /// This does not modify the order of elements in the data buffer, it just
    /// changes the strides used by indexing.
    pub fn transpose(&mut self) {
        self.layout.transpose();
    }

    /// Return an immutable copy of this view.
    pub fn as_view(&self) -> TensorView<T> {
        TensorView::new(self.data.as_ref(), &self.layout)
    }
}

impl<'a, T: Copy> TensorBase<T, &'a [T]> {
    /// Return the slice underlying this view.
    ///
    /// This is similar to [TensorBase::data], but the lifetime is that of the
    /// underlying storage rather than the view.
    pub fn to_data(&self) -> &'a [T] {
        self.data
    }

    /// Change the layout of this view to have the given shape.
    ///
    /// The current view must be contiguous and the new shape must have the
    /// same product as the current shape.
    pub fn reshape(&mut self, shape: &[usize]) {
        self.layout.reshape(shape);
    }

    /// Return a new view with the dimensions re-ordered according to `dims`.
    pub fn permuted(&self, dims: &[usize]) -> Self {
        Self {
            data: self.data,
            layout: self.layout.permuted(dims),
            element_type: PhantomData,
        }
    }

    /// Return a new view with the order of dimensions reversed.
    pub fn transposed(&self) -> Self {
        Self {
            data: self.data,
            layout: self.layout.transposed(),
            element_type: PhantomData,
        }
    }

    /// Return a new view with a given shape. This has the same requirements
    /// as `reshape`.
    pub fn reshaped(&self, shape: &[usize]) -> Self {
        Self {
            data: self.data,
            layout: self.layout.reshaped(shape),
            element_type: PhantomData,
        }
    }

    /// Return an unchecked version of this view.
    ///
    /// This provides faster indexing at the cost of not bounds-checking
    /// individual dimensions.
    ///
    /// Panics if the rank of this view is not `N`.
    pub fn unchecked_view<const N: usize>(&self) -> UncheckedView<T, N> {
        UncheckedView {
            data: self.data.as_ref(),
            strides: self.layout.strides().try_into().unwrap(),
        }
    }
}

impl<T: Copy, S: AsRef<[T]>> TensorLayout for TensorBase<T, S> {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

pub trait AsMatrix<'a> {
    fn as_matrix(&self) -> Matrix<'a>;
}

impl<'a> AsMatrix<'a> for TensorView<'a, f32> {
    fn as_matrix(&self) -> Matrix<'a> {
        assert!(
            self.layout.ndim() == 2,
            "Can only convert 2D view to matrix"
        );
        let shape = self.shape();
        Matrix::from_slice(
            self.data,
            shape[0],
            shape[1],
            Some((self.layout.stride(0), self.layout.stride(1))),
        )
    }
}

impl<I: TensorIndex, T: Copy, S: AsRef<[T]>> Index<I> for TensorBase<T, S> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        &self.data.as_ref()[self.offset(index)]
    }
}

impl<T: Copy, S: AsRef<[T]> + AsMut<[T]>> TensorBase<T, S> {
    /// Return the slice of the underlying array that is accessible through this
    /// view.
    ///
    /// WARNING: See notes about ordering in [Tensor::data].
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }

    /// Return a mutable iterator over elements of this view.
    pub fn iter_mut(&mut self) -> ElementsMut<T> {
        let layout = &self.layout;
        ElementsMut::new(self.data.as_mut(), layout)
    }

    /// Replace elements of this tensor with `f(element)`.
    ///
    /// This is the in-place version of `map`.
    ///
    /// The order in which elements are visited is unspecified and may not
    /// correspond to the logical order.
    pub fn apply<F: Fn(T) -> T>(&mut self, f: F) {
        // TODO: Skip unused elements when tensor is not contiguous.
        for val in self.data.as_mut().iter_mut() {
            *val = f(*val);
        }
    }

    /// Return a new mutable view of a subset of the elements in this view.
    ///
    /// Slices are specified in the same way as for [TensorView::slice].
    pub fn slice_mut(&mut self, range: &[SliceItem]) -> TensorViewMut<T> {
        let (offset, layout) = self.layout.slice(range);
        let data = &mut self.data.as_mut()[offset..offset + layout.end_offset()];

        TensorViewMut {
            data,
            layout,
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
}

impl<'a, T: Copy> TensorBase<T, &'a mut [T]> {
    /// Consume this view and return the underlying data slice.
    ///
    /// This differs from [Self::data_mut] as the lifetime of the returned slice
    /// is tied to the underlying tensor, rather than the view.
    pub fn into_data_mut(self) -> &'a mut [T] {
        self.data
    }

    /// Return a new view with the dimensions re-ordered according to `dims`.
    pub fn permuted(&mut self, dims: &[usize]) -> TensorBase<T, &mut [T]> {
        TensorBase {
            data: self.data,
            layout: self.layout.permuted(dims),
            element_type: PhantomData,
        }
    }

    /// Return a new view with the order of dimensions reversed.
    pub fn transposed(&mut self) -> TensorBase<T, &mut [T]> {
        TensorBase {
            data: self.data,
            layout: self.layout.transposed(),
            element_type: PhantomData,
        }
    }

    /// Return a new view with a given shape. This has the same requirements
    /// as `reshape`.
    pub fn reshaped(&mut self, shape: &[usize]) -> TensorBase<T, &mut [T]> {
        TensorBase {
            data: self.data,
            layout: self.layout.reshaped(shape),
            element_type: PhantomData,
        }
    }

    /// Return an unchecked version of this view.
    ///
    /// This provides faster indexing at the cost of not bounds-checking
    /// individual dimensions.
    ///
    /// Panics if the rank of this view is not `N`.
    pub fn unchecked_view_mut<const N: usize>(&mut self) -> UncheckedViewMut<T, N> {
        UncheckedViewMut {
            data: self.data.as_mut(),
            strides: self.layout.strides().try_into().unwrap(),
        }
    }
}

impl<I: TensorIndex, T: Copy, S: AsRef<[T]> + AsMut<[T]>> IndexMut<I> for TensorBase<T, S> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let offset = self.offset(index);
        &mut self.data.as_mut()[offset]
    }
}

/// Tensor is the core n-dimensional array type used for inputs, outputs and
/// intermediate values when executing a [crate::Model].
///
/// Tensor and its associated types [TensorView] and [TensorViewMut] are
/// conceptually a pair of a dynamically sized array (either owned or a
/// reference) and a _layout_ which specifies how to view the contents of that
/// array as an N-dimensional tensor. The layout specifies the number of
/// dimensions, size of each dimension and stride of each dimension (offset
/// between elements in the underlying array).
///
/// Information about a tensor or view's layout is available via the
/// [TensorLayout] trait.
///
/// By default, new tensors have a _contiguous_ layout, in which the stride of
/// the innermost (fastest-changing) dimension `Dn` is 1, the stride of
/// dimension `Di+1` is the size of dimension `Di` and so on. The layout will
/// become non-contiguous if the dimensions are permuted/transposed or if the
/// tensor is sliced in-place. Whether the tensor is contiguous does not matter
/// if accessing elements via indexing, slicing or iterators. It does matter if
/// accessing the underlying element buffer directly.
pub type Tensor<T = f32> = TensorBase<T, VecWithOffset<T>>;

impl<T: Copy> TensorBase<T, VecWithOffset<T>> {
    /// Create a new zero-filled tensor with a given shape.
    pub fn zeros(shape: &[usize]) -> Tensor<T>
    where
        T: Default,
    {
        let n_elts = shape.iter().product();
        let data = vec![T::default(); n_elts];
        Tensor {
            data: VecWithOffset::new(data),
            layout: Layout::new(shape),
            element_type: PhantomData,
        }
    }

    /// Create a new tensor from a given shape and set of elements. No copying
    /// is required.
    pub fn from_data(shape: &[usize], data: Vec<T>) -> Tensor<T> {
        assert!(
            shape[..].iter().product::<usize>() == data.len(),
            "Number of elements given by shape {:?} does not match data length {}",
            shape,
            data.len()
        );
        Tensor {
            data: VecWithOffset::new(data),
            layout: Layout::new(shape),
            element_type: PhantomData,
        }
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
    pub fn clone_with_shape(&self, shape: &[usize]) -> Tensor<T> {
        let data = if self.is_contiguous() {
            self.data().into()
        } else {
            self.iter().collect()
        };
        Self::from_data(shape, data)
    }

    /// Clip dimension `dim` to `[range.start, range.end)`. The new size for
    /// the dimension must be <= the old size.
    ///
    /// This is a fast operation since it just alters the start offset within
    /// the tensor's element buffer and length of the specified dimension.
    pub fn clip_dim(&mut self, dim: usize, range: Range<usize>) {
        let (start, end) = (range.start, range.end);

        assert!(start <= end, "start must be <= end");
        assert!(end <= self.shape()[dim], "end must be <= dim size");

        let start_offset = self.layout.stride(dim) * start;
        self.layout.resize_dim(dim, end - start);
        self.data
            .set_used_range(start_offset..start_offset + self.layout.end_offset());
    }

    /// Return a contiguous slice of `len` elements starting at `index`.
    /// `len` must be less than or equal to the size of the last dimension.
    ///
    /// Using a slice can allow for very efficient access to a range of elements
    /// in a single row or column (or whatever the last dimension represents).
    #[doc(hidden)]
    pub fn last_dim_slice<const N: usize>(&self, index: [usize; N], len: usize) -> &[T] {
        assert!(
            self.stride(N - 1) == 1,
            "last_dim_slice requires contiguous last dimension"
        );
        let offset = self.offset(index);
        &self.data[offset..offset + len]
    }

    /// Similar to `last_dim_slice`, but returns a mutable slice.
    #[doc(hidden)]
    pub fn last_dim_slice_mut<const N: usize>(
        &mut self,
        index: [usize; N],
        len: usize,
    ) -> &mut [T] {
        assert!(
            self.stride(N - 1) == 1,
            "last_dim_slice_mut requires contiguous last dimension"
        );
        let offset = self.offset(index);
        &mut self.data[offset..offset + len]
    }

    /// Convert the internal layout of elements to be contiguous, as reported
    /// by `is_contiguous`.
    ///
    /// This is a no-op if the tensor is already contiguous.
    pub fn make_contiguous(&mut self) {
        if self.is_contiguous() {
            return;
        }
        self.data = VecWithOffset::new(self.iter().collect());
        self.layout.make_contiguous();
    }

    /// Return a contiguous version of this tensor, either as a reference if
    /// the tensor is already contiguous, or a copy if not.
    pub fn as_contiguous(&self) -> Cow<Tensor<T>> {
        if self.is_contiguous() {
            Cow::Borrowed(self)
        } else {
            let mut contiguous_layout = self.layout.clone();
            contiguous_layout.make_contiguous();
            Cow::Owned(Tensor {
                data: VecWithOffset::new(self.iter().collect()),
                layout: contiguous_layout,
                element_type: PhantomData,
            })
        }
    }

    /// Update the shape of the tensor.
    ///
    /// The total number of elements for the new shape must be the same as the
    /// existing shape.
    ///
    /// This is a cheap operation if the tensor is contiguous, but requires
    /// copying data if the tensor has a non-contiguous layout.
    pub fn reshape(&mut self, shape: &[usize]) {
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
        self.layout = Layout::new(shape);
    }

    /// Insert a dimension of size one at index `dim`.
    pub fn insert_dim(&mut self, dim: usize) {
        let mut new_shape: Vec<usize> = self.shape().into();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape);
    }

    /// Return an _unchecked_ view of a subset of the data in this tensor.
    ///
    /// "Unchecked" means that individual dimensions of an index are not
    /// bounds-checked against the tensor's shape, but the final offset that
    /// is generated is.
    ///
    /// Base specifies zero or more indices to slice the view with, and N
    /// is the nubmer of indices to use for unchecked indexing. `B + N` must
    /// equal `self.ndim()`.
    #[doc(hidden)]
    pub fn unchecked_view<const B: usize, const N: usize>(
        &self,
        base: [usize; B],
    ) -> UncheckedView<T, N> {
        assert!(B + N == self.ndim());
        let offset = self.slice_offset(base);
        UncheckedView {
            data: &self.data()[offset..],
            strides: self.layout.strides()[self.ndim() - N..].try_into().unwrap(),
        }
    }

    /// Return an _unchecked_ mutable view of a subset of the data in this tensor.
    ///
    /// This is the same as [Tensor::unchecked_view] except that the returned view can
    /// be used to modify elements.
    #[doc(hidden)]
    pub fn unchecked_view_mut<const B: usize, const N: usize>(
        &mut self,
        base: [usize; B],
    ) -> UncheckedViewMut<T, N> {
        assert!(B + N == self.ndim());
        let offset = self.slice_offset(base);
        let strides = self.layout.strides()[self.ndim() - N..].try_into().unwrap();
        UncheckedViewMut {
            data: &mut self.data_mut()[offset..],
            strides,
        }
    }
}

impl Tensor<f32> {
    /// Serialize the tensor to a simple binary format.
    ///
    /// The serialized data is in little-endian order and has the structure:
    ///
    /// `[rank: u32][dim: u32 * rank][element: T * product(dims)]`
    ///
    /// Where `T` is the tensor's element type.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let ndim: u32 = self.ndim() as u32;
        writer.write_all(&ndim.to_le_bytes())?;
        for &dim in self.shape() {
            writer.write_all(&(dim as u32).to_le_bytes())?;
        }
        for el in self.iter() {
            writer.write_all(&el.to_le_bytes())?;
        }
        Ok(())
    }
}

impl<T: Copy + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.iter().eq(other.iter())
    }
}

impl<T: Copy> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        let data = self.data.clone();
        Tensor {
            data,
            layout: self.layout.clone(),
            element_type: PhantomData,
        }
    }
}

impl<T: Copy> FromIterator<T> for Tensor<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<_> = FromIterator::from_iter(iter);
        Tensor::from_vec(data)
    }
}

/// A view of a tensor which offers faster indexing at the cost of requiring
/// the rank to be known at compile time, and not bounds-checking individual
/// dimensions of an index.
///
/// Although individual dimensions are not bounds-checked, the generated offset
/// into the underlying buffer is, so unchecked views are not unsafe, in the
/// usual Rust sense. If a dimension index is out of bounds, but the offset
/// is still in-bounds, the view will access the wrong element.
///
/// UncheckedView and UncheckedViewMut can make operations which perform a
/// large number of indexed accesses, into tensors of a known rank, much faster.
pub struct UncheckedView<'a, T: Copy, const N: usize> {
    data: &'a [T],
    strides: [usize; N],
}

impl<'a, const N: usize, T: Copy> Index<[usize; N]> for UncheckedView<'a, T, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let mut offset = 0;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        &self.data[offset]
    }
}

/// Variant of [UncheckedView] which suppors mutable indexing into a view or
/// tensor.
pub struct UncheckedViewMut<'a, T: Copy, const N: usize> {
    data: &'a mut [T],
    strides: [usize; N],
}

impl<'a, const N: usize, T: Copy> Index<[usize; N]> for UncheckedViewMut<'a, T, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let mut offset = 0;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        &self.data[offset]
    }
}

impl<'a, const N: usize, T: Copy> IndexMut<[usize; N]> for UncheckedViewMut<'a, T, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let mut offset = 0;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        &mut self.data[offset]
    }
}

/// Create a new tensor with all values set to 0.
#[cfg(test)]
pub fn zeros<T: Copy + Default>(shape: &[usize]) -> Tensor<T> {
    Tensor::zeros(shape)
}

/// Create a new tensor filled with random values supplied by `rng`.
#[cfg(test)]
pub fn rand(shape: &[usize], rng: &mut XorShiftRng) -> Tensor {
    let mut t = zeros(shape);
    t.data_mut().fill_with(|| rng.next_f32());
    t
}

/// Create a new tensor with a given shape and values
#[cfg(test)]
pub fn from_data<T: Copy>(shape: &[usize], data: Vec<T>) -> Tensor<T> {
    Tensor::from_data(shape, data)
}

/// Create a new 0-dimensional (scalar) tensor from a single value.
#[cfg(test)]
pub fn from_scalar<T: Copy>(value: T) -> Tensor<T> {
    Tensor::from_scalar(value)
}

/// Create a new 1-dimensional tensor from a vector
#[cfg(test)]
pub fn from_vec<T: Copy>(data: Vec<T>) -> Tensor<T> {
    Tensor::from_vec(data)
}

/// Create a new 2D tensor from a nested array of slices.
#[cfg(test)]
pub fn from_2d_slice<T: Copy>(data: &[&[T]]) -> Tensor<T> {
    let rows = data.len();
    let cols = data.get(0).map(|first_row| first_row.len()).unwrap_or(0);

    let mut result = Vec::new();
    for row in data {
        assert!(cols == row.len(), "All row slices must have same length");
        result.extend_from_slice(row);
    }

    from_data(&[rows, cols], result)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::ops::IndexMut;

    use crate::rng::XorShiftRng;
    use crate::tensor;
    use crate::tensor::{
        from_2d_slice, from_data, from_scalar, from_vec, rand, zeros, SliceRange, Tensor,
        TensorLayout,
    };

    /// Create a tensor where the value of each element is its logical index
    /// plus one.
    fn steps(shape: &[usize]) -> Tensor<i32> {
        let mut x = zeros(shape);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = (index + 1) as i32;
        }
        x
    }

    #[test]
    fn test_apply() {
        let mut x = steps(&[3, 3]);
        x.apply(|el| el * el);
        let expected = from_data(&[3, 3], vec![1, 4, 9, 16, 25, 36, 49, 64, 81]);
        assert_eq!(x, expected);
    }

    #[test]
    fn test_clip_dim() {
        let mut x = steps(&[3, 3]);
        x.clip_dim(0, 1..2);
        x.clip_dim(1, 1..2);
        assert_eq!(x.iter().collect::<Vec<i32>>(), vec![5]);
    }

    #[test]
    fn test_clip_dim_start() {
        let mut x = steps(&[3, 3]);

        // Clip the start of the tensor, adjusting the `base` offset.
        x.clip_dim(0, 1..3);

        // Indexing should reflect the slice.
        assert_eq!(x.iter().collect::<Vec<i32>>(), &[4, 5, 6, 7, 8, 9]);
        assert_eq!(x[[0, 0]], 4);
        assert_eq!(*x.index_mut([0, 0]), 4);

        // Slices returned by `data`, `data_mut` should reflect the slice.
        assert_eq!(x.data(), &[4, 5, 6, 7, 8, 9]);
        assert_eq!(x.data_mut(), &[4, 5, 6, 7, 8, 9]);

        // Offsets should be relative to the sliced returned by `data`,
        // `data_mut`.
        assert_eq!(x.offsets().collect::<Vec<usize>>(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(x.offset([0, 0]), 0);
    }

    #[test]
    fn test_from_2d_slice() {
        let x = from_2d_slice(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);
        assert_eq!(x.shape(), &[3, 3]);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_from_scalar() {
        let x = from_scalar(5);
        assert_eq!(x.shape().len(), 0);
        assert_eq!(x.data(), &[5]);
    }

    #[test]
    fn test_from_vec() {
        let x = tensor!([1, 2, 3]);
        assert_eq!(x.shape(), &[3]);
        assert_eq!(x.data(), &[1, 2, 3]);
    }

    #[test]
    fn test_from_iterator() {
        let x: Tensor<i32> = FromIterator::from_iter(0..10);
        assert_eq!(x.shape(), &[10]);
        assert_eq!(x.data(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_stride() {
        let x = zeros::<f32>(&[2, 5, 7, 3]);
        assert_eq!(x.stride(3), 1);
        assert_eq!(x.stride(2), 3);
        assert_eq!(x.stride(1), 7 * 3);
        assert_eq!(x.stride(0), 5 * 7 * 3);
    }

    #[test]
    fn test_index() {
        let mut x = zeros::<f32>(&[2, 2]);

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
        let x = from_scalar(5.0);
        assert_eq!(x[[]], 5.0);
    }

    #[test]
    fn test_index_mut() {
        let mut x = zeros::<f32>(&[2, 2]);

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
    #[should_panic]
    fn test_index_panics_if_invalid() {
        let x = zeros::<f32>(&[2, 2]);
        x[[2, 0]];
    }

    #[test]
    #[should_panic]
    fn test_index_panics_if_wrong_dim_count() {
        let x = zeros::<f32>(&[2, 2]);
        x[[0, 0, 0]];
    }

    #[test]
    fn test_indices() {
        let x = zeros::<f32>(&[2, 2]);
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
        let scalar = from_scalar(5.0);
        assert_eq!(scalar.item(), Some(5.0));

        let vec_one_item = tensor!([5.0]);
        assert_eq!(vec_one_item.item(), Some(5.0));

        let vec_many_items = tensor!([1.0, 2.0]);
        assert_eq!(vec_many_items.item(), None);

        let matrix_one_item = from_data(&[1, 1], vec![5.0]);
        assert_eq!(matrix_one_item.item(), Some(5.0));
    }

    #[test]
    fn test_ndim() {
        let scalar = from_scalar(5.0);
        let vec = tensor!([5.0]);
        let matrix = from_data(&[1, 1], vec![5.0]);

        assert_eq!(scalar.ndim(), 0);
        assert_eq!(vec.ndim(), 1);
        assert_eq!(matrix.ndim(), 2);
    }

    #[test]
    fn test_partial_eq() {
        let x = tensor!([1, 2, 3, 4, 5]);
        let y = x.clone();
        let z = x.clone_with_shape(&[1, 5]);

        // Int tensors are equal if they have the same shape and elements.
        assert_eq!(&x, &y);
        assert_ne!(&x, &z);
    }

    #[test]
    fn test_dims() {
        let x = zeros::<f32>(&[10, 5, 3, 7]);
        let [i, j, k, l] = x.dims();

        assert_eq!(i, 10);
        assert_eq!(j, 5);
        assert_eq!(k, 3);
        assert_eq!(l, 7);
    }

    #[test]
    #[should_panic]
    fn test_dims_panics_if_wrong_array_length() {
        let x = zeros::<f32>(&[10, 5, 3, 7]);
        let [_i, _j, _k] = x.dims();
    }

    #[test]
    fn test_len() {
        let scalar = from_scalar(5);
        let vec = tensor!([1, 2, 3]);
        let matrix = from_data(&[2, 2], vec![1, 2, 3, 4]);

        assert_eq!(scalar.len(), 1);
        assert_eq!(vec.len(), 3);
        assert_eq!(matrix.len(), 4);
    }

    #[test]
    fn test_is_empty() {
        assert!(from_vec::<f32>(vec![]).is_empty());
        assert!(!tensor!([1]).is_empty());
        assert!(!from_scalar(5.0).is_empty());
    }

    #[test]
    fn test_reshape() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = rand(&[10, 5, 3, 7], &mut rng);
        let x_data: Vec<f32> = x.data().into();

        assert_eq!(x.shape(), &[10, 5, 3, 7]);

        x.reshape(&[10, 5, 3 * 7]);

        assert_eq!(x.shape(), &[10, 5, 3 * 7]);
        assert_eq!(x.data(), x_data);
    }

    #[test]
    fn test_reshape_non_contiguous() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = rand(&[10, 10], &mut rng);

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
        assert_eq!(x.iter().collect::<Vec<i32>>(), &[5, 6, 8, 9]);
    }

    #[test]
    fn test_reshape_copies_with_custom_strides() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = rand(&[10, 10], &mut rng);

        // Give the tensor a non-default stride
        x.clip_dim(1, 0..8);
        assert!(!x.is_contiguous());
        let x_elements: Vec<f32> = x.iter().collect();

        x.reshape(&[80]);

        // Since the tensor had a non-default stride, `reshape` will have copied
        // data.
        assert_eq!(x.shape(), &[80]);
        assert!(x.is_contiguous());
        assert_eq!(x.data(), x_elements);
    }

    #[test]
    #[should_panic(expected = "New shape must have same total elements as current shape")]
    fn test_reshape_with_wrong_size() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = rand(&[10, 5, 3, 7], &mut rng);
        x.reshape(&[10, 5]);
    }

    #[test]
    fn test_permute() {
        // Test with a vector (this is a no-op)
        let mut input = steps(&[5]);
        assert!(input.iter().eq([1, 2, 3, 4, 5].iter().copied()));
        input.permute(&[0]);
        assert!(input.iter().eq([1, 2, 3, 4, 5].iter().copied()));

        // Test with a matrix (ie. transpose the matrix)
        let mut input = steps(&[2, 3]);
        assert!(input.iter().eq([1, 2, 3, 4, 5, 6].iter().copied()));
        input.permute(&[1, 0]);
        assert_eq!(input.shape(), &[3, 2]);
        assert!(input.iter().eq([1, 4, 2, 5, 3, 6].iter().copied()));

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
    #[should_panic(expected = "Permutation is invalid")]
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
        assert!(input.iter().eq([1, 2, 3, 4, 5, 6].iter().copied()));
        input.transpose();
        assert_eq!(input.shape(), &[3, 2]);
        assert!(input.iter().eq([1, 4, 2, 5, 3, 6].iter().copied()));

        // Test with a higher-rank tensor
        let mut input = steps(&[1, 3, 7]);
        input.transpose();
        assert_eq!(input.shape(), [7, 3, 1]);
    }

    #[test]
    fn test_insert_dim() {
        let mut input = steps(&[2, 3]);
        input.insert_dim(1);
        assert_eq!(input.shape(), &[2, 1, 3]);

        input.insert_dim(1);
        assert_eq!(input.shape(), &[2, 1, 1, 3]);

        input.insert_dim(0);
        assert_eq!(input.shape(), &[1, 2, 1, 1, 3]);
    }

    #[test]
    fn test_clone_with_shape() {
        let mut rng = XorShiftRng::new(1234);
        let x = rand(&[10, 5, 3, 7], &mut rng);
        let y = x.clone_with_shape(&[10, 5, 3 * 7]);

        assert_eq!(y.shape(), &[10, 5, 3 * 7]);
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_unchecked_view() {
        let mut rng = XorShiftRng::new(1234);
        let x = rand(&[10, 5, 3, 7], &mut rng);
        let x_view = x.unchecked_view([5, 3]);

        for a in 0..x.shape()[2] {
            for b in 0..x.shape()[3] {
                assert_eq!(x[[5, 3, a, b]], x_view[[a, b]]);
            }
        }
    }

    #[test]
    fn test_unchecked_view_mut() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = rand(&[10, 5, 3, 7], &mut rng);

        let [_, _, a_size, b_size] = x.dims();
        let mut x_view = x.unchecked_view_mut([5, 3]);

        for a in 0..a_size {
            for b in 0..b_size {
                x_view[[a, b]] = (a + b) as f32;
            }
        }

        for a in 0..x.shape()[2] {
            for b in 0..x.shape()[3] {
                assert_eq!(x[[5, 3, a, b]], (a + b) as f32);
            }
        }
    }

    #[test]
    fn test_last_dim_slice() {
        let mut rng = XorShiftRng::new(1234);
        let x = rand(&[10, 5, 3, 7], &mut rng);
        let x_slice = x.last_dim_slice([5, 3, 2, 0], x.shape()[3]);

        for i in 0..x.shape()[3] {
            assert_eq!(x[[5, 3, 2, i]], x_slice[i]);
        }
    }

    #[test]
    fn test_last_dim_slice_mut() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = rand(&[10, 5, 3, 7], &mut rng);
        let x_slice = x.last_dim_slice_mut([5, 3, 2, 0], x.shape()[3]);

        for val in x_slice.iter_mut() {
            *val = 1.0;
        }

        for i in 0..x.shape()[3] {
            assert_eq!(x[[5, 3, 2, i]], 1.0);
        }
    }

    #[test]
    fn test_iter_for_contiguous_array() {
        for dims in 1..7 {
            let mut shape = Vec::new();
            for d in 0..dims {
                shape.push(d + 1);
            }
            let mut rng = XorShiftRng::new(1234);
            let x = rand(&shape, &mut rng);

            let elts: Vec<f32> = x.iter().collect();

            assert_eq!(x.data(), elts);
        }
    }

    #[test]
    fn test_iter_for_empty_array() {
        let empty = zeros::<f32>(&[3, 0, 5]);
        assert!(empty.iter().next().is_none());
    }

    #[test]
    fn test_iter_for_non_contiguous_array() {
        let mut x = zeros(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        // Initially tensor is contiguous, so data buffer and element sequence
        // match.
        assert_eq!(x.data(), x.iter().collect::<Vec<_>>());

        // Slice the tensor along an outer dimension. This will leave the tensor
        // contiguous, and hence `data` and `elements` should return the same
        // elements.
        x.clip_dim(0, 0..2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(x.iter().collect::<Vec<_>>(), &[1, 2, 3, 4, 5, 6]);
        // Test with step > 1 to exercise `Elements::nth`.
        assert_eq!(x.iter().step_by(2).collect::<Vec<_>>(), &[1, 3, 5]);

        // Slice the tensor along an inner dimension. The tensor will no longer
        // be contiguous and hence `elements` will return different results than
        // `data`.
        x.clip_dim(1, 0..2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5]);
        assert_eq!(x.iter().collect::<Vec<_>>(), &[1, 2, 4, 5]);
        // Test with step > 1 to exercise `Elements::nth`.
        assert_eq!(x.iter().step_by(2).collect::<Vec<_>>(), &[1, 4]);
    }

    // PyTorch and numpy do not allow iteration over a scalar, but it seems
    // consistent for `Tensor::iter` to always yield `Tensor::len` elements,
    // and `len` returns 1 for a scalar.
    #[test]
    fn test_iter_for_scalar() {
        let x = from_scalar(5.0);
        let elements = x.iter().collect::<Vec<_>>();
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
            let mut x = rand(&shape, &mut rng);

            let elts: Vec<f32> = x.iter().map(|x| x * 2.).collect();

            for elt in x.iter_mut() {
                *elt *= 2.;
            }

            assert_eq!(x.data(), elts);
        }
    }

    #[test]
    fn test_iter_mut_for_non_contiguous_array() {
        let mut x = zeros(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
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
    fn test_to_vec() {
        let mut x = steps(&[3, 3]);

        // Contiguous case. This should use the fast-path.
        assert_eq!(x.to_vec(), x.iter().collect::<Vec<_>>());

        // Non-contiguous case.
        x.clip_dim(1, 0..2);
        assert!(!x.is_contiguous());
        assert_eq!(x.to_vec(), x.iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_offsets() {
        let mut rng = XorShiftRng::new(1234);
        let mut x = rand(&[10, 10], &mut rng);

        let x_elts: Vec<_> = x.iter().collect();

        let x_offsets = x.offsets();
        let x_data = x.data_mut();
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
        let scalar = from_data(&[], vec![1.0]);
        assert_eq!(scalar.len(), 1);

        let matrix = from_data(&[2, 2], vec![1, 2, 3, 4]);
        assert_eq!(matrix.shape(), &[2, 2]);
        assert_eq!(matrix.data(), &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_with_wrong_len() {
        from_data(&[1], vec![1, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_if_scalar_data_empty() {
        from_data::<i32>(&[], vec![]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_if_scalar_data_has_many_elements() {
        from_data(&[], vec![1, 2, 3]);
    }

    #[test]
    fn test_is_contiguous() {
        let mut x = zeros(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        // Freshly-allocated tensor
        assert!(x.is_contiguous());

        // Tensor where outermost dimension has been clipped at the end.
        let mut y = x.clone();
        y.clip_dim(0, 0..2);
        assert!(y.is_contiguous());
        assert_eq!(y.data(), &[1, 2, 3, 4, 5, 6]);

        // Tensor where outermost dimension has been clipped at the start.
        let mut y = x.clone();
        y.clip_dim(0, 1..3);
        assert!(y.is_contiguous());
        assert_eq!(y.data(), &[4, 5, 6, 7, 8, 9]);

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
        let mut x = zeros(&[10]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
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
        assert_eq!(x.iter().collect::<Vec<i32>>(), &[5, 6, 8, 9]);
    }

    #[test]
    fn test_as_contiguous() {
        let mut x = steps(&[3, 3]);
        let y = x.as_contiguous();
        assert!(matches!(y, Cow::Borrowed(_)));

        x.permute(&[1, 0]);
        let y = x.as_contiguous();
        assert!(matches!(y, Cow::Owned(_)));
    }

    #[test]
    fn test_broadcast_iter() {
        let x = steps(&[1, 2, 1, 2]);
        assert_eq!(x.iter().collect::<Vec<i32>>(), &[1, 2, 3, 4]);

        // Broadcast a 1-size dimension to size 2
        let bx = x.broadcast_iter(&[2, 2, 1, 2]);
        assert_eq!(bx.collect::<Vec<i32>>(), &[1, 2, 3, 4, 1, 2, 3, 4]);

        // Broadcast a different 1-size dimension to size 2
        let bx = x.broadcast_iter(&[1, 2, 2, 2]);
        assert_eq!(bx.collect::<Vec<i32>>(), &[1, 2, 1, 2, 3, 4, 3, 4]);

        // Broadcast to a larger number of dimensions
        let x = steps(&[5]);
        let bx = x.broadcast_iter(&[1, 5]);
        assert_eq!(bx.collect::<Vec<i32>>(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_broadcast_iter_with_scalar() {
        let scalar = from_scalar(7);
        let bx = scalar.broadcast_iter(&[3, 3]);
        assert_eq!(bx.collect::<Vec<i32>>(), &[7, 7, 7, 7, 7, 7, 7, 7, 7]);
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
    fn test_broadcast_offsets() {
        let x = steps(&[2, 1, 4]);
        let to_shape = &[2, 2, 1, 4];

        let expected: Vec<i32> = x.broadcast_iter(to_shape).collect();
        let actual: Vec<i32> = x
            .broadcast_offsets(to_shape)
            .map(|off| x.data()[off])
            .collect();

        assert_eq!(&actual, &expected);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast to specified shape")]
    fn test_broadcast_offsets_with_invalid_shape() {
        let x = steps(&[2, 2]);
        x.broadcast_offsets(&[3, 2]);
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
    fn test_slice_iter() {
        let sr = |start, end| SliceRange::new(start, end, 1);
        let x = steps(&[3, 3]);

        // Slice that removes start of each dimension
        let slice: Vec<_> = x.slice_iter(&[sr(1, 3), sr(1, 3)]).collect();
        assert_eq!(slice, &[5, 6, 8, 9]);

        // Slice that removes end of each dimension
        let slice: Vec<_> = x.slice_iter(&[sr(0, 2), sr(0, 2)]).collect();
        assert_eq!(slice, &[1, 2, 4, 5]);

        // Slice that removes start and end of first dimension
        let slice: Vec<_> = x.slice_iter(&[sr(1, 2), sr(0, 3)]).collect();
        assert_eq!(slice, &[4, 5, 6]);

        // Slice that removes start and end of second dimension
        let slice: Vec<_> = x.slice_iter(&[sr(0, 3), sr(1, 2)]).collect();
        assert_eq!(slice, &[2, 5, 8]);
    }

    #[test]
    fn test_slice_iter_with_step() {
        let sr = SliceRange::new;
        let x = steps(&[10]);

        // Positive steps > 1.
        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, 2)]).collect();
        assert_eq!(slice, &[1, 3, 5, 7, 9]);

        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, 3)]).collect();
        assert_eq!(slice, &[1, 4, 7, 10]);

        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, 10)]).collect();
        assert_eq!(slice, &[1]);

        // Negative steps.
        let slice: Vec<_> = x.slice_iter(&[sr(10, -11, -1)]).collect();
        assert_eq!(slice, &[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);

        let slice: Vec<_> = x.slice_iter(&[sr(8, 0, -1)]).collect();
        assert_eq!(slice, &[9, 8, 7, 6, 5, 4, 3, 2]);

        let slice: Vec<_> = x.slice_iter(&[sr(10, 0, -2)]).collect();
        assert_eq!(slice, &[10, 8, 6, 4, 2]);

        let slice: Vec<_> = x.slice_iter(&[sr(10, 0, -10)]).collect();
        assert_eq!(slice, &[10]);
    }

    #[test]
    fn test_slice_iter_negative_indices() {
        let sr = |start, end| SliceRange::new(start, end, 1);
        let x = steps(&[10]);

        // Negative start
        let slice: Vec<_> = x.slice_iter(&[sr(-2, 10)]).collect();
        assert_eq!(slice, &[9, 10]);

        // Negative end
        let slice: Vec<_> = x.slice_iter(&[sr(7, -1)]).collect();
        assert_eq!(slice, &[8, 9]);

        // Negative start and end
        let slice: Vec<_> = x.slice_iter(&[sr(-3, -1)]).collect();
        assert_eq!(slice, &[8, 9]);
    }

    #[test]
    fn test_slice_iter_clamps_indices() {
        let sr = SliceRange::new;
        let x = steps(&[5]);

        // Test cases for positive steps (ie. traversing forwards).

        // Positive start out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(10, 11, 1)]).collect();
        assert_eq!(slice.len(), 0);

        // Positive end out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, 1)]).collect();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        // Negative start out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(-10, 5, 1)]).collect();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        // Negative end out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(-10, -5, 1)]).collect();
        assert_eq!(slice.len(), 0);

        // Test cases for negative steps (ie. traversing backwards).

        // Positive start out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(10, -6, -1)]).collect();
        assert_eq!(slice, &[5, 4, 3, 2, 1]);

        // Positive end out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(0, 10, -1)]).collect();
        assert_eq!(slice.len(), 0);

        // Negative start out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(-10, 5, -1)]).collect();
        assert_eq!(slice.len(), 0);

        // Negative end out of bounds
        let slice: Vec<_> = x.slice_iter(&[sr(-1, -10, -1)]).collect();
        assert_eq!(slice, &[5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_slice_iter_start_end_step_combinations() {
        let sr = SliceRange::new;
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
        let range = &[SliceRange::new(1, 4, 1), SliceRange::new(1, 4, 1)];
        let expected: Vec<_> = x.slice_iter(range).collect();
        let result: Vec<_> = x
            .slice_offsets(range)
            .map(|offset| x.data()[offset])
            .collect();

        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_write() -> std::io::Result<()> {
        use std::io::{Cursor, Read};
        let x = from_data(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
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

        for el in x.iter() {
            cursor.read(&mut tmp)?;
            let written_el = f32::from_le_bytes(tmp);
            assert_eq!(written_el, el);
        }

        Ok(())
    }
}
