use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};

use crate::errors::SliceError;
use crate::iterators::{BroadcastIter, Iter, IterMut, Lanes, LanesMut, MutViewRef, ViewRef};
use crate::layout::{DynLayout, Layout, MatrixLayout, NdLayout, OverlapPolicy};
use crate::RandomSource;
use crate::{IntoSliceItems, SliceItem};

mod iterators;
use iterators::{InnerIter, InnerIterMut};

/// The base type for multi-dimensional arrays. This consists of storage for
/// elements, plus a _layout_ which maps from a multi-dimensional array index
/// to a storage offset. This base type is not normally used directly but
/// instead through a type alias which selects the storage type and layout.
///
/// The storage can be owned (like a `Vec<T>`), borrowed (like `&[T]`) or
/// mutably borrowed (like `&mut [T]`). The layout can have a dimension count
/// that is determined statically (ie. forms part of the tensor's type), see
/// [NdLayout] or is only known at runtime, see [DynLayout].
#[derive(Debug)]
pub struct TensorBase<T, S: AsRef<[T]>, L: MutLayout> {
    data: S,
    layout: L,
    element_type: PhantomData<T>,
}

/// Trait implemented by all variants of [TensorBase], which provides a
/// `view` method to get an immutable view of the tensor, plus methods which
/// forward to the view.
///
/// Tensor views ([TensorView], [NdTensorView]) have specialized versions of
/// these methods which preserve the underlying data lifetime. This allows for
/// chaining operations on views together.
///
/// In other words, this trait is conceptually similar to the way
/// [std::ops::Deref] in the Rust standard library allows a `Vec<T>` to have all
/// the methods of an `&[T]`.
pub trait View: Layout {
    /// Type of element stored in this tensor.
    type Elem;

    /// The underlying layout of this tensor. It must have the same index
    /// type (eg. `[usize; N]` or `&[usize]`) as this view.
    type Layout: for<'a> MutLayout<Index<'a> = Self::Index<'a>>;

    /// Return a borrowed view of this tensor.
    fn view(&self) -> TensorBase<Self::Elem, &[Self::Elem], Self::Layout>;

    /// Return the layout of this tensor.
    fn layout(&self) -> &Self::Layout;

    /// Return a view of this tensor with a dynamic rank.
    fn as_dyn(&self) -> TensorBase<Self::Elem, &[Self::Elem], DynLayout>
    where
        Self::Layout: Into<DynLayout>,
    {
        self.view().as_dyn()
    }

    /// Broadcast this view to another shape with a static dimension count.
    fn broadcast<const M: usize>(&self, shape: [usize; M]) -> NdTensorView<Self::Elem, M> {
        self.view().broadcast(shape)
    }

    /// Broadcast this view to another shape with a dynamic dimension count.
    fn broadcast_dyn(&self, shape: &[usize]) -> TensorView<Self::Elem> {
        self.view().broadcast_dyn(shape)
    }

    /// Return an iterator over elements of this tensor, broadcast to `shape`.
    ///
    /// This is equivalent to `self.broadcast(shape).iter()` but has some
    /// additional optimizations.
    fn broadcast_iter(&self, shape: &[usize]) -> BroadcastIter<Self::Elem> {
        self.view().broadcast_iter(shape)
    }

    /// Return the layout of this tensor as a slice, if it is contiguous.
    fn data(&self) -> Option<&[Self::Elem]>;

    /// Return a reference to the element at a given index, or `None` if the
    /// index is invalid.
    fn get<I: AsIndex<Self::Layout>>(&self, index: I) -> Option<&Self::Elem>;

    /// Return an iterator over the innermost N dimensions.
    fn inner_iter<const N: usize>(&self) -> InnerIter<Self::Elem, Self::Layout, N> {
        self.view().inner_iter()
    }

    /// Insert a size-1 axis at the given index.
    fn insert_axis(&mut self, index: usize)
    where
        Self::Layout: ResizeLayout;

    /// Return the scalar value in this tensor if it has 0 dimensions.
    fn item(&self) -> Option<&Self::Elem> {
        self.view().item()
    }

    /// Return an iterator over elements in this tensor in their logical order.
    fn iter(&self) -> Iter<Self::Elem>;

    /// Return an iterator over 1D slices of this tensor along a given axis.
    fn lanes(&self, dim: usize) -> Lanes<Self::Elem> {
        self.view().lanes(dim)
    }

    /// Return a new tensor with the same shape, formed by applying `f` to each
    /// element in this tensor.
    fn map<F, U>(&self, f: F) -> TensorBase<U, Vec<U>, Self::Layout>
    where
        F: Fn(&Self::Elem) -> U,
    {
        self.view().map(f)
    }

    /// Re-order the axes of this tensor to move the axis at index `from` to
    /// `to`.
    ///
    /// Panics if `from` or `to` is >= `self.ndim()`.
    fn move_axis(&mut self, from: usize, to: usize);

    /// Convert this tensor to one with the same shape but a static dimension
    /// count.
    ///
    /// Panics if `self.ndim() != N`.
    fn nd_view<const N: usize>(&self) -> TensorBase<Self::Elem, &[Self::Elem], NdLayout<N>> {
        self.view().nd_view()
    }

    /// Permute the dimensions of this tensor.
    fn permute(&mut self, order: Self::Index<'_>);

    /// Return a view with dimensions permuted in the order given by `dims`.
    fn permuted(
        &self,
        dims: Self::Index<'_>,
    ) -> TensorBase<Self::Elem, &[Self::Elem], Self::Layout> {
        self.view().permuted(dims)
    }

    fn reshaped<S: ToLayout>(&self, shape: S) -> TensorBase<Self::Elem, &[Self::Elem], S::Layout> {
        self.view().reshaped(shape)
    }

    /// Reverse the order of dimensions in this tensor.
    fn transpose(&mut self);

    /// Return a view with the order of dimensions reversed.
    fn transposed(&self) -> TensorBase<Self::Elem, &[Self::Elem], Self::Layout> {
        self.view().transposed()
    }

    /// Slice this tensor and return a dynamic-rank view.
    ///
    /// Fails if the range has more dimensions than the view or is out of bounds
    /// for any dimension.
    fn try_slice<R: IntoSliceItems>(&self, range: R) -> Result<TensorView<Self::Elem>, SliceError> {
        self.view().try_slice(range)
    }

    /// Slice this tensor and return a static-rank view with `M` dimensions.
    ///
    /// Use [View::slice_dyn] instead the number of dimensions in the returned
    /// view is unknown at compile time.
    ///
    /// Panics if the dimension count is not `M`.
    fn slice<const M: usize, R: IntoSliceItems>(&self, range: R) -> NdTensorView<Self::Elem, M> {
        self.view().slice(range)
    }

    /// Slice this tensor and return a dynamic-rank view.
    fn slice_dyn<R: IntoSliceItems>(&self, range: R) -> TensorView<Self::Elem> {
        self.view().slice_dyn(range)
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

    /// Return a vector containing the elements of this tensor in their logical
    /// order, ie. as if the tensor were flattened into one dimension.
    fn to_vec(&self) -> Vec<Self::Elem>
    where
        Self::Elem: Clone;

    /// Return a vector with the same shape but with strides in contiguous order.
    ///
    /// Certain operations require or are faster with contiguous tensors.
    fn to_contiguous(&self) -> TensorBase<Self::Elem, Cow<[Self::Elem]>, Self::Layout>
    where
        Self::Elem: Clone,
    {
        self.view().to_contiguous()
    }

    /// Return a copy of this tensor with a given shape.
    fn to_shape<S: ToLayout>(&self, shape: S) -> TensorBase<Self::Elem, Vec<Self::Elem>, S::Layout>
    where
        Self::Elem: Clone;

    /// Return clone of this tensor which uniquely owns its elements.
    fn to_tensor(&self) -> TensorBase<Self::Elem, Vec<Self::Elem>, Self::Layout>
    where
        Self::Elem: Clone,
    {
        let data = self.to_vec();
        TensorBase::from_data(self.layout().shape(), data)
    }
}

/// MutLayout extends [Layout] with methods for creating, modifying and
/// transforming layouts.
pub trait MutLayout: Layout + Clone {
    /// Create a new contiguous layout with a given shape.
    fn from_shape(shape: Self::Index<'_>) -> Self;

    /// Broadcast this layout to a new shape with `M` dimensions.
    fn broadcast<const M: usize>(&self, shape: [usize; M]) -> NdLayout<M>;

    /// Broadcast this layout to a new shape with a variable number of dimensions.
    fn broadcast_dyn(&self, shape: &[usize]) -> DynLayout;

    fn move_axis(&mut self, from: usize, to: usize);

    fn permuted(&self, order: Self::Index<'_>) -> Self;

    fn reshaped<S: ToLayout>(&self, shape: S) -> S::Layout {
        assert!(
            self.is_contiguous(),
            "tried to reshape non-contiguous layout"
        );
        shape.to_layout()
    }

    fn transposed(&self) -> Self;

    /// Slice the layout and return a static rank layout with `M` dimensions.
    fn slice<const M: usize>(&self, range: &[SliceItem]) -> (Range<usize>, NdLayout<M>);

    /// Slice the layout and return a dynamic rank layout.
    fn slice_dyn(&self, range: &[SliceItem]) -> (Range<usize>, DynLayout);

    fn squeezed(&self) -> DynLayout;

    fn try_slice<R: IntoSliceItems>(
        &self,
        range: R,
    ) -> Result<(Range<usize>, DynLayout), SliceError>;
}

impl<const N: usize> MutLayout for NdLayout<N> {
    fn from_shape(shape: [usize; N]) -> Self {
        Self::from_shape(shape)
    }

    fn broadcast<const M: usize>(&self, shape: [usize; M]) -> NdLayout<M> {
        self.broadcast(shape)
    }

    fn broadcast_dyn(&self, shape: &[usize]) -> DynLayout {
        self.as_dyn().broadcast(shape)
    }

    fn move_axis(&mut self, from: usize, to: usize) {
        assert!(from < N && to < N);
        let mut dyn_layout = self.as_dyn();
        dyn_layout.move_axis(from, to);
        *self = NdLayout::try_from(&dyn_layout).unwrap();
    }

    fn permuted(&self, order: [usize; N]) -> NdLayout<N> {
        self.permuted(order)
    }

    fn transposed(&self) -> NdLayout<N> {
        self.transposed()
    }

    fn slice<const M: usize>(&self, range: &[SliceItem]) -> (Range<usize>, NdLayout<M>) {
        self.slice(range)
    }

    fn slice_dyn(&self, range: &[SliceItem]) -> (Range<usize>, DynLayout) {
        self.as_dyn().slice(range)
    }

    fn squeezed(&self) -> DynLayout {
        self.as_dyn().squeezed()
    }

    fn try_slice<R: IntoSliceItems>(
        &self,
        range: R,
    ) -> Result<(Range<usize>, DynLayout), SliceError> {
        let items = range.into_slice_items();
        self.as_dyn().try_slice(items.as_ref())
    }
}

impl MutLayout for DynLayout {
    fn from_shape(shape: &[usize]) -> Self {
        Self::from_shape(shape)
    }

    fn broadcast<const M: usize>(&self, shape: [usize; M]) -> NdLayout<M> {
        NdLayout::try_from(&self.broadcast(&shape)).expect("broadcast failed")
    }

    fn broadcast_dyn(&self, shape: &[usize]) -> DynLayout {
        self.broadcast(shape)
    }

    fn move_axis(&mut self, from: usize, to: usize) {
        self.move_axis(from, to)
    }

    fn permuted(&self, order: &[usize]) -> DynLayout {
        self.permuted(order)
    }

    fn transposed(&self) -> DynLayout {
        self.transposed()
    }

    fn slice<const M: usize>(&self, range: &[SliceItem]) -> (Range<usize>, NdLayout<M>) {
        let (offset_range, dyn_layout) = self.slice(range);
        let nd_layout = NdLayout::try_from(&dyn_layout)
            .expect("sliced tensor did not have expected dimension count");
        (offset_range, nd_layout)
    }

    fn slice_dyn(&self, range: &[SliceItem]) -> (Range<usize>, DynLayout) {
        self.slice(range)
    }

    fn squeezed(&self) -> DynLayout {
        self.squeezed()
    }

    fn try_slice<R: IntoSliceItems>(
        &self,
        range: R,
    ) -> Result<(Range<usize>, DynLayout), SliceError> {
        let items = range.into_slice_items();
        self.try_slice(items.as_ref())
    }
}

/// Trait for shapes which can be used to create a contiguous layout.
///
/// This is implemented for `[usize; N]` for creating static-rank layouts from
/// arrays, and `&[usize]` for creating dynamic-rank layouts from slices.
pub trait ToLayout {
    /// The type of layout produced from this shape.
    type Layout: MutLayout;

    /// Convert this shape into a contiguous layout.
    fn to_layout(self) -> Self::Layout;
}

impl<const N: usize> ToLayout for [usize; N] {
    type Layout = NdLayout<N>;

    fn to_layout(self) -> NdLayout<N> {
        NdLayout::from_shape(self)
    }
}

impl<'a> ToLayout for &'a [usize] {
    type Layout = DynLayout;

    fn to_layout(self) -> DynLayout {
        DynLayout::from_shape(self)
    }
}

/// Trait which extends [MutLayout] with support for changing the number of
/// dimensions in-place.
pub trait ResizeLayout: MutLayout {
    fn insert_axis(&mut self, index: usize);
}

impl ResizeLayout for DynLayout {
    fn insert_axis(&mut self, index: usize) {
        self.insert_dim(index)
    }
}

/// Trait for converting types into indices for use with a given layout.
///
/// Static-rank tensors can be indexed with `[usize; N]` arrays. Dynamic-rank
/// tensors can be indexed with any type that can be converted to a `&[usize]`
/// slice.
pub trait AsIndex<L: Layout> {
    fn as_index(&self) -> L::Index<'_>;
}

impl<T: AsRef<[usize]>> AsIndex<DynLayout> for T {
    fn as_index(&self) -> &[usize] {
        self.as_ref()
    }
}

impl<const N: usize> AsIndex<NdLayout<N>> for [usize; N] {
    fn as_index(&self) -> [usize; N] {
        *self
    }
}

impl<T, S: AsRef<[T]>, L: MutLayout> TensorBase<T, S, L> {
    /// Constructor a new tensor from a given shape and storage.
    pub fn from_data(shape: L::Index<'_>, data: S) -> TensorBase<T, S, L>
    where
        L: MutLayout,
    {
        let layout = L::from_shape(shape);
        assert!(
            data.as_ref().len() == layout.len(),
            "layout and data have different lengths"
        );
        TensorBase {
            data,
            layout,
            element_type: PhantomData,
        }
    }

    pub fn into_dyn(self) -> TensorBase<T, S, DynLayout>
    where
        L: Into<DynLayout>,
    {
        TensorBase {
            data: self.data,
            layout: self.layout.into(),
            element_type: PhantomData,
        }
    }

    /// Attempt to convert this tensor's layout to a static-rank layout with `N`
    /// dimensions.
    fn nd_layout<const N: usize>(&self) -> Option<NdLayout<N>> {
        if self.ndim() != N {
            return None;
        }
        let shape: [usize; N] = std::array::from_fn(|i| self.size(i));
        let strides: [usize; N] = std::array::from_fn(|i| self.stride(i));
        let layout = NdLayout::try_from_shape_and_strides(
            shape,
            Some(strides),
            self.len(),
            OverlapPolicy::AllowOverlap,
        )
        .expect("invalid layout");
        Some(layout)
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, L: MutLayout> TensorBase<T, S, L> {
    /// Replace each element in this tensor with the result of applying `f` to
    /// the element.
    pub fn apply<F: Fn(&T) -> T>(&mut self, f: F) {
        if self.is_contiguous() {
            self.data.as_mut().iter_mut().for_each(|x| *x = f(x));
        } else {
            self.iter_mut().for_each(|x| *x = f(x));
        }
    }

    /// Return a mutable view of this tensor with a dynamic dimension count.
    pub fn as_dyn_mut(&mut self) -> TensorBase<T, &mut [T], DynLayout>
    where
        L: Clone + Into<DynLayout>,
    {
        TensorBase {
            layout: self.layout.clone().into(),
            data: self.data.as_mut(),
            element_type: PhantomData,
        }
    }

    /// Copy elements from another tensor into this tensor.
    ///
    /// This tensor and `other` must have the same shape.
    pub fn copy_from<S2: AsRef<[T]>>(&mut self, other: &TensorBase<T, S2, L>)
    where
        T: Clone,
        L: Clone,
    {
        assert!(self.shape() == other.shape());
        for (out, x) in self.iter_mut().zip(other.iter()) {
            *out = x.clone();
        }
    }

    /// Return the data in this tensor as a slice if it is contiguous.
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        self.layout.is_contiguous().then_some(self.data.as_mut())
    }

    /// Replace all elements of this tensor with `value`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.apply(|_| value.clone())
    }

    /// Return a mutable reference to the element at `index`, or `None` if the
    /// index is invalid.
    pub fn get_mut<I: AsIndex<L>>(&mut self, index: I) -> Option<&mut T> {
        self.try_offset(index.as_index())
            .map(|offset| &mut self.data.as_mut()[offset])
    }

    /// Return the element at a given index, without performing any bounds-
    /// checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is valid for the tensor's shape.
    pub unsafe fn get_unchecked_mut<I: AsIndex<L>>(&mut self, index: I) -> &mut T {
        self.data
            .as_mut()
            .get_unchecked_mut(self.layout.offset_unchecked(index.as_index()))
    }

    pub(crate) fn mut_view_ref(&mut self) -> MutViewRef<T, L> {
        MutViewRef::new(self.data.as_mut(), &self.layout)
    }

    /// Return a mutable iterator over the N innermost dimensions of this tensor.
    pub fn inner_iter_mut<const N: usize>(&mut self) -> InnerIterMut<T, L, N> {
        InnerIterMut::new(self.view_mut())
    }

    /// Return a mutable iterator over the elements of this tensor, in their
    /// logical order.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self.mut_view_ref())
    }

    /// Return an iterator over mutable 1D slices of this tensor along a given
    /// dimension.
    pub fn lanes_mut(&mut self, dim: usize) -> LanesMut<T> {
        LanesMut::new(self.mut_view_ref(), dim)
    }

    /// Return a view of this tensor with a static dimension count.
    ///
    /// Panics if `self.ndim() != N`.
    pub fn nd_view_mut<const N: usize>(&mut self) -> TensorBase<T, &mut [T], NdLayout<N>> {
        assert!(self.ndim() == N, "ndim {} != {}", self.ndim(), N);
        TensorBase {
            layout: self.nd_layout().unwrap(),
            data: self.data.as_mut(),
            element_type: PhantomData,
        }
    }

    pub fn permuted_mut(&mut self, order: L::Index<'_>) -> TensorBase<T, &mut [T], L> {
        TensorBase {
            layout: self.layout.permuted(order),
            data: self.data.as_mut(),
            element_type: PhantomData,
        }
    }

    pub fn reshaped_mut<SH: ToLayout>(&mut self, shape: SH) -> TensorBase<T, &mut [T], SH::Layout> {
        TensorBase {
            layout: self.layout.reshaped(shape),
            data: self.data.as_mut(),
            element_type: PhantomData,
        }
    }

    /// Slice this tensor and return a static-rank view with `M` dimensions.
    ///
    /// Use [View::slice_dyn] instead the number of dimensions in the returned
    /// view is unknown at compile time.
    ///
    /// Panics if the dimension count is not `M`.
    pub fn slice_mut<const M: usize, R: IntoSliceItems>(
        &mut self,
        range: R,
    ) -> NdTensorViewMut<T, M> {
        let range = range.into_slice_items();
        let (offset_range, sliced_layout) = self.layout.slice(range.as_ref());
        NdTensorViewMut {
            data: &mut self.data.as_mut()[offset_range],
            layout: sliced_layout,
            element_type: PhantomData,
        }
    }

    /// Slice this tensor and return a dynamic-rank view.
    pub fn slice_mut_dyn<R: IntoSliceItems>(&mut self, range: R) -> TensorViewMut<T> {
        let range = range.into_slice_items();
        let (offset_range, sliced_layout) = self.layout.slice_dyn(range.as_ref());
        TensorViewMut {
            data: &mut self.data.as_mut()[offset_range],
            layout: sliced_layout,
            element_type: PhantomData,
        }
    }

    /// Slice this tensor and return a dynamic-rank view.
    ///
    /// Fails if the range has more dimensions than the view or is out of bounds
    /// for any dimension.
    pub fn try_slice_mut<R: IntoSliceItems>(
        &mut self,
        range: R,
    ) -> Result<TensorViewMut<T>, SliceError> {
        let (offset_range, layout) = self.layout.try_slice(range)?;
        Ok(TensorBase {
            data: &mut self.data.as_mut()[offset_range],
            layout,
            element_type: PhantomData,
        })
    }

    /// Return a mutable view of this tensor.
    pub fn view_mut(&mut self) -> TensorBase<T, &mut [T], L>
    where
        L: Clone,
    {
        TensorBase {
            data: self.data.as_mut(),
            layout: self.layout.clone(),
            element_type: PhantomData,
        }
    }
}

impl<T, L: Clone + MutLayout> TensorBase<T, Vec<T>, L> {
    /// Create a new 1D tensor filled with an arithmetic sequence of values
    /// in the range `[start, end)` separated by `step`. If `step` is omitted,
    /// it defaults to 1.
    pub fn arange(start: T, end: T, step: Option<T>) -> TensorBase<T, Vec<T>, L>
    where
        T: Copy + PartialOrd + From<bool> + std::ops::Add<Output = T>,
        [usize; 1]: AsIndex<L>,
    {
        let step = step.unwrap_or((true).into());
        let mut data = Vec::new();
        let mut curr = start;
        while curr < end {
            data.push(curr);
            curr = curr + step;
        }
        TensorBase::from_data([data.len()].as_index(), data)
    }

    /// Consume self and return the underlying data as a contiguous tensor.
    ///
    /// See also [TensorBase::to_vec].
    pub fn into_data(self) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_contiguous() {
            self.data
        } else {
            self.to_vec()
        }
    }

    /// Consume self and return a new contiguous tensor with the given shape.
    ///
    /// This avoids copying the data if it is already contiguous.
    pub fn into_shape<S: ToLayout>(self, shape: S) -> TensorBase<T, Vec<T>, S::Layout>
    where
        T: Clone,
    {
        TensorBase {
            data: self.into_data(),
            layout: shape.to_layout(),
            element_type: PhantomData,
        }
    }

    /// Create a new 0D tensor from a scalar value.
    pub fn from_scalar(value: T) -> TensorBase<T, Vec<T>, L>
    where
        [usize; 0]: AsIndex<L>,
    {
        TensorBase::from_data([].as_index(), vec![value])
    }

    /// Create a new tensor with a given shape and all elements set to `value`.
    pub fn full(shape: L::Index<'_>, value: T) -> TensorBase<T, Vec<T>, L>
    where
        T: Clone,
    {
        let n_elts = shape.as_ref().iter().product();
        let data = vec![value; n_elts];
        TensorBase::from_data(shape, data)
    }

    /// Make the underlying data in this tensor contiguous.
    pub fn make_contiguous(&mut self)
    where
        T: Clone,
    {
        if self.is_contiguous() {
            return;
        }
        self.data = self.to_vec();
        self.layout = L::from_shape(self.layout.shape());
    }

    /// Create a new tensor with a given shape and elements populated using
    /// numbers generated by `rand_src`.
    pub fn rand<R: RandomSource<T>>(
        shape: L::Index<'_>,
        rand_src: &mut R,
    ) -> TensorBase<T, Vec<T>, L> {
        let data: Vec<_> = std::iter::from_fn(|| Some(rand_src.next()))
            .take(shape.as_ref().iter().product())
            .collect();
        TensorBase::from_data(shape, data)
    }

    /// Create a new tensor with a given shape, with all elements set to their
    /// default value (ie. zero for numeric types).
    pub fn zeros(shape: L::Index<'_>) -> TensorBase<T, Vec<T>, L>
    where
        T: Clone + Default,
    {
        Self::full(shape, T::default())
    }
}

impl<'a, T, L: Clone + MutLayout> TensorBase<T, &'a [T], L> {
    /// Return a view of this tensor with a dynamic dimension count.
    pub fn as_dyn(&self) -> TensorBase<T, &'a [T], DynLayout>
    where
        L: Clone + Into<DynLayout>,
    {
        TensorBase {
            data: self.data,
            layout: self.layout.clone().into(),
            element_type: PhantomData,
        }
    }

    /// Broadcast this view to another shape with a static dimension count.
    pub fn broadcast<const M: usize>(&self, shape: [usize; M]) -> NdTensorView<'a, T, M> {
        NdTensorView {
            layout: self.layout.broadcast(shape),
            data: self.data,
            element_type: PhantomData,
        }
    }

    /// Broadcast this view to another shape with a dynamic dimension count.
    pub fn broadcast_dyn(&self, shape: &[usize]) -> TensorView<'a, T> {
        TensorView {
            layout: self.layout.broadcast_dyn(shape),
            data: self.data,
            element_type: PhantomData,
        }
    }

    pub fn broadcast_iter(&self, shape: &[usize]) -> BroadcastIter<'a, T> {
        BroadcastIter::new(self.view_ref(), shape.as_ref())
    }

    /// Return the data in this tensor as a slice if it is contiguous.
    pub fn data(&self) -> Option<&'a [T]> {
        self.layout.is_contiguous().then_some(self.data)
    }

    /// Return the element at a given index, without performing any bounds-
    /// checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is valid for the tensor's shape.
    pub unsafe fn get_unchecked<I: AsIndex<L>>(&self, index: I) -> &'a T {
        self.data
            .get_unchecked(self.layout.offset_unchecked(index.as_index()))
    }

    pub fn inner_iter<const N: usize>(&self) -> InnerIter<'a, T, L, N> {
        InnerIter::new(self.view())
    }

    /// Return the scalar value in this tensor if it has 0 dimensions.
    pub fn item(&self) -> Option<&'a T> {
        match self.ndim() {
            0 => Some(&self.data[0]),
            _ if self.len() == 1 => self.iter().next(),
            _ => None,
        }
    }

    /// Return an iterator over elements of this tensor in their logical order.
    pub fn iter(&self) -> Iter<'a, T> {
        Iter::new(self.view_ref())
    }

    /// Return an iterator over 1D slices of this tensor along a given dimension.
    pub fn lanes(&self, dim: usize) -> Lanes<'a, T> {
        Lanes::new(self.view_ref(), dim)
    }

    /// Return a view of this tensor with a static dimension count.
    ///
    /// Panics if `self.ndim() != N`.
    pub fn nd_view<const N: usize>(&self) -> TensorBase<T, &'a [T], NdLayout<N>> {
        assert!(self.ndim() == N, "ndim {} != {}", self.ndim(), N);
        TensorBase {
            data: self.data,
            layout: self.nd_layout().unwrap(),
            element_type: PhantomData,
        }
    }

    pub fn permuted(&self, order: L::Index<'_>) -> TensorBase<T, &'a [T], L> {
        TensorBase {
            data: self.data,
            layout: self.layout.permuted(order),
            element_type: PhantomData,
        }
    }

    pub fn reshaped<S: ToLayout>(&self, shape: S) -> TensorBase<T, &'a [T], S::Layout> {
        TensorBase {
            data: self.data,
            layout: self.layout.reshaped(shape),
            element_type: PhantomData,
        }
    }

    pub fn slice<const M: usize, R: IntoSliceItems>(&self, range: R) -> NdTensorView<'a, T, M> {
        let range = range.into_slice_items();
        let (offset_range, sliced_layout) = self.layout.slice(range.as_ref());
        NdTensorView {
            data: &self.data[offset_range],
            layout: sliced_layout,
            element_type: PhantomData,
        }
    }

    pub fn slice_dyn<R: IntoSliceItems>(&self, range: R) -> TensorView<'a, T> {
        let range = range.into_slice_items();
        let (offset_range, sliced_layout) = self.layout.slice_dyn(range.as_ref());
        TensorView {
            data: &self.data[offset_range],
            layout: sliced_layout,
            element_type: PhantomData,
        }
    }

    pub fn slice_iter(&self, range: &[SliceItem]) -> Iter<'a, T> {
        Iter::slice(self.view_ref(), range)
    }

    pub fn squeezed(&self) -> TensorView<'a, T> {
        TensorBase {
            data: self.data,
            layout: self.layout.squeezed(),
            element_type: PhantomData,
        }
    }

    /// Return a view of this tensor with elements stored in contiguous order.
    ///
    /// If the data is already contiguous, no copy is made, otherwise the
    /// elements are copied into a new buffer in contiguous order.
    pub fn to_contiguous(&self) -> TensorBase<T, Cow<'a, [T]>, L>
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
                layout: L::from_shape(self.layout.shape()),
                element_type: PhantomData,
            }
        }
    }

    pub fn transposed(&self) -> TensorBase<T, &'a [T], L> {
        TensorBase {
            data: self.data,
            layout: self.layout.transposed(),
            element_type: PhantomData,
        }
    }

    pub fn try_slice<R: IntoSliceItems>(&self, range: R) -> Result<TensorView<'a, T>, SliceError> {
        let (offset_range, layout) = self.layout.try_slice(range)?;
        Ok(TensorBase {
            data: &self.data[offset_range],
            layout,
            element_type: PhantomData,
        })
    }

    pub fn view(&self) -> TensorBase<T, &'a [T], L> {
        TensorBase {
            data: self.data,
            layout: self.layout.clone(),
            element_type: PhantomData,
        }
    }

    pub(crate) fn view_ref(&self) -> ViewRef<'a, '_, T, L> {
        ViewRef::new(self.data, &self.layout)
    }
}

impl<T, S: AsRef<[T]>, L: MutLayout> Layout for TensorBase<T, S, L> {
    type Index<'a> = L::Index<'a>;
    type Indices = L::Indices;

    fn ndim(&self) -> usize {
        self.layout.ndim()
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

    fn try_offset(&self, index: Self::Index<'_>) -> Option<usize> {
        self.layout.try_offset(index)
    }
}

impl<T, S: AsRef<[T]>, L: MutLayout + MatrixLayout> MatrixLayout for TensorBase<T, S, L> {
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

impl<T, S: AsRef<[T]>, L: MutLayout + Clone> View for TensorBase<T, S, L> {
    type Elem = T;
    type Layout = L;

    fn iter(&self) -> Iter<T> {
        self.view().iter()
    }

    fn data(&self) -> Option<&[Self::Elem]> {
        self.view().data()
    }

    fn insert_axis(&mut self, index: usize)
    where
        L: ResizeLayout,
    {
        self.layout.insert_axis(index)
    }

    fn layout(&self) -> &L {
        &self.layout
    }

    fn map<F, U>(&self, f: F) -> TensorBase<U, Vec<U>, L>
    where
        F: Fn(&Self::Elem) -> U,
    {
        let data: Vec<_> = self.iter().map(f).collect();
        TensorBase::from_data(self.shape(), data)
    }

    fn move_axis(&mut self, from: usize, to: usize) {
        self.layout.move_axis(from, to);
    }

    fn view(&self) -> TensorBase<T, &[T], L> {
        TensorBase {
            data: self.data.as_ref(),
            layout: self.layout.clone(),
            element_type: PhantomData,
        }
    }

    fn get<I: AsIndex<L>>(&self, index: I) -> Option<&Self::Elem> {
        self.try_offset(index.as_index())
            .map(|offset| &self.data.as_ref()[offset])
    }

    fn permute(&mut self, order: Self::Index<'_>) {
        self.layout = self.layout.permuted(order);
    }

    fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        if let Some(data) = self.data() {
            data.to_vec()
        } else {
            // TODO - Add fast path for low rank that doesn't use iterators.
            self.view().iter().cloned().collect()
        }
    }

    fn to_shape<SH: ToLayout>(
        &self,
        shape: SH,
    ) -> TensorBase<Self::Elem, Vec<Self::Elem>, SH::Layout>
    where
        T: Clone,
    {
        TensorBase {
            data: self.to_vec(),
            layout: shape.to_layout(),
            element_type: PhantomData,
        }
    }

    fn transpose(&mut self) {
        self.layout = self.layout.transposed();
    }
}

impl<T, S: AsRef<[T]>, const N: usize> TensorBase<T, S, NdLayout<N>> {
    /// Load an array of `M` elements from successive entries of a tensor along
    /// the `dim` axis.
    ///
    /// eg. If `base` is `[0, 1, 2]`, dim=0 and `M` = 4 this will return an
    /// array with values from indices `[0, 1, 2]`, `[1, 1, 2]` ... `[3, 1, 2]`.
    ///
    /// Panics if any of the array indices are out of bounds.
    #[inline]
    pub fn get_array<const M: usize>(&self, base: [usize; N], dim: usize) -> [T; M]
    where
        T: Copy + Default,
    {
        let offsets: [usize; M] = array_offsets(&self.layout, base, dim);
        let data = self.data.as_ref();
        let mut result = [T::default(); M];
        for i in 0..M {
            // Safety: `array_offsets` returns valid offsets
            result[i] = unsafe { *data.get_unchecked(offsets[i]) };
        }
        result
    }
}

/// Return the offsets of `M` successive elements along the `dim` axis, starting
/// at index `base`.
///
/// Panics if any of the M element indices are out of bounds.
fn array_offsets<const N: usize, const M: usize>(
    layout: &NdLayout<N>,
    base: [usize; N],
    dim: usize,
) -> [usize; M] {
    assert!(
        base[dim] < usize::MAX - M && layout.size(dim) >= base[dim] + M,
        "array indices invalid"
    );

    let offset = layout.offset(base);
    let stride = layout.stride(dim);
    let mut offsets = [0; M];
    for i in 0..M {
        offsets[i] = offset + i * stride;
    }
    offsets
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, const N: usize> TensorBase<T, S, NdLayout<N>> {
    /// Store an array of `M` elements into successive entries of a tensor along
    /// the `dim` axis.
    ///
    /// See [NdTensorBase::get_array] for more details.
    #[inline]
    pub fn set_array<const M: usize>(&mut self, base: [usize; N], dim: usize, values: [T; M])
    where
        T: Copy,
    {
        let offsets: [usize; M] = array_offsets(&self.layout, base, dim);
        let data = self.data.as_mut();

        for i in 0..M {
            // Safety: `array_offsets` returns valid offsets.
            unsafe { *data.get_unchecked_mut(offsets[i]) = values[i] };
        }
    }
}

impl<T, S: AsRef<[T]>> TensorBase<T, S, NdLayout<1>> {
    /// Convert this vector to a static array of length `M`.
    ///
    /// Panics if the length of this vector is not M.
    #[inline]
    pub fn to_array<const M: usize>(&self) -> [T; M]
    where
        T: Copy + Default,
    {
        self.get_array([0], 0)
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>> TensorBase<T, S, NdLayout<1>> {
    /// Fill this vector with values from a static array of length `M`.
    ///
    /// Panics if the length of this vector is not M.
    #[inline]
    pub fn assign_array<const M: usize>(&mut self, values: [T; M])
    where
        T: Copy + Default,
    {
        self.set_array([0], 0, values)
    }
}

/// View of a slice of a tensor with a static dimension count.
pub type NdTensorView<'a, T, const N: usize> = TensorBase<T, &'a [T], NdLayout<N>>;

/// Tensor with a static dimension count.
pub type NdTensor<T, const N: usize> = TensorBase<T, Vec<T>, NdLayout<N>>;

/// Mutable view of a slice of a tensor with a static dimension count.
pub type NdTensorViewMut<'a, T, const N: usize> = TensorBase<T, &'a mut [T], NdLayout<N>>;

/// Tensor with a dynamic dimension count.
pub type Tensor<T = f32> = TensorBase<T, Vec<T>, DynLayout>;

/// View of a slice of a tensor with a dynamic dimension count.
pub type TensorView<'a, T = f32> = TensorBase<T, &'a [T], DynLayout>;

/// Mutable view of a slice of a tensor with a dynamic dimension count.
pub type TensorViewMut<'a, T = f32> = TensorBase<T, &'a mut [T], DynLayout>;

impl<T, S: AsRef<[T]>, L: MutLayout, I: AsIndex<L>> Index<I> for TensorBase<T, S, L> {
    type Output = T;

    /// Return the element at a given index.
    ///
    /// Panics if the index is out of bounds along any dimension.
    fn index(&self, index: I) -> &Self::Output {
        let offset = self.layout.offset(index.as_index());
        &self.data.as_ref()[offset]
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, L: MutLayout, I: AsIndex<L>> IndexMut<I>
    for TensorBase<T, S, L>
{
    /// Return the element at a given index.
    ///
    /// Panics if the index is out of bounds along any dimension.
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let offset = self.layout.offset(index.as_index());
        &mut self.data.as_mut()[offset]
    }
}

impl<T, S: AsRef<[T]> + Clone, L: MutLayout + Clone> Clone for TensorBase<T, S, L> {
    fn clone(&self) -> TensorBase<T, S, L> {
        let data = self.data.clone();
        TensorBase {
            data,
            layout: self.layout.clone(),
            element_type: PhantomData,
        }
    }
}

impl<T: PartialEq, S: AsRef<[T]>, L: MutLayout, V: View<Elem = T>> PartialEq<V>
    for TensorBase<T, S, L>
{
    fn eq(&self, other: &V) -> bool {
        self.shape().as_ref() == other.shape().as_ref() && self.iter().eq(other.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::{NdTensor, NdTensorView, Tensor, View};
    use crate::layout::MatrixLayout;
    use crate::prelude::*;
    use crate::rng::XorShiftRng;
    use crate::SliceItem;

    #[test]
    fn test_apply() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor = NdTensor::from_data([2, 2], data);
        tensor.apply(|x| *x * 2.);
        assert_eq!(tensor.to_vec(), &[2., 4., 6., 8.]);
    }

    #[test]
    fn test_arange() {
        let x = Tensor::arange(2, 6, None);
        let y = NdTensor::arange(2, 6, None);
        assert_eq!(x.data(), Some([2, 3, 4, 5].as_slice()));
        assert_eq!(y.data(), Some([2, 3, 4, 5].as_slice()));
    }

    #[test]
    fn test_as_dyn() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);
        let dyn_view = tensor.as_dyn();
        assert_eq!(dyn_view.shape(), tensor.shape().as_ref());
        assert_eq!(dyn_view.to_vec(), tensor.to_vec());
    }

    #[test]
    fn test_as_dyn_mut() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor = NdTensor::from_data([2, 2], data);
        let mut dyn_view = tensor.as_dyn_mut();

        dyn_view[[0, 0]] = 9.;

        assert_eq!(tensor[[0, 0]], 9.);
    }

    #[test]
    fn test_assign_array() {
        let mut tensor = NdTensor::zeros([2, 2]);
        let mut transposed = tensor.view_mut();

        transposed.permute([1, 0]);
        transposed.slice_mut(0).assign_array([1, 2]);
        transposed.slice_mut(1).assign_array([3, 4]);

        assert_eq!(tensor.iter().copied().collect::<Vec<_>>(), [1, 3, 2, 4]);
    }

    #[test]
    fn test_broadcast() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);
        let view = tensor.broadcast([1, 1, 2, 2]);
        assert_eq!(view.shape(), [1, 1, 2, 2]);
    }

    #[test]
    fn test_broadcast_dyn() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);
        let view = tensor.broadcast_dyn(&[1, 1, 2, 2]);
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_broadcast_iter() {
        let tensor = NdTensor::from_data([1], vec![3]);
        let elems: Vec<_> = tensor.broadcast_iter(&[2, 2]).copied().collect();
        assert_eq!(elems, &[3, 3, 3, 3]);
    }

    #[test]
    fn test_clone() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);
        let cloned = tensor.clone();
        assert_eq!(tensor.shape(), cloned.shape());
        assert_eq!(tensor.to_vec(), cloned.to_vec());
    }

    #[test]
    fn test_copy_from() {
        let mut dest = Tensor::zeros(&[2, 2]);
        let src = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        dest.copy_from(&src);
        assert_eq!(dest.to_vec(), &[1., 2., 3., 4.]);
    }

    #[test]
    fn test_data() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let tensor = NdTensorView::from_data([2, 3], &data);
        assert_eq!(tensor.data(), Some(data.as_slice()));

        let permuted = tensor.permuted([1, 0]);
        assert_eq!(permuted.shape(), [3, 2]);
        assert_eq!(permuted.data(), None);
    }

    #[test]
    fn test_data_mut() {
        let mut data = vec![1., 2., 3., 4., 5., 6.];
        let mut tensor = NdTensor::from_data([2, 3], data.clone());
        assert_eq!(tensor.data_mut(), Some(data.as_mut_slice()));

        let mut permuted = tensor.permuted_mut([1, 0]);
        assert_eq!(permuted.shape(), [3, 2]);
        assert_eq!(permuted.data_mut(), None);
    }

    #[test]
    fn test_fill() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor = NdTensor::from_data([2, 2], data);
        tensor.fill(9.);
        assert_eq!(tensor.to_vec(), &[9., 9., 9., 9.]);
    }

    #[test]
    fn test_from_scalar() {
        let x = Tensor::from_scalar(5.);
        let y = NdTensor::from_scalar(6.);
        assert_eq!(x.item(), Some(&5.));
        assert_eq!(y.item(), Some(&6.));
    }

    #[test]
    fn test_full() {
        let tensor = NdTensor::full([2, 2], 2.);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(tensor.data(), Some([2., 2., 2., 2.].as_slice()));
    }

    #[test]
    fn test_get() {
        // NdLayout
        let data = vec![1., 2., 3., 4.];
        let tensor: NdTensor<f32, 2> = NdTensor::from_data([2, 2], data);
        assert_eq!(tensor.get([1, 1]), Some(&4.));
        assert_eq!(tensor.get([2, 1]), None);

        // DynLayout
        let data = vec![1., 2., 3., 4.];
        let tensor: Tensor<f32> = Tensor::from_data(&[2, 2], data);
        assert_eq!(tensor.get([1, 1]), Some(&4.));
        assert_eq!(tensor.get([2, 1]), None); // Invalid index
        assert_eq!(tensor.get([1, 2, 3]), None); // Incorrect dim count
    }

    #[test]
    fn test_get_array() {
        let tensor = NdTensor::arange(1, 17, None).into_shape([4, 2, 2]);

        // First dim, zero base.
        let values: [i32; 4] = tensor.get_array([0, 0, 0], 0);
        assert_eq!(values, [1, 5, 9, 13]);

        // First dim, different base.
        let values: [i32; 4] = tensor.get_array([0, 1, 1], 0);
        assert_eq!(values, [4, 8, 12, 16]);

        // Last dim, zero base.
        let values: [i32; 2] = tensor.get_array([0, 0, 0], 2);
        assert_eq!(values, [1, 2]);
    }

    #[test]
    fn test_get_mut() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor: NdTensor<f32, 2> = NdTensor::from_data([2, 2], data);
        if let Some(elem) = tensor.get_mut([1, 1]) {
            *elem = 9.;
        }
        assert_eq!(tensor[[1, 1]], 9.);
        assert_eq!(tensor.get_mut([2, 1]), None);
    }

    #[test]
    fn test_get_unchecked() {
        let ndtensor = NdTensor::arange(1, 5, None);
        for i in 0..ndtensor.size(0) {
            assert_eq!(
                unsafe { ndtensor.view().get_unchecked([i]) },
                &ndtensor[[i]]
            );
        }

        let tensor = Tensor::arange(1, 5, None);
        for i in 0..tensor.size(0) {
            assert_eq!(unsafe { tensor.view().get_unchecked([i]) }, &ndtensor[[i]]);
        }
    }

    #[test]
    fn test_get_unchecked_mut() {
        let mut ndtensor = NdTensor::arange(1, 5, None);
        for i in 0..ndtensor.size(0) {
            unsafe { *ndtensor.get_unchecked_mut([i]) += 1 }
        }
        assert_eq!(ndtensor.to_vec(), &[2, 3, 4, 5]);

        let mut tensor = Tensor::arange(1, 5, None);
        for i in 0..tensor.size(0) {
            unsafe { *tensor.get_unchecked_mut([i]) += 1 }
        }
        assert_eq!(tensor.to_vec(), &[2, 3, 4, 5]);
    }

    #[test]
    fn test_index_and_index_mut() {
        // NdLayout
        let data = vec![1., 2., 3., 4.];
        let mut tensor: NdTensor<f32, 2> = NdTensor::from_data([2, 2], data);
        assert_eq!(tensor[[1, 1]], 4.);
        tensor[[1, 1]] = 9.;
        assert_eq!(tensor[[1, 1]], 9.);

        // DynLayout
        let data = vec![1., 2., 3., 4.];
        let mut tensor: Tensor<f32> = Tensor::from_data(&[2, 2], data);
        assert_eq!(tensor[[1, 1]], 4.);
        tensor[&[1, 1]] = 9.;
        assert_eq!(tensor[[1, 1]], 9.);
    }

    #[test]
    fn test_into_data() {
        let tensor = NdTensor::from_data([2], vec![2., 3.]);
        assert_eq!(tensor.into_data(), vec![2., 3.]);
    }

    #[test]
    fn test_into_dyn() {
        let tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
        let dyn_tensor = tensor.into_dyn();
        assert_eq!(dyn_tensor.shape(), &[2, 2]);
        assert_eq!(dyn_tensor.data(), Some([1., 2., 3., 4.].as_slice()));
    }

    #[test]
    fn test_into_shape() {
        // Contiguous tensor.
        let tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
        let reshaped = tensor.into_shape([4]);
        assert_eq!(reshaped.shape(), [4]);
        assert_eq!(reshaped.data(), Some([1., 2., 3., 4.].as_slice()));

        // Non-contiguous tensor.
        let mut tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
        tensor.transpose();
        let reshaped = tensor.into_shape([4]);
        assert_eq!(reshaped.shape(), [4]);
        assert_eq!(reshaped.data(), Some([1., 3., 2., 4.].as_slice()));
    }

    #[test]
    fn test_inner_iter() {
        let tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let mut rows = tensor.inner_iter::<1>();

        let row = rows.next().unwrap();
        assert_eq!(row.shape(), [2]);
        assert_eq!(row.to_vec(), &[1, 2]);

        let row = rows.next().unwrap();
        assert_eq!(row.shape(), [2]);
        assert_eq!(row.to_vec(), &[3, 4]);

        assert_eq!(rows.next(), None);
    }

    #[test]
    fn test_inner_iter_mut() {
        let mut tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let mut rows = tensor.inner_iter_mut::<1>();

        let mut row = rows.next().unwrap();
        assert_eq!(row.shape(), [2]);
        row.apply(|x| x * 2);

        let mut row = rows.next().unwrap();
        assert_eq!(row.shape(), [2]);
        row.apply(|x| x * 2);

        assert_eq!(rows.next(), None);

        assert_eq!(tensor.to_vec(), &[2, 4, 6, 8]);
    }

    #[test]
    fn test_insert_axis() {
        let mut tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        tensor.insert_axis(0);
        assert_eq!(tensor.shape(), &[1, 2, 2]);
        tensor.insert_axis(3);
        assert_eq!(tensor.shape(), &[1, 2, 2, 1]);
    }

    #[test]
    fn test_item() {
        let tensor = NdTensor::from_data([], vec![5.]);
        assert_eq!(tensor.item(), Some(&5.));
        let tensor = NdTensor::from_data([2], vec![2., 3.]);
        assert_eq!(tensor.item(), None);

        let tensor = Tensor::from_data(&[], vec![5.]);
        assert_eq!(tensor.item(), Some(&5.));
        let tensor = Tensor::from_data(&[2], vec![2., 3.]);
        assert_eq!(tensor.item(), None);
    }

    #[test]
    fn test_iter() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);
        assert_eq!(
            tensor.iter().copied().collect::<Vec<_>>(),
            &[1., 2., 3., 4.]
        );

        let data = vec![1., 2., 3., 4.];
        let tensor = Tensor::from_data(&[2, 2], data);
        assert_eq!(
            tensor.iter().copied().collect::<Vec<_>>(),
            &[1., 2., 3., 4.]
        );
    }

    #[test]
    fn test_iter_mut() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor = NdTensor::from_data([2, 2], data);
        tensor.iter_mut().for_each(|x| *x *= 2.);
        assert_eq!(tensor.to_vec(), &[2., 4., 6., 8.]);
    }

    #[test]
    fn test_lanes() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);
        let mut lanes = tensor.lanes(1);
        assert_eq!(
            lanes.next().unwrap().copied().collect::<Vec<_>>(),
            &[1., 2.]
        );
        assert_eq!(
            lanes.next().unwrap().copied().collect::<Vec<_>>(),
            &[3., 4.]
        );
    }

    #[test]
    fn test_lanes_mut() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor = NdTensor::from_data([2, 2], data);
        let mut lanes = tensor.lanes_mut(1);
        assert_eq!(lanes.next().unwrap().collect::<Vec<_>>(), &[&1., &2.]);
        assert_eq!(lanes.next().unwrap().collect::<Vec<_>>(), &[&3., &4.]);
    }

    #[test]
    fn test_make_contiguous() {
        let mut tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
        assert!(tensor.is_contiguous());

        // No-op, since tensor is already contiguous.
        tensor.make_contiguous();
        assert!(tensor.is_contiguous());

        // On a non-contiguous tensor, the data should be shuffled.
        tensor.transpose();
        assert!(!tensor.is_contiguous());
        tensor.make_contiguous();
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.data(), Some([1., 3., 2., 4.].as_slice()));
    }

    #[test]
    fn test_map() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);
        let doubled = tensor.map(|x| x * 2.);
        assert_eq!(doubled.to_vec(), &[2., 4., 6., 8.]);
    }

    #[test]
    fn test_matrix_layout() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let tensor = NdTensorView::from_data([2, 3], &data);
        assert_eq!(tensor.rows(), 2);
        assert_eq!(tensor.row_stride(), 3);
        assert_eq!(tensor.cols(), 3);
        assert_eq!(tensor.col_stride(), 1);
    }

    #[test]
    fn test_move_axis() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let mut tensor = NdTensorView::from_data([2, 3], &data);

        tensor.move_axis(1, 0);
        assert_eq!(tensor.shape(), [3, 2]);
        assert_eq!(tensor.to_vec(), &[1., 4., 2., 5., 3., 6.]);

        tensor.move_axis(0, 1);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(tensor.to_vec(), &[1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn test_nd_view() {
        let tensor: Tensor<f32> = Tensor::zeros(&[1, 4, 5]);

        // Dynamic -> static rank conversion.
        let nd_view = tensor.nd_view::<3>();
        assert_eq!(nd_view.shape(), [1, 4, 5]);
        assert_eq!(nd_view.strides().as_ref(), tensor.strides());

        // Static -> static rank conversion. Pointless, but it should compile.
        let nd_view_2 = nd_view.nd_view::<3>();
        assert_eq!(nd_view_2.shape(), nd_view.shape());
    }

    #[test]
    fn test_nd_view_mut() {
        let mut tensor: Tensor<f32> = Tensor::zeros(&[1, 4, 5]);
        let mut nd_view = tensor.nd_view_mut::<3>();
        assert_eq!(nd_view.shape(), [1, 4, 5]);

        nd_view[[0, 0, 0]] = 9.;

        assert_eq!(tensor[[0, 0, 0]], 9.);
    }

    #[test]
    fn test_rand() {
        let mut rng = XorShiftRng::new(1234);
        let tensor = NdTensor::rand([2, 2], &mut rng);
        assert_eq!(tensor.shape(), [2, 2]);
        for &x in tensor.iter() {
            assert!(x >= 0. && x <= 1.);
        }
    }

    #[test]
    fn test_permute() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let mut tensor = NdTensorView::from_data([2, 3], &data);

        tensor.permute([1, 0]);

        assert_eq!(tensor.shape(), [3, 2]);
        assert_eq!(tensor.to_vec(), &[1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn test_permuted() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let tensor = NdTensorView::from_data([2, 3], &data);

        let permuted = tensor.permuted([1, 0]);

        assert_eq!(permuted.shape(), [3, 2]);
        assert_eq!(permuted.to_vec(), &[1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn test_permuted_mut() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let mut tensor = NdTensor::from_data([2, 3], data);

        let mut permuted = tensor.permuted_mut([1, 0]);
        permuted[[2, 1]] = 8.;

        assert_eq!(permuted.shape(), [3, 2]);
        assert_eq!(permuted.to_vec(), &[1., 4., 2., 5., 3., 8.]);
    }

    #[test]
    fn test_reshaped() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let tensor = NdTensorView::from_data([1, 1, 2, 1, 3], &data);

        // Reshape to static dim count
        let reshaped = tensor.reshaped([6]);
        assert_eq!(reshaped.shape(), [6]);

        // Reshape to dynamic dim count
        let reshaped = tensor.reshaped([6].as_slice());
        assert_eq!(reshaped.shape(), &[6]);
    }

    #[test]
    fn test_reshaped_mut() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let mut tensor = NdTensor::from_data([1, 1, 2, 1, 3], data);

        let mut reshaped = tensor.reshaped_mut([6]);
        reshaped[[0]] = 0.;
        reshaped[[5]] = 0.;

        assert_eq!(tensor.data(), Some([0., 2., 3., 4., 5., 0.].as_slice()));
    }

    #[test]
    fn test_set_array() {
        let mut tensor = NdTensor::arange(1, 17, None).into_shape([4, 2, 2]);
        tensor.set_array([0, 0, 0], 0, [-1, -2, -3, -4]);
        assert_eq!(
            tensor.iter().copied().collect::<Vec<_>>(),
            &[-1, 2, 3, 4, -2, 6, 7, 8, -3, 10, 11, 12, -4, 14, 15, 16]
        );
    }

    #[test]
    fn test_slice_with_ndlayout() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);

        let row_one = tensor.slice(0);
        assert_eq!(row_one[[0]], 1.);
        assert_eq!(row_one[[1]], 2.);

        let row_two = tensor.slice(1);
        assert_eq!(row_two[[0]], 3.);
        assert_eq!(row_two[[1]], 4.);
    }

    #[test]
    fn test_slice_dyn_with_ndlayout() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);

        let row_one = tensor.slice_dyn(0);
        assert_eq!(row_one[[0]], 1.);
        assert_eq!(row_one[[1]], 2.);

        let row_two = tensor.slice_dyn(1);
        assert_eq!(row_two[[0]], 3.);
        assert_eq!(row_two[[1]], 4.);
    }

    #[test]
    fn test_slice_with_dynlayout() {
        let data = vec![1., 2., 3., 4.];
        let tensor = Tensor::from_data(&[2, 2], data);

        let row_one = tensor.slice(0);
        assert_eq!(row_one[[0]], 1.);
        assert_eq!(row_one[[1]], 2.);

        let row_two = tensor.slice(1);
        assert_eq!(row_two[[0]], 3.);
        assert_eq!(row_two[[1]], 4.);
    }

    #[test]
    fn test_slice_dyn_with_dynlayout() {
        let data = vec![1., 2., 3., 4.];
        let tensor = Tensor::from_data(&[2, 2], data);

        let row_one = tensor.slice_dyn(0);
        assert_eq!(row_one[[0]], 1.);
        assert_eq!(row_one[[1]], 2.);

        let row_two = tensor.slice_dyn(1);
        assert_eq!(row_two[[0]], 3.);
        assert_eq!(row_two[[1]], 4.);
    }

    #[test]
    fn test_slice_iter() {
        let data = vec![1., 2., 3., 4.];
        let tensor = Tensor::from_data(&[2, 2], data);
        let row_one: Vec<_> = tensor
            .slice_iter(&[SliceItem::Index(0), SliceItem::full_range()])
            .copied()
            .collect();
        assert_eq!(row_one, &[1., 2.]);
    }

    #[test]
    fn test_slice_mut() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor = NdTensor::from_data([2, 2], data);

        let mut row = tensor.slice_mut(1);
        row[[0]] = 8.;
        row[[1]] = 9.;

        assert_eq!(tensor.to_vec(), &[1., 2., 8., 9.]);
    }

    #[test]
    fn test_slice_mut_dyn() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor = NdTensor::from_data([2, 2], data);

        let mut row = tensor.slice_mut_dyn(1);
        row[[0]] = 8.;
        row[[1]] = 9.;

        assert_eq!(tensor.to_vec(), &[1., 2., 8., 9.]);
    }

    #[test]
    fn test_squeezed() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let tensor = NdTensorView::from_data([1, 1, 2, 1, 3], &data);

        let squeezed = tensor.squeezed();

        assert_eq!(squeezed.shape(), &[2, 3]);
    }

    #[test]
    fn test_to_array() {
        let tensor = NdTensor::arange(1., 5., None).into_shape([2, 2]);
        let col0: [f32; 2] = tensor.view().transposed().slice::<1, _>(0).to_array();
        let col1: [f32; 2] = tensor.view().transposed().slice::<1, _>(1).to_array();
        assert_eq!(col0, [1., 3.]);
        assert_eq!(col1, [2., 4.]);
    }

    #[test]
    fn test_to_contiguous() {
        let data = vec![1., 2., 3., 4.];
        let tensor = NdTensor::from_data([2, 2], data);
        // TODO - Try both contiguous and non-contiguous tensors.
        let tensor = tensor.to_contiguous();
        // TODO - Check the actual storage from to_contiguous
        assert_eq!(tensor.to_vec(), &[1., 2., 3., 4.]);
    }

    #[test]
    fn test_to_shape() {
        let tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
        let flat = tensor.to_shape([4]);
        assert_eq!(flat.shape(), [4]);
        assert_eq!(flat.data(), Some([1, 2, 3, 4].as_slice()));
    }

    #[test]
    fn test_to_vec() {
        // Contiguous case
        let tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
        assert_eq!(tensor.to_vec(), &[1, 2, 3, 4]);

        // Non-contiguous case
        let mut tensor = tensor.clone();
        tensor.transpose();
        assert_eq!(tensor.to_vec(), &[1, 3, 2, 4]);
    }

    #[test]
    fn test_to_tensor() {
        let data = vec![1., 2., 3., 4.];
        let view = NdTensorView::from_data([2, 2], &data);
        let tensor = view.to_tensor();
        assert_eq!(tensor.shape(), view.shape());
        assert_eq!(tensor.to_vec(), view.to_vec());
    }

    #[test]
    fn test_transpose() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let mut tensor = NdTensorView::from_data([2, 3], &data);

        tensor.transpose();

        assert_eq!(tensor.shape(), [3, 2]);
        assert_eq!(tensor.to_vec(), &[1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn test_transposed() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let tensor = NdTensorView::from_data([2, 3], &data);

        let permuted = tensor.transposed();

        assert_eq!(permuted.shape(), [3, 2]);
        assert_eq!(permuted.to_vec(), &[1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn test_try_slice() {
        let data = vec![1., 2., 3., 4.];
        let tensor = Tensor::from_data(&[2, 2], data);

        let row = tensor.try_slice(0);
        assert!(row.is_ok());
        assert_eq!(row.unwrap().data(), Some([1., 2.].as_slice()));

        let row = tensor.try_slice(1);
        assert!(row.is_ok());

        let row = tensor.try_slice(2);
        assert!(row.is_err());
    }

    #[test]
    fn test_try_slice_mut() {
        let data = vec![1., 2., 3., 4.];
        let mut tensor = Tensor::from_data(&[2, 2], data);

        let mut row = tensor.try_slice_mut(0).unwrap();
        row[[0]] += 1.;
        row[[1]] += 1.;
        assert_eq!(row.data(), Some([2., 3.].as_slice()));

        let row = tensor.try_slice_mut(1);
        assert!(row.is_ok());

        let row = tensor.try_slice(2);
        assert!(row.is_err());
    }

    #[test]
    fn test_view() {
        let tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
        let view = tensor.view();
        assert_eq!(view.data(), Some([1, 2, 3, 4].as_slice()));
    }

    #[test]
    fn test_view_mut() {
        let mut tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
        let mut view = tensor.view_mut();
        view[[0, 0]] = 0;
        view[[1, 1]] = 0;
        assert_eq!(tensor.data(), Some([0, 2, 3, 0].as_slice()));
    }

    #[test]
    fn test_zeros() {
        let tensor = NdTensor::zeros([2, 2]);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(tensor.data(), Some([0, 0, 0, 0].as_slice()));
    }
}
