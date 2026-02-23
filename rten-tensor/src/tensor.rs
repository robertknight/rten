use std::borrow::Cow;
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Range};
use std::sync::Arc;

use crate::assume_init::AssumeInit;
use crate::copy::{
    copy_into, copy_into_slice, copy_into_uninit, copy_range_into_slice, map_into_slice,
};
use crate::errors::{DimensionError, ExpandError, FromDataError, ReshapeError, SliceError};
use crate::iterators::{
    AxisChunks, AxisChunksMut, AxisIter, AxisIterMut, InnerIter, InnerIterMut, Iter, IterMut,
    Lanes, LanesMut, for_each_mut,
};
use crate::layout::{
    AsIndex, BroadcastLayout, DynLayout, InsertDim, IntoLayout, Layout, LayoutExt, MatrixLayout,
    MutLayout, NdLayout, OverlapPolicy, RemoveDim, ResizeLayout, SliceWith, TrustedLayout,
};
use crate::overlap::may_have_internal_overlap;
use crate::slice_range::{IntoSliceItems, SliceItem};
use crate::storage::{
    Alloc, CowData, GlobalAlloc, IntoStorage, Storage, StorageMut, ViewData, ViewMutData,
};
use crate::type_num::IndexCount;
use crate::{Contiguous, RandomSource};

/// The base type for multi-dimensional arrays. This consists of storage for
/// elements, plus a _layout_ which maps from a multi-dimensional array index
/// to a storage offset. This base type is not normally used directly but
/// instead through a type alias which selects the storage type and layout.
///
/// The storage can be owned (like a `Vec<T>`), borrowed (like `&[T]`) or
/// mutably borrowed (like `&mut [T]`). The layout can have a dimension count
/// that is determined statically (ie. forms part of the tensor's type), see
/// [`NdLayout`] or is only known at runtime, see [`DynLayout`].
pub struct TensorBase<S: Storage, L: Layout> {
    data: S,

    // Layout mapping N-dimensional indices to offsets in `data`.
    //
    // Constructors must ensure:
    //
    // - Every index that is valid for `layout` must map to an offset that is
    //   less than `data.len()`. The minimum length for a layout is given by
    //   `Layout::min_data_len`.
    // - If `S` is a mutable storage type, no two indices of `layout` can map to
    //   the same offset. See the `may_have_internal_overlap` function.
    layout: L,
}

/// Trait implemented by all variants of [`TensorBase`], which provides a
/// `view` method to get an immutable view of the tensor, plus methods which
/// forward to such a view.
///
/// The purpose of this trait is to allow methods to be specialized for
/// immutable views by preserving the lifetime of the underlying data in
/// return types (eg. `iter` returns `&[T]` in the trait, but `&'a [T]` in
/// the view). This allows for chaining operations on views together (eg.
/// `tensor.slice(...).transpose()`) without needing to separate each step
/// into separate statements.
///
/// This trait is conceptually similar to the way [`std::ops::Deref`] in the Rust
/// standard library allows a `Vec<T>` to have all the methods of an `&[T]`.
///
/// If stable Rust gains support for specialization or a `Deref` trait that can
/// return non-references (see <https://github.com/rust-lang/rfcs/issues/997>)
/// this will become unnecessary.
pub trait AsView: Layout {
    /// Type of element stored in this tensor.
    type Elem;

    /// The underlying layout of this tensor. It must have the same index
    /// type (eg. `[usize; N]` or `&[usize]`) as this view.
    type Layout: Clone + for<'a> Layout<Index<'a> = Self::Index<'a>>;

    /// Return a borrowed view of this tensor.
    fn view(&self) -> TensorBase<ViewData<'_, Self::Elem>, Self::Layout>;

    /// Return the layout of this tensor.
    fn layout(&self) -> &Self::Layout;

    /// Return a view of this tensor using a borrowed [`CowData`] for storage.
    ///
    /// Together with [`into_cow`](TensorBase::into_cow), this is useful where
    /// code needs to conditionally copy or create a new tensor, and get either
    /// the borrowed or owned tensor into the same type.
    fn as_cow(&self) -> TensorBase<CowData<'_, Self::Elem>, Self::Layout>
    where
        [Self::Elem]: ToOwned,
    {
        self.view().as_cow()
    }

    /// Return a view of this tensor with a dynamic rank.
    fn as_dyn(&self) -> TensorBase<ViewData<'_, Self::Elem>, DynLayout> {
        self.view().as_dyn()
    }

    /// Return an iterator over slices of this tensor along a given axis.
    fn axis_chunks(&self, dim: usize, chunk_size: usize) -> AxisChunks<'_, Self::Elem, Self::Layout>
    where
        Self::Layout: MutLayout,
    {
        self.view().axis_chunks(dim, chunk_size)
    }

    /// Return an iterator over slices of this tensor along a given axis.
    fn axis_iter(&self, dim: usize) -> AxisIter<'_, Self::Elem, Self::Layout>
    where
        Self::Layout: MutLayout + RemoveDim,
    {
        self.view().axis_iter(dim)
    }

    /// Broadcast this view to another shape.
    ///
    /// If `shape` is an array (`[usize; N]`), the result will have a
    /// static-rank layout with `N` dims. If `shape` is a slice, the result will
    /// have a dynamic-rank layout.
    fn broadcast<S: IntoLayout>(&self, shape: S) -> TensorBase<ViewData<'_, Self::Elem>, S::Layout>
    where
        Self::Layout: BroadcastLayout<S::Layout>,
    {
        self.view().broadcast(shape)
    }

    /// Fallible variant of [`broadcast`](AsView::broadcast).
    fn try_broadcast<S: IntoLayout>(
        &self,
        shape: S,
    ) -> Result<TensorBase<ViewData<'_, Self::Elem>, S::Layout>, ExpandError>
    where
        Self::Layout: BroadcastLayout<S::Layout>,
    {
        self.view().try_broadcast(shape)
    }

    /// Copy elements from this tensor into `dest` in logical order.
    ///
    /// Returns the initialized slice. Panics if the length of `dest` does
    /// not match the number of elements in `self`.
    fn copy_into_slice<'a>(&self, dest: &'a mut [MaybeUninit<Self::Elem>]) -> &'a [Self::Elem]
    where
        Self::Elem: Copy;

    /// Return the layout of this tensor as a slice, if it is contiguous.
    fn data(&self) -> Option<&[Self::Elem]>;

    /// Return a reference to the element at a given index, or `None` if the
    /// index is invalid.
    fn get<I: AsIndex<Self::Layout>>(&self, index: I) -> Option<&Self::Elem>
    where
        Self::Layout: TrustedLayout,
    {
        self.view().get(index)
    }

    /// Return a reference to the element at a given index, without performing
    /// bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is valid for the tensor's shape.
    unsafe fn get_unchecked<I: AsIndex<Self::Layout>>(&self, index: I) -> &Self::Elem {
        let view = self.view();
        unsafe { view.get_unchecked(index) }
    }

    /// Index the tensor along a given axis.
    ///
    /// Returns a view with one dimension removed.
    ///
    /// Panics if `axis >= self.ndim()` or `index >= self.size(axis)`.
    fn index_axis(
        &self,
        axis: usize,
        index: usize,
    ) -> TensorBase<ViewData<'_, Self::Elem>, <Self::Layout as RemoveDim>::Output>
    where
        Self::Layout: MutLayout + RemoveDim,
    {
        self.view().index_axis(axis, index)
    }

    /// Return an iterator over the innermost N dimensions.
    fn inner_iter<const N: usize>(&self) -> InnerIter<'_, Self::Elem, NdLayout<N>> {
        self.view().inner_iter()
    }

    /// Return an iterator over the innermost `n` dimensions.
    ///
    /// Prefer [`inner_iter`](AsView::inner_iter) if `N` is known at compile time.
    fn inner_iter_dyn(&self, n: usize) -> InnerIter<'_, Self::Elem, DynLayout> {
        self.view().inner_iter_dyn(n)
    }

    /// Insert a size-1 axis at the given index.
    fn insert_axis(&mut self, index: usize)
    where
        Self::Layout: ResizeLayout;

    /// Remove a size-1 axis at the given index.
    ///
    /// This will panic if the index is out of bounds or the size of the index
    /// is not 1.
    fn remove_axis(&mut self, index: usize)
    where
        Self::Layout: ResizeLayout;

    /// Return the scalar value in this tensor if it has 0 dimensions.
    fn item(&self) -> Option<&Self::Elem> {
        self.view().item()
    }

    /// Return an iterator over elements in this tensor in their logical order.
    fn iter(&self) -> Iter<'_, Self::Elem>;

    /// Return an iterator over 1D slices of this tensor along a given axis.
    fn lanes(&self, dim: usize) -> Lanes<'_, Self::Elem>
    where
        Self::Layout: RemoveDim,
    {
        self.view().lanes(dim)
    }

    /// Return a new tensor with the same shape, formed by applying `f` to each
    /// element in this tensor.
    fn map<F, U>(&self, f: F) -> TensorBase<Vec<U>, Self::Layout>
    where
        F: Fn(&Self::Elem) -> U,
        Self::Layout: MutLayout,
    {
        self.view().map(f)
    }

    /// Variant of [`map`](AsView::map) which takes an allocator.
    fn map_in<A: Alloc, F, U>(&self, alloc: A, f: F) -> TensorBase<Vec<U>, Self::Layout>
    where
        F: Fn(&Self::Elem) -> U,
        Self::Layout: MutLayout,
    {
        self.view().map_in(alloc, f)
    }

    /// Merge consecutive dimensions to the extent possible without copying
    /// data or changing the iteration order.
    ///
    /// If the tensor is contiguous, this has the effect of flattening the
    /// tensor into a vector.
    fn merge_axes(&mut self)
    where
        Self::Layout: ResizeLayout;

    /// Re-order the axes of this tensor to move the axis at index `from` to
    /// `to`.
    ///
    /// Panics if `from` or `to` is >= `self.ndim()`.
    fn move_axis(&mut self, from: usize, to: usize)
    where
        Self::Layout: MutLayout;

    /// Convert this tensor to one with the same shape but a static dimension
    /// count.
    ///
    /// Panics if `self.ndim() != N`.
    fn nd_view<const N: usize>(&self) -> TensorBase<ViewData<'_, Self::Elem>, NdLayout<N>> {
        self.view().nd_view()
    }

    /// Permute the dimensions of this tensor.
    fn permute(&mut self, order: Self::Index<'_>)
    where
        Self::Layout: MutLayout;

    /// Return a view with dimensions permuted in the order given by `dims`.
    fn permuted(&self, order: Self::Index<'_>) -> TensorBase<ViewData<'_, Self::Elem>, Self::Layout>
    where
        Self::Layout: MutLayout,
    {
        self.view().permuted(order)
    }

    /// Return either a view or a copy of `self` with the given shape.
    ///
    /// The new shape must have the same number of elments as the current
    /// shape. The result will have a static rank if `shape` is an array or
    /// a dynamic rank if it is a slice.
    ///
    /// If `self` is contiguous this will return a view, as changing the shape
    /// can be done without moving data. Otherwise it will copy elements into
    /// a new tensor.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the new shape does not match the
    /// current shape.
    fn reshaped<S: Copy + IntoLayout>(
        &self,
        shape: S,
    ) -> TensorBase<CowData<'_, Self::Elem>, S::Layout>
    where
        Self::Elem: Clone,
        Self::Layout: MutLayout,
    {
        self.view().reshaped(shape)
    }

    /// A variant of [`reshaped`](AsView::reshaped) that allows specifying the
    /// allocator to use if a copy is needed.
    fn reshaped_in<A: Alloc, S: Copy + IntoLayout>(
        &self,
        alloc: A,
        shape: S,
    ) -> TensorBase<CowData<'_, Self::Elem>, S::Layout>
    where
        Self::Elem: Clone,
        Self::Layout: MutLayout,
    {
        self.view().reshaped_in(alloc, shape)
    }

    /// Reverse the order of dimensions in this tensor.
    fn transpose(&mut self)
    where
        Self::Layout: MutLayout;

    /// Return a view with the order of dimensions reversed.
    fn transposed(&self) -> TensorBase<ViewData<'_, Self::Elem>, Self::Layout>
    where
        Self::Layout: MutLayout,
    {
        self.view().transposed()
    }

    /// Slice this tensor and return a view.
    ///
    /// If both this tensor's layout and the range have a statically-known
    /// number of index terms, the result will have a static rank. Otherwise it
    /// will have a dynamic rank.
    ///
    /// ```
    /// use rten_tensor::prelude::*;
    /// use rten_tensor::NdTensor;
    ///
    /// let x = NdTensor::from([[1, 2], [3, 4]]);
    /// let col = x.slice((.., 1)); // `col` is an `NdTensorView<i32, 1>`
    /// assert_eq!(col.shape(), [2usize]);
    /// assert_eq!(col.to_vec(), [2, 4]);
    /// ```
    #[allow(clippy::type_complexity)]
    fn slice<R: IntoSliceItems + IndexCount>(
        &self,
        range: R,
    ) -> TensorBase<ViewData<'_, Self::Elem>, <Self::Layout as SliceWith<R, R::Count>>::Layout>
    where
        Self::Layout: SliceWith<R, R::Count>,
    {
        self.view().slice(range)
    }

    /// Slice this tensor along a given axis.
    fn slice_axis(
        &self,
        axis: usize,
        range: Range<usize>,
    ) -> TensorBase<ViewData<'_, Self::Elem>, Self::Layout>
    where
        Self::Layout: MutLayout,
    {
        self.view().slice_axis(axis, range)
    }

    /// A variant of [`slice`](Self::slice) that returns a result
    /// instead of panicking.
    #[allow(clippy::type_complexity)]
    fn try_slice<R: IntoSliceItems + IndexCount>(
        &self,
        range: R,
    ) -> Result<
        TensorBase<ViewData<'_, Self::Elem>, <Self::Layout as SliceWith<R, R::Count>>::Layout>,
        SliceError,
    >
    where
        Self::Layout: SliceWith<R, R::Count>,
    {
        self.view().try_slice(range)
    }

    /// Return a slice of this tensor as an owned tensor.
    ///
    /// This is more expensive than [`slice`](AsView::slice) as it copies the
    /// data, but is more flexible as it supports ranges with negative steps.
    #[allow(clippy::type_complexity)]
    fn slice_copy<R: Clone + IntoSliceItems + IndexCount>(
        &self,
        range: R,
    ) -> TensorBase<Vec<Self::Elem>, <Self::Layout as SliceWith<R, R::Count>>::Layout>
    where
        Self::Elem: Clone,
        Self::Layout: SliceWith<
                R,
                R::Count,
                Layout: for<'a> Layout<Index<'a>: TryFrom<&'a [usize], Error: Debug>>,
            >,
    {
        self.slice_copy_in(GlobalAlloc::new(), range)
    }

    /// Variant of [`slice_copy`](AsView::slice_copy) which takes an allocator.
    #[allow(clippy::type_complexity)]
    fn slice_copy_in<A: Alloc, R: Clone + IntoSliceItems + IndexCount>(
        &self,
        pool: A,
        range: R,
    ) -> TensorBase<Vec<Self::Elem>, <Self::Layout as SliceWith<R, R::Count>>::Layout>
    where
        Self::Elem: Clone,
        Self::Layout: SliceWith<
                R,
                R::Count,
                Layout: for<'a> Layout<Index<'a>: TryFrom<&'a [usize], Error: Debug>>,
            >,
    {
        // Fast path for slice ranges supported by `Tensor::slice`. This includes
        // all ranges except those with a negative step. This benefits from
        // optimizations that `Tensor::to_tensor` has for slices that are already
        // contiguous or have a small number of dims.
        if let Ok(slice_view) = self.try_slice(range.clone()) {
            return slice_view.to_tensor_in(pool);
        }

        let items = range.into_slice_items();
        let sliced_shape: Vec<_> = items
            .as_ref()
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(dim, item)| match item {
                SliceItem::Index(_) => None,
                SliceItem::Range(range) => Some(range.index_range(self.size(dim)).steps()),
            })
            .collect();
        let sliced_len = sliced_shape.iter().product();
        let mut sliced_data = pool.alloc(sliced_len);

        copy_range_into_slice(
            self.as_dyn(),
            &mut sliced_data.spare_capacity_mut()[..sliced_len],
            items.as_ref(),
        );

        // Safety: `copy_range_into_slice` initialized `sliced_len` elements.
        unsafe {
            sliced_data.set_len(sliced_len);
        }

        let sliced_shape = sliced_shape.as_slice().try_into().expect("slice failed");

        TensorBase::from_data(sliced_shape, sliced_data)
    }

    /// Return a view of this tensor with all dimensions of size 1 removed.
    fn squeezed(&self) -> TensorView<'_, Self::Elem>
    where
        Self::Layout: MutLayout,
    {
        self.view().squeezed()
    }

    /// Return a vector containing the elements of this tensor in their logical
    /// order, ie. as if the tensor were flattened into one dimension.
    fn to_vec(&self) -> Vec<Self::Elem>
    where
        Self::Elem: Clone;

    /// Variant of [`to_vec`](AsView::to_vec) which takes an allocator.
    fn to_vec_in<A: Alloc>(&self, alloc: A) -> Vec<Self::Elem>
    where
        Self::Elem: Clone;

    /// Return a tensor with the same shape as this tensor/view but with the
    /// data contiguous in memory and arranged in the same order as the
    /// logical/iteration order (used by `iter`).
    ///
    /// This will return a view if the data is already contiguous or copy
    /// data into a new buffer otherwise.
    ///
    /// Certain operations require or are faster with contiguous tensors.
    fn to_contiguous(&self) -> Contiguous<TensorBase<CowData<'_, Self::Elem>, Self::Layout>>
    where
        Self::Elem: Clone,
        Self::Layout: MutLayout,
    {
        self.view().to_contiguous()
    }

    /// Variant of [`to_contiguous`](AsView::to_contiguous) which takes an
    /// allocator.
    fn to_contiguous_in<A: Alloc>(
        &self,
        alloc: A,
    ) -> Contiguous<TensorBase<CowData<'_, Self::Elem>, Self::Layout>>
    where
        Self::Elem: Clone,
        Self::Layout: MutLayout,
    {
        self.view().to_contiguous_in(alloc)
    }

    /// Return a copy of this tensor with a given shape.
    fn to_shape<S: IntoLayout>(&self, shape: S) -> TensorBase<Vec<Self::Elem>, S::Layout>
    where
        Self::Elem: Clone,
        Self::Layout: MutLayout;

    /// Return a slice containing the elements of this tensor in their logical
    /// order, ie. as if the tensor were flattened into one dimension.
    ///
    /// Unlike [`data`](AsView::data) this will copy the elements if they are
    /// not contiguous. Unlike [`to_vec`](AsView::to_vec) this will not copy
    /// the elements if the tensor is already contiguous.
    fn to_slice(&self) -> Cow<'_, [Self::Elem]>
    where
        Self::Elem: Clone,
    {
        self.view().to_slice()
    }

    /// Return a copy of this tensor/view which uniquely owns its elements.
    fn to_tensor(&self) -> TensorBase<Vec<Self::Elem>, Self::Layout>
    where
        Self::Elem: Clone,
        Self::Layout: MutLayout,
    {
        self.to_tensor_in(GlobalAlloc::new())
    }

    /// Variant of [`to_tensor`](AsView::to_tensor) which takes an allocator.
    fn to_tensor_in<A: Alloc>(&self, alloc: A) -> TensorBase<Vec<Self::Elem>, Self::Layout>
    where
        Self::Elem: Clone,
        Self::Layout: MutLayout,
    {
        TensorBase::from_data(self.layout().shape(), self.to_vec_in(alloc))
    }

    /// Return a view which performs "weak" checking when indexing via
    /// `view[<index>]`. See [`WeaklyCheckedView`] for an explanation.
    fn weakly_checked_view(&self) -> WeaklyCheckedView<ViewData<'_, Self::Elem>, Self::Layout> {
        self.view().weakly_checked_view()
    }
}

impl<S: Storage, L: Layout> TensorBase<S, L> {
    /// Construct a new tensor from a given shape and storage.
    ///
    /// Panics if the data length does not match the product of `shape`.
    #[track_caller]
    pub fn from_data<D: IntoStorage<Output = S>>(shape: L::Index<'_>, data: D) -> TensorBase<S, L>
    where
        for<'a> L::Index<'a>: Clone,
        L: MutLayout,
    {
        let data = data.into_storage();
        let len = data.len();
        match Self::try_from_data(shape.clone(), data) {
            Ok(data) => data,
            Err(_) => panic!(
                "data length {} does not match shape {:?}",
                len,
                shape.as_ref()
            ),
        }
    }

    /// Construct a new tensor from a given shape and storage.
    ///
    /// This will fail if the data length does not match the product of `shape`.
    pub fn try_from_data<D: IntoStorage<Output = S>>(
        shape: L::Index<'_>,
        data: D,
    ) -> Result<TensorBase<S, L>, FromDataError>
    where
        L: MutLayout,
    {
        let data = data.into_storage();
        let layout = L::from_shape(shape);
        if layout.min_data_len() != data.len() {
            return Err(FromDataError::StorageLengthMismatch);
        }
        Ok(TensorBase { data, layout })
    }

    /// Create a tensor from a pre-created storage and layout.
    ///
    /// Panics if the storage length is too short for the layout, or the storage
    /// is mutable and the layout may map multiple indices to the same offset.
    pub fn from_storage_and_layout(data: S, layout: L) -> TensorBase<S, L> {
        assert!(data.len() >= layout.min_data_len());
        assert!(
            !S::MUTABLE
                || !may_have_internal_overlap(layout.shape().as_ref(), layout.strides().as_ref())
        );
        TensorBase { data, layout }
    }

    /// Create a tensor from a pre-created storage and layout.
    ///
    /// # Safety
    ///
    /// Caller must ensure storage length is sufficient for the layout, and
    /// that, if the storage is mutable, no two indices in the layout map to the
    /// same offset.
    pub(crate) unsafe fn from_storage_and_layout_unchecked(data: S, layout: L) -> TensorBase<S, L> {
        debug_assert!(data.len() >= layout.min_data_len());
        debug_assert!(
            !S::MUTABLE
                || !may_have_internal_overlap(layout.shape().as_ref(), layout.strides().as_ref())
        );
        TensorBase { data, layout }
    }

    /// Construct a new tensor from a given shape and storage, and custom
    /// strides.
    ///
    /// This will fail if the data length is incorrect for the shape and stride
    /// combination, or if the strides lead to overlap (see [`OverlapPolicy`]).
    /// See also [`TensorBase::from_slice_with_strides`] which is a similar method
    /// for immutable views that does allow overlapping strides.
    pub fn from_data_with_strides<D: IntoStorage<Output = S>>(
        shape: L::Index<'_>,
        data: D,
        strides: L::Index<'_>,
    ) -> Result<TensorBase<S, L>, FromDataError>
    where
        L: MutLayout,
    {
        let layout = L::from_shape_and_strides(shape, strides, OverlapPolicy::DisallowOverlap)?;
        let data = data.into_storage();
        if layout.min_data_len() > data.len() {
            return Err(FromDataError::StorageTooShort);
        }
        Ok(TensorBase { data, layout })
    }

    /// Convert the current tensor into a dynamic rank tensor without copying
    /// any data.
    pub fn into_dyn(self) -> TensorBase<S, DynLayout>
    where
        L: Into<DynLayout>,
    {
        TensorBase {
            data: self.data,
            layout: self.layout.into(),
        }
    }

    /// Return a tensor with a size-1 dimension inserted at `axis`.
    ///
    /// `axis` must be in the range `0..=self.ndim()`. Panics if `axis` is out
    /// of bounds.
    ///
    /// This is a zero-copy operation that only changes the layout metadata.
    #[track_caller]
    pub fn with_new_axis(self, axis: usize) -> TensorBase<S, <L as InsertDim>::Output>
    where
        L: InsertDim,
    {
        assert!(
            axis <= self.ndim(),
            "axis {} is out of bounds for tensor with {} dims",
            axis,
            self.ndim()
        );
        let layout = self.layout.insert_dim(axis);
        TensorBase {
            data: self.data,
            layout,
        }
    }

    /// Consume this tensor and return the underlying storage.
    ///
    /// Be aware that the underlying elements are not guaranteed to be contiguous.
    pub(crate) fn into_storage(self) -> S {
        self.data
    }

    /// Attempt to convert this tensor's layout to a static-rank layout with `N`
    /// dimensions.
    fn nd_layout<const N: usize>(&self) -> Option<NdLayout<N>> {
        if self.ndim() != N {
            return None;
        }
        let shape: [usize; N] = std::array::from_fn(|i| self.size(i));
        let strides: [usize; N] = std::array::from_fn(|i| self.stride(i));
        let layout = NdLayout::from_shape_and_strides(shape, strides, OverlapPolicy::AllowOverlap)
            .expect("invalid layout");
        Some(layout)
    }

    /// Return a raw pointer to the tensor's underlying data.
    pub fn data_ptr(&self) -> *const S::Elem {
        self.data.as_ptr()
    }
}

impl<S: StorageMut, L: Clone + Layout> TensorBase<S, L> {
    /// Return an iterator over mutable slices of this tensor along a given
    /// axis. Each view yielded has one dimension fewer than the current layout.
    pub fn axis_iter_mut(&mut self, dim: usize) -> AxisIterMut<'_, S::Elem, L>
    where
        L: RemoveDim,
    {
        AxisIterMut::new(self.view_mut(), dim)
    }

    /// Return an iterator over mutable slices of this tensor along a given
    /// axis. Each view yielded has the same rank as this tensor, but the
    /// dimension `dim` will only have `chunk_size` entries.
    pub fn axis_chunks_mut(
        &mut self,
        dim: usize,
        chunk_size: usize,
    ) -> AxisChunksMut<'_, S::Elem, L>
    where
        L: MutLayout,
    {
        AxisChunksMut::new(self.view_mut(), dim, chunk_size)
    }

    /// Replace each element in this tensor with the result of applying `f` to
    /// the element.
    pub fn apply<F: Fn(&S::Elem) -> S::Elem>(&mut self, f: F) {
        if let Some(data) = self.data_mut() {
            // Fast path for contiguous tensors.
            data.iter_mut().for_each(|x| *x = f(x));
        } else {
            for_each_mut(self.as_dyn_mut(), |x| *x = f(x));
        }
    }

    /// Return a mutable view of this tensor with a dynamic dimension count.
    pub fn as_dyn_mut(&mut self) -> TensorBase<ViewMutData<'_, S::Elem>, DynLayout> {
        TensorBase {
            layout: DynLayout::from(&self.layout),
            data: self.data.view_mut(),
        }
    }

    /// Copy elements from another tensor into this tensor.
    ///
    /// This tensor and `other` must have the same shape.
    pub fn copy_from<S2: Storage<Elem = S::Elem>>(&mut self, other: &TensorBase<S2, L>)
    where
        S::Elem: Clone,
        L: Clone,
    {
        assert!(
            self.shape() == other.shape(),
            "copy dest shape {:?} != src shape {:?}",
            self.shape(),
            other.shape()
        );

        if let Some(dest) = self.data_mut() {
            if let Some(src) = other.data() {
                dest.clone_from_slice(src);
            } else {
                // Drop all the existing values. This should be compiled away for
                // `Copy` types.
                let uninit_dest: &mut [MaybeUninit<S::Elem>] = unsafe { std::mem::transmute(dest) };
                for x in &mut *uninit_dest {
                    // Safety: All elements were initialized at the start of this
                    // block, and we haven't written to the slice yet.
                    unsafe { x.assume_init_drop() }
                }

                // Copy source into destination in contiguous order.
                copy_into_slice(other.as_dyn(), uninit_dest);
            }
        } else {
            copy_into(other.as_dyn(), self.as_dyn_mut());
        }
    }

    /// Return the data in this tensor as a slice if it is contiguous.
    pub fn data_mut(&mut self) -> Option<&mut [S::Elem]> {
        // The length of `self.data` must be at least the minimum required by
        // the layout, but it may be larger.
        let len = self.layout.min_data_len();
        let data = self.data.slice_mut(0..len);

        self.layout.is_contiguous().then(|| unsafe {
            // Safety: We verified the layout is contiguous.
            data.to_slice_mut()
        })
    }

    /// Index the tensor along a given axis.
    ///
    /// Returns a mutable view with one dimension removed.
    ///
    /// Panics if `axis >= self.ndim()` or `index >= self.size(axis)`.
    pub fn index_axis_mut(
        &mut self,
        axis: usize,
        index: usize,
    ) -> TensorBase<ViewMutData<'_, S::Elem>, <L as RemoveDim>::Output>
    where
        L: MutLayout + RemoveDim,
    {
        let (offsets, layout) = self.layout.index_axis(axis, index);
        TensorBase {
            data: self.data.slice_mut(offsets),
            layout,
        }
    }

    /// Return a mutable view of the tensor's underlying storage.
    pub fn storage_mut(&mut self) -> ViewMutData<'_, S::Elem> {
        self.data.view_mut()
    }

    /// Replace all elements of this tensor with `value`.
    pub fn fill(&mut self, value: S::Elem)
    where
        S::Elem: Clone,
    {
        self.apply(|_| value.clone())
    }

    /// Return a mutable reference to the element at `index`, or `None` if the
    /// index is invalid.
    pub fn get_mut<I: AsIndex<L>>(&mut self, index: I) -> Option<&mut S::Elem>
    where
        L: TrustedLayout,
    {
        self.offset(index.as_index()).map(|offset| unsafe {
            // Safety: We verified the offset is in-bounds.
            self.data.get_unchecked_mut(offset)
        })
    }

    /// Return the element at a given index, without performing any bounds-
    /// checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is valid for the tensor's shape.
    pub unsafe fn get_unchecked_mut<I: AsIndex<L>>(&mut self, index: I) -> &mut S::Elem {
        let offset = self.layout.offset_unchecked(index.as_index());
        unsafe { self.data.get_unchecked_mut(offset) }
    }

    pub(crate) fn mut_view_ref(&mut self) -> TensorBase<ViewMutData<'_, S::Elem>, &L> {
        TensorBase {
            data: self.data.view_mut(),
            layout: &self.layout,
        }
    }

    /// Return a mutable iterator over the N innermost dimensions of this tensor.
    pub fn inner_iter_mut<const N: usize>(&mut self) -> InnerIterMut<'_, S::Elem, NdLayout<N>>
    where
        L: MutLayout,
    {
        InnerIterMut::new(self.view_mut())
    }

    /// Return a mutable iterator over the n innermost dimensions of this tensor.
    ///
    /// Prefer [`inner_iter_mut`](TensorBase::inner_iter_mut) if `N` is known
    /// at compile time.
    pub fn inner_iter_dyn_mut(&mut self, n: usize) -> InnerIterMut<'_, S::Elem, DynLayout>
    where
        L: MutLayout,
    {
        InnerIterMut::new_dyn(self.view_mut(), n)
    }

    /// Return a mutable iterator over the elements of this tensor, in their
    /// logical order.
    pub fn iter_mut(&mut self) -> IterMut<'_, S::Elem> {
        IterMut::new(self.mut_view_ref())
    }

    /// Return an iterator over mutable 1D slices of this tensor along a given
    /// dimension.
    pub fn lanes_mut(&mut self, dim: usize) -> LanesMut<'_, S::Elem>
    where
        L: RemoveDim,
    {
        LanesMut::new(self.mut_view_ref(), dim)
    }

    /// Return a view of this tensor with a static dimension count.
    ///
    /// Panics if `self.ndim() != N`.
    pub fn nd_view_mut<const N: usize>(
        &mut self,
    ) -> TensorBase<ViewMutData<'_, S::Elem>, NdLayout<N>> {
        assert!(self.ndim() == N, "ndim {} != {}", self.ndim(), N);
        TensorBase {
            layout: self.nd_layout().unwrap(),
            data: self.data.view_mut(),
        }
    }

    /// Permute the order of dimensions according to the given order.
    ///
    /// See [`AsView::permuted`].
    pub fn permuted_mut(&mut self, order: L::Index<'_>) -> TensorBase<ViewMutData<'_, S::Elem>, L>
    where
        L: MutLayout,
    {
        TensorBase {
            layout: self.layout.permuted(order),
            data: self.data.view_mut(),
        }
    }

    /// Change the layout of the tensor without moving any data.
    ///
    /// This will return an error if the view is not contiguous.
    ///
    /// See also [`AsView::reshaped`].
    pub fn reshaped_mut<SH: IntoLayout>(
        &mut self,
        shape: SH,
    ) -> Result<TensorBase<ViewMutData<'_, S::Elem>, SH::Layout>, ReshapeError>
    where
        L: MutLayout,
    {
        let layout = self.layout.reshaped_for_view(shape)?;
        Ok(TensorBase {
            layout,
            data: self.data.view_mut(),
        })
    }

    /// Slice this tensor along a given axis.
    pub fn slice_axis_mut(
        &mut self,
        axis: usize,
        range: Range<usize>,
    ) -> TensorBase<ViewMutData<'_, S::Elem>, L>
    where
        L: MutLayout,
    {
        let (offset_range, sliced_layout) = self.layout.slice_axis(axis, range.clone()).unwrap();
        debug_assert_eq!(sliced_layout.size(axis), range.len());
        TensorBase {
            data: self.data.slice_mut(offset_range),
            layout: sliced_layout,
        }
    }

    /// Slice this tensor and return a mutable view.
    ///
    /// See [`slice`](AsView::slice) for notes on the layout of the returned
    /// view.
    pub fn slice_mut<R: IntoSliceItems + IndexCount>(
        &mut self,
        range: R,
    ) -> TensorBase<ViewMutData<'_, S::Elem>, <L as SliceWith<R, R::Count>>::Layout>
    where
        L: SliceWith<R, R::Count>,
    {
        self.try_slice_mut(range).expect("slice failed")
    }

    /// A variant of [`slice_mut`](Self::slice_mut) that returns a
    /// result instead of panicking.
    #[allow(clippy::type_complexity)]
    pub fn try_slice_mut<R: IntoSliceItems + IndexCount>(
        &mut self,
        range: R,
    ) -> Result<
        TensorBase<ViewMutData<'_, S::Elem>, <L as SliceWith<R, R::Count>>::Layout>,
        SliceError,
    >
    where
        L: SliceWith<R, R::Count>,
    {
        let (offset_range, sliced_layout) = self.layout.slice_with(range)?;
        Ok(TensorBase {
            data: self.data.slice_mut(offset_range),
            layout: sliced_layout,
        })
    }

    /// Return a mutable view of this tensor.
    pub fn view_mut(&mut self) -> TensorBase<ViewMutData<'_, S::Elem>, L>
    where
        L: Clone,
    {
        TensorBase {
            data: self.data.view_mut(),
            layout: self.layout.clone(),
        }
    }

    /// Return a mutable view that performs only "weak" checking when indexing,
    /// this is faster but can hide bugs. See [`WeaklyCheckedView`].
    pub fn weakly_checked_view_mut(&mut self) -> WeaklyCheckedView<ViewMutData<'_, S::Elem>, L> {
        WeaklyCheckedView {
            base: self.view_mut(),
        }
    }
}

impl<T, L: Clone + Layout> TensorBase<Vec<T>, L> {
    /// Create a new 1D tensor filled with an arithmetic sequence of values
    /// in the range `[start, end)` separated by `step`. If `step` is omitted,
    /// it defaults to 1.
    pub fn arange(start: T, end: T, step: Option<T>) -> TensorBase<Vec<T>, L>
    where
        T: Copy + PartialOrd + From<bool> + std::ops::Add<Output = T>,
        [usize; 1]: AsIndex<L>,
        L: MutLayout,
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

    /// Append elements from `other` to this tensor along a given axis.
    ///
    /// This will fail if the shapes of `self` and `other` do not match along
    /// dimensions other than `axis`, or if the current tensor has
    /// insufficient capacity to expand without re-allocating.
    pub fn append<S2: Storage<Elem = T>>(
        &mut self,
        axis: usize,
        other: &TensorBase<S2, L>,
    ) -> Result<(), ExpandError>
    where
        T: Copy,
        L: MutLayout,
    {
        let shape_match = self.ndim() == other.ndim()
            && (0..self.ndim()).all(|d| d == axis || self.size(d) == other.size(d));
        if !shape_match {
            return Err(ExpandError::ShapeMismatch);
        }

        let old_size = self.size(axis);
        let new_size = self.size(axis) + other.size(axis);

        let Some(new_layout) = self.expanded_layout(axis, new_size) else {
            return Err(ExpandError::InsufficientCapacity);
        };

        let new_data_len = new_layout.min_data_len();
        self.layout = new_layout;

        // Safety: The `copy_from` call below will initialize all elements
        // added to the tensor.
        assert!(self.data.capacity() >= new_data_len);
        unsafe {
            self.data.set_len(new_data_len);
        }

        self.slice_axis_mut(axis, old_size..new_size)
            .copy_from(other);

        Ok(())
    }

    /// Create a new 1D tensor from a `Vec<T>`.
    pub fn from_vec(vec: Vec<T>) -> TensorBase<Vec<T>, L>
    where
        [usize; 1]: AsIndex<L>,
        L: MutLayout,
    {
        TensorBase::from_data([vec.len()].as_index(), vec)
    }

    /// Clip dimension `dim` to `[range.start, range.end)`. The new size for
    /// the dimension must be <= the old size.
    ///
    /// This currently requires `T: Copy` to support efficiently moving data
    /// from the new start offset to the beginning of the element buffer.
    pub fn clip_dim(&mut self, dim: usize, range: Range<usize>)
    where
        T: Copy,
        L: MutLayout,
    {
        let (start, end) = (range.start, range.end);

        assert!(start <= end, "start must be <= end");
        assert!(end <= self.size(dim), "end must be <= dim size");

        self.layout.resize_dim(dim, end - start);

        let range = if self.is_empty() {
            0..0
        } else {
            let start_offset = start * self.layout.stride(dim);
            let end_offset = start_offset + self.layout.min_data_len();
            start_offset..end_offset
        };
        self.data.copy_within(range.clone(), 0);
        self.data.truncate(range.end - range.start);
    }

    /// Return true if this tensor can be expanded along a given axis to a
    /// new size without re-allocating.
    pub fn has_capacity(&self, axis: usize, new_size: usize) -> bool
    where
        L: MutLayout,
    {
        self.expanded_layout(axis, new_size).is_some()
    }

    /// Return the layout this tensor would have if the size of `axis` were
    /// expanded to `new_size`.
    ///
    /// Returns `None` if the tensor does not have capacity for the new size.
    fn expanded_layout(&self, axis: usize, new_size: usize) -> Option<L>
    where
        L: MutLayout,
    {
        let mut new_layout = self.layout.clone();
        new_layout.resize_dim(axis, new_size);
        let new_data_len = new_layout.min_data_len();

        let has_capacity = new_data_len <= self.data.capacity()
            && !may_have_internal_overlap(
                new_layout.shape().as_ref(),
                new_layout.strides().as_ref(),
            );

        has_capacity.then_some(new_layout)
    }

    /// Convert the storage of this tensor into an owned [`CowData`].
    ///
    /// This is useful in contexts where code needs to conditionally copy or
    /// create a new tensor. See [`AsView::as_cow`].
    pub fn into_cow(self) -> TensorBase<CowData<'static, T>, L> {
        let TensorBase { data, layout } = self;
        TensorBase {
            layout,
            data: CowData::Owned(data),
        }
    }

    /// Convert the storage of this tensor to be reference counted.
    ///
    /// This is a (relatively) cheap operation that does not copy the tensor
    /// data.
    pub fn into_arc(self) -> TensorBase<Arc<Vec<T>>, L> {
        let TensorBase { data, layout } = self;
        TensorBase {
            layout,
            data: Arc::new(data),
        }
    }

    /// Consume self and return the underlying data as a contiguous tensor.
    ///
    /// See also [`TensorBase::to_vec`].
    pub fn into_data(self) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_contiguous() {
            self.into_non_contiguous_data()
        } else {
            self.to_vec()
        }
    }

    /// Consume self and return the underlying data in whatever order the
    /// elements are currently stored.
    pub fn into_non_contiguous_data(mut self) -> Vec<T> {
        self.data.truncate(self.layout.min_data_len());
        self.data
    }

    /// Consume self and return a new contiguous tensor with the given shape.
    ///
    /// This avoids copying the data if it is already contiguous.
    #[track_caller]
    pub fn into_shape<S: Copy + IntoLayout>(self, shape: S) -> TensorBase<Vec<T>, S::Layout>
    where
        T: Clone,
        L: MutLayout,
    {
        let Ok(layout) = self.layout.reshaped_for_copy(shape) else {
            panic!(
                "element count mismatch reshaping {:?} to {:?}",
                self.shape(),
                shape
            );
        };
        TensorBase {
            layout,
            data: self.into_data(),
        }
    }

    /// Create a new tensor with a given shape and values generated by calling
    /// `f` repeatedly.
    ///
    /// Each call to `f` will receive an element index and should return the
    /// corresponding value. If the function does not need this index, use
    /// [`from_simple_fn`](TensorBase::from_simple_fn) instead, as it is faster.
    pub fn from_fn<F: FnMut(L::Index<'_>) -> T, Idx>(
        shape: L::Index<'_>,
        mut f: F,
    ) -> TensorBase<Vec<T>, L>
    where
        L::Indices: Iterator<Item = Idx>,
        Idx: AsIndex<L>,
        L: MutLayout,
    {
        let layout = L::from_shape(shape);
        let data: Vec<T> = layout.indices().map(|idx| f(idx.as_index())).collect();
        TensorBase { data, layout }
    }

    /// Create a new tensor with a given shape and values generated by calling
    /// `f` repeatedly.
    pub fn from_simple_fn<F: FnMut() -> T>(shape: L::Index<'_>, f: F) -> TensorBase<Vec<T>, L>
    where
        L: MutLayout,
    {
        Self::from_simple_fn_in(GlobalAlloc::new(), shape, f)
    }

    /// Variant of [`from_simple_fn`](TensorBase::from_simple_fn) that takes
    /// an allocator.
    pub fn from_simple_fn_in<A: Alloc, F: FnMut() -> T>(
        alloc: A,
        shape: L::Index<'_>,
        mut f: F,
    ) -> TensorBase<Vec<T>, L>
    where
        L: MutLayout,
    {
        let len = shape.as_ref().iter().product();
        let mut data = alloc.alloc(len);
        data.extend(std::iter::from_fn(|| Some(f())).take(len));
        TensorBase::from_data(shape, data)
    }

    /// Create a new 0D tensor from a scalar value.
    pub fn from_scalar(value: T) -> TensorBase<Vec<T>, L>
    where
        [usize; 0]: AsIndex<L>,
        L: MutLayout,
    {
        TensorBase::from_data([].as_index(), vec![value])
    }

    /// Create a new tensor with a given shape and all elements set to `value`.
    pub fn full(shape: L::Index<'_>, value: T) -> TensorBase<Vec<T>, L>
    where
        T: Clone,
        L: MutLayout,
    {
        Self::full_in(GlobalAlloc::new(), shape, value)
    }

    /// Variant of [`full`](TensorBase::full) which takes an allocator.
    pub fn full_in<A: Alloc>(alloc: A, shape: L::Index<'_>, value: T) -> TensorBase<Vec<T>, L>
    where
        T: Clone,
        L: MutLayout,
    {
        let len = shape.as_ref().iter().product();
        let mut data = alloc.alloc(len);
        data.resize(len, value);
        TensorBase::from_data(shape, data)
    }

    /// Make the underlying data in this tensor contiguous.
    ///
    /// This means that after calling `make_contiguous`, the elements are
    /// guaranteed to be stored in the same order as the logical order in
    /// which `iter` yields elements. This method is cheap if the storage is
    /// already contiguous.
    pub fn make_contiguous(&mut self)
    where
        T: Clone,
        L: MutLayout,
    {
        if self.is_contiguous() {
            return;
        }
        self.data = self.to_vec();
        self.layout = L::from_shape(self.layout.shape());
    }

    /// Create a new tensor with a given shape and elements populated using
    /// numbers generated by `rand_src`.
    ///
    /// A more general version of this method that generates values using any
    /// function is [`from_simple_fn`](Self::from_simple_fn).
    pub fn rand<R: RandomSource<T>>(shape: L::Index<'_>, rand_src: &mut R) -> TensorBase<Vec<T>, L>
    where
        L: MutLayout,
    {
        Self::from_simple_fn(shape, || rand_src.next())
    }

    /// Create a new tensor with a given shape, with all elements set to their
    /// default value (ie. zero for numeric types).
    pub fn zeros(shape: L::Index<'_>) -> TensorBase<Vec<T>, L>
    where
        T: Clone + Default,
        L: MutLayout,
    {
        Self::zeros_in(GlobalAlloc::new(), shape)
    }

    /// Variant of [`zeros`](TensorBase::zeros) which takes an allocator.
    pub fn zeros_in<A: Alloc>(alloc: A, shape: L::Index<'_>) -> TensorBase<Vec<T>, L>
    where
        T: Clone + Default,
        L: MutLayout,
    {
        // We delegate to `full_in` here and rely on compiler optimizations to
        // take advantage of the value being statically known to be zero.
        Self::full_in(alloc, shape, T::default())
    }

    /// Return a new tensor containing uninitialized elements.
    ///
    /// The caller must initialize elements and then call
    /// [`assume_init`](TensorBase::assume_init) to convert to an initialized
    /// `Tensor<T>`.
    pub fn uninit(shape: L::Index<'_>) -> TensorBase<Vec<MaybeUninit<T>>, L>
    where
        MaybeUninit<T>: Clone,
        L: MutLayout,
    {
        Self::uninit_in(GlobalAlloc::new(), shape)
    }

    /// Variant of [`uninit`](TensorBase::uninit) which takes an allocator.
    pub fn uninit_in<A: Alloc>(alloc: A, shape: L::Index<'_>) -> TensorBase<Vec<MaybeUninit<T>>, L>
    where
        L: MutLayout,
    {
        let len = shape.as_ref().iter().product();
        let mut data = alloc.alloc(len);

        // Safety: Since the contents of the `Vec` are `MaybeUninit`, we don't
        // need to initialize them.
        unsafe { data.set_len(len) }

        TensorBase::from_data(shape, data)
    }

    /// Concatenate a slice of tensors along a given dimension.
    ///
    /// All tensors must have the same shape, except along `dim` where the
    /// sizes may differ. Returns an error if `tensors` is empty or if shapes
    /// are incompatible.
    pub fn concat<S2: Storage<Elem = T>>(
        dim: usize,
        tensors: &[TensorBase<S2, L>],
    ) -> Result<TensorBase<Vec<T>, L>, ExpandError>
    where
        T: Copy,
        L: MutLayout,
    {
        let first = tensors.first().ok_or(ExpandError::ShapeMismatch)?;
        let total_dim_size: usize = tensors.iter().map(|t| t.size(dim)).sum();

        let mut target_layout = first.layout().clone();
        target_layout.resize_dim(dim, total_dim_size);

        let mut result = Self::with_capacity(target_layout.shape(), dim);
        for tensor in tensors {
            result.append(dim, tensor)?;
        }
        Ok(result)
    }

    /// Create a tensor which initially has zero elements, but can be expanded
    /// along a given dimension without reallocating.
    ///
    /// `shape` specifies the maximum shape that the tensor can be expanded to
    /// without reallocating. The initial shape will be the same, except for
    /// the dimension specified by `expand_dim`, which will be zero.
    pub fn with_capacity(shape: L::Index<'_>, expand_dim: usize) -> TensorBase<Vec<T>, L>
    where
        T: Copy,
        L: MutLayout,
    {
        Self::with_capacity_in(GlobalAlloc::new(), shape, expand_dim)
    }

    /// Variant of [`with_capacity`](Self::with_capacity) which takes an allocator.
    pub fn with_capacity_in<A: Alloc>(
        alloc: A,
        shape: L::Index<'_>,
        expand_dim: usize,
    ) -> TensorBase<Vec<T>, L>
    where
        T: Copy,
        L: MutLayout,
    {
        let mut tensor = Self::uninit_in(alloc, shape);
        tensor.clip_dim(expand_dim, 0..0);

        // Safety: Since at least one dimension has a size of zero, the tensor
        // has no elements and thus is fully initialized.
        unsafe { tensor.assume_init() }
    }
}

impl<T, L: Layout> TensorBase<CowData<'_, T>, L> {
    /// Consume self and return the underlying data in whatever order the
    /// elements are currently stored, if the storage is owned, or `None` if
    /// it is borrowed.
    pub fn into_non_contiguous_data(self) -> Option<Vec<T>> {
        match self.data {
            CowData::Owned(mut vec) => {
                vec.truncate(self.layout.min_data_len());
                Some(vec)
            }
            CowData::Borrowed(_) => None,
        }
    }
}

impl<T, S: Storage<Elem = MaybeUninit<T>> + AssumeInit, L: Layout + Clone> TensorBase<S, L>
where
    <S as AssumeInit>::Output: Storage<Elem = T>,
{
    /// Convert a tensor of potentially uninitialized elements to one of
    /// initialized elements.
    ///
    /// See also [`MaybeUninit::assume_init`].
    ///
    /// # Safety
    ///
    /// The caller must guarantee that all elements in this tensor have been
    /// initialized before calling `assume_init`.
    pub unsafe fn assume_init(self) -> TensorBase<<S as AssumeInit>::Output, L> {
        TensorBase {
            layout: self.layout,
            data: unsafe { self.data.assume_init() },
        }
    }

    /// Initialize this tensor with data from another view.
    ///
    /// This tensor and `other` must have the same shape.
    pub fn init_from<S2: Storage<Elem = T>>(
        mut self,
        other: &TensorBase<S2, L>,
    ) -> TensorBase<<S as AssumeInit>::Output, L>
    where
        T: Copy,
        S: StorageMut<Elem = MaybeUninit<T>>,
    {
        assert_eq!(self.shape(), other.shape(), "shape mismatch");

        match (self.data_mut(), other.data()) {
            // Source and dest are contiguous. Use a memcpy.
            (Some(self_data), Some(other_data)) => {
                let other_data: &[MaybeUninit<T>] = unsafe { std::mem::transmute(other_data) };
                self_data.clone_from_slice(other_data);
            }
            // Dest is contiguous.
            (Some(self_data), _) => {
                copy_into_slice(other.as_dyn(), self_data);
            }
            // Neither are contiguous.
            _ => {
                copy_into_uninit(other.as_dyn(), self.as_dyn_mut());
            }
        }

        unsafe { self.assume_init() }
    }
}

impl<'a, T, L: Clone + Layout> TensorBase<ViewData<'a, T>, L> {
    pub fn axis_iter(&self, dim: usize) -> AxisIter<'a, T, L>
    where
        L: MutLayout + RemoveDim,
    {
        AxisIter::new(self, dim)
    }

    pub fn axis_chunks(&self, dim: usize, chunk_size: usize) -> AxisChunks<'a, T, L>
    where
        L: MutLayout,
    {
        AxisChunks::new(self, dim, chunk_size)
    }

    /// Return a view of this tensor with a dynamic dimension count.
    ///
    /// See [`AsView::as_dyn`].
    pub fn as_dyn(&self) -> TensorBase<ViewData<'a, T>, DynLayout> {
        TensorBase {
            data: self.data,
            layout: DynLayout::from(&self.layout),
        }
    }

    /// Convert the storage of this view to a borrowed [`CowData`].
    ///
    /// See [`AsView::as_cow`].
    pub fn as_cow(&self) -> TensorBase<CowData<'a, T>, L> {
        TensorBase {
            layout: self.layout.clone(),
            data: CowData::Borrowed(self.data),
        }
    }

    /// Broadcast this view to another shape.
    ///
    /// See [`AsView::broadcast`].
    pub fn broadcast<S: IntoLayout>(&self, shape: S) -> TensorBase<ViewData<'a, T>, S::Layout>
    where
        L: BroadcastLayout<S::Layout>,
    {
        self.try_broadcast(shape).unwrap()
    }

    /// Broadcast this view to another shape.
    ///
    /// See [`AsView::broadcast`].
    pub fn try_broadcast<S: IntoLayout>(
        &self,
        shape: S,
    ) -> Result<TensorBase<ViewData<'a, T>, S::Layout>, ExpandError>
    where
        L: BroadcastLayout<S::Layout>,
    {
        Ok(TensorBase {
            layout: self.layout.broadcast(shape)?,
            data: self.data,
        })
    }

    /// Return the data in this tensor as a slice if it is contiguous, ie.
    /// the order of elements in the slice is the same as the logical order
    /// yielded by `iter`, and there are no gaps.
    pub fn data(&self) -> Option<&'a [T]> {
        // The length of `self.data` must be at least the minimum required by
        // the layout, but it may be larger.
        let len = self.layout.min_data_len();
        let data = self.data.slice(0..len);

        self.layout.is_contiguous().then(|| unsafe {
            // Safety: Storage is contigous
            data.as_slice()
        })
    }

    /// Return an immutable view of the tensor's underlying storage.
    pub fn storage(&self) -> ViewData<'a, T> {
        self.data.view()
    }

    pub fn get<I: AsIndex<L>>(&self, index: I) -> Option<&'a T>
    where
        L: TrustedLayout,
    {
        self.offset(index.as_index()).map(|offset|
                // Safety:
                // - No logically overlapping mutable view exist.
                // - For trusted layouts, offset is promised to be less than
                //   the storage length
                unsafe {
                self.data.get_unchecked(offset)
            })
    }

    /// Create a new view with a given shape and data slice, and custom strides.
    ///
    /// If you do not need to specify custom strides, use [`TensorBase::from_data`]
    /// instead. This method is similar to [`TensorBase::from_data_with_strides`],
    /// but allows strides that lead to internal overlap (see [`OverlapPolicy`]).
    pub fn from_slice_with_strides(
        shape: L::Index<'_>,
        data: &'a [T],
        strides: L::Index<'_>,
    ) -> Result<TensorBase<ViewData<'a, T>, L>, FromDataError>
    where
        L: MutLayout,
    {
        let layout = L::from_shape_and_strides(shape, strides, OverlapPolicy::AllowOverlap)?;
        if layout.min_data_len() > data.as_ref().len() {
            return Err(FromDataError::StorageTooShort);
        }
        Ok(TensorBase {
            data: data.into_storage(),
            layout,
        })
    }

    /// Return the element at a given index, without performing any bounds-
    /// checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is valid for the tensor's shape.
    pub unsafe fn get_unchecked<I: AsIndex<L>>(&self, index: I) -> &'a T {
        let offset = self.layout.offset_unchecked(index.as_index());
        unsafe { self.data.get_unchecked(offset) }
    }

    /// Index the tensor along a given axis.
    ///
    /// Returns a view with one dimension removed.
    ///
    /// Panics if `axis >= self.ndim()` or `index >= self.size(axis)`.
    pub fn index_axis(
        &self,
        axis: usize,
        index: usize,
    ) -> TensorBase<ViewData<'a, T>, <L as RemoveDim>::Output>
    where
        L: MutLayout + RemoveDim,
    {
        let (offsets, layout) = self.layout.index_axis(axis, index);
        TensorBase {
            data: self.data.slice(offsets),
            layout,
        }
    }

    /// Return an iterator over the inner `N` dimensions of this tensor.
    ///
    /// See [`AsView::inner_iter`].
    pub fn inner_iter<const N: usize>(&self) -> InnerIter<'a, T, NdLayout<N>> {
        InnerIter::new(self.view())
    }

    /// Return an iterator over the inner `n` dimensions of this tensor.
    ///
    /// See [`AsView::inner_iter_dyn`].
    pub fn inner_iter_dyn(&self, n: usize) -> InnerIter<'a, T, DynLayout> {
        InnerIter::new_dyn(self.view(), n)
    }

    /// Return the scalar value in this tensor if it has one element.
    pub fn item(&self) -> Option<&'a T> {
        match self.ndim() {
            0 => unsafe {
                // Safety: No logically overlapping mutable views exist.
                self.data.get(0)
            },
            _ if self.len() == 1 => self.iter().next(),
            _ => None,
        }
    }

    /// Return an iterator over elements of this tensor in their logical order.
    ///
    /// See [`AsView::iter`].
    pub fn iter(&self) -> Iter<'a, T> {
        Iter::new(self.view_ref())
    }

    /// Return an iterator over 1D slices of this tensor along a given dimension.
    ///
    /// See [`AsView::lanes`].
    pub fn lanes(&self, dim: usize) -> Lanes<'a, T>
    where
        L: RemoveDim,
    {
        assert!(dim < self.ndim());
        Lanes::new(self.view_ref(), dim)
    }

    /// Return a view of this tensor with a static dimension count.
    ///
    /// Panics if `self.ndim() != N`.
    pub fn nd_view<const N: usize>(&self) -> TensorBase<ViewData<'a, T>, NdLayout<N>> {
        assert!(self.ndim() == N, "ndim {} != {}", self.ndim(), N);
        TensorBase {
            data: self.data,
            layout: self.nd_layout().unwrap(),
        }
    }

    /// Permute the axes of this tensor according to `order`.
    ///
    /// See [`AsView::permuted`].
    pub fn permuted(&self, order: L::Index<'_>) -> TensorBase<ViewData<'a, T>, L>
    where
        L: MutLayout,
    {
        TensorBase {
            data: self.data,
            layout: self.layout.permuted(order),
        }
    }

    /// Return a view or owned tensor that has the given shape.
    ///
    /// See [`AsView::reshaped`].
    pub fn reshaped<S: Copy + IntoLayout>(&self, shape: S) -> TensorBase<CowData<'a, T>, S::Layout>
    where
        T: Clone,
        L: MutLayout,
    {
        self.reshaped_in(GlobalAlloc::new(), shape)
    }

    /// Variant of [`reshaped`](Self::reshaped) that takes an allocator.
    pub fn reshaped_in<A: Alloc, S: Copy + IntoLayout>(
        &self,
        alloc: A,
        shape: S,
    ) -> TensorBase<CowData<'a, T>, S::Layout>
    where
        T: Clone,
        L: MutLayout,
    {
        if let Ok(layout) = self.layout.reshaped_for_view(shape) {
            TensorBase {
                data: CowData::Borrowed(self.data),
                layout,
            }
        } else {
            let Ok(layout) = self.layout.reshaped_for_copy(shape) else {
                panic!(
                    "element count mismatch reshaping {:?} to {:?}",
                    self.shape(),
                    shape
                );
            };

            TensorBase {
                data: CowData::Owned(self.to_vec_in(alloc)),
                layout,
            }
        }
    }

    /// Slice this tensor and return a view. See [`AsView::slice`].
    pub fn slice<R: IntoSliceItems + IndexCount>(
        &self,
        range: R,
    ) -> TensorBase<ViewData<'a, T>, <L as SliceWith<R, R::Count>>::Layout>
    where
        L: SliceWith<R, R::Count>,
    {
        self.try_slice(range).expect("slice failed")
    }

    /// Slice this tensor along a given axis.
    pub fn slice_axis(&self, axis: usize, range: Range<usize>) -> TensorBase<ViewData<'a, T>, L>
    where
        L: MutLayout,
    {
        let (offset_range, sliced_layout) = self.layout.slice_axis(axis, range.clone()).unwrap();
        debug_assert_eq!(sliced_layout.size(axis), range.len());
        TensorBase {
            data: self.data.slice(offset_range),
            layout: sliced_layout,
        }
    }

    /// A variant of [`slice`](Self::slice) that returns a result
    /// instead of panicking.
    #[allow(clippy::type_complexity)]
    pub fn try_slice<R: IntoSliceItems + IndexCount>(
        &self,
        range: R,
    ) -> Result<TensorBase<ViewData<'a, T>, <L as SliceWith<R, R::Count>>::Layout>, SliceError>
    where
        L: SliceWith<R, R::Count>,
    {
        let (offset_range, sliced_layout) = self.layout.slice_with(range)?;
        Ok(TensorBase {
            data: self.data.slice(offset_range),
            layout: sliced_layout,
        })
    }

    /// Remove all size-one dimensions from this tensor.
    ///
    /// See [`AsView::squeezed`].
    pub fn squeezed(&self) -> TensorView<'a, T>
    where
        L: MutLayout,
    {
        TensorBase {
            data: self.data.view(),
            layout: self.layout.squeezed(),
        }
    }

    /// Divide this tensor into two views along a given axis.
    ///
    /// Returns a `(left, right)` tuple of views, where the `left` view
    /// contains the slice from `[0, mid)` along `axis` and the `right`
    /// view contains the slice from `[mid, end)` along `axis`.
    #[allow(clippy::type_complexity)]
    pub fn split_at(
        &self,
        axis: usize,
        mid: usize,
    ) -> (
        TensorBase<ViewData<'a, T>, L>,
        TensorBase<ViewData<'a, T>, L>,
    )
    where
        L: MutLayout,
    {
        let (left, right) = self.layout.split(axis, mid);
        let (left_offset_range, left_layout) = left;
        let (right_offset_range, right_layout) = right;
        let left_data = self.data.slice(left_offset_range.clone());
        let right_data = self.data.slice(right_offset_range.clone());

        debug_assert_eq!(left_data.len(), left_layout.min_data_len());
        let left_view = TensorBase {
            data: left_data,
            layout: left_layout,
        };

        debug_assert_eq!(right_data.len(), right_layout.min_data_len());
        let right_view = TensorBase {
            data: right_data,
            layout: right_layout,
        };

        (left_view, right_view)
    }

    /// Return a view of this tensor with elements stored in contiguous order.
    ///
    /// If the data is already contiguous, no copy is made, otherwise the
    /// elements are copied into a new buffer in contiguous order.
    pub fn to_contiguous(&self) -> Contiguous<TensorBase<CowData<'a, T>, L>>
    where
        T: Clone,
        L: MutLayout,
    {
        self.to_contiguous_in(GlobalAlloc::new())
    }

    /// Variant of [`to_contiguous`](TensorBase::to_contiguous) which takes
    /// an allocator.
    pub fn to_contiguous_in<A: Alloc>(&self, alloc: A) -> Contiguous<TensorBase<CowData<'a, T>, L>>
    where
        T: Clone,
        L: MutLayout,
    {
        let tensor = if let Some(data) = self.data() {
            TensorBase {
                data: CowData::Borrowed(data.into_storage()),
                layout: self.layout.clone(),
            }
        } else {
            let data = self.to_vec_in(alloc);
            TensorBase {
                data: CowData::Owned(data),
                layout: L::from_shape(self.layout.shape()),
            }
        };
        Contiguous::new(tensor).unwrap()
    }

    /// Return the underlying data as a flat slice if the tensor is contiguous,
    /// or a copy of the data as a flat slice otherwise.
    ///
    /// See [`AsView::to_slice`].
    pub fn to_slice(&self) -> Cow<'a, [T]>
    where
        T: Clone,
    {
        self.data()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(self.to_vec()))
    }

    /// Reverse the order of dimensions in this tensor. See [`AsView::transposed`].
    pub fn transposed(&self) -> TensorBase<ViewData<'a, T>, L>
    where
        L: MutLayout,
    {
        TensorBase {
            data: self.data,
            layout: self.layout.transposed(),
        }
    }

    pub fn try_slice_dyn<R: IntoSliceItems>(
        &self,
        range: R,
    ) -> Result<TensorView<'a, T>, SliceError>
    where
        L: MutLayout,
    {
        let (offset_range, layout) = self.layout.slice_dyn(range.into_slice_items().as_ref())?;
        Ok(TensorBase {
            data: self.data.slice(offset_range),
            layout,
        })
    }

    /// Return a read-only view of this tensor. See [`AsView::view`].
    pub fn view(&self) -> TensorBase<ViewData<'a, T>, L> {
        TensorBase {
            data: self.data,
            layout: self.layout.clone(),
        }
    }

    pub(crate) fn view_ref(&self) -> TensorBase<ViewData<'a, T>, &L> {
        TensorBase {
            data: self.data,
            layout: &self.layout,
        }
    }

    pub fn weakly_checked_view(&self) -> WeaklyCheckedView<ViewData<'a, T>, L> {
        WeaklyCheckedView { base: self.view() }
    }
}

impl<S: Storage, L: Layout> Layout for TensorBase<S, L> {
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

    fn offset(&self, index: Self::Index<'_>) -> Option<usize> {
        self.layout.offset(index)
    }
}

impl<S: Storage, L: Layout + MatrixLayout> MatrixLayout for TensorBase<S, L> {
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

impl<T, S: Storage<Elem = T>, L: Layout + Clone> AsView for TensorBase<S, L> {
    type Elem = T;
    type Layout = L;

    fn iter(&self) -> Iter<'_, T> {
        self.view().iter()
    }

    fn copy_into_slice<'a>(&self, dest: &'a mut [MaybeUninit<T>]) -> &'a [T]
    where
        T: Copy,
    {
        if let Some(data) = self.data() {
            // Safety: `[T]` and `[MaybeUninit<T>]` have same layout.
            let src_uninit = unsafe { std::mem::transmute::<&[T], &[MaybeUninit<T>]>(data) };
            dest.copy_from_slice(src_uninit);
            // Safety: `copy_from_slice` initializes the whole slice or panics
            // if there is a length mismatch.
            unsafe { dest.assume_init() }
        } else {
            copy_into_slice(self.as_dyn(), dest)
        }
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

    #[track_caller]
    fn remove_axis(&mut self, index: usize)
    where
        L: ResizeLayout,
    {
        self.layout.remove_axis(index)
    }

    fn merge_axes(&mut self)
    where
        L: ResizeLayout,
    {
        self.layout.merge_axes()
    }

    fn layout(&self) -> &L {
        &self.layout
    }

    fn map<F, U>(&self, f: F) -> TensorBase<Vec<U>, L>
    where
        F: Fn(&Self::Elem) -> U,
        L: MutLayout,
    {
        self.map_in(GlobalAlloc::new(), f)
    }

    fn map_in<A: Alloc, F, U>(&self, alloc: A, f: F) -> TensorBase<Vec<U>, L>
    where
        F: Fn(&Self::Elem) -> U,
        L: MutLayout,
    {
        let len = self.len();
        let mut buf = alloc.alloc(len);
        if let Some(data) = self.data() {
            // Fast path for contiguous tensors.
            buf.extend(data.iter().map(f));
        } else {
            let dest = &mut buf.spare_capacity_mut()[..len];
            map_into_slice(self.as_dyn(), dest, f);

            // Safety: `map_into` initialized all elements of `dest`.
            unsafe {
                buf.set_len(len);
            }
        };
        TensorBase::from_data(self.shape(), buf)
    }

    fn move_axis(&mut self, from: usize, to: usize)
    where
        L: MutLayout,
    {
        self.layout.move_axis(from, to);
    }

    fn view(&self) -> TensorBase<ViewData<'_, T>, L> {
        TensorBase {
            data: self.data.view(),
            layout: self.layout.clone(),
        }
    }

    // For `get` and `get_unchecked` we override the default implementation in
    // the trait to skip view creation.

    fn get<I: AsIndex<L>>(&self, index: I) -> Option<&Self::Elem> {
        self.offset(index.as_index()).map(|offset| unsafe {
            // Safety: We verified the offset is in-bounds
            self.data.get_unchecked(offset)
        })
    }

    unsafe fn get_unchecked<I: AsIndex<L>>(&self, index: I) -> &T {
        let offset = self.layout.offset_unchecked(index.as_index());
        unsafe { self.data.get_unchecked(offset) }
    }

    fn permute(&mut self, order: Self::Index<'_>)
    where
        L: MutLayout,
    {
        self.layout = self.layout.permuted(order);
    }

    fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_vec_in(GlobalAlloc::new())
    }

    fn to_vec_in<A: Alloc>(&self, alloc: A) -> Vec<T>
    where
        T: Clone,
    {
        let len = self.len();
        let mut buf = alloc.alloc(len);

        if let Some(data) = self.data() {
            buf.extend_from_slice(data);
        } else {
            copy_into_slice(self.as_dyn(), &mut buf.spare_capacity_mut()[..len]);

            // Safety: We initialized `len` elements.
            unsafe { buf.set_len(len) }
        }

        buf
    }

    fn to_shape<SH: IntoLayout>(&self, shape: SH) -> TensorBase<Vec<Self::Elem>, SH::Layout>
    where
        T: Clone,
        L: MutLayout,
    {
        TensorBase {
            data: self.to_vec(),
            layout: self
                .layout
                .reshaped_for_copy(shape)
                .expect("reshape failed"),
        }
    }

    fn transpose(&mut self)
    where
        L: MutLayout,
    {
        self.layout = self.layout.transposed();
    }
}

impl<T, S: Storage<Elem = T>, const N: usize> TensorBase<S, NdLayout<N>> {
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
        let mut result = [T::default(); M];
        for i in 0..M {
            // Safety: `array_offsets` returns valid offsets
            result[i] = unsafe { *self.data.get_unchecked(offsets[i]) };
        }
        result
    }
}

impl<T> TensorBase<Vec<T>, DynLayout> {
    /// Reshape this tensor in place. This is cheap if the tensor is contiguous,
    /// as only the layout will be changed, but requires copying data otherwise.
    #[track_caller]
    pub fn reshape(&mut self, shape: &[usize])
    where
        T: Clone,
    {
        self.reshape_in(GlobalAlloc::new(), shape)
    }

    /// Variant of [`reshape`](TensorBase::reshape) which takes an allocator.
    #[track_caller]
    pub fn reshape_in<A: Alloc>(&mut self, alloc: A, shape: &[usize])
    where
        T: Clone,
    {
        if !self.is_contiguous() {
            self.data = self.to_vec_in(alloc);
        }
        let Ok(layout) = self.layout.reshaped_for_copy(shape) else {
            panic!(
                "element count mismatch reshaping {:?} to {:?}",
                self.shape(),
                shape
            );
        };
        self.layout = layout;
    }
}

impl<'a, T, L: Layout> TensorBase<ViewMutData<'a, T>, L> {
    /// Divide this tensor into two mutable views along a given axis.
    ///
    /// Returns a `(left, right)` tuple of views, where the `left` view
    /// contains the slice from `[0, mid)` along `axis` and the `right`
    /// view contains the slice from `[mid, end)` along `axis`.
    #[allow(clippy::type_complexity)]
    pub fn split_at_mut(
        self,
        axis: usize,
        mid: usize,
    ) -> (
        TensorBase<ViewMutData<'a, T>, L>,
        TensorBase<ViewMutData<'a, T>, L>,
    )
    where
        L: MutLayout,
    {
        let (left, right) = self.layout.split(axis, mid);
        let (left_offset_range, left_layout) = left;
        let (right_offset_range, right_layout) = right;
        let (left_data, right_data) = self
            .data
            .split_mut(left_offset_range.clone(), right_offset_range.clone());

        debug_assert_eq!(left_data.len(), left_layout.min_data_len());
        let left_view = TensorBase {
            data: left_data,
            layout: left_layout,
        };

        debug_assert_eq!(right_data.len(), right_layout.min_data_len());
        let right_view = TensorBase {
            data: right_data,
            layout: right_layout,
        };

        (left_view, right_view)
    }

    /// Consume this view and return a mutable slice, if the tensor is
    /// contiguous.
    pub fn into_slice_mut(self) -> Option<&'a mut [T]> {
        let len = self.layout.min_data_len();
        self.is_contiguous().then(|| {
            // Safety: We verified that the slice is contiguous.
            let slice = unsafe { self.data.to_slice_mut() };
            &mut slice[..len]
        })
    }
}

impl<T, L: MutLayout> FromIterator<T> for TensorBase<Vec<T>, L>
where
    [usize; 1]: AsIndex<L>,
{
    /// Create a new 1D tensor filled with an arithmetic sequence of values
    /// in the range `[start, end)` separated by `step`. If `step` is omitted,
    /// it defaults to 1.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> TensorBase<Vec<T>, L> {
        let data: Vec<T> = iter.into_iter().collect();
        TensorBase::from_data([data.len()].as_index(), data)
    }
}

impl<T, L: MutLayout> From<Vec<T>> for TensorBase<Vec<T>, L>
where
    [usize; 1]: AsIndex<L>,
{
    /// Create a 1D tensor from a vector.
    fn from(vec: Vec<T>) -> Self {
        Self::from_data([vec.len()].as_index(), vec)
    }
}

impl<'a, T, L: MutLayout> From<&'a [T]> for TensorBase<ViewData<'a, T>, L>
where
    [usize; 1]: AsIndex<L>,
{
    /// Create a 1D view from a slice.
    fn from(slice: &'a [T]) -> Self {
        Self::from_data([slice.len()].as_index(), slice)
    }
}

impl<'a, T, L: MutLayout, const N: usize> From<&'a [T; N]> for TensorBase<ViewData<'a, T>, L>
where
    [usize; 1]: AsIndex<L>,
{
    /// Create a 1D view from a slice of known length.
    fn from(slice: &'a [T; N]) -> Self {
        Self::from_data([slice.len()].as_index(), slice.as_slice())
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

    let offset = layout.must_offset(base);
    let stride = layout.stride(dim);
    let mut offsets = [0; M];
    for i in 0..M {
        offsets[i] = offset + i * stride;
    }
    offsets
}

impl<T, S: StorageMut<Elem = T>, const N: usize> TensorBase<S, NdLayout<N>> {
    /// Store an array of `M` elements into successive entries of a tensor along
    /// the `dim` axis.
    ///
    /// See [`TensorBase::get_array`] for more details.
    #[inline]
    pub fn set_array<const M: usize>(&mut self, base: [usize; N], dim: usize, values: [T; M])
    where
        T: Copy,
    {
        let offsets: [usize; M] = array_offsets(&self.layout, base, dim);

        for i in 0..M {
            // Safety: `array_offsets` returns valid offsets.
            unsafe { *self.data.get_unchecked_mut(offsets[i]) = values[i] };
        }
    }
}

impl<T, S: Storage<Elem = T>> TensorBase<S, NdLayout<1>> {
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

impl<T, S: StorageMut<Elem = T>> TensorBase<S, NdLayout<1>> {
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

/// View of a tensor with N dimensions.
pub type NdTensorView<'a, T, const N: usize> = TensorBase<ViewData<'a, T>, NdLayout<N>>;

/// Owned tensor with N dimensions.
pub type NdTensor<T, const N: usize> = TensorBase<Vec<T>, NdLayout<N>>;

/// Mutable view of a tensor with N dimensions.
pub type NdTensorViewMut<'a, T, const N: usize> = TensorBase<ViewMutData<'a, T>, NdLayout<N>>;

/// Owned or borrowed tensor with N dimensions.
///
/// `CowNdTensor`s can be created using [`as_cow`](TensorBase::as_cow) (to
/// borrow) or [`into_cow`](TensorBase::into_cow).
///
/// The name comes from [`std::borrow::Cow`].
pub type CowNdTensor<'a, T, const N: usize> = TensorBase<CowData<'a, T>, NdLayout<N>>;

/// View of a 2D tensor.
pub type Matrix<'a, T = f32> = NdTensorView<'a, T, 2>;

/// Mutable view of a 2D tensor.
pub type MatrixMut<'a, T = f32> = NdTensorViewMut<'a, T, 2>;

/// Owned tensor with a dynamic dimension count.
pub type Tensor<T = f32> = TensorBase<Vec<T>, DynLayout>;

/// View of a tensor with a dynamic dimension count.
pub type TensorView<'a, T = f32> = TensorBase<ViewData<'a, T>, DynLayout>;

/// Mutable view of a tensor with a dynamic dimension count.
pub type TensorViewMut<'a, T = f32> = TensorBase<ViewMutData<'a, T>, DynLayout>;

/// Owned or borrowed tensor with a dynamic dimension count.
///
/// `CowTensor`s can be created using [`as_cow`](TensorBase::as_cow) (to
/// borrow) or [`into_cow`](TensorBase::into_cow).
///
/// The name comes from [`std::borrow::Cow`].
pub type CowTensor<'a, T> = TensorBase<CowData<'a, T>, DynLayout>;

/// Reference-counted tensor with a dynamic dimension count.
///
/// This uses `Arc<Vec<T>>` rather than `Arc<[T]>` as the backing storage. This
/// adds an extra indirection when accessing the data, but it enables cheap
/// conversion between owned and reference-counted tensors.
pub type ArcTensor<T> = TensorBase<Arc<Vec<T>>, DynLayout>;

/// Reference-counted tensor with N dimensions.
///
/// See also the notes for [`ArcTensor`].
pub type ArcNdTensor<T, const N: usize> = TensorBase<Arc<Vec<T>>, NdLayout<N>>;

impl<T, S: Storage<Elem = T>, L: TrustedLayout, I: AsIndex<L>> Index<I> for TensorBase<S, L> {
    type Output = T;

    /// Return the element at a given index.
    ///
    /// Panics if the index is out of bounds along any dimension.
    fn index(&self, index: I) -> &Self::Output {
        let offset = self.layout.must_offset(index.as_index());

        // Safety: `TrustedLayout` guarantees offsets are < `min_data_len`.
        // TensorBase guarantees storage length is >= `min_data_len`.
        unsafe { self.data.get_unchecked(offset) }
    }
}

impl<T, S: StorageMut<Elem = T>, L: TrustedLayout, I: AsIndex<L>> IndexMut<I> for TensorBase<S, L> {
    /// Return the element at a given index.
    ///
    /// Panics if the index is out of bounds along any dimension.
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index = index.as_index();
        let offset = self.layout.must_offset(index);

        // Safety: `TrustedLayout` guarantees offsets are < `min_data_len`.
        // TensorBase guarantees storage length is >= `min_data_len`.
        unsafe { self.data.get_unchecked_mut(offset) }
    }
}

impl<T, S: Storage<Elem = T> + Clone, L: Layout + Clone> Clone for TensorBase<S, L> {
    fn clone(&self) -> TensorBase<S, L> {
        let data = self.data.clone();
        TensorBase {
            data,
            layout: self.layout.clone(),
        }
    }
}

impl<T, S: Storage<Elem = T> + Copy, L: Layout + Copy> Copy for TensorBase<S, L> {}

impl<T: PartialEq, S: Storage<Elem = T>, L: Layout + Clone, V: AsView<Elem = T>> PartialEq<V>
    for TensorBase<S, L>
{
    fn eq(&self, other: &V) -> bool {
        self.shape().as_ref() == other.shape().as_ref() && self.iter().eq(other.iter())
    }
}

impl<T, S: Storage<Elem = T>, const N: usize> From<TensorBase<S, NdLayout<N>>>
    for TensorBase<S, DynLayout>
{
    fn from(tensor: TensorBase<S, NdLayout<N>>) -> Self {
        Self {
            data: tensor.data,
            layout: tensor.layout.into(),
        }
    }
}

impl<T, S1: Storage<Elem = T>, S2: Storage<Elem = T>, const N: usize>
    TryFrom<TensorBase<S1, DynLayout>> for TensorBase<S2, NdLayout<N>>
where
    S1: Into<S2>,
{
    type Error = DimensionError;

    /// Convert a tensor or view with dynamic rank into a static rank one.
    ///
    /// Fails if `value` does not have `N` dimensions.
    fn try_from(value: TensorBase<S1, DynLayout>) -> Result<Self, Self::Error> {
        let layout: NdLayout<N> = value.layout().try_into()?;
        Ok(TensorBase {
            data: value.data.into(),
            layout,
        })
    }
}

/// Trait for scalar (ie. non-array) values.
///
/// This is used to prevent generic types from being inferred as array types
/// in [`TensorBase::from`].
pub trait Scalar {}

macro_rules! impl_scalar {
    ($ty:ty) => {
        impl Scalar for $ty {}
    };
}
impl_scalar!(bool);
impl_scalar!(u8);
impl_scalar!(i8);
impl_scalar!(u16);
impl_scalar!(i16);
impl_scalar!(u32);
impl_scalar!(i32);
impl_scalar!(u64);
impl_scalar!(i64);
impl_scalar!(usize);
impl_scalar!(isize);
impl_scalar!(f32);
impl_scalar!(f64);
impl_scalar!(String);

// The `T: Scalar` bound avoids ambiguity when choosing a `Tensor::from`
// impl for a nested array literal, as it prevents `T` from matching an array
// type.

impl<T: Clone + Scalar, L: MutLayout> From<T> for TensorBase<Vec<T>, L>
where
    [usize; 0]: AsIndex<L>,
{
    /// Construct a scalar tensor from a scalar value.
    fn from(value: T) -> Self {
        Self::from_scalar(value)
    }
}

impl<T: Clone + Scalar, L: MutLayout, const D0: usize> From<[T; D0]> for TensorBase<Vec<T>, L>
where
    [usize; 1]: AsIndex<L>,
{
    /// Construct a 1D tensor from a 1D array.
    fn from(value: [T; D0]) -> Self {
        let data: Vec<T> = value.iter().cloned().collect();
        Self::from_data([D0].as_index(), data)
    }
}

impl<T: Clone + Scalar, L: MutLayout, const D0: usize, const D1: usize> From<[[T; D1]; D0]>
    for TensorBase<Vec<T>, L>
where
    [usize; 2]: AsIndex<L>,
{
    /// Construct a 2D tensor from a nested array.
    fn from(value: [[T; D1]; D0]) -> Self {
        let data: Vec<_> = value.iter().flat_map(|y| y.iter()).cloned().collect();
        Self::from_data([D0, D1].as_index(), data)
    }
}

impl<T: Clone + Scalar, L: MutLayout, const D0: usize, const D1: usize, const D2: usize>
    From<[[[T; D2]; D1]; D0]> for TensorBase<Vec<T>, L>
where
    [usize; 3]: AsIndex<L>,
{
    /// Construct a 3D tensor from a nested array.
    fn from(value: [[[T; D2]; D1]; D0]) -> Self {
        let data: Vec<_> = value
            .iter()
            .flat_map(|y| y.iter().flat_map(|z| z.iter()))
            .cloned()
            .collect();
        Self::from_data([D0, D1, D2].as_index(), data)
    }
}

/// A view of a tensor which does "weak" checking when indexing via
/// `view[<index>]`. This means that it does not bounds-check individual
/// dimensions, but does bounds-check the computed offset.
///
/// This offers a middle-ground between regular indexing, which bounds-checks
/// each index element, and unchecked indexing, which does no bounds-checking
/// at all and is thus unsafe.
pub struct WeaklyCheckedView<S: Storage, L: Layout> {
    base: TensorBase<S, L>,
}

impl<T, S: Storage<Elem = T>, L: Layout> Layout for WeaklyCheckedView<S, L> {
    type Index<'a> = L::Index<'a>;
    type Indices = L::Indices;

    fn ndim(&self) -> usize {
        self.base.ndim()
    }

    fn offset(&self, index: Self::Index<'_>) -> Option<usize> {
        self.base.offset(index)
    }

    fn len(&self) -> usize {
        self.base.len()
    }

    fn shape(&self) -> Self::Index<'_> {
        self.base.shape()
    }

    fn strides(&self) -> Self::Index<'_> {
        self.base.strides()
    }

    fn indices(&self) -> Self::Indices {
        self.base.indices()
    }
}

impl<T, S: Storage<Elem = T>, L: Layout, I: AsIndex<L>> Index<I> for WeaklyCheckedView<S, L> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        let offset = self.base.layout.offset_unchecked(index.as_index());
        unsafe {
            // Safety: See comments in [Storage] trait.
            self.base.data.get(offset).expect("invalid offset")
        }
    }
}

impl<T, S: StorageMut<Elem = T>, L: Layout, I: AsIndex<L>> IndexMut<I> for WeaklyCheckedView<S, L> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let offset = self.base.layout.offset_unchecked(index.as_index());
        unsafe {
            // Safety: See comments in [Storage] trait.
            self.base.data.get_mut(offset).expect("invalid offset")
        }
    }
}

#[cfg(test)]
mod tests;
