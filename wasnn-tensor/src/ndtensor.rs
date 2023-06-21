use std::error::Error;
use std::fmt::{Display, Formatter};
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use super::index_iterator::NdIndexIterator;
use super::iterators::{Elements, ElementsMut};
use super::layout::Layout;
use super::overlap::may_have_internal_overlap;
use super::range::SliceItem;
use super::IntoSliceItems;
use super::TensorBase;

/// Describes how to view a linear buffer as an `N`-dimensional array.
#[derive(Clone, Copy)]
pub struct NdLayout<const N: usize> {
    shape: [usize; N],
    strides: [usize; N],
}

/// Specifies whether a tensor or view may have an overlapping layout.
///
/// An overlapping layout is one in which multiple valid indices map to the same
/// offset in storage. To comply with Rust's rules for mutable aliases, views
/// must disallow overlap in tensors/views which can yield mutable element
/// references. They may allow overlap in immutable views.
enum OverlapPolicy {
    AllowOverlap,
    DisallowOverlap,
}

impl<const N: usize> NdLayout<N> {
    /// Return the number of elements in the array.
    #[allow(dead_code)] // Remove when this is used outside of tests
    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Convert a layout with dynamic rank to a layout with a static rank.
    ///
    /// Panics if `l` does not have N dimensions.
    fn from_dyn(l: Layout) -> Self {
        assert!(l.ndim() == N, "Dynamic layout dims != {}", N);
        NdLayout {
            shape: l.shape().try_into().unwrap(),
            strides: l.strides().try_into().unwrap(),
        }
    }

    /// Convert this layout to one with a dynamic rank.
    fn as_dyn(&self) -> Layout {
        Layout::new_with_strides(&self.shape, &self.strides)
    }

    /// Return true if all components of `index` are in-bounds.
    fn index_valid(&self, index: [usize; N]) -> bool {
        let mut valid = true;
        for i in 0..N {
            valid = valid && index[i] < self.shape[i]
        }
        valid
    }

    /// Return the offset in the slice that an index maps to.
    fn offset(&self, index: [usize; N]) -> usize {
        assert!(
            self.index_valid(index),
            "Index {:?} out of bounds for shape {:?}",
            index,
            self.shape
        );
        self.offset_unchecked(index)
    }

    /// Return the offset in the slice that an index maps to, or `None` if it
    /// is out of bounds.
    fn try_offset(&self, index: [usize; N]) -> Option<usize> {
        if !self.index_valid(index) {
            return None;
        }
        Some(self.offset_unchecked(index))
    }

    /// Return the offset in the slice that an index maps to.
    ///
    /// Unlike `offset`, this does not bounds-check elements of `index` against
    /// the corresponding shape. Hence the returned offset may be out of bounds.
    fn offset_unchecked(&self, index: [usize; N]) -> usize {
        let mut offset = 0;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        offset
    }

    /// Return the minimum length required for the element data buffer used
    /// with this layout.
    fn min_data_len(&self) -> usize {
        if self.shape.iter().any(|&size| size == 0) {
            return 0;
        }
        let max_offset: usize = zip(self.shape.iter(), self.strides.iter())
            .map(|(size, stride)| (size - 1) * stride)
            .sum();
        max_offset + 1
    }

    /// Return the strides that a contiguous layout with a given shape would
    /// have.
    fn contiguous_strides(shape: [usize; N]) -> [usize; N] {
        let mut strides = [0; N];
        for i in 0..N {
            strides[i] = shape[i + 1..].iter().product();
        }
        strides
    }

    /// Create a layout with a given shape and a contiguous layout.
    fn from_shape(shape: [usize; N]) -> Self {
        Self {
            shape,
            strides: Self::contiguous_strides(shape),
        }
    }

    /// Create a layout with given shape and strides, intended for use with
    /// data storage of length `data_len`.
    ///
    /// `overlap` determines whether this method will fail if the layout
    /// may have internal overlap.
    fn try_from_shape_and_strides(
        shape: [usize; N],
        strides: Option<[usize; N]>,
        data_len: usize,
        overlap: OverlapPolicy,
    ) -> Result<NdLayout<N>, FromDataError> {
        let layout = NdLayout {
            shape,
            strides: strides.unwrap_or(NdLayout::contiguous_strides(shape)),
        };

        if data_len < layout.min_data_len() {
            return Err(FromDataError::StorageTooShort);
        }

        match overlap {
            OverlapPolicy::DisallowOverlap => {
                if may_have_internal_overlap(&layout.shape, &layout.strides) {
                    return Err(FromDataError::MayOverlap);
                }
            }
            OverlapPolicy::AllowOverlap => {}
        }

        Ok(layout)
    }
}

impl NdLayout<2> {
    fn transposed(self) -> NdLayout<2> {
        NdLayout {
            shape: [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

/// Provides methods for querying the shape and data layout of an [NdTensorView].
pub trait NdTensorLayout<const N: usize> {
    #[doc(hidden)]
    fn layout(&self) -> &NdLayout<N>;

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.layout().len()
    }

    /// Returns true if the array has no elements.
    fn is_empty(&self) -> bool {
        self.layout().len() == 0
    }

    /// Returns an array of the sizes of each dimension.
    fn shape(&self) -> [usize; N] {
        self.layout().shape
    }

    /// Returns the size of the dimension `dim`.
    fn size(&self, dim: usize) -> usize {
        self.layout().shape[dim]
    }

    /// Returns an array of the strides of each dimension.
    fn strides(&self) -> [usize; N] {
        self.layout().strides
    }

    /// Returns the offset between adjacent indices along dimension `dim`.
    fn stride(&self, dim: usize) -> usize {
        self.layout().strides[dim]
    }

    /// Return an iterator over all valid indices in this tensor.
    fn indices(&self) -> NdIndexIterator<N> {
        NdIndexIterator::from_shape(self.shape())
    }
}

/// Provides convenience methods for querying the shape and strides of a matrix.
pub trait MatrixLayout {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row_stride(&self) -> usize;
    fn col_stride(&self) -> usize;
}

impl<NTL: NdTensorLayout<2>> MatrixLayout for NTL {
    fn rows(&self) -> usize {
        self.size(0)
    }

    fn cols(&self) -> usize {
        self.size(1)
    }

    fn row_stride(&self) -> usize {
        self.stride(0)
    }

    fn col_stride(&self) -> usize {
        self.stride(1)
    }
}

/// Provides a view of an array of elements as an N-dimensional tensor.
///
/// `T` is the element type, `S` is the storage and `N` is the number of
/// dimensions. The storage may be owned (eg. a Vec) or a slice.
///
/// ## Notes
///
/// This struct uses patterns from
/// <https://lab.whitequark.org/notes/2016-12-13/abstracting-over-mutability-in-rust/>
/// to support owned, borrowed, mutable and immutable element storage.
#[derive(Clone, Copy)]
pub struct NdTensorBase<T, S: AsRef<[T]>, const N: usize> {
    data: S,
    layout: NdLayout<N>,

    /// Avoids compiler complaining `T` is unused.
    element_type: PhantomData<T>,
}

/// Errors that can occur when constructing an [NdTensorBase] from existing
/// data.
#[derive(PartialEq, Debug)]
pub enum FromDataError {
    /// Some indices will map to offsets that are beyond the end of the storage.
    StorageTooShort,

    /// Some indices will map to the same offset within the storage.
    ///
    /// This error can only occur when the storage is mutable.
    MayOverlap,
}

impl Display for FromDataError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FromDataError::StorageTooShort => write!(f, "Data too short"),
            FromDataError::MayOverlap => write!(f, "May have internal overlap"),
        }
    }
}

impl Error for FromDataError {}

impl<T, S: AsRef<[T]>, const N: usize> NdTensorBase<T, S, N> {
    /// Constructs a tensor from the associated storage type.
    ///
    /// For creating views, prefer [NdTensorBase::from_slice] instead, as it
    /// supports more flexible strides.
    pub fn from_data(
        data: S,
        shape: [usize; N],
        strides: Option<[usize; N]>,
    ) -> Result<NdTensorBase<T, S, N>, FromDataError> {
        NdLayout::try_from_shape_and_strides(
            shape,
            strides,
            data.as_ref().len(),
            // Since this tensor may be mutable, having multiple indices yield
            // the same element reference would lead to violations of Rust's
            // mutable aliasing rules.
            OverlapPolicy::DisallowOverlap,
        )
        .map(|layout| NdTensorBase {
            data,
            layout,
            element_type: PhantomData,
        })
    }

    /// Return the underlying elements, in the order they are stored.
    ///
    /// See [NdTensorBase::to_data] for a variant for [NdTensorView] where
    /// the returned lifetime matches the underlying slice.
    pub fn data(&self) -> &[T] {
        self.data.as_ref()
    }

    /// Return the element at a given index, or `None` if the index is out of
    /// bounds in any dimension.
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        self.layout
            .try_offset(index)
            .and_then(|offset| self.data().get(offset))
    }

    /// Return the element at a given index, without performing any bounds-
    /// checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is valid for the tensor's shape.
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &T {
        self.data()
            .get_unchecked(self.layout.offset_unchecked(index))
    }

    /// Return an immutable view of this tensor.
    pub fn view(&self) -> NdTensorView<T, N> {
        NdTensorView {
            data: self.data.as_ref(),
            layout: self.layout,
            element_type: PhantomData,
        }
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
    pub fn slice<const M: usize, const K: usize, R: IntoSliceItems<K>>(
        &self,
        range: R,
    ) -> NdTensorView<T, M> {
        self.slice_dyn::<M>(&range.into_slice_items())
    }

    /// Return an immutable view of part of this tensor.
    ///
    /// This is like [NdTensorBase::slice] but supports a dynamic number of slice
    /// items.
    pub fn slice_dyn<const M: usize>(&self, range: &[SliceItem]) -> NdTensorView<T, M> {
        let (offset, sliced_layout) = self.layout.as_dyn().slice(range);
        assert!(sliced_layout.ndim() == M, "sliced dims != {}", M);
        NdTensorView {
            data: &self.data.as_ref()[offset..],
            layout: NdLayout::from_dyn(sliced_layout),
            element_type: PhantomData,
        }
    }

    /// Return a view of this tensor with a dynamic dimension count.
    pub fn as_dyn(&self) -> TensorBase<T, &[T]>
    where
        T: Copy,
    {
        TensorBase::new(self.data.as_ref(), &self.layout.as_dyn())
    }

    /// Return an iterator over elements of this tensor.
    pub fn iter(&self) -> Elements<T>
    where
        T: Copy,
    {
        Elements::from_view(&self.as_dyn())
    }

    /// Return a copy of this view that owns its data. For [NdTensorView] this
    /// is different than cloning the view, as that returns a view which has
    /// its own layout, but the same underlying data buffer.
    pub fn to_owned(&self) -> NdTensor<T, N>
    where
        T: Clone,
    {
        NdTensor {
            data: self.data.as_ref().to_vec(),
            layout: self.layout,
            element_type: PhantomData,
        }
    }
}

impl<'a, T, const N: usize> NdTensorBase<T, &'a [T], N> {
    /// Constructs a view from a slice.
    pub fn from_slice(
        data: &'a [T],
        shape: [usize; N],
        strides: Option<[usize; N]>,
    ) -> Result<Self, FromDataError> {
        NdLayout::try_from_shape_and_strides(
            shape,
            strides,
            data.as_ref().len(),
            // Since this view is immutable, having multiple indices yield the
            // same element reference won't violate Rust's aliasing rules.
            OverlapPolicy::AllowOverlap,
        )
        .map(|layout| NdTensorBase {
            data,
            layout,
            element_type: PhantomData,
        })
    }
}

// Note: `S` refers to `[T]` here rather than `&[T]` so we can preserve
// liftimes on the result.
impl<'a, T, S: AsRef<[T]> + ?Sized, const N: usize> NdTensorBase<T, &'a S, N> {
    /// Return the underlying elements of the view.
    ///
    /// This method differs from [NdTensorBase::data] in that the lifetime of the
    /// result is that of the underlying data, rather than the view.
    pub fn to_data(&self) -> &'a [T] {
        self.data.as_ref()
    }

    /// Return a view of this tensor which uses unchecked indexing.
    pub fn unchecked(&self) -> UncheckedNdTensor<T, &'a [T], N> {
        let base = NdTensorBase {
            data: self.data.as_ref(),
            layout: self.layout,
            element_type: PhantomData,
        };
        UncheckedNdTensor { base }
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, const N: usize> NdTensorBase<T, S, N> {
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
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
    /// of dims.
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
        let (offset, sliced_layout) = self.layout.as_dyn().slice(range);
        assert!(sliced_layout.ndim() == M, "sliced dims != {}", M);
        NdTensorViewMut {
            data: &mut self.data.as_mut()[offset..],
            layout: NdLayout::from_dyn(sliced_layout),
            element_type: PhantomData,
        }
    }

    /// Return a mutable view of this tensor which uses unchecked indexing.
    pub fn unchecked_mut(&mut self) -> UncheckedNdTensor<T, &mut [T], N> {
        let base = NdTensorBase {
            data: self.data.as_mut(),
            layout: self.layout,
            element_type: PhantomData,
        };
        UncheckedNdTensor { base }
    }

    /// Return a view of this tensor with a dynamic dimension count.
    pub fn as_dyn_mut(&mut self) -> TensorBase<T, &mut [T]>
    where
        T: Copy,
    {
        TensorBase::new(self.data.as_mut(), &self.layout.as_dyn())
    }

    /// Return a mutable iterator over elements of this tensor.
    pub fn iter_mut(&mut self) -> ElementsMut<T>
    where
        T: Copy,
    {
        ElementsMut::new(self.data.as_mut(), &self.layout.as_dyn())
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

impl<T, S: AsRef<[T]>, const N: usize> NdTensorLayout<N> for NdTensorBase<T, S, N> {
    fn layout(&self) -> &NdLayout<N> {
        &self.layout
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

/// N-dimensional tensor.
pub type NdTensor<T, const N: usize> = NdTensorBase<T, Vec<T>, N>;

/// N-dimensional view of a slice of data.
///
/// See [NdTensorBase] for available methods.
pub type NdTensorView<'a, T, const N: usize> = NdTensorBase<T, &'a [T], N>;

/// Mutable N-dimensional view of a slice of data.
///
/// See [NdTensorBase] for available methods.
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

#[cfg(test)]
mod tests {
    use super::{
        FromDataError, MatrixLayout, NdTensor, NdTensorLayout, NdTensorView, NdTensorViewMut,
    };
    use crate::TensorLayout;

    /// Return elements of `tensor` in their logical order.
    fn tensor_elements<T: Copy, const N: usize>(tensor: NdTensorView<T, N>) -> Vec<T> {
        tensor.iter().collect()
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
        let elements: Vec<_> = dyn_view.iter().collect();
        assert_eq!(elements, &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
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
    fn test_ndtensor_get() {
        let tensor = NdTensor::<i32, 3>::zeros([5, 10, 15]);

        assert_eq!(tensor.get([0, 0, 0]), Some(&0));
        assert_eq!(tensor.get([4, 9, 14]), Some(&0));
        assert_eq!(tensor.get([5, 9, 14]), None);
        assert_eq!(tensor.get([4, 10, 14]), None);
        assert_eq!(tensor.get([4, 9, 15]), None);
    }

    #[test]
    fn test_ndtensor_get_unchecked() {
        let tensor = NdTensor::<i32, 3>::zeros([5, 10, 15]);
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
        let elements: Vec<_> = tensor.iter().collect();
        assert_eq!(elements, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_ndtensor_iter_mut() {
        let mut tensor = NdTensor::<i32, 2>::zeros([2, 2]);
        tensor
            .iter_mut()
            .enumerate()
            .for_each(|(i, el)| *el = i as i32);
        let elements: Vec<_> = tensor.iter().collect();
        assert_eq!(elements, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_ndtensor_to_owned() {
        let data = vec![1., 2., 3., 4.];
        let view = NdTensorView::<f32, 2>::from_slice(&data, [2, 2], None).unwrap();
        let owned = view.to_owned();
        assert_eq!(owned.shape(), view.shape());
        assert_eq!(owned.strides(), view.strides());
        assert_eq!(owned.data(), view.data());
    }

    #[test]
    fn test_ndtensor_transposed() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::<i32, 2>::from_slice(&data, [2, 2], None).unwrap();
        assert_eq!(tensor_elements(view), &[1, 2, 3, 4]);
        let view = view.transposed();
        assert_eq!(tensor_elements(view), &[1, 3, 2, 4]);
    }

    #[test]
    fn test_ndtensor_slice() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let view = NdTensorView::<i32, 2>::from_slice(&data, [4, 4], None).unwrap();
        let slice: NdTensorView<_, 2> = view.slice([1..3, 1..3]);
        assert_eq!(tensor_elements(slice), &[6, 7, 10, 11]);
    }

    #[test]
    #[should_panic(expected = "sliced dims != 3")]
    fn test_ndtensor_slice_wrong_dims() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::<i32, 2>::from_slice(&data, [2, 2], None).unwrap();
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
        let mat = NdTensorView::<f32, 2>::from_slice(&data, [2, 2], None).unwrap();
        assert_eq!(mat.data(), data);
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.row_stride(), 2);
        assert_eq!(mat.col_stride(), 1);
    }
}
