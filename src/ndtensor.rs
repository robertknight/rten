use std::error::Error;
use std::fmt::{Display, Formatter};
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

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
    fn len(&self) -> usize {
        self.shape.iter().product()
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
        assert!(self.index_valid(index), "Index is out of bounds");
        let mut offset = 0;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        offset
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

    /// Returns true if a tensor with this layout would have no elements.
    fn is_empty(&self) -> bool {
        self.shape.iter().any(|&size| size == 0)
    }

    /// Return true if multiple indices may map to the same offset.
    ///
    /// Determining whether arbitrary shapes and strides will overlap is
    /// difficult [1][2] so this method is conservative. It verifies that, after
    /// sorting dimensions in order of increasing stride, each dimension's
    /// stride is larger than the maximum offset that is reachable by indexing
    /// the previous dimensions. This correctly reports that there is no overlap
    /// for layouts that are contiguous or produced by slicing other
    /// non-overlapping layouts. However it is possible to construct
    /// combinations of shapes and strides for which no two indicies map to the
    /// same offset, but for which this method returns true. For example when
    /// `shape == [4, 4]` and `strides == [3, 4]` the offsets are `[0, 3, 4, 6,
    /// 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21]`. The maximum offset for
    /// the dimension with the smallest stride is `(4-1)*3 == 9`, which is
    /// greater than the next-smallest stride. Hence this method will report
    /// there may be overlap, even though there is not.
    ///
    /// [1] See https://github.com/numpy/numpy/blob/main/numpy/core/src/common/mem_overlap.c
    ///     and in particular references to internal overlap.
    /// [2] See also references to "memory overlap" in PyTorch source and
    ///     issues.
    fn may_have_internal_overlap(&self) -> bool {
        if self.is_empty() {
            return false;
        }

        // Sort dimensions in order of increasing stride.
        let mut stride_shape = [(0, 0); N];
        for i in 0..N {
            stride_shape[i] = (self.strides[i], self.shape[i])
        }
        stride_shape.sort_unstable();

        // Verify that the stride for each dimension fully "steps over" the
        // previous dimension.
        let mut max_offset = 0;
        for (stride, shape) in stride_shape {
            if stride <= max_offset {
                return true;
            }
            max_offset += (shape - 1) * stride;
        }
        false
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
                if layout.may_have_internal_overlap() {
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
/// This uses patterns from
/// https://lab.whitequark.org/notes/2016-12-13/abstracting-over-mutability-in-rust/
/// to allow the same type to work with owned, borrowed, mutable and immutable
/// element storage.
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
    pub fn data(&self) -> &'a [T] {
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

    /// Return a mutable view of this tensor which uses unchecked indexing.
    pub fn unchecked_mut(&mut self) -> UncheckedNdTensor<T, &mut [T], N> {
        let base = NdTensorBase {
            data: self.data.as_mut(),
            layout: self.layout,
            element_type: PhantomData,
        };
        UncheckedNdTensor { base }
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
    use crate::ndtensor::{FromDataError, MatrixLayout, NdTensorLayout, NdTensorView};

    /// Return elements of `matrix` in their logical order.
    ///
    /// TODO - Replace this once generic iteration is implemented for NdTensorBase.
    fn matrix_elements<T: Copy>(matrix: NdTensorView<T, 2>) -> Vec<T> {
        let mut result = Vec::with_capacity(matrix.len());
        for row in 0..matrix.size(0) {
            for col in 0..matrix.size(1) {
                result.push(matrix[[row, col]]);
            }
        }
        result
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
    fn test_ndtensor_transposed() {
        let data = vec![1, 2, 3, 4];
        let view = NdTensorView::<i32, 2>::from_slice(&data, [2, 2], None).unwrap();
        assert_eq!(matrix_elements(view), &[1, 2, 3, 4]);
        let view = view.transposed();
        assert_eq!(matrix_elements(view), &[1, 3, 2, 4]);
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
