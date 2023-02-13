use std::iter::zip;
use std::ops::{Index, IndexMut};

/// Describes how to view a slice as an `N`-dimensional array.
#[derive(Clone, Copy)]
pub struct NdLayout<const N: usize> {
    shape: [usize; N],
    strides: [usize; N],
}

impl NdLayout<2> {
    fn transposed(self) -> NdLayout<2> {
        NdLayout {
            shape: [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

impl<const N: usize> NdLayout<N> {
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

    /// Return the maximum index in the slice that any valid index will map to.
    fn max_offset(&self) -> usize {
        if self.shape.iter().any(|&size| size == 0) {
            return 0;
        }
        zip(self.shape.iter(), self.strides.iter())
            .map(|(size, stride)| (size - 1) * stride)
            .sum()
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
}

/// Provides methods for querying the shape and data layout of an [NdTensorView].
pub trait NdTensorLayout<const N: usize> {
    #[doc(hidden)]
    fn layout(&self) -> &NdLayout<N>;

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

/// Provides a view of a slice as an N-dimensional tensor.
#[derive(Clone, Copy)]
pub struct NdTensorView<'a, T, const N: usize> {
    data: &'a [T],
    layout: NdLayout<N>,
}

impl<'a, T, const N: usize> NdTensorView<'a, T, N> {
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// Constructs an NdTensorView from a slice.
    ///
    /// Panics if the slice is too short for the dimensions and strides specified.
    pub fn from_slice(
        data: &'a [T],
        shape: [usize; N],
        strides: Option<[usize; N]>,
    ) -> NdTensorView<'a, T, N> {
        let layout = NdLayout {
            shape,
            strides: strides.unwrap_or(NdLayout::contiguous_strides(shape)),
        };
        assert!(data.len() >= layout.max_offset(), "Slice is too short");
        NdTensorView { data, layout }
    }
}

impl<'a, T, const N: usize> Index<[usize; N]> for NdTensorView<'a, T, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.data[self.layout.offset(index)]
    }
}

impl<'a, T, const N: usize> NdTensorLayout<N> for NdTensorView<'a, T, N> {
    fn layout(&self) -> &NdLayout<N> {
        &self.layout
    }
}

/// Provides methods specific to 2D tensors (matrices).
impl<'a, T> NdTensorView<'a, T, 2> {
    /// Return a new view which transposes the columns and rows.
    pub fn transposed(self) -> Self {
        NdTensorView {
            data: self.data,
            layout: self.layout.transposed(),
        }
    }
}

/// Provides a view of a mutable slice as an N-dimensional tensor.
pub struct NdTensorViewMut<'a, T, const N: usize> {
    data: &'a mut [T],
    layout: NdLayout<N>,
}

impl<'a, T, const N: usize> NdTensorViewMut<'a, T, N> {
    pub fn data(&mut self) -> &mut [T] {
        self.data
    }

    /// Constructs an NdTensorViewMut from a slice.
    ///
    /// Panics if the slice is too short for the dimensions and strides specified.
    pub fn from_slice(data: &'a mut [T], shape: [usize; N], strides: Option<[usize; N]>) -> Self {
        let layout = NdLayout {
            shape,
            strides: strides.unwrap_or(NdLayout::contiguous_strides(shape)),
        };
        assert!(data.len() >= layout.max_offset(), "Slice is too short");
        Self { data, layout }
    }
}

impl<'a, T, const N: usize> NdTensorLayout<N> for NdTensorViewMut<'a, T, N> {
    fn layout(&self) -> &NdLayout<N> {
        &self.layout
    }
}

impl<'a, T, const N: usize> Index<[usize; N]> for NdTensorViewMut<'a, T, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.data[self.layout.offset(index)]
    }
}

impl<'a, T, const N: usize> IndexMut<[usize; N]> for NdTensorViewMut<'a, T, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let offset = self.layout.offset(index);
        &mut self.data[offset]
    }
}

/// Alias for a 2D tensor view.
pub type Matrix<'a, T = f32> = NdTensorView<'a, T, 2>;

/// Alias for a mutable 2D tensor view.
pub type MatrixMut<'a, T = f32> = NdTensorViewMut<'a, T, 2>;
