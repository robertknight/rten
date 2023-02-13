use std::iter::zip;
use std::ops::{Index, IndexMut};

/// Describes how to view a slice as an `N`-dimensional array.
#[derive(Clone, Copy)]
struct Layout<const N: usize> {
    shape: [usize; N],
    strides: [usize; N],
}

impl Layout<2> {
    fn transposed(self) -> Layout<2> {
        Layout {
            shape: [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

impl<const N: usize> Layout<N> {
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
}

/// Provides a view of a slice as a matrix.
#[derive(Clone, Copy)]
pub struct Matrix<'a, T = f32> {
    data: &'a [T],
    layout: Layout<2>,
}

impl<'a, T> Matrix<'a, T> {
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn rows(&self) -> usize {
        self.layout.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.layout.shape[1]
    }

    pub fn row_stride(&self) -> usize {
        self.layout.strides[0]
    }

    pub fn col_stride(&self) -> usize {
        self.layout.strides[1]
    }

    /// Return a new view which transposes the columns and rows.
    pub fn transposed(self) -> Matrix<'a, T> {
        Matrix {
            data: self.data,
            layout: self.layout.transposed(),
        }
    }

    /// Constructs a Matrix from a slice.
    ///
    /// `strides` specifies the row and column strides or (rows, 1) (ie. row-
    /// major layout) if None.
    ///
    /// Panics if the slice is too short for the dimensions and strides specified.
    pub fn from_slice(
        data: &'a [T],
        rows: usize,
        cols: usize,
        strides: Option<(usize, usize)>,
    ) -> Matrix<'a, T> {
        let (row_stride, col_stride) = strides.unwrap_or((rows, 1));
        let layout = Layout {
            shape: [rows, cols],
            strides: [row_stride, col_stride],
        };
        assert!(data.len() >= layout.max_offset(), "Slice is too short");
        Matrix { data, layout }
    }
}

impl<'a, T> Index<[usize; 2]> for Matrix<'a, T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[self.layout.offset(index)]
    }
}

pub struct MatrixMut<'a, T = f32> {
    data: &'a mut [T],
    layout: Layout<2>,
}

impl<'a, T> MatrixMut<'a, T> {
    pub fn data(&mut self) -> &mut [T] {
        self.data
    }

    pub fn rows(&self) -> usize {
        self.layout.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.layout.shape[1]
    }

    pub fn row_stride(&self) -> usize {
        self.layout.strides[0]
    }

    pub fn col_stride(&self) -> usize {
        self.layout.strides[1]
    }

    /// Constructs a Matrix from a slice.
    ///
    /// `strides` specifies the row and column strides or (rows, 1) (ie. row-
    /// major layout) if None.
    ///
    /// Panics if the slice is too short for the dimensions and strides specified.
    pub fn from_slice(
        data: &'a mut [T],
        rows: usize,
        cols: usize,
        strides: Option<(usize, usize)>,
    ) -> MatrixMut<'_, T> {
        let (row_stride, col_stride) = strides.unwrap_or((rows, 1));
        let layout = Layout {
            shape: [rows, cols],
            strides: [row_stride, col_stride],
        };
        assert!(data.len() >= layout.max_offset(), "Slice is too short");
        MatrixMut { data, layout }
    }
}

impl<'a, T> Index<[usize; 2]> for MatrixMut<'a, T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[self.layout.offset(index)]
    }
}

impl<'a, T> IndexMut<[usize; 2]> for MatrixMut<'a, T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let offset = self.layout.offset(index);
        &mut self.data[offset]
    }
}
