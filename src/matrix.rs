use std::ops::Index;

/// Provides a view of a slice as a matrix.
#[derive(Copy, Clone)]
pub struct Matrix<'a, T = f32> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
}

impl<'a, T> Matrix<'a, T> {
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// Return a new view which transposes the columns and rows.
    pub fn transposed(self) -> Matrix<'a, T> {
        Matrix {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// Return true if the slice length is valid for the dimensions and strides
    /// of the matrix.
    fn valid(&self) -> bool {
        if self.rows == 0 || self.cols == 0 {
            true // Min slice len is 0, and all slice lengths are >= 0.
        } else {
            let max_offset = (self.rows - 1) * self.row_stride + (self.cols - 1) * self.col_stride;
            self.data.len() > max_offset
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
    ) -> Matrix<'_, T> {
        let (row_stride, col_stride) = strides.unwrap_or((rows, 1));
        let m = Matrix {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        };
        assert!(m.valid(), "Slice is too short");
        m
    }
}

impl<'a, T> Index<[usize; 2]> for Matrix<'a, T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [row, col] = index;
        &self.data[row * self.row_stride + col * self.col_stride]
    }
}
