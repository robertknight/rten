use std::iter::zip;
use std::marker::PhantomData;
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
///
/// `T` is the element type, `S` is the storage (eg. `&[T]` or `&mut [T]`)
/// and `N` is the number of dimensions.
///
/// This uses patterns from
/// https://lab.whitequark.org/notes/2016-12-13/abstracting-over-mutability-in-rust/
/// to allow the same type to work with owned, borrowed, mutable and immutable
/// element storage.
#[derive(Clone, Copy)]
pub struct NdTensor<T, S: AsRef<[T]>, const N: usize> {
    data: S,
    layout: NdLayout<N>,

    /// Avoids compiler complaining `T` is unused.
    element_type: PhantomData<T>,
}

impl<T, S: AsRef<[T]>, const N: usize> NdTensor<T, S, N> {
    /// Constructs an NdTensorView from a slice.
    ///
    /// Panics if the slice is too short for the dimensions and strides specified.
    pub fn from_slice(
        data: S,
        shape: [usize; N],
        strides: Option<[usize; N]>,
    ) -> NdTensor<T, S, N> {
        // TODO - Check that the strides here do not allow for multiple
        // elements to alias.
        let layout = NdLayout {
            shape,
            strides: strides.unwrap_or(NdLayout::contiguous_strides(shape)),
        };
        assert!(
            data.as_ref().len() >= layout.min_data_len(),
            "Slice is too short"
        );
        NdTensor {
            data,
            layout,
            element_type: PhantomData,
        }
    }
}

// Note: `S` refers to `[T]` here rather than `&[T]` so we can preserve
// liftimes on the result.
impl<'a, T, S: AsRef<[T]> + ?Sized, const N: usize> NdTensor<T, &'a S, N> {
    pub fn data(&self) -> &'a [T] {
        self.data.as_ref()
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, const N: usize> NdTensor<T, S, N> {
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
}

impl<T, S: AsRef<[T]>, const N: usize> Index<[usize; N]> for NdTensor<T, S, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.data.as_ref()[self.layout.offset(index)]
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>, const N: usize> IndexMut<[usize; N]> for NdTensor<T, S, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let offset = self.layout.offset(index);
        &mut self.data.as_mut()[offset]
    }
}

impl<T, S: AsRef<[T]>, const N: usize> NdTensorLayout<N> for NdTensor<T, S, N> {
    fn layout(&self) -> &NdLayout<N> {
        &self.layout
    }
}

/// Provides methods specific to 2D tensors (matrices).
impl<T, S: AsRef<[T]>> NdTensor<T, S, 2> {
    /// Return a new view which transposes the columns and rows.
    pub fn transposed(self) -> Self {
        NdTensor {
            data: self.data,
            layout: self.layout.transposed(),
            element_type: PhantomData,
        }
    }
}

/// Alias for viewing a slice as a 2D matrix.
pub type Matrix<'a, T = f32> = NdTensor<T, &'a [T], 2>;

/// Alias for viewing a mutable slice as a 2D matrix.
pub type MatrixMut<'a, T = f32> = NdTensor<T, &'a mut [T], 2>;
