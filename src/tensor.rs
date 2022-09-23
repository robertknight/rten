use std::ops::{Index, IndexMut};

use crate::rng::XorShiftRNG;

/// n-dimensional array
pub struct Tensor<T: Copy = f32> {
    /// The underlying buffer of elements
    data: Vec<T>,

    /// The size of each dimension of the array
    shape: Vec<usize>,
}

impl<T: Copy> Tensor<T> {
    /// Return a copy of this tensor with each element replaced by `f(element)`
    pub fn map<F>(&self, f: F) -> Tensor<T>
    where
        F: FnMut(&T) -> T,
    {
        let data = self.data.iter().map(f).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn clone(&self) -> Tensor<T> {
        let data = self.data.clone();
        let shape = self.shape.clone();
        Tensor { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the number of elements between successive entries in the `dim`
    /// dimension.
    pub fn stride(&self, dim: usize) -> usize {
        if dim == self.shape.len() - 1 {
            1
        } else {
            self.shape[dim + 1..].iter().fold(1, |stride, n| stride * n)
        }
    }

    /// Return the offset of an element that corresponds to a given index.
    ///
    /// The length of `index` must match the tensor's dimension count.
    pub fn offset<const N: usize>(&self, index: [usize; N]) -> usize {
        let shape = &self.shape;
        if shape.len() != N {
            panic!(
                "Cannot access {} dim tensor with {} dim index",
                shape.len(),
                N
            );
        }
        let mut offset = 0;
        for i in 0..N {
            if index[i] >= self.shape[i] {
                panic!("Invalid index {} for dim {}", index[i], i);
            }
            offset += index[i] * self.stride(i)
        }
        offset
    }

    /// Return the shape of this tensor as a fixed-sized array.
    ///
    /// The tensor's dimension count must match `N`.
    pub fn dims<const N: usize>(&self) -> [usize; N] {
        if self.shape.len() != N {
            panic!(
                "Cannot extract {} dim tensor as {} dim array",
                self.shape.len(),
                N
            );
        }
        self.shape[..].try_into().unwrap()
    }

    /// Return a view of a subset of the data in this tensor.
    ///
    /// This provides faster indexing, at the cost of not bounds-checking
    /// individual dimensions, although generated offsets into the data buffer
    /// are still checked.
    ///
    /// N specifies the number of dimensions used for indexing into the view
    /// and `base` specifies a fixed index to add to all indexes. `base` must
    /// have the same number of dimensions as this tensor. N can be the same
    /// or less. If less, it refers to the last N dimensions.
    pub fn unchecked_view<const B: usize, const N: usize>(
        &self,
        base: [usize; B],
    ) -> UncheckedView<T, N> {
        let offset = self.offset(base);
        let mut strides = [0; N];
        for i in 0..N {
            strides[i] = self.stride(self.shape.len() - N + i);
        }
        let mut shape = [0; N];
        for i in 0..N {
            shape[i] = self.shape[self.shape.len() - N + i];
        }
        UncheckedView {
            tensor: self,
            offset,
            strides,
        }
    }
}

impl<const N: usize, T: Copy> Index<[usize; N]> for Tensor<T> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.data[self.offset(index)]
    }
}

impl<const N: usize, T: Copy> IndexMut<[usize; N]> for Tensor<T> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let offset = self.offset(index);
        &mut self.data[offset]
    }
}

pub struct UncheckedView<'a, T: Copy, const N: usize> {
    tensor: &'a Tensor<T>,
    offset: usize,
    strides: [usize; N],
}

impl<'a, const N: usize, T: Copy> Index<[usize; N]> for UncheckedView<'a, T, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let mut offset = self.offset;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        &self.tensor.data[offset]
    }
}

/// Create a new tensor with all values set to 0.
pub fn zero_tensor<T: Copy + Default>(shape: Vec<usize>) -> Tensor<T> {
    let mut data = Vec::new();
    let n_elts = shape.iter().fold(1, |elts, dim| elts * dim);
    data.resize(n_elts, T::default());
    Tensor { data, shape }
}

/// Create a new tensor filled with random values supplied by `rng`.
pub fn random_tensor(shape: Vec<usize>, rng: &mut XorShiftRNG) -> Tensor {
    let mut t = zero_tensor(shape);
    t.data.fill_with(|| rng.next_f32());
    t
}

/// Create a new tensor with a given shape and values
pub fn from_data<T: Copy>(shape: Vec<usize>, data: Vec<T>) -> Tensor<T> {
    Tensor { data, shape }
}

#[cfg(test)]
mod tests {
    use crate::tensor::zero_tensor;

    #[test]
    fn test_stride() {
        let x = zero_tensor::<f32>(vec![2, 5, 7, 3]);
        assert_eq!(x.stride(3), 1);
        assert_eq!(x.stride(2), 3);
        assert_eq!(x.stride(1), 7 * 3);
        assert_eq!(x.stride(0), 5 * 7 * 3);
    }

    #[test]
    fn test_index() {
        let mut x = zero_tensor::<f32>(vec![2, 2]);

        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        x.data[3] = 4.0;

        assert_eq!(x[[0, 0]], 1.0);
        assert_eq!(x[[0, 1]], 2.0);
        assert_eq!(x[[1, 0]], 3.0);
        assert_eq!(x[[1, 1]], 4.0);
    }

    #[test]
    fn test_index_mut() {
        let mut x = zero_tensor::<f32>(vec![2, 2]);

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
        let x = zero_tensor::<f32>(vec![2, 2]);
        x[[2, 0]];
    }

    #[test]
    #[should_panic]
    fn test_index_panics_if_wrong_dim_count() {
        let x = zero_tensor::<f32>(vec![2, 2]);
        x[[0, 0, 0]];
    }

    #[test]
    fn test_dims() {
        let x = zero_tensor::<f32>(vec![10, 5, 3, 7]);
        let [i, j, k, l] = x.dims();

        assert_eq!(i, 10);
        assert_eq!(j, 5);
        assert_eq!(k, 3);
        assert_eq!(l, 7);
    }

    #[test]
    #[should_panic]
    fn test_dims_panics_if_wrong_array_length() {
        let x = zero_tensor::<f32>(vec![10, 5, 3, 7]);
        let [_i, _j, _k] = x.dims();
    }
}
