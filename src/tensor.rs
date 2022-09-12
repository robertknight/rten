use std::ops::{Index, IndexMut};

use crate::rng::XorShiftRNG;

/// n-dimensional array
pub struct Tensor {
    /// The underlying buffer of elements
    pub data: Vec<f32>,

    /// The size of each dimension of the array
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Return a copy of this tensor with each element replaced by `f(element)`
    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: FnMut(&f32) -> f32,
    {
        let data = self.data.iter().map(f).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn clone(&self) -> Tensor {
        let data = self.data.clone();
        let shape = self.shape.clone();
        Tensor { data, shape }
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
            if index[i] > self.shape[i] {
                panic!("Invalid index {} for dim {}", index[i], i);
            }
            offset += index[i] * self.stride(i)
        }
        offset
    }
}

impl<const N: usize> Index<[usize; N]> for Tensor {
    type Output = f32;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.data[self.offset(index)]
    }
}

impl<const N: usize> IndexMut<[usize; N]> for Tensor {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let offset = self.offset(index);
        &mut self.data[offset]
    }
}

/// Create a new tensor with all values set to 0.
pub fn zero_tensor(shape: Vec<usize>) -> Tensor {
    let mut data = Vec::new();
    let n_elts = shape.iter().fold(1, |elts, dim| elts * dim);
    data.resize(n_elts, 0.0f32);
    Tensor { data, shape }
}

/// Create a new tensor filled with random values supplied by `rng`.
pub fn random_tensor(shape: Vec<usize>, rng: &mut XorShiftRNG) -> Tensor {
    let mut t = zero_tensor(shape);
    t.data.fill_with(|| rng.next_f32());
    t
}

/// Create a new tensor with a given shape and values
pub fn from_data(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor { data, shape }
}

/// Return dimensions of a 3D tensor as a tuple
pub fn dims3(x: &Tensor) -> (usize, usize, usize) {
    let shape = &x.shape;
    if shape.len() != 3 {
        panic!("Expected tensor to have 3 dimensions");
    }
    (shape[0], shape[1], shape[2])
}

/// Return dimensions of a 4D tensor as a tuple
pub fn dims4(x: &Tensor) -> (usize, usize, usize, usize) {
    let shape = &x.shape;
    if shape.len() != 4 {
        panic!("Expected tensor to have 4 dimensions");
    }
    (shape[0], shape[1], shape[2], shape[3])
}

#[cfg(test)]
mod tests {
    use crate::tensor::{zero_tensor, Tensor};

    #[test]
    fn test_stride() {
        let x = zero_tensor(vec![2, 5, 7, 3]);
        assert_eq!(x.stride(3), 1);
        assert_eq!(x.stride(2), 3);
        assert_eq!(x.stride(1), 7 * 3);
        assert_eq!(x.stride(0), 5 * 7 * 3);
    }

    #[test]
    fn test_index() {
        let mut x = zero_tensor(vec![2, 2]);

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
        let mut x = zero_tensor(vec![2, 2]);

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
        let mut x = zero_tensor(vec![2, 2]);
        x[[2, 0]];
    }

    #[test]
    #[should_panic]
    fn test_index_panics_if_wrong_dim_count() {
        let mut x = zero_tensor(vec![2, 2]);
        x[[0, 0, 0]];
    }
}
