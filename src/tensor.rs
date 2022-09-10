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
        self.shape[dim..].iter().fold(1, |stride, n| stride * n)
    }

    fn offset3(&self, index: [usize; 3]) -> usize {
        let shape = &self.shape;
        let stride_1 = shape[2];
        let stride_0 = shape[1] * stride_1;
        let offset = index[0] * stride_0 + index[1] * stride_1 + index[2];

        if offset > self.data.len() {
            panic!("Computed offset for 3-index {:?} is invalid", index);
        }

        offset
    }

    fn offset4(&self, index: [usize; 4]) -> usize {
        let shape = &self.shape;
        let stride_2 = shape[3];
        let stride_1 = shape[2] * stride_2;
        let stride_0 = shape[1] * stride_1;
        let offset = index[0] * stride_0 + index[1] * stride_1 + index[2] * stride_2 + index[3];

        if offset > self.data.len() {
            panic!("Computed offset for 4-index {:?} is invalid", index);
        }

        offset
    }
}

impl Index<[usize; 1]> for Tensor {
    type Output = f32;
    fn index(&self, index: [usize; 1]) -> &Self::Output {
        &self.data[index[0]]
    }
}

impl IndexMut<[usize; 1]> for Tensor {
    fn index_mut(&mut self, index: [usize; 1]) -> &mut Self::Output {
        &mut self.data[index[0]]
    }
}

impl Index<[usize; 3]> for Tensor {
    type Output = f32;

    fn index(&self, index: [usize; 3]) -> &Self::Output {
        &self.data[self.offset3(index)]
    }
}

impl IndexMut<[usize; 3]> for Tensor {
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        let offset = self.offset3(index);
        &mut self.data[offset]
    }
}

impl Index<[usize; 4]> for Tensor {
    type Output = f32;

    fn index(&self, index: [usize; 4]) -> &Self::Output {
        &self.data[self.offset4(index)]
    }
}

impl IndexMut<[usize; 4]> for Tensor {
    fn index_mut(&mut self, index: [usize; 4]) -> &mut Self::Output {
        let offset = self.offset4(index);
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
