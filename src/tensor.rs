use std::ops::{Index, IndexMut};

use crate::rng::XorShiftRNG;

/// n-dimensional array
pub struct Tensor<T: Copy = f32> {
    /// The underlying buffer of elements
    data: Vec<T>,

    /// The offset in the buffer of the first element
    base: usize,

    /// The size of each dimension of the array
    shape: Vec<usize>,

    /// The stride of each dimension of the array
    strides: Vec<usize>,
}

impl<T: Copy> Tensor<T> {
    /// Return a copy of this tensor with each element replaced by `f(element)`
    pub fn map<F>(&self, f: F) -> Tensor<T>
    where
        F: FnMut(T) -> T,
    {
        let data = self.elements().map(f).collect();
        Tensor {
            data,
            base: 0,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    pub fn clone(&self) -> Tensor<T> {
        let data = self.data.clone();
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        Tensor {
            data,
            base: self.base,
            shape,
            strides,
        }
    }

    /// Clone this tensor with a new shape. The new shape must have the same
    /// total number of elements as the existing shape. See `reshape`.
    pub fn clone_with_shape(&self, shape: &[usize]) -> Tensor<T> {
        let mut clone = self.clone();
        clone.reshape(shape);
        clone
    }

    /// Return the total number of elements in this tensor.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Clip the size of a dimension to `new_size`. The new size must be <=
    /// the previous size.
    ///
    /// This is a fast operation since it just alters the size of the dimension,
    /// but retains the existing strides.
    pub fn resize_dim(&mut self, dim: usize, new_size: usize) {
        if new_size > self.shape[dim] {
            panic!("New size must be <= old size");
        }
        self.shape[dim] = new_size;
    }

    /// Return a contiguous slice of `len` elements starting at `index`.
    /// `len` must be less than or equal to the size of the last dimension.
    ///
    /// Using a slice can allow for very efficient access to a range of elements
    /// in a single row or column (or whatever the last dimension represents).
    pub fn last_dim_slice<const N: usize>(&self, index: [usize; N], len: usize) -> &[T] {
        let offset = self.offset(index);
        &self.data[offset..offset + len]
    }

    /// Similar to `last_dim_slice`, but returns a mutable slice.
    pub fn last_dim_slice_mut<const N: usize>(
        &mut self,
        index: [usize; N],
        len: usize,
    ) -> &mut [T] {
        let offset = self.offset(index);
        &mut self.data[offset..offset + len]
    }

    /// Return the underlying element buffer for this tensor.
    ///
    /// If the tensor is contiguous, the buffer will contain the same elements
    /// in the same order as yielded by `elements`.
    pub fn data(&self) -> &[T] {
        &self.data[self.base..]
    }

    /// Return the underlying element buffer for this tensor.
    ///
    /// See notes for `data` about the ordering and validity of elements.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data[self.base..]
    }

    /// Return a slice of the sizes of each dimension.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return true if the logical order of elements in this tensor matches the
    /// order in which they are stored in the underlying buffer, and there are
    /// no gaps.
    pub fn is_contiguous(&self) -> bool {
        if self.strides[self.shape.len() - 1] != 1 {
            return false;
        }

        for dim in (0..self.shape.len() - 2).rev() {
            if self.strides[dim] != self.shape[dim + 1] * self.strides[dim + 1] {
                return false;
            }
        }

        true
    }

    /// Return an iterator over elements of this tensor, in their logical order.
    pub fn elements(&self) -> Elements<T> {
        Elements::new(self)
    }

    /// Update the shape of the tensor without altering the data layout.
    ///
    /// The total number of elements for the new shape must be the same as the
    /// existing shape.
    pub fn reshape(&mut self, shape: &[usize]) {
        let len: usize = shape.iter().product();
        let current_len = self.len();

        if len != current_len {
            panic!("New shape must have same total elements as current shape");
        }

        self.shape = shape.into();

        // TODO - Handle non-default strides
        self.strides = strides_for_shape(&shape);
    }

    /// Return the number of elements between successive entries in the `dim`
    /// dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
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
        let mut offset = self.base;
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
        UncheckedView {
            tensor: self,
            offset,
            strides: self.strides[self.shape.len() - N..].try_into().unwrap(),
        }
    }

    /// Return a mutable view of a subset of the data in this tensor.
    ///
    /// This is the same as `unchecked_view` except that the returned view can
    /// be used to modify elements.
    pub fn unchecked_view_mut<const B: usize, const N: usize>(
        &mut self,
        base: [usize; B],
    ) -> UncheckedViewMut<T, N> {
        let offset = self.offset(base);
        let strides = self.strides[self.shape.len() - N..].try_into().unwrap();
        UncheckedViewMut {
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

pub struct UncheckedViewMut<'a, T: Copy, const N: usize> {
    tensor: &'a mut Tensor<T>,
    offset: usize,
    strides: [usize; N],
}

impl<'a, const N: usize, T: Copy> Index<[usize; N]> for UncheckedViewMut<'a, T, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let mut offset = self.offset;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        &self.tensor.data[offset]
    }
}

impl<'a, const N: usize, T: Copy> IndexMut<[usize; N]> for UncheckedViewMut<'a, T, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let mut offset = self.offset;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        &mut self.tensor.data[offset]
    }
}

struct ElementsDim {
    /// Current index for this dimension
    index: usize,

    /// Maximum index value for this dimension
    max_index: usize,

    /// Amount to increase offset by every time index is incremented in this
    /// dimension
    stride: usize,
}

/// Iterator over elements of a tensor
pub struct Elements<'a, T: Copy> {
    /// Remaining elements to visit
    len: usize,

    /// Index of next element within each dimension. The stride and max value
    /// of each dimension are also copied into this struct for faster access
    /// during iteration.
    dims: Vec<ElementsDim>,

    /// Offset of next element to return in `data`
    offset: usize,

    /// Data buffer of the tensor
    data: &'a [T],
}

impl<'a, T: Copy> Elements<'a, T> {
    fn new(tensor: &'a Tensor<T>) -> Elements<'a, T> {
        let dims = tensor
            .shape
            .iter()
            .enumerate()
            .map(|(dim, &len)| ElementsDim {
                index: 0,
                max_index: if len > 0 { len - 1 } else { 0 },
                stride: tensor.strides[dim],
            })
            .collect();

        Elements {
            data: &tensor.data,

            len: tensor.len(),
            offset: tensor.base,
            dims,
        }
    }
}

impl<'a, T: Copy> Iterator for Elements<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let element = self.data[self.offset];

        self.len -= 1;

        // Find dimension where the last element has not been reached.
        let mut dim = self.dims.len() - 1;
        while dim > 0 && self.dims[dim].index >= self.dims[dim].max_index {
            // Reset offset back to the start of this dimension.
            self.offset -= self.dims[dim].index * self.dims[dim].stride;
            self.dims[dim].index = 0;
            dim -= 1;
        }

        self.dims[dim].index += 1;
        self.offset += self.dims[dim].stride;

        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for Elements<'a, T> {}

/// Return the default strides for a given tensor shape.
///
/// The returned strides are for a tightly packed tensor (ie. no unused
/// elements) where all elements are stored in the default order.
fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    for i in 0..shape.len() {
        let stride = shape[i + 1..].iter().product();
        strides.push(stride);
    }
    strides
}

/// Create a new tensor with all values set to 0.
pub fn zero_tensor<T: Copy + Default>(shape: &[usize]) -> Tensor<T> {
    let n_elts = shape.iter().product();
    let data = vec![T::default(); n_elts];
    let strides = strides_for_shape(&shape);
    Tensor {
        data,
        base: 0,
        shape: shape.into(),
        strides,
    }
}

/// Create a new tensor filled with random values supplied by `rng`.
pub fn random_tensor(shape: &[usize], rng: &mut XorShiftRNG) -> Tensor {
    let mut t = zero_tensor(shape);
    t.data.fill_with(|| rng.next_f32());
    t
}

/// Create a new tensor with a given shape and values
pub fn from_data<T: Copy>(shape: Vec<usize>, data: Vec<T>) -> Tensor<T> {
    let strides = strides_for_shape(&shape);
    Tensor {
        data,
        base: 0,
        shape,
        strides,
    }
}

#[cfg(test)]
mod tests {
    use crate::rng::XorShiftRNG;
    use crate::tensor::{random_tensor, zero_tensor};

    #[test]
    fn test_stride() {
        let x = zero_tensor::<f32>(&[2, 5, 7, 3]);
        assert_eq!(x.stride(3), 1);
        assert_eq!(x.stride(2), 3);
        assert_eq!(x.stride(1), 7 * 3);
        assert_eq!(x.stride(0), 5 * 7 * 3);
    }

    #[test]
    fn test_index() {
        let mut x = zero_tensor::<f32>(&[2, 2]);

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
        let mut x = zero_tensor::<f32>(&[2, 2]);

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
        let x = zero_tensor::<f32>(&[2, 2]);
        x[[2, 0]];
    }

    #[test]
    #[should_panic]
    fn test_index_panics_if_wrong_dim_count() {
        let x = zero_tensor::<f32>(&[2, 2]);
        x[[0, 0, 0]];
    }

    #[test]
    fn test_dims() {
        let x = zero_tensor::<f32>(&[10, 5, 3, 7]);
        let [i, j, k, l] = x.dims();

        assert_eq!(i, 10);
        assert_eq!(j, 5);
        assert_eq!(k, 3);
        assert_eq!(l, 7);
    }

    #[test]
    #[should_panic]
    fn test_dims_panics_if_wrong_array_length() {
        let x = zero_tensor::<f32>(&[10, 5, 3, 7]);
        let [_i, _j, _k] = x.dims();
    }

    #[test]
    fn test_reshape() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = random_tensor(&[10, 5, 3, 7], &mut rng);
        let x_data: Vec<f32> = x.data().into();

        assert_eq!(x.shape(), &[10, 5, 3, 7]);

        x.reshape(&[10, 5, 3 * 7]);

        assert_eq!(x.shape(), &[10, 5, 3 * 7]);
        assert_eq!(x.data(), x_data);
    }

    #[test]
    #[should_panic(expected = "New shape must have same total elements as current shape")]
    fn test_reshape_with_wrong_size() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = random_tensor(&[10, 5, 3, 7], &mut rng);
        x.reshape(&[10, 5]);
    }

    #[test]
    fn test_clone_with_shape() {
        let mut rng = XorShiftRNG::new(1234);
        let x = random_tensor(&[10, 5, 3, 7], &mut rng);
        let y = x.clone_with_shape(&[10, 5, 3 * 7]);

        assert_eq!(y.shape(), &[10, 5, 3 * 7]);
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_unchecked_view() {
        let mut rng = XorShiftRNG::new(1234);
        let x = random_tensor(&[10, 5, 3, 7], &mut rng);
        let x_view = x.unchecked_view([5, 3, 0, 0]);

        for a in 0..x.shape()[2] {
            for b in 0..x.shape()[3] {
                assert_eq!(x[[5, 3, a, b]], x_view[[a, b]]);
            }
        }
    }

    #[test]
    fn test_unchecked_view_mut() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = random_tensor(&[10, 5, 3, 7], &mut rng);

        let [_, _, a_size, b_size] = x.dims();
        let mut x_view = x.unchecked_view_mut([5, 3, 0, 0]);

        for a in 0..a_size {
            for b in 0..b_size {
                x_view[[a, b]] = (a + b) as f32;
            }
        }

        for a in 0..x.shape()[2] {
            for b in 0..x.shape()[3] {
                assert_eq!(x[[5, 3, a, b]], (a + b) as f32);
            }
        }
    }

    #[test]
    fn test_last_dim_slice() {
        let mut rng = XorShiftRNG::new(1234);
        let x = random_tensor(&[10, 5, 3, 7], &mut rng);
        let x_slice = x.last_dim_slice([5, 3, 2, 0], x.shape()[3]);

        for i in 0..x.shape()[3] {
            assert_eq!(x[[5, 3, 2, i]], x_slice[i]);
        }
    }

    #[test]
    fn test_last_dim_slice_mut() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = random_tensor(&[10, 5, 3, 7], &mut rng);
        let x_slice = x.last_dim_slice_mut([5, 3, 2, 0], x.shape()[3]);

        for val in x_slice.iter_mut() {
            *val = 1.0;
        }

        for i in 0..x.shape()[3] {
            assert_eq!(x[[5, 3, 2, i]], 1.0);
        }
    }

    #[test]
    fn test_elements_for_contiguous_array() {
        for dims in 1..7 {
            let mut shape = Vec::new();
            for d in 0..dims {
                shape.push(d + 1);
            }
            let mut rng = XorShiftRNG::new(1234);
            let x = random_tensor(&shape, &mut rng);

            let elts: Vec<f32> = x.elements().collect();

            assert_eq!(x.data(), elts);
        }
    }

    #[test]
    fn test_elements_for_empty_array() {
        let empty = zero_tensor::<f32>(&[3, 0, 5]);
        assert!(empty.elements().next().is_none());
    }

    #[test]
    fn test_elements_for_non_contiguous_array() {
        let mut x = zero_tensor(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        // Initially tensor is contiguous, so data buffer and element sequence
        // match.
        assert_eq!(x.data(), x.elements().collect::<Vec<_>>());

        // Slice the tensor. Afterwards the data buffer will be the same but
        // `elements` will only iterate over the new shape.
        x.resize_dim(0, 2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(x.elements().collect::<Vec<_>>(), &[1, 2, 3, 4, 5, 6]);

        x.resize_dim(1, 2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(x.elements().collect::<Vec<_>>(), &[1, 2, 4, 5]);
    }

    #[test]
    fn test_is_contiguous() {
        let mut x = zero_tensor(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        assert!(x.is_contiguous());
        x.resize_dim(0, 2);
        assert!(x.is_contiguous());
    }
}
