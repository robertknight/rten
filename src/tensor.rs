use std::iter::{repeat, zip};
use std::ops::{Index, IndexMut, Range};

#[cfg(test)]
use crate::rng::XorShiftRNG;

/// n-dimensional array
#[derive(Debug)]
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

/// Trait for indexing a `Tensor`
pub trait TensorIndex {
    /// Return the number of dimensions in the index.
    fn len(&self) -> usize;

    /// Return the index for dimension `dim`
    fn index(&self, dim: usize) -> usize;
}

impl<const N: usize> TensorIndex for [usize; N] {
    fn len(&self) -> usize {
        N
    }

    fn index(&self, dim: usize) -> usize {
        self[dim]
    }
}

impl TensorIndex for &[usize] {
    fn len(&self) -> usize {
        (self as &[usize]).len()
    }

    fn index(&self, dim: usize) -> usize {
        self[dim]
    }
}

impl<T: Copy> Tensor<T> {
    /// Return a copy of this tensor with each element replaced by `f(element)`
    pub fn map<F, U: Copy>(&self, f: F) -> Tensor<U>
    where
        F: FnMut(T) -> U,
    {
        let data = self.elements().map(f).collect();
        Tensor {
            data,
            base: 0,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
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

    /// Return the number of dimensions the tensor has, aka. the rank of the
    /// tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Clip dimension `dim` to `[start, end)`. The new size for the dimension
    /// must be <= the old size.
    ///
    /// This is a fast operation since it just alters the start offset within
    /// the tensor's element buffer and length of the specified dimension.
    pub fn clip_dim(&mut self, dim: usize, start: usize, end: usize) {
        if end > self.shape[dim] {
            panic!("New end must be <= old end");
        }
        self.base += self.strides[dim] * start;
        self.shape[dim] = end - start;
    }

    /// Return a contiguous slice of `len` elements starting at `index`.
    /// `len` must be less than or equal to the size of the last dimension.
    ///
    /// Using a slice can allow for very efficient access to a range of elements
    /// in a single row or column (or whatever the last dimension represents).
    pub fn last_dim_slice<const N: usize>(&self, index: [usize; N], len: usize) -> &[T] {
        assert!(
            self.strides[N - 1] == 1,
            "last_dim_slice requires contiguous last dimension"
        );
        let offset = self.offset(index);
        &self.data[offset..offset + len]
    }

    /// Similar to `last_dim_slice`, but returns a mutable slice.
    pub fn last_dim_slice_mut<const N: usize>(
        &mut self,
        index: [usize; N],
        len: usize,
    ) -> &mut [T] {
        assert!(
            self.strides[N - 1] == 1,
            "last_dim_slice_mut requires contiguous last dimension"
        );
        let offset = self.offset(index);
        &mut self.data[offset..offset + len]
    }

    /// Return the underlying element buffer for this tensor.
    ///
    /// If the tensor is contiguous, the buffer will contain the same elements
    /// in the same order as yielded by `elements`. In other cases the buffer
    /// may have unused indexes or a different ordering.
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
        let mut product = 1;
        for (dim, len) in self.shape.iter().enumerate().rev() {
            if self.strides[dim] != product {
                return false;
            }
            product *= len;
        }
        true
    }

    /// Return an iterator over elements of this tensor, in their logical order.
    pub fn elements(&self) -> Elements<T> {
        Elements::new(self)
    }

    /// Returns the single item if this tensor is a 0-dimensional tensor
    /// (ie. a scalar)
    pub fn item(&self) -> Option<T> {
        match self.ndim() {
            0 => Some(self.data[self.base]),
            _ if self.len() == 1 => self.elements().next(),
            _ => None,
        }
    }

    /// Return an iterator over elements of this tensor, broadcasted to `shape`.
    ///
    /// A broadcasted iterator behaves as if the tensor had the broadcasted
    /// shape, repeating elements as necessary to fill the given dimensions.
    /// Broadcasting is only possible if the actual and broadcast shapes are
    /// compatible according to ONNX's rules. See
    /// <https://github.com/onnx/onnx/blob/main/docs/Operators.md>.
    ///
    /// See also <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>
    /// for worked examples of how broadcasting works.
    pub fn broadcast_elements(&self, shape: &[usize]) -> Elements<T> {
        if !self.can_broadcast(shape) {
            panic!("Broadcast shape is not compatible with actual shape");
        }
        if self.shape == shape {
            Elements::new(self)
        } else {
            Elements::broadcast(self, shape)
        }
    }

    /// Return true if the element's shape can be broadcast to `shape` using
    /// `broadcast_elements`.
    ///
    /// See <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md> for
    /// conditions in which broadcasting is allowed.
    pub fn can_broadcast(&self, shape: &[usize]) -> bool {
        if self.shape == shape {
            return true;
        } else if self.ndim() > shape.len() {
            return false;
        }

        // For two shapes to be compatible for broadcasting, each dimension must
        // either be the same or be 1.
        //
        // If the tensor has fewer dimensions, pretend that it was prefixed with
        // 1-length dimensions to make the dimension counts equal.
        let self_dims = self.shape.iter().copied();
        let target_dims = shape[shape.len() - self.shape.len()..].iter().copied();

        zip(self_dims, target_dims).all(|(a, b)| a == b || a == 1 || b == 1)
    }

    /// Return an iterator over a subset of elements in this tensor.
    ///
    /// `indices` is a slice of `(start, end)` tuples for each dimension.
    pub fn slice_elements(&self, indices: &[(usize, usize)]) -> Elements<T> {
        Elements::slice(self, indices)
    }

    /// Return an iterator over offsets of elements in this tensor.
    ///
    /// The returned offsets can be used to index the data buffer returned by
    /// `data` and `data_mut`.
    ///
    /// Unlike `slice_elements`, the returned `Offsets` struct does not hold
    /// a reference to this tensor, so it is possible to modify the tensor while
    /// iterating over offsets.
    ///
    /// `indices` is a slice of `(start, end)` tuples for each dimension.
    pub fn slice_offsets(&self, indices: &[(usize, usize)]) -> Offsets {
        Offsets::slice(self, indices)
    }

    /// Update the shape of the tensor.
    ///
    /// The total number of elements for the new shape must be the same as the
    /// existing shape.
    ///
    /// This is a cheap operation if the tensor is contiguous, but requires
    /// copying data if the tensor has a non-contiguous layout.
    pub fn reshape(&mut self, shape: &[usize]) {
        let len: usize = shape.iter().product();
        let current_len = self.len();

        if len != current_len {
            panic!("New shape must have same total elements as current shape");
        }

        // We currently always copy data whenever the input is non-contiguous.
        // However there are cases of custom strides where copies could be
        // avoided. See https://pytorch.org/docs/stable/generated/torch.Tensor.view.html.
        if !self.is_contiguous() {
            self.base = 0;
            self.data = self.elements().collect();
        }

        self.shape = shape.into();
        self.strides = strides_for_shape(shape);
    }

    /// Re-order the dimensions according to `dims`.
    ///
    /// This does not modify the order of elements in the data buffer, it merely
    /// updates the strides used by indexing.
    pub fn permute(&mut self, dims: &[usize]) {
        if dims.len() != self.ndim() {
            panic!("Permute dims length does not match dimension count");
        }
        self.strides = dims.iter().map(|&dim| self.strides[dim]).collect();
        self.shape = dims.iter().map(|&dim| self.shape[dim]).collect();
    }

    /// Return the number of elements between successive entries in the `dim`
    /// dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    /// Return the offset of an element that corresponds to a given index.
    ///
    /// The length of `index` must match the tensor's dimension count.
    ///
    /// Panicks if the index length is incorrect or the value of an index
    /// exceeds the size of the corresponding dimension.
    pub fn offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        let shape = &self.shape;
        assert!(
            shape.len() == index.len(),
            "Cannot access {} dim tensor with {} dim index",
            shape.len(),
            index.len()
        );
        let mut offset = self.base;
        for i in 0..index.len() {
            assert!(
                index.index(i) < self.shape[i],
                "Invalid index {} for dim {}",
                index.index(i),
                i
            );
            offset += index.index(i) * self.stride(i)
        }
        offset
    }

    /// Return the shape of this tensor as a fixed-sized array.
    ///
    /// The tensor's dimension count must match `N`.
    pub fn dims<const N: usize>(&self) -> [usize; N] {
        if self.ndim() != N {
            panic!(
                "Cannot extract {} dim tensor as {} dim array",
                self.ndim(),
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
            data: &self.data,
            offset,
            strides: self.strides[self.ndim() - N..].try_into().unwrap(),
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
        let strides = self.strides[self.ndim() - N..].try_into().unwrap();
        UncheckedViewMut {
            data: &mut self.data,
            offset,
            strides,
        }
    }
}

impl<T: Copy + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.elements().eq(other.elements())
    }
}

impl<T: Copy> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
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
}

impl<I: TensorIndex, T: Copy> Index<I> for Tensor<T> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        &self.data[self.offset(index)]
    }
}

impl<I: TensorIndex, T: Copy> IndexMut<I> for Tensor<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let offset = self.offset(index);
        &mut self.data[offset]
    }
}

pub struct UncheckedView<'a, T: Copy, const N: usize> {
    data: &'a [T],
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
        &self.data[offset]
    }
}

pub struct UncheckedViewMut<'a, T: Copy, const N: usize> {
    data: &'a mut [T],
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
        &self.data[offset]
    }
}

impl<'a, const N: usize, T: Copy> IndexMut<[usize; N]> for UncheckedViewMut<'a, T, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let mut offset = self.offset;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        &mut self.data[offset]
    }
}

#[derive(Debug)]
struct ElementsDim {
    /// Current index for this dimension
    index: usize,

    /// Maximum index value for this dimension
    max_index: usize,

    /// Amount to increase offset by every time index is incremented in this
    /// dimension
    stride: usize,
}

/// Struct with shared functionality for iterating over elements, indexes and
/// offsets of a tensor.
struct ElementsBase {
    /// Remaining elements to visit
    len: usize,

    /// Offset of next element to return in `data`
    offset: usize,

    /// True if the tensor data is contiguous in memory
    contiguous: bool,

    /// Index of next element within each dimension. The stride and max value
    /// of each dimension are also copied into this struct for faster access
    /// during iteration.
    ///
    /// This is not used if the tensor is contiguous.
    dims: Vec<ElementsDim>,
}

impl ElementsBase {
    fn new<T: Copy>(tensor: &Tensor<T>) -> ElementsBase {
        let contiguous = tensor.is_contiguous();
        let dims = if contiguous {
            Vec::new()
        } else {
            tensor
                .shape
                .iter()
                .enumerate()
                .map(|(dim, &len)| ElementsDim {
                    index: 0,
                    max_index: if len > 0 { len - 1 } else { 0 },
                    stride: tensor.strides[dim],
                })
                .collect()
        };

        ElementsBase {
            len: tensor.len(),
            offset: tensor.base,
            dims,
            contiguous,
        }
    }

    fn broadcast<T: Copy>(tensor: &Tensor<T>, shape: &[usize]) -> ElementsBase {
        // nb. We require that the broadcast shape has a length >= the actual
        // shape.
        let added_dims = shape.len() - tensor.shape().len();
        let padded_tensor_shape = repeat(&0).take(added_dims).chain(tensor.shape().iter());
        let dims = zip(padded_tensor_shape, shape.iter())
            .enumerate()
            .map(|(dim, (&actual_len, &broadcast_len))| ElementsDim {
                index: 0,
                max_index: if broadcast_len > 0 {
                    broadcast_len - 1
                } else {
                    0
                },

                // If the dimension is being broadcast, set its stride to 0 so
                // that when we increment in this dimension, we just repeat
                // elements. Otherwise, use the real stride.
                stride: if actual_len == broadcast_len {
                    tensor.strides[dim - added_dims]
                } else {
                    0
                },
            })
            .collect();

        ElementsBase {
            len: shape.iter().product(),
            offset: tensor.base,
            dims,
            contiguous: false,
        }
    }

    fn slice<T: Copy>(tensor: &Tensor<T>, ranges: &[(usize, usize)]) -> ElementsBase {
        if ranges.len() != tensor.ndim() {
            panic!(
                "slice dimensions {} do not match tensor dimensions {}",
                ranges.len(),
                tensor.ndim()
            );
        }
        let mut offset = tensor.base;
        let mut dims = Vec::with_capacity(ranges.len());

        for (dim, (start, end)) in ranges.iter().copied().enumerate() {
            let slice_dim_size = end.saturating_sub(start);
            let dim_size = tensor.shape[dim];

            if slice_dim_size > tensor.shape[dim] {
                panic!(
                    "slice range {}..{} for dimension {} exceeds dimension size {}",
                    start, end, dim, dim_size
                );
            }

            let stride = tensor.strides[dim];
            offset += stride * start;

            dims.push(ElementsDim {
                index: 0,
                max_index: slice_dim_size - 1,
                stride,
            });
        }

        ElementsBase {
            len: ranges.iter().map(|(start, end)| end - start).product(),
            offset,
            dims,
            contiguous: false,
        }
    }

    fn step(&mut self) {
        self.len -= 1;

        // Fast path for contiguous tensors
        if self.contiguous {
            self.offset += 1;
            return;
        }

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
    }
}

/// Iterator over elements of a tensor.
pub struct Elements<'a, T: Copy> {
    base: ElementsBase,

    /// Data buffer of the tensor
    data: &'a [T],
}

impl<'a, T: Copy> Elements<'a, T> {
    fn new(tensor: &'a Tensor<T>) -> Elements<'a, T> {
        Elements {
            base: ElementsBase::new(tensor),
            data: &tensor.data,
        }
    }

    fn broadcast(tensor: &'a Tensor<T>, shape: &[usize]) -> Elements<'a, T> {
        Elements {
            base: ElementsBase::broadcast(tensor, shape),
            data: &tensor.data,
        }
    }

    fn slice(tensor: &'a Tensor<T>, ranges: &[(usize, usize)]) -> Elements<'a, T> {
        Elements {
            base: ElementsBase::slice(tensor, ranges),
            data: &tensor.data,
        }
    }
}

impl<'a, T: Copy> Iterator for Elements<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.base.len == 0 {
            return None;
        }
        let element = self.data[self.base.offset];
        self.base.step();
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for Elements<'a, T> {}

/// Iterator over element offsets of a tensor.
///
/// `Offsets` does not hold a reference to the tensor, allowing the tensor to
/// be modified during iteration. It is the caller's responsibilty not to modify
/// the tensor in ways that invalidate the offset sequence returned by this
/// iterator.
pub struct Offsets {
    base: ElementsBase,
}

impl Offsets {
    fn slice<T: Copy>(tensor: &Tensor<T>, ranges: &[(usize, usize)]) -> Offsets {
        Offsets {
            base: ElementsBase::slice(tensor, ranges),
        }
    }
}

impl Iterator for Offsets {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.base.len == 0 {
            return None;
        }
        let offset = self.base.offset;
        self.base.step();
        Some(offset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }
}

impl ExactSizeIterator for Offsets {}

/// An iterator over indices within a given range.
///
/// This struct does not implement the `Iterator` trait because such iterators
/// cannot return references to data they contain. Use a `while` loop instead.
pub struct IndexIterator {
    first: bool,
    current: Vec<usize>,
    ranges: Vec<Range<usize>>,
}

impl IndexIterator {
    /// Return an iterator over all the indices where each dimension lies
    /// within the corresponding range in `ranges`.
    ///
    /// If `ranges` is empty, the iterator yields a single empty index. This
    /// is consistent with `ndindex` in eg. numpy.
    pub fn from_ranges(ranges: &[Range<usize>]) -> IndexIterator {
        let current = ranges.iter().map(|r| r.start).collect();
        IndexIterator {
            first: true,
            current,
            ranges: ranges.into(),
        }
    }

    /// Return an iterator over all the indices where each dimension is between
    /// `0` and `shape[dim]`.
    pub fn from_shape(shape: &[usize]) -> IndexIterator {
        let ranges = shape.iter().map(|&size| 0..size).collect();
        let current = vec![0; shape.len()];
        IndexIterator {
            first: true,
            current,
            ranges,
        }
    }

    /// Reset the iterator back to the first index.
    pub fn reset(&mut self) {
        self.first = true;
        for i in 0..self.ranges.len() {
            self.current[i] = self.ranges[i].start;
        }
    }

    /// Return the index index in the sequence, or `None` after all indices
    /// have been returned.
    pub fn next(&mut self) -> Option<&[usize]> {
        if self.current.is_empty() {
            if self.first {
                self.first = false;
                return Some(&self.current[..]);
            } else {
                return None;
            }
        }

        // Find dimension where the last element has not been reached.
        let mut dim = self.current.len() - 1;
        while dim > 0 && self.current[dim] >= self.ranges[dim].end - 1 {
            self.current[dim] = self.ranges[dim].start;
            dim -= 1;
        }

        if self.first {
            self.first = false;
        } else {
            self.current[dim] += 1;
        }

        if self.current[dim] >= self.ranges[dim].end {
            return None;
        }

        Some(&self.current[..])
    }
}

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
    let strides = strides_for_shape(shape);
    Tensor {
        data,
        base: 0,
        shape: shape.into(),
        strides,
    }
}

/// Create a new tensor filled with random values supplied by `rng`.
#[cfg(test)]
pub fn random_tensor(shape: &[usize], rng: &mut XorShiftRNG) -> Tensor {
    let mut t = zero_tensor(shape);
    t.data.fill_with(|| rng.next_f32());
    t
}

/// Create a new tensor with a given shape and values
pub fn from_data<T: Copy>(shape: Vec<usize>, data: Vec<T>) -> Tensor<T> {
    if shape[..].iter().product::<usize>() != data.len() {
        panic!(
            "Number of elements given by shape {:?} does not match data length {}",
            &shape[..],
            data.len()
        );
    }
    let strides = strides_for_shape(&shape);
    Tensor {
        data,
        base: 0,
        shape,
        strides,
    }
}

/// Create a new 0-dimensional (scalar) tensor from a single value.
pub fn from_scalar<T: Copy>(value: T) -> Tensor<T> {
    from_data(vec![], vec![value])
}

/// Create a new 1-dimensional tensor from a vector
pub fn from_vec<T: Copy>(data: Vec<T>) -> Tensor<T> {
    from_data(vec![data.len()], data)
}

/// Create a new 2D tensor from a nested array of slices.
pub fn from_2d_slice<T: Copy>(data: &[&[T]]) -> Tensor<T> {
    let rows = data.len();
    let cols = data.get(0).map(|first_row| first_row.len()).unwrap_or(0);

    let mut result = Vec::new();
    for row in data {
        assert!(cols == row.len(), "All row slices must have same length");
        result.extend_from_slice(row);
    }

    from_data(vec![rows, cols], result)
}

#[cfg(test)]
mod tests {
    use crate::rng::XorShiftRNG;
    use crate::tensor::{
        from_2d_slice, from_data, from_scalar, from_vec, random_tensor, zero_tensor, IndexIterator,
        Tensor,
    };

    /// Create a tensor where the value of each element is its logical index
    /// plus one.
    fn steps(shape: &[usize]) -> Tensor<i32> {
        let mut x = zero_tensor(shape);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = (index + 1) as i32;
        }
        x
    }

    #[test]
    fn test_clip_dim() {
        let mut x = steps(&[3, 3]);
        x.clip_dim(0, 1, 2);
        x.clip_dim(1, 1, 2);
        assert_eq!(x.elements().collect::<Vec<i32>>(), vec![5]);
    }

    #[test]
    fn test_from_2d_slice() {
        let x = from_2d_slice(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);
        assert_eq!(x.shape(), &[3, 3]);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_from_scalar() {
        let x = from_scalar(5);
        assert_eq!(x.shape(), &[]);
        assert_eq!(x.data(), &[5]);
    }

    #[test]
    fn test_from_vec() {
        let x = from_vec(vec![1, 2, 3]);
        assert_eq!(x.shape(), &[3]);
        assert_eq!(x.data(), &[1, 2, 3]);
    }

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

        // Index with fixed-sized array.
        assert_eq!(x[[0, 0]], 1.0);
        assert_eq!(x[[0, 1]], 2.0);
        assert_eq!(x[[1, 0]], 3.0);
        assert_eq!(x[[1, 1]], 4.0);

        // Index with slice.
        assert_eq!(x[[0, 0].as_slice()], 1.0);
        assert_eq!(x[[0, 1].as_slice()], 2.0);
        assert_eq!(x[[1, 0].as_slice()], 3.0);
        assert_eq!(x[[1, 1].as_slice()], 4.0);
    }

    #[test]
    fn test_index_scalar() {
        let x = from_scalar(5.0);
        assert_eq!(x[[]], 5.0);
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
    fn test_item() {
        let scalar = from_scalar(5.0);
        assert_eq!(scalar.item(), Some(5.0));

        let vec_one_item = from_vec(vec![5.0]);
        assert_eq!(vec_one_item.item(), Some(5.0));

        let vec_many_items = from_vec(vec![1.0, 2.0]);
        assert_eq!(vec_many_items.item(), None);

        let matrix_one_item = from_data(vec![1, 1], vec![5.0]);
        assert_eq!(matrix_one_item.item(), Some(5.0));
    }

    #[test]
    fn test_ndim() {
        let scalar = from_scalar(5.0);
        let vec = from_vec(vec![5.0]);
        let matrix = from_data(vec![1, 1], vec![5.0]);

        assert_eq!(scalar.ndim(), 0);
        assert_eq!(vec.ndim(), 1);
        assert_eq!(matrix.ndim(), 2);
    }

    #[test]
    fn test_partial_eq() {
        let x = from_vec(vec![1, 2, 3, 4, 5]);
        let y = x.clone();
        let z = x.clone_with_shape(&[1, 5]);

        // Int tensors are equal if they have the same shape and elements.
        assert_eq!(&x, &y);
        assert_ne!(&x, &z);
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
    fn test_len() {
        let scalar = from_scalar(5);
        let vec = from_vec(vec![1, 2, 3]);
        let matrix = from_data(vec![2, 2], vec![1, 2, 3, 4]);

        assert_eq!(scalar.len(), 1);
        assert_eq!(vec.len(), 3);
        assert_eq!(matrix.len(), 4);
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
    fn test_reshape_non_contiguous() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = random_tensor(&[10, 10], &mut rng);

        // Set the input up so that it is non-contiguous and has a non-zero
        // `base` offset.
        x.permute(&[1, 0]);
        x.clip_dim(0, 2, 8);

        // Reshape the tensor. This should copy the data and reset the `base`
        // offset.
        x.reshape(&[x.shape().iter().product()]);

        // After reshaping, we should be able to successfully read all the elements.
        let elts: Vec<_> = x.elements().collect();
        assert_eq!(elts.len(), 60);
    }

    #[test]
    fn test_reshape_copies_with_custom_strides() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = random_tensor(&[10, 10], &mut rng);

        // Give the tensor a non-default stride
        x.clip_dim(1, 0, 8);
        assert!(!x.is_contiguous());
        let x_elements: Vec<f32> = x.elements().collect();

        x.reshape(&[80]);

        // Since the tensor had a non-default stride, `reshape` will have copied
        // data.
        assert_eq!(x.shape(), &[80]);
        assert!(x.is_contiguous());
        assert_eq!(x.data(), x_elements);
    }

    #[test]
    #[should_panic(expected = "New shape must have same total elements as current shape")]
    fn test_reshape_with_wrong_size() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = random_tensor(&[10, 5, 3, 7], &mut rng);
        x.reshape(&[10, 5]);
    }

    #[test]
    fn test_permute() {
        // Test with a vector (this is a no-op)
        let mut input = steps(&[5]);
        assert!(input.elements().eq([1, 2, 3, 4, 5].iter().copied()));
        input.permute(&[0]);
        assert!(input.elements().eq([1, 2, 3, 4, 5].iter().copied()));

        // Test with a matrix (ie. transpose the matrix)
        let mut input = steps(&[2, 3]);
        assert!(input.elements().eq([1, 2, 3, 4, 5, 6].iter().copied()));
        input.permute(&[1, 0]);
        assert_eq!(input.shape(), &[3, 2]);
        assert!(input.elements().eq([1, 4, 2, 5, 3, 6].iter().copied()));

        // Test with a higher-rank tensor. For this test we don't list out the
        // full permuted element sequence, but just check the shape and strides
        // were updated.
        let mut input = steps(&[3, 4, 5]);
        let (stride_0, stride_1, stride_2) = (input.stride(0), input.stride(1), input.stride(2));
        input.permute(&[2, 0, 1]);
        assert_eq!(input.shape(), &[5, 3, 4]);
        assert_eq!(
            (input.stride(0), input.stride(1), input.stride(2)),
            (stride_2, stride_0, stride_1)
        );
    }

    #[test]
    #[should_panic(expected = "Permute dims length does not match dimension count")]
    fn test_permute_wrong_dim_count() {
        let mut input = steps(&[2, 3]);
        input.permute(&[1, 2, 3]);
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
        x.clip_dim(0, 0, 2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(x.elements().collect::<Vec<_>>(), &[1, 2, 3, 4, 5, 6]);

        x.clip_dim(1, 0, 2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(x.elements().collect::<Vec<_>>(), &[1, 2, 4, 5]);
    }

    // PyTorch and numpy do not allow iteration over a scalar, but it seems
    // consistent for `Tensor::elements` to always yield `Tensor::len` elements,
    // and `len` returns 1 for a scalar.
    #[test]
    fn test_elements_for_scalar() {
        let x = from_scalar(5.0);
        let elements = x.elements().collect::<Vec<_>>();
        assert_eq!(&elements, &[5.0]);
    }

    #[test]
    fn test_from_data() {
        let scalar = from_data(vec![], vec![1.0]);
        assert_eq!(scalar.len(), 1);

        let matrix = from_data(vec![2, 2], vec![1, 2, 3, 4]);
        assert_eq!(matrix.shape(), &[2, 2]);
        assert_eq!(matrix.data(), &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_with_wrong_len() {
        from_data(vec![1], vec![1, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_if_scalar_data_empty() {
        from_data::<i32>(vec![], vec![]);
    }

    #[test]
    #[should_panic]
    fn test_from_data_panics_if_scalar_data_has_many_elements() {
        from_data(vec![], vec![1, 2, 3]);
    }

    #[test]
    fn test_is_contiguous() {
        let mut x = zero_tensor(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        assert!(x.is_contiguous());
        x.clip_dim(0, 0, 2);
        assert!(x.is_contiguous());
        x.clip_dim(1, 0, 2);
        assert!(!x.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_1d() {
        let mut x = zero_tensor(&[10]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        assert!(x.is_contiguous());
        x.clip_dim(0, 0, 5);
        assert!(x.is_contiguous());
    }

    #[test]
    fn test_broadcast_elements() {
        let x = steps(&[1, 2, 1, 2]);
        assert_eq!(x.elements().collect::<Vec<i32>>(), &[1, 2, 3, 4]);

        // Broadcast a 1-size dimension to size 2
        let bx = x.broadcast_elements(&[2, 2, 1, 2]);
        assert_eq!(bx.collect::<Vec<i32>>(), &[1, 2, 3, 4, 1, 2, 3, 4]);

        // Broadcast a different 1-size dimension to size 2
        let bx = x.broadcast_elements(&[1, 2, 2, 2]);
        assert_eq!(bx.collect::<Vec<i32>>(), &[1, 2, 1, 2, 3, 4, 3, 4]);

        // Broadcast to a larger number of dimensions
        let x = steps(&[5]);
        let bx = x.broadcast_elements(&[1, 5]);
        assert_eq!(bx.collect::<Vec<i32>>(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_broadcast_elements_with_scalar() {
        let scalar = from_scalar(7);
        let bx = scalar.broadcast_elements(&[3, 3]);
        assert_eq!(bx.collect::<Vec<i32>>(), &[7, 7, 7, 7, 7, 7, 7, 7, 7]);
    }

    #[test]
    #[should_panic(expected = "Broadcast shape is not compatible with actual shape")]
    fn test_broadcast_elements_with_invalid_shape() {
        let x = steps(&[2, 2]);
        x.broadcast_elements(&[3, 2]);
    }

    #[test]
    #[should_panic(expected = "Broadcast shape is not compatible with actual shape")]
    fn test_broadcast_elements_with_shorter_shape() {
        let x = steps(&[2, 2]);
        x.broadcast_elements(&[4]);
    }

    #[test]
    fn test_slice_elements() {
        let x = steps(&[3, 3]);

        // Slice that removes start of each dimension
        let slice: Vec<_> = x.slice_elements(&[(1, 3), (1, 3)]).collect();
        assert_eq!(slice, &[5, 6, 8, 9]);

        // Slice that removes end of each dimension
        let slice: Vec<_> = x.slice_elements(&[(0, 2), (0, 2)]).collect();
        assert_eq!(slice, &[1, 2, 4, 5]);

        // Slice that removes start and end of first dimension
        let slice: Vec<_> = x.slice_elements(&[(1, 2), (0, 3)]).collect();
        assert_eq!(slice, &[4, 5, 6]);

        // Slice that removes start and end of second dimension
        let slice: Vec<_> = x.slice_elements(&[(0, 3), (1, 2)]).collect();
        assert_eq!(slice, &[2, 5, 8]);
    }

    // These tests assume the correctness of `slice_elements`, given the tests
    // above, and check for consistency between the results of `slice_offsets`
    // and `slice_elements`.
    #[test]
    fn test_slice_offsets() {
        let x = steps(&[5, 5]);

        // Range that removes the start and end of each dimension.
        let range = &[(1, 4), (1, 4)];
        let expected: Vec<_> = x.slice_elements(range).collect();
        let result: Vec<_> = x
            .slice_offsets(range)
            .map(|offset| x.data()[offset])
            .collect();

        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_index_iterator() {
        // Empty iterator
        let mut iter = IndexIterator::from_ranges(&[0..0]);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        // Scalar index iterator
        let mut iter = IndexIterator::from_ranges(&[]);
        assert_eq!(iter.next(), Some(&[] as &[usize]));
        assert_eq!(iter.next(), None);

        // 1D index iterator
        let mut iter = IndexIterator::from_ranges(&[0..5]);
        let mut visited: Vec<Vec<usize>> = Vec::new();
        while let Some(index) = iter.next() {
            visited.push(index.into());
        }
        assert_eq!(visited, vec![vec![0], vec![1], vec![2], vec![3], vec![4]]);

        // 2D index iterator
        let mut iter = IndexIterator::from_ranges(&[2..4, 2..4]);
        let mut visited: Vec<Vec<usize>> = Vec::new();
        while let Some(index) = iter.next() {
            visited.push(index.into());
        }

        assert_eq!(
            visited,
            vec![vec![2, 2], vec![2, 3], vec![3, 2], vec![3, 3],]
        );
    }
}
