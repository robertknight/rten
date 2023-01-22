use std::borrow::Cow;
use std::fmt::Debug;
use std::io;
use std::ops::{Index, IndexMut, Range};

#[cfg(test)]
use crate::rng::XorShiftRNG;

mod index_iterator;
mod iterators;
mod layout;
mod range;

pub use self::index_iterator::IndexIterator;
pub use self::iterators::{BroadcastElements, Elements, ElementsMut, Offsets};
use self::layout::Layout;
pub use self::range::SliceRange;

/// TensorView provides a view onto data owned by a Tensor.
///
/// Conceptually the relationship between TensorView and Tensor is similar to
/// that between slice and Vec.
#[derive(Clone)]
pub struct TensorView<'a, T: Copy = f32> {
    data: &'a [T],
    layout: Cow<'a, Layout>,
}

impl<'a, T: Copy> TensorView<'a, T> {
    fn new(data: &'a [T], layout: &'a Layout) -> TensorView<'a, T> {
        TensorView {
            data,
            layout: Cow::Borrowed(layout),
        }
    }

    /// Return a slice of the sizes of each dimension.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Change the layout of this view to put dimensions in the order specified
    /// by `dims`.
    pub fn permute(&mut self, dims: &[usize]) {
        self.layout.to_mut().permute(dims);
    }

    /// Return an iterator over elements of this tensor, in their logical order.
    pub fn iter(&self) -> Elements<'a, T> {
        Elements::new(self)
    }

    /// Return a new contiguous tensor with the same shape and elements as this
    /// view.
    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor::from_data(self.shape().into(), self.iter().collect())
    }
}

/// TensorViewMut provides a mutable view onto data owned by a Tensor.
///
/// Conceptually the relationship between TensorViewMut and Tensor is similar to
/// that between a mutable slice and Vec.
pub struct TensorViewMut<'a, T: Copy = f32> {
    data: &'a mut [T],
    layout: Cow<'a, Layout>,
}

impl<'a, T: Copy> TensorViewMut<'a, T> {
    fn new(data: &'a mut [T], layout: &'a Layout) -> TensorViewMut<'a, T> {
        TensorViewMut {
            data,
            layout: Cow::Borrowed(layout),
        }
    }

    /// Return a slice of the sizes of each dimension.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Change the layout of this view to put dimensions in the order specified
    /// by `dims`.
    pub fn permute(&mut self, dims: &[usize]) {
        self.layout.to_mut().permute(dims);
    }

    pub fn iter_mut(&mut self) -> ElementsMut<T> {
        ElementsMut::new(self)
    }
}

/// Tensor is the core n-dimensional array type used for inputs, outputs and
/// intermediate values when executing an ML graph.
#[derive(Debug)]
pub struct Tensor<T: Copy = f32> {
    /// The underlying buffer of elements
    data: Vec<T>,

    /// The offset in the buffer of the first element. This is initially 0 but
    /// will be changed if the tensor is sliced.
    base: usize,

    layout: Layout,
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
    /// Create a new zero-filled tensor with a given shape.
    pub fn zeros(shape: &[usize]) -> Tensor<T>
    where
        T: Default,
    {
        let n_elts = shape.iter().product();
        let data = vec![T::default(); n_elts];
        Tensor {
            data,
            base: 0,
            layout: Layout::new(shape),
        }
    }

    /// Create a new tensor from a given shape and set of elements. No copying
    /// is required.
    pub fn from_data(shape: Vec<usize>, data: Vec<T>) -> Tensor<T> {
        if shape[..].iter().product::<usize>() != data.len() {
            panic!(
                "Number of elements given by shape {:?} does not match data length {}",
                &shape[..],
                data.len()
            );
        }
        Tensor {
            data,
            base: 0,
            layout: Layout::new(&shape),
        }
    }

    /// Create a new 0-dimensional (scalar) tensor from a single value.
    pub fn from_scalar(value: T) -> Tensor<T> {
        from_data(vec![], vec![value])
    }

    /// Create a new 1-dimensional tensor from a vector. No copying is required.
    pub fn from_vec(data: Vec<T>) -> Tensor<T> {
        from_data(vec![data.len()], data)
    }

    /// Replace elements of this tensor with `f(element)`.
    ///
    /// This is the in-place version of `map`.
    ///
    /// The order in which elements are visited is unspecified and may not
    /// correspond to the logical order.
    pub fn apply<F: Fn(T) -> T>(&mut self, f: F) {
        // TODO: Skip unused elements when tensor is not contiguous.
        for val in self.data_mut().iter_mut() {
            *val = f(*val);
        }
    }

    /// Return a copy of this tensor with each element replaced by `f(element)`.
    ///
    /// The order in which elements are visited is unspecified and may not
    /// correspond to the logical order.
    pub fn map<F, U: Copy>(&self, f: F) -> Tensor<U>
    where
        F: Fn(T) -> U,
    {
        let data = self.iter().map(f).collect();
        Tensor {
            data,
            base: 0,
            layout: self.layout.clone(),
        }
    }

    /// Clone this tensor with a new shape. The new shape must have the same
    /// total number of elements as the existing shape. See `reshape`.
    pub fn clone_with_shape(&self, shape: &[usize]) -> Tensor<T> {
        let data = if self.is_contiguous() {
            self.data().into()
        } else {
            self.iter().collect()
        };
        Self::from_data(shape.into(), data)
    }

    /// Return an iterator over all valid indices in this tensor.
    ///
    /// The returned iterator does not implement the `Iterator` trait but has
    /// a similar API. See `IndexIterator` docs.
    pub fn indices(&self) -> IndexIterator {
        IndexIterator::from_shape(self.shape())
    }

    /// Return the total number of elements in this tensor.
    pub fn len(&self) -> usize {
        self.layout.len()
    }

    /// Return true if this tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    /// Return the number of dimensions the tensor has, aka. the rank of the
    /// tensor.
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// Clip dimension `dim` to `[range.start, range.end)`. The new size for
    /// the dimension must be <= the old size.
    ///
    /// This is a fast operation since it just alters the start offset within
    /// the tensor's element buffer and length of the specified dimension.
    pub fn clip_dim(&mut self, dim: usize, range: Range<usize>) {
        let (start, end) = (range.start, range.end);

        assert!(start <= end, "start must be <= end");
        assert!(end <= self.shape()[dim], "end must be <= dim size");

        self.base += self.layout.stride(dim) * start;
        self.layout.resize_dim(dim, end - start);

        if self.is_contiguous() {
            // Truncate buffer to preserve invariant that `Tensor::data` yields
            // the same elements as `Tensor::elements` for a contiguous tensor.
            self.data.truncate(self.base + self.len());
        }
    }

    /// Return a contiguous slice of `len` elements starting at `index`.
    /// `len` must be less than or equal to the size of the last dimension.
    ///
    /// Using a slice can allow for very efficient access to a range of elements
    /// in a single row or column (or whatever the last dimension represents).
    pub fn last_dim_slice<const N: usize>(&self, index: [usize; N], len: usize) -> &[T] {
        assert!(
            self.stride(N - 1) == 1,
            "last_dim_slice requires contiguous last dimension"
        );
        let offset = self.base + self.offset(index);
        &self.data[offset..offset + len]
    }

    /// Similar to `last_dim_slice`, but returns a mutable slice.
    pub fn last_dim_slice_mut<const N: usize>(
        &mut self,
        index: [usize; N],
        len: usize,
    ) -> &mut [T] {
        assert!(
            self.stride(N - 1) == 1,
            "last_dim_slice_mut requires contiguous last dimension"
        );
        let offset = self.base + self.offset(index);
        &mut self.data[offset..offset + len]
    }

    /// Return a copy of the elements of this tensor as a contiguous vector
    /// in row-major order.
    ///
    /// This is slightly more efficient than `iter().collect()` in the case
    /// where the tensor is already contiguous.
    pub fn to_vec(&self) -> Vec<T> {
        if self.is_contiguous() {
            self.data().to_vec()
        } else {
            self.iter().collect()
        }
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
        self.layout.shape()
    }

    /// Return true if the logical order of elements in this tensor matches the
    /// order of elements in the slice returned by `data()` and `data_mut()`,
    /// with no gaps.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Convert the internal layout of elements to be contiguous, as reported
    /// by `is_contiguous`.
    ///
    /// This is a no-op if the tensor is already contiguous.
    pub fn make_contiguous(&mut self) {
        if self.is_contiguous() {
            return;
        }
        self.data = self.iter().collect();
        self.base = 0;
        self.layout.make_contiguous();
    }

    /// Return a contiguous version of this tensor, either as a reference if
    /// the tensor is already contiguous, or a copy if not.
    pub fn as_contiguous(&self) -> Cow<Tensor<T>> {
        if self.is_contiguous() {
            Cow::Borrowed(self)
        } else {
            let mut copy = self.clone();
            copy.make_contiguous();
            Cow::Owned(copy)
        }
    }

    /// Return an iterator over elements of this tensor, in their logical order.
    pub fn iter(&self) -> Elements<T> {
        Elements::new(&self.view())
    }

    /// Return an iterator over offsets of elements in this tensor, in their
    /// logical order.
    ///
    /// See also the notes for `slice_offsets`.
    pub fn offsets(&self) -> Offsets {
        Offsets::new(&self.layout)
    }

    /// Returns the single item if this tensor is a 0-dimensional tensor
    /// (ie. a scalar)
    pub fn item(&self) -> Option<T> {
        match self.ndim() {
            0 => Some(self.data[self.base]),
            _ if self.len() == 1 => self.iter().next(),
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
    pub fn broadcast_elements(&self, shape: &[usize]) -> BroadcastElements<'_, T> {
        if !self.can_broadcast_to(shape) {
            panic!("Cannot broadcast to specified shape");
        }
        BroadcastElements::new(&self.view(), shape)
    }

    /// Return an iterator over offsets of this tensor, broadcasted to `shape`.
    ///
    /// This is very similar to `broadcast_elements`, except that the iterator
    /// yields offsets into rather than elements of the data buffer.
    pub fn broadcast_offsets(&self, shape: &[usize]) -> Offsets {
        if !self.can_broadcast_to(shape) {
            panic!("Cannot broadcast to specified shape");
        }
        Offsets::broadcast(&self.layout, shape)
    }

    /// Return true if the element's shape can be broadcast to `shape` using
    /// `broadcast_elements`. The result of the broadcasted tensor will have
    /// exactly the shape `shape`.
    ///
    /// See <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md> for
    /// conditions in which broadcasting is allowed.
    pub fn can_broadcast_to(&self, shape: &[usize]) -> bool {
        self.layout.can_broadcast_to(shape)
    }

    /// Return true if the element's shape can be broadcast with `shape` using
    /// `broadcast_elements`.
    ///
    /// The shape of the result may be larger than either the current shape
    /// or `shape`. eg. If a tensor of shape `[1, 5]` is broadcast with one
    /// of size `[2, 1, 1]` the result has shape `[2, 1, 5]`.
    ///
    /// See <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md> for
    /// conditions in which broadcasting is allowed.
    pub fn can_broadcast_with(&self, shape: &[usize]) -> bool {
        self.layout.can_broadcast_with(shape)
    }

    /// Return an iterator over a subset of elements in this tensor.
    pub fn slice_elements(&self, ranges: &[SliceRange]) -> Elements<T> {
        Elements::slice(&self.view(), ranges)
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
    /// Note that the offset order of the returned iterator will become incorrect
    /// if the tensor shape is subsequently modified.
    pub fn slice_offsets(&self, ranges: &[SliceRange]) -> Offsets {
        Offsets::slice(&self.layout, ranges)
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
        self.make_contiguous();
        self.layout = Layout::new(shape);
    }

    /// Re-order the dimensions according to `dims`.
    ///
    /// This does not modify the order of elements in the data buffer, it merely
    /// updates the strides used by indexing.
    pub fn permute(&mut self, dims: &[usize]) {
        self.layout.permute(dims);
    }

    /// Insert a dimension of size one at index `dim`.
    pub fn insert_dim(&mut self, dim: usize) {
        let mut new_shape: Vec<usize> = self.shape().into();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape);
    }

    /// Return the number of elements between successive entries in the `dim`
    /// dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.layout.stride(dim)
    }

    /// Return the offset of an element in the slices returned by `data`
    /// and `data_mut`.
    ///
    /// The length of `index` must match the tensor's dimension count.
    ///
    /// Panicks if the index length is incorrect or the value of an index
    /// exceeds the size of the corresponding dimension.
    pub fn offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        self.layout.offset(index)
    }

    /// Return the shape of this tensor as a fixed-sized array.
    ///
    /// The tensor's dimension count must match `N`.
    pub fn dims<const N: usize>(&self) -> [usize; N] {
        self.layout.dims()
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
            data: self.data(),
            offset,
            strides: self.layout.strides()[self.ndim() - N..].try_into().unwrap(),
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
        let strides = self.layout.strides()[self.ndim() - N..].try_into().unwrap();
        UncheckedViewMut {
            data: self.data_mut(),
            offset,
            strides,
        }
    }

    pub fn view(&self) -> TensorView<T> {
        TensorView::new(self.data(), &self.layout)
    }

    pub fn view_mut(&mut self) -> TensorViewMut<T> {
        // We slice `self.data` here rather than using `self.data_mut()` to
        // avoid a borrow-checker complaint.
        let data = &mut self.data[self.base..];
        TensorViewMut::new(data, &self.layout)
    }
}

impl Tensor<f32> {
    /// Serialize the tensor to a simple binary format.
    ///
    /// The serialized data is in little-endian order and has the structure:
    ///
    /// [rank: u32][dim: u32 * rank][element: T * product(dims)]
    ///
    /// Where `T` is the tensor's element type.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let ndim: u32 = self.ndim() as u32;
        writer.write_all(&ndim.to_le_bytes())?;
        for &dim in self.shape() {
            writer.write_all(&(dim as u32).to_le_bytes())?;
        }
        for el in self.iter() {
            writer.write_all(&el.to_le_bytes())?;
        }
        Ok(())
    }
}

impl<T: Copy + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.iter().eq(other.iter())
    }
}

impl<T: Copy> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        let data = self.data.clone();
        Tensor {
            data,
            base: self.base,
            layout: self.layout.clone(),
        }
    }
}

impl<T: Copy> FromIterator<T> for Tensor<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<_> = FromIterator::from_iter(iter);
        Tensor::from_vec(data)
    }
}

impl<I: TensorIndex, T: Copy> Index<I> for Tensor<T> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        &self.data[self.base + self.offset(index)]
    }
}

impl<I: TensorIndex, T: Copy> IndexMut<I> for Tensor<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let offset = self.base + self.offset(index);
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

/// Create a new tensor with all values set to 0.
pub fn zeros<T: Copy + Default>(shape: &[usize]) -> Tensor<T> {
    Tensor::zeros(shape)
}

/// Create a new tensor filled with random values supplied by `rng`.
#[cfg(test)]
pub fn rand(shape: &[usize], rng: &mut XorShiftRNG) -> Tensor {
    let mut t = zeros(shape);
    t.data.fill_with(|| rng.next_f32());
    t
}

/// Create a new tensor with a given shape and values
pub fn from_data<T: Copy>(shape: Vec<usize>, data: Vec<T>) -> Tensor<T> {
    Tensor::from_data(shape, data)
}

/// Create a new 0-dimensional (scalar) tensor from a single value.
#[cfg(test)]
pub fn from_scalar<T: Copy>(value: T) -> Tensor<T> {
    Tensor::from_scalar(value)
}

/// Create a new 1-dimensional tensor from a vector
#[cfg(test)]
pub fn from_vec<T: Copy>(data: Vec<T>) -> Tensor<T> {
    Tensor::from_vec(data)
}

/// Create a new 2D tensor from a nested array of slices.
#[cfg(test)]
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
    use std::borrow::Cow;
    use std::ops::IndexMut;

    use crate::rng::XorShiftRNG;
    use crate::tensor::{
        from_2d_slice, from_data, from_scalar, from_vec, rand, zeros, SliceRange, Tensor,
    };

    /// Create a tensor where the value of each element is its logical index
    /// plus one.
    fn steps(shape: &[usize]) -> Tensor<i32> {
        let mut x = zeros(shape);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = (index + 1) as i32;
        }
        x
    }

    #[test]
    fn test_apply() {
        let mut x = steps(&[3, 3]);
        x.apply(|el| el * el);
        let expected = from_data(vec![3, 3], vec![1, 4, 9, 16, 25, 36, 49, 64, 81]);
        assert_eq!(x, expected);
    }

    #[test]
    fn test_clip_dim() {
        let mut x = steps(&[3, 3]);
        x.clip_dim(0, 1..2);
        x.clip_dim(1, 1..2);
        assert_eq!(x.iter().collect::<Vec<i32>>(), vec![5]);
    }

    #[test]
    fn test_clip_dim_start() {
        let mut x = steps(&[3, 3]);

        // Clip the start of the tensor, adjusting the `base` offset.
        x.clip_dim(0, 1..3);

        // Indexing should reflect the slice.
        assert_eq!(x.iter().collect::<Vec<i32>>(), &[4, 5, 6, 7, 8, 9]);
        assert_eq!(x[[0, 0]], 4);
        assert_eq!(*x.index_mut([0, 0]), 4);

        // Slices returned by `data`, `data_mut` should reflect the slice.
        assert_eq!(x.data(), &[4, 5, 6, 7, 8, 9]);
        assert_eq!(x.data_mut(), &[4, 5, 6, 7, 8, 9]);

        // Offsets should be relative to the sliced returned by `data`,
        // `data_mut`.
        assert_eq!(x.offsets().collect::<Vec<usize>>(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(x.offset([0, 0]), 0);
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
        assert_eq!(x.shape().len(), 0);
        assert_eq!(x.data(), &[5]);
    }

    #[test]
    fn test_from_vec() {
        let x = from_vec(vec![1, 2, 3]);
        assert_eq!(x.shape(), &[3]);
        assert_eq!(x.data(), &[1, 2, 3]);
    }

    #[test]
    fn test_from_iterator() {
        let x: Tensor<i32> = FromIterator::from_iter(0..10);
        assert_eq!(x.shape(), &[10]);
        assert_eq!(x.data(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_stride() {
        let x = zeros::<f32>(&[2, 5, 7, 3]);
        assert_eq!(x.stride(3), 1);
        assert_eq!(x.stride(2), 3);
        assert_eq!(x.stride(1), 7 * 3);
        assert_eq!(x.stride(0), 5 * 7 * 3);
    }

    #[test]
    fn test_index() {
        let mut x = zeros::<f32>(&[2, 2]);

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
        let mut x = zeros::<f32>(&[2, 2]);

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
        let x = zeros::<f32>(&[2, 2]);
        x[[2, 0]];
    }

    #[test]
    #[should_panic]
    fn test_index_panics_if_wrong_dim_count() {
        let x = zeros::<f32>(&[2, 2]);
        x[[0, 0, 0]];
    }

    #[test]
    fn test_indices() {
        let x = zeros::<f32>(&[2, 2]);
        let x_indices = {
            let mut indices = Vec::new();
            let mut iter = x.indices();
            while let Some(index) = iter.next() {
                indices.push(index.to_vec());
            }
            indices
        };
        assert_eq!(
            x_indices,
            &[vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1],]
        );
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
        let x = zeros::<f32>(&[10, 5, 3, 7]);
        let [i, j, k, l] = x.dims();

        assert_eq!(i, 10);
        assert_eq!(j, 5);
        assert_eq!(k, 3);
        assert_eq!(l, 7);
    }

    #[test]
    #[should_panic]
    fn test_dims_panics_if_wrong_array_length() {
        let x = zeros::<f32>(&[10, 5, 3, 7]);
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
    fn test_is_empty() {
        assert!(from_vec::<f32>(vec![]).is_empty());
        assert!(!from_vec(vec![1]).is_empty());
        assert!(!from_scalar(5.0).is_empty());
    }

    #[test]
    fn test_reshape() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = rand(&[10, 5, 3, 7], &mut rng);
        let x_data: Vec<f32> = x.data().into();

        assert_eq!(x.shape(), &[10, 5, 3, 7]);

        x.reshape(&[10, 5, 3 * 7]);

        assert_eq!(x.shape(), &[10, 5, 3 * 7]);
        assert_eq!(x.data(), x_data);
    }

    #[test]
    fn test_reshape_non_contiguous() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = rand(&[10, 10], &mut rng);

        // Set the input up so that it is non-contiguous and has a non-zero
        // `base` offset.
        x.permute(&[1, 0]);
        x.clip_dim(0, 2..8);

        // Reshape the tensor. This should copy the data and reset the `base`
        // offset.
        x.reshape(&[x.shape().iter().product()]);

        // After reshaping, we should be able to successfully read all the elements.
        // Note this test doesn't check that the correct elements were read.
        let elts: Vec<_> = x.iter().collect();
        assert_eq!(elts.len(), 60);

        // Set up another input so it is non-contiguous and has a non-zero `base` offset.
        let mut x = steps(&[3, 3]);
        x.clip_dim(0, 1..3);
        x.clip_dim(1, 1..3);

        // Flatten the input with reshape.
        x.reshape(&[4]);

        // Check that the correct elements were read.
        assert_eq!(x.iter().collect::<Vec<i32>>(), &[5, 6, 8, 9]);
    }

    #[test]
    fn test_reshape_copies_with_custom_strides() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = rand(&[10, 10], &mut rng);

        // Give the tensor a non-default stride
        x.clip_dim(1, 0..8);
        assert!(!x.is_contiguous());
        let x_elements: Vec<f32> = x.iter().collect();

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
        let mut x = rand(&[10, 5, 3, 7], &mut rng);
        x.reshape(&[10, 5]);
    }

    #[test]
    fn test_permute() {
        // Test with a vector (this is a no-op)
        let mut input = steps(&[5]);
        assert!(input.iter().eq([1, 2, 3, 4, 5].iter().copied()));
        input.permute(&[0]);
        assert!(input.iter().eq([1, 2, 3, 4, 5].iter().copied()));

        // Test with a matrix (ie. transpose the matrix)
        let mut input = steps(&[2, 3]);
        assert!(input.iter().eq([1, 2, 3, 4, 5, 6].iter().copied()));
        input.permute(&[1, 0]);
        assert_eq!(input.shape(), &[3, 2]);
        assert!(input.iter().eq([1, 4, 2, 5, 3, 6].iter().copied()));

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
    fn test_insert_dim() {
        let mut input = steps(&[2, 3]);
        input.insert_dim(1);
        assert_eq!(input.shape(), &[2, 1, 3]);

        input.insert_dim(1);
        assert_eq!(input.shape(), &[2, 1, 1, 3]);

        input.insert_dim(0);
        assert_eq!(input.shape(), &[1, 2, 1, 1, 3]);
    }

    #[test]
    fn test_clone_with_shape() {
        let mut rng = XorShiftRNG::new(1234);
        let x = rand(&[10, 5, 3, 7], &mut rng);
        let y = x.clone_with_shape(&[10, 5, 3 * 7]);

        assert_eq!(y.shape(), &[10, 5, 3 * 7]);
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_unchecked_view() {
        let mut rng = XorShiftRNG::new(1234);
        let x = rand(&[10, 5, 3, 7], &mut rng);
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
        let mut x = rand(&[10, 5, 3, 7], &mut rng);

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
        let x = rand(&[10, 5, 3, 7], &mut rng);
        let x_slice = x.last_dim_slice([5, 3, 2, 0], x.shape()[3]);

        for i in 0..x.shape()[3] {
            assert_eq!(x[[5, 3, 2, i]], x_slice[i]);
        }
    }

    #[test]
    fn test_last_dim_slice_mut() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = rand(&[10, 5, 3, 7], &mut rng);
        let x_slice = x.last_dim_slice_mut([5, 3, 2, 0], x.shape()[3]);

        for val in x_slice.iter_mut() {
            *val = 1.0;
        }

        for i in 0..x.shape()[3] {
            assert_eq!(x[[5, 3, 2, i]], 1.0);
        }
    }

    #[test]
    fn test_iter_for_contiguous_array() {
        for dims in 1..7 {
            let mut shape = Vec::new();
            for d in 0..dims {
                shape.push(d + 1);
            }
            let mut rng = XorShiftRNG::new(1234);
            let x = rand(&shape, &mut rng);

            let elts: Vec<f32> = x.iter().collect();

            assert_eq!(x.data(), elts);
        }
    }

    #[test]
    fn test_iter_for_empty_array() {
        let empty = zeros::<f32>(&[3, 0, 5]);
        assert!(empty.iter().next().is_none());
    }

    #[test]
    fn test_iter_for_non_contiguous_array() {
        let mut x = zeros(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        // Initially tensor is contiguous, so data buffer and element sequence
        // match.
        assert_eq!(x.data(), x.iter().collect::<Vec<_>>());

        // Slice the tensor along an outer dimension. This will leave the tensor
        // contiguous, and hence `data` and `elements` should return the same
        // elements.
        x.clip_dim(0, 0..2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(x.iter().collect::<Vec<_>>(), &[1, 2, 3, 4, 5, 6]);
        // Test with step > 1 to exercise `Elements::nth`.
        assert_eq!(x.iter().step_by(2).collect::<Vec<_>>(), &[1, 3, 5]);

        // Slice the tensor along an inner dimension. The tensor will no longer
        // be contiguous and hence `elements` will return different results than
        // `data`.
        x.clip_dim(1, 0..2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(x.iter().collect::<Vec<_>>(), &[1, 2, 4, 5]);
        // Test with step > 1 to exercise `Elements::nth`.
        assert_eq!(x.iter().step_by(2).collect::<Vec<_>>(), &[1, 4]);
    }

    // PyTorch and numpy do not allow iteration over a scalar, but it seems
    // consistent for `Tensor::iter` to always yield `Tensor::len` elements,
    // and `len` returns 1 for a scalar.
    #[test]
    fn test_iter_for_scalar() {
        let x = from_scalar(5.0);
        let elements = x.iter().collect::<Vec<_>>();
        assert_eq!(&elements, &[5.0]);
    }

    #[test]
    fn test_iter_mut_for_contiguous_array() {
        for dims in 1..7 {
            let mut shape = Vec::new();
            for d in 0..dims {
                shape.push(d + 1);
            }
            let mut rng = XorShiftRNG::new(1234);
            let mut x = rand(&shape, &mut rng);

            let elts: Vec<f32> = x.iter().map(|x| x * 2.).collect();

            for elt in x.view_mut().iter_mut() {
                *elt *= 2.;
            }

            assert_eq!(x.data(), elts);
        }
    }

    #[test]
    fn test_iter_mut_for_non_contiguous_array() {
        let mut x = zeros(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }
        x.permute(&[1, 0]);

        let x_doubled: Vec<usize> = x.iter().map(|x| x * 2).collect();
        for elt in x.view_mut().iter_mut() {
            *elt *= 2;
        }
        assert_eq!(x.to_vec(), x_doubled);
    }

    #[test]
    fn test_to_vec() {
        let mut x = steps(&[3, 3]);

        // Contiguous case. This should use the fast-path.
        assert_eq!(x.to_vec(), x.iter().collect::<Vec<_>>());

        // Non-contiguous case.
        x.clip_dim(1, 0..2);
        assert!(!x.is_contiguous());
        assert_eq!(x.to_vec(), x.iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_offsets() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = rand(&[10, 10], &mut rng);

        let x_elts: Vec<_> = x.iter().collect();

        let x_offsets = x.offsets();
        let x_data = x.data_mut();
        let x_elts_from_offset: Vec<_> = x_offsets.map(|off| x_data[off]).collect();

        assert_eq!(x_elts, x_elts_from_offset);
    }

    #[test]
    fn test_offsets_nth() {
        let x = steps(&[3]);
        let mut iter = x.offsets();
        assert_eq!(iter.nth(0), Some(0));
        assert_eq!(iter.nth(0), Some(1));
        assert_eq!(iter.nth(0), Some(2));
        assert_eq!(iter.nth(0), None);

        let x = steps(&[10]);
        let mut iter = x.offsets();
        assert_eq!(iter.nth(1), Some(1));
        assert_eq!(iter.nth(5), Some(7));
        assert_eq!(iter.nth(1), Some(9));
        assert_eq!(iter.nth(0), None);
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
        let mut x = zeros(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        // Freshly-allocated tensor
        assert!(x.is_contiguous());

        // Tensor where outermost dimension has been clipped at the end.
        let mut y = x.clone();
        y.clip_dim(0, 0..2);
        assert!(y.is_contiguous());
        assert_eq!(y.data(), &[1, 2, 3, 4, 5, 6]);

        // Tensor where outermost dimension has been clipped at the start.
        let mut y = x.clone();
        y.clip_dim(0, 1..3);
        assert!(y.is_contiguous());
        assert_eq!(y.data(), &[4, 5, 6, 7, 8, 9]);

        // Tensor where inner dimension has been clipped at the start.
        let mut y = x.clone();
        y.clip_dim(1, 1..3);
        assert!(!y.is_contiguous());

        // Tensor where inner dimension has been clipped at the end.
        let mut y = x.clone();
        y.clip_dim(1, 0..2);
        assert!(!y.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_1d() {
        let mut x = zeros(&[10]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        assert!(x.is_contiguous());
        x.clip_dim(0, 0..5);
        assert!(x.is_contiguous());
    }

    #[test]
    fn test_make_contiguous() {
        let mut x = steps(&[3, 3]);
        assert!(x.is_contiguous());

        // Clip outer dimension at start. This will modify the base offset.
        x.clip_dim(0, 1..3);

        // Clip inner dimension at start. This will modify the strides.
        x.clip_dim(1, 1..3);
        assert!(!x.is_contiguous());

        x.make_contiguous();
        assert!(x.is_contiguous());
        assert_eq!(x.iter().collect::<Vec<i32>>(), &[5, 6, 8, 9]);
    }

    #[test]
    fn test_as_contiguous() {
        let mut x = steps(&[3, 3]);
        let y = x.as_contiguous();
        assert!(matches!(y, Cow::Borrowed(_)));

        x.permute(&[1, 0]);
        let y = x.as_contiguous();
        assert!(matches!(y, Cow::Owned(_)));
    }

    #[test]
    fn test_broadcast_elements() {
        let x = steps(&[1, 2, 1, 2]);
        assert_eq!(x.iter().collect::<Vec<i32>>(), &[1, 2, 3, 4]);

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
    #[should_panic(expected = "Cannot broadcast to specified shape")]
    fn test_broadcast_elements_with_invalid_shape() {
        let x = steps(&[2, 2]);
        x.broadcast_elements(&[3, 2]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast to specified shape")]
    fn test_broadcast_elements_with_shorter_shape() {
        let x = steps(&[2, 2]);
        x.broadcast_elements(&[4]);
    }

    #[test]
    fn test_broadcast_offsets() {
        let x = steps(&[2, 1, 4]);
        let to_shape = &[2, 2, 1, 4];

        let expected: Vec<i32> = x.broadcast_elements(to_shape).collect();
        let actual: Vec<i32> = x
            .broadcast_offsets(to_shape)
            .map(|off| x.data()[off])
            .collect();

        assert_eq!(&actual, &expected);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast to specified shape")]
    fn test_broadcast_offsets_with_invalid_shape() {
        let x = steps(&[2, 2]);
        x.broadcast_offsets(&[3, 2]);
    }

    #[test]
    fn test_can_broadcast_to() {
        let x = steps(&[1, 5, 10]);
        assert!(x.can_broadcast_to(&[2, 5, 10]));
        assert!(x.can_broadcast_to(&[1, 5, 10]));
        assert!(!x.can_broadcast_to(&[1, 1, 10]));
    }

    #[test]
    fn test_can_broadcast_with() {
        let x = steps(&[1, 5, 10]);
        assert!(x.can_broadcast_with(&[2, 5, 10]));
        assert!(x.can_broadcast_with(&[1, 5, 10]));
        assert!(x.can_broadcast_with(&[1, 1, 10]));
    }

    #[test]
    fn test_slice_elements() {
        let sr = |start, end| SliceRange::new(start, end, 1);
        let x = steps(&[3, 3]);

        // Slice that removes start of each dimension
        let slice: Vec<_> = x.slice_elements(&[sr(1, 3), sr(1, 3)]).collect();
        assert_eq!(slice, &[5, 6, 8, 9]);

        // Slice that removes end of each dimension
        let slice: Vec<_> = x.slice_elements(&[sr(0, 2), sr(0, 2)]).collect();
        assert_eq!(slice, &[1, 2, 4, 5]);

        // Slice that removes start and end of first dimension
        let slice: Vec<_> = x.slice_elements(&[sr(1, 2), sr(0, 3)]).collect();
        assert_eq!(slice, &[4, 5, 6]);

        // Slice that removes start and end of second dimension
        let slice: Vec<_> = x.slice_elements(&[sr(0, 3), sr(1, 2)]).collect();
        assert_eq!(slice, &[2, 5, 8]);
    }

    #[test]
    fn test_slice_elements_with_step() {
        let sr = SliceRange::new;
        let x = steps(&[10]);

        // Positive steps > 1.
        let slice: Vec<_> = x.slice_elements(&[sr(0, 10, 2)]).collect();
        assert_eq!(slice, &[1, 3, 5, 7, 9]);

        let slice: Vec<_> = x.slice_elements(&[sr(0, 10, 3)]).collect();
        assert_eq!(slice, &[1, 4, 7, 10]);

        let slice: Vec<_> = x.slice_elements(&[sr(0, 10, 10)]).collect();
        assert_eq!(slice, &[1]);

        // Negative steps.
        let slice: Vec<_> = x.slice_elements(&[sr(10, -11, -1)]).collect();
        assert_eq!(slice, &[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);

        let slice: Vec<_> = x.slice_elements(&[sr(8, 0, -1)]).collect();
        assert_eq!(slice, &[9, 8, 7, 6, 5, 4, 3, 2]);

        let slice: Vec<_> = x.slice_elements(&[sr(10, 0, -2)]).collect();
        assert_eq!(slice, &[10, 8, 6, 4, 2]);

        let slice: Vec<_> = x.slice_elements(&[sr(10, 0, -10)]).collect();
        assert_eq!(slice, &[10]);
    }

    #[test]
    fn test_slice_elements_negative_indices() {
        let sr = |start, end| SliceRange::new(start, end, 1);
        let x = steps(&[10]);

        // Negative start
        let slice: Vec<_> = x.slice_elements(&[sr(-2, 10)]).collect();
        assert_eq!(slice, &[9, 10]);

        // Negative end
        let slice: Vec<_> = x.slice_elements(&[sr(7, -1)]).collect();
        assert_eq!(slice, &[8, 9]);

        // Negative start and end
        let slice: Vec<_> = x.slice_elements(&[sr(-3, -1)]).collect();
        assert_eq!(slice, &[8, 9]);
    }

    #[test]
    fn test_slice_elements_clamps_indices() {
        let sr = SliceRange::new;
        let x = steps(&[5]);

        // Test cases for positive steps (ie. traversing forwards).

        // Positive start out of bounds
        let slice: Vec<_> = x.slice_elements(&[sr(10, 11, 1)]).collect();
        assert_eq!(slice.len(), 0);

        // Positive end out of bounds
        let slice: Vec<_> = x.slice_elements(&[sr(0, 10, 1)]).collect();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        // Negative start out of bounds
        let slice: Vec<_> = x.slice_elements(&[sr(-10, 5, 1)]).collect();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        // Negative end out of bounds
        let slice: Vec<_> = x.slice_elements(&[sr(-10, -5, 1)]).collect();
        assert_eq!(slice.len(), 0);

        // Test cases for negative steps (ie. traversing backwards).

        // Positive start out of bounds
        let slice: Vec<_> = x.slice_elements(&[sr(10, -6, -1)]).collect();
        assert_eq!(slice, &[5, 4, 3, 2, 1]);

        // Positive end out of bounds
        let slice: Vec<_> = x.slice_elements(&[sr(0, 10, -1)]).collect();
        assert_eq!(slice.len(), 0);

        // Negative start out of bounds
        let slice: Vec<_> = x.slice_elements(&[sr(-10, 5, -1)]).collect();
        assert_eq!(slice.len(), 0);

        // Negative end out of bounds
        let slice: Vec<_> = x.slice_elements(&[sr(-1, -10, -1)]).collect();
        assert_eq!(slice, &[5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_slice_elements_start_end_step_combinations() {
        let sr = SliceRange::new;
        let x = steps(&[3]);

        // Test various combinations of slice starts, ends and steps that are
        // positive and negative, in-bounds and out-of-bounds, and ensure they
        // don't cause a panic.
        for start in -5..5 {
            for end in -5..5 {
                for step in -5..5 {
                    if step == 0 {
                        continue;
                    }
                    x.slice_elements(&[sr(start, end, step)]).for_each(drop);
                }
            }
        }
    }

    // These tests assume the correctness of `slice_elements`, given the tests
    // above, and check for consistency between the results of `slice_offsets`
    // and `slice_elements`.
    #[test]
    fn test_slice_offsets() {
        let x = steps(&[5, 5]);

        // Range that removes the start and end of each dimension.
        let range = &[SliceRange::new(1, 4, 1), SliceRange::new(1, 4, 1)];
        let expected: Vec<_> = x.slice_elements(range).collect();
        let result: Vec<_> = x
            .slice_offsets(range)
            .map(|offset| x.data()[offset])
            .collect();

        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_write() -> std::io::Result<()> {
        use std::io::{Cursor, Read};
        let x = from_data(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let mut buf: Vec<u8> = Vec::new();

        x.write(&mut buf)?;

        assert_eq!(buf.len(), 4 + x.ndim() * 4 + x.len() * 4);

        let mut cursor = Cursor::new(buf);
        let mut tmp = [0u8; 4];

        cursor.read(&mut tmp)?;
        let ndim = u32::from_le_bytes(tmp);
        assert_eq!(ndim, x.ndim() as u32);

        for &size in x.shape().iter() {
            cursor.read(&mut tmp)?;
            let written_size = u32::from_le_bytes(tmp);
            assert_eq!(written_size, size as u32);
        }

        for el in x.iter() {
            cursor.read(&mut tmp)?;
            let written_el = f32::from_le_bytes(tmp);
            assert_eq!(written_el, el);
        }

        Ok(())
    }
}
