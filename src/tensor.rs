use std::borrow::Cow;
use std::fmt::Debug;
use std::io;
use std::iter::{repeat, zip, Cycle, Take};
use std::ops::{Index, IndexMut, Range, RangeTo};
use std::slice::Iter;

#[cfg(test)]
use crate::rng::XorShiftRNG;

/// A range for slicing a Tensor.
///
/// This has two main differences from a standard Rust range (`std::ops::Range`):
///
/// - A non-zero step between indices can be specified. The step can be negative,
///   which means that the dimension should be traversed in reverse order.
/// - The `start` and `end` indexes can also be negative, in which case they
///   count backwards from the end of the array.
///
/// This system for specifying slicing and indexing follows NumPy, which in
/// turn strongly influenced slicing in ONNX.
#[derive(Clone, Copy, Debug)]
pub struct SliceRange {
    pub start: isize,
    pub end: isize,

    /// The steps between adjacent elements selected by this range. This
    /// is private so this module can enforce the invariant that it is non-zero.
    step: isize,
}

impl SliceRange {
    /// Create a new range from `start` to `end`. The `start` index is inclusive
    /// and the `end` value is exclusive.
    ///
    /// Panicks if the `step` size is 0.
    pub fn new(start: isize, end: isize, step: isize) -> SliceRange {
        assert!(step != 0, "Slice step cannot be 0");
        SliceRange { start, end, step }
    }

    /// Return the number of elements that would be retained if using this range
    /// to slice a dimension of size `dim_size`.
    pub fn steps(&self, dim_size: usize) -> usize {
        let clamped = self.clamp(dim_size);

        let start_idx = Self::offset_from_start(clamped.start, dim_size);
        let end_idx = Self::offset_from_start(clamped.end, dim_size);

        if (clamped.step > 0 && end_idx <= start_idx) || (clamped.step < 0 && end_idx >= start_idx)
        {
            return 0;
        }

        let steps = if clamped.step > 0 {
            1 + (end_idx - start_idx - 1) / clamped.step
        } else {
            1 + (start_idx - end_idx - 1) / -clamped.step
        };

        steps.max(0) as usize
    }

    /// Return a copy of this range with indexes adjusted so that they are valid
    /// for a tensor dimension of size `dim_size`.
    ///
    /// Valid indexes depend on direction that the dimension is traversed
    /// (forwards if `self.step` is positive or backwards if negative). They
    /// start at the first element going in that direction and end after the
    /// last element.
    pub fn clamp(&self, dim_size: usize) -> SliceRange {
        let len = dim_size as isize;

        let min_idx;
        let max_idx;

        if self.step > 0 {
            // When traversing forwards, the range of valid +ve indexes is `[0,
            // len]` and for -ve indexes `[-len, -1]`.
            min_idx = -len;
            max_idx = len;
        } else {
            // When traversing backwards, the range of valid +ve indexes are
            // `[0, len-1]` and for -ve indexes `[-len-1, -1]`.
            min_idx = -len - 1;
            max_idx = len - 1;
        }

        SliceRange::new(
            self.start.clamp(min_idx, max_idx),
            self.end.clamp(min_idx, max_idx),
            self.step,
        )
    }

    /// Resolve the range endpoints to positive indices in `[0, dim_size)`.
    ///
    /// If `self.step` is positive, the returned range counts forwards from
    /// the first index of the dimension, otherwise it counts backwards from
    /// the last index.
    pub fn resolve(&self, dim_size: usize) -> Range<usize> {
        let clamped = self.clamp(dim_size);

        let offset_fn = if self.step > 0 {
            Self::offset_from_start
        } else {
            Self::offset_from_end
        };

        let start = offset_fn(clamped.start, dim_size) as usize;
        let end = offset_fn(clamped.end, dim_size) as usize;

        start..end
    }

    /// Resolve an index to an offset from the first index of the dimension.
    fn offset_from_start(index: isize, dim_size: usize) -> isize {
        if index >= 0 {
            index
        } else {
            dim_size as isize + index
        }
    }

    /// Resolve an index to an offset from the last index of the dimension.
    fn offset_from_end(index: isize, dim_size: usize) -> isize {
        if index >= 0 {
            dim_size as isize - 1 - index
        } else {
            dim_size as isize + index
        }
    }
}

impl<T> From<Range<T>> for SliceRange
where
    T: TryInto<isize>,
    <T as TryInto<isize>>::Error: Debug,
{
    fn from(r: Range<T>) -> SliceRange {
        let start = r.start.try_into().unwrap();
        let end = r.end.try_into().unwrap();
        SliceRange::new(start, end, 1)
    }
}

impl<T> From<RangeTo<T>> for SliceRange
where
    T: TryInto<isize>,
    <T as TryInto<isize>>::Error: Debug,
{
    fn from(r: RangeTo<T>) -> SliceRange {
        let end = r.end.try_into().unwrap();
        SliceRange::new(0, end, 1)
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
    /// Create a new zero-filled tensor with a given shape.
    pub fn zeros(shape: &[usize]) -> Tensor<T>
    where
        T: Default,
    {
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
        let strides = strides_for_shape(&shape);
        Tensor {
            data,
            base: 0,
            shape,
            strides,
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
        let data = if self.is_contiguous() {
            self.data().into()
        } else {
            self.elements().collect()
        };
        Self::from_data(shape.into(), data)
    }

    /// Return an iterator over all valid indices in this tensor.
    ///
    /// The returned iterator does not implement the `Iterator` trait but has
    /// a similar API. See `IndexIterator` docs.
    pub fn indices(&self) -> IndexIterator {
        IndexIterator::from_shape(&self.shape)
    }

    /// Return the total number of elements in this tensor.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return true if this tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of dimensions the tensor has, aka. the rank of the
    /// tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Clip dimension `dim` to `[range.start, range.end)`. The new size for
    /// the dimension must be <= the old size.
    ///
    /// This is a fast operation since it just alters the start offset within
    /// the tensor's element buffer and length of the specified dimension.
    pub fn clip_dim(&mut self, dim: usize, range: Range<usize>) {
        let (start, end) = (range.start, range.end);

        assert!(start <= end, "start must be <= end");
        assert!(end <= self.shape[dim], "end must be <= dim size");

        self.base += self.strides[dim] * start;
        self.shape[dim] = end - start;

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
            self.strides[N - 1] == 1,
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
            self.strides[N - 1] == 1,
            "last_dim_slice_mut requires contiguous last dimension"
        );
        let offset = self.base + self.offset(index);
        &mut self.data[offset..offset + len]
    }

    /// Return a copy of the elements of this tensor as a contiguous vector
    /// in row-major order.
    ///
    /// This is slightly more efficient than `elements().collect()` in the case
    /// where the tensor is already contiguous.
    pub fn elements_vec(&self) -> Vec<T> {
        if self.is_contiguous() {
            self.data().to_vec()
        } else {
            self.elements().collect()
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
        &self.shape
    }

    /// Return true if the logical order of elements in this tensor matches the
    /// order of elements in the slice returned by `data()` and `data_mut()`,
    /// with no gaps.
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

    /// Convert the internal layout of elements to be contiguous, as reported
    /// by `is_contiguous`.
    ///
    /// This is a no-op if the tensor is already contiguous.
    pub fn make_contiguous(&mut self) {
        if self.is_contiguous() {
            return;
        }
        self.data = self.elements().collect();
        self.base = 0;
        self.strides = strides_for_shape(&self.shape);
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
    pub fn elements(&self) -> Elements<T> {
        Elements::new(self)
    }

    /// Return an iterator over offsets of elements in this tensor, in their
    /// logical order.
    ///
    /// See also the notes for `slice_offsets`.
    pub fn offsets(&self) -> Offsets {
        Offsets::new(self)
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
    pub fn broadcast_elements(&self, shape: &[usize]) -> BroadcastElements<'_, T> {
        if !self.can_broadcast_to(shape) {
            panic!("Cannot broadcast to specified shape");
        }
        BroadcastElements::new(self, shape)
    }

    /// Return an iterator over offsets of this tensor, broadcasted to `shape`.
    ///
    /// This is very similar to `broadcast_elements`, except that the iterator
    /// yields offsets into rather than elements of the data buffer.
    pub fn broadcast_offsets(&self, shape: &[usize]) -> Offsets {
        if !self.can_broadcast_to(shape) {
            panic!("Cannot broadcast to specified shape");
        }
        Offsets::broadcast(self, shape)
    }

    /// Return true if the element's shape can be broadcast to `shape` using
    /// `broadcast_elements`. The result of the broadcasted tensor will have
    /// exactly the shape `shape`.
    ///
    /// See <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md> for
    /// conditions in which broadcasting is allowed.
    pub fn can_broadcast_to(&self, shape: &[usize]) -> bool {
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

        zip(self_dims, target_dims).all(|(a, b)| a == b || a == 1)
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
        if self.shape == shape {
            return true;
        }

        // For two shapes to be compatible for broadcasting, each dimension must
        // either be the same or be 1.
        //
        // If the tensor has fewer dimensions, pretend that it was prefixed with
        // 1-length dimensions to make the dimension counts equal.

        let a = self.shape.as_slice();
        let b = shape;

        let a_pad = b.len().saturating_sub(a.len());
        let b_pad = a.len().saturating_sub(b.len());

        let a_iter = a.iter().copied().rev().chain(repeat(1).take(a_pad));
        let b_iter = b.iter().copied().rev().chain(repeat(1).take(b_pad));

        zip(a_iter, b_iter).all(|(a, b)| a == b || a == 1 || b == 1)
    }

    /// Return an iterator over a subset of elements in this tensor.
    pub fn slice_elements(&self, ranges: &[SliceRange]) -> Elements<T> {
        Elements::slice(self, ranges)
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
        Offsets::slice(self, ranges)
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

    /// Insert a dimension of size one at index `dim`.
    pub fn insert_dim(&mut self, dim: usize) {
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape);
    }

    /// Return the number of elements between successive entries in the `dim`
    /// dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    /// Return the offset of an element in the slices returned by `data`
    /// and `data_mut`.
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
        let mut offset = 0;
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
            data: self.data(),
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
            data: self.data_mut(),
            offset,
            strides,
        }
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
        for &dim in self.shape.iter() {
            writer.write_all(&(dim as u32).to_le_bytes())?;
        }
        for el in self.elements() {
            writer.write_all(&el.to_le_bytes())?;
        }
        Ok(())
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

/// IterPos tracks the position within a single dimension of an IndexingIter.
#[derive(Debug)]
struct IterPos {
    /// Current step along this dimension. Each step corresponds to advancing
    /// one or more indexes either forwards or backwards.
    step: usize,

    /// Number of steps to take along this dimension before resetting.
    steps: usize,

    /// Adjustment for element buffer offset for each step along this dimension.
    offset_step: isize,
}

impl IterPos {
    fn step(&mut self) -> bool {
        if self.step < self.steps - 1 {
            self.step += 1;
            true
        } else {
            false
        }
    }
}

/// Helper for iterating over offsets of elements in a tensor.
#[derive(Debug)]
struct IndexingIterBase {
    /// Remaining elements to visit
    len: usize,

    /// Offset of the next element to return from the tensor's element buffer.
    offset: isize,

    /// Current position within each dimension.
    pos: Vec<IterPos>,
}

impl IndexingIterBase {
    /// Create an iterator over element offsets in `tensor`.
    fn new<T: Copy>(tensor: &Tensor<T>) -> IndexingIterBase {
        let dims = tensor
            .shape
            .iter()
            .enumerate()
            .map(|(dim, &len)| IterPos {
                step: 0,
                steps: len,
                offset_step: tensor.strides[dim] as isize,
            })
            .collect();

        IndexingIterBase {
            len: tensor.len(),
            offset: 0,
            pos: dims,
        }
    }

    /// Create an iterator over offsets of elements in `tensor`, as if it had
    /// a given `shape`. This will repeat offsets as necessary.
    fn broadcast<T: Copy>(tensor: &Tensor<T>, shape: &[usize]) -> IndexingIterBase {
        // nb. We require that the broadcast shape has a length >= the actual
        // shape.
        let added_dims = shape.len() - tensor.shape().len();
        let padded_tensor_shape = repeat(&0).take(added_dims).chain(tensor.shape().iter());
        let dims = zip(padded_tensor_shape, shape.iter())
            .enumerate()
            .map(|(dim, (&actual_len, &broadcast_len))| IterPos {
                step: 0,
                steps: broadcast_len,

                // If the dimension is being broadcast, set its stride to 0 so
                // that when we increment in this dimension, we just repeat
                // elements. Otherwise, use the real stride.
                offset_step: if actual_len == broadcast_len {
                    tensor.strides[dim - added_dims] as isize
                } else {
                    0
                },
            })
            .collect();

        IndexingIterBase {
            len: shape.iter().product(),
            offset: 0,
            pos: dims,
        }
    }

    /// Create an iterator over offsets of a subset of elements in `tensor`.
    fn slice<T: Copy>(tensor: &Tensor<T>, ranges: &[SliceRange]) -> IndexingIterBase {
        if ranges.len() != tensor.ndim() {
            panic!(
                "slice dimensions {} do not match tensor dimensions {}",
                ranges.len(),
                tensor.ndim()
            );
        }
        let mut offset = 0;
        let dims: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(dim, range)| {
                let len = tensor.shape[dim];
                let range = range.clamp(len);
                let stride = tensor.strides[dim];

                let start_index = if range.start >= 0 {
                    range.start
                } else {
                    (len as isize) + range.start
                };

                // Clamped ranges either have a start index that is valid, or
                // that is one before/after the first/last valid index
                // (depending on step direction). If invalid, the slice is
                // empty.
                if start_index >= 0 && start_index < (len as isize) {
                    offset += stride * start_index as usize;
                } else {
                    assert!(range.steps(len) == 0);
                }

                IterPos {
                    step: 0,
                    steps: range.steps(len),
                    offset_step: (stride as isize) * range.step,
                }
            })
            .collect();

        IndexingIterBase {
            len: dims.iter().map(|dim| dim.steps).product(),
            offset: offset as isize,
            pos: dims,
        }
    }

    /// Advance the iterator by stepping along dimension `dim`.
    ///
    /// The caller must calculate `stride`, the number of indices being stepped
    /// over.
    fn step_dim(&mut self, mut dim: usize, stride: usize) {
        self.len -= stride;
        let mut pos = &mut self.pos[dim];
        while !pos.step() {
            // End of range reached for dimension `dim`. Rewind offset by
            // amount it moved since iterating from the start of this dimension.
            self.offset -= pos.offset_step * (pos.steps as isize - 1);
            pos.step = 0;

            if dim == 0 {
                break;
            }

            dim -= 1;
            pos = &mut self.pos[dim];
        }
        self.offset += pos.offset_step;
    }

    /// Advance iterator by one index.
    fn step(&mut self) {
        self.step_dim(self.pos.len() - 1, 1);
    }

    /// Advance iterator by up to `n` indices.
    fn step_by(&mut self, n: usize) {
        let mut n = n.min(self.len);
        while n > 0 {
            // Find the outermost dimension that we can step along which will
            // advance the iterator by <= N elements.
            let mut dim = self.pos.len() - 1;
            let mut stride = 1;
            while dim > 0 {
                let next_stride = stride * self.pos[dim].steps;
                if next_stride >= n {
                    break;
                }
                dim -= 1;
                stride = next_stride;
            }

            // Step along the selected dimension.
            let n_steps = n / stride;
            for _ in 0..n_steps {
                n -= stride;
                self.step_dim(dim, stride);
            }
        }
    }
}

/// Iterator over elements of a tensor.
pub struct Elements<'a, T: Copy> {
    iter: ElementsIter<'a, T>,
}

/// Alternate implementations of `Elements`.
///
/// When the tensor has a contiguous layout, this iterator is just a thin
/// wrapper around a slice iterator.
enum ElementsIter<'a, T: Copy> {
    Direct(Iter<'a, T>),
    Indexing(IndexingIter<'a, T>),
}

impl<'a, T: Copy> Elements<'a, T> {
    fn new(tensor: &'a Tensor<T>) -> Elements<'a, T> {
        if tensor.is_contiguous() {
            Elements {
                iter: ElementsIter::Direct(tensor.data().iter()),
            }
        } else {
            Elements {
                iter: ElementsIter::Indexing(IndexingIter::new(tensor)),
            }
        }
    }

    fn slice(tensor: &'a Tensor<T>, ranges: &[SliceRange]) -> Elements<'a, T> {
        let iter = IndexingIter {
            base: IndexingIterBase::slice(tensor, ranges),
            data: tensor.data(),
        };
        Elements {
            iter: ElementsIter::Indexing(iter),
        }
    }
}

impl<'a, T: Copy> Iterator for Elements<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self.iter {
            ElementsIter::Direct(ref mut iter) => iter.next().copied(),
            ElementsIter::Indexing(ref mut iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.iter {
            ElementsIter::Direct(iter) => iter.size_hint(),
            ElementsIter::Indexing(iter) => iter.size_hint(),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.iter {
            ElementsIter::Direct(ref mut iter) => iter.nth(n).copied(),
            ElementsIter::Indexing(ref mut iter) => {
                iter.base.step_by(n);
                iter.next()
            }
        }
    }
}

impl<'a, T: Copy> ExactSizeIterator for Elements<'a, T> {}

struct IndexingIter<'a, T: Copy> {
    base: IndexingIterBase,

    /// Data buffer of the tensor
    data: &'a [T],
}

impl<'a, T: Copy> IndexingIter<'a, T> {
    fn new(tensor: &'a Tensor<T>) -> IndexingIter<'a, T> {
        IndexingIter {
            base: IndexingIterBase::new(tensor),
            data: tensor.data(),
        }
    }

    fn broadcast(tensor: &'a Tensor<T>, shape: &[usize]) -> IndexingIter<'a, T> {
        IndexingIter {
            base: IndexingIterBase::broadcast(tensor, shape),
            data: tensor.data(),
        }
    }
}

impl<'a, T: Copy> Iterator for IndexingIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.base.len == 0 {
            return None;
        }
        let element = self.data[self.base.offset as usize];
        self.base.step();
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for IndexingIter<'a, T> {}

/// Iterator over element offsets of a tensor.
///
/// `Offsets` does not hold a reference to the tensor, allowing the tensor to
/// be modified during iteration. It is the caller's responsibilty not to modify
/// the tensor in ways that invalidate the offset sequence returned by this
/// iterator.
pub struct Offsets {
    base: IndexingIterBase,
}

impl Offsets {
    fn new<T: Copy>(tensor: &Tensor<T>) -> Offsets {
        Offsets {
            base: IndexingIterBase::new(tensor),
        }
    }

    fn broadcast<T: Copy>(tensor: &Tensor<T>, shape: &[usize]) -> Offsets {
        Offsets {
            base: IndexingIterBase::broadcast(tensor, shape),
        }
    }

    fn slice<T: Copy>(tensor: &Tensor<T>, ranges: &[SliceRange]) -> Offsets {
        Offsets {
            base: IndexingIterBase::slice(tensor, ranges),
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
        Some(offset as usize)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.base.len, Some(self.base.len))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.base.step_by(n);
        self.next()
    }
}

impl ExactSizeIterator for Offsets {}

/// Iterator over elements of a tensor which broadcasts to a different shape.
///
/// This iterator will repeat elements of the underlying tensor until the total
/// number yielded matches the product of the shape being broadcast to.
pub struct BroadcastElements<'a, T: Copy> {
    iter: BroadcastElementsIter<'a, T>,
}

/// Alternate implementations for broadcasting. See notes in
/// `BroadcastElements::can_broadcast_by_cycling`.
enum BroadcastElementsIter<'a, T: Copy> {
    Direct(Take<Cycle<Iter<'a, T>>>),
    Indexing(IndexingIter<'a, T>),
}

impl<'a, T: Copy> BroadcastElements<'a, T> {
    fn new(tensor: &'a Tensor<T>, to_shape: &[usize]) -> BroadcastElements<'a, T> {
        let iter =
            if tensor.is_contiguous() && Self::can_broadcast_by_cycling(tensor.shape(), to_shape) {
                let iter_len = to_shape.iter().product();
                BroadcastElementsIter::Direct(tensor.data().iter().cycle().take(iter_len))
            } else {
                BroadcastElementsIter::Indexing(IndexingIter::broadcast(tensor, to_shape))
            };
        BroadcastElements { iter }
    }

    /// Return true if a tensor with shape `from_shape` can be broadcast to shape
    /// `to_shape` by cycling through all of its elements repeatedly.
    ///
    /// This requires that, after left-padding `from_shape` with 1s to match the
    /// length of `to_shape`, all non-1 dimensions in `from_shape` are contiguous
    /// at the end. For example, `[1, 5, 10]` can be broadcast to `[3, 4, 5, 10]`
    /// by cycling, but `[5, 1, 10]` cannot be broadcast to `[5, 4, 10]` this way,
    /// as the inner (`[1, 10]`) dimensions will need to be repeated 4 times before
    /// moving to the next index in the outermost dimension.
    ///
    /// If the tensor can be broadcast via cycling, and is also contiguous, it can
    /// be broadcast efficiently using `tensor.data().iter().cycle()`.
    fn can_broadcast_by_cycling(from_shape: &[usize], to_shape: &[usize]) -> bool {
        assert!(to_shape.len() >= from_shape.len());

        let excess_dims = to_shape.len() - from_shape.len();
        let mut dims_to_check = to_shape.len() - excess_dims;

        while dims_to_check > 0 {
            if from_shape[dims_to_check - 1] != to_shape[excess_dims + dims_to_check - 1] {
                break;
            }
            dims_to_check -= 1;
        }

        while dims_to_check > 0 {
            if from_shape[dims_to_check - 1] != 1 {
                return false;
            }
            dims_to_check -= 1;
        }

        true
    }
}

impl<'a, T: Copy> Iterator for BroadcastElements<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self.iter {
            BroadcastElementsIter::Direct(ref mut iter) => iter.next().copied(),
            BroadcastElementsIter::Indexing(ref mut iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.iter {
            BroadcastElementsIter::Direct(iter) => iter.size_hint(),
            BroadcastElementsIter::Indexing(iter) => iter.size_hint(),
        }
    }
}

impl<'a, T: Copy> ExactSizeIterator for BroadcastElements<'a, T> {}

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
        from_2d_slice, from_data, from_scalar, from_vec, rand, zeros, IndexIterator, SliceRange,
        Tensor,
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
    fn test_slice_range_resolve() {
        // +ve endpoints, +ve step
        assert_eq!(SliceRange::new(0, 5, 1).resolve(10), 0..5);
        assert_eq!(SliceRange::new(15, 20, 1).resolve(10), 10..10);

        // -ve endpoints, +ve step
        assert_eq!(SliceRange::new(-5, -1, 1).resolve(10), 5..9);
        assert_eq!(SliceRange::new(-20, -1, 1).resolve(10), 0..9);

        // +ve endpoints, -ve step
        assert_eq!(SliceRange::new(5, 0, -1).resolve(10), 4..9);
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
        assert_eq!(x.elements().collect::<Vec<i32>>(), vec![5]);
    }

    #[test]
    fn test_clip_dim_start() {
        let mut x = steps(&[3, 3]);

        // Clip the start of the tensor, adjusting the `base` offset.
        x.clip_dim(0, 1..3);

        // Indexing should reflect the slice.
        assert_eq!(x.elements().collect::<Vec<i32>>(), &[4, 5, 6, 7, 8, 9]);
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
        let elts: Vec<_> = x.elements().collect();
        assert_eq!(elts.len(), 60);

        // Set up another input so it is non-contiguous and has a non-zero `base` offset.
        let mut x = steps(&[3, 3]);
        x.clip_dim(0, 1..3);
        x.clip_dim(1, 1..3);

        // Flatten the input with reshape.
        x.reshape(&[4]);

        // Check that the correct elements were read.
        assert_eq!(x.elements().collect::<Vec<i32>>(), &[5, 6, 8, 9]);
    }

    #[test]
    fn test_reshape_copies_with_custom_strides() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = rand(&[10, 10], &mut rng);

        // Give the tensor a non-default stride
        x.clip_dim(1, 0..8);
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
        let mut x = rand(&[10, 5, 3, 7], &mut rng);
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
    fn test_elements_for_contiguous_array() {
        for dims in 1..7 {
            let mut shape = Vec::new();
            for d in 0..dims {
                shape.push(d + 1);
            }
            let mut rng = XorShiftRNG::new(1234);
            let x = rand(&shape, &mut rng);

            let elts: Vec<f32> = x.elements().collect();

            assert_eq!(x.data(), elts);
        }
    }

    #[test]
    fn test_elements_for_empty_array() {
        let empty = zeros::<f32>(&[3, 0, 5]);
        assert!(empty.elements().next().is_none());
    }

    #[test]
    fn test_elements_for_non_contiguous_array() {
        let mut x = zeros(&[3, 3]);
        for (index, elt) in x.data_mut().iter_mut().enumerate() {
            *elt = index + 1;
        }

        // Initially tensor is contiguous, so data buffer and element sequence
        // match.
        assert_eq!(x.data(), x.elements().collect::<Vec<_>>());

        // Slice the tensor along an outer dimension. This will leave the tensor
        // contiguous, and hence `data` and `elements` should return the same
        // elements.
        x.clip_dim(0, 0..2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(x.elements().collect::<Vec<_>>(), &[1, 2, 3, 4, 5, 6]);
        // Test with step > 1 to exercise `Elements::nth`.
        assert_eq!(x.elements().step_by(2).collect::<Vec<_>>(), &[1, 3, 5]);

        // Slice the tensor along an inner dimension. The tensor will no longer
        // be contiguous and hence `elements` will return different results than
        // `data`.
        x.clip_dim(1, 0..2);
        assert_eq!(x.data(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(x.elements().collect::<Vec<_>>(), &[1, 2, 4, 5]);
        // Test with step > 1 to exercise `Elements::nth`.
        assert_eq!(x.elements().step_by(2).collect::<Vec<_>>(), &[1, 4]);
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
    fn test_elements_vec() {
        let mut x = steps(&[3, 3]);

        // Contiguous case. This should use the fast-path.
        assert_eq!(x.elements_vec(), x.elements().collect::<Vec<_>>());

        // Non-contiguous case.
        x.clip_dim(1, 0..2);
        assert!(!x.is_contiguous());
        assert_eq!(x.elements_vec(), x.elements().collect::<Vec<_>>());
    }

    #[test]
    fn test_offsets() {
        let mut rng = XorShiftRNG::new(1234);
        let mut x = rand(&[10, 10], &mut rng);

        let x_elts: Vec<_> = x.elements().collect();

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
        assert_eq!(x.elements().collect::<Vec<i32>>(), &[5, 6, 8, 9]);
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

        for el in x.elements() {
            cursor.read(&mut tmp)?;
            let written_el = f32::from_le_bytes(tmp);
            assert_eq!(written_el, el);
        }

        Ok(())
    }
}
