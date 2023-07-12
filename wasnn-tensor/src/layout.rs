use std::iter::{repeat, zip};
use std::ops::Range;

use smallvec::SmallVec;

use crate::errors::FromDataError;
use crate::index_iterator::{DynIndices, NdIndices};
use crate::iterators::Offsets;
use crate::overlap::may_have_internal_overlap;
use crate::range::{SliceItem, SliceRange};
use crate::tensor::TensorIndex;

/// Return true if `permutation` is a valid permutation of dimensions for
/// a tensor of rank `ndim`.
pub fn is_valid_permutation(ndim: usize, permutation: &[usize]) -> bool {
    permutation.len() == ndim
        && (0..ndim).all(|dim| permutation.iter().filter(|d| **d == dim).count() == 1)
}

/// Provides methods for querying the shape and strides of a tensor.
pub trait Layout {
    /// Type used to represent indices.
    ///
    /// It is assumed that this type can also represent the shape and strides
    /// of the tensor.
    type Index<'a>: AsRef<[usize]>
    where
        Self: 'a;

    /// Iterator over indices in this tensor.
    type Indices;

    /// Return the number of dimensions.
    fn ndim(&self) -> usize;

    /// Returns the number of elements in the array.
    fn len(&self) -> usize;

    /// Return true if this layout describes a contiguous tensor, where the
    /// logical order of elements matches the order in which they are stored.
    fn is_contiguous(&self) -> bool {
        let (shape, strides) = (self.shape(), self.strides());
        let mut product = 1;
        for (dim, len) in shape.as_ref().iter().enumerate().rev() {
            if strides.as_ref()[dim] != product {
                return false;
            }
            product *= len;
        }
        true
    }

    /// Returns true if the array has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an array of the sizes of each dimension.
    fn shape(&self) -> Self::Index<'_>;

    /// Returns the size of the dimension `dim`.
    fn size(&self, dim: usize) -> usize {
        self.shape().as_ref()[dim]
    }

    /// Returns an array of the strides of each dimension.
    fn strides(&self) -> Self::Index<'_>;

    /// Returns the offset between adjacent indices along dimension `dim`.
    fn stride(&self, dim: usize) -> usize {
        self.strides().as_ref()[dim]
    }

    /// Return an iterator over all valid indices in this tensor.
    fn indices(&self) -> Self::Indices;

    /// Return true if this layout's shape can be broadcast to the given shape.
    fn can_broadcast_to(&self, target_shape: &[usize]) -> bool {
        if self.shape().as_ref() == target_shape {
            return true;
        } else if self.ndim() > target_shape.len() {
            return false;
        }

        // For two shapes to be compatible for broadcasting, each dimension must
        // either be the same or be 1.
        //
        // If the tensor has fewer dimensions, pretend that it was prefixed with
        // 1-length dimensions to make the dimension counts equal.
        let target_dims = target_shape[target_shape.len() - self.shape().len()..]
            .iter()
            .copied();

        zip(self.shape().as_ref().iter().copied(), target_dims).all(|(a, b)| a == b || a == 1)
    }

    /// Return true if the tensor/view can be broadcast with another tensor or
    /// view with a given `shape` as part of a binary operation.
    ///
    /// The shape of the result may be larger than either the current shape
    /// or `shape`. eg. If a tensor of shape `[1, 5]` is broadcast with one
    /// of size `[2, 1, 1]` the result has shape `[2, 1, 5]`.
    ///
    /// See <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md> for
    /// conditions in which broadcasting is allowed.
    fn can_broadcast_with(&self, shape: &[usize]) -> bool {
        if self.shape().as_ref() == shape {
            return true;
        }

        // For two shapes to be compatible for broadcasting, each dimension must
        // either be the same or be 1.
        //
        // If the tensor has fewer dimensions, pretend that it was prefixed with
        // 1-length dimensions to make the dimension counts equal.

        let current_shape = self.shape();
        let a = current_shape.as_ref();
        let b = shape;

        let a_pad = b.len().saturating_sub(a.len());
        let b_pad = a.len().saturating_sub(b.len());

        let a_iter = a.iter().copied().rev().chain(repeat(1).take(a_pad));
        let b_iter = b.iter().copied().rev().chain(repeat(1).take(b_pad));

        zip(a_iter, b_iter).all(|(a, b)| a == b || a == 1 || b == 1)
    }
}

/// Provides methods for querying the shape and data layout of a [Tensor]
/// or [TensorView].
pub trait TensorLayout: Layout {
    /// Returns the internal struct that contains layout information for the tensor.
    #[doc(hidden)]
    fn layout(&self) -> &DynLayout;

    /// Return the offset of an element in the array.
    ///
    /// The length of `index` must match the tensor's dimension count.
    ///
    /// Panics if the index length is incorrect or the value of an index
    /// exceeds the size of the corresponding dimension.
    fn offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        self.layout().offset(index)
    }

    /// Return the offset of the first element in a slice of the array.
    ///
    /// This is the same as `slice`, except that `index` can have fewer
    /// dimensions than the tensor, in which case the index is implicitly
    /// zero-padded on the right.
    fn slice_offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        self.layout().slice_offset(index)
    }

    /// Return an iterator over offsets of elements in this tensor, in their
    /// logical order.
    ///
    /// See also the notes for `slice_offsets`.
    fn offsets(&self) -> Offsets {
        Offsets::new(self.layout())
    }

    /// Return an iterator over offsets of this tensor, broadcasted to `shape`.
    ///
    /// This is very similar to `broadcast_iter`, except that the iterator
    /// yields offsets into rather than elements of the data buffer.
    fn broadcast_offsets(&self, shape: &[usize]) -> Offsets {
        assert!(
            self.can_broadcast_to(shape),
            "Cannot broadcast to specified shape"
        );
        Offsets::broadcast(self.layout(), shape)
    }

    /// Return an iterator over offsets of elements in this tensor.
    ///
    /// Note that the offset order of the returned iterator will become incorrect
    /// if the tensor's layout is modified during iteration.
    fn slice_offsets(&self, ranges: &[SliceRange]) -> Offsets {
        Offsets::slice(self.layout(), ranges)
    }
}

/// Provides convenience methods for querying the shape and strides of a matrix.
pub trait MatrixLayout {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row_stride(&self) -> usize;
    fn col_stride(&self) -> usize;
}

/// Specifies whether a tensor or view may have an overlapping layout.
///
/// An overlapping layout is one in which multiple valid indices map to the same
/// offset in storage. To comply with Rust's rules for mutable aliases, mutable
/// tensors/views must disallow overlap.
pub enum OverlapPolicy {
    AllowOverlap,
    DisallowOverlap,
}

/// Defines the valid indices for an N-dimensional array and how to map them
/// to offsets in a linear buffer, where N is known at compile time.
#[derive(Clone, Copy, Debug)]
pub struct NdLayout<const N: usize> {
    shape: [usize; N],
    strides: [usize; N],
}

impl<const N: usize> Layout for NdLayout<N> {
    type Index<'a> = [usize; N];
    type Indices = NdIndices<N>;

    fn ndim(&self) -> usize {
        N
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    fn shape(&self) -> Self::Index<'_> {
        self.shape
    }

    fn strides(&self) -> Self::Index<'_> {
        self.strides
    }

    fn indices(&self) -> Self::Indices {
        NdIndices::from_shape(self.shape)
    }
}

impl MatrixLayout for NdLayout<2> {
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

impl<const N: usize> NdLayout<N> {
    /// Convert a layout with dynamic rank to a layout with a static rank.
    ///
    /// Panics if `l` does not have N dimensions.
    pub fn from_dyn(l: DynLayout) -> Self {
        assert!(l.ndim() == N, "Dynamic layout dims != {}", N);
        NdLayout {
            shape: l.shape().try_into().unwrap(),
            strides: l.strides().try_into().unwrap(),
        }
    }

    /// Convert this layout to one with a dynamic rank.
    pub fn as_dyn(&self) -> DynLayout {
        DynLayout::with_strides(&self.shape, &self.strides)
    }

    /// Return true if all components of `index` are in-bounds.
    pub fn index_valid(&self, index: [usize; N]) -> bool {
        let mut valid = true;
        for i in 0..N {
            valid = valid && index[i] < self.shape[i]
        }
        valid
    }

    /// Return the offset in the slice that an index maps to.
    pub fn offset(&self, index: [usize; N]) -> usize {
        assert!(
            self.index_valid(index),
            "Index {:?} out of bounds for shape {:?}",
            index,
            self.shape
        );
        self.offset_unchecked(index)
    }

    /// Return the offset in the slice that an index maps to, or `None` if it
    /// is out of bounds.
    pub fn try_offset(&self, index: [usize; N]) -> Option<usize> {
        if !self.index_valid(index) {
            return None;
        }
        Some(self.offset_unchecked(index))
    }

    /// Return the offset in the slice that an index maps to.
    ///
    /// Unlike `offset`, this does not bounds-check elements of `index` against
    /// the corresponding shape. Hence the returned offset may be out of bounds.
    pub fn offset_unchecked(&self, index: [usize; N]) -> usize {
        let mut offset = 0;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        offset
    }

    /// Return the minimum length required for the element data buffer used
    /// with this layout.
    pub fn min_data_len(&self) -> usize {
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
    pub fn contiguous_strides(shape: [usize; N]) -> [usize; N] {
        let mut strides = [0; N];
        for i in 0..N {
            strides[i] = shape[i + 1..].iter().product();
        }
        strides
    }

    /// Create a layout with a given shape and a contiguous layout.
    pub fn from_shape(shape: [usize; N]) -> Self {
        Self {
            shape,
            strides: Self::contiguous_strides(shape),
        }
    }

    /// Create a layout with given shape and strides, intended for use with
    /// data storage of length `data_len`.
    ///
    /// `overlap` determines whether this method will fail if the layout
    /// may have internal overlap.
    pub fn try_from_shape_and_strides(
        shape: [usize; N],
        strides: Option<[usize; N]>,
        data_len: usize,
        overlap: OverlapPolicy,
    ) -> Result<NdLayout<N>, FromDataError> {
        let layout = NdLayout {
            shape,
            strides: strides.unwrap_or(NdLayout::contiguous_strides(shape)),
        };

        if data_len < layout.min_data_len() {
            return Err(FromDataError::StorageTooShort);
        }

        match overlap {
            OverlapPolicy::DisallowOverlap => {
                if may_have_internal_overlap(&layout.shape, &layout.strides) {
                    return Err(FromDataError::MayOverlap);
                }
            }
            OverlapPolicy::AllowOverlap => {}
        }

        Ok(layout)
    }

    /// Swap strides of this layout to put axes in the given order.
    ///
    /// Values in `dims` must be < N.
    pub fn permuted(&self, dims: [usize; N]) -> Self {
        Self::from_dyn(self.as_dyn().permuted(&dims))
    }

    /// Return a layout with the same number of elements but a given shape.
    ///
    /// Panics if the layout is not contiguous.
    pub fn reshaped<const M: usize>(&self, shape: [usize; M]) -> NdLayout<M> {
        NdLayout::from_dyn(self.as_dyn().reshaped(&shape))
    }
}

impl NdLayout<2> {
    pub fn transposed(self) -> NdLayout<2> {
        NdLayout {
            shape: [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

/// Defines the valid indices for an N-dimensional array and how to map them
/// to offsets in a linear buffer, where N can be varied at runtime.
///
/// The layout specifies the size of each dimension of the tensor (the _shape_)
/// and the stride (gap) between offsets in each dimension.
///
/// ## Safety and internal overlap
///
/// Rust requires that only one mutable reference can exist for any value. To
/// ensure this, mutable iteration over a tensor must visit each element only
/// once. This means that in the tensor's Layout, every valid index must map to
/// a unique offset. Verifying this for the general case of arbitrary shape and
/// strides is non-trivial. See notes in `mem_overlap.c` in the NumPy source. In
/// this library the problem is simplified by limiting the stride patterns that
/// can be constructed. All Layout functions must uphold the invariant:
///
///   Every Layout either has one or more strides set to zero, or every valid
///   index must map to a unique offset.
///
/// Zero-strides are used for broadcasting, which is widely used and easy to
/// check for.
#[derive(Clone, Debug)]
pub struct DynLayout {
    /// Array of dimension sizes followed by the corresponding dimension strides.
    ///
    /// Since we always have the same number of stride and shape dims, these
    /// are combined into one array to avoid redundantly storing separate
    /// lengths for each.
    shape_and_strides: SmallVec<[usize; 8]>,
}

impl Layout for DynLayout {
    type Index<'a> = &'a [usize];
    type Indices = DynIndices;

    /// Return the number of elements in the tensor shape described by this layout.
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of dimensions.
    fn ndim(&self) -> usize {
        self.shape_and_strides.len() / 2
    }

    /// Return the sizes of each dimension.
    fn shape(&self) -> &[usize] {
        &self.shape_and_strides[0..self.ndim()]
    }

    /// Returns the size of the dimension `dim`.
    fn size(&self, dim: usize) -> usize {
        self.shape_and_strides[dim]
    }

    /// Return the stride (offset between elements) in the tensor's element array.
    fn strides(&self) -> &[usize] {
        &self.shape_and_strides[self.ndim()..]
    }

    /// Return the stride for a specific dimension.
    fn stride(&self, dim: usize) -> usize {
        self.shape_and_strides[self.ndim() + dim]
    }

    fn indices(&self) -> DynIndices {
        DynIndices::from_shape(self.shape())
    }
}

impl DynLayout {
    /// Construct a layout with dimension sizes given by `shape` and default
    /// (contiguous) strides.
    pub fn new(shape: &[usize]) -> DynLayout {
        DynLayout {
            shape_and_strides: Self::contiguous_shape_and_strides(shape),
        }
    }

    /// Construct a layout with dimension sizes given by `shape` and given
    /// strides.
    ///
    /// Panics if `strides` may lead to internal overlap (multiple indices
    /// map to the same data offset), unless strides contains a `0`. See
    /// struct notes.
    pub fn with_strides(shape: &[usize], strides: &[usize]) -> DynLayout {
        assert!(
            strides.iter().any(|s| *s == 0) || !may_have_internal_overlap(shape, strides),
            "Layout may have internal overlap"
        );
        let mut shape_and_strides = SmallVec::with_capacity(shape.len() + strides.len());
        shape_and_strides.extend_from_slice(shape);
        shape_and_strides.extend_from_slice(strides);
        DynLayout { shape_and_strides }
    }

    /// Move the index at axis `from` to `to`, keeping the relative order of
    /// other dimensions the same. This is like NumPy's `moveaxis` function.
    pub fn move_axis(&mut self, from: usize, to: usize) {
        let ndim = self.ndim();
        assert!(from < ndim && to < ndim);

        let size = self.shape_and_strides.remove(from);
        let stride = self.shape_and_strides.remove(ndim - 1 + from);
        self.shape_and_strides.insert(to, size);
        self.shape_and_strides.insert(ndim + to, stride);
    }

    /// Compute the new layout and offset of the first element for a slice into
    /// an existing tensor view.
    ///
    /// Returns a tuple of (offset_range, layout) for the sliced view.
    pub fn slice(&self, range: &[SliceItem]) -> (Range<usize>, DynLayout) {
        assert!(
            self.ndim() >= range.len(),
            "Slice dims must be <= current dims"
        );
        assert!(
            zip(self.shape().iter(), range.iter())
                .all(|(dim_size, slice_item)| slice_item.valid_for(*dim_size)),
            "Slice range is invalid for tensor shape"
        );

        let padded_range = range
            .iter()
            .chain(repeat(&SliceItem::RangeFull))
            .take(self.ndim())
            .enumerate();

        let offset = padded_range
            .clone()
            .map(|(dim, item)| {
                let start = match item {
                    SliceItem::Index(idx) => *idx,
                    SliceItem::Range(r) => r.start,
                    SliceItem::RangeFrom(r) => r.start,
                    SliceItem::RangeFull => 0,
                };
                self.stride(dim) * start
            })
            .sum();

        let retained_dims = padded_range.clone().filter_map(|(dim, item)| match item {
            SliceItem::Index(_) => None,
            SliceItem::Range(range) => Some((dim, range.clone())),
            SliceItem::RangeFrom(range) => Some((dim, range.start..self.size(dim))),
            SliceItem::RangeFull => Some((dim, 0..self.size(dim))),
        });

        let shape_and_strides = retained_dims
            .clone()
            .map(|(_, item)| item.end - item.start)
            .chain(retained_dims.map(|(dim, _)| self.stride(dim)))
            .collect();

        let layout = Self { shape_and_strides };
        (offset..offset + layout.end_offset(), layout)
    }

    /// Return one past the maximum offset into the tensor/view's data buffer
    /// that will be accessed when indexing into it using this layout.
    pub fn end_offset(&self) -> usize {
        let shape = self.shape();
        if shape.iter().any(|&size| size == 0) {
            return 0;
        }
        let strides = self.strides();
        (0..self.ndim())
            .map(|dim| (shape[dim] - 1) * strides[dim])
            .sum::<usize>()
            + 1
    }

    pub fn resize_dim(&mut self, dim: usize, new_size: usize) {
        self.shape_and_strides[dim] = new_size;
    }

    /// Return true if this is a broadcasting layout which repeats dimensions.
    pub fn is_broadcast(&self) -> bool {
        self.strides().iter().any(|&stride| stride == 0)
    }

    pub fn make_contiguous(&mut self) {
        self.shape_and_strides = Self::contiguous_shape_and_strides(self.shape());
    }

    fn permute_iter<I: Clone + Iterator<Item = usize>>(&mut self, dims: I) {
        let strides = self.strides();
        let shape = self.shape();
        let shape_iter = dims.clone().map(|dim| shape[dim]);
        let stride_iter = dims.map(|dim| strides[dim]);
        self.shape_and_strides = shape_iter.chain(stride_iter).collect();
    }

    /// Swap the order of dimensions in this layout to the order described by
    /// `dims`.
    pub fn permute(&mut self, dims: &[usize]) {
        assert!(
            is_valid_permutation(self.ndim(), dims),
            "permutation is invalid"
        );
        self.permute_iter(dims.iter().copied());
    }

    /// Return a copy of this layout with dimensions re-ordered according to
    /// `dims`.
    pub fn permuted(&self, dims: &[usize]) -> DynLayout {
        let mut permuted = self.clone();
        permuted.permute(dims);
        permuted
    }

    /// Reverse the order of dimensions in this layout.
    pub fn transpose(&mut self) {
        self.permute_iter((0..self.ndim()).rev());
    }

    /// Return a copy of this layout with the order of dimensions reversed.
    pub fn transposed(&self) -> DynLayout {
        let mut transposed = self.clone();
        transposed.transpose();
        transposed
    }

    /// Insert a dimension of size one at index `dim`.
    pub fn insert_dim(&mut self, dim: usize) {
        let mut new_shape: SmallVec<[usize; 4]> = self.shape().into();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape);
    }

    /// Change the shape of this layout to `shape`.
    ///
    /// `shape` must have the same product as the current shape (ie. must
    /// specify the same number of elements) and the layout must be contiguous.
    pub fn reshape(&mut self, shape: &[usize]) {
        assert!(
            shape.iter().product::<usize>() == self.len(),
            "new shape must have same number of elements as current shape"
        );
        assert!(
            self.is_contiguous(),
            "can only reshape a contiguous tensor/view"
        );
        *self = DynLayout::new(shape);
    }

    pub fn reshaped(&self, shape: &[usize]) -> DynLayout {
        let mut reshaped = self.clone();
        reshaped.reshape(shape);
        reshaped
    }

    /// Return the offset of the element with a given index.
    pub fn offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        assert!(
            self.ndim() == index.len(),
            "Cannot index {} dim tensor with {} dim index",
            self.ndim(),
            index.len()
        );
        self.slice_offset(index)
    }

    /// Return the offset of the slice that begins at the given index.
    pub fn slice_offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        assert!(index.len() <= self.ndim());
        let shape = self.shape();
        let mut offset = 0;
        for i in 0..index.len() {
            assert!(
                index.index(i) < shape[i],
                "Invalid index {} for dim {}",
                index.index(i),
                i
            );
            offset += index.index(i) * self.stride(i)
        }
        offset
    }

    /// Return a copy of this layout with dimensions of size 1 removed.
    pub fn squeezed(&self) -> DynLayout {
        let shape = self.shape().iter().copied().filter(|&size| size != 1);
        let strides = zip(self.shape().iter().copied(), self.strides().iter().copied())
            .filter_map(|(size, stride)| if size != 1 { Some(stride) } else { None });
        DynLayout {
            shape_and_strides: shape.chain(strides).collect(),
        }
    }

    pub fn dims<const N: usize>(&self) -> [usize; N] {
        assert!(
            self.ndim() == N,
            "Cannot extract {} dim tensor as {} dim array",
            self.ndim(),
            N
        );
        self.shape().try_into().unwrap()
    }

    /// Create a shape-and-strides array for a contiguous layout.
    fn contiguous_shape_and_strides(shape: &[usize]) -> SmallVec<[usize; 8]> {
        let mut strides_and_shape = SmallVec::with_capacity(shape.len() * 2);
        strides_and_shape.extend_from_slice(shape);
        for i in 0..shape.len() {
            let stride = shape[i + 1..].iter().product();
            strides_and_shape.push(stride);
        }
        strides_and_shape
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::layout::DynLayout;
    use crate::{Layout, SliceItem};

    #[test]
    fn test_with_strides() {
        struct Case {
            shape: &'static [usize],
            strides: &'static [usize],
        }

        let cases = [
            // Contiguous layout
            Case {
                shape: &[10, 10],
                strides: &[10, 1],
            },
            // Broadcasting layout
            Case {
                shape: &[10, 10],
                strides: &[10, 0],
            },
        ];

        for case in cases {
            let layout = DynLayout::with_strides(case.shape, case.strides);
            assert_eq!(layout.shape(), case.shape);
            assert_eq!(layout.strides(), case.strides);
        }
    }

    #[test]
    #[should_panic(expected = "Layout may have internal overlap")]
    fn test_with_strides_overlap() {
        DynLayout::with_strides(&[10, 10], &[1, 2]);
    }

    #[test]
    fn test_move_axis() {
        let mut layout = DynLayout::new(&[2, 4, 8]);
        assert_eq!(layout.strides(), [32, 8, 1]);

        layout.move_axis(1, 0);
        assert_eq!(layout.shape(), [4, 2, 8]);
        assert_eq!(layout.strides(), [8, 32, 1]);

        layout.move_axis(0, 1);
        assert_eq!(layout.shape(), [2, 4, 8]);
        assert_eq!(layout.strides(), [32, 8, 1]);

        layout.move_axis(2, 1);
        assert_eq!(layout.shape(), [2, 8, 4]);
        assert_eq!(layout.strides(), [32, 1, 8]);
    }

    #[test]
    #[should_panic]
    fn test_move_axis_invalid_from() {
        let mut layout = DynLayout::new(&[2, 4, 8]);
        layout.move_axis(3, 0);
    }

    #[test]
    #[should_panic]
    fn test_move_axis_invalid_to() {
        let mut layout = DynLayout::new(&[2, 4, 8]);
        layout.move_axis(0, 3);
    }

    #[test]
    #[should_panic(expected = "permutation is invalid")]
    fn test_permute_invalid_len() {
        let mut layout = DynLayout::new(&[5, 5]);
        layout.permute(&[1, 0, 3]);
    }

    #[test]
    #[should_panic(expected = "permutation is invalid")]
    fn test_permute_too_few_dims() {
        let mut layout = DynLayout::new(&[5, 5]);
        layout.permute(&[1]);
    }

    #[test]
    #[should_panic(expected = "permutation is invalid")]
    fn test_permute_repeated_dims() {
        let mut layout = DynLayout::new(&[5, 5]);
        layout.permute(&[1, 1]);
    }

    #[test]
    fn test_squeezed() {
        let layout = DynLayout::new(&[1, 1, 10, 20]);
        let squeezed = layout.squeezed();
        assert_eq!(squeezed.shape(), &[10, 20]);
        assert_eq!(squeezed.strides(), &[20, 1]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_index() {
        let layout = DynLayout::new(&[3, 5]);
        layout.slice(&[SliceItem::Index(4), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_range() {
        let layout = DynLayout::new(&[3, 5]);
        layout.slice(&[SliceItem::Range(1..4), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_from_range() {
        let layout = DynLayout::new(&[3, 5]);
        layout.slice(&[SliceItem::RangeFrom(4..), SliceItem::Index(0)]);
    }

    #[test]
    fn test_size_stride() {
        let layout = DynLayout::new(&[10, 20, 30]);
        for (dim, (&size, &stride)) in
            zip(layout.shape().iter(), layout.strides().iter()).enumerate()
        {
            assert_eq!(layout.size(dim), size);
            assert_eq!(layout.stride(dim), stride);
        }
    }
}
