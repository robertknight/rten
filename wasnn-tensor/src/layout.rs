use std::iter::{repeat, zip};
use std::ops::Range;

use smallvec::SmallVec;

use super::overlap::may_have_internal_overlap;
use super::range::SliceItem;
use super::TensorIndex;

/// Return true if `permutation` is a valid permutation of dimensions for
/// a tensor of rank `ndim`.
pub fn is_valid_permutation(ndim: usize, permutation: &[usize]) -> bool {
    permutation.len() == ndim
        && (0..ndim).all(|dim| permutation.iter().filter(|d| **d == dim).count() == 1)
}

/// Describes how to map coordinates in an N-dimensional array / tensor to
/// offsets in the underlying array of elements.
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
pub struct Layout {
    /// Array of dimension sizes followed by the corresponding dimension strides.
    ///
    /// Since we always have the same number of stride and shape dims, these
    /// are combined into one array to avoid redundantly storing separate
    /// lengths for each.
    shape_and_strides: SmallVec<[usize; 8]>,
}

impl Layout {
    /// Construct a layout with dimension sizes given by `shape` and default
    /// (contiguous) strides.
    pub fn new(shape: &[usize]) -> Layout {
        Layout {
            shape_and_strides: Self::contiguous_shape_and_strides(shape),
        }
    }

    /// Construct a layout with dimension sizes given by `shape` and given
    /// strides.
    ///
    /// Panics if `strides` may lead to internal overlap (multiple indices
    /// map to the same data offset), unless strides contains a `0`. See
    /// struct notes.
    pub fn new_with_strides(shape: &[usize], strides: &[usize]) -> Layout {
        assert!(
            strides.iter().any(|s| *s == 0) || !may_have_internal_overlap(shape, strides),
            "Layout may have internal overlap"
        );
        let mut shape_and_strides = SmallVec::with_capacity(shape.len() + strides.len());
        shape_and_strides.extend_from_slice(shape);
        shape_and_strides.extend_from_slice(strides);
        Layout { shape_and_strides }
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
    pub fn slice(&self, range: &[SliceItem]) -> (Range<usize>, Layout) {
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
            SliceItem::RangeFrom(range) => Some((dim, range.start..self.shape()[dim])),
            SliceItem::RangeFull => Some((dim, 0..self.shape()[dim])),
        });

        let shape_and_strides = retained_dims
            .clone()
            .map(|(_, item)| item.end - item.start)
            .chain(retained_dims.map(|(dim, _)| self.stride(dim)))
            .collect();

        let layout = Self { shape_and_strides };
        (offset..offset + layout.end_offset(), layout)
    }

    /// Return the number of elements in the tensor shape described by this layout.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape_and_strides.len() / 2
    }

    /// Return the sizes of each dimension.
    pub fn shape(&self) -> &[usize] {
        &self.shape_and_strides[0..self.ndim()]
    }

    /// Returns the size of the dimension `dim`.
    pub fn size(&self, dim: usize) -> usize {
        self.shape_and_strides[dim]
    }

    /// Return the stride (offset between elements) in the tensor's element array.
    pub fn strides(&self) -> &[usize] {
        &self.shape_and_strides[self.ndim()..]
    }

    /// Return the stride for a specific dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.shape_and_strides[self.ndim() + dim]
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

    /// Return true if this layout describes a contiguous tensor, where the
    /// logical order of elements matches the order in which they are stored.
    pub fn is_contiguous(&self) -> bool {
        let mut product = 1;
        for (dim, len) in self.shape().iter().enumerate().rev() {
            if self.stride(dim) != product {
                return false;
            }
            product *= len;
        }
        true
    }

    pub fn make_contiguous(&mut self) {
        self.shape_and_strides = Self::contiguous_shape_and_strides(self.shape());
    }

    /// Return true if this layout's shape can be broadcast to the given shape.
    pub fn can_broadcast_to(&self, shape: &[usize]) -> bool {
        if self.shape() == shape {
            return true;
        } else if self.ndim() > shape.len() {
            return false;
        }

        // For two shapes to be compatible for broadcasting, each dimension must
        // either be the same or be 1.
        //
        // If the tensor has fewer dimensions, pretend that it was prefixed with
        // 1-length dimensions to make the dimension counts equal.
        let self_dims = self.shape().iter().copied();
        let target_dims = shape[shape.len() - self.shape().len()..].iter().copied();

        zip(self_dims, target_dims).all(|(a, b)| a == b || a == 1)
    }

    /// Return true if this layout's shape can be broadcast with another layout
    /// that has shape `shape`.
    pub fn can_broadcast_with(&self, shape: &[usize]) -> bool {
        if self.shape() == shape {
            return true;
        }

        // For two shapes to be compatible for broadcasting, each dimension must
        // either be the same or be 1.
        //
        // If the tensor has fewer dimensions, pretend that it was prefixed with
        // 1-length dimensions to make the dimension counts equal.

        let a = self.shape();
        let b = shape;

        let a_pad = b.len().saturating_sub(a.len());
        let b_pad = a.len().saturating_sub(b.len());

        let a_iter = a.iter().copied().rev().chain(repeat(1).take(a_pad));
        let b_iter = b.iter().copied().rev().chain(repeat(1).take(b_pad));

        zip(a_iter, b_iter).all(|(a, b)| a == b || a == 1 || b == 1)
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
            "Permutation is invalid"
        );
        self.permute_iter(dims.iter().copied());
    }

    /// Return a copy of this layout with dimensions re-ordered according to
    /// `dims`.
    pub fn permuted(&self, dims: &[usize]) -> Layout {
        let mut permuted = self.clone();
        permuted.permute(dims);
        permuted
    }

    /// Reverse the order of dimensions in this layout.
    pub fn transpose(&mut self) {
        self.permute_iter((0..self.ndim()).rev());
    }

    /// Return a copy of this layout with the order of dimensions reversed.
    pub fn transposed(&self) -> Layout {
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
            "New shape must have same number of elements as current shape"
        );
        assert!(
            self.is_contiguous(),
            "Can only reshape a contiguous tensor/view"
        );
        *self = Layout::new(shape);
    }

    pub fn reshaped(&self, shape: &[usize]) -> Layout {
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
    pub fn squeezed(&self) -> Layout {
        let shape = self.shape().iter().copied().filter(|&size| size != 1);
        let strides = zip(self.shape().iter().copied(), self.strides().iter().copied())
            .filter_map(|(size, stride)| if size != 1 { Some(stride) } else { None });
        Layout {
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

    use crate::layout::Layout;
    use crate::SliceItem;

    #[test]
    fn test_new_with_strides() {
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
            let layout = Layout::new_with_strides(case.shape, case.strides);
            assert_eq!(layout.shape(), case.shape);
            assert_eq!(layout.strides(), case.strides);
        }
    }

    #[test]
    #[should_panic(expected = "Layout may have internal overlap")]
    fn test_new_with_strides_overlap() {
        Layout::new_with_strides(&[10, 10], &[1, 2]);
    }

    #[test]
    fn test_move_axis() {
        let mut layout = Layout::new(&[2, 4, 8]);
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
        let mut layout = Layout::new(&[2, 4, 8]);
        layout.move_axis(3, 0);
    }

    #[test]
    #[should_panic]
    fn test_move_axis_invalid_to() {
        let mut layout = Layout::new(&[2, 4, 8]);
        layout.move_axis(0, 3);
    }

    #[test]
    #[should_panic(expected = "Permutation is invalid")]
    fn test_permute_invalid_len() {
        let mut layout = Layout::new(&[5, 5]);
        layout.permute(&[1, 0, 3]);
    }

    #[test]
    #[should_panic(expected = "Permutation is invalid")]
    fn test_permute_too_few_dims() {
        let mut layout = Layout::new(&[5, 5]);
        layout.permute(&[1]);
    }

    #[test]
    #[should_panic(expected = "Permutation is invalid")]
    fn test_permute_repeated_dims() {
        let mut layout = Layout::new(&[5, 5]);
        layout.permute(&[1, 1]);
    }

    #[test]
    fn test_squeezed() {
        let layout = Layout::new(&[1, 1, 10, 20]);
        let squeezed = layout.squeezed();
        assert_eq!(squeezed.shape(), &[10, 20]);
        assert_eq!(squeezed.strides(), &[20, 1]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_index() {
        let layout = Layout::new(&[3, 5]);
        layout.slice(&[SliceItem::Index(4), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_range() {
        let layout = Layout::new(&[3, 5]);
        layout.slice(&[SliceItem::Range(1..4), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_from_range() {
        let layout = Layout::new(&[3, 5]);
        layout.slice(&[SliceItem::RangeFrom(4..), SliceItem::Index(0)]);
    }

    #[test]
    fn test_size_stride() {
        let layout = Layout::new(&[10, 20, 30]);
        for (dim, (&size, &stride)) in
            zip(layout.shape().iter(), layout.strides().iter()).enumerate()
        {
            assert_eq!(layout.size(dim), size);
            assert_eq!(layout.stride(dim), stride);
        }
    }
}
