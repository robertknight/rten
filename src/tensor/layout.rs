use std::iter::{repeat, zip};

use smallvec::SmallVec;

use super::range::SliceItem;
use super::TensorIndex;

/// Describes how to map offsets in a buffer of elements to coordinates in an
/// N-dimensional array / tensor.
///
/// Logically this data consists of the size of each dimension of the tensor,
/// and the stride (gap) between offsets in that dimension.
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

    /// Compute the new layout and offset of the first element for a slice into
    /// an existing tensor view.
    ///
    /// Returns a tuple of (offset, layout) for the sliced view.
    pub fn slice(&self, range: &[SliceItem]) -> (usize, Layout) {
        assert!(
            self.ndim() >= range.len(),
            "Slice dims must be <= current dims"
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
                    SliceItem::RangeFull => 0,
                };
                self.stride(dim) * start
            })
            .sum();

        let retained_dims = padded_range.clone().filter_map(|(dim, item)| match item {
            SliceItem::Index(_) => None,
            SliceItem::Range(range) => Some((dim, range.clone())),
            SliceItem::RangeFull => Some((dim, 0..self.shape()[dim])),
        });

        let shape_and_strides = retained_dims
            .clone()
            .map(|(_, item)| item.end - item.start)
            .chain(retained_dims.map(|(dim, _)| self.stride(dim)))
            .collect();

        (offset, Self { shape_and_strides })
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

    /// Return the stride (offset between elements) in the tensor's data buffer.
    pub fn strides(&self) -> &[usize] {
        &self.shape_and_strides[self.ndim()..]
    }

    /// Return the stride for a specific dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.shape_and_strides[self.ndim() + dim]
    }

    /// Return one past the maximum offset into the tensor/view's data buffer
    /// that will be accessed when indexing into it using the mapping defined
    /// by this layout.
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

    /// Return true if this layout describes viewing a tensor with N elements
    /// as a larger tensor with some multiple of N elements.
    ///
    /// To enforce Rust's invariant that only one mutable reference to a value
    /// can exist at once, broadcasted views / iterators must be read-only.
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

    /// Swap the order of dimensions in this layout to the order described by
    /// `dims`.
    pub fn permute(&mut self, dims: &[usize]) {
        if dims.len() != self.ndim() {
            panic!("Permute dims length does not match dimension count");
        }
        let strides = self.strides();
        let shape = self.shape();
        self.shape_and_strides = dims
            .iter()
            .map(|&dim| shape[dim])
            .chain(dims.iter().map(|&dim| strides[dim]))
            .collect();
    }

    /// Return a copy of this layout with dimensions re-ordered according to
    /// `dims`.
    pub fn permuted(&self, dims: &[usize]) -> Layout {
        let mut permuted = self.clone();
        permuted.permute(dims);
        permuted
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

    pub fn offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        let shape = self.shape();
        assert!(
            shape.len() == index.len(),
            "Cannot access {} dim tensor with {} dim index",
            shape.len(),
            index.len()
        );
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

    pub fn dims<const N: usize>(&self) -> [usize; N] {
        if self.ndim() != N {
            panic!(
                "Cannot extract {} dim tensor as {} dim array",
                self.ndim(),
                N
            );
        }
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
