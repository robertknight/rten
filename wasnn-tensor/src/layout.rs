use std::iter::{repeat, zip};
use std::ops::Range;

use smallvec::{smallvec, SmallVec};

use crate::errors::{DimensionError, FromDataError, SliceError};
use crate::index_iterator::{DynIndices, NdIndices};
use crate::overlap::{is_contiguous, may_have_internal_overlap};
use crate::range::SliceItem;
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
    type Index<'a>: AsRef<[usize]> + std::fmt::Debug + PartialEq<Self::Index<'a>>
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
        is_contiguous(self.shape(), self.strides())
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

    #[inline]
    fn shape(&self) -> Self::Index<'_> {
        self.shape
    }

    #[inline]
    fn strides(&self) -> Self::Index<'_> {
        self.strides
    }

    fn indices(&self) -> Self::Indices {
        NdIndices::from_shape(self.shape)
    }
}

impl MatrixLayout for NdLayout<2> {
    #[inline]
    fn rows(&self) -> usize {
        self.size(0)
    }

    #[inline]
    fn cols(&self) -> usize {
        self.size(1)
    }

    #[inline]
    fn row_stride(&self) -> usize {
        self.stride(0)
    }

    #[inline]
    fn col_stride(&self) -> usize {
        self.stride(1)
    }
}

/// Compute the shape and strides of a layout after slicing with `range`.
///
/// Returns an `(ndim, offset)` tuple for the number of dimensions in the
/// slice and the offset of the first element in the parent view's data.
///
/// This function is generic to allow for specialized variants to be generated
/// when slicing with statically known input or output shape sizes.
fn slice_layout<I: AsRef<[usize]>, O: AsMut<[usize]>>(
    in_shape: I,
    in_strides: I,
    mut out_shape: O,
    mut out_strides: O,
    range: &[SliceItem],
) -> Result<(usize, usize), SliceError> {
    let in_shape = in_shape.as_ref();
    let in_strides = in_strides.as_ref();
    let out_shape = out_shape.as_mut();
    let out_strides = out_strides.as_mut();

    let mut ndim = 0;
    let mut offset = 0;

    for (in_dim, (&size, &stride)) in zip(in_shape.iter(), in_strides.iter()).enumerate() {
        let (offset_adjust, new_size_stride) = match range.get(in_dim) {
            Some(&SliceItem::Index(idx)) => {
                let size = size as isize;
                let pos_idx = if idx >= 0 { idx } else { idx + size };
                if pos_idx < 0 || pos_idx >= size {
                    return Err(SliceError::InvalidIndex);
                }
                (stride * pos_idx as usize, None)
            }
            Some(SliceItem::Range(range)) => {
                let resolved = range.resolve(size).ok_or(SliceError::InvalidRange)?;
                let step: usize = range
                    .step()
                    .try_into()
                    .map_err(|_| SliceError::InvalidStep)?;
                let new_size = if step == 1 {
                    // Fast path when no custom step is used.
                    resolved.end - resolved.start
                } else {
                    range.steps(size)
                };
                let new_stride = stride * step;
                (stride * resolved.start, Some((new_size, new_stride)))
            }
            None => (0, Some((size, stride))),
        };

        offset += offset_adjust;
        if let Some((new_size, new_stride)) = new_size_stride {
            out_shape[ndim] = new_size;
            out_strides[ndim] = new_stride;
            ndim += 1;
        }
    }

    Ok((ndim, offset))
}

/// Return an iterator over the strides of a layout that broadcasts a view
/// with shape `from_shape` and strides `from_strides` to `to_shape`.
fn broadcast_strides<'a>(
    from_shape: &'a [usize],
    from_strides: &'a [usize],
    to_shape: &'a [usize],
) -> impl Iterator<Item = usize> + 'a {
    let pad = to_shape.len() - from_shape.len();
    repeat(0)
        .take(pad)
        .chain(from_shape.iter().zip(from_strides.iter()).enumerate().map(
            move |(i, (size, stride))| {
                if *size == 1 && to_shape[i + pad] > 1 {
                    0
                } else {
                    *stride
                }
            },
        ))
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

    /// Construct a layout which broadcasts elements to `to_shape` by setting
    /// the stride to `0` in broadcasted dimensions.
    pub fn broadcast<const M: usize>(&self, to_shape: [usize; M]) -> NdLayout<M> {
        assert!(
            self.can_broadcast_to(&to_shape),
            "Cannot broadcast to specified shape"
        );
        let mut strides = [0usize; M];
        for (i, stride) in broadcast_strides(&self.shape(), &self.strides(), &to_shape).enumerate()
        {
            strides[i] = stride;
        }

        NdLayout {
            shape: to_shape,
            strides,
        }
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

    /// Compute the new layout and offset of the first element for a slice into
    /// an existing tensor view.
    ///
    /// Returns a tuple of (offset_range, layout) for the sliced view.
    pub fn slice<const M: usize>(&self, range: &[SliceItem]) -> (Range<usize>, NdLayout<M>) {
        assert!(
            self.ndim() >= range.len(),
            "Slice dims must be <= current dims"
        );

        let mut shape: [usize; M] = [0; M];
        let mut strides: [usize; M] = [0; M];

        let (ndim, offset) =
            slice_layout(&self.shape, &self.strides, &mut shape, &mut strides, range).unwrap();

        assert!(ndim == M, "sliced dims != {}", M);

        let layout = NdLayout { shape, strides };
        (offset..offset + layout.min_data_len(), layout)
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

impl<'a, const N: usize> TryFrom<&'a DynLayout> for NdLayout<N> {
    type Error = DimensionError;

    /// Convert a dynamic layout into a static layout with N dims. Fails if
    /// `value.ndim() != N`.
    fn try_from(value: &'a DynLayout) -> Result<NdLayout<N>, DimensionError> {
        let shape: [usize; N] = value.shape().try_into().map_err(|_| DimensionError {})?;
        let strides: [usize; N] = value.strides().try_into().map_err(|_| DimensionError {})?;
        Ok(NdLayout { shape, strides })
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
    #[inline]
    fn ndim(&self) -> usize {
        self.shape_and_strides.len() / 2
    }

    /// Return the sizes of each dimension.
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape_and_strides[0..self.ndim()]
    }

    /// Returns the size of the dimension `dim`.
    #[inline]
    fn size(&self, dim: usize) -> usize {
        self.shape_and_strides[dim]
    }

    /// Return the stride (offset between elements) in the tensor's element array.
    #[inline]
    fn strides(&self) -> &[usize] {
        &self.shape_and_strides[self.ndim()..]
    }

    /// Return the stride for a specific dimension.
    #[inline]
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

    /// Construct a layout which broadcasts elements to `to_shape` by setting
    /// the stride to `0` in broadcasted dimensions.
    pub fn broadcast(&self, to_shape: &[usize]) -> DynLayout {
        assert!(
            self.can_broadcast_to(to_shape),
            "Cannot broadcast to specified shape"
        );

        let mut shape_and_strides = SmallVec::with_capacity(to_shape.len() * 2);
        shape_and_strides.extend(to_shape.iter().copied());
        shape_and_strides.extend(broadcast_strides(self.shape(), self.strides(), to_shape));

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
    ///
    /// Panics if the range is invalid for the current layout.
    pub fn slice(&self, range: &[SliceItem]) -> (Range<usize>, DynLayout) {
        match self.try_slice(range) {
            Ok(result) => result,

            // These error conversions preserve existing error messages in
            // various tests.
            Err(SliceError::InvalidRange) => panic!("Slice range is invalid for tensor shape"),
            Err(SliceError::InvalidIndex) => panic!("Slice index is invalid for tensor shape"),
            Err(SliceError::InvalidStep) => panic!("Cannot slice with negative step"),
            Err(err) => panic!("{:?}", err),
        }
    }

    /// Compute the new layout and offset of the first element for a slice into
    /// an existing tensor view.
    ///
    /// Returns a tuple of (offset_range, layout) for the sliced view, or an
    /// error if the range is invalid.
    pub fn try_slice(&self, range: &[SliceItem]) -> Result<(Range<usize>, DynLayout), SliceError> {
        if self.ndim() < range.len() {
            return Err(SliceError::TooManyDims);
        }

        let out_dims = self.ndim()
            - range
                .iter()
                .filter(|item| matches!(item, SliceItem::Index(_)))
                .count();
        let mut shape_and_strides = smallvec![0; out_dims * 2];
        let (out_shape, out_strides) = shape_and_strides.as_mut_slice().split_at_mut(out_dims);

        let (_ndim, offset) =
            slice_layout(self.shape(), self.strides(), out_shape, out_strides, range)?;

        let layout = Self { shape_and_strides };
        Ok((offset..offset + layout.end_offset(), layout))
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

    /// Return true if iterating over elements in this layout will visit
    /// elements multiple times.
    pub fn is_broadcast(&self) -> bool {
        !self.is_empty() && self.strides().iter().any(|&stride| stride == 0)
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
        let ndim = self.ndim();
        let new_size = 1;

        // Choose stride for new dimension as if we were inserting it at the
        // beginning. If `dim != 0` then the result is as if we inserted the
        // dim at the start and then permuted the layout.
        let (max_stride, size_for_max_stride) = self
            .strides()
            .iter()
            .copied()
            .zip(self.shape().iter().copied())
            .max_by_key(|(stride, _size)| *stride)
            .unwrap_or((1, 1));
        let new_stride = max_stride * size_for_max_stride;

        self.shape_and_strides.insert(dim, new_size);
        self.shape_and_strides.insert(ndim + 1 + dim, new_stride);
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

    /// Return the offset in the slice that an index maps to, or `None` if it
    /// is out of bounds.
    #[inline]
    pub fn try_offset<Idx: TensorIndex>(&self, index: Idx) -> Option<usize> {
        let shape = self.shape();
        let strides = self.strides();
        let mut valid = index.len() == shape.len();
        let mut offset = 0;
        for (idx, (size, stride)) in index.iter().zip(shape.iter().zip(strides.iter())) {
            valid = valid && idx < size;
            offset += idx * stride;
        }
        valid.then_some(offset)
    }

    /// Return the offset of the element with a given index.
    pub fn offset<Idx: TensorIndex>(&self, index: Idx) -> usize {
        self.try_offset(index).expect("invalid index")
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

impl<const N: usize> From<&NdLayout<N>> for DynLayout {
    fn from(value: &NdLayout<N>) -> DynLayout {
        DynLayout::with_strides(&value.shape(), &value.strides())
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::layout::DynLayout;
    use crate::{Layout, SliceItem};

    #[test]
    fn test_is_broadcast() {
        // Non-empty, contiguous layout
        let layout = DynLayout::new(&[5, 5]);
        assert!(!layout.is_broadcast());

        // Empty layout
        let layout = DynLayout::new(&[5, 0]);
        assert!(!layout.is_broadcast());

        // Broadcasting layout
        let layout = DynLayout::with_strides(&[5, 5], &[0, 0]);
        assert!(layout.is_broadcast());
    }

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
    #[should_panic(expected = "Slice index is invalid for tensor shape")]
    fn test_slice_invalid_index() {
        let layout = DynLayout::new(&[3, 5]);
        layout.slice(&[SliceItem::Index(4), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Slice index is invalid for tensor shape")]
    fn test_slice_invalid_negative_index() {
        let layout = DynLayout::new(&[3, 5]);
        layout.slice(&[SliceItem::Index(-4)]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_range() {
        let layout = DynLayout::new(&[3, 5]);
        layout.slice(&[SliceItem::Range((1..4).into()), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_from_range() {
        let layout = DynLayout::new(&[3, 5]);
        layout.slice(&[SliceItem::Range((4..).into()), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Cannot slice with negative step")]
    fn test_slice_negative_step() {
        let layout = DynLayout::new(&[3, 5]);
        layout.slice(&[SliceItem::full_range(), SliceItem::range(0, None, -1)]);
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
