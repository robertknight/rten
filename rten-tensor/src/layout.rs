use std::iter::{repeat, zip};
use std::ops::Range;

use smallvec::{smallvec, SmallVec};

use crate::errors::{DimensionError, FromDataError, ReshapeError, SliceError};
use crate::index_iterator::{DynIndices, NdIndices};
use crate::overlap::{is_contiguous, may_have_internal_overlap};
use crate::slice_range::{IntoSliceItems, SliceItem};

/// Return true if `permutation` is a valid permutation of dimensions for
/// a tensor of rank `ndim`.
pub fn is_valid_permutation(ndim: usize, permutation: &[usize]) -> bool {
    permutation.len() == ndim
        && (0..ndim).all(|dim| permutation.iter().filter(|d| **d == dim).count() == 1)
}

/// Layouts describe the shape of a tensor, ie. the number of dimensions and
/// size of each, and the mapping between indices and offsets in the data
/// storage.
///
/// The main implementations are [NdLayout], where the dimension count is known
/// statically, and [DynLayout], where the dimension count is only known at
/// runtime.
pub trait Layout {
    /// Type used to represent indices.
    ///
    /// It is assumed that this type can also represent the shape and strides
    /// of the tensor.
    type Index<'a>: AsRef<[usize]> + Clone + std::fmt::Debug + PartialEq<Self::Index<'a>>;

    /// Iterator over indices in this tensor.
    type Indices;

    /// Map an index to a storage offset.
    ///
    /// Panics if any dimension of the index is out of bounds.
    #[inline]
    fn offset(&self, index: Self::Index<'_>) -> usize {
        self.try_offset(index.clone()).unwrap_or_else(|| {
            panic!(
                "index {:?} out of bounds for shape {:?}",
                index.as_ref(),
                self.shape().as_ref()
            );
        })
    }

    /// Map an index to a storage offset, without checking if it is valid for
    /// the tensor's shape.
    ///
    /// This method is not itself unsafe, because it only computes a storage
    /// offset but does not access any data. Using the offset to index into
    /// storage with a bounds check is unsafe however.
    fn offset_unchecked(&self, index: Self::Index<'_>) -> usize {
        index
            .as_ref()
            .iter()
            .zip(self.strides().as_ref())
            .map(|(idx, stride)| *idx * *stride)
            .sum()
    }

    /// Map an index to a storage offset, or return `None` if the index is out
    /// of bounds along any dimension.
    fn try_offset(&self, index: Self::Index<'_>) -> Option<usize>;

    /// Return the number of dimensions.
    fn ndim(&self) -> usize;

    /// Returns the number of elements in the array.
    fn len(&self) -> usize;

    /// Return true if this layout describes a contiguous tensor, where the
    /// logical order of elements matches the order in which they are stored.
    fn is_contiguous(&self) -> bool {
        is_contiguous(self.shape(), self.strides())
    }

    /// Return true if iterating over elements in this layout will visit
    /// elements multiple times.
    fn is_broadcast(&self) -> bool {
        !self.is_empty() && self.strides().as_ref().iter().any(|&stride| stride == 0)
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
        let target_dims = target_shape[target_shape.len() - self.shape().as_ref().len()..]
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

    /// Return the minimum length required for the element data buffer used
    /// with this layout.
    fn min_data_len(&self) -> usize {
        if self.shape().as_ref().iter().any(|&size| size == 0) {
            return 0;
        }
        let max_offset: usize = zip(self.shape().as_ref().iter(), self.strides().as_ref().iter())
            .map(|(size, stride)| (size - 1) * stride)
            .sum();
        max_offset + 1
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
    fn try_offset(&self, index: [usize; N]) -> Option<usize> {
        if !self.index_valid(index) {
            return None;
        }
        Some(self.offset_unchecked(index))
    }

    #[inline]
    fn offset_unchecked(&self, index: [usize; N]) -> usize {
        let mut offset = 0;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        offset
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
        self.into()
    }

    /// Return true if all components of `index` are in-bounds.
    pub fn index_valid(&self, index: [usize; N]) -> bool {
        let mut valid = true;
        for i in 0..N {
            valid = valid && index[i] < self.shape[i]
        }
        valid
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
        strides: [usize; N],
        overlap: OverlapPolicy,
    ) -> Result<NdLayout<N>, FromDataError> {
        let layout = NdLayout { shape, strides };

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
        assert!(is_valid_permutation(N, &dims), "permutation is invalid");
        let mut shape = [0; N];
        let mut strides = [0; N];
        for i in 0..N {
            shape[i] = self.shape[dims[i]];
            strides[i] = self.strides[dims[i]];
        }
        NdLayout { shape, strides }
    }

    /// Reverse the order of dimensions in this layout.
    pub fn transposed(&self) -> Self {
        let dims = std::array::from_fn(|i| N - i - 1);
        self.permuted(dims)
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

    pub fn resize_dim(&mut self, dim: usize, new_size: usize) {
        self.shape[dim] = new_size;
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

    #[inline]
    fn try_offset(&self, index: Self::Index<'_>) -> Option<usize> {
        let shape = self.shape();
        let strides = self.strides();
        let mut valid = index.as_ref().len() == shape.len();
        let mut offset = 0;
        for (idx, (size, stride)) in index.as_ref().iter().zip(shape.iter().zip(strides.iter())) {
            valid = valid && idx < size;
            offset += idx * stride;
        }
        valid.then_some(offset)
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
    pub fn from_shape(shape: &[usize]) -> DynLayout {
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
    pub fn try_from_shape_and_strides(
        shape: &[usize],
        strides: &[usize],
        overlap: OverlapPolicy,
    ) -> Result<DynLayout, FromDataError> {
        let mut shape_and_strides = SmallVec::with_capacity(shape.len() + strides.len());
        shape_and_strides.extend_from_slice(shape);
        shape_and_strides.extend_from_slice(strides);
        let layout = DynLayout { shape_and_strides };

        match overlap {
            OverlapPolicy::DisallowOverlap => {
                if may_have_internal_overlap(layout.shape(), layout.strides()) {
                    return Err(FromDataError::MayOverlap);
                }
            }
            OverlapPolicy::AllowOverlap => {}
        }

        Ok(layout)
    }

    /// Create a new `DynLayout` with the same shape and strides as `layout`.
    pub fn from_layout<L: Layout>(layout: &L) -> DynLayout {
        DynLayout::try_from_shape_and_strides(
            layout.shape().as_ref(),
            layout.strides().as_ref(),
            OverlapPolicy::AllowOverlap,
        )
        .expect("invalid layout")
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
        Ok((offset..offset + layout.min_data_len(), layout))
    }

    pub fn resize_dim(&mut self, dim: usize, new_size: usize) {
        self.shape_and_strides[dim] = new_size;
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

    /// Return the offset of the slice that begins at the given index.
    pub fn slice_offset<Idx: AsRef<[usize]>>(&self, index: Idx) -> usize {
        let index = index.as_ref();

        assert!(index.len() <= self.ndim());
        let shape = self.shape();
        let mut offset = 0;
        for i in 0..index.len() {
            assert!(
                index[i] < shape[i],
                "Invalid index {} for dim {}",
                index[i],
                i
            );
            offset += index[i] * self.stride(i)
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
        let mut strides_and_shape: SmallVec<[usize; 8]> = SmallVec::from_slice(shape);
        strides_and_shape.resize(shape.len() * 2, 0);
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides_and_shape[shape.len() + i] = stride;
            stride *= shape[i];
        }
        strides_and_shape
    }
}

impl<const N: usize> From<&NdLayout<N>> for DynLayout {
    fn from(value: &NdLayout<N>) -> DynLayout {
        DynLayout::from_layout(value)
    }
}

impl<const N: usize> From<NdLayout<N>> for DynLayout {
    fn from(value: NdLayout<N>) -> DynLayout {
        DynLayout::from_layout(&value)
    }
}

/// MutLayout extends [Layout] with methods for creating, modifying and
/// transforming layouts.
pub trait MutLayout: Layout + Clone {
    /// Create a new contiguous layout with a given shape.
    fn from_shape(shape: Self::Index<'_>) -> Self;

    /// Create a layout with custom strides.
    ///
    /// The strides specify the offset gap between successive entries along a
    /// given axis. `overlap` controls whether the layout is allowed to map
    /// multiple indices to the same element. This can be true for immutable
    /// views, but must be false for tensors or views that are mutable.
    fn from_shape_and_strides(
        shape: Self::Index<'_>,
        strides: Self::Index<'_>,
        overlap: OverlapPolicy,
    ) -> Result<Self, FromDataError>;

    /// Move the axis at position `from` to `to` by swapping their strides.
    fn move_axis(&mut self, from: usize, to: usize);

    /// Return a layout with the axes permuted according to the given order.
    fn permuted(&self, order: Self::Index<'_>) -> Self;

    /// Return a new layout formed by reshaping this one to `shape`.
    ///
    /// This has the same requirements as
    /// [`reshaped_for_copy`](MutLayout::reshaped_for_copy) but also requires
    /// that the layout is contiguous.
    fn reshaped_for_view<S: IntoLayout>(&self, shape: S) -> Result<S::Layout, ReshapeError> {
        if !self.is_contiguous() {
            return Err(ReshapeError::NotContiguous);
        }
        self.reshaped_for_copy(shape)
    }

    /// Return a new layout formed by reshaping this one to `shape`.
    fn reshaped_for_copy<S: IntoLayout>(&self, shape: S) -> Result<S::Layout, ReshapeError> {
        let layout = shape.into_layout();
        if layout.len() != self.len() {
            return Err(ReshapeError::LengthMismatch);
        }
        Ok(layout)
    }

    // Modify the size of a dimension. This does not alter the strides.
    fn resize_dim(&mut self, dim: usize, size: usize);

    /// Reverse the order of dimensions. This is equivalent to
    /// `self.permuted([N-1, N-2, ... 0])`.
    fn transposed(&self) -> Self;

    /// Slice the layout and return a static rank layout with `M` dimensions.
    fn slice<const M: usize>(&self, range: &[SliceItem]) -> (Range<usize>, NdLayout<M>);

    /// Slice the layout and return a dynamic rank layout.
    fn slice_dyn(&self, range: &[SliceItem]) -> (Range<usize>, DynLayout);

    /// Return a layout with all size-one dimensions removed.
    fn squeezed(&self) -> DynLayout;

    /// Attempt to slice the layout or return an error if the range is invalid
    /// for the layout's shape.
    fn try_slice<R: IntoSliceItems>(
        &self,
        range: R,
    ) -> Result<(Range<usize>, DynLayout), SliceError>;
}

/// Trait for broadcasting a layout from one shape to another.
pub trait BroadcastLayout<L: MutLayout> {
    /// Broadcast the `self` layout to a given shape.
    fn broadcast<S: IntoLayout<Layout = L>>(&self, shape: S) -> L;
}

impl<const N: usize, const M: usize> BroadcastLayout<NdLayout<M>> for NdLayout<N> {
    fn broadcast<S: IntoLayout<Layout = NdLayout<M>>>(&self, shape: S) -> NdLayout<M> {
        let shape: [usize; M] = shape.as_ref().try_into().unwrap();
        self.broadcast(shape)
    }
}

impl<const N: usize> BroadcastLayout<DynLayout> for NdLayout<N> {
    fn broadcast<S: IntoLayout<Layout = DynLayout>>(&self, shape: S) -> DynLayout {
        let dyn_layout: DynLayout = self.into();
        dyn_layout.broadcast(shape.as_ref())
    }
}

impl BroadcastLayout<DynLayout> for DynLayout {
    fn broadcast<S: IntoLayout<Layout = DynLayout>>(&self, shape: S) -> DynLayout {
        self.broadcast(shape.as_ref())
    }
}

impl<const N: usize> BroadcastLayout<NdLayout<N>> for DynLayout {
    fn broadcast<S: IntoLayout<Layout = NdLayout<N>>>(&self, shape: S) -> NdLayout<N> {
        let dyn_broadcast = self.broadcast(shape.as_ref());
        (&dyn_broadcast).try_into().unwrap()
    }
}

impl<const N: usize> MutLayout for NdLayout<N> {
    fn from_shape(shape: [usize; N]) -> Self {
        Self::from_shape(shape)
    }

    fn from_shape_and_strides(
        shape: Self::Index<'_>,
        strides: Self::Index<'_>,
        overlap: OverlapPolicy,
    ) -> Result<Self, FromDataError> {
        Self::try_from_shape_and_strides(shape, strides, overlap)
    }

    fn move_axis(&mut self, from: usize, to: usize) {
        assert!(from < N && to < N);
        let mut dyn_layout = self.as_dyn();
        dyn_layout.move_axis(from, to);
        *self = NdLayout::try_from(&dyn_layout).unwrap();
    }

    fn permuted(&self, order: [usize; N]) -> NdLayout<N> {
        self.permuted(order)
    }

    fn resize_dim(&mut self, dim: usize, size: usize) {
        self.resize_dim(dim, size)
    }

    fn transposed(&self) -> NdLayout<N> {
        self.transposed()
    }

    fn slice<const M: usize>(&self, range: &[SliceItem]) -> (Range<usize>, NdLayout<M>) {
        self.slice(range)
    }

    fn slice_dyn(&self, range: &[SliceItem]) -> (Range<usize>, DynLayout) {
        self.as_dyn().slice(range)
    }

    fn squeezed(&self) -> DynLayout {
        self.as_dyn().squeezed()
    }

    fn try_slice<R: IntoSliceItems>(
        &self,
        range: R,
    ) -> Result<(Range<usize>, DynLayout), SliceError> {
        let items = range.into_slice_items();
        self.as_dyn().try_slice(items.as_ref())
    }
}

impl MutLayout for DynLayout {
    fn from_shape(shape: &[usize]) -> Self {
        Self::from_shape(shape)
    }

    fn from_shape_and_strides(
        shape: &[usize],
        strides: &[usize],
        overlap: OverlapPolicy,
    ) -> Result<Self, FromDataError> {
        Self::try_from_shape_and_strides(shape, strides, overlap)
    }

    fn move_axis(&mut self, from: usize, to: usize) {
        self.move_axis(from, to)
    }

    fn permuted(&self, order: &[usize]) -> DynLayout {
        self.permuted(order)
    }

    fn resize_dim(&mut self, dim: usize, size: usize) {
        self.resize_dim(dim, size)
    }

    fn transposed(&self) -> DynLayout {
        self.transposed()
    }

    fn slice<const M: usize>(&self, range: &[SliceItem]) -> (Range<usize>, NdLayout<M>) {
        let (offset_range, dyn_layout) = self.slice(range);
        let nd_layout = NdLayout::try_from(&dyn_layout).unwrap_or_else(|_| {
            panic!(
                "expected sliced tensor to have {} dims but it has {}",
                M,
                dyn_layout.ndim()
            );
        });
        (offset_range, nd_layout)
    }

    fn slice_dyn(&self, range: &[SliceItem]) -> (Range<usize>, DynLayout) {
        self.slice(range)
    }

    fn squeezed(&self) -> DynLayout {
        self.squeezed()
    }

    fn try_slice<R: IntoSliceItems>(
        &self,
        range: R,
    ) -> Result<(Range<usize>, DynLayout), SliceError> {
        let items = range.into_slice_items();
        self.try_slice(items.as_ref())
    }
}

/// Trait for shapes which can be used to create a contiguous layout.
///
/// This is implemented for `[usize; N]` for creating static-rank layouts from
/// arrays, and `&[usize]` for creating dynamic-rank layouts from slices.
pub trait IntoLayout: AsRef<[usize]> {
    /// The type of layout produced from this shape.
    type Layout: MutLayout;

    /// Convert this shape into a contiguous layout.
    fn into_layout(self) -> Self::Layout;
}

impl<const N: usize> IntoLayout for [usize; N] {
    type Layout = NdLayout<N>;

    #[inline]
    fn into_layout(self) -> NdLayout<N> {
        NdLayout::from_shape(self)
    }
}

impl<'a> IntoLayout for &'a [usize] {
    type Layout = DynLayout;

    #[inline]
    fn into_layout(self) -> DynLayout {
        DynLayout::from_shape(self)
    }
}

/// Trait which extends [MutLayout] with support for changing the number of
/// dimensions in-place.
///
/// This is only implemented for [DynLayout], since layouts that have a static
/// rank cannot change their dimension count at runtime.
pub trait ResizeLayout: MutLayout {
    /// Insert a size-one axis at the given index in the shape. This will have
    /// the same stride as the dimension that follows it.
    fn insert_axis(&mut self, index: usize);

    /// Merge consecutive axes where possible.
    ///
    /// This "simplifies" the layout by minimizing the number of dimensions
    /// while preserving the iteration order.
    fn merge_axes(&mut self);
}

impl ResizeLayout for DynLayout {
    fn insert_axis(&mut self, index: usize) {
        self.insert_dim(index)
    }

    fn merge_axes(&mut self) {
        if self.ndim() == 0 {
            return;
        }

        let mut shape = SmallVec::<[usize; 4]>::new();
        let mut strides = SmallVec::<[usize; 4]>::new();

        shape.push(self.size(self.ndim() - 1));
        strides.push(self.stride(self.ndim() - 1));

        for (&outer_size, &outer_stride) in
            self.shape().iter().zip(self.strides().iter()).rev().skip(1)
        {
            let inner_stride = strides.last().unwrap();
            let inner_size = shape.last().unwrap();
            let can_merge = outer_size == 1 || (outer_stride == inner_stride * inner_size);

            if can_merge {
                let prev_size = shape.last_mut().unwrap();
                *prev_size *= outer_size;
            } else {
                shape.push(outer_size);
                strides.push(outer_stride);
            }
        }

        shape.reverse();
        strides.reverse();

        self.shape_and_strides = shape.iter().chain(strides.iter()).copied().collect();
    }
}

/// Trait for converting types into indices for use with a given layout.
///
/// Static-rank tensors can be indexed with `[usize; N]` arrays. Dynamic-rank
/// tensors can be indexed with any type that can be converted to an `&[usize]`
/// slice.
pub trait AsIndex<L: Layout> {
    /// Convert `self` into an index for use the layout `L`.
    fn as_index(&self) -> L::Index<'_>;
}

impl<T: AsRef<[usize]>> AsIndex<DynLayout> for T {
    fn as_index(&self) -> &[usize] {
        self.as_ref()
    }
}

impl<const N: usize> AsIndex<NdLayout<N>> for [usize; N] {
    fn as_index(&self) -> [usize; N] {
        *self
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::OverlapPolicy;
    use crate::errors::ReshapeError;
    use crate::layout::{DynLayout, Layout, MutLayout, ResizeLayout};
    use crate::SliceItem;

    #[test]
    fn test_is_broadcast() {
        // Non-empty, contiguous layout
        let layout = DynLayout::from_shape(&[5, 5]);
        assert!(!layout.is_broadcast());

        // Empty layout
        let layout = DynLayout::from_shape(&[5, 0]);
        assert!(!layout.is_broadcast());

        // Broadcasting layout
        let layout =
            DynLayout::try_from_shape_and_strides(&[5, 5], &[0, 0], OverlapPolicy::AllowOverlap)
                .unwrap();
        assert!(layout.is_broadcast());
    }

    #[test]
    fn test_try_from_shape_and_strides() {
        struct Case<'a> {
            shape: &'a [usize],
            strides: &'a [usize],
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
            let layout = DynLayout::try_from_shape_and_strides(
                case.shape,
                case.strides,
                OverlapPolicy::AllowOverlap,
            )
            .unwrap();
            assert_eq!(layout.shape(), case.shape);
            assert_eq!(layout.strides(), case.strides);
        }
    }

    #[test]
    fn test_move_axis() {
        let mut layout = DynLayout::from_shape(&[2, 4, 8]);
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
        let mut layout = DynLayout::from_shape(&[2, 4, 8]);
        layout.move_axis(3, 0);
    }

    #[test]
    #[should_panic]
    fn test_move_axis_invalid_to() {
        let mut layout = DynLayout::from_shape(&[2, 4, 8]);
        layout.move_axis(0, 3);
    }

    #[test]
    #[should_panic(expected = "permutation is invalid")]
    fn test_permute_invalid_len() {
        let mut layout = DynLayout::from_shape(&[5, 5]);
        layout.permute(&[1, 0, 3]);
    }

    #[test]
    #[should_panic(expected = "permutation is invalid")]
    fn test_permute_too_few_dims() {
        let mut layout = DynLayout::from_shape(&[5, 5]);
        layout.permute(&[1]);
    }

    #[test]
    #[should_panic(expected = "permutation is invalid")]
    fn test_permute_repeated_dims() {
        let mut layout = DynLayout::from_shape(&[5, 5]);
        layout.permute(&[1, 1]);
    }

    #[test]
    fn test_reshaped() {
        struct Case<'a> {
            layout: DynLayout,
            new_shape: &'a [usize],
            for_copy: bool,
            error: Option<ReshapeError>,
        }

        let cases = [
            // Reshapes that don't allow copying.
            Case {
                layout: DynLayout::from_shape(&[2, 2]),
                new_shape: &[4],
                for_copy: false,
                error: None,
            },
            Case {
                layout: DynLayout::from_shape(&[2, 2]).transposed(),
                new_shape: &[4],
                for_copy: false,
                error: Some(ReshapeError::NotContiguous),
            },
            Case {
                layout: DynLayout::from_shape(&[2, 2]),
                new_shape: &[3],
                for_copy: false,
                error: Some(ReshapeError::LengthMismatch),
            },
            // Reshapes that do allow copying.
            Case {
                layout: DynLayout::from_shape(&[2, 2]).transposed(),
                new_shape: &[4],
                for_copy: true,
                error: None,
            },
            Case {
                layout: DynLayout::from_shape(&[2, 2]),
                new_shape: &[3],
                for_copy: false,
                error: Some(ReshapeError::LengthMismatch),
            },
        ];

        for Case {
            layout,
            new_shape,
            for_copy,
            error,
        } in cases
        {
            let reshaped = if for_copy {
                layout.reshaped_for_copy(new_shape)
            } else {
                layout.reshaped_for_view(new_shape)
            };

            assert_eq!(reshaped.as_ref().err(), error.as_ref());
            if let Ok(new_layout) = reshaped {
                assert_eq!(new_layout.shape(), new_shape);
            }
        }
    }

    #[test]
    fn test_squeezed() {
        let layout = DynLayout::from_shape(&[1, 1, 10, 20]);
        let squeezed = layout.squeezed();
        assert_eq!(squeezed.shape(), &[10, 20]);
        assert_eq!(squeezed.strides(), &[20, 1]);
    }

    #[test]
    #[should_panic(expected = "Slice index is invalid for tensor shape")]
    fn test_slice_invalid_index() {
        let layout = DynLayout::from_shape(&[3, 5]);
        layout.slice(&[SliceItem::Index(4), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Slice index is invalid for tensor shape")]
    fn test_slice_invalid_negative_index() {
        let layout = DynLayout::from_shape(&[3, 5]);
        layout.slice(&[SliceItem::Index(-4)]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_range() {
        let layout = DynLayout::from_shape(&[3, 5]);
        layout.slice(&[SliceItem::Range((1..4).into()), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Slice range is invalid for tensor shape")]
    fn test_slice_invalid_from_range() {
        let layout = DynLayout::from_shape(&[3, 5]);
        layout.slice(&[SliceItem::Range((4..).into()), SliceItem::Index(0)]);
    }

    #[test]
    #[should_panic(expected = "Cannot slice with negative step")]
    fn test_slice_negative_step() {
        let layout = DynLayout::from_shape(&[3, 5]);
        layout.slice(&[SliceItem::full_range(), SliceItem::range(0, None, -1)]);
    }

    #[test]
    fn test_size_stride() {
        let layout = DynLayout::from_shape(&[10, 20, 30]);
        for (dim, (&size, &stride)) in
            zip(layout.shape().iter(), layout.strides().iter()).enumerate()
        {
            assert_eq!(layout.size(dim), size);
            assert_eq!(layout.stride(dim), stride);
        }
    }

    #[test]
    fn test_merge_axes() {
        struct Case<'a> {
            shape: &'a [usize],
            strides: &'a [usize],
            merged_shape: &'a [usize],
            merged_strides: &'a [usize],
        }

        let cases = [
            // Empty shape
            Case {
                shape: &[],
                strides: &[],
                merged_shape: &[],
                merged_strides: &[],
            },
            // Vector
            Case {
                shape: &[10],
                strides: &[2],
                merged_shape: &[10],
                merged_strides: &[2],
            },
            // Simple contiguous layout
            Case {
                shape: &[10, 10],
                strides: &[10, 1],
                merged_shape: &[100],
                merged_strides: &[1],
            },
            // Transposed matrix
            Case {
                shape: &[10, 10],
                strides: &[1, 10],
                merged_shape: &[10, 10],
                merged_strides: &[1, 10],
            },
            // Leading 1-sized dims
            Case {
                shape: &[1, 10, 10],
                strides: &[10, 1, 10],
                merged_shape: &[10, 10],
                merged_strides: &[1, 10],
            },
            // Inner 1-sized dims
            Case {
                shape: &[2, 1, 1, 2],
                strides: &[2, 2, 2, 1],
                merged_shape: &[4],
                merged_strides: &[1],
            },
            // Inner 1-sized dims that have been shifted over from the left,
            // ie. where the 1-sized dims where inserted at the left and then
            // shifted over to the middle.
            Case {
                shape: &[2, 1, 1, 2],
                strides: &[2, 4, 4, 1],
                merged_shape: &[4],
                merged_strides: &[1],
            },
        ];

        for Case {
            shape,
            strides,
            merged_shape,
            merged_strides,
        } in cases
        {
            let mut layout =
                DynLayout::try_from_shape_and_strides(shape, strides, OverlapPolicy::AllowOverlap)
                    .unwrap();
            layout.merge_axes();
            assert_eq!(layout.shape(), merged_shape);
            assert_eq!(layout.strides(), merged_strides);
        }
    }
}
