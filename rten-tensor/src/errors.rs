//! Error types that are reported by various tensor operations.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Error in a tensor operation if the dimension count is incorrect.
#[derive(Debug, PartialEq)]
pub struct DimensionError {}

impl Display for DimensionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dim count is incorrect")
    }
}

impl Error for DimensionError {}

/// Errors that can occur when constructing a tensor from existing data.
#[derive(Debug, PartialEq)]
pub enum FromDataError {
    /// Some indices will map to offsets that are beyond the end of the storage.
    StorageTooShort,

    /// The storage length was expected to exactly match the product of the
    /// shape, and it did not.
    StorageLengthMismatch,

    /// Some indices will map to the same offset within the storage.
    ///
    /// This error can only occur when the storage is mutable.
    MayOverlap,
}

impl Display for FromDataError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FromDataError::StorageTooShort => write!(f, "data too short"),
            FromDataError::StorageLengthMismatch => write!(f, "data length mismatch"),
            FromDataError::MayOverlap => write!(f, "may have internal overlap"),
        }
    }
}

impl Error for FromDataError {}

/// Errors that can occur when slicing a tensor.
#[derive(Clone, Debug, PartialEq)]
pub enum SliceError {
    /// The slice spec has more dimensions than the tensor being sliced.
    TooManyDims,

    /// An index in the slice spec is out of bounds for the corresponding tensor
    /// dimension.
    InvalidIndex,

    /// A range in the slice spec is out of bounds for the corresponding tensor
    /// dimension.
    InvalidRange,

    /// The step in a slice range is negative, in a context where this is not
    /// supported.
    InvalidStep,

    /// There is a mismatch between the actual and expected number of axes
    /// in the output slice.
    OutputDimsMismatch,
}

impl Display for SliceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SliceError::TooManyDims => write!(f, "slice spec has too many dims"),
            SliceError::InvalidIndex => write!(f, "slice index is invalid"),
            SliceError::InvalidRange => write!(f, "slice range is invalid"),
            SliceError::InvalidStep => write!(f, "slice step is invalid"),
            SliceError::OutputDimsMismatch => {
                write!(f, "slice output dims does not match expected dims")
            }
        }
    }
}

impl Error for SliceError {}

/// Errors that can occur while reshaping a layout or tensor.
#[derive(Clone, Debug, PartialEq)]
pub enum ReshapeError {
    /// Attempted to reshape a tensor without copying data, but the layout
    /// is not contiguous.
    NotContiguous,

    /// The reshaped layout would have a different length than the current
    /// layout.
    LengthMismatch,
}

impl std::fmt::Display for ReshapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ReshapeError::NotContiguous => write!(f, "view is not contiguous"),
            ReshapeError::LengthMismatch => write!(f, "new shape has a different length"),
        }
    }
}

/// Errors that can occur while expanding a tensor.
#[derive(Clone, Debug, PartialEq)]
pub enum ExpandError {
    /// The shape of the source and destination tensor do not match, excluding
    /// the dimensions along which expansion is happening.
    ShapeMismatch,

    /// The tensor cannot be resized without copying into a new buffer.
    InsufficientCapacity,
}

impl std::fmt::Display for ExpandError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpandError::ShapeMismatch => {
                write!(
                    f,
                    "non-expanding dimensions of source and destination do not match"
                )
            }
            ExpandError::InsufficientCapacity => {
                write!(f, "insufficient capacity for new dimension size")
            }
        }
    }
}
