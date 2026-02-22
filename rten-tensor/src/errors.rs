//! Error types that are reported by various tensor operations.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::slice_range::SliceRange;

/// Error in a tensor operation if the dimension count is incorrect.
#[derive(Debug, PartialEq)]
pub struct DimensionError {
    /// Actual number of dimensions the tensor has.
    pub actual: usize,

    /// Expected number of dimensions.
    pub expected: usize,
}

impl Display for DimensionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tensor has {} dims but expected {}",
            self.actual, self.expected
        )
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
    TooManyDims {
        /// Number of axes in the tensor.
        ndim: usize,
        /// Number of items in the slice spec.
        range_ndim: usize,
    },

    /// The slice spec specified an axis that is equal to or greater than the
    /// dimension count.
    InvalidAxis { axis: usize },

    /// An index in the slice spec is out of bounds for the corresponding tensor
    /// dimension.
    InvalidIndex {
        /// Axis that the error applies to.
        axis: usize,
        /// Index in the slice range.
        index: isize,
        /// Size of the dimension.
        size: usize,
    },

    /// A range in the slice spec is out of bounds for the corresponding tensor
    /// dimension.
    InvalidRange {
        /// Axis that the error applies to.
        axis: usize,

        /// The range item.
        range: SliceRange,

        /// Size of the dimension.
        size: usize,
    },

    /// The step in a slice range is negative, in a context where this is not
    /// supported.
    InvalidStep {
        /// Axis that the error applies to.
        axis: usize,

        /// Size of the dimension.
        step: isize,
    },

    /// There is a mismatch between the actual and expected number of axes
    /// in the output slice.
    OutputDimsMismatch { actual: usize, expected: usize },
}

impl Display for SliceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SliceError::TooManyDims { ndim, range_ndim } => {
                write!(
                    f,
                    "slice range has {} items but tensor has only {} dims",
                    range_ndim, ndim
                )
            }
            SliceError::InvalidAxis { axis } => write!(f, "slice axis {} is invalid", axis),
            SliceError::InvalidIndex { axis, index, size } => write!(
                f,
                "slice index {} is invalid for axis ({}) of size {}",
                index, axis, size
            ),
            SliceError::InvalidRange { axis, range, size } => write!(
                f,
                "slice range {:?} is invalid for axis ({}) of size {}",
                range, axis, size
            ),
            SliceError::InvalidStep { axis, step } => {
                write!(f, "slice step {} is invalid for axis {}", step, axis)
            }
            SliceError::OutputDimsMismatch { actual, expected } => {
                write!(
                    f,
                    "slice output dims {} does not match expected dims {}",
                    actual, expected
                )
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

impl Error for ExpandError {}
