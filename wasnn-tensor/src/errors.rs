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

    /// Some indices will map to the same offset within the storage.
    ///
    /// This error can only occur when the storage is mutable.
    MayOverlap,
}

impl Display for FromDataError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FromDataError::StorageTooShort => write!(f, "Data too short"),
            FromDataError::MayOverlap => write!(f, "May have internal overlap"),
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
}

impl Display for SliceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SliceError::TooManyDims => write!(f, "slice spec has too many dims"),
            SliceError::InvalidIndex => write!(f, "slice index is invalid"),
            SliceError::InvalidRange => write!(f, "slice range is invalid"),
            SliceError::InvalidStep => write!(f, "slice step is invalid"),
        }
    }
}

impl Error for SliceError {}
