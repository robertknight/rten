use std::error::Error;
use std::fmt;
use std::fmt::Display;

/// Errors with matrix multiplication inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum GemmError {
    /// Columns of LHS and RHS inputs do not match.
    KSizeMismatch,
    /// Bias vector length does not match the corresponding output matrix size.
    WrongBiasSize,
    /// The buffer provided for the output is too short.
    OutputNotLargeEnough,
    /// The data was packed with a kernel that uses a different layout than
    /// the current kernel.
    PackedDataKernelMismatch,
    /// The data was packed with different cache blocking parameters than are
    /// currently being used.
    PackedDataBlockingMismatch,
}

impl Display for GemmError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::KSizeMismatch => {
                write!(fmt, "columns of matrix `a` must match rows of matrix `b`")
            }
            Self::WrongBiasSize => write!(fmt, "bias vector length is incorrect"),
            Self::OutputNotLargeEnough => write!(fmt, "output buffer is too small"),
            Self::PackedDataKernelMismatch => {
                write!(fmt, "matrix was packed with a different kernel")
            }
            Self::PackedDataBlockingMismatch => {
                write!(fmt, "matrix was packed with a different blocking size")
            }
        }
    }
}

impl Error for GemmError {}
