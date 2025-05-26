use std::error::Error;
use std::fmt;
use std::fmt::Display;

/// Errors with matrix multiplication inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum GemmError {
    /// Number of LHS and RHS batch inputs do not match.
    BatchSizeMismatch,
    /// Number of columns in LHS does not match rows of RHS.
    KSizeMismatch,
    /// Bias vector length does not match the corresponding output matrix size.
    WrongBiasSize,
    /// Quantization parameter size does not match corresponding input size.
    WrongQuantParamSize,
    /// The buffer provided for the output is too short.
    OutputSizeMismatch,
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
            Self::BatchSizeMismatch => {
                write!(fmt, "batches of `a` and `b` matrices must have same length")
            }
            Self::KSizeMismatch => {
                write!(fmt, "columns of matrix `a` must match rows of matrix `b`")
            }
            Self::WrongBiasSize => write!(fmt, "bias vector length is incorrect"),
            Self::WrongQuantParamSize => {
                write!(fmt, "quantization parameter size does not match input")
            }
            Self::OutputSizeMismatch => write!(fmt, "output buffer has wrong length"),
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
