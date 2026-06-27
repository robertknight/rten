//! rten-serialize supports reading and writing tensors in popular formats.
//!
//! All formats are disabled by support and must be enabled using the
//! corresponding crate features.
//!
//! # Features
//!
//!  - **npy** - Enable the .npy format
//!  - **npz** - Enable the .npz format
//!
//! # Supported formats
//!
//! [NumPy's .npy format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
//! is a simple format for reading and writing single tensors.
//!
//! [NumPy's .npz format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
//! is an archive format for reading and writing multiple tensors.
//!
//! # Usage
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use rten_serialize::npy as npy;
//! use rten_serialize::npz as npz;
//! use rten_tensor::prelude::*;
//! use rten_tensor::NdTensor;
//!
//! // Write a single tensor.
//! let matrix = NdTensor::from([[1i32, 2], [3, 4]]);
//! npy::write_to_file("tensor.npy", &matrix)?;
//!
//! // Read a single tensor.
//! let matrix_2 = npy::read_from_file("tensor.npy")?.into_rank::<2>()?;
//! assert_eq!(matrix, matrix_2);
//!
//! // Write multiple tensors.
//! npz::write_to_file("tensors.npz", [("some_matrix", matrix.view())])?;
//!
//! // Read a map of name to tensor
//! let tensors = npz::read_from_file("tensors.npz")?;
//! let matrix_3 = tensors
//!     .get("some_matrix")
//!     .ok_or("tensor not found")?
//!     .view()
//!     .into_rank::<2>()?;
//! assert_eq!(matrix, matrix_3);
//!
//! # Ok(()) }
//! ```

/// Read and write tensors in NumPy's `.npy` format.
#[cfg(feature = "npy")]
pub mod npy;

/// Read and write tensors in NumPy's `.npz` format.
#[cfg(feature = "npz")]
pub mod npz;
