//! rten-serialize supports reading and writing tensors in popular formats.
//!
//! All formats are disabled by default and must be enabled using the
//! corresponding crate features.
//!
//! # Features
//!
//!  - **npy** - Enable the .npy format
//!  - **npz** - Enable the .npz format
//!  - **npz-compression** - Enable reading compressed (deflate) .npz archives,
//!    such as those produced by `numpy.savez_compressed`
//!
//! # Data types
//!
//! Readers return tensors as a [`Value`], and writers accept them as a
//! [`View`]. These are enums that hold a tensor of any supported
//! [element type](Element), since a single file may contain tensors with
//! different element types. Use [`Value::into_type`] / [`Value::as_type`] to
//! extract a typed tensor.
//!
//! # Supported formats
//!
//! [NumPy's .npy format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
//! is a simple format for reading and writing single tensors.
//!
//! [NumPy's .npz format](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
//! is an archive format for reading and writing multiple tensors. Archives are
//! always written uncompressed (matching `numpy.savez`). Reading compressed
//! archives requires the **npz-compression** feature.
#![cfg_attr(
    feature = "npz",
    doc = r#"
# Usage

```
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use rten_serialize::{npy, npz};
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;

let matrix = NdTensor::from([[1i32, 2], [3, 4]]);

// Write and read back a single tensor.
npy::write_to_file("tensor.npy", matrix.view())?;
let matrix_2 = npy::read_from_file("tensor.npy")?.into_type::<i32>()?.into_rank::<2>()?;
assert_eq!(matrix, matrix_2);

// Write and read back a map of named tensors.
npz::write_to_file("tensors.npz", [("some_matrix", matrix.view())])?;
let mut tensors = npz::read_from_file("tensors.npz")?;

// Borrow a tensor from the map as a typed view.
let matrix_3 = tensors
    .get("some_matrix")
    .ok_or("tensor not found")?
    .as_type::<i32>()?
    .into_rank::<2>()?;
assert_eq!(matrix, matrix_3);

// Or remove an entry to extract its tensor as an owned value.
let matrix_4 = tensors
    .remove("some_matrix")
    .ok_or("tensor not found")?
    .into_type::<i32>()?
    .into_rank::<2>()?;
assert_eq!(matrix, matrix_4);

# Ok(()) }
```
"#
)]

mod value;
pub use value::{DataType, Element, TypeError, Value, View};

/// Read and write tensors in NumPy's `.npy` format.
#[cfg(feature = "npy")]
pub mod npy;

/// Read and write tensors in NumPy's `.npz` format.
#[cfg(feature = "npz")]
pub mod npz;
