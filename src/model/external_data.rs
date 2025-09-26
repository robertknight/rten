//! Functions for loading tensor data stored externally to the main model file.
//!
//! This is used for ONNX models when `TensorProto`s reference external data
//! files.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::ModelLoadError;
use rten_base::byte_cast::{Pod, cast_pod_arc, cast_uninit_pod_mut_slice};
use rten_tensor::AssumeInit;

/// Specifies the location of tensor data which is stored externally from the
/// main model file.
pub struct ExternalDataLocation<'a> {
    /// Filename of the file containing the external data.
    pub path: &'a str,

    /// Offset of the start of the tensor data in the file.
    pub offset: u64,

    /// Length of the tensor data in bytes.
    pub length: u64,
}

/// Errors reading tensor data from an external file.
#[derive(Debug)]
pub enum ExternalDataError {
    /// An IO error occurred when accessing the external file.
    IoError(std::io::Error),

    /// The length of the external data is not a multiple of the element size
    /// (eg. if reading i32 elements, the length must be a multiple of 4).
    InvalidLength,

    /// External data is not supported in the current environment.
    NotSupported,
}

impl std::fmt::Display for ExternalDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(err) => write!(f, "io error: {}", err),
            Self::InvalidLength => write!(f, "length is not a multiple of element size"),
            Self::NotSupported => {
                write!(f, "external data not supported on this system")
            }
        }
    }
}

impl std::error::Error for ExternalDataError {}

impl From<ExternalDataError> for ModelLoadError {
    fn from(err: ExternalDataError) -> ModelLoadError {
        ModelLoadError::ExternalDataError(Box::new(err))
    }
}

/// Trait for loading data from an external file.
pub trait ExternalDataLoader {
    /// Load 8-bit elements from an external file.
    fn load_u8(&self, location: ExternalDataLocation) -> Result<Arc<[u8]>, ExternalDataError>;

    /// Load 32-bit elements from an external file.
    fn load_u32(&self, location: ExternalDataLocation) -> Result<Arc<[u32]>, ExternalDataError>;

    /// Load 64-bit elements from an external file.
    fn load_u64(&self, location: ExternalDataLocation) -> Result<Arc<[u64]>, ExternalDataError>;
}

/// External file loader that uses standard file IO.
pub struct ExternalFileLoader {
    /// Path to the directory external data files.
    base_path: PathBuf,

    /// Map of external data file name to open file.
    files: RefCell<HashMap<String, File>>,
}

impl ExternalFileLoader {
    pub fn new(model_path: &Path) -> Self {
        let base_path = model_path.parent().expect("should have a parent dir");
        Self {
            base_path: base_path.to_path_buf(),
            files: HashMap::new().into(),
        }
    }

    /// Read a chunk of a file specified by `location` into `buf`.
    ///
    /// The path specified by `location.path` is resolved relative to this
    /// loader's base path.
    ///
    /// The size of the file chunk specified by `location.length` must match
    /// the length of `buf`.
    fn read<T: Pod>(
        &self,
        location: ExternalDataLocation,
        mut buf: Arc<[MaybeUninit<T>]>,
    ) -> Result<Arc<[T]>, ExternalDataError> {
        // On a big-endian system we'd need to transmute bytes after reading
        // if we're reading multi-byte elements.
        if cfg!(target_endian = "big") {
            return Err(ExternalDataError::NotSupported);
        }

        let mut files = self.files.borrow_mut();
        let file = get_or_open_file(&mut files, &self.base_path, location.path)?;
        file.seek(SeekFrom::Start(location.offset))
            .map_err(ExternalDataError::IoError)?;
        let dst_slice = Arc::get_mut(&mut buf).unwrap();
        let dst_bytes = cast_uninit_pod_mut_slice::<T, u8>(dst_slice).unwrap();

        // FIXME: This should use `read_buf` to fill the buffer when that is
        // stabilized. Passing an uninitialized slice to `read_exact` is UB. We
        // are relying on the impl for `File` to follow the recommendations for
        // `read_exact` that impls should only write to the slice and not read
        // from it.
        file.read_exact(unsafe { dst_bytes.assume_init() })
            .map_err(ExternalDataError::IoError)?;

        // FIXME: We assume here that `read_exact` initialized the buffer,
        // although the contract does not guarantee this.
        Ok(unsafe { buf.assume_init() })
    }
}

impl ExternalDataLoader for ExternalFileLoader {
    fn load_u8(&self, location: ExternalDataLocation) -> Result<Arc<[u8]>, ExternalDataError> {
        let uninit_data = Arc::<[u8]>::new_uninit_slice(location.length as usize);
        let data = self.read(location, uninit_data)?;
        Ok(cast_pod_arc(data).unwrap())
    }

    fn load_u32(&self, location: ExternalDataLocation) -> Result<Arc<[u32]>, ExternalDataError> {
        let Some(len_u32) = location.length.checked_div(size_of::<u32>() as u64) else {
            return Err(ExternalDataError::InvalidLength);
        };
        let uninit_data = Arc::<[u32]>::new_uninit_slice(len_u32 as usize);
        let data = self.read(location, uninit_data)?;
        Ok(cast_pod_arc(data).unwrap())
    }

    fn load_u64(&self, location: ExternalDataLocation) -> Result<Arc<[u64]>, ExternalDataError> {
        let Some(len_u64) = location.length.checked_div(size_of::<u64>() as u64) else {
            return Err(ExternalDataError::InvalidLength);
        };
        let uninit_data = Arc::<[u64]>::new_uninit_slice(len_u64 as usize);
        let data = self.read(location, uninit_data)?;
        Ok(cast_pod_arc(data).unwrap())
    }
}

/// Get a handle to the open file named `path` in `files` or open the file for reading.
fn get_or_open_file<'a>(
    files: &'a mut HashMap<String, File>,
    base_path: &Path,
    path: &str,
) -> Result<&'a mut File, ExternalDataError> {
    if files.get(path).is_none() {
        // TODO - Disallow file paths that don't contain an `onnx_data` extension
        // or contain a directory path.
        let mut file_path = base_path.to_path_buf();
        file_path.push(path);
        let file = File::open(file_path).map_err(ExternalDataError::IoError)?;
        files.insert(path.to_string(), file);
    }
    Ok(files.get_mut(path).unwrap())
}
