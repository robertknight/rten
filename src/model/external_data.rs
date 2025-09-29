//! Functions for loading tensor data stored externally to the main model file.
//!
//! This is used for ONNX models when `TensorProto`s reference external data
//! files.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::ModelLoadError;
use rten_base::byte_cast::{Pod, cast_pod_slice};

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
    /// (eg. if reading i32 elements, the length must be a multiple of 4), or
    /// is larger than the maximum supported length (4GB in a 32-bit environment).
    InvalidLength,

    /// External data is not supported in the current environment.
    NotSupported,

    /// The length of the external data file is too short for the offset and
    /// length of the external data.
    TooShort {
        required_len: usize,
        actual_len: usize,
    },
}

impl std::fmt::Display for ExternalDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(err) => write!(f, "io error: {}", err),
            Self::InvalidLength => write!(f, "length is not a multiple of element size"),
            Self::NotSupported => {
                write!(f, "external data not supported on this system")
            }
            Self::TooShort {
                required_len,
                actual_len,
            } => write!(
                f,
                "external data file too short. read {required_len} of {actual_len} bytes"
            ),
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
    fn load_u8(&self, location: ExternalDataLocation) -> Result<Vec<u8>, ExternalDataError>;

    /// Load 32-bit elements from an external file.
    fn load_u32(&self, location: ExternalDataLocation) -> Result<Vec<u32>, ExternalDataError>;

    /// Load 64-bit elements from an external file.
    fn load_u64(&self, location: ExternalDataLocation) -> Result<Vec<u64>, ExternalDataError>;
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
    fn read<T: Pod>(&self, location: ExternalDataLocation) -> Result<Vec<T>, ExternalDataError> {
        // On a big-endian system we'd need to transmute bytes after reading
        // if we're reading multi-byte elements.
        if cfg!(target_endian = "big") {
            return Err(ExternalDataError::NotSupported);
        }

        let elem_size = size_of::<T>();
        assert!(elem_size != 0);

        if !location.length.is_multiple_of(elem_size as u64) || location.length > usize::MAX as u64
        {
            return Err(ExternalDataError::InvalidLength);
        }
        let vec_len = location.length as usize / elem_size;

        let mut files = self.files.borrow_mut();
        let mut file = get_or_open_file(&mut files, &self.base_path, location.path)?;
        file.seek(SeekFrom::Start(location.offset))
            .map_err(ExternalDataError::IoError)?;

        // Ideally we would fill the buffer in one call via [`Read::read_buf`].
        // Since that API is not stabilized yet, we fill in small chunks, which
        // requires extra copying.
        let mut remaining = vec_len;
        let mut buf = Vec::with_capacity(remaining);
        const TMP_SIZE: usize = 4096;
        let mut tmp = [0u8; TMP_SIZE];

        loop {
            let tmp_size = (remaining * elem_size).min(TMP_SIZE);
            let n_read =
                read_fill(&mut file, &mut tmp[..tmp_size]).map_err(ExternalDataError::IoError)?;

            // Reinterpret the bytes assuming we're on a little-endian system.
            // To support big-endian we'd need to byte-swap.
            let chunk = cast_pod_slice(&tmp[..n_read]).unwrap();
            remaining -= chunk.len();
            buf.extend_from_slice(chunk);

            if n_read < tmp.len() || remaining == 0 {
                break;
            }
        }

        if buf.len() != vec_len {
            return Err(ExternalDataError::TooShort {
                required_len: vec_len,
                actual_len: buf.len(),
            });
        }

        Ok(buf)
    }
}

/// Read from `src` repeatedly until `buf` is filled or we reach the end of the
/// file.
fn read_fill<R: Read>(mut src: R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut total_read = 0;
    loop {
        let n = src.read(&mut buf[total_read..])?;
        total_read += n;
        if n == 0 || total_read == buf.len() {
            break;
        }
    }
    Ok(total_read)
}

impl ExternalDataLoader for ExternalFileLoader {
    fn load_u8(&self, location: ExternalDataLocation) -> Result<Vec<u8>, ExternalDataError> {
        self.read(location)
    }

    fn load_u32(&self, location: ExternalDataLocation) -> Result<Vec<u32>, ExternalDataError> {
        self.read(location)
    }

    fn load_u64(&self, location: ExternalDataLocation) -> Result<Vec<u64>, ExternalDataError> {
        self.read(location)
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
