//! Functions for loading tensor data stored externally to the main model file.
//!
//! This is used for ONNX models when `TensorProto`s reference external data
//! files.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use super::ModelLoadError;

/// Specifies the location of tensor data which is stored externally from the
/// main model file.
#[derive(Debug)]
pub struct DataLocation {
    /// Name of the external data file.
    pub path: String,

    /// Offset of the start of the tensor data in bytes.
    pub offset: u64,

    /// Length of the tensor data in bytes.
    pub length: u64,
}

/// Errors reading tensor data from an external file.
#[derive(Debug)]
pub enum ExternalDataError {
    /// An IO error occurred when accessing the external file.
    IoError(std::io::Error),

    /// The length of the external data is too large.
    InvalidLength,

    /// An invalid path was specified.
    InvalidPath(PathBuf),

    /// External data is not supported in the current environment.
    NotSupported,

    /// The length of the external data file is too short for the offset and
    /// length of the external data.
    TooShort {
        /// Minimum length the file would need to be in bytes.
        required_len: usize,
        /// Actual length of the file in bytes.
        actual_len: usize,
    },

    /// The external data file path is disallowed.
    DisallowedPath(PathBuf),
}

impl std::fmt::Display for ExternalDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(err) => write!(f, "io error: {}", err),
            Self::InvalidLength => write!(f, "invalid data length"),
            Self::InvalidPath(path) => write!(f, "invalid path \"{}\"", path.display()),
            Self::NotSupported => write!(f, "external data not supported"),
            Self::TooShort {
                required_len,
                actual_len,
            } => write!(
                f,
                "file too short. required {} actual {}",
                required_len, actual_len
            ),
            Self::DisallowedPath(path) => {
                write!(f, "disallowed path \"{}\"", path.display(),)
            }
        }
    }
}

impl std::error::Error for ExternalDataError {}

impl From<std::io::Error> for ExternalDataError {
    fn from(val: std::io::Error) -> Self {
        Self::IoError(val)
    }
}

impl From<ExternalDataError> for ModelLoadError {
    fn from(err: ExternalDataError) -> ModelLoadError {
        ModelLoadError::ExternalDataError(Box::new(err))
    }
}

/// Trait for loading data from an external file.
pub trait DataLoader {
    /// Load data from the file and offset specified by `location`.
    fn load(&self, location: &DataLocation) -> Result<Vec<u8>, ExternalDataError>;
}

/// Check if `path` is an allowed path for external data files for a given
/// model path.
///
/// Any data loaded from an external data file can potentially be returned in
/// inference results (directly or indirectly). Hence some measures are taken to
/// prevent loading of data from files not intended for this. The ONNX
/// documentation states that the only restriction is that parent directory
/// components ("..") are disallowed
/// (https://onnx.ai/onnx/repo-docs/ExternalData.html#external-data-field). This
/// implementation imposes additional restrictions:
///
///  - The file must have one of the known extensions used for tensor data
///    ("data", "onnx_data", "onnx_data_N" etc.)
///  - The file must be a relative path with only a filename component (ie. it
///    must be located in the same directory as the model)
fn is_allowed_external_data_path(path: &Path) -> bool {
    // Data file path must be relative and consist only of a filename.
    let mut components = path.components();
    let Some(Component::Normal(_)) = components.next() else {
        return false;
    };
    if components.next().is_some() {
        return false;
    }

    // Check for allowed extension. The most common extensions used are "data"
    // or "onnx_data", but large files are sometimes split into pieces eg.
    // "onnx_data_N".
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.starts_with("data") => true,
        Some(ext) if ext.starts_with("onnx_data") => true,
        _ => false,
    }
}

/// External data loader that uses standard file IO.
pub struct FileLoader {
    /// Path to directory containing external data.
    dir_path: PathBuf,

    /// Map of external data file name to open file.
    files: RefCell<HashMap<PathBuf, File>>,
}

impl FileLoader {
    /// Create an external data loader which loads data for the model file
    /// specified by `model_path`.
    ///
    /// Data file paths will be resolved relative to the directory containing
    /// `model_path`.
    pub fn new(model_path: &Path) -> Result<Self, ExternalDataError> {
        // Resolve the path now to avoid the possibility of loading data from
        // an unexpected location if `model_path` is relative and the current
        // working directory changes before a data file is loaded.
        let model_path = if !cfg!(target_arch = "wasm32") {
            model_path.canonicalize()?
        } else {
            // On WASM / WASI `Path::canonicalize` is not available.
            model_path.to_path_buf()
        };

        if !model_path.is_file() {
            return Err(ExternalDataError::InvalidPath(model_path));
        }
        let dir_path = model_path
            .parent()
            // Since `model_path` is a file path, it cannot be the root ("/").
            .expect("should have parent dir")
            .to_path_buf();

        Ok(Self {
            dir_path,
            files: HashMap::new().into(),
        })
    }

    fn read(&self, location: &DataLocation) -> Result<Vec<u8>, ExternalDataError> {
        // On a big-endian system we'd need to perform byte-swapping while loading.
        if cfg!(target_endian = "big") {
            return Err(ExternalDataError::NotSupported);
        }

        // On a 32-bit systems assume we can't load more than 2GB of data.
        if location.length > isize::MAX as u64 {
            return Err(ExternalDataError::InvalidLength);
        }
        let vec_len = location.length as usize;

        let mut files = self.files.borrow_mut();
        let mut file = get_or_open_file(&mut files, &self.dir_path, Path::new(&location.path))?;
        file.seek(SeekFrom::Start(location.offset))
            .map_err(ExternalDataError::IoError)?;

        // Ideally we would fill the buffer in one call via [`Read::read_buf`].
        // Since that API is not stabilized yet, we fill in small chunks, which
        // requires extra copying.
        let mut remaining = vec_len;
        let mut buf = Vec::with_capacity(remaining);

        // Buffer size chosen to match BufReader's default.
        const TMP_SIZE: usize = 8192;
        let mut tmp = [0u8; TMP_SIZE];

        loop {
            let tmp_size = remaining.min(TMP_SIZE);
            let n_read =
                read_fill(&mut file, &mut tmp[..tmp_size]).map_err(ExternalDataError::IoError)?;
            let chunk = &tmp[..n_read];
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

impl DataLoader for FileLoader {
    fn load(&self, location: &DataLocation) -> Result<Vec<u8>, ExternalDataError> {
        self.read(location)
    }
}

fn get_or_open_file<'a>(
    files: &'a mut HashMap<PathBuf, File>,
    dir_path: &Path,
    data_path: &Path,
) -> Result<&'a mut File, ExternalDataError> {
    let data_path = Path::new(data_path);
    if !is_allowed_external_data_path(data_path) {
        return Err(ExternalDataError::DisallowedPath(data_path.into()));
    }

    // Check if we already opened the file.
    if files.get(data_path).is_none() {
        let mut file_path = dir_path.to_path_buf();
        file_path.push(data_path);
        let file = File::open(file_path).map_err(ExternalDataError::IoError)?;
        files.insert(data_path.into(), file);
    }

    Ok(files.get_mut(data_path).unwrap())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{DataLoader, DataLocation, FileLoader};
    use rten_testing::TestCases;

    fn temp_dir() -> PathBuf {
        if cfg!(target_arch = "wasm32") {
            // temp_dir is not available on WASM / WASI, so just use the current
            // directory.
            PathBuf::new()
        } else {
            std::env::temp_dir()
        }
    }

    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn new(name: impl AsRef<Path>, content: &[u8]) -> std::io::Result<Self> {
            let mut path = temp_dir();
            path.push(name);
            std::fs::write(&path, content)?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            std::fs::remove_file(&self.path).expect("should remove file");
        }
    }

    #[test]
    fn test_file_loader() {
        let bytes: Vec<u8> = (0..32).collect();
        let model_file = TempFile::new("test_load_external_data.onnx", &[]).unwrap();
        let data_file = TempFile::new("test_load_external_data.onnx.data", &bytes).unwrap();

        let data_filename = data_file
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        #[derive(Debug)]
        struct Case {
            location: DataLocation,
            expected: Result<Vec<u8>, String>,
        }

        let cases = [
            // Part of file
            Case {
                location: DataLocation {
                    path: data_filename.clone(),
                    offset: 8,
                    length: 8,
                },
                expected: Ok(bytes[8..16].into()),
            },
            // Full file
            Case {
                location: DataLocation {
                    path: data_filename.clone(),
                    offset: 0,
                    length: bytes.len() as u64,
                },
                expected: Ok(bytes.clone()),
            },
            // Empty path
            Case {
                location: DataLocation {
                    path: String::new(),
                    offset: 0,
                    length: 0,
                },
                expected: Err("disallowed path".into()),
            },
            // Path containing parent directory
            Case {
                location: DataLocation {
                    path: "../foo.data".into(),
                    offset: 0,
                    length: 0,
                },
                expected: Err("disallowed path".into()),
            },
            // Path with disallowed extension
            Case {
                location: DataLocation {
                    path: "not_a_data_file.md".into(),
                    offset: 0,
                    length: 0,
                },
                expected: Err("disallowed path".into()),
            },
            // File does not exist
            Case {
                location: DataLocation {
                    path: "file_does_not_exist.data".into(),
                    offset: 0,
                    length: 0,
                },
                expected: Err("No such file or directory".into()),
            },
            // Range extends beyond end of file
            Case {
                location: DataLocation {
                    path: data_filename,
                    offset: 0,
                    length: 36,
                },
                expected: Err("file too short".into()),
            },
        ];

        cases.test_each(|case| {
            let loader = FileLoader::new(model_file.path()).unwrap();
            let data = loader.load(&case.location).map_err(|e| e.to_string());
            match (&data, &case.expected) {
                (Ok(actual), Ok(expected)) => assert_eq!(actual, expected),
                (Err(actual), Err(expected)) => assert!(
                    actual.contains(expected),
                    "{} does not contain {}",
                    actual,
                    expected
                ),
                (actual, expected) => assert_eq!(actual.is_ok(), expected.is_ok()),
            }
        });
    }
}
