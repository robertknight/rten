use std::ffi::OsStr;
use std::path::Path;

use rten_base::num::AsUsize;

/// File type of a machine learning model.
#[derive(Debug, PartialEq)]
pub enum FileType {
    /// RTen file format.
    ///
    /// See `docs/rten-file-format.md` in this repository.
    Rten,
    /// ONNX file format.
    Onnx,
}

impl FileType {
    /// Return the file type that corresponds to the extension of `path`.
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension().unwrap_or_default();

        if ext.eq_ignore_ascii_case(OsStr::new("rten")) {
            Some(FileType::Rten)
        } else if ext.eq_ignore_ascii_case(OsStr::new("onnx")) {
            Some(FileType::Onnx)
        } else {
            None
        }
    }

    /// Infer file type from the content of a file.
    pub fn from_buffer(data: &[u8]) -> Option<Self> {
        let magic: Option<[u8; 4]> = data.get(..4).unwrap_or_default().try_into().ok();

        // The checks here are ordered from most to least reliable.

        // rten files using the v2 format and later start with a 4-byte file
        // type identifier.
        if magic == Some(*b"RTEN") {
            return Some(FileType::Rten);
        }

        #[cfg(feature = "onnx_format")]
        {
            use rten_onnx::onnx::is_onnx_model;
            use rten_onnx::protobuf::ValueReader;

            // ONNX models are serialized Protocol Buffers messages with no file
            // type identifier, so we attempt some lightweight protobuf parsing.
            if is_onnx_model(ValueReader::from_buf(data)) {
                return Some(FileType::Onnx);
            }
        }

        // rten files using the v1 format don't have a file type identifier.
        // They are FlatBuffers messages which start with a u32 offset pointing
        // to the root table, as described at
        // https://flatbuffers.dev/internals/#encoding-example.
        if let Some(root_offset) = magic.map(u32::from_le_bytes)
            && data.len() >= root_offset.as_usize()
        {
            return Some(FileType::Rten);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;
    use std::path::Path;

    use super::FileType;

    #[test]
    fn test_file_type_from_path() {
        #[derive(Debug)]
        struct Case<'a> {
            path: &'a Path,
            file_type: Option<FileType>,
        }

        let cases = [
            Case {
                path: Path::new("foo.rten"),
                file_type: Some(FileType::Rten),
            },
            Case {
                path: Path::new("foo.onnx"),
                file_type: Some(FileType::Onnx),
            },
            Case {
                path: Path::new("foo.md"),
                file_type: None,
            },
            Case {
                path: Path::new("foo.ONNX"),
                file_type: Some(FileType::Onnx),
            },
            Case {
                path: Path::new("foo.RTeN"),
                file_type: Some(FileType::Rten),
            },
        ];

        cases.test_each(|case| {
            assert_eq!(FileType::from_path(case.path), case.file_type);
        });
    }

    #[test]
    fn test_file_type_from_buffer() {
        #[derive(Debug)]
        struct Case {
            buf: Vec<u8>,
            expected: Option<FileType>,
        }

        let cases = [
            Case {
                buf: b"RTEN".into(),
                expected: Some(FileType::Rten),
            },
            Case {
                buf: b"".into(),
                expected: None,
            },
            Case {
                buf: {
                    (128u32)
                        .to_le_bytes()
                        .into_iter()
                        .chain(std::iter::repeat(0).take(128))
                        .collect()
                },
                expected: Some(FileType::Rten),
            },
            Case {
                buf: b"unknown format".into(),
                expected: None,
            },
        ];

        cases.test_each(|case| {
            assert_eq!(FileType::from_buffer(&case.buf), case.expected);
        });
    }
}
