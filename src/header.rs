use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::number::LeBytes;

/// Read little-endian encoded primitive values from a byte buffer.
struct ValueReader<'a> {
    pos: usize,
    buf: &'a [u8],
}

impl<'a> ValueReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { pos: 0, buf }
    }

    /// Return the next N bytes from the buffer, or None if there aren't enough.
    fn read_n<const N: usize>(&mut self) -> Option<[u8; N]> {
        let chunk = self.buf.get(self.pos..self.pos + N)?;
        self.pos += N;
        Some(chunk.try_into().unwrap())
    }

    /// Read a little-endian encoded value.
    ///
    /// Returns None if there are not enough bytes left in the buffer.
    fn read<T: LeBytes>(&mut self) -> Option<T> {
        let chunk = self
            .buf
            .get(self.pos..self.pos + std::mem::size_of::<T>())?;
        self.pos += chunk.len();

        let chunk_array = chunk.try_into().unwrap();
        Some(T::from_le_bytes(chunk_array))
    }
}

/// Errors produced when reading the header for an RTen model file.
#[derive(Clone, Debug, PartialEq)]
pub enum HeaderError {
    /// The header is incomplete
    TooShort,

    /// The file format version specified in the header is unsupported.
    UnsupportedVersion,

    /// The header doesn't start with the magic bytes "RTEN".
    InvalidMagic,

    /// A segment offset in the header is invalid.
    InvalidOffset,

    /// A segment length in the header is invalid.
    InvalidLength,
}

/// Header for an RTen model file.
///
/// This specifies the file version and offset of the model data and tensor
/// data within the file.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    /// Major version of the file format. Currently 2.
    pub version: u32,

    /// Offset of the FlatBuffers data describing the model.
    pub model_offset: u64,

    /// Length of the FlatBuffers data describing the model.
    pub model_len: u64,

    /// Offset of tensor data stored outside the model.
    pub tensor_data_offset: u64,
}

impl Header {
    /// Size of the serialized header in bytes.
    pub const LEN: usize = 32;

    /// Read the file header from a byte buffer.
    ///
    /// `buf` is expected to be a slice that contains the entire file, as its
    /// length is used to validate offsets in the header.
    pub fn from_buf(buf: &[u8]) -> Result<Header, HeaderError> {
        let too_short = Err(HeaderError::TooShort);

        // This could be passed in separately if we wanted to avoid needing to
        // read or mmap the entire file just to read the header.
        let file_size = buf.len() as u64;

        let mut reader = ValueReader::new(buf);

        let Some(magic) = reader.read_n::<4>() else {
            return too_short;
        };
        if &magic != b"RTEN" {
            return Err(HeaderError::InvalidMagic);
        }

        let Some(version) = reader.read() else {
            return too_short;
        };
        if version != 2 {
            return Err(HeaderError::UnsupportedVersion);
        }

        let Some(model_offset) = reader.read::<u64>() else {
            return too_short;
        };
        if model_offset < Self::LEN as u64 || model_offset > file_size {
            return Err(HeaderError::InvalidOffset);
        }
        let Some(model_len) = reader.read() else {
            return too_short;
        };
        if model_offset.saturating_add(model_len) > file_size {
            return Err(HeaderError::InvalidLength);
        }

        let Some(tensor_data_offset) = reader.read() else {
            return too_short;
        };
        if tensor_data_offset < Self::LEN as u64 || tensor_data_offset > file_size {
            return Err(HeaderError::InvalidOffset);
        }

        Ok(Header {
            version,
            model_offset,
            model_len,
            tensor_data_offset,
        })
    }

    /// Serialize this header to a byte buffer.
    pub fn to_buf(&self) -> Vec<u8> {
        let mut buffer = Vec::new();

        buffer.extend(b"RTEN");
        buffer.extend(self.version.to_le_bytes());
        buffer.extend(self.model_offset.to_le_bytes());
        buffer.extend(self.model_len.to_le_bytes());
        buffer.extend(self.tensor_data_offset.to_le_bytes());

        buffer
    }
}

impl Display for HeaderError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            HeaderError::TooShort => write!(fmt, "header is too short"),
            HeaderError::UnsupportedVersion => write!(fmt, "unsupported file version"),
            HeaderError::InvalidMagic => write!(fmt, "incorrect file magic"),
            HeaderError::InvalidOffset => write!(fmt, "segment offset is invalid"),
            HeaderError::InvalidLength => write!(fmt, "segment length is invalid"),
        }
    }
}

impl Error for HeaderError {}

#[cfg(test)]
mod tests {
    use super::{Header, HeaderError};

    #[test]
    fn test_read_header() {
        let expected_header = Header {
            version: 2,
            // nb. Values must be >= header size and <= length of buffer.
            model_offset: 32,
            model_len: 32,
            tensor_data_offset: 64,
        };

        let mut header_buf = expected_header.to_buf();
        header_buf.extend([0; 32]);
        let header = Header::from_buf(&header_buf).unwrap();

        assert_eq!(header, expected_header);
    }

    #[test]
    fn test_invalid_header() {
        struct Case {
            buf: Vec<u8>,
            expected: HeaderError,
        }

        let cases = [
            Case {
                buf: Vec::new(),
                expected: HeaderError::TooShort,
            },
            Case {
                buf: b"This is some random ASCII text and not a valid header".to_vec(),
                expected: HeaderError::InvalidMagic,
            },
            Case {
                buf: Header {
                    version: 10,
                    model_offset: 0,
                    model_len: 0,
                    tensor_data_offset: 0,
                }
                .to_buf(),
                expected: HeaderError::UnsupportedVersion,
            },
            // Offsets too small.
            Case {
                buf: Header {
                    version: 2,
                    model_offset: 0,
                    model_len: 0,
                    tensor_data_offset: 0,
                }
                .to_buf(),
                expected: HeaderError::InvalidOffset,
            },
            // Offsets exceed buffer size.
            Case {
                buf: Header {
                    version: 2,
                    model_offset: 500,
                    model_len: 0,
                    tensor_data_offset: 500,
                }
                .to_buf(),
                expected: HeaderError::InvalidOffset,
            },
            // Offset + length exceeds buffer size
            Case {
                buf: Header {
                    version: 2,
                    model_offset: 32,
                    model_len: 1024,
                    tensor_data_offset: 0,
                }
                .to_buf(),
                expected: HeaderError::InvalidLength,
            },
        ];

        for Case { buf, expected } in cases {
            let result = Header::from_buf(&buf);
            assert_eq!(result, Err(expected));
        }
    }
}
