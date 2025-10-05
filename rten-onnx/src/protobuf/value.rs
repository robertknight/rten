//! Traits and types for reading primitive values in Protocol Buffers messages.

use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom};

use crate::protobuf::errors::{ErrorKind, ProtobufError};
use crate::protobuf::varint::read_varint;

/// Specifies the types produced when reading certain types of message field.
///
/// This determines whether the reader produces owned types (`String`, `Vec`),
/// slices (`&str`, `&[u8]`) or something else.
///
/// See [`ReadValue::Types`].
pub trait FieldTypes {
    /// Type produced when reading string fields.
    type String: AsRef<str>;

    /// Type produced when reading bytes fields.
    type Bytes: AsRef<[u8]>;
}

/// Implementation of [`FieldTypes`] that uses owned types.
pub struct OwnedValues;

impl FieldTypes for OwnedValues {
    type String = String;
    type Bytes = Vec<u8>;
}

/// Trait for reading primitive values from a Protocol Buffers message.
///
/// The basic building block of Protocol Buffers messages are primitives that
/// can be one of 4 types: variable-length integer (varints), 32-bit value,
/// 64-bit value or a variable-length sequence of bytes. Strings are represented
/// as sequences of UTF-8 bytes and embedded messages are also sequences of bytes.
///
/// This trait provides methods to read these primitives, or skip over
/// variable-sized fields. It also provides a [`position`](ReadValue::position)
/// method which allows the caller to determine when the end of an embedded
/// message has been reached. The types used to represent variable-length fields
/// are specified via the `Types` associated type.
pub trait ReadValue {
    /// Collection of types used for fields that are read.
    type Types: FieldTypes;

    /// Read a 4-byte little-endian value as an i32.
    fn read_i32(&mut self) -> Result<i32, ProtobufError>;

    /// Read a 4-byte little-endian value as an i64.
    fn read_i64(&mut self) -> Result<i64, ProtobufError>;

    /// Read an LEB128-encoded varint.
    ///
    /// See <https://protobuf.dev/programming-guides/encoding/#varints> for
    /// details.
    ///
    /// Implementations will usually delegate to
    /// [`read_varint](crate::protobuf::varint::read_varint).
    fn read_varint(&mut self) -> Result<u64, ProtobufError>;

    /// Read the value of a `bytes` field.
    fn read_bytes(
        &mut self,
        len: usize,
    ) -> Result<<Self::Types as FieldTypes>::Bytes, ProtobufError>;

    /// Read a string encoded as `len` bytes of UTF-8.
    fn read_string(
        &mut self,
        len: usize,
    ) -> Result<<Self::Types as FieldTypes>::String, ProtobufError>;

    /// Skip over `len` bytes.
    fn skip(&mut self, len: usize) -> Result<(), ProtobufError>;

    /// Return the current position of the reader.
    fn position(&self) -> u64;
}

/// A Protocol Buffers primitive reader that returns owned values.
///
/// This implements the [`ReadValue`] trait and returns `Vec` when reading a
/// bytes field, `String` for strings fields and so on.
#[derive(Default)]
pub struct ValueReader<R> {
    inner: R,
}

impl<R: BufRead + Seek + Position> ValueReader<R> {
    /// Create a value reader from an underlying file or buffer.
    ///
    /// See [`from_buf`](Self::from_buf) and [`from_file`](Self::from_file)
    /// for convenient wrappers for this which create readers from byte buffers
    /// and files.
    pub fn new(inner: R) -> Self {
        Self { inner }
    }
}

impl<T: AsRef<[u8]>> ValueReader<Cursor<T>> {
    /// Convenience method that creates a reader from a byte buffer.
    pub fn from_buf(buf: T) -> Self {
        Self::new(Cursor::new(buf))
    }
}

impl ValueReader<ReadPos<BufReader<File>>> {
    /// Convenience method that creates a reader from a file.
    pub fn from_file(file: File) -> Self {
        Self::new(ReadPos::new(BufReader::new(file)))
    }
}

impl<R: BufRead + Seek + Position> ReadValue for ValueReader<R> {
    type Types = OwnedValues;

    fn read_i32(&mut self) -> Result<i32, ProtobufError> {
        let mut buf = [0; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, ProtobufError> {
        let mut buf = [0; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_varint(&mut self) -> Result<u64, ProtobufError> {
        let value = read_varint(&mut self.inner)?;
        Ok(value)
    }

    fn read_bytes(
        &mut self,
        len: usize,
    ) -> Result<<Self::Types as FieldTypes>::Bytes, ProtobufError> {
        let mut buf = vec![0; len];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn read_string(
        &mut self,
        len: usize,
    ) -> Result<<Self::Types as FieldTypes>::String, ProtobufError> {
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes).map_err(|_| ProtobufError::new(ErrorKind::InvalidUtf8))
    }

    fn skip(&mut self, len: usize) -> Result<(), ProtobufError> {
        self.inner.seek_relative(len as i64)?;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.inner.position()
    }
}

/// Trait for readers that can report their current position cheaply.
///
/// This can be implemented for any reader using [`ReadPos`].
pub trait Position {
    fn position(&self) -> u64;
}

impl<T> Position for Cursor<T> {
    fn position(&self) -> u64 {
        Cursor::position(self)
    }
}

/// Reader adapter that tracks the current read position.
pub struct ReadPos<R: Read> {
    inner: R,
    pos: u64,
}

impl<R: Read + Seek> ReadPos<R> {
    /// Create a new `ReadPos` that wraps a reader.
    pub fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    /// Return the wrapped reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> Read for ReadPos<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n_read = self.inner.read(buf)?;
        self.pos += n_read as u64;
        Ok(n_read)
    }
}

impl<R: Read + Seek> Seek for ReadPos<R> {
    fn seek(&mut self, seek: SeekFrom) -> std::io::Result<u64> {
        match seek {
            SeekFrom::Current(offset) => {
                self.inner.seek_relative(offset)?;
                self.pos = (self.pos as i64 + offset) as u64;
                Ok(self.pos)
            }
            SeekFrom::Start(_) | SeekFrom::End(_) => {
                self.inner.seek(seek)?;
                self.pos = self.inner.stream_position()?;
                Ok(self.pos)
            }
        }
    }

    // It is important to implement `seek_relative` here in order to delegate
    // to `seek_relative` on the underlying reader. This is much more efficient
    // for `BufReader`.
    fn seek_relative(&mut self, offset: i64) -> std::io::Result<()> {
        self.inner.seek_relative(offset)?;
        self.pos = (self.pos as i64 + offset) as u64;
        Ok(())
    }
}

impl<R: BufRead> BufRead for ReadPos<R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        self.inner.consume(amount);
        self.pos += amount as u64;
    }
}

impl<R: BufRead> Position for ReadPos<R> {
    fn position(&self) -> u64 {
        self.pos
    }
}

/// Wraps a [`ReadValue`] to limit the maximum offset in the stream that can
/// be read up to.
///
/// This is used for reading embedded messages for example.
pub(crate) struct LimitReader<'a, R: ReadValue> {
    inner: &'a mut R,
    end: u64,
}

impl<'a, R: ReadValue> LimitReader<'a, R> {
    /// Create a reader which reads up to `len` bytes of `inner`.
    pub fn new(inner: &'a mut R, len: u64) -> Self {
        Self {
            end: inner.position() + len,
            inner,
        }
    }

    /// Create a sub-reader which reads up to `len` bytes of this reader.
    pub fn sub_limit(&mut self, len: u64) -> LimitReader<'_, R> {
        LimitReader {
            end: self.inner.position() + len,
            inner: self.inner,
        }
    }

    fn check_has_bytes(&self, len: usize) -> Result<(), ProtobufError> {
        if self.position() + (len as u64) <= self.end {
            Ok(())
        } else {
            Err(ProtobufError::new(ErrorKind::Eof))
        }
    }
}

impl<'a, R: ReadValue> ReadValue for LimitReader<'a, R> {
    type Types = R::Types;

    fn read_i32(&mut self) -> Result<i32, ProtobufError> {
        self.check_has_bytes(4)?;
        let val = self.inner.read_i32()?;
        Ok(val)
    }

    fn read_i64(&mut self) -> Result<i64, ProtobufError> {
        self.check_has_bytes(8)?;
        let val = self.inner.read_i64()?;
        Ok(val)
    }

    fn read_varint(&mut self) -> Result<u64, ProtobufError> {
        // Varints are at least 1 byte long, and can be up to 10.
        self.check_has_bytes(1)?;
        self.inner.read_varint()
    }

    fn read_bytes(
        &mut self,
        len: usize,
    ) -> Result<<Self::Types as FieldTypes>::Bytes, ProtobufError> {
        self.check_has_bytes(len)?;
        let bytes = self.inner.read_bytes(len)?;
        Ok(bytes)
    }

    fn read_string(
        &mut self,
        len: usize,
    ) -> Result<<Self::Types as FieldTypes>::String, ProtobufError> {
        self.check_has_bytes(len)?;
        let string = self.inner.read_string(len)?;
        Ok(string)
    }

    fn skip(&mut self, len: usize) -> Result<(), ProtobufError> {
        self.check_has_bytes(len)?;
        self.inner.skip(len)?;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.inner.position()
    }
}

#[cfg(test)]
mod tests {
    use super::{LimitReader, ReadValue, ValueReader};
    use crate::protobuf::ErrorKind;
    use crate::protobuf::varint::encode_varint;

    fn test_read_value<R: ReadValue>(make_reader: impl Fn(Vec<u8>) -> R) {
        let mut buf = Vec::new();
        buf.extend((42i32).to_le_bytes());
        buf.extend((84i64).to_le_bytes());
        buf.extend(encode_varint(1234));
        buf.extend([1, 2, 3, 4]);
        buf.extend("hello world".as_bytes());

        // Read each value.
        let mut reader = make_reader(buf.clone());
        assert_eq!(reader.position(), 0);

        assert_eq!(reader.read_i32().unwrap(), 42);
        assert_eq!(reader.position(), 4);

        assert_eq!(reader.read_i64().unwrap(), 84);
        assert_eq!(reader.position(), 12);

        assert_eq!(reader.read_varint().unwrap(), 1234);
        assert_eq!(reader.position(), 14);

        assert_eq!(reader.read_bytes(4).unwrap().as_ref(), [1, 2, 3, 4]);
        assert_eq!(reader.position(), 18);

        assert_eq!(reader.read_string(11).unwrap().as_ref(), "hello world");
        assert_eq!(reader.position(), 29);

        // Read buffer again, but skip over some values.
        let mut reader = make_reader(buf);
        assert_eq!(reader.read_i32().unwrap(), 42);
        assert_eq!(reader.read_i64().unwrap(), 84);
        assert_eq!(reader.read_varint().unwrap(), 1234);
        reader.skip(4).unwrap();
        assert_eq!(reader.read_string(11).unwrap().as_ref(), "hello world");
    }

    #[test]
    fn test_value_reader() {
        test_read_value(|buf| ValueReader::from_buf(buf));
    }

    #[test]
    fn test_limit_reader() {
        let mut buf = Vec::new();
        buf.extend((42i32).to_le_bytes());
        buf.extend((84i64).to_le_bytes());
        buf.extend(encode_varint(1234));
        buf.extend([1, 2, 3, 4]);
        buf.extend("hello world".as_bytes());

        let limit = buf.len();
        buf.extend(encode_varint(5678));

        let mut reader = ValueReader::from_buf(buf);

        // Create a sub-reader which only reads up to `limit`.
        let mut lr = LimitReader::new(&mut reader, limit as u64);
        assert_eq!(lr.read_i32().unwrap(), 42);
        assert_eq!(lr.read_i64().unwrap(), 84);
        assert_eq!(lr.read_varint().unwrap(), 1234);
        assert_eq!(lr.read_bytes(4).unwrap(), [1, 2, 3, 4]);
        assert_eq!(lr.read_string(11).unwrap(), "hello world");

        let eof = lr.read_varint().err().unwrap();
        assert!(matches!(eof.kind(), ErrorKind::Eof));
    }
}
