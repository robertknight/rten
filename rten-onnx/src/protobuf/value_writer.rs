//! Traits and types for writing primitive values in Protocol Buffers messages.

use std::fs::File;
use std::io::{BufWriter, Write};

use crate::protobuf::errors::ProtobufError;
use crate::protobuf::varint::{MAX_VARINT_LEN, encode_varint, varint_len};

/// Trait for writing primitive values to a Protocol Buffers message.
pub trait WriteValue {
    /// True if this writer only counts the bytes that would be written, rather
    /// than writing them.
    ///
    /// Writers of embedded messages use this to avoid recursing into a message
    /// whose length has already been measured.
    const COUNT_ONLY: bool = false;

    /// Write a 4-byte little-endian value.
    fn write_i32(&mut self, val: i32) -> Result<(), ProtobufError>;

    /// Write an 8-byte little-endian value.
    fn write_i64(&mut self, val: i64) -> Result<(), ProtobufError>;

    /// Write an LEB128-encoded varint.
    ///
    /// See <https://protobuf.dev/programming-guides/encoding/#varints>.
    fn write_varint(&mut self, val: u64) -> Result<(), ProtobufError>;

    /// Write the content of a `bytes` or `string` field.
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), ProtobufError>;

    /// Flush any buffered data to the underlying output.
    fn flush(&mut self) -> Result<(), ProtobufError>;

    /// Return the number of bytes written so far.
    fn position(&self) -> u64;
}

impl<W: WriteValue + ?Sized> WriteValue for &mut W {
    const COUNT_ONLY: bool = W::COUNT_ONLY;

    fn write_i32(&mut self, val: i32) -> Result<(), ProtobufError> {
        (**self).write_i32(val)
    }

    fn write_i64(&mut self, val: i64) -> Result<(), ProtobufError> {
        (**self).write_i64(val)
    }

    fn write_varint(&mut self, val: u64) -> Result<(), ProtobufError> {
        (**self).write_varint(val)
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), ProtobufError> {
        (**self).write_bytes(bytes)
    }

    fn flush(&mut self) -> Result<(), ProtobufError> {
        (**self).flush()
    }

    fn position(&self) -> u64 {
        (**self).position()
    }
}

/// A Protocol Buffers primitive writer that writes to an output stream.
pub struct ValueWriter<W> {
    inner: W,
    pos: u64,
}

impl<W: Write> ValueWriter<W> {
    /// Create a value writer which writes to `inner`.
    pub fn new(inner: W) -> Self {
        Self { inner, pos: 0 }
    }

    /// Return the wrapped writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl ValueWriter<BufWriter<File>> {
    /// Convenience method that creates a writer which writes to a file.
    pub fn from_file(file: File) -> Self {
        Self::new(BufWriter::new(file))
    }
}

impl<W: Write> WriteValue for ValueWriter<W> {
    fn write_i32(&mut self, val: i32) -> Result<(), ProtobufError> {
        self.inner.write_all(&val.to_le_bytes())?;
        self.pos += 4;
        Ok(())
    }

    fn write_i64(&mut self, val: i64) -> Result<(), ProtobufError> {
        self.inner.write_all(&val.to_le_bytes())?;
        self.pos += 8;
        Ok(())
    }

    fn write_varint(&mut self, val: u64) -> Result<(), ProtobufError> {
        let mut buf = [0; MAX_VARINT_LEN];
        let encoded = encode_varint(val, &mut buf);
        self.inner.write_all(encoded)?;
        self.pos += encoded.len() as u64;
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), ProtobufError> {
        self.inner.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ProtobufError> {
        self.inner.flush()?;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.pos
    }
}

/// A Protocol Buffers primitive writer that counts bytes instead of writing
/// them.
///
/// This is used to determine the length of a message before writing it.
#[derive(Default)]
pub struct CountingWriter {
    pos: u64,
}

impl CountingWriter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl WriteValue for CountingWriter {
    const COUNT_ONLY: bool = true;

    fn write_i32(&mut self, _val: i32) -> Result<(), ProtobufError> {
        self.pos += 4;
        Ok(())
    }

    fn write_i64(&mut self, _val: i64) -> Result<(), ProtobufError> {
        self.pos += 8;
        Ok(())
    }

    fn write_varint(&mut self, val: u64) -> Result<(), ProtobufError> {
        self.pos += varint_len(val) as u64;
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), ProtobufError> {
        self.pos += bytes.len() as u64;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ProtobufError> {
        Ok(())
    }

    fn position(&self) -> u64 {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::{CountingWriter, ValueWriter, WriteValue};
    use crate::protobuf::varint::encode_varint_vec;
    use crate::protobuf::{ReadValue, ValueReader};

    fn write_values<W: WriteValue>(writer: &mut W) {
        assert_eq!(writer.position(), 0);

        writer.write_i32(42).unwrap();
        assert_eq!(writer.position(), 4);

        writer.write_i64(84).unwrap();
        assert_eq!(writer.position(), 12);

        writer.write_varint(1234).unwrap();
        assert_eq!(writer.position(), 14);

        writer.write_bytes(&[1, 2, 3, 4]).unwrap();
        assert_eq!(writer.position(), 18);

        writer.write_bytes("hello world".as_bytes()).unwrap();
        assert_eq!(writer.position(), 29);

        writer.flush().unwrap();
    }

    #[test]
    fn test_value_writer() {
        let mut buf = Vec::new();
        let mut writer = ValueWriter::new(&mut buf);
        write_values(&mut writer);

        // Verify expected bytes were written.
        let mut expected = Vec::new();
        expected.extend((42i32).to_le_bytes());
        expected.extend((84i64).to_le_bytes());
        expected.extend(encode_varint_vec(1234));
        expected.extend([1, 2, 3, 4]);
        expected.extend("hello world".as_bytes());
        assert_eq!(buf, expected);

        // Verify that message can be read with `ValueReader`.
        let mut reader = ValueReader::from_buf(buf);
        assert_eq!(reader.read_i32().unwrap(), 42);
        assert_eq!(reader.read_i64().unwrap(), 84);
        assert_eq!(reader.read_varint().unwrap(), 1234);
        assert_eq!(reader.read_bytes(4).unwrap(), [1, 2, 3, 4]);
        assert_eq!(reader.read_string(11).unwrap(), "hello world");
    }

    #[test]
    fn test_counting_writer() {
        let mut writer = CountingWriter::new();
        write_values(&mut writer);
        assert_eq!(writer.position(), 29);
    }
}
