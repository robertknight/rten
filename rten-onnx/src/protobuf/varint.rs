//! Read varints from buffered streams.
//!
//! Variable length integers (_varints_) are the default encoding of integers
//! in Protocol Buffers messages, including field tags and numbers.
//!
//! See <https://protobuf.dev/programming-guides/encoding/#varints>.

use std::io::BufRead;

/// Maximum number of bytes for an encoded varint.
///
/// A decoded varint is a u64 value. Each byte contains 7 value bits and one
/// continuation bit. Hence we need 9 "full" bytes plus one bit from the 10th byte.
const MAX_VARINT_LEN: usize = 10;

#[derive(Debug)]
pub enum VarintError {
    /// The input reader is already at the end of stream.
    Eof,
    /// The varint length exceeds 64-bits, or the end of the stream was reached
    /// before a full varint was read.
    InvalidVarint,
    /// An IO error was encountered while reading from the input reader.
    IoError(std::io::Error),
}

impl From<std::io::Error> for VarintError {
    fn from(val: std::io::Error) -> Self {
        Self::IoError(val)
    }
}

/// Read a varint value of up to 64-bits.
///
/// This will read between one and ten bytes from `src`.
pub fn read_varint<R: BufRead>(mut src: R) -> Result<u64, VarintError> {
    let mut index = 0;
    let mut value = 0;

    'outer: loop {
        let buf = src.fill_buf()?;
        if buf.is_empty() {
            return Err(VarintError::Eof);
        }

        let buf_len = buf.len().min(MAX_VARINT_LEN - index);
        let buf = &buf[..buf_len];

        for (i, byte) in buf.iter().copied().enumerate() {
            // High bit is continuation bit. Low 7 bits are the payload.
            value |= ((byte & 0x7f) as u64) << (index * 7);
            if byte <= 0x7f {
                // Only one value bit from the last byte may be used.
                if index + 1 == MAX_VARINT_LEN && byte > 0x01 {
                    break 'outer;
                }
                src.consume(i + 1);
                return Ok(value);
            }
            index += 1;
        }

        src.consume(buf_len);
        if index > MAX_VARINT_LEN {
            break;
        }
    }

    Err(VarintError::InvalidVarint)
}

#[cfg(test)]
pub fn encode_varint(mut val: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(10);

    loop {
        let mut byte = (val & 0x7f) as u8;
        if val <= 0x7f {
            bytes.push(byte);
            break;
        } else {
            byte |= 0x80;
            bytes.push(byte);
            val = val >> 7;
        }
    }

    bytes
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, Cursor, Read};

    use super::{VarintError, encode_varint, read_varint};

    /// Like `Cursor`, but behaves as if the internal buffer only has a capacity
    /// of one.
    struct OneByteCursor<'a> {
        buf: &'a [u8],
    }

    impl<'a> OneByteCursor<'a> {
        fn new(buf: &'a [u8]) -> Self {
            Self { buf }
        }
    }

    impl<'a> Read for OneByteCursor<'a> {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            // Read is a supertrait of `BufRead`, but we don't expect
            // `read_varint` to use it.
            unimplemented!("unexpected call")
        }
    }

    impl<'a> BufRead for OneByteCursor<'a> {
        fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
            let len = self.buf.len().min(1);
            Ok(&self.buf[..len])
        }

        fn consume(&mut self, len: usize) {
            assert!(len <= 1);
            (_, self.buf) = self.buf.split_at(1);
        }
    }

    #[test]
    fn test_read_varint() {
        let mut values: Vec<u64> = (0..1024).collect();
        values.push(u64::MAX);
        for val in values {
            let buf = encode_varint(val);
            let mut cur = Cursor::new(buf);
            let decoded_val = read_varint(&mut cur).unwrap();
            assert_eq!(decoded_val, val);
        }
    }

    #[test]
    fn test_read_varint_sequence() {
        // Example from https://protobuf.dev/programming-guides/encoding/#simple.
        let buf = vec![0x08, 0x96, 0x01];
        let mut cur = Cursor::new(buf);
        let val = read_varint(&mut cur).unwrap();
        assert_eq!(val, 8);

        let val = read_varint(&mut cur).unwrap();
        assert_eq!(val, 150);

        let val = read_varint(&mut cur);
        assert!(matches!(val, Err(VarintError::Eof)));

        // Longer sequence of varints.
        let values = [0, 1, 150, u64::MAX, 2];
        let buf: Vec<u8> = values.iter().copied().flat_map(encode_varint).collect();

        let mut decoded_values = Vec::new();
        let mut cur = Cursor::new(buf);
        loop {
            match read_varint(&mut cur) {
                Ok(val) => decoded_values.push(val),
                Err(VarintError::Eof) => break,
                Err(_) => {
                    panic!("decode failed")
                }
            }
        }

        assert_eq!(decoded_values, values);
    }

    // Test the case where the reader's internal buffer runs out and needs to
    // be refilled before we reach the end of the varint.
    #[test]
    fn test_read_varint_refill() {
        let val = 65535;
        let buf = encode_varint(val);
        let mut cur = OneByteCursor::new(&buf);
        let decoded = read_varint(&mut cur).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_invalid_varint() {
        let mut buf = encode_varint(u64::MAX);
        assert_eq!(buf.len(), 10);
        buf[9] += 1;
        let decoded = read_varint(&mut Cursor::new(buf));
        assert!(matches!(decoded, Err(VarintError::InvalidVarint)));
    }
}
