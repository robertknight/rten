use rten_tensor::{AsView, Layout, Tensor, TensorView};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

mod dtype;
use dtype::{Element, ElementKind};

use crate::value::{DataType, Value, View, dispatch_data_type, match_view};

/// Magic bytes at the start of every `.npy` file.
const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Alignment of the header in bytes. The magic string, version, header length
/// and header dictionary are padded to a multiple of this.
const HEADER_ALIGN: usize = 64;

fn invalid_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Serialize a tensor to a writer.
pub fn write<'a, V: Into<View<'a>>>(writer: impl io::Write, array: V) -> io::Result<()> {
    let view = array.into();
    match_view!(view, array => write_typed(writer, array))
}

/// Serialize a tensor to a file.
pub fn write_to_file<'a, V: Into<View<'a>>>(path: impl AsRef<Path>, array: V) -> io::Result<()> {
    write(File::create(path)?, array)
}

fn write_typed<T: Element>(writer: impl io::Write, array: TensorView<T>) -> io::Result<()> {
    let mut writer = io::BufWriter::new(writer);

    let header = build_header::<T>(array.shape())?;
    writer.write_all(&header)?;

    for x in array.iter() {
        writer.write_all(x.to_le_bytes().as_ref())?;
    }
    writer.flush()
}

/// Read a tensor from a reader.
///
/// The returned [`Value`] holds a tensor whose element type matches the dtype
/// stored in the file. Use [`Value::into_type`] / [`Value::as_type`] to extract
/// a typed tensor.
pub fn read(mut reader: impl io::Read) -> io::Result<Value> {
    let header = read_header(&mut reader)?;
    let data_type = data_type_from_dtype(&header.dtype).ok_or_else(|| {
        invalid_data(format!(
            "unsupported npy dtype `{}{}`",
            header.dtype.kind.as_char(),
            header.dtype.item_size
        ))
    })?;
    let value = dispatch_data_type!(data_type, T => {
        Value::from(read_typed::<T>(&header, reader)?)
    });
    Ok(value)
}

fn read_typed<T: Element>(header: &Header, mut reader: impl io::Read) -> io::Result<Tensor<T>> {
    let n_elements = header
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| invalid_data("array element count overflows"))?;
    let n_bytes = n_elements
        .checked_mul(T::ITEM_SIZE)
        .ok_or_else(|| invalid_data("array size in bytes overflows"))?;

    // Reject implausibly large arrays so a corrupt or malicious header cannot
    // request a huge allocation.
    if n_bytes > u32::MAX as usize {
        return Err(invalid_data("array is too large"));
    }
    let mut data = Vec::with_capacity(n_bytes);
    reader
        .by_ref()
        .take(n_bytes as u64)
        .read_to_end(&mut data)?;
    if data.len() != n_bytes {
        return Err(invalid_data("array data is truncated"));
    }

    // FIXME: This copies every element even when the data is already in the
    // requested little-endian, C-contiguous layout and could be used directly.
    let swap_bytes = header.dtype.big_endian && T::ITEM_SIZE > 1;
    let values: Vec<T> = data
        .chunks_exact(T::ITEM_SIZE)
        .map(|chunk| {
            let mut bytes = T::Bytes::try_from(chunk)
                .unwrap_or_else(|_| unreachable!("chunk is T::ITEM_SIZE bytes"));
            if swap_bytes {
                bytes.as_mut().reverse();
            }
            T::from_le_bytes(bytes)
        })
        .collect();

    let values = if header.fortran_order {
        fortran_order_to_row_major(values, &header.shape)
    } else {
        values
    };

    Ok(Tensor::from_data(&header.shape, values))
}

/// Read a tensor from a file.
pub fn read_from_file(path: impl AsRef<Path>) -> io::Result<Value> {
    let file = io::BufReader::new(File::open(path)?);
    read(file)
}

/// Map a parsed npy dtype to the corresponding [`DataType`], or `None` if it is
/// not a supported scalar type.
fn data_type_from_dtype(dtype: &DType) -> Option<DataType> {
    let data_type = match (dtype.kind, dtype.item_size) {
        (ElementKind::Bool, 1) => DataType::Bool,
        (ElementKind::Int, 1) => DataType::Int8,
        (ElementKind::Int, 2) => DataType::Int16,
        (ElementKind::Int, 4) => DataType::Int32,
        (ElementKind::Int, 8) => DataType::Int64,
        (ElementKind::Uint, 1) => DataType::UInt8,
        (ElementKind::Uint, 2) => DataType::UInt16,
        (ElementKind::Uint, 4) => DataType::UInt32,
        (ElementKind::Uint, 8) => DataType::UInt64,
        (ElementKind::Float, 4) => DataType::Float32,
        (ElementKind::Float, 8) => DataType::Float64,
        _ => return None,
    };
    Some(data_type)
}

fn fortran_order_to_row_major<T: Clone>(values: Vec<T>, shape: &[usize]) -> Vec<T> {
    if shape.len() < 2 {
        return values;
    }

    let reversed_shape = shape.iter().copied().rev().collect::<Vec<_>>();
    let reverse_axes = (0..shape.len()).rev().collect::<Vec<_>>();

    Tensor::from_vec(values)
        .reshaped(reversed_shape.as_slice())
        .permuted(&reverse_axes)
        .to_vec()
}

/// Parsed `descr` field of a `.npy` header.
struct DType {
    /// Whether multi-byte elements are stored big-endian.
    big_endian: bool,
    kind: ElementKind,
    item_size: usize,
}

/// Parsed `.npy` header.
struct Header {
    dtype: DType,
    fortran_order: bool,
    shape: Vec<usize>,
}

/// Read and parse the header of a `.npy` file, leaving `reader` positioned at
/// the start of the array data.
fn read_header(mut reader: impl io::Read) -> io::Result<Header> {
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid_data("not an npy file"));
    }

    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;

    // The header length is a u16 in format version 1 and a u32 in versions 2
    // and 3. Versions differ only in the header encoding, which does not affect
    // the fields this crate reads.
    let header_len = match version[0] {
        1 => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            u16::from_le_bytes(buf) as usize
        }
        2 | 3 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        }
        major => {
            return Err(invalid_data(format!("unsupported npy version {major}")));
        }
    };

    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let header =
        std::str::from_utf8(&header).map_err(|_| invalid_data("npy header is not valid UTF-8"))?;

    parse_header(header)
}

/// Parse the Python dictionary literal that makes up a `.npy` header.
///
/// This supports only the small subset of Python syntax that NumPy emits: a
/// dictionary mapping single-quoted string keys to a single-quoted string
/// (`descr`), a boolean (`fortran_order`) or a tuple of integers (`shape`).
fn parse_header(header: &str) -> io::Result<Header> {
    let mut parser = HeaderParser::new(header);
    let mut descr = None;
    let mut fortran_order = None;
    let mut shape = None;

    // NumPy tolerates leading whitespace before the dictionary when loading, so
    // accept it here too.
    parser.skip_whitespace();
    parser.expect(b'{')?;
    loop {
        parser.skip_whitespace();
        if parser.consume(b'}') {
            break;
        }

        let key = parser.parse_string()?;
        parser.skip_whitespace();
        parser.expect(b':')?;
        parser.skip_whitespace();

        match key.as_str() {
            "descr" => descr = Some(parse_descr(&parser.parse_string()?)?),
            "fortran_order" => fortran_order = Some(parser.parse_bool()?),
            "shape" => shape = Some(parser.parse_shape()?),
            _ => return Err(invalid_data(format!("unexpected npy header key `{key}`"))),
        }

        parser.skip_whitespace();
        // A comma separates entries and NumPy also writes a trailing comma
        // before the closing brace.
        parser.consume(b',');
    }

    Ok(Header {
        dtype: descr.ok_or_else(|| invalid_data("npy header is missing `descr`"))?,
        fortran_order: fortran_order
            .ok_or_else(|| invalid_data("npy header is missing `fortran_order`"))?,
        shape: shape.ok_or_else(|| invalid_data("npy header is missing `shape`"))?,
    })
}

/// Parse a NumPy dtype string such as `<i4` or `|u1`.
fn parse_descr(descr: &str) -> io::Result<DType> {
    let invalid = || invalid_data(format!("invalid npy dtype `{descr}`"));

    let mut chars = descr.chars();
    let big_endian = match chars.next() {
        Some('>') => true,
        // `=` is native byte order.
        Some('=') => cfg!(target_endian = "big"),
        // `|` ("not applicable") is used for single-byte types.
        Some('<' | '|') => false,
        _ => return Err(invalid()),
    };
    let kind = chars
        .next()
        .and_then(ElementKind::from_char)
        .ok_or_else(invalid)?;
    // The order and kind characters are both ASCII, so byte index 2 is a
    // character boundary and the remainder is the item size in bytes.
    let item_size = descr[2..].parse::<usize>().map_err(|_| invalid())?;

    Ok(DType {
        big_endian,
        kind,
        item_size,
    })
}

/// Parser for the contents of a `.npy` header.
struct HeaderParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> HeaderParser<'a> {
    fn new(s: &'a str) -> Self {
        HeaderParser {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b) if b.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    /// Consume the next byte if it equals `byte`, returning whether it did.
    fn consume(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// Consume the next byte, requiring it to equal `byte`.
    fn expect(&mut self, byte: u8) -> io::Result<()> {
        if self.consume(byte) {
            Ok(())
        } else {
            Err(invalid_data(format!(
                "expected `{}` in npy header",
                byte as char
            )))
        }
    }

    /// Parse a single-quoted string.
    fn parse_string(&mut self) -> io::Result<String> {
        self.expect(b'\'')?;
        let start = self.pos;
        while let Some(byte) = self.peek() {
            if byte == b'\'' {
                let value = std::str::from_utf8(&self.bytes[start..self.pos])
                    .map_err(|_| invalid_data("npy header string is not valid UTF-8"))?
                    .to_string();
                self.pos += 1;
                return Ok(value);
            }
            self.pos += 1;
        }
        Err(invalid_data("unterminated string in npy header"))
    }

    /// Parse a `True` or `False` literal.
    fn parse_bool(&mut self) -> io::Result<bool> {
        if self.bytes[self.pos..].starts_with(b"True") {
            self.pos += 4;
            Ok(true)
        } else if self.bytes[self.pos..].starts_with(b"False") {
            self.pos += 5;
            Ok(false)
        } else {
            Err(invalid_data("expected `True` or `False` in npy header"))
        }
    }

    /// Parse a tuple of non-negative integers, e.g. `(2, 3)`, `(5,)` or `()`.
    fn parse_shape(&mut self) -> io::Result<Vec<usize>> {
        self.expect(b'(')?;
        let mut shape = Vec::new();
        loop {
            self.skip_whitespace();
            if self.consume(b')') {
                break;
            }
            shape.push(self.parse_usize()?);
            self.skip_whitespace();
            self.consume(b',');
        }
        Ok(shape)
    }

    fn parse_usize(&mut self) -> io::Result<usize> {
        let start = self.pos;
        while matches!(self.peek(), Some(b) if b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(invalid_data("expected integer in npy header"));
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .unwrap()
            .parse::<usize>()
            .map_err(|_| invalid_data("integer in npy header is out of range"))
    }
}

/// Build the magic string, version, length and header dictionary for a tensor
/// with the given element type and shape.
fn build_header<T: Element>(shape: &[usize]) -> io::Result<Vec<u8>> {
    let mut dims = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    // A 1-tuple requires a trailing comma to distinguish it from a parenthesized
    // expression.
    if shape.len() == 1 {
        dims.push(',');
    }
    let mut dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({dims}), }}",
        T::DESCR
    );

    // Pad with spaces and a trailing newline so the total header length is a
    // multiple of `HEADER_ALIGN`. Version 1 uses a 2-byte length field.
    let prefix_len = MAGIC.len() + 2 + 2;
    let unpadded_len = prefix_len + dict.len() + 1;
    let padding = unpadded_len.next_multiple_of(HEADER_ALIGN) - unpadded_len;
    dict.reserve(padding + 1);
    dict.extend(std::iter::repeat_n(' ', padding));
    dict.push('\n');

    // The version 1 length field is a `u16`. NumPy switches to version 2 for
    // larger headers, but a header this size requires a tensor with thousands
    // of dimensions, so reject it rather than emit a truncated length.
    let dict_len =
        u16::try_from(dict.len()).map_err(|_| invalid_data("npy header is too large to encode"))?;

    let mut header = Vec::with_capacity(prefix_len + dict.len());
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&[1, 0]); // Format version 1.0.
    header.extend_from_slice(&dict_len.to_le_bytes());
    header.extend_from_slice(dict.as_bytes());
    Ok(header)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_TEST_FILE: AtomicUsize = AtomicUsize::new(0);

    fn temp_file(name: &str) -> std::path::PathBuf {
        let id = NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("rten-npy-{}-{id}-{name}", std::process::id()))
    }

    /// Verifies that data written to an in-memory NPY stream can be read back
    /// with the same shape and values.
    #[test]
    fn test_round_trip_npy_in_memory() {
        let tensor: Tensor<i32> = [[1, 2, 3], [4, 5, 6]].into();
        let mut buffer = Vec::new();

        write(&mut buffer, tensor.view()).unwrap();
        let read = read(&buffer[..]).unwrap().into_type::<i32>().unwrap();

        assert_eq!(read, tensor);
    }

    /// Round-trip every supported element type through the in-memory writer
    /// and reader.
    #[test]
    fn test_round_trip_npy_dtypes() {
        macro_rules! check {
            ($ty:ty, $tensor:expr) => {{
                let tensor: Tensor<$ty> = $tensor;
                let mut buffer = Vec::new();
                write(&mut buffer, tensor.view()).unwrap();
                let read = read(&buffer[..]).unwrap().into_type::<$ty>().unwrap();
                assert_eq!(&read, &tensor);
            }};
        }

        check!(i8, [1i8, -2, 3].into());
        check!(i16, [1i16, -2, 3].into());
        check!(i32, [1i32, -2, 3].into());
        check!(i64, [1i64, -2, 3].into());
        check!(u8, [1u8, 2, 3].into());
        check!(u16, [1u16, 2, 3].into());
        check!(u32, [1u32, 2, 3].into());
        check!(u64, [1u64, 2, 3].into());
        check!(f32, [1.5f32, -2.5, 3.5].into());
        check!(f64, [1.5f64, -2.5, 3.5].into());
        check!(bool, [true, false, true].into());
    }

    /// Exercises the file convenience helpers by writing and then reading a
    /// tensor from a real file path.
    #[test]
    fn test_round_trip_npy_file_helpers() {
        let path = temp_file("round-trip.npy");
        let tensor: Tensor<f32> = [[1., 2.], [3., 4.]].into();

        write_to_file(&path, tensor.view()).unwrap();
        let read = read_from_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(read.into_type::<f32>().unwrap(), tensor);
    }

    /// Round-trips a scalar tensor, whose header has an empty shape
    /// tuple `()`.
    #[test]
    fn test_round_trip_npy_scalar() {
        let tensor = Tensor::from_scalar(42i32);
        let mut buffer = Vec::new();

        write(&mut buffer, tensor.view()).unwrap();
        let read = read(&buffer[..]).unwrap().into_type::<i32>().unwrap();

        assert_eq!(&read, &tensor);
    }

    /// Confirms malformed input is surfaced as an I/O error instead of being
    /// accepted as an empty or partially decoded tensor.
    #[test]
    fn test_read_npy_rejects_invalid_data() {
        let err = read(&b"not an npy file"[..]).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_read_npy_reports_stored_dtype() {
        let tensor: Tensor<i32> = [1, 2, 3].into();
        let mut buffer = Vec::new();
        write(&mut buffer, tensor.view()).unwrap();

        let value = read(&buffer[..]).unwrap();

        assert_eq!(value.dtype(), DataType::Int32);
    }

    /// Builds a minimal version 1.0 `.npy` file with the given header
    /// dictionary and data.
    fn npy_with_header(dict: &str, data: &[u8]) -> Vec<u8> {
        let mut header = dict.as_bytes().to_vec();
        header.push(b'\n');
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn test_read_npy_rejects_malformed_headers() {
        let cases = [
            // Missing required keys.
            "{'descr': '<i4'}",
            "{'fortran_order': False, 'shape': (1,)}",
            // Malformed dtype strings.
            "{'descr': 'i4', 'fortran_order': False, 'shape': (1,)}",
            "{'descr': '<', 'fortran_order': False, 'shape': (1,)}",
            "{'descr': '<i', 'fortran_order': False, 'shape': (1,)}",
            // Structurally broken dictionaries.
            "{'descr': '<i4', 'fortran_order': False, 'shape': (1,",
            "not a dict",
            "{'descr': '<i4', 'fortran_order': Maybe, 'shape': (1,)}",
        ];

        for case in cases {
            let bytes = npy_with_header(case, &[0, 0, 0, 0]);
            let err = read(&bytes[..]).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidData, "case: {case}");
        }
    }

    #[test]
    fn test_read_npy_accepts_leading_header_whitespace() {
        let bytes = npy_with_header(
            "  \n{'descr': '<i4', 'fortran_order': False, 'shape': (2,)}",
            &[1, 0, 0, 0, 2, 0, 0, 0],
        );

        let read = read(&bytes[..]).unwrap().into_type::<i32>().unwrap();

        assert_eq!(read, Tensor::from([1, 2]));
    }

    #[test]
    fn test_read_npy_big_endian() {
        let bytes = npy_with_header(
            "{'descr': '>i4', 'fortran_order': False, 'shape': (2,)}",
            // 1 and 2 encoded as big-endian i32.
            &[0, 0, 0, 1, 0, 0, 0, 2],
        );

        let read = read(&bytes[..]).unwrap().into_type::<i32>().unwrap();

        assert_eq!(read, Tensor::from([1, 2]));
    }

    /// Covers the rank-2 boundary in Fortran-to-row-major conversion. The
    /// input is the Fortran flat representation of `[[1, 2, 3], [4, 5, 6]]`.
    #[test]
    fn test_fortran_order_to_row_major_converts_rank_2_data() {
        let converted = fortran_order_to_row_major(vec![1, 4, 2, 5, 3, 6], &[2, 3]);

        assert_eq!(converted, [1, 2, 3, 4, 5, 6]);
    }

    /// Checks that equivalent NumPy-generated C-order and Fortran-order files
    /// decode to the same row-major tensor.
    #[test]
    fn test_read_npy_c_and_fortran_order_fixtures_match() {
        let c_order = include_bytes!("../tests/fixtures/order_c_i32.npy");
        let fortran_order = include_bytes!("../tests/fixtures/order_fortran_i32.npy");

        let c_tensor = read(&c_order[..]).unwrap().into_type::<i32>().unwrap();
        let fortran_tensor = read(&fortran_order[..])
            .unwrap()
            .into_type::<i32>()
            .unwrap();
        let expected = Tensor::from([
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ]);

        assert_eq!(c_tensor, expected);
        assert_eq!(fortran_tensor, expected);
        assert_eq!(fortran_tensor, c_tensor);
    }
}
