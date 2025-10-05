use crate::protobuf::errors::{ErrorKind, ProtobufError};
use crate::protobuf::value::{FieldTypes, LimitReader, ReadValue};

/// Wire-type and associated value of a field.
///
/// See <https://protobuf.dev/programming-guides/encoding/#structure>.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FieldValue {
    /// Integer value encoded as a varint.
    Varint(u64),

    /// 64-bit fixed-width value.
    I64(i64),

    /// A variable-length value with a size specified in bytes.
    Len(u64),

    /// Deprecated start-of-group type.
    Sgroup,

    /// Deprecated end-of-group type.
    Egroup,

    /// 32-bit fixed-width value.
    I32(i32),
}

#[cfg(test)]
impl FieldValue {
    /// Encode a field with the value and wire type of `self` and the given
    /// field number.
    fn encode(self, number: u64) -> Vec<u8> {
        use crate::protobuf::varint::encode_varint;
        let encode_type_number = |wire_type| encode_varint(wire_type | (number << 3));

        let mut buf = Vec::new();
        match self {
            Self::Varint(val) => {
                buf.extend(encode_type_number(0));
                buf.extend(encode_varint(val));
            }
            Self::I64(val) => {
                buf.extend(encode_type_number(1));
                buf.extend(val.to_le_bytes());
            }
            Self::Len(len) => {
                buf.extend(encode_type_number(2));
                buf.extend(encode_varint(len));
            }
            Self::Sgroup => {
                buf.extend(encode_type_number(3));
            }
            Self::Egroup => {
                buf.extend(encode_type_number(4));
            }
            Self::I32(val) => {
                buf.extend(encode_type_number(5));
                buf.extend(val.to_le_bytes());
            }
        }
        buf
    }
}

/// Read a single field of a message.
///
/// `Field`s are produced by iterating over fields of a message using
/// [`Fields::next`].
///
/// Fields have a number and a value. The value can either be a fixed-length
/// type or a variable length value (bytes, string, embedded message, packed
/// repeated field). If the field has a variable length value, then it must be
/// either read (eg. using [`read_bytes`](Self::read_bytes)) or skipped using
/// [`skip`](Self::skip).
///
/// # Repeated fields
///
/// Repeated fields with a primitive type may have either a packed or un-packed
/// representation. The `read_repeated_*` methods return iterators which handle
/// both cases. The returned iterators will yield a single value if the field
/// is unpacked, or all values in a packed block if the field is packed.
pub struct Field<'r, R: ReadValue> {
    reader: LimitReader<'r, R>,
    number: u64,
    value: FieldValue,

    /// Flag indicating whether a variable length field has already been
    /// consumed (ie. read or skipped over).
    consumed: bool,

    /// Debug name of the message type this field belongs to.
    context: Option<&'static str>,

    /// Unconsumed field ID slot in the parent [`Fields`].
    unconsumed_field: &'r mut Option<u64>,
}

impl<'r, R: ReadValue> Field<'r, R> {
    /// Return the field number.
    pub fn number(&self) -> u64 {
        self.number
    }

    /// Return the field value.
    ///
    /// For variable length fields, this value only includes the length. To
    /// actually read the value, one of the `read_` methods needs to be called
    /// on `self`.
    pub fn value(&self) -> FieldValue {
        self.value
    }

    /// Read the bytes in this field.
    pub fn read_bytes(&mut self) -> Result<<R::Types as FieldTypes>::Bytes, ProtobufError> {
        match self.value {
            FieldValue::Len(len) => {
                self.consume_field()?;
                Ok(self.reader.read_bytes(len as usize)?)
            }
            _ => Err(self.error(ErrorKind::FieldTypeMismatch)),
        }
    }

    /// Read the UTF-8 encoded string in this field.
    pub fn read_string(&mut self) -> Result<<R::Types as FieldTypes>::String, ProtobufError> {
        match self.value {
            FieldValue::Len(len) => {
                self.consume_field()?;
                Ok(self.reader.read_string(len as usize)?)
            }
            _ => Err(self.error(ErrorKind::FieldTypeMismatch)),
        }
    }

    /// Begin reading the embedded message in this field.
    ///
    /// The returned [`Fields`] instance must be fully iterated over before
    /// continuing to read fields from `self`. `context` is the name of the
    /// embedded message type being read. It is used to add context to any errors
    /// encountered.
    pub fn read_message(
        &mut self,
        context: Option<&'static str>,
    ) -> Result<Fields<'_, impl ReadValue<Types = R::Types>>, ProtobufError> {
        match self.value {
            FieldValue::Len(len) => {
                self.consume_field()?;
                Ok(Fields {
                    reader: self.reader.sub_limit(len),
                    context,
                    unconsumed_field: None,
                })
            }
            _ => Err(self.error(ErrorKind::FieldTypeMismatch)),
        }
    }

    /// Skip over the contents of this field.
    ///
    /// This should be called for all unknown fields of a message which are not
    /// read, although it currently does nothing if the field has a fixed
    /// length.
    pub fn skip(&mut self) -> Result<(), ProtobufError> {
        if let FieldValue::Len(len) = self.value {
            self.consume_field()?;
            self.reader.skip(len as usize)?;
        }
        Ok(())
    }

    fn get_varint(&self) -> Result<u64, ProtobufError> {
        match self.value {
            FieldValue::Varint(val) => Ok(val),
            _ => Err(self.error(ErrorKind::FieldTypeMismatch)),
        }
    }

    /// Get the value of a field with schema type `int32`.
    pub fn get_int32(&self) -> Result<i32, ProtobufError> {
        self.get_varint().map(|v| v as i32)
    }

    /// Get the value of a field where the schema type is an enum.
    pub fn get_enum(&self) -> Result<i32, ProtobufError> {
        self.get_int32()
    }

    /// Get the value of a field with schema type `int64`.
    pub fn get_int64(&self) -> Result<i64, ProtobufError> {
        self.get_varint().map(|v| v as i64)
    }

    /// Get the value of a field with schema type `float`.
    pub fn get_float(&self) -> Result<f32, ProtobufError> {
        match self.value {
            FieldValue::I32(val) => Ok(f32::from_le_bytes(val.to_le_bytes())),
            _ => Err(self.error(ErrorKind::FieldTypeMismatch)),
        }
    }

    /// Get one or multiple values from a `repeated int32` field.
    pub fn read_repeated_int32(
        &mut self,
    ) -> Result<impl Iterator<Item = Result<i32, ProtobufError>>, ProtobufError> {
        self.read_repeated_varint(|x| x as i32)
    }

    /// Get one or multiple values from a `repeated int64` field.
    pub fn read_repeated_int64(
        &mut self,
    ) -> Result<impl Iterator<Item = Result<i64, ProtobufError>>, ProtobufError> {
        self.read_repeated_varint(|x| x as i64)
    }

    /// Get one or multiple values from a `repeated uint64` field.
    pub fn read_repeated_uint64(
        &mut self,
    ) -> Result<impl Iterator<Item = Result<u64, ProtobufError>>, ProtobufError> {
        self.read_repeated_varint(|x| x)
    }

    /// Get one or multiple values from a `repeated float` field.
    pub fn read_repeated_float(
        &mut self,
    ) -> Result<impl Iterator<Item = Result<f32, ProtobufError>>, ProtobufError> {
        self.read_repeated_fixed32(f32::from_le_bytes)
    }

    /// Get one or multiple values from a `repeated double` field.
    pub fn read_repeated_double(
        &mut self,
    ) -> Result<impl Iterator<Item = Result<f64, ProtobufError>>, ProtobufError> {
        self.read_repeated_fixed64(f64::from_le_bytes)
    }

    /// Get the value of a repeated varint field.
    fn read_repeated_varint<T: Copy>(
        &mut self,
        from_u64: impl Fn(u64) -> T,
    ) -> Result<Repeated<T, impl Iterator<Item = Result<T, ProtobufError>>>, ProtobufError> {
        let repeated = match self.value {
            FieldValue::Varint(val) => Repeated::Unpacked(Some(from_u64(val))),
            FieldValue::Len(len) => {
                let consumed = &mut self.consumed;
                let mut reader = self.reader.sub_limit(len);
                let iter = std::iter::from_fn(move || match reader.read_varint() {
                    Ok(val) => Some(Ok(from_u64(val))),
                    Err(err) if matches!(err.kind(), ErrorKind::Eof) => {
                        *consumed = true;
                        None
                    }
                    Err(err) => Some(Err(err)),
                });
                Repeated::Packed(iter)
            }
            _ => {
                return Err(self.error(ErrorKind::FieldTypeMismatch));
            }
        };
        Ok(repeated)
    }

    /// Get the value of a repeated 32-bit scalar field.
    fn read_repeated_fixed32<T: Copy>(
        &mut self,
        from_le_bytes: impl Fn([u8; 4]) -> T,
    ) -> Result<Repeated<T, impl Iterator<Item = Result<T, ProtobufError>>>, ProtobufError> {
        let repeated = match self.value {
            FieldValue::I32(val) => Repeated::Unpacked(Some(from_le_bytes(val.to_le_bytes()))),
            FieldValue::Len(len) => {
                let consumed = &mut self.consumed;
                let mut reader = self.reader.sub_limit(len);
                let iter = std::iter::from_fn(move || match reader.read_i32() {
                    Ok(val) => Some(Ok(from_le_bytes(val.to_le_bytes()))),
                    Err(err) if matches!(err.kind(), ErrorKind::Eof) => {
                        *consumed = true;
                        None
                    }
                    Err(err) => Some(Err(err)),
                });
                Repeated::Packed(iter)
            }
            _ => {
                return Err(self.error(ErrorKind::FieldTypeMismatch));
            }
        };
        Ok(repeated)
    }

    /// Get the value of a repeated 64-bit scalar field.
    fn read_repeated_fixed64<T: Copy>(
        &mut self,
        from_le_bytes: impl Fn([u8; 8]) -> T,
    ) -> Result<Repeated<T, impl Iterator<Item = Result<T, ProtobufError>>>, ProtobufError> {
        let repeated = match self.value {
            FieldValue::I64(val) => Repeated::Unpacked(Some(from_le_bytes(val.to_le_bytes()))),
            FieldValue::Len(len) => {
                let consumed = &mut self.consumed;
                let mut reader = self.reader.sub_limit(len);
                let iter = std::iter::from_fn(move || match reader.read_i64() {
                    Ok(val) => Some(Ok(from_le_bytes(val.to_le_bytes()))),
                    Err(err) if matches!(err.kind(), ErrorKind::Eof) => {
                        *consumed = true;
                        None
                    }
                    Err(err) => Some(Err(err)),
                });
                Repeated::Packed(iter)
            }
            _ => {
                return Err(self.error(ErrorKind::FieldTypeMismatch));
            }
        };
        Ok(repeated)
    }

    /// Mark a field as having been read.
    fn consume_field(&mut self) -> Result<(), ProtobufError> {
        if !self.consumed {
            self.consumed = true;
            Ok(())
        } else {
            Err(self.error(ErrorKind::FieldAlreadyConsumed))
        }
    }

    fn error(&self, kind: ErrorKind) -> ProtobufError {
        ProtobufError::new(kind).with_context(self.context, Some(self.number))
    }
}

impl<R: ReadValue> Drop for Field<'_, R> {
    fn drop(&mut self) {
        if !self.consumed {
            // Record field number in parent `Fields` so it can report this on
            // the next call to `Fields::next`.
            *self.unconsumed_field = Some(self.number);
        }
    }
}

/// Iterator over a repeated scalar field.
///
/// Repeated scalar fields may use either a packed or un-packed representation.
/// See https://protobuf.dev/programming-guides/encoding/#repeated.
enum Repeated<T: Copy, I: Iterator<Item = Result<T, ProtobufError>>> {
    Unpacked(Option<T>),
    Packed(I),
}

impl<T: Copy, I: Iterator<Item = Result<T, ProtobufError>>> Iterator for Repeated<T, I> {
    type Item = Result<T, ProtobufError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Unpacked(val) => val.take().map(Ok),
            Self::Packed(packed) => packed.next(),
        }
    }
}

/// Iterator over fields of a message.
///
/// This type has an [`Iterator`]-like interface but does not implement the
/// [`Iterator`] trait because it needs to borrow from itself (it is a _lending
/// iterator_). Instead, you
/// use it with a `while` loop:
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use rten_onnx::protobuf::{Fields, ValueReader};
///
/// // A minimal but valid message. This could also be a `BufReader<File>`
/// // or other buffered reader.
/// let message = vec![0x08, 0x96, 0x01];
///
/// // Incrementally decode message fields.
/// let mut value_reader = ValueReader::from_buf(message);
/// let mut fields = Fields::new(&mut value_reader, None);
/// while let Some(mut field) = fields.next()? {
///     // Process field according to its number. Variable length fields must
///     // always be read with `field.read_*` or skipped with `field.skip()`.
///     //
///     // It is recommended to call `skip` on all unknown fields.
///     field.skip();
/// }
/// # Ok(()) }
/// ```
pub struct Fields<'r, R: ReadValue> {
    reader: LimitReader<'r, R>,

    /// Debug name of the message type.
    context: Option<&'static str>,

    /// The number of the last variable length field which was not consumed
    /// before being dropped. This is used to report an error when attempting
    /// to read the next field.
    unconsumed_field: Option<u64>,
}

impl<'r, R: ReadValue> Fields<'r, R> {
    /// Read a message from `reader`.
    ///
    /// `context` is the name of the message type being read, for debugging
    /// purposes.
    pub fn new(reader: &'r mut R, context: Option<&'static str>) -> Self {
        Self {
            reader: LimitReader::new(reader, u64::MAX),
            context,
            unconsumed_field: None,
        }
    }

    /// Read the next field of the message.
    ///
    /// This returns `Ok(Some(field))` if a field was read, `Ok(None)` if the
    /// end of the input was reached or `Err(err)` if an error was encountered.
    ///
    /// The returned field mutably borrows from the reader, so it must be
    /// processed before `next` can be called again.
    ///
    /// If the returned field has a variable-length type (string, bytes, embedded
    /// message, packed repeated field) then its data must either be consumed
    /// by calling one of the `Field::read_*` methods or skipped over using
    /// [`Field::skip`].
    #[allow(clippy::should_implement_trait)] // Not an Iterator because this borrows from self.
    pub fn next(&mut self) -> Result<Option<Field<'_, R>>, ProtobufError> {
        if let Some(number) = self.unconsumed_field {
            return Err(ProtobufError::new(ErrorKind::FieldNotConsumed)
                .with_context(self.context, Some(number)));
        }

        let tag = match self.reader.read_varint() {
            Ok(tag) => tag,
            Err(err) if matches!(err.kind(), ErrorKind::Eof) => return Ok(None),
            Err(err) => return Err(err),
        };
        let number = tag >> 3;
        let wire_type = tag & 0x7;

        let mut len = 0;
        let value = match wire_type {
            0 => self.reader.read_varint().map(FieldValue::Varint),
            1 => self.reader.read_i64().map(FieldValue::I64),
            2 => self.reader.read_varint().map(|val| {
                len = val;
                FieldValue::Len(val)
            }),
            3 => Ok(FieldValue::Sgroup),
            4 => Ok(FieldValue::Egroup),
            5 => self.reader.read_i32().map(FieldValue::I32),
            _ => Err(ProtobufError::new(ErrorKind::InvalidWireType)),
        }
        .map_err(|err| err.with_context(self.context, Some(number)))?;

        Ok(Some(Field {
            reader: self.reader.sub_limit(len),
            number,
            consumed: !matches!(value, FieldValue::Len(_)),
            value,
            context: self.context,
            unconsumed_field: &mut self.unconsumed_field,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{FieldValue, Fields};
    use crate::protobuf::varint::encode_varint;
    use crate::protobuf::{ErrorKind, ProtobufError, ValueReader};

    fn read_fields(buf: &[u8]) -> Result<Vec<(u64, FieldValue)>, ProtobufError> {
        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, Some("TestMessage"));

        let mut field_vals = Vec::new();
        while let Some(mut field) = fields.next()? {
            field_vals.push((field.number(), field.value()));
            field.skip()?;
        }
        Ok(field_vals)
    }

    #[test]
    fn test_iter_fields_simple() {
        let buf = vec![0x08, 0x96, 0x01];
        let fields = read_fields(&buf).unwrap();
        assert_eq!(fields, [(1, FieldValue::Varint(150))]);
    }

    #[test]
    fn test_iter_fields() {
        let mut buf = Vec::new();
        buf.extend(FieldValue::Varint(1234).encode(1));
        buf.extend(FieldValue::I32(456).encode(2));
        buf.extend(FieldValue::Len(4).encode(3));
        buf.extend([1, 2, 3, 4]);
        buf.extend(FieldValue::I64(678).encode(4));
        buf.extend(FieldValue::Sgroup.encode(5));
        buf.extend(FieldValue::Egroup.encode(6));

        let fields = read_fields(&buf).unwrap();

        assert_eq!(
            fields,
            [
                (1, FieldValue::Varint(1234)),
                (2, FieldValue::I32(456)),
                (3, FieldValue::Len(4)),
                (4, FieldValue::I64(678)),
                (5, FieldValue::Sgroup),
                (6, FieldValue::Egroup),
            ]
        );
    }

    #[test]
    fn test_unconsumed_field() {
        let mut buf = Vec::new();
        buf.extend(FieldValue::Len(4).encode(3));
        buf.extend([1, 2, 3, 4]);
        buf.extend(FieldValue::I64(678).encode(4));

        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, Some("TestMessage"));

        let len_field = fields.next().unwrap().unwrap();
        assert_eq!(len_field.number(), 3);
        std::mem::drop(len_field);

        // Read next field without either skipping or consuming the variable-length field.
        let err = fields.next().err().unwrap();
        assert!(matches!(err.kind(), ErrorKind::FieldNotConsumed));
        assert_eq!(err.context(), Some("TestMessage"));
        assert_eq!(err.field(), Some(3));
    }

    #[test]
    fn test_read_bytes() {
        let mut buf = Vec::new();
        buf.extend(FieldValue::Len(3).encode(1));
        buf.extend([1, 2, 3]);

        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, Some("TestMessage"));
        let mut len_field = fields.next().unwrap().unwrap();
        let bytes = len_field.read_bytes().unwrap();

        assert_eq!(bytes, [1, 2, 3]);
    }

    #[test]
    fn test_read_string() {
        let mut buf = Vec::new();
        buf.extend(FieldValue::Len(5).encode(1));
        buf.extend("hello".as_bytes());

        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, Some("TestMessage"));
        let mut len_field = fields.next().unwrap().unwrap();
        let string = len_field.read_string().unwrap();

        assert_eq!(string, "hello");
    }

    #[test]
    fn test_read_message() {
        let mut sub_msg = Vec::new();
        sub_msg.extend(FieldValue::I32(1).encode(3));
        sub_msg.extend(FieldValue::I32(2).encode(4));

        let mut buf = Vec::new();
        buf.extend(FieldValue::Len(sub_msg.len() as u64).encode(1));
        buf.extend(sub_msg);
        buf.extend(FieldValue::Varint(3).encode(2));

        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, Some("TestMessage"));

        // Read embedded message.
        let mut sub_field = fields.next().unwrap().unwrap();
        let mut sub_fields = sub_field.read_message(Some("SubMessage")).unwrap();
        sub_fields.next().unwrap();
        sub_fields.next().unwrap();
        assert!(sub_fields.next().unwrap().is_none());
        std::mem::drop(sub_field);

        // Read next field in the parent message.
        let final_field = fields.next().unwrap().unwrap();
        assert_eq!(final_field.number(), 2);
        std::mem::drop(final_field);
        assert!(fields.next().unwrap().is_none());
    }

    #[test]
    fn test_read_repeated_varint() {
        let mut buf = Vec::new();
        buf.extend(FieldValue::Varint(1).encode(1)); // Non-packed repeated
        buf.extend(FieldValue::Len(2).encode(1)); // Packed repeated
        buf.extend(encode_varint(2));
        buf.extend(encode_varint(3));

        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, Some("TestMessage"));

        let mut vals = Vec::new();
        while let Some(mut field) = fields.next().unwrap() {
            match field.number() {
                1 => {
                    for val in field.read_repeated_int32().unwrap() {
                        vals.push(val.unwrap());
                    }
                }
                _ => field.skip().unwrap(),
            }
        }
        assert_eq!(vals, [1, 2, 3]);
    }

    #[test]
    fn test_read_repeated_fixed32() {
        let mut buf = Vec::new();
        buf.extend(FieldValue::I32(0).encode(1)); // Non-packed repeated
        buf.extend(FieldValue::Len(8).encode(1)); // Packed repeated
        buf.extend((1.0f32).to_le_bytes());
        buf.extend((2.0f32).to_le_bytes());

        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, Some("TestMessage"));

        let mut vals = Vec::new();
        while let Some(mut field) = fields.next().unwrap() {
            match field.number() {
                1 => {
                    for val in field.read_repeated_float().unwrap() {
                        vals.push(val.unwrap());
                    }
                }
                _ => field.skip().unwrap(),
            }
        }
        assert_eq!(vals, [0., 1., 2.]);
    }

    #[test]
    fn test_read_repeated_fixed64() {
        let mut buf = Vec::new();
        buf.extend(FieldValue::I64(0).encode(1)); // Non-packed repeated
        buf.extend(FieldValue::Len(16).encode(1)); // Packed repeated
        buf.extend((1.0f64).to_le_bytes());
        buf.extend((2.0f64).to_le_bytes());

        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, Some("TestMessage"));

        let mut vals = Vec::new();
        while let Some(mut field) = fields.next().unwrap() {
            match field.number() {
                1 => {
                    for val in field.read_repeated_double().unwrap() {
                        vals.push(val.unwrap());
                    }
                }
                _ => field.skip().unwrap(),
            }
        }
        assert_eq!(vals, [0., 1., 2.]);
    }
}
