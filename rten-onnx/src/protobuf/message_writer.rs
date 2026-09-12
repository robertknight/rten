use std::any::type_name;

use crate::protobuf::errors::{ErrorKind, ProtobufError};
use crate::protobuf::value_writer::{CountingWriter, WriteValue};
use crate::protobuf::varint::varint_len;

// Wire types. See <https://protobuf.dev/programming-guides/encoding/#structure>.
const WIRE_VARINT: u64 = 0;
const WIRE_LEN: u64 = 2;
const WIRE_I32: u64 = 5;

/// Writes the fields of a single message.
pub struct MessageWriter<'w, W: WriteValue> {
    writer: &'w mut W,

    /// Total length of embedded message content which was not written because
    /// the writer only measures lengths.
    ///
    /// When [`WriteValue::COUNT_ONLY`] is set, the length of each embedded
    /// message is measured by [`EncodeMessage::encoded_len`] and recorded here
    /// rather than visiting the message's fields a second time.
    skipped_len: u64,

    /// Debug name of the message type.
    context: Option<&'static str>,
}

impl<'w, W: WriteValue> MessageWriter<'w, W> {
    /// Create a writer which writes the fields of a message to `writer`.
    ///
    /// `context` is the name of the message type being written, for debugging
    /// purposes.
    pub fn new(writer: &'w mut W, context: Option<&'static str>) -> Self {
        Self {
            writer,
            skipped_len: 0,
            context,
        }
    }

    /// Write the value of a field with schema type `int32`.
    pub fn write_int32(&mut self, number: u64, val: i32) -> Result<(), ProtobufError> {
        // `int32` values are sign-extended to 64 bits before being encoded.
        self.write_varint_field(number, val as i64 as u64)
    }

    /// Write the value of a field with schema type `int64`.
    pub fn write_int64(&mut self, number: u64, val: i64) -> Result<(), ProtobufError> {
        self.write_varint_field(number, val as u64)
    }

    /// Write the value of a field where the schema type is an enum.
    pub fn write_enum(&mut self, number: u64, val: i32) -> Result<(), ProtobufError> {
        self.write_int32(number, val)
    }

    /// Write the value of a field with schema type `float`.
    pub fn write_float(&mut self, number: u64, val: f32) -> Result<(), ProtobufError> {
        self.write_tag(number, WIRE_I32)?;
        self.writer.write_i32(i32::from_le_bytes(val.to_le_bytes()))
    }

    /// Write the value of a field with schema type `string`.
    pub fn write_string(&mut self, number: u64, val: &str) -> Result<(), ProtobufError> {
        self.write_bytes(number, val.as_bytes())
    }

    /// Write the value of a field with schema type `bytes`.
    pub fn write_bytes(&mut self, number: u64, val: &[u8]) -> Result<(), ProtobufError> {
        self.write_tag(number, WIRE_LEN)?;
        self.writer.write_varint(val.len() as u64)?;
        self.writer.write_bytes(val)
    }

    /// Write an embedded message as the value of a field.
    pub fn write_message<M: EncodeMessage>(
        &mut self,
        number: u64,
        msg: &M,
    ) -> Result<(), ProtobufError> {
        let len = msg.encoded_len()?;

        self.write_tag(number, WIRE_LEN)?;
        self.writer.write_varint(len)?;

        if W::COUNT_ONLY {
            self.skipped_len += len;
            return Ok(());
        }

        let start = self.writer.position();
        let context = Some(type_name::<M>());
        msg.encode_fields(&mut MessageWriter::new(&mut *self.writer, context))?;

        // Verify that `encode_fields` wrote the number of bytes computed by
        // `encoded_len`.
        let written = self.writer.position() - start;
        if written != len {
            return Err(ProtobufError::new(ErrorKind::LengthMismatch).with_context(context, None));
        }

        Ok(())
    }

    /// Write the elements of an un-packed `repeated int64` field.
    pub fn write_repeated_int64(&mut self, number: u64, vals: &[i64]) -> Result<(), ProtobufError> {
        for val in vals {
            self.write_int64(number, *val)?;
        }
        Ok(())
    }

    /// Write the elements of an un-packed `repeated float` field.
    pub fn write_repeated_float(&mut self, number: u64, vals: &[f32]) -> Result<(), ProtobufError> {
        for val in vals {
            self.write_float(number, *val)?;
        }
        Ok(())
    }

    /// Write the elements of a `repeated string` field.
    pub fn write_repeated_string<S: AsRef<str>>(
        &mut self,
        number: u64,
        vals: &[S],
    ) -> Result<(), ProtobufError> {
        for val in vals {
            self.write_string(number, val.as_ref())?;
        }
        Ok(())
    }

    /// Write the elements of a repeated field with an embedded message type.
    pub fn write_repeated_message<M: EncodeMessage>(
        &mut self,
        number: u64,
        msgs: &[M],
    ) -> Result<(), ProtobufError> {
        for msg in msgs {
            self.write_message(number, msg)?;
        }
        Ok(())
    }

    /// Write the elements of a packed `repeated int32` field.
    pub fn write_packed_int32(&mut self, number: u64, vals: &[i32]) -> Result<(), ProtobufError> {
        self.write_packed_varints(number, vals.iter().map(|val| *val as i64 as u64))
    }

    /// Write the elements of a packed `repeated int64` field.
    pub fn write_packed_int64(&mut self, number: u64, vals: &[i64]) -> Result<(), ProtobufError> {
        self.write_packed_varints(number, vals.iter().map(|val| *val as u64))
    }

    /// Write the elements of a packed `repeated float` field.
    pub fn write_packed_float(&mut self, number: u64, vals: &[f32]) -> Result<(), ProtobufError> {
        if vals.is_empty() {
            return Ok(());
        }
        self.write_tag(number, WIRE_LEN)?;
        self.writer.write_varint(vals.len() as u64 * 4)?;
        for val in vals {
            self.writer
                .write_i32(i32::from_le_bytes(val.to_le_bytes()))?;
        }
        Ok(())
    }

    /// Write the elements of a packed `repeated double` field.
    pub fn write_packed_double(&mut self, number: u64, vals: &[f64]) -> Result<(), ProtobufError> {
        if vals.is_empty() {
            return Ok(());
        }
        self.write_tag(number, WIRE_LEN)?;
        self.writer.write_varint(vals.len() as u64 * 8)?;
        for val in vals {
            self.writer
                .write_i64(i64::from_le_bytes(val.to_le_bytes()))?;
        }
        Ok(())
    }

    fn write_packed_varints(
        &mut self,
        number: u64,
        vals: impl Iterator<Item = u64> + Clone,
    ) -> Result<(), ProtobufError> {
        let len: u64 = vals.clone().map(|val| varint_len(val) as u64).sum();
        if len == 0 {
            return Ok(());
        }
        self.write_tag(number, WIRE_LEN)?;
        self.writer.write_varint(len)?;
        for val in vals {
            self.writer.write_varint(val)?;
        }
        Ok(())
    }

    fn write_varint_field(&mut self, number: u64, val: u64) -> Result<(), ProtobufError> {
        self.write_tag(number, WIRE_VARINT)?;
        self.writer.write_varint(val)
    }

    fn write_tag(&mut self, number: u64, wire_type: u64) -> Result<(), ProtobufError> {
        self.writer
            .write_varint((number << 3) | wire_type)
            .map_err(|err| err.with_context(self.context, Some(number)))
    }
}

/// Defines how to serialize a type as an encoded message.
pub trait EncodeMessage {
    /// Write the fields of this message.
    ///
    /// Implementations must write the same fields with the same values each
    /// time they are called for a given message, as the message is visited once
    /// to determine its encoded length and once to write it.
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError>;

    /// Return the length of this message in bytes when encoded.
    ///
    /// This visits the fields of the message using [`CountingWriter`].
    fn encoded_len(&self) -> Result<u64, ProtobufError> {
        let mut counter = CountingWriter::new();
        let context = Some(type_name::<Self>());
        let mut fields = MessageWriter::new(&mut counter, context);
        self.encode_fields(&mut fields)?;
        let skipped_len = fields.skipped_len;
        Ok(counter.position() + skipped_len)
    }

    /// Encode this message to a writer.
    fn encode<W: WriteValue>(&self, mut writer: W) -> Result<(), ProtobufError> {
        let context = Some(type_name::<Self>());
        {
            let mut fields = MessageWriter::new(&mut writer, context);
            self.encode_fields(&mut fields)?;
        }
        writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{EncodeMessage, MessageWriter};
    use crate::protobuf::{
        ErrorKind, FieldValue, Fields, ProtobufError, ValueReader, ValueWriter, WriteValue,
    };

    /// Encode a message and return the encoded bytes.
    fn encode<M: EncodeMessage>(msg: &M) -> Vec<u8> {
        let mut buf = Vec::new();
        msg.encode(ValueWriter::new(&mut buf)).unwrap();

        assert_eq!(msg.encoded_len().unwrap(), buf.len() as u64);

        buf
    }

    /// Read the number and value of each field in an encoded message.
    fn read_fields(buf: &[u8]) -> Vec<(u64, FieldValue)> {
        let mut reader = ValueReader::from_buf(buf);
        let mut fields = Fields::new(&mut reader, None);

        let mut field_vals = Vec::new();
        while let Some(mut field) = fields.next().unwrap() {
            field_vals.push((field.number(), field.value()));
            field.skip().unwrap();
        }
        field_vals
    }

    // Message which writes each kind of scalar field.
    struct Scalars;

    impl EncodeMessage for Scalars {
        fn encode_fields<W: WriteValue>(
            &self,
            fields: &mut MessageWriter<W>,
        ) -> Result<(), ProtobufError> {
            fields.write_int32(1, -5)?;
            fields.write_int64(2, 1234)?;
            fields.write_enum(3, 7)?;
            fields.write_float(4, 0.5)?;
            fields.write_string(5, "hello")?;
            fields.write_bytes(6, &[1, 2, 3])
        }
    }

    #[test]
    fn test_write_scalar_fields() {
        let buf = encode(&Scalars);

        let mut reader = ValueReader::from_buf(&buf);
        let mut fields = Fields::new(&mut reader, None);

        let field = fields.next().unwrap().unwrap();
        assert_eq!(field.number(), 1);
        assert_eq!(field.get_int32().unwrap(), -5);
        drop(field);

        let field = fields.next().unwrap().unwrap();
        assert_eq!(field.number(), 2);
        assert_eq!(field.get_int64().unwrap(), 1234);
        drop(field);

        let field = fields.next().unwrap().unwrap();
        assert_eq!(field.number(), 3);
        assert_eq!(field.get_enum().unwrap(), 7);
        drop(field);

        let field = fields.next().unwrap().unwrap();
        assert_eq!(field.number(), 4);
        assert_eq!(field.get_float().unwrap(), 0.5);
        drop(field);

        let mut field = fields.next().unwrap().unwrap();
        assert_eq!(field.number(), 5);
        assert_eq!(field.read_string().unwrap(), "hello");
        drop(field);

        let mut field = fields.next().unwrap().unwrap();
        assert_eq!(field.number(), 6);
        assert_eq!(field.read_bytes().unwrap(), [1, 2, 3]);
        drop(field);

        assert!(fields.next().unwrap().is_none());
    }

    // Message which writes repeated fields using packed and un-packed
    // representations.
    struct Repeated;

    impl EncodeMessage for Repeated {
        fn encode_fields<W: WriteValue>(
            &self,
            fields: &mut MessageWriter<W>,
        ) -> Result<(), ProtobufError> {
            fields.write_repeated_int64(1, &[1, 2])?;
            fields.write_repeated_float(2, &[1.0, 2.0])?;
            fields.write_repeated_string(3, &["a".to_string(), "b".to_string()])?;
            fields.write_packed_int32(4, &[3, 4])?;
            fields.write_packed_int64(5, &[5, 6])?;
            fields.write_packed_float(6, &[3.0, 4.0])?;
            fields.write_packed_double(7, &[5.0, 6.0])?;

            // Empty repeated fields should be omitted entirely.
            fields.write_repeated_int64(8, &[])?;
            fields.write_repeated_float(9, &[])?;
            fields.write_packed_int32(10, &[])?;
            fields.write_packed_int64(11, &[])?;
            fields.write_packed_float(12, &[])?;
            fields.write_packed_double(13, &[])
        }
    }

    #[test]
    fn test_write_repeated_fields() {
        let buf = encode(&Repeated);

        assert_eq!(
            read_fields(&buf),
            [
                // repeated int64
                (1, FieldValue::Varint(1)),
                (1, FieldValue::Varint(2)),
                // repeated float
                (2, FieldValue::I32(1.0f32.to_bits() as i32)),
                (2, FieldValue::I32(2.0f32.to_bits() as i32)),
                // repeated string
                (3, FieldValue::Len(1)),
                (3, FieldValue::Len(1)),
                // packed i32
                (4, FieldValue::Len(2)),
                // packed i64
                (5, FieldValue::Len(2)),
                // packed float
                (6, FieldValue::Len(8)),
                // packed double
                (7, FieldValue::Len(16)),
            ]
        );

        // Verify the result can be read using `ValueReader`.
        let mut reader = ValueReader::from_buf(&buf);
        let mut fields = Fields::new(&mut reader, None);
        let mut int64s = Vec::new();
        let mut floats = Vec::new();
        let mut strings = Vec::new();
        let mut int32s = Vec::new();
        let mut packed_int64s = Vec::new();
        let mut packed_floats = Vec::new();
        let mut doubles = Vec::new();

        while let Some(mut field) = fields.next().unwrap() {
            match field.number() {
                1 => int64s.extend(field.read_repeated_int64().unwrap().map(|x| x.unwrap())),
                2 => floats.extend(field.read_repeated_float().unwrap().map(|x| x.unwrap())),
                3 => strings.push(field.read_string().unwrap()),
                4 => int32s.extend(field.read_repeated_int32().unwrap().map(|x| x.unwrap())),
                5 => packed_int64s.extend(field.read_repeated_int64().unwrap().map(|x| x.unwrap())),
                6 => packed_floats.extend(field.read_repeated_float().unwrap().map(|x| x.unwrap())),
                7 => doubles.extend(field.read_repeated_double().unwrap().map(|x| x.unwrap())),
                _ => panic!("unexpected field {}", field.number()),
            }
        }

        assert_eq!(int64s, [1, 2]);
        assert_eq!(floats, [1.0, 2.0]);
        assert_eq!(strings, ["a", "b"]);
        assert_eq!(int32s, [3, 4]);
        assert_eq!(packed_int64s, [5, 6]);
        assert_eq!(packed_floats, [3.0, 4.0]);
        assert_eq!(doubles, [5.0, 6.0]);
    }

    // Message with several levels of nesting. Each level writes one scalar
    // field followed by `depth` nested messages.
    struct Nested {
        depth: u32,
    }

    impl EncodeMessage for Nested {
        fn encode_fields<W: WriteValue>(
            &self,
            fields: &mut MessageWriter<W>,
        ) -> Result<(), ProtobufError> {
            fields.write_int32(1, self.depth as i32)?;
            if self.depth > 0 {
                let child = Nested {
                    depth: self.depth - 1,
                };
                fields.write_repeated_message(2, &[child])?;
            }
            Ok(())
        }
    }

    #[test]
    fn test_write_nested_messages() {
        let msg = Nested { depth: 3 };
        let buf = encode(&msg);

        // Read the nested messages back. This will fail if the length prefix
        // written for any level is incorrect.
        fn read_nested(buf: &[u8], depths: &mut Vec<i32>) {
            let mut reader = ValueReader::from_buf(buf);
            let mut fields = Fields::new(&mut reader, None);
            while let Some(mut field) = fields.next().unwrap() {
                match field.number() {
                    1 => depths.push(field.get_int32().unwrap()),
                    2 => {
                        let nested = field.read_bytes().unwrap();
                        read_nested(&nested, depths);
                    }
                    _ => panic!("unexpected field {}", field.number()),
                }
            }
        }

        let mut depths = Vec::new();
        read_nested(&buf, &mut depths);

        assert_eq!(depths, [3, 2, 1, 0]);
    }

    // Message which writes a different number of fields each time it is
    // encoded.
    struct Inconsistent {
        encode_count: Cell<u32>,
    }

    impl EncodeMessage for Inconsistent {
        fn encode_fields<W: WriteValue>(
            &self,
            fields: &mut MessageWriter<W>,
        ) -> Result<(), ProtobufError> {
            self.encode_count.set(self.encode_count.get() + 1);
            for i in 0..self.encode_count.get() {
                fields.write_int32(1, i as i32)?;
            }
            Ok(())
        }
    }

    struct InconsistentParent {
        child: Inconsistent,
    }

    impl EncodeMessage for InconsistentParent {
        fn encode_fields<W: WriteValue>(
            &self,
            fields: &mut MessageWriter<W>,
        ) -> Result<(), ProtobufError> {
            fields.write_message(1, &self.child)
        }
    }

    #[test]
    fn test_inconsistent_message_length() {
        let msg = InconsistentParent {
            child: Inconsistent {
                encode_count: Cell::new(0),
            },
        };

        let mut buf = Vec::new();
        let err = msg
            .encode(ValueWriter::new(&mut buf))
            .expect_err("expected error");

        assert!(matches!(err.kind(), ErrorKind::LengthMismatch));
    }
}
