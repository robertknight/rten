use crate::protobuf::{Field, FieldTypes, Fields, ProtobufError, ReadValue};

/// Defines how to deserialize a type from an encoded message.
///
/// # Usage
///
/// Given the Protocol Buffers schema:
///
/// ```proto
/// message Message {
///     int32 int_field = 1;
///     string string_field = 2;
///     bytes bytes_field = 3;
/// }
/// ```
///
/// A decoder could be written as follows:
///
/// ```
/// use std::io::{BufRead, Cursor};
/// use std::sync::Arc;
/// use rten_onnx::protobuf::{DecodeMessage, Fields, OwnedValues, ReadPos, ProtobufError,
/// ReadValue, ValueReader};
///
/// #[derive(Default)]
/// struct Message {
///     int_field: i32,
///     string_field: Option<String>,
///
///     // Use `Arc` here as an example of customizing the storage of
///     // deserialized types.
///     bytes_field: Option<Arc<Vec<u8>>>,
/// }
///
/// impl Message {
///     const INT_FIELD: u64 = 1;
///     const STRING_FIELD: u64 = 2;
///     const BYTES_FIELD: u64 = 3;
/// }
///
/// impl DecodeMessage for Message {
///     type Types = OwnedValues;
///
///     fn decode_fields<R: ReadValue<Types = Self::Types>>(mut fields: Fields<R>) -> Result<Self, ProtobufError> {
///         let mut msg = Message::default();
///         while let Some(mut field) = fields.next()? {
///             match field.number() {
///                 Self::INT_FIELD => { msg.int_field = field.get_int32()?; }
///                 Self::STRING_FIELD => {
///                     msg.string_field = Some(field.read_string()?);
///                 }
///                 Self::BYTES_FIELD => {
///                     msg.bytes_field = Some(Arc::new(field.read_bytes()?));
///                 }
///                 _ => {
///                     // Skip over fields that are unrecognized or not useful
///                     // for us.
///                     field.skip()?;
///                 }
///             }
///         }
///         Ok(msg)
///     }
/// }
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let message: &[u8] = &[
///         0x08, 0x96, 0x01, // int_field = 150
///         0x12, 0x02, 0x68, 0x69, // string_field = "hi"
///         0x1A, 0x02, 0x01, 0x02, // bytes_field = [0x01, 0x02]
///     ];
///     let reader = Cursor::new(message);
///     let value_reader = ValueReader::new(reader);
///     let msg = Message::decode(value_reader)?;
///
///     assert_eq!(msg.int_field, 150);
///     assert_eq!(msg.string_field.unwrap(), "hi");
///     assert_eq!(msg.bytes_field.unwrap().as_slice(), [0x01, 0x02]);
///
///     Ok(())
/// }
/// ```
pub trait DecodeMessage: Sized {
    /// Specifies the types of collections used for `bytes` and `strings` fields
    /// by the reader passed to [`decode`](DecodeMessage::decode).
    ///
    /// The most common value is [`OwnedValues`](crate::protobuf::OwnedValues).
    type Types: FieldTypes;

    /// Decode a message from a reader.
    fn decode<R>(mut reader: R) -> Result<Self, ProtobufError>
    where
        R: ReadValue<Types = Self::Types>,
    {
        let ctx = Some(std::any::type_name::<Self>());
        Self::decode_fields(Fields::new(&mut reader, ctx))
    }

    /// Decode a message from an iterator over message fields.
    fn decode_fields<R>(fields: Fields<R>) -> Result<Self, ProtobufError>
    where
        R: ReadValue<Types = Self::Types>;

    /// Decode a message stored in a field in a parent message.
    fn decode_field<R>(field: &mut Field<R>) -> Result<Self, ProtobufError>
    where
        R: ReadValue<Types = Self::Types>,
    {
        let ctx = Some(std::any::type_name::<Self>());
        Self::decode_fields(field.read_message(ctx)?)
    }
}
