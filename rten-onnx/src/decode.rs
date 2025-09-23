//! Utilities for parsing Protocol Buffers messages with minimal copies.

use std::error::Error;
use std::fmt;
use std::fmt::Display;

// Protozero is used as the low-level parser. Note that it is not exposed
// in the public API of the crate. This enables swapping out the parsing library
// in future if that is ever needed.
use protozero::Message;
use protozero::field::{Field, FieldValue};

/// Decode a value from a protocol buffers message.
pub trait DecodeMessage<'a>: Sized {
    /// Decode an instance of this type.
    fn decode(buf: &'a [u8]) -> Result<Self, DecodeError>;
}

/// Update the fields of struct by decoding fields from a Protocol Buffers
/// message.
pub(crate) trait DecodeField<'a> {
    /// Update the field of `self` that corresponds to `field.number` by
    /// using `field.value`.
    ///
    /// If the field is a scalar, this it will be replaced. If it is a vector,
    /// it will be appended to.
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError>;
}

impl<'a, D: DecodeField<'a> + Default> DecodeMessage<'a> for D {
    fn decode(msg: &'a [u8]) -> Result<Self, DecodeError> {
        let msg = Message::new(msg);
        let mut proto = Self::default();
        for field in msg.fields() {
            let field = field?;
            proto.decode_field(field)?;
        }
        Ok(proto)
    }
}

/// Update the value of a field from a Protocol Buffers message field.
pub(crate) trait DecodeFrom<'a> {
    /// Update `self` with the value of a message field.
    ///
    /// If `Self` is a scalar type, its value will be replaced. If it is a
    /// vector, it will be appended to.
    fn decode_from(&mut self, value: FieldValue<'a>) -> Result<(), DecodeError>;
}

impl<'a, M: DecodeMessage<'a>> DecodeFrom<'a> for Option<M> {
    fn decode_from(&mut self, value: FieldValue<'a>) -> Result<(), DecodeError> {
        *self = Some(M::decode(value.get_bytes()?)?);
        Ok(())
    }
}

impl<'a, M: DecodeMessage<'a>> DecodeFrom<'a> for Vec<M> {
    fn decode_from(&mut self, value: FieldValue<'a>) -> Result<(), DecodeError> {
        self.push(M::decode(value.get_bytes()?)?);
        Ok(())
    }
}

/// Impl [`DecodeFrom`] for a scalar field type.
macro_rules! impl_decode_from {
    ($type:ty, $accessor:ident) => {
        impl<'a> DecodeFrom<'a> for $type {
            fn decode_from(&mut self, value: FieldValue<'a>) -> Result<(), DecodeError> {
                *self = value.$accessor()?;
                Ok(())
            }
        }

        impl<'a> DecodeFrom<'a> for Option<$type> {
            fn decode_from(&mut self, value: FieldValue<'a>) -> Result<(), DecodeError> {
                *self = Some(value.$accessor()?);
                Ok(())
            }
        }
    };
}

/// Impl [`DecodeFrom`] for a vector field type.
macro_rules! impl_decode_from_repeated {
    // Decode from a protobuf field which may use a packed representation.
    // See https://protobuf.dev/programming-guides/encoding/#repeated.
    ($type:ty, $accessor:ident, packed) => {
        impl<'a> DecodeFrom<'a> for Vec<$type> {
            fn decode_from(&mut self, value: FieldValue<'a>) -> Result<(), DecodeError> {
                let values = value.$accessor()?;
                for value in values {
                    self.push(value?);
                }
                Ok(())
            }
        }
    };

    // Decode from a protobuf field which uses a non-packed representation.
    ($type:ty, $accessor:ident, non_packed) => {
        impl<'a> DecodeFrom<'a> for Vec<$type> {
            fn decode_from(&mut self, value: FieldValue<'a>) -> Result<(), DecodeError> {
                let value = value.$accessor()?;
                self.push(value);
                Ok(())
            }
        }
    };
}

impl_decode_from!(&'a [u8], get_bytes);
impl_decode_from!(&'a str, get_string);
impl_decode_from_repeated!(&'a str, get_string, non_packed);
impl_decode_from!(f32, get_float);
impl_decode_from_repeated!(f32, get_repeated_float, packed);
impl_decode_from!(f64, get_double);
impl_decode_from_repeated!(f64, get_repeated_double, packed);
impl_decode_from!(i32, get_int32);
impl_decode_from_repeated!(i32, get_repeated_int32, packed);
impl_decode_from!(i64, get_int64);
impl_decode_from_repeated!(i64, get_repeated_int64, packed);

/// Implement the [`DecodeFrom`] trait for an i32 newtype representing an enum.
macro_rules! impl_decode_from_enum {
    ($type:ident) => {
        impl<'a> DecodeFrom<'a> for Option<$type> {
            fn decode_from(&mut self, value: FieldValue<'a>) -> Result<(), DecodeError> {
                *self = Some($type(value.get_enum()?));
                Ok(())
            }
        }
    };
}

pub(crate) use impl_decode_from_enum;

/// Errors decoding protocol buffers messages.
#[derive(Debug)]
pub enum DecodeError {
    /// An unknown field ID was encountered visiting a message, and the decoder
    /// is configured to report these.
    UnknownField { msg: &'static str, tag: u64 },
    /// An unknown error during parsing.
    Other,
}

impl Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownField { msg, tag } => {
                write!(f, "unknown field tag {} in {} message", tag, msg)
            }
            Self::Other => write!(f, "Protocol Buffer parse failed"),
        }
    }
}

impl Error for DecodeError {}

impl From<protozero::Error> for DecodeError {
    fn from(_val: protozero::Error) -> DecodeError {
        // protozero::Error is a zero-sized type, so there is nothing to store
        // in the enum variant.
        DecodeError::Other
    }
}

pub(crate) fn unknown_field(msg: &'static str, field: Field) -> Result<(), DecodeError> {
    Err(DecodeError::UnknownField {
        msg,
        tag: field.number,
    })
}
