use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::protobuf::varint::VarintError;

/// Errors parsing Protocol Buffers messages.
#[derive(Debug)]
pub struct ProtobufError {
    kind: ErrorKind,
    context: Option<&'static str>,
    field: Option<u64>,
}

impl ProtobufError {
    pub fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            context: None,
            field: None,
        }
    }

    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Return the message type associated with this error.
    pub fn context(&self) -> Option<&str> {
        self.context
    }

    /// Return the field number associated with this error.
    pub fn field(&self) -> Option<u64> {
        self.field
    }

    /// Associate a message type and/or field number with this error.
    pub fn with_context(mut self, context: Option<&'static str>, field: Option<u64>) -> Self {
        self.context = context;
        self.field = field;
        self
    }
}

impl std::fmt::Display for ProtobufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "error in message {} field {}: {}",
            self.context.unwrap_or_default(),
            self.field.unwrap_or(0),
            self.kind
        )
    }
}

impl Error for ProtobufError {}

impl From<std::io::Error> for ProtobufError {
    fn from(val: std::io::Error) -> Self {
        Self::new(ErrorKind::IoError(val))
    }
}

impl From<VarintError> for ProtobufError {
    fn from(val: VarintError) -> Self {
        match val {
            VarintError::Eof => Self::new(ErrorKind::Eof),
            VarintError::InvalidVarint => Self::new(ErrorKind::InvalidVarint),
            VarintError::IoError(err) => Self::new(ErrorKind::IoError(err)),
        }
    }
}

/// Enum describing the kind of a [`ProtobufError`] error.
#[derive(Debug)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An IO error occured while decoding the message.
    IoError(std::io::Error),

    /// An invalid varint value was encountered.
    ///
    /// This can be reported if a varint value is encountered that contains
    /// more than 64 bits of value data.
    InvalidVarint,

    /// The end of the file was reached unexpectedly.
    Eof,

    /// Attempted to read a field value of a type that doesn't match the wire
    /// type.
    FieldTypeMismatch,

    /// A repeated field has a length that is not a multiple of the element size.
    FieldLengthMismatch,

    /// A field has an invalid wire type.
    ///
    /// Protocol Buffers defines 6 wire types, but uses 3 bits to encode them.
    /// Hence there are two unused values.
    InvalidWireType,

    /// Attempted to read a field value which has already been read.
    FieldAlreadyConsumed,

    /// A string field contained invalid UTF-8.
    InvalidUtf8,

    /// A variable length field was not read or skipped over.
    FieldNotConsumed,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorKind::IoError(err) => write!(f, "io error: {err}"),
            ErrorKind::InvalidVarint => write!(f, "invalid varint"),
            ErrorKind::Eof => write!(f, "end of file"),
            ErrorKind::FieldTypeMismatch => write!(f, "field type mismatch"),
            ErrorKind::FieldLengthMismatch => write!(f, "field length mismatch"),
            ErrorKind::InvalidWireType => write!(f, "invalid wire type"),
            ErrorKind::FieldAlreadyConsumed => write!(f, "field already consumed"),
            ErrorKind::InvalidUtf8 => write!(f, "invalid UTF-8 in string"),
            ErrorKind::FieldNotConsumed => {
                write!(f, "variable-length field not consumed or skipped")
            }
        }
    }
}
