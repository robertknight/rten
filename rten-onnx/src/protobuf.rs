//! Low-level incremental Protocol Buffers message decoder.
//!
//! This module provides a low-level API for incrementally decoding [Protocol
//! Buffers](https://protobuf.dev/) messages from a buffered reader. Decoders
//! have control over which types are used for message fields and which fields
//! are read or skipped. Unlike with many implementations, it is not necessary
//! to read the whole message into memory before decoding.
//!
//! # Prerequisites
//!
//! To use this library, it is helpful to have an understanding of how Protocol
//! Buffers messages are encoded. See
//! <https://protobuf.dev/programming-guides/encoding/> for a guide.
//!
//! # Defining decoders
//!
//! The main way to use this library is to define a type into which a message
//! will be deserialized and implement [`DecodeMessage`] for it. See the
//! [`DecodeMessage`] documentation for an example.
//!
//! `DecodeMessage` implementations will use the lower-level [`Fields`] type to
//! visit fields of a message and update fields of the returned struct.
//!
//! # Deserializing messages
//!
//! Given a file or buffer containing a message, a type to deserialize into
//! and a `DecodeMessage` implementation for it, the steps to deserialize are:
//!
//! 1. Create a buffered reader (eg. `BufReader<File>` for a file or `Cursor`
//!    for an in-memory buffer)
//! 2. Wrap the buffered reader in a [`ReadPos`] to add position tracking. This
//!    is not needed for readers which already provide this (eg. `Cursor`)
//! 3. Create a Protocol Buffers value reader that wraps the buffered reader.
//!    The value reader implementation determines the types used to represent
//!    field values. The most common choice is [`ValueReader`].
//! 4. Invoke the [`decode`](DecodeMessage) function for your type.
//!
//! See the [`DecodeMessage`] trait for a full example.

mod errors;
mod field;
mod message;
mod value;
pub mod varint;

pub use errors::{ErrorKind, ProtobufError};
pub use field::{Field, FieldValue, Fields};
pub use message::DecodeMessage;
pub use value::{FieldTypes, OwnedValues, ReadPos, ReadValue, ValueReader};
