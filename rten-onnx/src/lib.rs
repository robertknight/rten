//! ONNX model file parser.
//!
//! This crate is used to parse [ONNX][onnx] model files, with a focus on
//! minimizing dependencies, parse time and memory overhead.
//!
//! # About ONNX models
//!
//! ONNX models are [Protocol Buffers][protobuf] messages using the `ModelProto`
//! schema from
//! [onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3). The
//! `ModelProto` message describes the model structure as a graph. The weights
//! and other parameters may be included as part of the Protocol Buffers message
//! or stored in external data files.
//!
//! # Design
//!
//! ONNX model files can be large (up to 2GB) and contain many strings and large
//! tensor data fields. To minimize parse time and memory usage, parsing
//! functions return structures which contain references into the input buffer.
//! This avoids allocations for all strings and most tensor data, which is
//! usually stored as bytes (`TensorProto.raw_bytes`).
//!
//! In order to have control over allocations and minimize compile times, this
//! crate uses a low-level parser for the Protocol Buffers wire format
//! ([protozero]) with some helper traits, rather than use a runtime based on
//! code generation like [Prost][prost]. This is viable because the schema of
//! ONNX model files is relatively small and changes infrequently.
//!
//! [onnx]: https://onnx.ai/onnx/
//! [protobuf]: https://protobuf.dev/
//! [protozero]: https://github.com/kalcutter/protozero-rs
//! [prost]: https://github.com/tokio-rs/prost

#![forbid(unsafe_code)]

mod decode;
pub mod onnx;

pub use decode::{DecodeError, DecodeMessage};
