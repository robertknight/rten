//! This crate provides a parser for [ONNX][onnx] ML model files.
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
//! When the weights are stored internally within the model file, they are
//! usually stored as `bytes` fields in `TensorProto.raw_data`.
//!
//! # Usage
//!
//! To read a model from a file:
//!
//! ```no_run
//! use std::error::Error;
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! use rten_onnx::onnx::ModelProto;
//! use rten_onnx::protobuf::{DecodeMessage, ReadPos, ValueReader};
//!
//! fn main() -> Result<(), Box<dyn Error>> {
//!     let file = File::open("model.onnx")?;
//!     let reader = ReadPos::new(BufReader::new(file));
//!     let value_reader = ValueReader::new(reader);
//!     let model = ModelProto::decode(value_reader)?;
//!
//!     let op_count = model.graph.as_ref().map(|g| g.node.len()).unwrap_or(0);
//!     let weight_count = model.graph.as_ref().map(|g| g.initializer.len()).unwrap_or(0);
//!
//!     println!("Model has {} operators and {} weights", op_count, weight_count);
//!     
//!     Ok(())
//! }
//! ```
//!
//! To read a model from an in-memory buffer, use:
//!
//! ```no_run
//! # use rten_onnx::{onnx::ModelProto, protobuf::{ValueReader, DecodeMessage}};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let buffer = std::fs::read("model.onnx")?;
//! use std::io::Cursor;
//!
//! let reader = Cursor::new(&buffer);
//! let value_reader = ValueReader::new(reader);
//! let model = ModelProto::decode(value_reader)?;
//! # Ok(()) }
//! ```
//!
//! # Design
//!
//! An ONNX model consists of a graph describing the model architecture and
//! weights. The data describing the graph is usually a few hundred KB. The
//! weights are orders of magnitude larger. The time taken to load a model is
//! therefore dominated by reading, copying or processing weights.
//!
//! The fastest way to read ML models from disk is to map the file into memory
//! using `mmap` and use the mapped memory directly as the data storage for
//! weight tensors. This requires that the data is appropriately aligned
//! relative to the start of the file. Tensors with `f32` data must have an
//! alignment of at least 4 bytes for example. The next best option is to read
//! the file into one large buffer and use slices of it as storage for tensors.
//! This also requires the data to be appropriately aligned.
//!
//! For ONNX files which store tensor data in external files, it is possible to
//! use either of the above strategies to load the weights. The `.onnx` file
//! itself contains only the model graph and can be parsed efficiently using
//! standard Protocol Buffers parsers.
//!
//! ONNX models with embedded weights however intersperse the ML graph and the
//! weights, and there is no guaranteed alignment for the fields containing the
//! weights. To minimize load time of these models rten-onnx uses a custom
//! Protocol Buffers parser which allows reading data for `bytes` and `strings`
//! fields directly from a file into suitably aligned buffers. This contrasts
//! for example with libraries like [Prost](https://github.com/tokio-rs/prost)
//! which require the file to be read into a buffer before it is parsed,
//! incurring an extra copy for every tensor.
//!
//! In addition to minimizing weight copies, the parser is also able to skip
//! reading data that is not necessary for inference, such as doc strings and
//! metadata fields.
//!
//! [onnx]: https://onnx.ai/onnx/
//! [protobuf]: https://protobuf.dev/
//! [prost]: https://github.com/tokio-rs/prost

// This is a crate for parsing potentially untrusted files, so it is preferable
// to avoid unsafe code.
#![forbid(unsafe_code)]

pub mod onnx;
pub mod protobuf;
