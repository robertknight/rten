//! Low-level crate for parsing `.rten`-format machine learning model files.
//!
//! # About .rten model files
//!
//! RTen model files contain both the model graph and tensor data. The model
//! graph is in [FlatBuffers](https://flatbuffers.dev) format and closely
//! follows the [ONNX](https://onnx.ai/onnx/) specification. Tensor data is
//! stored following the model graph.
//!
//! See [this document](https://github.com/robertknight/rten/blob/main/docs/rten-file-format.md)
//! for more details of the format and the rationale for its design.

/// Schema for the model graph, generated using the FlatBuffers compiler.
///
/// See `schema.fbs` for the FlatBuffers source.
#[allow(
    clippy::extra_unused_lifetimes,
    clippy::missing_safety_doc,
    mismatched_lifetime_syntaxes,
    unused_imports
)]
pub mod schema_generated;

/// Parse the header of a .rten model file.
pub mod header;

pub use schema_generated as schema;
