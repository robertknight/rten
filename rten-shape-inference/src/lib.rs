//! Shape inference for ONNX graphs.
//!
//! # About shape inference
//!
//! Some ONNX model optimizations depend upon knowledge about the shapes and
//! types of various values in the graph. These values may have dynamic sizes
//! that depend on model inputs. In a typical language model for example, the
//! input has dynamic dimensions for the batch size and sequence length.
//!
//! The goal of shape inference is to take information embedded in the model
//! about the shapes of model inputs and trace how graph operators transform,
//! extract and otherwise process tensor shapes, and produce metadata about the
//! shape of each value in the graph.
//!
//! As an example, suppose a model has an image input of shape (batch, 3,
//! height, width) and computes a mask with shape (batch, height, width). This
//! could be done with a sequence of operators such as:
//!
//! ```text
//! S = Shape(Image) // ["batch", 3, "height", "width"]
//! B = Gather(S, axis=0, indices=0) // "batch"
//! BV = Unsqueeze(B, axis=0) // ["batch"]
//! H = Gather(S, axis=0, indices=2) // "height"
//! HV = Unsqueeze(H, axis=0) // ["height"]
//! W = Gather(S, axis=0, indices=3) // "width"
//! WV = Unsqueeze(H, axis=0) // ["width"]
//! S2 = Concat<axis=0>(BV, HV, WV) // ["batch", "height", "width"]
//! Mask = ConstantOfShape<value=1>(S2) // shape("batch", "height", "width")
//! ```
//!
//! Shape inference of this graph involves following the extraction of the of
//! the input shape, its transformation and uses in order to determine the shape
//! of the output. If there was an optimization to combine all these nodes into
//! one, which depended on knowing that the output shape was the same as the
//! input minus the second dimension, the results of shape inference could be
//! used to verify this.
pub mod infer_shapes;
pub mod sym_gen;
pub mod sym_tensor;
