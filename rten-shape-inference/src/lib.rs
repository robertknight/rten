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
//!
//! # Crate overview
//!
//! The main export of this crate is the [`InferShapes`] trait, plus types which
//! implement it in [`ops`]. This trait computes the output shapes produced for
//! a given set of input shapes. The shapes are symbolic, meaning they can
//! represent variables that change at inference time.
//!
//! Many ONNX operators have the same shape inference rules, so there is an M:1
//! mapping between operators and shape inference implementations. For most
//! operators, shape inference only represents how the operator transforms the
//! shapes of tensors, but not their values. For a subset of operators, shape
//! inference can also understand how the operator transforms the values of
//! inputs, where the values are scalars or vectors of symbolic expressions.
//! This is needed for understanding subgraphs in ONNX models that extract and
//! transform shapes. For example, shape inference for the `Concat` op can
//! express that concatentating vectors `["batch"]` and `["height" / 2, "width"
//! / 2]` produces the output `["batch", "height" / 2, "width" / 2]`.
//!
//! # Symbolic values
//!
//! Symbolic values are multi-dimensional array types where the dimension sizes
//! and elements are _symbolic expressions_. Expressions can be known integers,
//! named symbols, or composite expressions involving these. Values are
//! represented by [`SymTensor`] and expressions by [`SymExpr`].

mod infer_shapes;
pub mod ops;
mod sym_expr;
mod sym_gen;
mod sym_tensor;

pub use infer_shapes::{BinaryOp, InferShapes, InferShapesError, ReductionOp, UnaryOp, VariadicOp};
pub use sym_expr::{EvalError, SymExpr, Symbol, SymbolMap};
pub use sym_gen::SymbolGen;
pub use sym_tensor::{Constant, SymTensor};
