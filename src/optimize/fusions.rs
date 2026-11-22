//! Traits for defining operator fusions and implementations of fusions.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
use rustc_hash::FxHashSet;

use rten_shape_inference::{SymExpr, Symbol};
use rten_tensor::{ArcTensor, AsView, Layout, NdTensorView, SliceRange, Tensor, TensorView};

use crate::buffer_pool::BufferPool;
use crate::graph::{
    Constant, ConstantNode, ConstantNodeData, Dimension, Graph, Node, NodeId, OperatorNode,
    TypedConstant,
};
use crate::operator::Operator;
use crate::ops::transform_inputs::TransformInputsBuilder;
use crate::ops::{
    Add, AddSoftmax, BatchNormalization, Cast, ComputeShape, Conv, ConvDynamicQuant, ConvInteger,
    ConvIntegerToFloat, ConvTranspose, ConvTransposeDynamicQuant, DequantizeLinear, Div, Dst,
    FusedMatMul, Gelu, GroupedQueryAttentionMatMul, LayerNormalization, MatMul, MatMulDynamicQuant,
    MatMulIntegerToFloat, Mul, Pass, Pow, RMSNormalization, Reciprocal, ReduceMean,
    RepeatInterleave, ScalarOp, Shape, Silu, Sin, Softmax, Sqrt, Src, Sub, Swish, SymbolInfo,
    TileOp, TileProgram, Transpose, dequantize_linear,
};
use crate::optimize::pattern_matcher::{Match, Pattern};
use crate::optimize::{DynamicQuantOp, DynamicQuantOptions};
use crate::value::{DataType, ValueType};

#[derive(Debug)]
pub struct FusedOp {
    /// The name of the graph node.
    pub name: Option<String>,

    pub fused_op: Arc<dyn Operator + Send + Sync>,

    /// IDs of input value nodes.
    pub input_ids: Vec<Option<NodeId>>,

    /// New constants to create and use as inputs to the fused operator.
    ///
    /// Each `(index, constant)` entry causes `constant` to be added to the
    /// graph and its node ID placed at position `index` in [`input_ids`]. The
    /// corresponding slot in [`input_ids`] should be `None`.
    ///
    /// [`input_ids`]: FusedOp::input_ids
    pub new_constant_inputs: Vec<(usize, Constant)>,

    /// IDs of output value nodes.
    pub output_ids: Vec<Option<NodeId>>,

    /// IDs of additional value nodes which are used by the unfused graph but
    /// not required by the fused operator.
    pub unused_input_ids: Vec<Option<NodeId>>,
}

#[derive(Debug)]
pub enum Fusion {
    /// Replace a subgraph with a single operation.
    Op(FusedOp),

    /// Replace a subgraph's outputs with one of its inputs.
    Identity { input_id: NodeId, output_id: NodeId },

    /// Replace an operator with a constant value.
    ///
    /// The replaced operator must have a single output value.
    Constant {
        /// Input IDs for the subgraph being replaced.
        input_ids: Vec<NodeId>,

        /// ID of the value node being replaced.
        output_id: NodeId,

        /// Constant name and value.
        value: Constant,
    },
}

impl Fusion {
    /// Create a fusion with a given operator, name and input and output nodes.
    fn from_op(
        name: Option<&str>,
        fused_op: Arc<dyn Operator + Send + Sync>,
        input_ids: &[Option<NodeId>],
        output_ids: &[Option<NodeId>],
        unused_input_ids: &[Option<NodeId>],
    ) -> Fusion {
        Fusion::Op(FusedOp {
            name: name.map(|s| s.to_string()),
            fused_op,
            input_ids: input_ids.to_vec(),
            new_constant_inputs: Vec::new(),
            output_ids: output_ids.to_vec(),
            unused_input_ids: unused_input_ids.to_vec(),
        })
    }

    /// Return all inputs to the fused subgraph.
    pub fn input_ids(&self) -> Vec<NodeId> {
        match self {
            Fusion::Op(op) => op.input_ids.iter().copied().flatten().collect(),
            Fusion::Identity {
                input_id,
                output_id: _,
            } => [*input_id].into(),
            Fusion::Constant { input_ids, .. } => input_ids.clone(),
        }
    }

    /// Return all inputs to the unfused subgraph which are unused in the
    /// fused version.
    pub fn unused_input_ids(&self) -> Vec<NodeId> {
        match self {
            Fusion::Op(op) => op.unused_input_ids.iter().copied().flatten().collect(),
            Fusion::Identity { .. } => Vec::new(),
            Fusion::Constant { .. } => Vec::new(),
        }
    }

    /// Return all outputs from the fused subgraph.
    pub fn output_ids(&self) -> Vec<NodeId> {
        match self {
            Fusion::Op(op) => op.output_ids.iter().copied().flatten().collect(),
            Fusion::Identity {
                input_id: _,
                output_id,
            } => [*output_id].into(),
            Fusion::Constant { output_id, .. } => [*output_id].into(),
        }
    }

    /// Return a short identifier for the kind of fusion this is.
    pub fn name(&self) -> &str {
        match self {
            Fusion::Op(op) => op.fused_op.name(),
            Fusion::Identity { .. } => "identity",
            Fusion::Constant { .. } => "constant",
        }
    }
}

/// Reasons why a fusion was not created.
#[derive(Debug)]
pub enum FusionError {
    /// No match for the fusion's basic pattern was found at the graph node
    /// being tested.
    NoMatch,
    /// The fusion was not applied because it would have no effect.
    ///
    /// For example, a fusion that combines MatMul + Mul will have no effect
    /// if the scale is 1.
    NoEffect,
    /// The fusion's pattern matched, but a precondition failed.
    ///
    /// For example, the fusion might require a certain input to be constant
    /// or have a specific rank, and it does not.
    CheckFailed(&'static str),
}

/// Interface for graph visitors which match graph patterns and return fused
/// operations.
pub trait FusionVisitor {
    type State;

    /// Identifier for this fusion.
    fn name(&self) -> &str;

    /// Prepare for a graph traversal by creating pattern matchers or other
    /// required state.
    fn prepare(&self, graph: &Graph) -> Self::State;

    /// Visit an operator in the graph and potentially return a fusion for it.
    fn maybe_fuse(
        &self,
        state: &Self::State,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError>;
}

/// Define a fusion for an operator using a graph pattern.
///
/// This provides a simpler way to define fusions than implementing
/// [`FusionVisitor`]. A `PatternFusion` can be converted into a `FusionVisitor`
/// using [`into_visitor`](PatternFusion::into_visitor).
pub trait PatternFusion {
    /// The operator produced by this fusion.
    type Operator: Operator + Send + Sync;

    /// Return an identifier for this fusion.
    fn name(&self) -> &str;

    /// Return the graph pattern to match.
    fn pattern(&self) -> Pattern;

    /// Return the names of symbols in the pattern which will become inputs
    /// to the fused operator.
    fn inputs(&self) -> &[&str];

    /// Return the names of symbols in the pattern which are inputs to the
    /// unfused operator that become unused in the fused operator.
    ///
    /// The default implementation returns an empty slice, meaning that the
    /// fused operator must use all non-constant inputs specified by the pattern.
    fn unused_inputs(&self) -> &[&str] {
        &[]
    }

    /// Create a fused operator given a successful match for the pattern.
    ///
    /// This can fail if there are additional requirements which cannot be
    /// expressed in the pattern.
    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Result<Self::Operator, FusionError>;

    /// Wrap this fusion into a [`FusionVisitor`].
    fn into_visitor(self) -> impl FusionVisitor<State = Pattern>
    where
        Self: Sized + 'static,
    {
        PatternFusionVisitor::new(self)
    }
}

/// Wraps a [`PatternFusion`] to implement [`FusionVisitor`].
struct PatternFusionVisitor<PF: PatternFusion + 'static> {
    fusion: PF,
    pattern: Pattern,
}

impl<PF: PatternFusion + 'static> PatternFusionVisitor<PF> {
    fn new(fusion: PF) -> Self {
        let pattern = fusion.pattern();

        // Sanity check: Make sure input symbols appear in the pattern.
        for input in fusion.inputs() {
            assert!(
                pattern.contains_symbol(input),
                "pattern does not contain symbol \"{}\"",
                input
            );
        }

        Self { fusion, pattern }
    }
}

impl<PF: PatternFusion + 'static> FusionVisitor for PatternFusionVisitor<PF> {
    type State = Pattern;

    fn name(&self) -> &str {
        self.fusion.name()
    }

    fn prepare(&self, _: &Graph) -> Pattern {
        // Patterns are ref-counted and don't depend on the pattern, so we can
        // create them once and do a cheap clone before each graph traversal.
        self.pattern.clone()
    }

    fn maybe_fuse(
        &self,
        pattern: &Pattern,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let pat_match = pattern
            .test(op_node_id, graph)
            .ok_or(FusionError::NoMatch)?;
        let input_ids: Vec<_> = self
            .fusion
            .inputs()
            .iter()
            .map(|name| pat_match.node_id(name))
            .collect();
        let unused_input_ids: Vec<_> = self
            .fusion
            .unused_inputs()
            .iter()
            .map(|name| pat_match.node_id(name))
            .collect();
        let fused_op = self.fusion.maybe_fuse(&pat_match, graph)?;
        let fusion = Fusion::from_op(
            op_node.name(),
            Arc::new(fused_op),
            &input_ids,
            op_node.output_ids(),
            &unused_input_ids,
        );
        Ok(fusion)
    }
}

/// Additional graph querying methods used in fusions.
trait GraphQuery {
    /// Extract the scalar value from a constant node.
    fn get_scalar<T>(&self, node_id: NodeId) -> Option<T>
    where
        Constant: TypedConstant<T>;

    /// Extract the vector value from a constant node.
    fn get_vector<T>(&self, node_id: NodeId) -> Option<&[T]>
    where
        Constant: TypedConstant<T>;

    /// Extract the operator from an operator node.
    fn get_operator<Op: 'static>(&self, node_id: NodeId) -> Option<&Op>;

    /// Return the rank of a constant or value node.
    ///
    /// This may be `None` if this is a value node which does not have shape
    /// information.
    fn get_rank(&self, node_id: NodeId) -> Option<usize>;
}

impl GraphQuery for Graph {
    fn get_scalar<T>(&self, node_id: NodeId) -> Option<T>
    where
        Constant: TypedConstant<T>,
    {
        self.get_node(node_id).and_then(|node| match node {
            Node::Constant(const_node) => const_node.as_scalar(),
            _ => None,
        })
    }

    fn get_vector<T>(&self, node_id: NodeId) -> Option<&[T]>
    where
        Constant: TypedConstant<T>,
    {
        self.get_node(node_id).and_then(|node| match node {
            Node::Constant(const_node) => const_node.as_vector(),
            _ => None,
        })
    }

    fn get_operator<Op: 'static>(&self, node_id: NodeId) -> Option<&Op> {
        self.get_node(node_id).and_then(|node| match node {
            Node::Operator(op_node) => op_node.operator().downcast_ref(),
            _ => None,
        })
    }

    fn get_rank(&self, node_id: NodeId) -> Option<usize> {
        self.get_node(node_id).and_then(|node| match node {
            Node::Constant(const_node) => Some(const_node.ndim()),
            Node::Value(val) => val.ndim(),
            _ => None,
        })
    }
}

pub struct ReciprocalFusion {}

impl PatternFusion for ReciprocalFusion {
    type Operator = Reciprocal;

    fn name(&self) -> &str {
        "ReciprocalFusion"
    }

    fn pattern(&self) -> Pattern {
        1. / Pattern::symbol("x")
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Result<Reciprocal, FusionError> {
        Ok(Reciprocal {})
    }
}

/// Convert `ReduceMean(X, axes)` to `ReduceMean<axes>(X)` where `axes` is a constant.
pub struct ReduceMeanAxesFusion {}

impl PatternFusion for ReduceMeanAxesFusion {
    type Operator = ReduceMean;

    fn name(&self) -> &str {
        "ReduceMeanAxesFusion"
    }

    fn pattern(&self) -> Pattern {
        let axes = Pattern::const_symbol("axes");
        let x = Pattern::symbol("x");
        Pattern::binary_op("ReduceMean", x, axes).with_name("mean")
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Result<ReduceMean, FusionError> {
        let mean_op = g
            .get_operator::<ReduceMean>(pat_match.node_id("mean").unwrap())
            .ok_or(FusionError::CheckFailed("wrong op"))?;
        let axes = g
            .get_vector::<i32>(pat_match.node_id("axes").unwrap())
            .ok_or(FusionError::CheckFailed("axes not a vector"))?;

        Ok(ReduceMean {
            axes: Some(axes.to_vec()),
            keep_dims: mean_op.keep_dims,
            noop_with_empty_axes: false,
        })
    }
}

pub struct GeluFusion {}

impl PatternFusion for GeluFusion {
    type Operator = Gelu;

    fn name(&self) -> &str {
        "GeluFusion"
    }

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        let sqrt_2 = (2.0f32).sqrt();
        let x_scaled = Pattern::any_of([x.clone() / sqrt_2, x.clone() * (1. / sqrt_2)].into());
        x.clone() * (Pattern::unary_op("Erf", x_scaled) + 1.0) * 0.5
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Result<Gelu, FusionError> {
        Ok(Gelu { approximate: false })
    }
}

/// Remove operations such as `Identity(x)` or `Add(x, 0)` which have no effect.
pub struct IdentityFusion {}

impl FusionVisitor for IdentityFusion {
    type State = Pattern;

    fn name(&self) -> &str {
        "IdentityFusion"
    }

    fn prepare(&self, _: &Graph) -> Pattern {
        let x = Pattern::symbol("x");

        // Use exact constants here so that we don't match expressions like
        // `x + epsilon` which appear in eg. root mean square operations.
        let zero = Pattern::exact_constant(0.);
        let one = Pattern::exact_constant(1.0);

        Pattern::any_of(
            [
                // Unary op identities
                Pattern::unary_op("Identity", x.clone()),
                // Binary op identities
                x.clone() + zero.clone(),
                x.clone() - zero.clone(),
                x.clone() * one.clone(),
                x.clone() / one.clone(),
            ]
            .into(),
        )
    }

    fn maybe_fuse(
        &self,
        pattern: &Pattern,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let pat_match = pattern
            .test(op_node_id, graph)
            .ok_or(FusionError::NoMatch)?;
        let input_id = pat_match.node_id("x").unwrap();
        let &[Some(output_id)] = op_node.output_ids() else {
            return Err(FusionError::CheckFailed("wrong output count"));
        };
        Ok(Fusion::Identity {
            input_id,
            output_id,
        })
    }
}

/// Eliminate Cast operations where the output dtype is the same as the input
/// dtype.
pub struct CastElimination {}

impl FusionVisitor for CastElimination {
    type State = ();

    fn name(&self) -> &str {
        "CastElimination"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let to_dtype = op_node
            .operator()
            .downcast_ref::<Cast>()
            .map(|op| op.to)
            .ok_or(FusionError::NoMatch)?;

        let &[Some(input_id)] = op_node.input_ids() else {
            return Err(FusionError::CheckFailed("wrong input count"));
        };

        let input_dtype = graph
            .get_node(input_id)
            .and_then(|n| n.dtype())
            .ok_or(FusionError::CheckFailed("missing input dtype"))?;
        if input_dtype != ValueType::Tensor(to_dtype) {
            // This Cast op is not a no-op.
            return Err(FusionError::CheckFailed("input dtype != output type"));
        }

        let &[Some(output_id)] = op_node.output_ids() else {
            return Err(FusionError::CheckFailed("wrong output count"));
        };

        Ok(Fusion::Identity {
            input_id,
            output_id,
        })
    }
}

pub struct ApproxGeluFusion {}

impl PatternFusion for ApproxGeluFusion {
    type Operator = Gelu;

    fn name(&self) -> &str {
        "ApproxGeluFusion"
    }

    fn pattern(&self) -> Pattern {
        // Pattern for tanh approximate of gelu. See
        // https://onnx.ai/onnx/operators/onnx__Gelu.html.
        let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let x = Pattern::symbol("x");
        x.clone()
            * 0.5
            * (1.
                + Pattern::unary_op(
                    "Tanh",
                    sqrt_2_pi * (x.clone() + Pattern::binary_op("Pow", x.clone(), 3.0) * 0.044715),
                ))
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Result<Gelu, FusionError> {
        Ok(Gelu { approximate: true })
    }
}

pub struct SiluFusion {}

impl PatternFusion for SiluFusion {
    type Operator = Silu;

    fn name(&self) -> &str {
        "SiluFusion"
    }

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        x.clone() * Pattern::unary_op("Sigmoid", x.clone())
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Result<Silu, FusionError> {
        Ok(Silu {})
    }
}

pub struct SwishFusion {}

impl PatternFusion for SwishFusion {
    type Operator = Swish;

    fn name(&self) -> &str {
        "SwishFusion"
    }

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        let alpha = Pattern::const_symbol("alpha");
        x.clone() * Pattern::unary_op("Sigmoid", alpha * x.clone())
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Result<Swish, FusionError> {
        let alpha_input = pat_match.node_id("alpha").expect("missing symbol");
        let alpha = g
            .get_scalar(alpha_input)
            .ok_or(FusionError::CheckFailed("alpha not a scalar"))?;
        Ok(Swish { alpha })
    }
}

trait OperatorAxis {
    /// Get the axis to which an operator is applied.
    ///
    /// Some operators (eg. ReduceMean) support dynamic `axes` inputs. In this
    /// case we rely on canonicalization passes to convert constant values to
    /// attributes first.
    fn get_axis(&self) -> Option<i32>;
}

impl OperatorAxis for ReduceMean {
    fn get_axis(&self) -> Option<i32> {
        match self.axes.as_deref() {
            Some([axis]) => Some(*axis),
            _ => None,
        }
    }
}

impl OperatorAxis for Softmax {
    fn get_axis(&self) -> Option<i32> {
        Some(self.axis as i32)
    }
}

/// Test if an operator is applied to the last axis of its input.
fn op_applied_to_last_axis<Op: OperatorAxis + 'static>(graph: &Graph, node_id: NodeId) -> bool {
    let Some(op_node) = graph.get_node(node_id).and_then(|n| n.as_operator()) else {
        return false;
    };

    let Some(axis) = op_node
        .operator()
        .downcast_ref::<Op>()
        .and_then(|op| op.get_axis())
    else {
        return false;
    };

    if axis == -1 {
        return true;
    }

    // For positive values we require the ReduceMean input to have shape information.
    let input_last_axis = op_node
        .input_ids()
        .first()
        .copied()
        .flatten()
        .and_then(|id| graph.get_rank(id))
        .and_then(|ndim| ndim.checked_sub(1))
        .map(|axis| axis as i32);

    input_last_axis == Some(axis)
}

/// Identify and fuse common patterns for `LayerNormalization(X)`.
pub struct LayerNormalizationFusion {}

impl PatternFusion for LayerNormalizationFusion {
    type Operator = LayerNormalization;

    fn name(&self) -> &str {
        "LayerNormalizationFusion"
    }

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");

        // First step: Center values
        let center_pat =
            x.clone() - Pattern::unary_op("ReduceMean", x.clone()).with_name("center_mean");

        // Middle step: Normalize variance
        let epsilon = Pattern::const_symbol("epsilon");
        let normalize_variance_pat = center_pat.clone()
            / Pattern::unary_op(
                "Sqrt",
                epsilon
                    + Pattern::unary_op(
                        "ReduceMean",
                        Pattern::binary_op("Pow", center_pat.clone(), 2.0),
                    )
                    .with_name("norm_mean"),
            );

        // Final step: Scale, and optionally shift, the normalized values
        let bias = Pattern::const_symbol("bias");
        let scale = Pattern::const_symbol("scale");
        let shift_scale_pat = (normalize_variance_pat.clone() * scale.clone()) + bias;
        let scale_pat = normalize_variance_pat.clone() * scale;

        Pattern::any_of([shift_scale_pat, scale_pat].into())
    }

    fn inputs(&self) -> &[&str] {
        &["x", "scale", "bias"]
    }

    fn maybe_fuse(
        &self,
        pat_match: &Match,
        graph: &Graph,
    ) -> Result<LayerNormalization, FusionError> {
        let norm_mean = pat_match.node_id("norm_mean").unwrap();
        if !op_applied_to_last_axis::<ReduceMean>(graph, norm_mean) {
            // The LayerNormalization operator supports taking the mean over
            // multiple trailing axes. However this fusion only supports the
            // common case of taking the mean over one axis.
            return Err(FusionError::CheckFailed("not applied to last axis"));
        }

        let center_mean = pat_match.node_id("center_mean").unwrap();
        if !op_applied_to_last_axis::<ReduceMean>(graph, center_mean) {
            return Err(FusionError::CheckFailed("not applied to last axis"));
        }

        let epsilon_input = pat_match.node_id("epsilon").unwrap();
        let epsilon = graph
            .get_scalar(epsilon_input)
            .ok_or(FusionError::CheckFailed("epsilon not a scalar"))?;

        Ok(LayerNormalization {
            axis: -1,
            epsilon: Some(epsilon),
        })
    }
}

/// Fuse `RMSNormalization(x)`.
///
/// See https://pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html.
pub struct RMSNormalizationFusion {}

impl PatternFusion for RMSNormalizationFusion {
    type Operator = RMSNormalization;

    fn name(&self) -> &str {
        "RMSNormalizationFusion"
    }

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        let scale = Pattern::const_symbol("scale");
        let epsilon = Pattern::const_symbol("epsilon");

        // The scaling of the input is canonically written as `x / (sqrt(rms) + epsilon)`.
        //
        // Here we test for `x * 1/(sqrt(rms) + epsilon)` because that is the
        // observed pattern in models like T5. Ideally we would recognize both.
        x.clone()
            * Pattern::unary_op(
                "Reciprocal",
                Pattern::unary_op(
                    "Sqrt",
                    epsilon
                        + Pattern::unary_op(
                            "ReduceMean",
                            Pattern::binary_op("Pow", x.clone(), 2.0),
                        )
                        .with_name("norm_mean"),
                ),
            )
            * scale
    }

    fn inputs(&self) -> &[&str] {
        &["x", "scale"]
    }

    fn maybe_fuse(&self, rms_match: &Match, graph: &Graph) -> Result<Self::Operator, FusionError> {
        let epsilon_input = rms_match.node_id("epsilon").unwrap();
        let epsilon = graph
            .get_scalar(epsilon_input)
            .ok_or(FusionError::CheckFailed("epsilon not a scalar"))?;
        let norm_mean = rms_match.node_id("norm_mean").unwrap();

        if !op_applied_to_last_axis::<ReduceMean>(graph, norm_mean) {
            return Err(FusionError::CheckFailed("not applied to last axis"));
        }

        Ok(RMSNormalization {
            axis: -1,
            epsilon: Some(epsilon),
        })
    }
}

/// Fuse `Add(MatMul(a, b), bias)` into `FusedMatMul(a, b, bias)`.
pub struct MatMulAddFusion {}

impl PatternFusion for MatMulAddFusion {
    type Operator = FusedMatMul;

    fn name(&self) -> &str {
        "MatMulAddFusion"
    }

    fn pattern(&self) -> Pattern {
        let a = Pattern::symbol("a");
        let b = Pattern::symbol("b");
        let bias = Pattern::const_symbol("bias");
        Pattern::binary_op(
            "Add",
            Pattern::binary_op("MatMul", a.clone(), b.clone()),
            bias.clone(),
        )
    }

    fn inputs(&self) -> &[&str] {
        &["a", "b", "bias"]
    }

    fn maybe_fuse(
        &self,
        matmul_add_match: &Match,
        graph: &Graph,
    ) -> Result<FusedMatMul, FusionError> {
        let bias_input = matmul_add_match.node_id("bias").unwrap();
        let is_bias_a_vector = match graph.get_node(bias_input) {
            Some(Node::Constant(const_node)) => const_node.shape().len() == 1,
            _ => false,
        };

        if !is_bias_a_vector {
            return Err(FusionError::CheckFailed("bias not a vector"));
        }

        Ok(FusedMatMul { alpha: None })
    }
}

/// Fuse multiplication or division of MatMul inputs and outputs by
/// scalars.
///
/// A subgraph of the form `Mul(MatMul(Mul(X, c), Mul(Y, d)), e)` where c, d
/// and e are constants can be rewritten as `FusedMatMul(X, Y, alpha=c * d *
/// e)`. Each `Mul(X, c)` can also be expressed as `Div(X, 1/c)`.
pub struct MatMulScaleFusion {}

impl FusionVisitor for MatMulScaleFusion {
    type State = ();

    fn name(&self) -> &str {
        "MatMulScaleFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let binary_op_input_ids = |op: &OperatorNode| -> Option<[NodeId; 2]> {
            match op.input_ids() {
                [Some(lhs_id), Some(rhs_id)] => Some([*lhs_id, *rhs_id]),
                _ => None,
            }
        };

        // Test if `op_node` is a Mul or Div node with one constant scalar
        // input and one non-constant input. If so, returns the constant scalar
        // which the node multiplies the other input by and the ID of the other
        // input.
        let get_scale_factor = |graph: &Graph, op_node: &OperatorNode| -> Option<(f32, NodeId)> {
            let op_type = op_node.operator().name();
            if !["Mul", "Div"].contains(&op_type) {
                return None;
            }

            let [lhs, rhs] = binary_op_input_ids(op_node)?;
            let lhs_scalar = graph.get_scalar(lhs);
            let rhs_scalar = graph.get_scalar(rhs);

            match op_type {
                "Mul" => match (lhs_scalar, rhs_scalar) {
                    (Some(lhs_scale), None) => Some((lhs_scale, rhs)),
                    (None, Some(rhs_scale)) => Some((rhs_scale, lhs)),
                    _ => None,
                },
                "Div" => match (lhs_scalar, rhs_scalar) {
                    (None, Some(rhs_scale)) => Some((1. / rhs_scale, lhs)),
                    _ => None,
                },
                _ => None,
            }
        };

        // Accumulated scale factor from scalings applied to MatMul inputs
        // and outputs.
        let mut alpha = 1.0;

        // Check if this is a Mul/Div node scaling the output of a MatMul.
        let matmul_node = if ["Mul", "Div"].contains(&op_node.operator().name()) {
            let (output_scale, scale_input) =
                get_scale_factor(graph, op_node).ok_or(FusionError::NoMatch)?;
            alpha *= output_scale;
            let (_, scale_input_op) = graph
                .get_source_node(scale_input)
                .ok_or(FusionError::NoMatch)?;
            scale_input_op
        } else {
            op_node
        };

        if matmul_node.operator().name() != "MatMul" {
            return Err(FusionError::NoMatch);
        }

        let [matmul_lhs, matmul_rhs] =
            binary_op_input_ids(matmul_node).ok_or(FusionError::NoMatch)?;
        let lhs_input = if let Some((_, lhs_source_op)) = graph.get_source_node(matmul_lhs) {
            let (lhs_scale, lhs_input) =
                get_scale_factor(graph, lhs_source_op).unwrap_or((1.0, matmul_lhs));
            alpha *= lhs_scale;
            lhs_input
        } else {
            // MatMul LHS is not computed by an upstream operator.
            matmul_lhs
        };

        let rhs_input = if let Some((_, rhs_source_op)) = graph.get_source_node(matmul_rhs) {
            let (rhs_scale, rhs_input) =
                get_scale_factor(graph, rhs_source_op).unwrap_or((1.0, matmul_rhs));
            alpha *= rhs_scale;
            rhs_input
        } else {
            // MatMul RHS is not computed by an upstream operator.
            matmul_rhs
        };

        if alpha == 1.0 {
            // Scale factor of 1 has no effect.
            return Err(FusionError::NoEffect);
        }

        Ok(Fusion::from_op(
            matmul_node.name(),
            Arc::new(FusedMatMul { alpha: Some(alpha) }),
            &[Some(lhs_input), Some(rhs_input)],
            op_node.output_ids(),
            &[],
        ))
    }
}

pub struct MatMulIntegerToFloatFusion {}

impl PatternFusion for MatMulIntegerToFloatFusion {
    type Operator = MatMulIntegerToFloat;

    fn name(&self) -> &str {
        "MatMulIntegerToFloatFusion"
    }

    fn pattern(&self) -> Pattern {
        let scale = Pattern::symbol("scale");
        let a = Pattern::symbol("a");
        let b = Pattern::symbol("b");
        let a_zero = Pattern::symbol("a_zero");
        let b_zero = Pattern::symbol("b_zero");

        Pattern::unary_op(
            "Cast",
            Pattern::operator("MatMulInteger", [a, b, a_zero, b_zero]),
        )
        .with_name("cast")
            * scale
    }

    fn inputs(&self) -> &[&str] {
        &["a", "b", "a_zero", "b_zero", "scale"]
    }

    fn maybe_fuse(&self, pat_match: &Match, graph: &Graph) -> Result<Self::Operator, FusionError> {
        let scale = pat_match.node_id("scale").unwrap();
        let scale_shape = graph
            .get_node(scale)
            .ok_or(FusionError::NoMatch)?
            .shape()
            .ok_or(FusionError::CheckFailed("unknown scale shape"))?;

        // Fusion supports scalar or vector scale which can be broadcast to
        // MatMulInteger output shape.
        if scale_shape.len() > 1 {
            return Err(FusionError::NoMatch);
        }

        Ok(MatMulIntegerToFloat::default())
    }
}

pub struct ConvIntegerToFloatFusion {}

impl PatternFusion for ConvIntegerToFloatFusion {
    type Operator = ConvIntegerToFloat;

    fn name(&self) -> &str {
        "ConvIntegerToFloatFusion"
    }

    fn pattern(&self) -> Pattern {
        let scale = Pattern::symbol("scale");
        let x = Pattern::symbol("x");
        let w = Pattern::symbol("w");
        let x_zero = Pattern::symbol("x_zero");
        let w_zero = Pattern::symbol("w_zero");

        Pattern::unary_op(
            "Cast",
            Pattern::operator("ConvInteger", [x, w, x_zero, w_zero]).with_name("conv"),
        ) * scale
    }

    fn inputs(&self) -> &[&str] {
        &["x", "w", "x_zero", "w_zero", "scale"]
    }

    fn maybe_fuse(&self, pat_match: &Match, graph: &Graph) -> Result<Self::Operator, FusionError> {
        let scale = pat_match.node_id("scale").unwrap();
        let scale_shape = graph
            .get_node(scale)
            .ok_or(FusionError::NoMatch)?
            .shape()
            .ok_or(FusionError::CheckFailed("unknown scale shape"))?;

        let is_scalar = matches!(scale_shape.as_ref(), [] | [Dimension::Fixed(1)]);
        if !is_scalar {
            return Err(FusionError::NoMatch);
        }

        let conv_id = pat_match.node_id("conv").unwrap();
        let conv: &ConvInteger = graph
            .get_operator(conv_id)
            .ok_or(FusionError::CheckFailed("expected ConvInteger operator"))?;

        Ok(ConvIntegerToFloat::new(conv.clone()))
    }
}

/// Quantize `weight`, with shape `[O, ...]`, to int8 with a symmetric scale
/// per output channel.
///
/// Returns the quantized weights and scales.
fn quantize_per_output_channel(weight: Tensor<f32>) -> (Tensor<i8>, Vec<f32>) {
    let weight = weight.to_contiguous();
    let out_chans = weight.size(0);
    let row_len = (weight.len() / out_chans.max(1)).max(1);
    let mut quantized = Tensor::<i8>::uninit(weight.shape());

    // Quantize output channels in parallel, as weights can be large.
    let scales = quantized
        .data_mut()
        .unwrap()
        .par_chunks_mut(row_len)
        .zip(weight.data().par_chunks(row_len))
        .map(|(q_row, row)| {
            let max_abs = row.iter().fold(0f32, |m, x| m.max(x.abs()));
            let scale = if max_abs > 0. { max_abs / 127. } else { 1. };
            for (q, &x) in q_row.iter_mut().zip(row) {
                q.write((x / scale).round_ties_even() as i8);
            }
            scale
        })
        .collect();

    // Safety: Each row of `quantized` was initialized.
    (unsafe { quantized.assume_init() }, scales)
}

/// Fuse `Conv(X, DequantizeLinear(W, scale, zero_point), B)`, where `W` is an
/// int8 constant, into `ConvDynamicQuant(X, W, scale, B)`.
///
/// The `DequantizeLinear` may be followed by a `Cast` to f32. If `W` is not
/// quantized per output channel (eg. it uses blocked quantization, as is
/// common for int4 weights), it is requantized to int8 with a scale per output
/// channel.
///
/// The fused operator quantizes activations dynamically and uses int8 matrix
/// multiplication. This changes numerics, so it is only applied to
/// operators for which `rules` enable it.
///
/// This must run before constant propagation, which would replace the
/// `DequantizeLinear` with dequantized weights.
pub struct ConvDynamicQuantFusion {
    pub rules: DynamicQuantOptions,
}

impl FusionVisitor for ConvDynamicQuantFusion {
    type State = ();

    fn name(&self) -> &str {
        "ConvDynamicQuantFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let Some(conv) = op_node.operator().downcast_ref::<Conv>() else {
            return Err(FusionError::NoMatch);
        };
        let (input, weight, bias) = match op_node.input_ids() {
            [Some(x), Some(w)] => (*x, *w, None),
            [Some(x), Some(w), b] => (*x, *w, *b),
            _ => return Err(FusionError::NoMatch),
        };

        let dq_node = dequantize_linear_source(graph, weight).ok_or(FusionError::NoMatch)?;
        let dq = dq_node
            .operator()
            .downcast_ref::<DequantizeLinear>()
            .unwrap();
        if conv.groups != 1 {
            return Err(FusionError::CheckFailed("groups != 1"));
        }

        let (weight_q, scale, zero_point) = match dq_node.input_ids() {
            [Some(w), Some(s)] => (*w, *s, None),
            [Some(w), Some(s), zp] => (*w, *s, *zp),
            _ => return Err(FusionError::NoMatch),
        };

        let Some(Node::Constant(weight_const)) = graph.get_node(weight_q) else {
            return Err(FusionError::CheckFailed("weight is not a constant"));
        };
        let Some(weight_view) = TypedConstant::<i8>::as_typed_view(weight_const) else {
            return Err(FusionError::CheckFailed("weight is not int8"));
        };
        if !self
            .rules
            .enables(DynamicQuantOp::Conv, op_node.name(), weight_const.name())
        {
            return Err(FusionError::CheckFailed("operator not selected"));
        }
        let out_chans = weight_const.shape().first().copied().unwrap_or(0);
        let weight_rank = weight_const.shape().len() as isize;

        let Some(Node::Constant(scale_const)) = graph.get_node(scale) else {
            return Err(FusionError::CheckFailed("scale is not a constant"));
        };
        let Some(scale_view) = TypedConstant::<f32>::as_typed_view(scale_const) else {
            return Err(FusionError::CheckFailed("scale is not f32"));
        };

        // Zero point must be zero.
        if let Some(zp) = zero_point {
            let Some(Node::Constant(zp_const)) = graph.get_node(zp) else {
                return Err(FusionError::CheckFailed("zero point is not a constant"));
            };
            let all_zero = TypedConstant::<i8>::as_typed_view(zp_const)
                .is_some_and(|zp| zp.iter().all(|&z| z == 0));
            if !all_zero {
                return Err(FusionError::CheckFailed("zero point is not zero"));
            }
        }

        let per_channel_axis = dq.axis == 0 || dq.axis == -weight_rank;
        let per_channel_scale =
            TypedConstant::<f32>::as_vector(scale_const).is_some_and(|s| s.len() == out_chans);
        let per_tensor_scale = TypedConstant::<f32>::as_scalar(scale_const);

        // Use the weights as-is if they are quantized per tensor or per
        // output channel. Otherwise (eg. for blocked quantization), dequantize
        // the weights and requantize them per output channel.
        let (weight_input, scale_input, new_constant_inputs, unused_input_ids) =
            if dq.block_size == 0 && per_channel_axis && per_channel_scale {
                (Some(weight_q), Some(scale), Vec::new(), vec![zero_point])
            } else if dq.block_size == 0
                && let Some(scale_val) = per_tensor_scale
            {
                (
                    Some(weight_q),
                    None,
                    vec![(2, vector_constant(vec![scale_val; out_chans]))],
                    vec![zero_point, Some(scale)],
                )
            } else {
                let pool = BufferPool::new();
                let weight_f32 =
                    dequantize_linear(&pool, weight_view, scale_view, None, dq.axis, dq.block_size)
                        .map_err(|_| FusionError::CheckFailed("failed to dequantize weights"))?;
                let (weight_requant, scale_requant) = quantize_per_output_channel(weight_f32);
                let name = weight_const.name().map(|name| format!("{name}_requant"));
                (
                    None,
                    None,
                    vec![
                        (1, Constant::new(name.as_deref(), weight_requant.into_arc())),
                        (2, vector_constant(scale_requant)),
                    ],
                    vec![Some(weight_q), Some(scale), zero_point],
                )
            };

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: Arc::new(ConvDynamicQuant {
                padding: conv.padding.clone(),
                strides: conv.strides.clone(),
                dilations: conv.dilations.clone(),
            }),
            input_ids: vec![Some(input), weight_input, scale_input, bias],
            new_constant_inputs,
            output_ids: op_node.output_ids().to_vec(),
            unused_input_ids,
        }))
    }
}

/// Fuse `ConvTranspose(X, DequantizeLinear(W, scale, zero_point), B)`, where
/// `W` is an int8 constant, into `ConvTransposeDynamicQuant(X, W', scale', B)`.
///
/// The `DequantizeLinear` may be followed by a `Cast` to f32. The weights are
/// requantized to int8 with a scale for each output channel and kernel
/// position, which is what the matrix multiplication in the fused operator
/// requires.
///
/// The fused operator quantizes activations dynamically and uses int8 matrix
/// multiplication. This changes numerics, so it is only applied to
/// operators for which `rules` enable it.
///
/// This must run before constant propagation, which would replace the
/// `DequantizeLinear` with dequantized weights.
pub struct ConvTransposeDynamicQuantFusion {
    pub rules: DynamicQuantOptions,
}

impl FusionVisitor for ConvTransposeDynamicQuantFusion {
    type State = ();

    fn name(&self) -> &str {
        "ConvTransposeDynamicQuantFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let Some(conv) = op_node.operator().downcast_ref::<ConvTranspose>() else {
            return Err(FusionError::NoMatch);
        };
        let (input, weight, bias) = match op_node.input_ids() {
            [Some(x), Some(w)] => (*x, *w, None),
            [Some(x), Some(w), b] => (*x, *w, *b),
            _ => return Err(FusionError::NoMatch),
        };
        if conv.groups != 1 {
            return Err(FusionError::CheckFailed("groups != 1"));
        }

        let dq_node = dequantize_linear_source(graph, weight).ok_or(FusionError::NoMatch)?;
        let dq = dq_node
            .operator()
            .downcast_ref::<DequantizeLinear>()
            .unwrap();
        let (weight_q, scale, zero_point) = match dq_node.input_ids() {
            [Some(w), Some(s)] => (*w, *s, None),
            [Some(w), Some(s), zp] => (*w, *s, *zp),
            _ => return Err(FusionError::NoMatch),
        };

        let Some(Node::Constant(weight_const)) = graph.get_node(weight_q) else {
            return Err(FusionError::CheckFailed("weight is not a constant"));
        };
        let Some(weight_view) = TypedConstant::<i8>::as_typed_view(weight_const) else {
            return Err(FusionError::CheckFailed("weight is not int8"));
        };
        if !self.rules.enables(
            DynamicQuantOp::ConvTranspose,
            op_node.name(),
            weight_const.name(),
        ) {
            return Err(FusionError::CheckFailed("operator not selected"));
        }
        let Some(Node::Constant(scale_const)) = graph.get_node(scale) else {
            return Err(FusionError::CheckFailed("scale is not a constant"));
        };
        let Some(scale_view) = TypedConstant::<f32>::as_typed_view(scale_const) else {
            return Err(FusionError::CheckFailed("scale is not f32"));
        };
        let zero_point_view = match zero_point.map(|zp| graph.get_node(zp)) {
            None => None,
            Some(Some(Node::Constant(zp))) => Some(
                TypedConstant::<i8>::as_typed_view(zp)
                    .ok_or(FusionError::CheckFailed("zero point is not int8"))?,
            ),
            Some(_) => return Err(FusionError::CheckFailed("zero point is not a constant")),
        };

        // Requantize the `[C, O, ...kernel]` weights with a scale for each
        // column of the `[C, O * prod(kernel)]` matrix.
        let pool = BufferPool::new();
        let weight_f32 = dequantize_linear(
            &pool,
            weight_view,
            scale_view,
            zero_point_view,
            dq.axis,
            dq.block_size,
        )
        .map_err(|_| FusionError::CheckFailed("failed to dequantize weights"))?;
        let shape = weight_f32.shape().to_vec();
        let in_c = shape.first().copied().unwrap_or(0);
        let weight_mat = weight_f32.reshaped([in_c, weight_f32.len() / in_c.max(1)]);
        let (weight_t, scale_requant) =
            quantize_per_output_channel(weight_mat.transposed().to_tensor().into_dyn());
        let mut weight_requant = weight_t.transposed().to_tensor();
        weight_requant.reshape(&shape);
        let name = weight_const.name().map(|name| format!("{name}_requant"));

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: Arc::new(ConvTransposeDynamicQuant {
                dilations: conv.dilations.clone(),
                padding: conv.padding.clone(),
                strides: conv.strides.clone(),
                output_padding: conv.output_padding.clone(),
            }),
            input_ids: vec![Some(input), None, None, bias],
            new_constant_inputs: vec![
                (1, Constant::new(name.as_deref(), weight_requant.into_arc())),
                (2, vector_constant(scale_requant)),
            ],
            output_ids: op_node.output_ids().to_vec(),
            unused_input_ids: vec![Some(weight_q), Some(scale), zero_point],
        }))
    }
}

/// Fuse `MatMul(A, DequantizeLinear(W, scale, zero_point))`, where `W` is an
/// int8 constant matrix, into `MatMulDynamicQuant(A, W, scale)`.
///
/// The `DequantizeLinear` may be followed by a `Cast` to f32. If `W` is not
/// quantized per column, it is requantized to int8 with a scale per column.
///
/// The fused operator quantizes each row of `A` dynamically and uses int8
/// matrix multiplication. This changes numerics, so it is only applied to
/// operators for which `rules` enable it.
///
/// This must run before constant propagation, which would replace the
/// `DequantizeLinear` with dequantized weights.
pub struct MatMulDynamicQuantFusion {
    pub rules: DynamicQuantOptions,
}

impl FusionVisitor for MatMulDynamicQuantFusion {
    type State = ();

    fn name(&self) -> &str {
        "MatMulDynamicQuantFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        if op_node.operator().downcast_ref::<MatMul>().is_none() {
            return Err(FusionError::NoMatch);
        }
        let &[Some(input), Some(weight)] = op_node.input_ids() else {
            return Err(FusionError::NoMatch);
        };

        let dq_node = dequantize_linear_source(graph, weight).ok_or(FusionError::NoMatch)?;
        let dq = dq_node
            .operator()
            .downcast_ref::<DequantizeLinear>()
            .unwrap();
        let (weight_q, scale, zero_point) = match dq_node.input_ids() {
            [Some(w), Some(s)] => (*w, *s, None),
            [Some(w), Some(s), zp] => (*w, *s, *zp),
            _ => return Err(FusionError::NoMatch),
        };

        let Some(Node::Constant(weight_const)) = graph.get_node(weight_q) else {
            return Err(FusionError::CheckFailed("weight is not a constant"));
        };
        let Some(weight_view) = TypedConstant::<i8>::as_typed_view(weight_const) else {
            return Err(FusionError::CheckFailed("weight is not int8"));
        };
        if weight_view.ndim() != 2 {
            return Err(FusionError::CheckFailed("weight is not a matrix"));
        }
        if !self
            .rules
            .enables(DynamicQuantOp::MatMul, op_node.name(), weight_const.name())
        {
            return Err(FusionError::CheckFailed("operator not selected"));
        }
        let Some(Node::Constant(scale_const)) = graph.get_node(scale) else {
            return Err(FusionError::CheckFailed("scale is not a constant"));
        };
        let Some(scale_view) = TypedConstant::<f32>::as_typed_view(scale_const) else {
            return Err(FusionError::CheckFailed("scale is not f32"));
        };
        if let Some(zp) = zero_point {
            let all_zero = match graph.get_node(zp) {
                Some(Node::Constant(zp_const)) => TypedConstant::<i8>::as_typed_view(zp_const)
                    .is_some_and(|zp| zp.iter().all(|&z| z == 0)),
                _ => false,
            };
            if !all_zero {
                return Err(FusionError::CheckFailed("zero point is not zero"));
            }
        }

        let n_cols = weight_view.size(1);
        let per_column = dq.block_size == 0
            && (dq.axis == 1 || dq.axis == -1)
            && TypedConstant::<f32>::as_vector(scale_const).is_some_and(|s| s.len() == n_cols);

        // Use the weights as-is if they are quantized per column. Otherwise,
        // dequantize the weights and requantize them per column.
        let (weight_input, scale_input, new_constant_inputs) = if per_column {
            (Some(weight_q), Some(scale), Vec::new())
        } else {
            let (weight_requant, scale_requant) =
                requantize_per_column(weight_view, scale_view, dq)?;
            let name = weight_const.name().map(|name| format!("{name}_requant"));
            (
                None,
                None,
                vec![
                    (1, Constant::new(name.as_deref(), weight_requant.into_arc())),
                    (2, vector_constant(scale_requant)),
                ],
            )
        };

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: Arc::new(MatMulDynamicQuant {}),
            input_ids: vec![Some(input), weight_input, scale_input],
            new_constant_inputs,
            output_ids: op_node.output_ids().to_vec(),
            // The dequantized weights may be shared with other operators, so
            // only the `MatMul` is replaced. The dequantization operators are
            // removed after optimization if they become unused.
            unused_input_ids: vec![Some(weight)],
        }))
    }
}

/// Dequantize a `[K, N]` int8 weight matrix and requantize it to int8 with a
/// symmetric scale per column.
fn requantize_per_column(
    weight: TensorView<i8>,
    scale: TensorView<f32>,
    dq: &DequantizeLinear,
) -> Result<(Tensor<i8>, Vec<f32>), FusionError> {
    let pool = BufferPool::new();
    let weight_f32 = dequantize_linear(&pool, weight, scale, None, dq.axis, dq.block_size)
        .map_err(|_| FusionError::CheckFailed("failed to dequantize weights"))?;
    let (weight_t, scale) = quantize_per_output_channel(weight_f32.transposed().to_tensor());
    Ok((weight_t.transposed().to_tensor(), scale))
}

/// Requantize int8 weights of dynamically quantized `MatMul`s per column.
///
/// This replaces `DequantizeLinear(W, scale, zero_point)`, where `W` is an
/// int8 constant matrix which is not quantized per column (eg. because it
/// uses blocked quantization), with `DequantizeLinear(W', scale')`, where `W'`
/// is quantized per column. This is done only if the output is used, possibly
/// via a `Cast` to f32, only as the weights of `MatMul` operators for which
/// `rules` enable dynamic quantization.
///
/// This prepares weights for [`MatMulDynamicQuantFusion`]. Requantizing the
/// weights here, rather than in that fusion, means that weights shared by
/// multiple `MatMul`s are requantized and stored once.
pub struct MatMulWeightRequantizeFusion {
    pub rules: DynamicQuantOptions,
}

impl MatMulWeightRequantizeFusion {
    /// Return true if `value` is used only as the weights of `MatMul`
    /// operators which will be dynamically quantized, possibly via a `Cast`
    /// to f32.
    ///
    /// `weight_name` is the name of the quantized weights that `value` is
    /// computed from.
    fn used_only_as_quantized_matmul_weights(
        &self,
        graph: &Graph,
        value: NodeId,
        weight_name: Option<&str>,
        allow_cast: bool,
    ) -> bool {
        if graph.output_ids().contains(&value) {
            return false;
        }
        let Some(consumers) = graph.get_consumers(value) else {
            return false;
        };
        consumers.iter().all(|&consumer| {
            let Some(Node::Operator(op)) = graph.get_node(consumer) else {
                return false;
            };
            if op.operator().downcast_ref::<MatMul>().is_some() {
                return op.input_ids() == [op.input_ids()[0], Some(value)]
                    && self
                        .rules
                        .enables(DynamicQuantOp::MatMul, op.name(), weight_name);
            }
            match (op.operator().downcast_ref::<Cast>(), op.output_ids()) {
                (Some(cast), &[Some(cast_out)]) if allow_cast && cast.to == DataType::Float => {
                    self.used_only_as_quantized_matmul_weights(graph, cast_out, weight_name, false)
                }
                _ => false,
            }
        })
    }
}

impl FusionVisitor for MatMulWeightRequantizeFusion {
    type State = ();

    fn name(&self) -> &str {
        "MatMulWeightRequantizeFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let Some(dq) = op_node.operator().downcast_ref::<DequantizeLinear>() else {
            return Err(FusionError::NoMatch);
        };
        let (weight_q, scale, zero_point) = match op_node.input_ids() {
            [Some(w), Some(s)] => (*w, *s, None),
            [Some(w), Some(s), zp] => (*w, *s, *zp),
            _ => return Err(FusionError::NoMatch),
        };
        let &[Some(output)] = op_node.output_ids() else {
            return Err(FusionError::NoMatch);
        };

        let Some(Node::Constant(weight_const)) = graph.get_node(weight_q) else {
            return Err(FusionError::NoMatch);
        };
        let Some(weight_view) = TypedConstant::<i8>::as_typed_view(weight_const) else {
            return Err(FusionError::NoMatch);
        };
        if weight_view.ndim() != 2 {
            return Err(FusionError::NoMatch);
        }
        let Some(Node::Constant(scale_const)) = graph.get_node(scale) else {
            return Err(FusionError::NoMatch);
        };
        let Some(scale_view) = TypedConstant::<f32>::as_typed_view(scale_const) else {
            return Err(FusionError::NoMatch);
        };

        let n_cols = weight_view.size(1);
        let per_column = dq.block_size == 0
            && (dq.axis == 1 || dq.axis == -1)
            && scale_view.ndim() == 1
            && scale_view.len() == n_cols;
        if per_column {
            return Err(FusionError::NoEffect);
        }
        if let Some(zp) = zero_point {
            let all_zero = match graph.get_node(zp) {
                Some(Node::Constant(zp_const)) => TypedConstant::<i8>::as_typed_view(zp_const)
                    .is_some_and(|zp| zp.iter().all(|&z| z == 0)),
                _ => false,
            };
            if !all_zero {
                return Err(FusionError::CheckFailed("zero point is not zero"));
            }
        }
        if !self.used_only_as_quantized_matmul_weights(graph, output, weight_const.name(), true) {
            return Err(FusionError::CheckFailed(
                "not used only as weights of dynamically quantized MatMuls",
            ));
        }

        let (weight_requant, scale_requant) = requantize_per_column(weight_view, scale_view, dq)?;
        let name = weight_const.name().map(|name| format!("{name}_requant"));

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: Arc::new(DequantizeLinear {
                axis: 1,
                block_size: 0,
            }),
            input_ids: vec![None, None],
            new_constant_inputs: vec![
                (0, Constant::new(name.as_deref(), weight_requant.into_arc())),
                (1, vector_constant(scale_requant)),
            ],
            output_ids: op_node.output_ids().to_vec(),
            unused_input_ids: vec![Some(weight_q), Some(scale), zero_point],
        }))
    }
}

/// Fuse `Add(MatMulDynamicQuant(A, B, B_scale), bias)`, where `bias` is a
/// constant vector, into `MatMulDynamicQuant(A, B, B_scale, bias)`.
pub struct MatMulDynamicQuantAddFusion {}

impl FusionVisitor for MatMulDynamicQuantAddFusion {
    type State = Pattern;

    fn name(&self) -> &str {
        "MatMulDynamicQuantAddFusion"
    }

    fn prepare(&self, _: &Graph) -> Pattern {
        Pattern::operator(
            "MatMulDynamicQuant",
            [
                Pattern::symbol("a"),
                Pattern::const_symbol("b"),
                Pattern::const_symbol("b_scale"),
            ],
        ) + Pattern::const_symbol("bias")
    }

    fn maybe_fuse(
        &self,
        pattern: &Pattern,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let pat_match = pattern
            .test(op_node_id, graph)
            .ok_or(FusionError::NoMatch)?;
        let b = pat_match.node_id("b").unwrap();
        let bias = pat_match.node_id("bias").unwrap();

        let n_cols = match graph.get_node(b) {
            Some(Node::Constant(b)) => b.shape().last().copied(),
            _ => None,
        };
        let bias_len = graph.get_vector::<f32>(bias).map(|v| v.len());
        if n_cols.is_none() || bias_len != n_cols {
            return Err(FusionError::CheckFailed("bias is not a per-column vector"));
        }

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: Arc::new(MatMulDynamicQuant {}),
            input_ids: vec![
                pat_match.node_id("a"),
                Some(b),
                pat_match.node_id("b_scale"),
                Some(bias),
            ],
            new_constant_inputs: Vec::new(),
            output_ids: op_node.output_ids().to_vec(),
            unused_input_ids: Vec::new(),
        }))
    }
}

/// Fuse `Add(ConvDynamicQuant(X, W, W_scale, B), R)` into
/// `ConvDynamicQuant(X, W, W_scale, B, R)`.
///
/// This avoids a separate pass over the output to add residual connections,
/// if `R` has the same shape as the output.
pub struct ConvDynamicQuantAddFusion {}

impl FusionVisitor for ConvDynamicQuantAddFusion {
    type State = ();

    fn name(&self) -> &str {
        "ConvDynamicQuantAddFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        if op_node.operator().downcast_ref::<Add>().is_none() {
            return Err(FusionError::NoMatch);
        }
        let [Some(lhs), Some(rhs)] = op_node.input_ids() else {
            return Err(FusionError::NoMatch);
        };
        if lhs == rhs {
            return Err(FusionError::CheckFailed(
                "residual is the convolution output",
            ));
        }

        // Find the operand which is the output of a convolution without a
        // residual input.
        let conv_source = |id: NodeId| {
            let (_, src) = graph.get_source_node(id)?;
            let conv = src.operator().downcast_ref::<ConvDynamicQuant>()?;
            (src.input_ids().len() <= 4).then_some((conv, src))
        };
        let (residual, conv, conv_node) = if let Some((conv, src)) = conv_source(*lhs) {
            (*rhs, conv, src)
        } else if let Some((conv, src)) = conv_source(*rhs) {
            (*lhs, conv, src)
        } else {
            return Err(FusionError::NoMatch);
        };

        let mut input_ids = conv_node.input_ids().to_vec();
        input_ids.resize(4, None);
        input_ids.push(Some(residual));

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: Arc::new(conv.clone()),
            input_ids,
            new_constant_inputs: Vec::new(),
            output_ids: op_node.output_ids().to_vec(),
            unused_input_ids: Vec::new(),
        }))
    }
}

/// Return the `DequantizeLinear` operator which produces `value`, looking
/// through a `Cast` to f32 (used when the dequantized values are f16).
fn dequantize_linear_source(graph: &Graph, value: NodeId) -> Option<&OperatorNode> {
    let (_, mut src) = graph.get_source_node(value)?;
    if let Some(cast) = src.operator().downcast_ref::<Cast>() {
        if cast.to != DataType::Float {
            return None;
        }
        let cast_input = src.input_ids().first().copied().flatten()?;
        src = graph.get_source_node(cast_input)?.1;
    }
    src.operator()
        .downcast_ref::<DequantizeLinear>()
        .is_some()
        .then_some(src)
}

/// Fuses transpose operations on an operator's inputs.
///
/// A Transpose operator on its own will move data around in memory. In the
/// fused operator, the strides of the input tensor view are just permuted
/// before calling the underlying operator. This is cheaper provided that the
/// underlying operator can efficiently handle non-contiguous inputs.
pub struct TransposeFusion {}

impl FusionVisitor for TransposeFusion {
    type State = ();

    fn name(&self) -> &str {
        "TransposeFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        // Filter against a set of operators which are known to efficiently
        // handle transposed inputs.
        if ![
            // Operators which pack blocks of non-contiguous inputs before
            // computing with them.
            "MatMul",
            "FusedMatMul",
            // Operators which copy chunks of the input using methods that can
            // efficiently handle transposed layouts.
            "Concat",
            "Expand",
            "Slice",
            "Split",
        ]
        .contains(&op_node.operator().name())
        {
            return Err(FusionError::NoMatch);
        }

        // Check the operator's inputs to see if any come from the output of
        // an operation that we can fuse into this one.
        let mut fused_op = TransformInputsBuilder::new();
        let mut fused_inputs = Cow::Borrowed(op_node.input_ids());

        for (i, input) in op_node.input_ids().iter().enumerate() {
            // Get the operator that produced this input.
            let Some((_, source_node)) = input.and_then(|input| graph.get_source_node(input))
            else {
                continue;
            };

            // Check if the operator can be fused into this one.
            if let Some(transpose) = source_node.operator().downcast_ref::<Transpose>() {
                let &[transpose_input] = source_node.input_ids() else {
                    continue;
                };
                fused_op = fused_op.permute(i, transpose.perm.clone());
                fused_inputs.to_mut()[i] = transpose_input;
            }
        }

        if !fused_op.has_transforms() {
            return Err(FusionError::NoEffect);
        }

        Ok(Fusion::from_op(
            op_node.name(),
            Arc::new(fused_op.build(op_node.clone_operator())),
            &fused_inputs,
            op_node.output_ids(),
            &[],
        ))
    }
}

/// Fuse `Add(QK, M) -> Softmax` operations.
///
/// This is common in attention operations where QK is the query-key product and
/// M is a mask or score matrix.
pub struct AddSoftmaxFusion {}

impl PatternFusion for AddSoftmaxFusion {
    type Operator = AddSoftmax;

    fn name(&self) -> &str {
        "AddSoftmaxFusion"
    }

    fn pattern(&self) -> Pattern {
        let query_dot_keys = Pattern::symbol("qk");
        let mask = Pattern::symbol("mask");
        Pattern::unary_op(
            "Softmax",
            Pattern::binary_op("Add", query_dot_keys.clone(), mask.clone()),
        )
        .with_name("softmax")
    }

    fn inputs(&self) -> &[&str] {
        &["qk", "mask"]
    }

    fn maybe_fuse(&self, pat_match: &Match, graph: &Graph) -> Result<AddSoftmax, FusionError> {
        let softmax_id = pat_match.node_id("softmax").unwrap();
        let softmax_op = graph
            .get_operator::<Softmax>(softmax_id)
            .ok_or(FusionError::NoMatch)?;

        // This fusion is currently restricted to the case where it is known
        // to be applied over the last, likely-contiguous lane. This is the case
        // in attention operations.
        //
        // It would be possible to extend this to support non-last axes, but the
        // operator would need to be modified to handle that efficiently.
        if !op_applied_to_last_axis::<Softmax>(graph, softmax_id) {
            return Err(FusionError::CheckFailed("op not applied to last axis"));
        }

        Ok(AddSoftmax {
            flush_nans_to_zero: softmax_op.flush_nans_to_zero,
        })
    }
}

/// Converts `Slice(Shape(X), start, end)` into a constant where:
///
/// - `start` and `end` are constants
/// - `X.shape[start..end]` contains only dimensions with known sizes
pub struct ShapeSliceToConstant {}

impl FusionVisitor for ShapeSliceToConstant {
    type State = Pattern;

    fn name(&self) -> &str {
        "ShapeSliceToConstant"
    }

    fn prepare(&self, _: &Graph) -> Self::State {
        let x = Pattern::symbol("x");
        let starts = Pattern::const_symbol("starts");
        let ends = Pattern::const_symbol("ends");
        Pattern::operator("Slice", [Pattern::unary_op("Shape", x), starts, ends])
    }

    fn maybe_fuse(
        &self,
        pattern: &Self::State,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let pat_match = pattern
            .test(op_node_id, graph)
            .ok_or(FusionError::NoMatch)?;
        let x_id = pat_match.node_id("x").ok_or(FusionError::NoMatch)?;
        let starts_id = pat_match.node_id("starts").ok_or(FusionError::NoMatch)?;
        let ends_id = pat_match.node_id("ends").ok_or(FusionError::NoMatch)?;

        let x_shape = graph
            .get_node(x_id)
            .ok_or(FusionError::NoMatch)?
            .shape()
            .ok_or(FusionError::CheckFailed("unknown input shape"))?;
        let ndim = x_shape.len();
        let x_shape = NdTensorView::from_data([ndim], &*x_shape);

        // Check for constant starts/ends.
        let Some(&[start]) = graph.get_vector::<i32>(starts_id) else {
            return Err(FusionError::CheckFailed("start is not constant"));
        };
        let Some(&[end]) = graph.get_vector::<i32>(ends_id) else {
            return Err(FusionError::CheckFailed("end is not constant"));
        };

        // Clamp dimensions here the same as the `Slice` op does.
        let dim_range = SliceRange::new(start as isize, Some(end as isize), 1).clamp(ndim);

        // Extract the selected dimensions and check they all have a fixed size.
        let dims: Option<Vec<_>> = x_shape
            .slice(dim_range)
            .iter()
            .map(|dim| match dim {
                Dimension::Fixed(size) => Some(*size as i32),
                Dimension::Symbolic(_) => None,
            })
            .collect();
        let dims = dims.ok_or(FusionError::CheckFailed("sliced dims have dynamic size"))?;

        // Slice ops should always have one output, but exit early if not.
        let &[Some(output_id)] = op_node.output_ids() else {
            return Err(FusionError::CheckFailed("wrong output count"));
        };

        Ok(Fusion::Constant {
            input_ids: [x_id].into(),
            output_id,
            value: ConstantNode::new(
                op_node.name(),
                ConstantNodeData::Arc(ArcTensor::from_data(&[dims.len()], Arc::new(dims))),
            )
            .into(),
        })
    }
}

/// Fuses patterns for RepeatInterleave.
///
/// The basic pattern for RepeatInterleave looks like:
///
/// ```text
/// T1 = Unsqueeze(X, axes) // Insert new axis
/// T2 = Expand(T1, tmp_shape) // Repeat along new axis
/// Y = Reshape(T2, out_shape) // Combine new axis and previous axis
/// ```
///
/// Under the constraint that the input and output shapes are the same except
/// for a single axis, where the input and output sizes are both fixed and the
/// output size is a multiple of the input size. eg. (batch, 8, seq, dim) =>
/// (batch, 24, seq, dim).
///
/// This pattern on its own has some value by enabling fusion of `Unsqueeze` and
/// `Expand`, which is useful if the Unsqueeze can't run in-place. The more
/// significant value comes as a building block of fusions for Grouped-query
/// Attention (GQA) and other operations.
pub struct RepeatInterleaveFusion {}

impl PatternFusion for RepeatInterleaveFusion {
    type Operator = RepeatInterleave;

    fn name(&self) -> &str {
        "RepeatInterleaveFusion"
    }

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        let axes = Pattern::const_symbol("axes");
        let t1 = Pattern::binary_op("Unsqueeze", x, axes);
        let expand_shape = Pattern::symbol("expand_shape");
        let expanded = Pattern::binary_op("Expand", t1, expand_shape);
        let reshape_shape = Pattern::symbol("reshape_shape");
        Pattern::binary_op("Reshape", expanded, reshape_shape).with_name("reshape")
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn unused_inputs(&self) -> &[&str] {
        &["expand_shape", "reshape_shape"]
    }

    fn maybe_fuse(
        &self,
        pat_match: &Match,
        graph: &Graph,
    ) -> Result<RepeatInterleave, FusionError> {
        let x_id = pat_match.node_id("x").ok_or(FusionError::NoMatch)?;

        let reshape_id = pat_match.node_id("reshape").ok_or(FusionError::NoMatch)?;
        let reshape_op = graph
            .get_node(reshape_id)
            .and_then(|n| n.as_operator())
            .ok_or(FusionError::NoMatch)?;
        let &[Some(reshape_out)] = reshape_op.output_ids() else {
            return Err(FusionError::CheckFailed("wrong output count"));
        };

        // Get shape metadata. This requires that shape inference has been run
        // on the model.
        let in_shape = graph
            .get_node(x_id)
            .ok_or(FusionError::NoMatch)?
            .shape()
            .ok_or(FusionError::CheckFailed("unknown input shape"))?;
        let out_shape = graph
            .get_node(reshape_out)
            .ok_or(FusionError::NoMatch)?
            .shape()
            .ok_or(FusionError::CheckFailed("unknown output shape"))?;

        // Find repeated axis and number of repeats.
        //
        // This operator requires that exactly one axis is repeated. All other
        // axes must have the same size (either a static value or symbolic
        // name).
        if in_shape.len() != out_shape.len() {
            return Err(FusionError::CheckFailed("in rank != out rank"));
        }

        let mut axis_repeats = None;
        for (i, (size_in, size_out)) in in_shape.iter().zip(out_shape.iter()).enumerate() {
            if size_in == size_out {
                continue;
            }

            if axis_repeats.is_some() {
                // Another axis is already repeated.
                return Err(FusionError::CheckFailed("multiple axes repeated"));
            }

            match (size_in, size_out) {
                (Dimension::Fixed(fixed_in), Dimension::Fixed(fixed_out)) => {
                    if fixed_out.is_multiple_of(*fixed_in) {
                        axis_repeats = Some((i, fixed_out / fixed_in));
                    } else {
                        // Repeat count must be an integer.
                        return Err(FusionError::CheckFailed("repeat count not an integer"));
                    }
                }
                _ => {
                    // At least one of the non-equal axes has a symbolic size.
                    return Err(FusionError::CheckFailed("repeat count unknown"));
                }
            }
        }

        let Some((axis, repeats)) = axis_repeats else {
            // All axes have the same size. We _could_ create this fusion by
            // picking any axis and setting the repeat count to one,
            // but haven't found a use for this.
            return Err(FusionError::NoEffect);
        };

        Ok(RepeatInterleave { axis, repeats })
    }
}

/// Fuses Softmax + Where + IsNaN operations.
///
/// This fuses graphs which represent "safe softmax" operations.
/// See https://github.com/pytorch/pytorch/pull/159973.
///
/// ```text
/// Y = Softmax(X)
/// Where(IsNaN(Y), 0., Y)
/// ```
pub struct SafeSoftmaxFusion {}

impl PatternFusion for SafeSoftmaxFusion {
    type Operator = Softmax;

    fn name(&self) -> &str {
        "SafeSoftmaxFusion"
    }

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        let y = Pattern::unary_op("Softmax", x).with_name("softmax");

        let cond = Pattern::unary_op("IsNaN", y.clone());
        Pattern::operator("Where", [cond, Pattern::constant(0.), y])
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Result<Softmax, FusionError> {
        let softmax_id = pat_match.node_id("softmax").unwrap();
        let softmax_op = g
            .get_operator::<Softmax>(softmax_id)
            .ok_or(FusionError::NoMatch)?;
        Ok(Softmax {
            flush_nans_to_zero: true,
            ..*softmax_op
        })
    }
}

/// Replace `Shape` operators with `ComputeShape` operators which generate the
/// same output using dimension sizes that are either precomputed or extracted
/// from graph inputs.
///
/// The benefit of this fusion is to remove a consumer of the value which feeds
/// into the Shape operator. This potentially frees up the source operator to be
/// included in fusions. For example, given:
///
/// ```text
/// T = Sigmoid(X)
/// S = Shape(T)
/// Y = Mul(X, T)
/// ```
///
/// The `Shape` operator prevents fusing this subgraph into `Silu(X)`, because
/// it relies on the intermediate value `S`. If however we determine that `S`
/// can be computed from another node closer to the graph inputs, we remove the
/// `Shape(T)` and free up the remaining `Mul(X, Sigmoid(X))` to be fused.
pub struct ComputeShapeFusion {}

impl FusionVisitor for ComputeShapeFusion {
    // Map of symbolic_dimension => (input_id, dimension_index)
    type State = HashMap<String, (NodeId, u32)>;

    fn name(&self) -> &str {
        "ComputeShapeFusion"
    }

    fn prepare(&self, graph: &Graph) -> Self::State {
        let mut map = HashMap::new();

        for id in graph.input_ids() {
            let Some(node) = graph.get_node(*id) else {
                continue;
            };
            let Some(shape) = node.shape() else {
                continue;
            };

            for (dim_idx, dim) in shape.iter().enumerate() {
                match dim {
                    Dimension::Symbolic(name) => {
                        if !map.contains_key(name) {
                            map.insert(name.to_string(), (*id, dim_idx as u32));
                        }
                    }
                    Dimension::Fixed(_) => {}
                }
            }
        }

        map
    }

    fn maybe_fuse(
        &self,
        state: &Self::State,
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let shape_op = op_node
            .operator()
            .downcast_ref::<Shape>()
            .ok_or(FusionError::NoMatch)?;

        // Shape operators which slice their inputs are not supported yet.
        if shape_op.start.is_some() || shape_op.end.is_some() {
            return Err(FusionError::CheckFailed(
                "slice has start or end attributes",
            ));
        }

        let &[Some(shape_source)] = op_node.input_ids() else {
            return Err(FusionError::CheckFailed("wrong input count"));
        };
        let dims = graph
            .get_node(shape_source)
            .ok_or(FusionError::NoMatch)?
            .shape()
            .ok_or(FusionError::CheckFailed("unknown input shape"))?;

        if dims.iter().any(|dim| match dim {
            Dimension::Symbolic(name) => !state.contains_key(name),
            Dimension::Fixed(_) => false,
        }) {
            return Err(FusionError::CheckFailed("unknown symbolic dim"));
        }

        let mut input_ids = Vec::new();
        let symbols: Vec<SymbolInfo> = dims
            .iter()
            .filter_map(|dim| match dim {
                Dimension::Fixed(_) => None,
                Dimension::Symbolic(name) => {
                    let (input_id, dim) = state
                        .get(name.as_str())
                        .copied()
                        .expect("should have symbolic name");
                    let idx =
                        if let Some(used_idx) = input_ids.iter().position(|id| *id == input_id) {
                            used_idx
                        } else {
                            input_ids.push(input_id);
                            input_ids.len() - 1
                        };
                    Some(SymbolInfo {
                        name: name.clone(),
                        input: idx as u32,
                        axis: dim,
                    })
                }
            })
            .collect();

        let shape: Vec<SymExpr> = dims
            .iter()
            .map(|dim| match dim {
                Dimension::Fixed(size) => SymExpr::Value(*size as i32),
                Dimension::Symbolic(name) => SymExpr::Var(
                    Symbol {
                        name: name.to_string(),
                        positive: true,
                        synthetic: false,
                    }
                    .into(),
                ),
            })
            .collect();

        let input_ids: Vec<_> = input_ids.into_iter().map(Some).collect();

        let compute_shape = ComputeShape { shape, symbols };

        Ok(Fusion::from_op(
            op_node.name(),
            Arc::new(compute_shape),
            &input_ids,
            op_node.output_ids(),
            op_node.input_ids(),
        ))
    }
}

/// Fuse `MatMul(Q, RepeatInterleave(K_or_V))`.
///
/// This is a fusion for both the Q@K^T and QK@V matmuls in Grouped-query
/// Attention.
pub struct GroupedQueryAttentionMatMulFusion {}

impl PatternFusion for GroupedQueryAttentionMatMulFusion {
    type Operator = GroupedQueryAttentionMatMul;

    fn name(&self) -> &str {
        "GroupedQueryAttentionMatMulFusion"
    }

    fn pattern(&self) -> Pattern {
        let a = Pattern::symbol("a");
        let b = Pattern::symbol("b");
        let t1 = Pattern::unary_op("RepeatInterleave", b).with_name("repeat");
        Pattern::any_of(vec![
            // Vanilla matmul used for QK @ V.
            Pattern::binary_op("MatMul", a.clone(), t1.clone()),
            // Matmul with transposed RHS and scaling applied to the input, used
            // for Q @ K.
            Pattern::binary_op(
                "FusedMatMul",
                a.clone(),
                Pattern::unary_op("Transpose", t1.clone()).with_name("transpose"),
            )
            .with_name("scaled_matmul"),
        ])
    }

    fn inputs(&self) -> &[&str] {
        &["a", "b"]
    }

    fn maybe_fuse(
        &self,
        pat_match: &Match,
        graph: &Graph,
    ) -> Result<GroupedQueryAttentionMatMul, FusionError> {
        let repeat_id = pat_match.node_id("repeat").ok_or(FusionError::NoMatch)?;
        let &RepeatInterleave { axis, repeats } = graph
            .get_operator::<RepeatInterleave>(repeat_id)
            .ok_or(FusionError::NoMatch)?;

        // Only the second (head) dimension can be repeated.
        if axis != 1 {
            return Err(FusionError::CheckFailed("repeated axis is not 1"));
        }

        // Only 4D inputs are supported. We expect the LHS input to have shape
        // (batch, query_heads, seq, d_model) and the RHS input to have shape
        // (batch, kv_heads, seq, d_model) where `query_heads = kv_heads *
        // repeats`.
        let b_id = pat_match.node_id("a").unwrap();
        let a_shape = graph
            .get_node(b_id)
            .and_then(|n| n.shape())
            .ok_or(FusionError::CheckFailed("unknown LHS rank"))?;
        if a_shape.len() != 4 {
            return Err(FusionError::CheckFailed("unsupported LHS rank"));
        }

        let b_id = pat_match.node_id("b").unwrap();
        let b_shape = graph
            .get_node(b_id)
            .and_then(|n| n.shape())
            .ok_or(FusionError::CheckFailed("unknown RHS rank"))?;
        if b_shape.len() != 4 {
            return Err(FusionError::CheckFailed("unsupported RHS rank"));
        }

        // Check that query_heads is a multiple of kv_heads.
        let Dimension::Fixed(query_heads) = a_shape[1] else {
            return Err(FusionError::CheckFailed("unknown query heads"));
        };
        let Dimension::Fixed(kv_heads) = b_shape[1] else {
            return Err(FusionError::CheckFailed("unknown KV heads"));
        };
        if query_heads != kv_heads * repeats {
            return Err(FusionError::CheckFailed(
                "query_heads != kv_heads * repeats",
            ));
        }

        // Set the scale and transpose flag if this is the Q @ K^T matmul.
        //
        // We rely on the Mul + MatMul being first fused into FusedMatMul by
        // MatMulScaleFusion.
        let mut alpha = None;
        let mut transpose_rhs = false;

        if let Some(transpose_op) = pat_match
            .node_id("transpose")
            .and_then(|id| graph.get_operator::<Transpose>(id))
            // Permute must transpose only last two dims
            && transpose_op.perm.as_deref() == Some(&[0, 1, 3, 2])
        {
            transpose_rhs = true;

            if let Some(matmul) = pat_match
                .node_id("scaled_matmul")
                .and_then(|id| graph.get_operator::<FusedMatMul>(id))
            {
                alpha = matmul.alpha;
            }
        }

        Ok(GroupedQueryAttentionMatMul {
            repeats,
            alpha,
            transpose_rhs,
        })
    }
}

/// If `shape` has the form `[1, C, 1, ...]`, return `C`.
fn channel_count(shape: &[usize]) -> Option<usize> {
    if shape.len() < 2 {
        return None;
    }
    shape
        .iter()
        .enumerate()
        .all(|(axis, &size)| axis == 1 || size == 1)
        .then(|| shape[1])
}

/// Create a constant containing a tensor.
fn tensor_constant(shape: &[usize], data: Vec<f32>) -> Constant {
    ConstantNode::new(
        None,
        ConstantNodeData::Arc(ArcTensor::from_data(shape, Arc::new(data))),
    )
    .into()
}

/// Create a constant containing a vector.
fn vector_constant(data: Vec<f32>) -> Constant {
    tensor_constant(&[data.len()], data)
}

/// Convert a constant with shape `[1, C, 1, ...]` into a `[C]` vector.
fn channel_vector(graph: &Graph, const_id: NodeId, rank: usize) -> Result<Vec<f32>, FusionError> {
    let Some(Node::Constant(const_node)) = graph.get_node(const_id) else {
        return Err(FusionError::NoMatch);
    };
    if const_node.shape().len() != rank || channel_count(const_node.shape()).is_none() {
        return Err(FusionError::CheckFailed("constant is not per-channel"));
    }
    let Some(view) = TypedConstant::<f32>::as_typed_view(const_node) else {
        return Err(FusionError::CheckFailed("constant is not an f32 constant"));
    };
    Ok(view.to_vec())
}

/// Fuse `Add(Mul(X, scale), bias)` into `BatchNormalization(X, scale, bias, mean, var)`.
pub struct BatchNormalizationFusion {}

impl FusionVisitor for BatchNormalizationFusion {
    type State = Pattern;

    fn name(&self) -> &str {
        "BatchNormalizationFusion"
    }

    fn prepare(&self, _: &Graph) -> Pattern {
        Pattern::binary_op(
            "Add",
            Pattern::binary_op("Mul", Pattern::symbol("x"), Pattern::const_symbol("scale")),
            Pattern::const_symbol("bias"),
        )
    }

    fn maybe_fuse(
        &self,
        pattern: &Pattern,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let pat_match = pattern
            .test(op_node_id, graph)
            .ok_or(FusionError::NoMatch)?;
        let (Some(x), Some(scale_id), Some(bias_id)) = (
            pat_match.node_id("x"),
            pat_match.node_id("scale"),
            pat_match.node_id("bias"),
        ) else {
            return Err(FusionError::NoMatch);
        };

        // Extract `[C]` vectors from `[1, C, 1, 1]` scale and bias tensors.
        let x_shape = graph
            .get_node(x)
            .and_then(|node| node.shape())
            .ok_or(FusionError::CheckFailed("unknown input shape"))?;
        let scale = channel_vector(graph, scale_id, x_shape.len())?;
        let bias = channel_vector(graph, bias_id, x_shape.len())?;
        if scale.len() != bias.len() {
            return Err(FusionError::CheckFailed("scale and bias sizes differ"));
        }
        let channels = scale.len();

        // Check `C` matches channel count of input. Otherwise the `Mul` op
        // will either fail or broadcast the input.
        match x_shape.get(1) {
            Some(Dimension::Fixed(size)) if *size == channels => {}
            _ => {
                return Err(FusionError::CheckFailed(
                    "input channel count does not match scale",
                ));
            }
        }

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: Arc::new(BatchNormalization { epsilon: 0. }),
            input_ids: vec![Some(x), None, None, None, None],
            new_constant_inputs: vec![
                (1, vector_constant(scale)),
                (2, vector_constant(bias)),
                // input mean and variance. Set to 0 and 1 respectively so
                // they have no effect.
                (3, vector_constant(vec![0.; channels])),
                (4, vector_constant(vec![1.; channels])),
            ],
            output_ids: op_node.output_ids().to_vec(),
            unused_input_ids: Vec::new(),
        }))
    }
}

/// Fuse `Add(Conv(X, W), bias)` into `Conv(X, W, bias)`.
pub struct ConvAddFusion {}

impl FusionVisitor for ConvAddFusion {
    type State = ();

    fn name(&self) -> &str {
        "ConvAddFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        // Match an `Add` with two inputs.
        if op_node.operator().name() != "Add" {
            return Err(FusionError::NoMatch);
        }
        let [Some(lhs), Some(rhs)] = op_node.input_ids() else {
            return Err(FusionError::NoMatch);
        };
        let (lhs, rhs) = (*lhs, *rhs);

        // One input must be the output of a `Conv`, the other the bias. `Add`
        // is commutative, so try both orderings.
        let is_conv = |id: NodeId| {
            graph
                .get_source_node(id)
                .filter(|(_, conv_op)| conv_op.operator().downcast_ref::<Conv>().is_some())
        };
        let ((_, conv_op), bias_id) = if let Some(conv) = is_conv(lhs) {
            (conv, rhs)
        } else if let Some(conv) = is_conv(rhs) {
            (conv, lhs)
        } else {
            return Err(FusionError::NoMatch);
        };

        // The `Conv` must not already have a bias input.
        let (conv_input, conv_weight) = match conv_op.input_ids() {
            [Some(input), Some(weight)] | [Some(input), Some(weight), None] => (*input, *weight),
            _ => return Err(FusionError::CheckFailed("conv already has a bias")),
        };

        // Determine the `Conv`'s rank and output channel count from the weights,
        // which have shape `[out_channels, in_channels / groups, ...kernel]`.
        let weight_shape = graph
            .get_node(conv_weight)
            .and_then(|n| n.shape())
            .ok_or(FusionError::CheckFailed("unknown conv weight shape"))?;
        let out_channels = match weight_shape.first() {
            Some(Dimension::Fixed(size)) => *size,
            _ => return Err(FusionError::CheckFailed("unknown conv output channels")),
        };
        let conv_rank = weight_shape.len();

        // Check bias is a constant with shape `[1, out_channels, ...]` that
        // can be reshaped to `[out_channels]`.
        let Some(Node::Constant(bias_const)) = graph.get_node(bias_id) else {
            return Err(FusionError::NoMatch);
        };
        let bias_shape = bias_const.shape();
        if bias_shape.len() != conv_rank || channel_count(bias_shape) != Some(out_channels) {
            return Err(FusionError::CheckFailed("bias is not a per-channel bias"));
        }
        let Some(bias_view) = TypedConstant::<f32>::as_typed_view(bias_const) else {
            return Err(FusionError::CheckFailed("bias is not an f32 constant"));
        };

        // Create the `[out_channels]` bias constant.
        let bias_const = vector_constant(bias_view.iter().copied().collect());

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: conv_op.clone_operator(),
            input_ids: vec![Some(conv_input), Some(conv_weight), None],
            new_constant_inputs: vec![(2, bias_const)],
            output_ids: op_node.output_ids().to_vec(),
            unused_input_ids: Vec::new(),
        }))
    }
}

/// Fuse `BatchNormalization(Conv(X, W, b))` into `Conv(X, W', b')`.
///
/// Fuse `BatchNormalization(Conv(X, W, b), scale, bias, mean, var)` into a
/// `Conv` operator with adjusted weights and bias. For each channel the new
/// weights `W'` and bias `b'` are computed as:
///
///    a = scale / (var + epsilon).sqrt()
///    W' = W * a
///    b' = (b - mean) * a + bias
///
/// This requires the weights, bias, scale and variance to all be constants.
pub struct ConvBatchNormFusion {}

impl FusionVisitor for ConvBatchNormFusion {
    type State = ();

    fn name(&self) -> &str {
        "ConvBatchNormFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let Some(batch_norm) = op_node.operator().downcast_ref::<BatchNormalization>() else {
            return Err(FusionError::NoMatch);
        };
        let [
            Some(bn_input),
            Some(scale),
            Some(bn_bias),
            Some(mean),
            Some(var),
        ] = op_node.input_ids()
        else {
            return Err(FusionError::NoMatch);
        };

        // The normalized input must be the output of a `Conv`.
        let Some((_, conv_op)) = graph
            .get_source_node(*bn_input)
            .filter(|(_, op)| op.operator().downcast_ref::<Conv>().is_some())
        else {
            return Err(FusionError::NoMatch);
        };
        let (conv_input, conv_weight, conv_bias) = match conv_op.input_ids() {
            [Some(input), Some(weight)] | [Some(input), Some(weight), None] => {
                (*input, *weight, None)
            }
            [Some(input), Some(weight), Some(bias)] => (*input, *weight, Some(*bias)),
            _ => return Err(FusionError::NoMatch),
        };

        // Weights have shape `[out_channels, in_channels / groups, ...kernel]`.
        let Some(Node::Constant(weight_const)) = graph.get_node(conv_weight) else {
            return Err(FusionError::CheckFailed("conv weights are not constant"));
        };
        let Some(weight_view) = TypedConstant::<f32>::as_typed_view(weight_const) else {
            return Err(FusionError::CheckFailed("conv weights are not f32"));
        };
        let weight_shape = weight_const.shape().to_vec();
        let Some((&out_channels, kernel_shape)) = weight_shape.split_first() else {
            return Err(FusionError::CheckFailed("conv weights are a scalar"));
        };

        let stats = [scale, bn_bias, mean, var].map(|id| graph.get_vector::<f32>(*id));
        let [Some(scale), Some(bn_bias), Some(mean), Some(var)] = stats else {
            return Err(FusionError::CheckFailed(
                "batch norm inputs are not f32 vectors",
            ));
        };
        if [scale, bn_bias, mean, var]
            .iter()
            .any(|stat| stat.len() != out_channels)
        {
            return Err(FusionError::CheckFailed(
                "batch norm input size does not match conv output channels",
            ));
        }

        let conv_bias = match conv_bias {
            Some(bias_id) => match graph.get_vector::<f32>(bias_id) {
                Some(bias) if bias.len() == out_channels => Some(bias),
                _ => return Err(FusionError::CheckFailed("conv bias is not an f32 vector")),
            },
            None => None,
        };

        let mut weight_data = weight_view.to_vec();
        let mut bias_data = Vec::with_capacity(out_channels);

        let elements_per_channel: usize = kernel_shape.iter().product();
        if elements_per_channel == 0 {
            return Err(FusionError::CheckFailed("conv weights are empty"));
        }

        for (chan, chan_weights) in weight_data.chunks_mut(elements_per_channel).enumerate() {
            let alpha = scale[chan] / (var[chan] + batch_norm.epsilon).sqrt();
            for weight in chan_weights {
                *weight *= alpha;
            }
            let bias = conv_bias.map(|bias| bias[chan]).unwrap_or(0.);
            bias_data.push((bias - mean[chan]) * alpha + bn_bias[chan]);
        }

        Ok(Fusion::Op(FusedOp {
            name: op_node.name().map(|s| s.to_string()),
            fused_op: conv_op.clone_operator(),
            input_ids: vec![Some(conv_input), None, None],
            new_constant_inputs: vec![
                (1, tensor_constant(&weight_shape, weight_data)),
                (2, vector_constant(bias_data)),
            ],
            output_ids: op_node.output_ids().to_vec(),
            unused_input_ids: Vec::new(),
        }))
    }
}

/// Describes how a value used by a subgraph which is fused into a
/// [`TileProgram`] relates to the subgraph's output.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ProgramValue {
    /// The value has the same shape as the output.
    Tensor,
    /// The value is broadcast against the output and has a size of one in
    /// the last `inner_dims` dimensions of the output.
    Scalar { inner_dims: usize },
}

/// Classify a value used by a subgraph whose output has shape `out_shape`.
///
/// Returns `None` if the value's type or shape is unknown or unsupported.
fn program_value(graph: &Graph, value_id: NodeId, out_shape: &[Dimension]) -> Option<ProgramValue> {
    let node = graph.get_node(value_id)?;
    if node.dtype() != Some(ValueType::Tensor(DataType::Float)) {
        return None;
    }
    let shape = node.shape()?;
    if shape.as_ref() == out_shape {
        return Some(ProgramValue::Tensor);
    }

    // Align the value's dimensions with the output's, as broadcasting does.
    let pad = out_shape.len().checked_sub(shape.len())?;
    let is_one = |dim: usize| dim < pad || shape[dim - pad] == Dimension::Fixed(1);

    let inner_dims = (0..out_shape.len())
        .rev()
        .take_while(|dim| is_one(*dim))
        .count();
    let outer_dims = out_shape.len() - inner_dims;
    if inner_dims == 0
        || !(0..outer_dims).all(|dim| is_one(dim) || shape[dim - pad] == out_shape[dim])
    {
        return None;
    }
    Some(ProgramValue::Scalar { inner_dims })
}

/// An operator which can be evaluated by a [`TileProgram`], with the IDs of
/// its inputs.
#[derive(Clone, Copy, Debug)]
enum ProgramStep {
    // Steps which produce a tensor.
    Add {
        a: NodeId,
        b: NodeId,
    },
    AddScalar {
        src: NodeId,
        scalar: NodeId,
    },
    DivScalar {
        src: NodeId,
        scalar: NodeId,
    },
    MulScalar {
        src: NodeId,
        scalar: NodeId,
    },
    MulScalarAdd {
        src: NodeId,
        scalar: NodeId,
        addend: NodeId,
    },
    Sin {
        src: NodeId,
    },
    SinSquared {
        src: NodeId,
        exponent: NodeId,
    },
    Square {
        src: NodeId,
        exponent: NodeId,
    },
    SubScalar {
        src: NodeId,
        scalar: NodeId,
    },

    // Steps which produce a scalar for each row.
    /// Mean of the last `inner_dims` dimensions of `src`.
    ReduceMean {
        src: NodeId,
        inner_dims: usize,
    },
    ScalarAdd {
        a: NodeId,
        b: NodeId,
    },
    ScalarSqrt {
        src: NodeId,
    },
}

impl ProgramStep {
    /// Return the step which evaluates `op_node`, if it is a supported
    /// operator whose inputs and output are tensors of shape `out_shape` or
    /// scalars.
    fn from_operator(
        graph: &Graph,
        op_node: &OperatorNode,
        out_shape: &[Dimension],
    ) -> Option<ProgramStep> {
        use ProgramValue::{Scalar, Tensor};

        let [Some(output_id)] = op_node.output_ids() else {
            return None;
        };
        let input = |index: usize| op_node.input_ids().get(index).copied().flatten();
        let kind = |value_id: NodeId| program_value(graph, value_id, out_shape);
        let is_tensor = |value_id: NodeId| kind(value_id) == Some(Tensor);

        let op = op_node.operator();
        let step = if op.downcast_ref::<Add>().is_some() {
            let (a, b) = (input(0)?, input(1)?);
            match (kind(a)?, kind(b)?) {
                (Tensor, Tensor) => ProgramStep::Add { a, b },
                (Tensor, Scalar { .. }) => ProgramStep::AddScalar { src: a, scalar: b },
                (Scalar { .. }, Tensor) => ProgramStep::AddScalar { src: b, scalar: a },
                (Scalar { .. }, Scalar { .. }) => ProgramStep::ScalarAdd { a, b },
            }
        } else if op.downcast_ref::<Sub>().is_some() {
            let (src, scalar) = (input(0)?, input(1)?);
            match (kind(src)?, kind(scalar)?) {
                (Tensor, Scalar { .. }) => ProgramStep::SubScalar { src, scalar },
                _ => return None,
            }
        } else if op.downcast_ref::<Mul>().is_some() {
            let (a, b) = (input(0)?, input(1)?);
            match (kind(a)?, kind(b)?) {
                (Tensor, Scalar { .. }) => ProgramStep::MulScalar { src: a, scalar: b },
                (Scalar { .. }, Tensor) => ProgramStep::MulScalar { src: b, scalar: a },
                _ => return None,
            }
        } else if op.downcast_ref::<Div>().is_some() {
            let (src, scalar) = (input(0)?, input(1)?);
            match (kind(src)?, kind(scalar)?) {
                (Tensor, Scalar { .. }) => ProgramStep::DivScalar { src, scalar },
                _ => return None,
            }
        } else if op.downcast_ref::<Pow>().is_some() {
            let (src, exponent) = (input(0)?, input(1)?);
            if !is_tensor(src) || graph.get_scalar::<f32>(exponent) != Some(2.) {
                return None;
            }
            ProgramStep::Square { src, exponent }
        } else if op.downcast_ref::<Sin>().is_some() {
            let src = input(0)?;
            if !is_tensor(src) {
                return None;
            }
            ProgramStep::Sin { src }
        } else if op.downcast_ref::<Sqrt>().is_some() {
            let src = input(0)?;
            if is_tensor(src) {
                return None;
            }
            kind(src)?;
            ProgramStep::ScalarSqrt { src }
        } else if let Some(reduce) = op.downcast_ref::<ReduceMean>() {
            let src = input(0)?;
            if !is_tensor(src) || input(1).is_some() || !reduce.keep_dims {
                return None;
            }

            // The reduced axes must be the last `inner_dims` axes.
            let rank = out_shape.len() as i32;
            let mut axes: Vec<_> = reduce
                .axes
                .as_deref()?
                .iter()
                .map(|&axis| if axis < 0 { axis + rank } else { axis })
                .collect();
            axes.sort();
            let inner_dims = axes.len();
            if inner_dims == 0 || !axes.into_iter().eq(rank - inner_dims as i32..rank) {
                return None;
            }
            ProgramStep::ReduceMean { src, inner_dims }
        } else {
            return None;
        };

        // Check that the output is a tensor or scalar, as expected for the
        // step.
        let output_is_scalar = matches!(kind(*output_id)?, Scalar { .. });
        (output_is_scalar == step.is_scalar()).then_some(step)
    }

    /// Return true if this step produces a scalar for each row.
    fn is_scalar(&self) -> bool {
        matches!(
            self,
            ProgramStep::ReduceMean { .. }
                | ProgramStep::ScalarAdd { .. }
                | ProgramStep::ScalarSqrt { .. }
        )
    }

    /// Return the IDs of the inputs which have the same shape as the
    /// subgraph's output.
    fn tensor_inputs(&self) -> impl Iterator<Item = NodeId> + use<> {
        let inputs = match *self {
            ProgramStep::Add { a, b } => [Some(a), Some(b)],
            ProgramStep::MulScalarAdd { src, addend, .. } => [Some(src), Some(addend)],
            ProgramStep::AddScalar { src, .. }
            | ProgramStep::DivScalar { src, .. }
            | ProgramStep::MulScalar { src, .. }
            | ProgramStep::ReduceMean { src, .. }
            | ProgramStep::Sin { src }
            | ProgramStep::SinSquared { src, .. }
            | ProgramStep::Square { src, .. }
            | ProgramStep::SubScalar { src, .. } => [Some(src), None],
            ProgramStep::ScalarAdd { .. } | ProgramStep::ScalarSqrt { .. } => [None, None],
        };
        inputs.into_iter().flatten()
    }

    /// Return the IDs of the inputs which are scalars.
    fn scalar_inputs(&self) -> impl Iterator<Item = NodeId> + use<> {
        let inputs = match *self {
            ProgramStep::ScalarAdd { a, b } => [Some(a), Some(b)],
            ProgramStep::AddScalar { scalar, .. }
            | ProgramStep::DivScalar { scalar, .. }
            | ProgramStep::MulScalar { scalar, .. }
            | ProgramStep::MulScalarAdd { scalar, .. }
            | ProgramStep::SubScalar { scalar, .. } => [Some(scalar), None],
            ProgramStep::ScalarSqrt { src } => [Some(src), None],
            ProgramStep::Add { .. }
            | ProgramStep::ReduceMean { .. }
            | ProgramStep::Sin { .. }
            | ProgramStep::SinSquared { .. }
            | ProgramStep::Square { .. } => [None, None],
        };
        inputs.into_iter().flatten()
    }

    /// Return the IDs of all inputs which are tensors or scalars.
    fn inputs(&self) -> impl Iterator<Item = NodeId> + use<> {
        self.tensor_inputs().chain(self.scalar_inputs())
    }
}

/// Fuse subgraphs of elementwise operators, and reductions over the trailing
/// dimensions of their inputs, into a [`TileProgram`]. This evaluates all of
/// the operators on one small tile of the inputs at a time.
///
/// A subgraph is formed from the visited operator and the supported operators
/// which it depends on. Shape inference must show that every value in the
/// subgraph is either a float tensor with the same shape as the output, or a
/// _scalar_ which is constant along one or more trailing dimensions of the
/// output (eg. a scalar, or per-channel values of shape `[1, C, 1]` for an
/// `[N, C, T]` output). Scalars may be inputs of the subgraph or be computed
/// by it, by reducing those trailing dimensions of a tensor. Values which are
/// used outside the subgraph become inputs to it.
pub struct TileProgramFusion {}

impl FusionVisitor for TileProgramFusion {
    type State = ();

    fn name(&self) -> &str {
        "TileProgramFusion"
    }

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _: &(),
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Result<Fusion, FusionError> {
        let [Some(output_id)] = op_node.output_ids() else {
            return Err(FusionError::NoMatch);
        };
        let output_id = *output_id;
        let out_shape = graph
            .get_node(output_id)
            .and_then(|node| node.shape())
            .ok_or(FusionError::NoMatch)?;
        let root_step =
            ProgramStep::from_operator(graph, op_node, &out_shape).ok_or(FusionError::NoMatch)?;

        // Find the supported operators which the visited operator depends on,
        // directly or via other supported operators. This maps operator IDs
        // to the operator's output and the step which evaluates it.
        let mut steps = HashMap::new();
        let mut pending = vec![(op_node_id, output_id, root_step)];
        while let Some((op_id, op_output, step)) = pending.pop() {
            if steps.insert(op_id, (op_output, step)).is_some() {
                continue;
            }
            for input in step.inputs() {
                let Some((src_id, src_op)) = graph.get_source_node(input) else {
                    continue;
                };
                if let Some(src_step) = ProgramStep::from_operator(graph, src_op, &out_shape) {
                    pending.push((src_id, input, src_step));
                }
            }
        }
        let source_step = |value_id: NodeId, steps: &HashMap<NodeId, _>| {
            graph
                .get_source_node(value_id)
                .map(|(src_id, _)| src_id)
                .filter(|src_id| steps.contains_key(src_id))
        };

        // Remove operators whose outputs are used outside the subgraph, which
        // may in turn disconnect other operators from the subgraph's output.
        // Repeat until the subgraph has no such operators. This leaves the
        // subgraph's operators sorted so that each comes after its inputs.
        let mut order = loop {
            let mut order = Vec::new();
            let mut visited = FxHashSet::default();
            let mut stack = vec![(op_node_id, false)];
            while let Some((op_id, inputs_visited)) = stack.pop() {
                if inputs_visited {
                    order.push(op_id);
                } else if visited.insert(op_id) {
                    stack.push((op_id, true));
                    let (_, step) = steps[&op_id];
                    stack.extend(
                        step.inputs()
                            .filter_map(|input| source_step(input, &steps))
                            .map(|src_id| (src_id, false)),
                    );
                }
            }

            let used_outside = order.iter().copied().find(|op_id| {
                let (op_output, _) = steps[op_id];
                *op_id != op_node_id
                    && (graph.output_ids().contains(&op_output)
                        || graph.get_consumers(op_output).is_some_and(|consumers| {
                            consumers.iter().any(|consumer| !visited.contains(consumer))
                        }))
            });
            match used_outside {
                Some(op_id) => {
                    steps.remove(&op_id);
                }
                None => break order,
            }
        };
        if order.len() < 2 {
            return Err(FusionError::NoMatch);
        }

        // Count the uses of each tensor within the subgraph.
        let mut uses: HashMap<NodeId, usize> = HashMap::new();
        for op_id in &order {
            let (_, step) = steps[op_id];
            for input in step.tensor_inputs() {
                *uses.entry(input).or_default() += 1;
            }
        }

        // Combine `Sin` and `Square` steps, if the sine is not used by any
        // other operator.
        for op_id in order.clone() {
            let (op_output, ProgramStep::Square { src, exponent }) = steps[&op_id] else {
                continue;
            };
            let Some(sin_id) = source_step(src, &steps) else {
                continue;
            };
            let (_, ProgramStep::Sin { src: sin_src }) = steps[&sin_id] else {
                continue;
            };
            if uses[&src] != 1 {
                continue;
            }
            let step = ProgramStep::SinSquared {
                src: sin_src,
                exponent,
            };
            steps.insert(op_id, (op_output, step));
            order.retain(|id| *id != sin_id);
        }

        // Combine `MulScalar` and `Add` steps, if the product is not used by
        // any other operator.
        for op_id in order.clone() {
            let (op_output, ProgramStep::Add { a, b }) = steps[&op_id] else {
                continue;
            };
            let combined = [(b, a), (a, b)].into_iter().find_map(|(product, addend)| {
                let mul_id = source_step(product, &steps)?;
                match steps[&mul_id] {
                    (_, ProgramStep::MulScalar { src, scalar }) if uses[&product] == 1 => Some((
                        mul_id,
                        ProgramStep::MulScalarAdd {
                            src,
                            scalar,
                            addend,
                        },
                    )),
                    _ => None,
                }
            });
            if let Some((mul_id, step)) = combined {
                steps.insert(op_id, (op_output, step));
                order.retain(|id| *id != mul_id);
            }
        }

        // Map of value ID to the step which computes it.
        let value_steps: HashMap<NodeId, ProgramStep> =
            order.iter().map(|op_id| steps[op_id]).collect();
        let steps = || order.iter().map(|op_id| steps[op_id]);

        // Assign an index to each scalar: first the subgraph's scalar inputs,
        // then the scalars it computes, then the reciprocals of divisors.
        let mut scalar_inputs = Vec::new();
        for (_, step) in steps() {
            for scalar in step.scalar_inputs() {
                if !value_steps.contains_key(&scalar) && !scalar_inputs.contains(&scalar) {
                    scalar_inputs.push(scalar);
                }
            }
        }
        let mut scalar_ids: HashMap<NodeId, usize> = scalar_inputs
            .iter()
            .enumerate()
            .map(|(index, id)| (*id, index))
            .collect();
        for (value_id, step) in steps() {
            if step.is_scalar() {
                scalar_ids.insert(value_id, scalar_ids.len());
            }
        }
        let mut reciprocal_ids: HashMap<NodeId, usize> = HashMap::new();
        for (_, step) in steps() {
            if let ProgramStep::DivScalar { scalar, .. } = step {
                let next_id = scalar_ids.len() + reciprocal_ids.len();
                reciprocal_ids.entry(scalar).or_insert(next_id);
            }
        }

        // Get the number of trailing dimensions which form a row. Reductions
        // must all be over the same dimensions, and every scalar input must
        // be constant within a row.
        let mut reduce_dims = steps().filter_map(|(_, step)| match step {
            ProgramStep::ReduceMean { inner_dims, .. } => Some(inner_dims),
            _ => None,
        });
        let scalar_input_dims = || {
            scalar_inputs.iter().filter_map(|scalar| {
                match program_value(graph, *scalar, &out_shape) {
                    Some(ProgramValue::Scalar { inner_dims }) => Some(inner_dims),
                    _ => None,
                }
            })
        };
        let inner_dims = if let Some(inner_dims) = reduce_dims.next() {
            if reduce_dims.any(|dims| dims != inner_dims) {
                return Err(FusionError::CheckFailed(
                    "reductions are over different dimensions",
                ));
            }
            inner_dims
        } else {
            scalar_input_dims().min().unwrap_or(out_shape.len())
        };
        if scalar_input_dims().any(|dims| dims < inner_dims) {
            return Err(FusionError::CheckFailed("scalar input varies within rows"));
        }

        // Skip subgraphs whose rows are known to be short. A tile never spans
        // more than one row, so the cost of evaluating the program would then
        // be dominated by the overhead for each tile.
        const MIN_ROW_LEN: usize = 256;
        let row_len: Option<usize> = out_shape[out_shape.len() - inner_dims..]
            .iter()
            .map(|dim| match dim {
                Dimension::Fixed(size) => Some(*size),
                Dimension::Symbolic(_) => None,
            })
            .product();
        if row_len.is_some_and(|len| len < MIN_ROW_LEN) {
            return Err(FusionError::CheckFailed("rows are too short"));
        }

        // Find the first pass in which each value can be used. The mean of a
        // tensor is available in the pass after the one which reads the
        // tensor. Other values are available as soon as their inputs are.
        // Inputs of the subgraph are available in the first pass.
        let mut value_pass: HashMap<NodeId, usize> = HashMap::new();
        for (value_id, step) in steps() {
            let inputs_pass = step
                .inputs()
                .map(|input| value_pass.get(&input).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let pass = match step {
                ProgramStep::ReduceMean { .. } => inputs_pass + 1,
                _ => inputs_pass,
            };
            value_pass.insert(value_id, pass);
        }
        let n_passes = value_pass[&output_id] + 1;

        let mut inputs = Vec::new();
        let mut passes = Vec::new();
        for pass in 0..n_passes {
            // Compute the scalars which become available in this pass.
            let mut scalar_ops = Vec::new();
            for (value_id, step) in steps() {
                if value_pass[&value_id] != pass {
                    continue;
                }
                let dst = || scalar_ids[&value_id];
                match step {
                    ProgramStep::ReduceMean { .. } => {
                        scalar_ops.push(ScalarOp::DivRowLen { scalar: dst() })
                    }
                    ProgramStep::ScalarAdd { a, b } => scalar_ops.push(ScalarOp::Add {
                        dst: dst(),
                        a: scalar_ids[&a],
                        b: scalar_ids[&b],
                    }),
                    ProgramStep::ScalarSqrt { src } => scalar_ops.push(ScalarOp::Sqrt {
                        dst: dst(),
                        src: scalar_ids[&src],
                    }),
                    _ => {}
                }
            }
            let mut reciprocals: Vec<_> = reciprocal_ids
                .iter()
                .filter(|(scalar, _)| value_pass.get(*scalar).copied().unwrap_or(0) == pass)
                .map(|(scalar, dst)| ScalarOp::Reciprocal {
                    dst: *dst,
                    src: scalar_ids[scalar],
                })
                .collect();
            reciprocals.sort_by_key(|op| match op {
                ScalarOp::Reciprocal { dst, .. } => *dst,
                _ => 0,
            });
            scalar_ops.extend(reciprocals);

            // Find the tensors which this pass must produce: those which are
            // reduced to a scalar that becomes available in the next pass,
            // or the output in the last pass.
            let reductions: Vec<(NodeId, usize)> = steps()
                .filter_map(|(value_id, step)| match step {
                    ProgramStep::ReduceMean { src, .. } if value_pass[&value_id] == pass + 1 => {
                        Some((src, scalar_ids[&value_id]))
                    }
                    _ => None,
                })
                .collect();
            let mut targets: Vec<NodeId> = reductions.iter().map(|(src, _)| *src).collect();
            if pass == n_passes - 1 {
                targets.push(output_id);
            }

            // Find the steps needed to compute those tensors, sorted so that
            // each comes after its inputs. Registers do not persist between
            // passes, so this includes steps which were also computed in
            // earlier passes.
            let mut pass_steps = Vec::new();
            let mut visited = FxHashSet::default();
            let mut stack: Vec<_> = targets.iter().rev().map(|id| (*id, false)).collect();
            while let Some((value_id, inputs_visited)) = stack.pop() {
                let Some(step) = value_steps.get(&value_id) else {
                    continue;
                };
                if inputs_visited {
                    pass_steps.push((value_id, *step));
                } else if visited.insert(value_id) {
                    stack.push((value_id, true));
                    stack.extend(step.tensor_inputs().map(|input| (input, false)));
                }
            }

            // Count the uses of each tensor within the pass.
            let mut uses: HashMap<NodeId, usize> = HashMap::new();
            let step_inputs = pass_steps.iter().flat_map(|(_, step)| step.tensor_inputs());
            for input in step_inputs.chain(reductions.iter().map(|(src, _)| *src)) {
                *uses.entry(input).or_default() += 1;
            }

            // Map of value ID to the register which holds the value, for
            // values computed in this pass.
            let mut value_regs: HashMap<NodeId, usize> = HashMap::new();
            let mut free_regs = Vec::new();
            let mut n_regs = 0;

            // Get the source of a tensor and release its register after the
            // last use, so that it can be reused as a destination.
            let mut src = |value_id: NodeId,
                           value_regs: &HashMap<NodeId, usize>,
                           free_regs: &mut Vec<usize>| {
                if let Some(&reg) = value_regs.get(&value_id) {
                    let remaining = uses.get_mut(&value_id).unwrap();
                    *remaining -= 1;
                    if *remaining == 0 {
                        free_regs.push(reg);
                    }
                    Src::Reg(reg)
                } else {
                    let index = inputs
                        .iter()
                        .position(|id| *id == value_id)
                        .unwrap_or_else(|| {
                            inputs.push(value_id);
                            inputs.len() - 1
                        });
                    Src::Input(index)
                }
            };

            let mut tile_ops = Vec::new();
            for (value_id, step) in pass_steps {
                let srcs: Vec<Src> = step
                    .tensor_inputs()
                    .map(|input| src(input, &value_regs, &mut free_regs))
                    .collect();
                let a = srcs[0];

                let dst = if value_id == output_id {
                    Dst::Output
                } else {
                    // Use the lowest free register.
                    free_regs.sort_by(|a, b| b.cmp(a));
                    let reg = free_regs.pop().unwrap_or_else(|| {
                        n_regs += 1;
                        n_regs - 1
                    });
                    value_regs.insert(value_id, reg);
                    Dst::Reg(reg)
                };

                tile_ops.push(match step {
                    ProgramStep::Add { .. } => TileOp::Add { dst, a, b: srcs[1] },
                    ProgramStep::AddScalar { scalar, .. } => TileOp::AddScalar {
                        dst,
                        src: a,
                        scalar: scalar_ids[&scalar],
                    },
                    ProgramStep::DivScalar { scalar, .. } => TileOp::MulScalar {
                        dst,
                        src: a,
                        scalar: reciprocal_ids[&scalar],
                    },
                    ProgramStep::MulScalar { scalar, .. } => TileOp::MulScalar {
                        dst,
                        src: a,
                        scalar: scalar_ids[&scalar],
                    },
                    ProgramStep::MulScalarAdd { scalar, .. } => TileOp::MulScalarAdd {
                        dst,
                        src: a,
                        scalar: scalar_ids[&scalar],
                        addend: srcs[1],
                    },
                    ProgramStep::Sin { .. } => TileOp::Sin { dst, src: a },
                    ProgramStep::SinSquared { .. } => TileOp::SinSquared { dst, src: a },
                    ProgramStep::Square { .. } => TileOp::Square { dst, src: a },
                    ProgramStep::SubScalar { scalar, .. } => TileOp::SubScalar {
                        dst,
                        src: a,
                        scalar: scalar_ids[&scalar],
                    },
                    ProgramStep::ReduceMean { .. }
                    | ProgramStep::ScalarAdd { .. }
                    | ProgramStep::ScalarSqrt { .. } => {
                        unreachable!("pass steps produce tensors")
                    }
                });
            }
            for (value_id, scalar) in reductions {
                tile_ops.push(TileOp::Sum {
                    src: src(value_id, &value_regs, &mut free_regs),
                    scalar,
                });
            }

            passes.push(Pass {
                scalar_ops,
                tile_ops,
            });
        }

        let unused_inputs: Vec<_> = steps()
            .filter_map(|(_, step)| match step {
                ProgramStep::SinSquared { exponent, .. } | ProgramStep::Square { exponent, .. } => {
                    Some(Some(exponent))
                }
                _ => None,
            })
            .collect();

        let program = TileProgram::new(passes, inputs.len(), scalar_inputs.len(), inner_dims);
        let input_ids: Vec<_> = inputs.into_iter().chain(scalar_inputs).map(Some).collect();
        Ok(Fusion::from_op(
            op_node.name(),
            Arc::new(program),
            &input_ids,
            &[Some(output_id)],
            &unused_inputs,
        ))
    }
}

#[cfg(test)]
mod tests {
    // Tests for fusions are currently defined in the main `optimize.rs` module.
}
