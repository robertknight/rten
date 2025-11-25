//! Traits for defining operator fusions and implementations of fusions.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use rten_tensor::{ArcTensor, NdTensorView, SliceRange};

use crate::graph::{
    Constant, ConstantNode, ConstantNodeData, Dimension, Graph, Node, NodeId, OperatorNode,
    TypedConstant,
};
use crate::operator::Operator;
use crate::ops::transform_inputs::TransformInputsBuilder;
use crate::ops::{
    AddSoftmax, Cast, ComputeShape, DimSpec, DynamicQuantizeLinear, FusedMatMul, Gelu,
    LayerNormalization, MatMulIntegerToFloat, Mul, Reciprocal, ReduceMean, RepeatInterleave,
    RmsNormalization, Shape, Silu, Softmax, Swish, Transpose,
};
use crate::optimize::pattern_matcher::{Match, Pattern};

pub struct FusedOp {
    /// The name of the graph node.
    pub name: Option<String>,

    pub fused_op: Arc<dyn Operator + Send + Sync>,

    /// IDs of input value nodes.
    pub input_ids: Vec<Option<NodeId>>,

    /// IDs of output value nodes.
    pub output_ids: Vec<Option<NodeId>>,

    /// IDs of additional value nodes which are used by the unfused graph but
    /// not required by the fused operator.
    pub unused_input_ids: Vec<Option<NodeId>>,
}

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
}

/// Interface for graph visitors which match graph patterns and return fused
/// operations.
pub trait FusionVisitor {
    type State;

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
    ) -> Option<Fusion>;
}

/// Define a fusion for an operator using a graph pattern.
///
/// This provides a simpler way to define fusions than implementing
/// [`FusionVisitor`]. A `PatternFusion` can be converted into a `FusionVisitor`
/// using [`into_visitor`](PatternFusion::into_visitor).
pub trait PatternFusion {
    /// The operator produced by this fusion.
    type Operator: Operator + Send + Sync;

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
    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Option<Self::Operator>;

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
    ) -> Option<Fusion> {
        let pat_match = pattern.test(op_node_id, graph)?;
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
        Some(fusion)
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

    fn pattern(&self) -> Pattern {
        1. / Pattern::symbol("x")
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Option<Reciprocal> {
        Some(Reciprocal {})
    }
}

/// Convert `ReduceMean(X, axes)` to `ReduceMean<axes>(X)` where `axes` is a constant.
pub struct ReduceMeanAxesFusion {}

impl PatternFusion for ReduceMeanAxesFusion {
    type Operator = ReduceMean;

    fn pattern(&self) -> Pattern {
        let axes = Pattern::const_symbol("axes");
        let x = Pattern::symbol("x");
        Pattern::binary_op("ReduceMean", x, axes).with_name("mean")
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Option<ReduceMean> {
        let mean_op = g.get_operator::<ReduceMean>(pat_match.node_id("mean").unwrap())?;
        let axes = g.get_vector::<i32>(pat_match.node_id("axes").unwrap())?;

        Some(ReduceMean {
            axes: Some(axes.to_vec()),
            keep_dims: mean_op.keep_dims,
            noop_with_empty_axes: false,
        })
    }
}

pub struct GeluFusion {}

impl PatternFusion for GeluFusion {
    type Operator = Gelu;

    fn pattern(&self) -> Pattern {
        // The expression for GELU is usually written as `x * 0.5 * (...)`
        // instead of `x * (...) * 0.5`. Ideally our graph pattern matcher
        // would be smart enough to let us write one pattern and have it match
        // either structure. However it isn't. The pattern used matches PyTorch's
        // `nn.GELU`.
        let x = Pattern::symbol("x");
        x.clone() * (Pattern::unary_op("Erf", x.clone() / (2.0f32).sqrt()) + 1.0) * 0.5
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Option<Gelu> {
        Some(Gelu { approximate: false })
    }
}

/// Remove operations such as `Identity(x)` or `Add(x, 0)` which have no effect.
pub struct IdentityFusion {}

impl FusionVisitor for IdentityFusion {
    type State = Pattern;

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
    ) -> Option<Fusion> {
        let pat_match = pattern.test(op_node_id, graph)?;
        let input_id = pat_match.node_id("x")?;
        let &[Some(output_id)] = op_node.output_ids() else {
            return None;
        };
        Some(Fusion::Identity {
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

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let to_dtype = op_node.operator().downcast_ref::<Cast>().map(|op| op.to)?;

        let &[Some(input_id)] = op_node.input_ids() else {
            return None;
        };

        let input_dtype = graph.get_node(input_id).and_then(|n| n.dtype())?;
        if input_dtype != to_dtype {
            // This Cast op is not a no-op.
            return None;
        }

        let &[Some(output_id)] = op_node.output_ids() else {
            return None;
        };

        Some(Fusion::Identity {
            input_id,
            output_id,
        })
    }
}

pub struct ApproxGeluFusion {}

impl PatternFusion for ApproxGeluFusion {
    type Operator = Gelu;

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

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Option<Gelu> {
        Some(Gelu { approximate: true })
    }
}

pub struct SiluFusion {}

impl PatternFusion for SiluFusion {
    type Operator = Silu;

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        x.clone() * Pattern::unary_op("Sigmoid", x.clone())
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Option<Silu> {
        Some(Silu {})
    }
}

pub struct SwishFusion {}

impl PatternFusion for SwishFusion {
    type Operator = Swish;

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        let beta = Pattern::const_symbol("beta");
        x.clone() * Pattern::unary_op("Sigmoid", beta * x.clone())
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Option<Swish> {
        let beta_input = pat_match.node_id("beta").expect("missing symbol");
        let beta = g.get_scalar(beta_input)?;
        Some(Swish { beta })
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

    fn maybe_fuse(&self, pat_match: &Match, graph: &Graph) -> Option<LayerNormalization> {
        let norm_mean = pat_match.node_id("norm_mean").unwrap();
        if !op_applied_to_last_axis::<ReduceMean>(graph, norm_mean) {
            // The LayerNormalization operator supports taking the mean over
            // multiple trailing axes. However this fusion only supports the
            // common case of taking the mean over one axis.
            return None;
        }

        let center_mean = pat_match.node_id("center_mean").unwrap();
        if !op_applied_to_last_axis::<ReduceMean>(graph, center_mean) {
            return None;
        }

        let epsilon_input = pat_match.node_id("epsilon").unwrap();
        let epsilon = graph.get_scalar(epsilon_input)?;

        Some(LayerNormalization {
            axis: -1,
            epsilon: Some(epsilon),
        })
    }
}

/// Fuse `RMSNormalization(x)`.
///
/// See https://pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html.
pub struct RmsNormalizationFusion {}

impl PatternFusion for RmsNormalizationFusion {
    type Operator = RmsNormalization;

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

    fn maybe_fuse(&self, rms_match: &Match, graph: &Graph) -> Option<Self::Operator> {
        let epsilon_input = rms_match.node_id("epsilon").unwrap();
        let epsilon = graph.get_scalar(epsilon_input)?;
        let norm_mean = rms_match.node_id("norm_mean").unwrap();

        if !op_applied_to_last_axis::<ReduceMean>(graph, norm_mean) {
            return None;
        }

        Some(RmsNormalization {
            axis: -1,
            epsilon: Some(epsilon),
        })
    }
}

/// Fuse `Add(MatMul(a, b), bias)` into `FusedMatMul(a, b, bias)`.
pub struct MatMulAddFusion {}

impl PatternFusion for MatMulAddFusion {
    type Operator = FusedMatMul;

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

    fn maybe_fuse(&self, matmul_add_match: &Match, graph: &Graph) -> Option<FusedMatMul> {
        let bias_input = matmul_add_match.node_id("bias").unwrap();
        let is_bias_a_vector = match graph.get_node(bias_input) {
            Some(Node::Constant(const_node)) => const_node.shape().len() == 1,
            _ => false,
        };

        if !is_bias_a_vector {
            return None;
        }

        Some(FusedMatMul { alpha: None })
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

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
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
            let (output_scale, scale_input) = get_scale_factor(graph, op_node)?;
            alpha *= output_scale;
            let (_, scale_input_op) = graph.get_source_node(scale_input)?;
            scale_input_op
        } else {
            op_node
        };

        if matmul_node.operator().name() != "MatMul" {
            return None;
        }

        let [matmul_lhs, matmul_rhs] = binary_op_input_ids(matmul_node)?;
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
            return None;
        }

        Some(Fusion::from_op(
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

    fn maybe_fuse(&self, pat_match: &Match, graph: &Graph) -> Option<Self::Operator> {
        let a = pat_match.node_id("a").unwrap();
        let a_zero = pat_match.node_id("a_zero").unwrap();

        // Check that the candidate inputs are all outputs from a
        // DynamicQuantizeLinear node. This allows us to be sure that the
        // inputs will have the expected shape.
        let (a_src_id, quantize_op) = graph.get_source_node(a)?;
        let (a_zero_src_id, _) = graph.get_source_node(a_zero)?;

        let [
            Some(quant_out_data),
            Some(quant_out_scale),
            Some(quant_out_zero),
        ] = quantize_op.output_ids()
        else {
            return None;
        };

        // The data and zero point should come from the same DynamicQuantizeLinear op.
        if a_src_id != a_zero_src_id
            || quantize_op
                .operator()
                .downcast_ref::<DynamicQuantizeLinear>()
                .is_none()
            || a != *quant_out_data
            || a_zero != *quant_out_zero
        {
            return None;
        }

        // The scale should come from `Mul(dyn_scale, const_scale)` where
        // `dyn_scale` is the scale output of the DynamicQuantizeLinear op and
        // `const_scale` is a vector.
        let scale = pat_match.node_id("scale").unwrap();
        let (_, scale_src) = graph.get_source_node(scale)?;
        let [Some(dyn_scale), Some(const_scale)] = scale_src.input_ids() else {
            return None;
        };
        if scale_src.operator().downcast_ref::<Mul>().is_none()
            || graph.get_rank(*const_scale) != Some(1)
            || quant_out_scale != dyn_scale
        {
            return None;
        }

        Some(MatMulIntegerToFloat::default())
    }
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

    fn prepare(&self, _: &Graph) {}

    fn maybe_fuse(
        &self,
        _state: &(),
        graph: &Graph,
        _op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
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
            return None;
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
            return None;
        }

        Some(Fusion::from_op(
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

    fn maybe_fuse(&self, pat_match: &Match, graph: &Graph) -> Option<AddSoftmax> {
        let softmax_id = pat_match.node_id("softmax").unwrap();
        let softmax_op = graph.get_operator::<Softmax>(softmax_id)?;

        // This fusion is currently restricted to the case where it is known
        // to be applied over the last, likely-contiguous lane. This is the case
        // in attention operations.
        //
        // It would be possible to extend this to support non-last axes, but the
        // operator would need to be modified to handle that efficiently.
        if !op_applied_to_last_axis::<Softmax>(graph, softmax_id) {
            return None;
        }

        Some(AddSoftmax {
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
    ) -> Option<Fusion> {
        let pat_match = pattern.test(op_node_id, graph)?;
        let x_id = pat_match.node_id("x")?;
        let starts_id = pat_match.node_id("starts")?;
        let ends_id = pat_match.node_id("ends")?;

        let x_shape = graph.get_node(x_id)?.shape()?;
        let ndim = x_shape.len();
        let x_shape = NdTensorView::from_data([ndim], &*x_shape);

        // Check for constant starts/ends.
        let Some(&[start]) = graph.get_vector::<i32>(starts_id) else {
            return None;
        };
        let Some(&[end]) = graph.get_vector::<i32>(ends_id) else {
            return None;
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
        let dims = dims?;

        // Slice ops should always have one output, but exit early if not.
        let &[Some(output_id)] = op_node.output_ids() else {
            return None;
        };

        Some(Fusion::Constant {
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

    fn maybe_fuse(&self, pat_match: &Match, graph: &Graph) -> Option<RepeatInterleave> {
        let x_id = pat_match.node_id("x")?;

        let reshape_id = pat_match.node_id("reshape")?;
        let reshape_op = graph.get_node(reshape_id).and_then(|n| n.as_operator())?;
        let &[Some(reshape_out)] = reshape_op.output_ids() else {
            return None;
        };

        // Get shape metadata. This requires that shape inference has been run
        // on the model.
        let in_shape = graph.get_node(x_id)?.shape()?;
        let out_shape = graph.get_node(reshape_out)?.shape()?;

        // Find repeated axis and number of repeats.
        //
        // This operator requires that exactly one axis is repeated. All other
        // axes must have the same size (either a static value or symbolic
        // name).
        if in_shape.len() != out_shape.len() {
            return None;
        }

        let mut axis_repeats = None;
        for (i, (size_in, size_out)) in in_shape.iter().zip(out_shape.iter()).enumerate() {
            if size_in == size_out {
                continue;
            }

            if axis_repeats.is_some() {
                // Another axis is already repeated.
                return None;
            }

            match (size_in, size_out) {
                (Dimension::Fixed(fixed_in), Dimension::Fixed(fixed_out)) => {
                    if fixed_out.is_multiple_of(*fixed_in) {
                        axis_repeats = Some((i, fixed_out / fixed_in));
                    } else {
                        // Repeat count must be an integer.
                        return None;
                    }
                }
                _ => {
                    // At least one of the non-equal axes has a symbolic size.
                    return None;
                }
            }
        }

        let Some((axis, repeats)) = axis_repeats else {
            // All axes have the same size. We _could_ create this fusion by
            // picking any axis and setting the repeat count to one,
            // but haven't found a use for this.
            return None;
        };

        Some(RepeatInterleave { axis, repeats })
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

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        let y = Pattern::unary_op("Softmax", x).with_name("softmax");

        let cond = Pattern::unary_op("IsNaN", y.clone());
        Pattern::operator("Where", [cond, Pattern::constant(0.), y])
    }

    fn inputs(&self) -> &[&str] {
        &["x"]
    }

    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Option<Softmax> {
        let softmax_id = pat_match.node_id("softmax").unwrap();
        let softmax_op = g.get_operator::<Softmax>(softmax_id)?;
        Some(Softmax {
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
    ) -> Option<Fusion> {
        let shape_op = op_node.operator().downcast_ref::<Shape>()?;

        // Shape operators which slice their inputs are not supported yet.
        if shape_op.start.is_some() || shape_op.end.is_some() {
            return None;
        }

        let &[Some(shape_source)] = op_node.input_ids() else {
            return None;
        };
        let dims = graph.get_node(shape_source)?.shape()?;

        if dims.iter().any(|dim| match dim {
            Dimension::Symbolic(name) => !state.contains_key(name),
            Dimension::Fixed(_) => false,
        }) {
            return None;
        }

        let mut input_ids = Vec::new();
        let shape: Vec<DimSpec> = dims
            .iter()
            .map(|dim| match dim {
                Dimension::Fixed(size) => DimSpec::Static(*size as u32),
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
                    DimSpec::Dynamic {
                        input: idx as u32,
                        dim,
                    }
                }
            })
            .collect();

        let input_ids: Vec<_> = input_ids.into_iter().map(Some).collect();

        let compute_shape = ComputeShape { shape };

        Some(Fusion::from_op(
            op_node.name(),
            Arc::new(compute_shape),
            &input_ids,
            op_node.output_ids(),
            op_node.input_ids(),
        ))
    }
}

#[cfg(test)]
mod tests {
    // Tests for fusions are currently defined in the main `optimize.rs` module.
}
