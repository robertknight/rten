//! Traits for defining operator fusions and implementations of fusions.

use crate::downcast::DowncastDyn;
use crate::graph::{Graph, Node, NodeId, OperatorNode, TypedConstant};
use crate::ops::transform_inputs::TransformInputsBuilder;
use crate::ops::{
    AddSoftmax, FusedMatMul, Gelu, LayerNormalization, Operator, Reciprocal, ReduceMean,
    RmsNormalization, Silu, Softmax, Swish, Transpose,
};
use crate::optimize::pattern_matcher::{Match, Pattern};

/// Defines a fused operator which replaces a subgraph.
pub struct Fusion {
    /// The name of the graph node.
    pub name: Option<String>,

    pub fused_op: Box<dyn Operator + Send + Sync>,

    /// IDs of input value nodes.
    pub input_ids: Vec<Option<NodeId>>,

    /// IDs of output value nodes.
    pub output_ids: Vec<Option<NodeId>>,
}

impl Fusion {
    /// Create a fusion with a given operator, name and input and output nodes.
    fn from_op(
        name: Option<&str>,
        fused_op: Box<dyn Operator + Send + Sync>,
        input_ids: &[Option<NodeId>],
        output_ids: &[Option<NodeId>],
    ) -> Fusion {
        Fusion {
            name: name.map(|s| s.to_string()),
            fused_op,
            input_ids: input_ids.to_vec(),
            output_ids: output_ids.to_vec(),
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

    /// Return the names of input symbols in the pattern.
    ///
    /// The default implementation assumes the pattern has a single dynamic
    /// input variable named "x".
    fn inputs(&self) -> &[&str] {
        &["x"]
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
        PatternFusionVisitor(self)
    }
}

/// Wraps a [`PatternFusion`] to implement [`FusionVisitor`].
struct PatternFusionVisitor<PF: PatternFusion + 'static>(PF);

impl<PF: PatternFusion + 'static> FusionVisitor for PatternFusionVisitor<PF> {
    type State = Pattern;

    fn prepare(&self, _: &Graph) -> Pattern {
        self.0.pattern()
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
            .0
            .inputs()
            .iter()
            .map(|name| Some(pat_match.node_id(name).expect("missing symbol")))
            .collect();
        let fused_op = self.0.maybe_fuse(&pat_match, graph)?;
        let fusion = Fusion::from_op(
            op_node.name(),
            Box::new(fused_op),
            &input_ids,
            op_node.output_ids(),
        );
        Some(fusion)
    }
}

/// Additional graph querying methods used in fusions.
trait GraphQuery {
    /// Extract the scalar value from a constant node.
    fn get_scalar(&self, node_id: NodeId) -> Option<f32>;
}

impl GraphQuery for Graph {
    fn get_scalar(&self, node_id: NodeId) -> Option<f32> {
        self.get_node(node_id).and_then(|node| match node {
            Node::Constant(const_node) => const_node.as_scalar(),
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

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Option<Reciprocal> {
        Some(Reciprocal {})
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

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Option<Gelu> {
        Some(Gelu { approximate: false })
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

    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Option<Swish> {
        let beta_input = pat_match.node_id("beta").expect("missing symbol");
        let beta = g.get_scalar(beta_input)?;
        Some(Swish { beta })
    }
}

/// Test if a node is a `ReduceMean` operator that reduces over its last axis.
fn mean_op_reduces_last_axis(graph: &Graph, node_id: NodeId) -> bool {
    match graph.get_node(node_id) {
        Some(Node::Operator(op_node)) => {
            let Some(mean_op) = op_node.operator().downcast_ref::<ReduceMean>() else {
                return false;
            };

            // The last axis can be specified with either a positive or
            // negative value. We only support the negative case as that
            // is easier to handle and used in popular models.
            if mean_op.axes.as_deref() == Some(&[-1]) {
                true
            } else if let Some(axes_input) = op_node.input_ids().get(1).copied().flatten() {
                match graph.get_node(axes_input) {
                    Some(Node::Constant(val)) => val.as_vector() == Some(&[-1]),
                    _ => false,
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Identify and fuse common patterns for `LayerNormalization(X)`.
pub struct LayerNormalizationFusion {}

pub struct LayerNormFusionState {
    center_pat: Pattern,
    normalize_variance_pat: Pattern,
    scale_pat: Pattern,
    shift_scale_pat: Pattern,
}

impl FusionVisitor for LayerNormalizationFusion {
    type State = LayerNormFusionState;

    fn prepare(&self, _graph: &Graph) -> Self::State {
        let x = Pattern::symbol("x");

        // LayerNormalization has three steps. Pattern matching only supports a
        // single expression, so we use three patterns and match them in reverse
        // order (ie. starting from the output of the final step).

        // First step: Center values
        let center_pat =
            x.clone() - Pattern::unary_op("ReduceMean", x.clone()).with_name("center_mean");

        // Middle step: Normalize variance
        let epsilon = Pattern::const_symbol("epsilon");
        let normalize_variance_pat = x.clone()
            / Pattern::unary_op(
                "Sqrt",
                epsilon
                    + Pattern::unary_op("ReduceMean", Pattern::binary_op("Pow", x.clone(), 2.0))
                        .with_name("norm_mean"),
            );

        // Final step: Scale, and optionally shift, the normalized values
        let bias = Pattern::const_symbol("bias");
        let scale = Pattern::const_symbol("scale");
        let shift_scale_pat = (x.clone() * scale.clone()) + bias;
        let scale_pat = x.clone() * scale;

        LayerNormFusionState {
            center_pat,
            normalize_variance_pat,
            shift_scale_pat,
            scale_pat,
        }
    }

    fn maybe_fuse(
        &self,
        state: &LayerNormFusionState,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let LayerNormFusionState {
            center_pat,
            normalize_variance_pat,
            scale_pat,
            shift_scale_pat,
        } = state;

        let (shift_scale_input, bias_input, scale_input) =
            if let Some(shift_scale_match) = shift_scale_pat.test(op_node_id, graph) {
                // Found match for scale + bias.
                let shift_scale_input = shift_scale_match.node_id("x").unwrap();
                let bias_input = shift_scale_match.node_id("bias").unwrap();
                let scale_input = shift_scale_match.node_id("scale").unwrap();
                (shift_scale_input, Some(bias_input), scale_input)
            } else if let Some(scale_match) = scale_pat.test(op_node_id, graph) {
                // Found match for scale only.
                let x_input = scale_match.node_id("x").unwrap();
                let scale_input = scale_match.node_id("scale").unwrap();
                (x_input, None, scale_input)
            } else {
                return None;
            };

        let norm_match = normalize_variance_pat.test(shift_scale_input, graph)?;
        let norm_input = norm_match.node_id("x").unwrap();
        let epsilon_input = norm_match.node_id("epsilon").unwrap();
        let norm_mean = norm_match.node_id("norm_mean").unwrap();
        if !mean_op_reduces_last_axis(graph, norm_mean) {
            // The LayerNormalization operator supports taking the mean over
            // multiple trailing axes. However this fusion only supports the
            // common case of taking the mean over one axis.
            return None;
        }

        let center_match = center_pat.test(norm_input, graph)?;
        let center_input = center_match.node_id("x").unwrap();
        let center_mean = center_match.node_id("center_mean").unwrap();
        if !mean_op_reduces_last_axis(graph, center_mean) {
            return None;
        }

        let epsilon = graph.get_scalar(epsilon_input)?;

        Some(Fusion::from_op(
            op_node.name(),
            Box::new(LayerNormalization {
                axis: -1,
                epsilon: Some(epsilon),
            }),
            &[Some(center_input), Some(scale_input), bias_input],
            op_node.output_ids(),
        ))
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

        if !mean_op_reduces_last_axis(graph, norm_mean) {
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
            Box::new(FusedMatMul { alpha: Some(alpha) }),
            &[Some(lhs_input), Some(rhs_input)],
            op_node.output_ids(),
        ))
    }
}

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

        let has_transposed_input = op_node.input_ids().iter().any(|input| {
            input
                .and_then(|input| graph.get_source_node(input))
                .and_then(|(_, src_op)| src_op.operator().downcast_ref::<Transpose>())
                .is_some()
        });
        if !has_transposed_input {
            return None;
        }

        let mut fused_op = TransformInputsBuilder::new(op_node.clone_operator());
        let mut fused_inputs = op_node.input_ids().to_vec();

        for (i, input) in op_node.input_ids().iter().enumerate() {
            let Some((_, source_node)) = input.and_then(|input| graph.get_source_node(input))
            else {
                continue;
            };

            let &[transpose_input] = source_node.input_ids() else {
                continue;
            };
            let Some(transpose) = source_node.operator().downcast_ref::<Transpose>() else {
                continue;
            };

            fused_op = fused_op.permute(i, transpose.perm.clone());
            fused_inputs[i] = transpose_input;
        }

        Some(Fusion::from_op(
            op_node.name(),
            Box::new(fused_op.build()),
            &fused_inputs,
            op_node.output_ids(),
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
        let softmax_node = graph.get_node(softmax_id).and_then(|n| n.as_operator())?;
        let softmax_op = softmax_node.operator().downcast_ref::<Softmax>()?;

        // This fusion is currently restricted to the case where it is known
        // to be applied over the last, likely-contiguous lane. This is the case
        // in attention operations.
        //
        // A case we're not handling here is where the operation is applied over
        // the last lane, but `axis` is specified as a positive value. If we
        // knew the ranks of the inputs, we could handle that as well.
        //
        // It would be possible to extend this to support non-last axes, but the
        // operator would need to be modified to handle that efficiently.
        if softmax_op.axis != -1 {
            return None;
        }

        Some(AddSoftmax {})
    }
}

#[cfg(test)]
mod tests {
    // Tests for fusions are currently defined in the main `optimize.rs` module.
}
