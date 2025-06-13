use std::any::Any;
use std::error::Error;
use std::fmt::{Display, Formatter};

use rten_tensor::Tensor;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::downcast::DowncastDyn;
use crate::graph::{
    CaptureEnv, Constant, ConstantNode, Graph, Node, NodeId, OperatorNode, RunError, TypedConstant,
};
use crate::ops::transform_inputs::TransformInputsBuilder;
use crate::ops::{
    AddSoftmax, FusedMatMul, Gelu, LayerNormalization, Operator, ReduceMean, RmsNormalization,
    Silu, Softmax, Swish, Transpose,
};
use crate::Value;

mod pattern_matcher;

use pattern_matcher::{Match, Pattern};

/// Errors that occur while applying graph optimizations.
#[derive(Debug, PartialEq)]
pub enum OptimizeError {
    /// An error occurred while evaluating parts of the graph (eg. as part
    /// of constant propagation).
    RunError(RunError),
}

impl Display for OptimizeError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Self::RunError(err) => write!(f, "partial evaluation failed: {}", err),
        }
    }
}

impl Error for OptimizeError {}

/// Additional graph querying methods used by the optimizer.
trait GraphExt {
    /// Extract the scalar value from a constant node.
    fn get_scalar(&self, node_id: NodeId) -> Option<f32>;
}

impl GraphExt for Graph {
    fn get_scalar(&self, node_id: NodeId) -> Option<f32> {
        self.get_node(node_id).and_then(|node| match node {
            Node::Constant(const_node) => const_node.as_scalar(),
            _ => None,
        })
    }
}

/// Holds a [`Graph`] and associated data structures while it is being mutated
/// by an optimizer, and provides operations to update the graph.
struct GraphMutator {
    /// Map of (value_node_id, operator_node_ids) for each value node that
    /// is an input to one or more operators.
    edges: FxHashMap<NodeId, Vec<NodeId>>,
    graph: Graph,
    output_ids: Vec<NodeId>,
}

impl GraphMutator {
    fn from_graph(graph: Graph) -> GraphMutator {
        // Map of value_node => operator_node.
        let edges: FxHashMap<NodeId, Vec<NodeId>> = graph.iter().fold(
            FxHashMap::default(),
            |mut edges, (node_id, node)| match node {
                Node::Operator(op_node) => {
                    for &edge_start in op_node.input_ids().iter().flatten() {
                        if let Some(edge_ends) = edges.get_mut(&edge_start) {
                            edge_ends.push(node_id);
                        } else {
                            edges.insert(edge_start, vec![node_id]);
                        }
                    }
                    edges
                }
                _ => edges,
            },
        );
        GraphMutator {
            output_ids: graph.output_ids().to_vec(),
            edges,
            graph,
        }
    }

    /// Add a new constant value to the graph.
    fn add_constant<T>(&mut self, name: Option<&str>, value: Tensor<T>) -> NodeId
    where
        Constant: From<ConstantNode<T>>,
    {
        self.graph.add_constant(name, value)
    }

    /// Add a new constant value to the graph.
    fn add_constant_node(&mut self, const_node: Constant) -> NodeId {
        self.graph.add_constant_node(const_node)
    }

    /// Add a new operator to the graph.
    fn add_operator(
        &mut self,
        name: Option<&str>,
        op: Box<dyn Operator + Send + Sync>,
        inputs: &[Option<NodeId>],
        outputs: &[Option<NodeId>],
    ) {
        let op_id = self.graph.add_op(name, op, inputs, outputs);
        for input_id in inputs.iter().filter_map(|id| *id) {
            if let Some(op_ids) = self.edges.get_mut(&input_id) {
                op_ids.push(op_id);
            } else {
                self.edges.insert(input_id, vec![op_id]);
            }
        }
    }

    /// Return a reference to the graph.
    ///
    /// Note there is no mutable variant of this method. All graph updates must
    /// be done via methods of this struct.
    fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Update the output IDs of the graph and return it.
    fn finalize_graph(mut self) -> Graph {
        self.graph.set_output_ids(&self.output_ids);
        self.graph
    }

    /// Iterate over operator nodes and their IDs.
    fn iter_operators(&self) -> impl Iterator<Item = (NodeId, &OperatorNode)> {
        self.graph.iter().filter_map(|(node_id, node)| match node {
            Node::Operator(op) => Some((node_id, op)),
            _ => None,
        })
    }

    /// Iterate over each operator node in the graph and potentially apply a
    /// fusion which combines this node and adjacent nodes.
    fn apply_fusion<F: Fn(&Self, NodeId, &OperatorNode) -> Option<Fusion>>(
        &mut self,
        create_fusion: F,
    ) {
        struct Replacement {
            unfused_ops: Vec<NodeId>,
            fusion: Fusion,
        }

        let fusions: Vec<_> = self
            .iter_operators()
            .filter_map(|(op_node_id, op_node)| {
                let fusion = create_fusion(self, op_node_id, op_node)?;

                // Check for outputs of intermediate steps in the fused subgraph
                // used by operators outside of the subgraph. If any are found,
                // we can't fuse the subgraph as the intermediate value would no
                // longer be available.
                let mut input_ids: Vec<_> = fusion.input_ids.iter().flatten().copied().collect();

                // Execution planning disallows duplicate input IDs. An operator
                // however is allowed to use the same value for multiple inputs.
                input_ids.sort();
                input_ids.dedup();

                let output_ids: Vec<_> = fusion.output_ids.iter().flatten().copied().collect();
                let unfused_ops = self.graph.execution_plan(&input_ids, &output_ids).unwrap();
                let reused_output = find_operator_output_used_outside_subgraph(
                    &self.graph,
                    &self.edges,
                    &unfused_ops,
                    &output_ids,
                );
                if reused_output.is_some() {
                    return None;
                }

                Some(Replacement {
                    fusion,
                    unfused_ops,
                })
            })
            .collect();

        for Replacement {
            fusion,
            unfused_ops,
        } in fusions
        {
            // Remove all the nodes from the unfused subgraph.
            self.graph.remove_nodes(&unfused_ops);
            for consumer_ops in self.edges.values_mut() {
                consumer_ops.retain(|op_id| !unfused_ops.contains(op_id));
            }

            // Add the fused operator. We do this afterwards to avoid node name
            // conflicts.
            self.add_operator(
                fusion.name.as_deref(),
                fusion.fused_op,
                &fusion.input_ids,
                &fusion.output_ids,
            );
        }
    }

    fn output_ids(&self) -> &[NodeId] {
        &self.output_ids
    }

    fn set_captures(&mut self, captures: &[NodeId]) {
        self.graph.set_captures(captures)
    }

    /// Replace `old_value_id` with `new_value_id` in operator inputs and graph
    /// outputs.
    fn replace_value(&mut self, old_value_id: NodeId, new_value_id: NodeId) {
        // Replace `old_value_id` in graph outputs.
        for output_id in self.output_ids.iter_mut().filter(|id| **id == old_value_id) {
            *output_id = new_value_id;
        }

        // Replace `old_value_id` in operator inputs.
        let Some(old_value_op_ids) = self.edges.remove(&old_value_id) else {
            return;
        };

        for &op_id in &old_value_op_ids {
            let Some(Node::Operator(op_node)) = self.graph.get_node_mut(op_id) else {
                panic!("operator node not found");
            };
            op_node.replace_input(old_value_id, new_value_id);
        }

        if let Some(new_value_op_ids) = self.edges.get_mut(&new_value_id) {
            new_value_op_ids.extend(old_value_op_ids);
        } else {
            self.edges.insert(new_value_id, old_value_op_ids);
        }
    }
}

/// Find an operator output in a subgraph which is used outside the subgraph,
/// excluding outputs listed in `output_ids`, which are the final outputs of the
/// subgraph.
fn find_operator_output_used_outside_subgraph(
    graph: &Graph,
    edges: &FxHashMap<NodeId, Vec<NodeId>>,
    subgraph_ops: &[NodeId],
    output_ids: &[NodeId],
) -> Option<NodeId> {
    for op_id in subgraph_ops {
        let op = graph
            .get_node(*op_id)
            .and_then(|n| n.as_operator())
            .expect("node ID should be a valid operator ID");

        for output in op.output_ids().iter().flatten() {
            if output_ids.contains(output) {
                continue;
            }

            // Check for intermediate output used as graph output.
            if graph.output_ids().contains(output) {
                return Some(*output);
            }

            // Check for intermediate output used as input to operator node
            // outside subgraph.
            let Some(consumers) = edges.get(output) else {
                continue;
            };
            for consumer in consumers {
                if !subgraph_ops.contains(consumer) {
                    return Some(*output);
                }
            }
        }
    }
    None
}

/// Defines a fused operator which replaces a subgraph.
struct Fusion {
    name: Option<String>,
    fused_op: Box<dyn Operator + Send + Sync>,
    input_ids: Vec<Option<NodeId>>,
    output_ids: Vec<Option<NodeId>>,
}

impl Fusion {
    /// Create a fusion with a given operator, name and input nodes.
    ///
    /// `output_id` specifies the output ID of the subgraph that this fusion
    /// replaces.
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

/// Applies optimizations to a [`Graph`] to enable faster inference.
pub struct GraphOptimizer {}

impl GraphOptimizer {
    /// Create a new optimizer with the default set of optimizations enabled.
    pub fn new() -> Self {
        GraphOptimizer {}
    }

    /// Apply optimizations to a graph.
    ///
    /// The graph's input and output nodes, identified by
    /// [`input_ids`](Graph::input_ids) and [`output_ids`](Graph::output_ids)
    /// will be preserved. Other nodes may be modified, removed or replaced.
    ///
    /// Returns the optimized graph.
    pub fn optimize(
        &self,
        graph: Graph,
        capture_env: Option<&CaptureEnv>,
    ) -> Result<Graph, OptimizeError> {
        let mut graph_mut = GraphMutator::from_graph(graph);

        if let Some(capture_env) = capture_env {
            self.convert_captured_values_to_constants(&mut graph_mut, capture_env)?;
        }
        self.propagate_constants(&mut graph_mut)?;

        // Fuse operators.
        //
        // The ordering is significant as fusions are tried in turn until a
        // match is found.
        self.apply_fusions(
            &mut graph_mut,
            &[
                &SiluFusion {}.into_visitor(),
                &SwishFusion {}.into_visitor(),
                &GeluFusion {}.into_visitor(),
                &ApproxGeluFusion {}.into_visitor(),
                &LayerNormalizationFusion {},
                &RmsNormalizationFusion {},
                &MatMulAddFusion {},
                &MatMulScaleFusion {},
                &AddSoftmaxFusion {},
            ],
        )?;

        // Fuse view operations (transpose etc.) with computations.
        self.apply_fusions(&mut graph_mut, &[&TransposeFusion {}])?;

        Ok(graph_mut.finalize_graph())
    }

    /// Replace captured values in a graph with constants if the captured value
    /// resolves to a constant node in the parent graph.
    ///
    /// This pass should be performed before other transformations which
    /// require certain nodes to be constants.
    fn convert_captured_values_to_constants(
        &self,
        graph: &mut GraphMutator,
        capture_env: &CaptureEnv,
    ) -> Result<(), OptimizeError> {
        let captured_constants: Vec<(NodeId, Constant)> = graph
            .graph()
            .captures()
            .iter()
            .filter_map(|&capture_id| {
                let cap_name = graph.graph().get_node(capture_id).and_then(|n| n.name())?;
                let Some(Node::Constant(const_node)) = capture_env.get_node(cap_name) else {
                    return None;
                };
                const_node
                    .clone_ref()
                    .map(|local_const| (capture_id, local_const))
            })
            .collect();

        let mut new_captures: FxHashSet<_> = graph.graph().captures().iter().copied().collect();

        for (capture_id, local_const) in captured_constants {
            let const_id = graph.add_constant_node(local_const);
            new_captures.remove(&capture_id);
            graph.replace_value(capture_id, const_id);
        }

        let new_captures: Vec<_> = new_captures.into_iter().collect();
        graph.set_captures(&new_captures);

        Ok(())
    }

    /// Apply constant propagation to replace parts of the graph which depend
    /// only on constant values with a pre-computed constant.
    fn propagate_constants(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        // Do a partial run with no inputs. This evaluates all nodes that
        // transitively depend only on constants.
        let leaves = graph
            .graph()
            .partial_run(vec![], graph.output_ids(), None)
            .map_err(OptimizeError::RunError)?;

        // Take the resulting (value_node_id, value) list, create new constant
        // nodes in the graph with the value and replace references to
        // `value_node_id` in operator inputs and model outputs with the new
        // constant.
        for (value_node_id, output) in leaves {
            let const_name = graph
                .graph()
                .get_node(value_node_id)
                .and_then(|n| n.name())
                .map(|name| name.to_string());
            let const_name = const_name.as_deref();

            let const_id = match output {
                Value::FloatTensor(tensor) => graph.add_constant(const_name, tensor),
                Value::Int32Tensor(tensor) => graph.add_constant(const_name, tensor),
                Value::Int8Tensor(tensor) => graph.add_constant(const_name, tensor),
                Value::UInt8Tensor(tensor) => graph.add_constant(const_name, tensor),
            };
            graph.replace_value(value_node_id, const_id);
        }

        Ok(())
    }

    /// Traverse the graph and test each operator node against a sequence of
    /// fusion visitors, applying the first returned fusion if any.
    ///
    /// This function visits each operator only once, so won't combine multiple
    /// fusions. Fusions that can be combined must be applied in separate
    /// passes.
    fn apply_fusions(
        &self,
        graph: &mut GraphMutator,
        visitors: &[&dyn FusionVisitor],
    ) -> Result<(), OptimizeError> {
        // Create the prepared state once and then re-use it for each operator
        // visited.
        let prepared_state: Vec<_> = visitors.iter().map(|f| f.prepare(graph.graph())).collect();

        graph.apply_fusion(|graph, op_node_id, op_node| {
            for (visitor, state) in visitors.iter().zip(&prepared_state) {
                if let Some(fusion) =
                    visitor.maybe_fuse(state.as_ref(), graph.graph(), op_node_id, op_node)
                {
                    return Some(fusion);
                }
            }
            None
        });

        Ok(())
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
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

/// Defines a fusion that matches a graph pattern and replaces it with a fused
/// operator.
///
/// This is a simplified version of [`FusionVisitor`].
trait UnaryOpFusion {
    type Operator: Operator + Send + Sync;

    /// Return the graph pattern to match.
    ///
    /// The pattern expression should have a single input named "x".
    fn pattern(&self) -> Pattern;

    /// Create a fused operator given a successful match for the pattern.
    ///
    /// This can fail if there are additional requirements which cannot be
    /// expressed in the pattern.
    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Result<Self::Operator, ()>;

    /// Wrap this fusion into a [`FusionVisitor`].
    fn into_visitor(self) -> impl FusionVisitor
    where
        Self: Sized + 'static,
    {
        UnaryOpFusionVisitor(self)
    }
}

struct GeluFusion {}

impl UnaryOpFusion for GeluFusion {
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

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Result<Self::Operator, ()> {
        Ok(Gelu { approximate: false })
    }
}

struct ApproxGeluFusion {}

impl UnaryOpFusion for ApproxGeluFusion {
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

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Result<Self::Operator, ()> {
        Ok(Gelu { approximate: true })
    }
}

struct SiluFusion {}

impl UnaryOpFusion for SiluFusion {
    type Operator = Silu;

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        x.clone() * Pattern::unary_op("Sigmoid", x.clone())
    }

    fn maybe_fuse(&self, _: &Match, _: &Graph) -> Result<Silu, ()> {
        Ok(Silu {})
    }
}

struct SwishFusion {}

impl UnaryOpFusion for SwishFusion {
    type Operator = Swish;

    fn pattern(&self) -> Pattern {
        let x = Pattern::symbol("x");
        let beta = Pattern::const_symbol("beta");
        x.clone() * Pattern::unary_op("Sigmoid", beta * x.clone())
    }

    fn maybe_fuse(&self, pat_match: &Match, g: &Graph) -> Result<Swish, ()> {
        let beta_input = pat_match.node_id("beta").expect("missing symbol");
        let Some(beta) = g.get_scalar(beta_input) else {
            return Err(());
        };
        Ok(Swish { beta })
    }
}

/// Wraps a [`UnaryOpFusion`] to implement [`FusionVisitor`].
struct UnaryOpFusionVisitor<F: UnaryOpFusion + 'static>(F);

impl<U: UnaryOpFusion + 'static> FusionVisitor for UnaryOpFusionVisitor<U> {
    fn prepare(&self, _: &Graph) -> Box<dyn Any> {
        Box::new(self.0.pattern())
    }

    fn maybe_fuse(
        &self,
        state: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let pattern: &Pattern = state.downcast_ref().unwrap();
        let pat_match = pattern.test(op_node_id, graph)?;
        let input_id = pat_match.node_id("x").expect("missing symbol");
        let fused_op = self.0.maybe_fuse(&pat_match, graph).ok()?;
        let fusion = Fusion::from_op(
            op_node.name(),
            Box::new(fused_op),
            &[Some(input_id)],
            op_node.output_ids(),
        );
        Some(fusion)
    }
}

/// Interface for graph visitors which match graph patterns and return fused
/// operations.
trait FusionVisitor {
    /// Prepare for a graph traversal by creating pattern matchers or other
    /// required state.
    fn prepare(&self, graph: &Graph) -> Box<dyn Any>;

    /// Visit an operator in the graph and potentially return a fusion for it.
    ///
    /// `state` is the result of a call to [`prepare`](FusionVisitor::prepare)
    /// before traversing the graph.
    fn maybe_fuse(
        &self,
        state: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion>;
}

/// Identify and fuse common patterns for `LayerNormalization(X)`.
struct LayerNormalizationFusion {}

struct LayerNormFusionState {
    center_pat: Pattern,
    normalize_variance_pat: Pattern,
    scale_pat: Pattern,
    shift_scale_pat: Pattern,
}

impl FusionVisitor for LayerNormalizationFusion {
    fn prepare(&self, _graph: &Graph) -> Box<dyn Any> {
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

        Box::new(LayerNormFusionState {
            center_pat,
            normalize_variance_pat,
            shift_scale_pat,
            scale_pat,
        })
    }

    fn maybe_fuse(
        &self,
        state: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let LayerNormFusionState {
            center_pat,
            normalize_variance_pat,
            scale_pat,
            shift_scale_pat,
        } = state.downcast_ref().unwrap();

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
struct RmsNormalizationFusion {}

impl FusionVisitor for RmsNormalizationFusion {
    fn prepare(&self, _: &Graph) -> Box<dyn Any> {
        let x = Pattern::symbol("x");
        let scale = Pattern::const_symbol("scale");
        let epsilon = Pattern::const_symbol("epsilon");

        // The scaling of the input is canonically written as `x / (sqrt(rms) + epsilon)`.
        //
        // Here we test for `x * 1/(sqrt(rms) + epsilon)` because that is the
        // observed pattern in models like T5. Ideally we would recognize both.
        let pattern = x.clone()
            * (1.
                / Pattern::unary_op(
                    "Sqrt",
                    epsilon
                        + Pattern::unary_op(
                            "ReduceMean",
                            Pattern::binary_op("Pow", x.clone(), 2.0),
                        )
                        .with_name("norm_mean"),
                ))
            * scale;
        Box::new(pattern)
    }

    fn maybe_fuse(
        &self,
        state: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let pattern: &Pattern = state.downcast_ref().unwrap();
        let rms_match = pattern.test(op_node_id, graph)?;
        let x_input = rms_match.node_id("x").unwrap();
        let epsilon_input = rms_match.node_id("epsilon").unwrap();
        let epsilon = graph.get_scalar(epsilon_input)?;
        let scale_input = rms_match.node_id("scale").unwrap();
        let norm_mean = rms_match.node_id("norm_mean").unwrap();

        if !mean_op_reduces_last_axis(graph, norm_mean) {
            return None;
        }

        Some(Fusion::from_op(
            op_node.name(),
            Box::new(RmsNormalization {
                axis: -1,
                epsilon: Some(epsilon),
            }),
            &[Some(x_input), Some(scale_input)],
            op_node.output_ids(),
        ))
    }
}

/// Fuse `Add(MatMul(a, b), bias)` into `FusedMatMul(a, b, bias)`.
struct MatMulAddFusion {}

impl FusionVisitor for MatMulAddFusion {
    fn prepare(&self, _: &Graph) -> Box<dyn Any> {
        let a = Pattern::symbol("a");
        let b = Pattern::symbol("b");
        let bias = Pattern::const_symbol("bias");
        let matmul_add_pat = Pattern::binary_op(
            "Add",
            Pattern::binary_op("MatMul", a.clone(), b.clone()),
            bias.clone(),
        );
        Box::new(matmul_add_pat)
    }

    fn maybe_fuse(
        &self,
        pattern: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let pattern: &Pattern = pattern.downcast_ref().unwrap();
        let matmul_add_match = pattern.test(op_node_id, graph)?;

        let a_input = matmul_add_match.node_id("a").unwrap();
        let b_input = matmul_add_match.node_id("b").unwrap();
        let bias_input = matmul_add_match.node_id("bias").unwrap();

        let is_bias_a_vector = match graph.get_node(bias_input) {
            Some(Node::Constant(const_node)) => const_node.shape().len() == 1,
            _ => false,
        };

        if !is_bias_a_vector {
            return None;
        }

        Some(Fusion::from_op(
            op_node.name(),
            Box::new(FusedMatMul { alpha: None }),
            &[Some(a_input), Some(b_input), Some(bias_input)],
            op_node.output_ids(),
        ))
    }
}

/// Fuse multiplication or division of MatMul inputs and outputs by
/// scalars.
///
/// A subgraph of the form `Mul(MatMul(Mul(X, c), Mul(Y, d)), e)` where c, d
/// and e are constants can be rewritten as `FusedMatMul(X, Y, alpha=c * d *
/// e)`. Each `Mul(X, c)` can also be expressed as `Div(X, 1/c)`.
struct MatMulScaleFusion {}

impl FusionVisitor for MatMulScaleFusion {
    fn prepare(&self, _: &Graph) -> Box<dyn Any> {
        Box::new(())
    }

    fn maybe_fuse(
        &self,
        _state: &dyn Any,
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

struct TransposeFusion {}

impl FusionVisitor for TransposeFusion {
    fn prepare(&self, _: &Graph) -> Box<dyn Any> {
        Box::new(())
    }

    fn maybe_fuse(
        &self,
        _state: &dyn Any,
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
struct AddSoftmaxFusion {}

impl FusionVisitor for AddSoftmaxFusion {
    fn prepare(&self, _: &Graph) -> Box<dyn Any> {
        let query_dot_keys = Pattern::symbol("qk");
        let mask = Pattern::symbol("mask");
        let pat = Pattern::unary_op(
            "Softmax",
            Pattern::binary_op("Add", query_dot_keys.clone(), mask.clone()),
        );
        Box::new(pat)
    }

    fn maybe_fuse(
        &self,
        pattern: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let pattern: &Pattern = pattern.downcast_ref().unwrap();
        let pat_match = pattern.test(op_node_id, graph)?;

        let qk = pat_match.node_id("qk").unwrap();
        let mask = pat_match.node_id("mask").unwrap();
        let softmax_op = op_node.operator().downcast_ref::<Softmax>()?;

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

        Some(Fusion::from_op(
            op_node.name(),
            Box::new(AddSoftmax {}),
            &[Some(qk), Some(mask)],
            op_node.output_ids(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::Arc;

    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use super::{GraphOptimizer, OptimizeError};
    use crate::constant_storage::{ArcSlice, ArcTensorView, ConstantStorage};
    use crate::downcast::DowncastDyn;
    use crate::graph::builder::Expr;
    use crate::graph::{CaptureEnv, Constant, Graph, Node, NodeId};
    use crate::ops::{
        Add, Erf, FusedMatMul, Gelu, LayerNormalization, MatMul, Neg, Pow, ReduceMean,
        RmsNormalization, Sigmoid, Softmax, Sqrt, Swish, Tanh, Transpose,
    };
    use crate::slice_cast::cast_pod_slice;

    fn optimize_graph(graph: Graph) -> Result<Graph, OptimizeError> {
        let optimizer = GraphOptimizer::new();
        optimizer.optimize(graph, None)
    }

    fn arc_tensor_view(val: f32) -> ArcTensorView<f32> {
        let const_data = Vec::from(val.to_le_bytes());
        let const_storage = Arc::new(ConstantStorage::Buffer(const_data));
        let slice = ArcSlice::new(
            const_storage.clone(),
            cast_pod_slice(const_storage.data()).unwrap(),
        )
        .unwrap();
        ArcTensorView::from_data(&[], slice)
    }

    /// Extends [`Expr`] with methods to create expressions for specific
    /// operations.
    ///
    /// For example `a.matmul(b)` returns an expression for a `MatMul` graph
    /// node with `a` and `b` as inputs.
    trait OpExprs {
        fn erf(&self) -> Expr;
        fn pow(&self, rhs: Expr) -> Expr;
        fn matmul(&self, rhs: Expr) -> Expr;
        fn mean(&self) -> Expr;
        fn sigmoid(&self) -> Expr;
        fn square(&self) -> Expr;
        fn sqrt(&self) -> Expr;
        fn softmax(&self, axis: isize) -> Expr;
        fn tanh(&self) -> Expr;
        fn transpose(&self) -> Expr;
    }

    impl OpExprs for Expr {
        fn erf(&self) -> Expr {
            self.unary(Erf {})
        }

        fn matmul(&self, rhs: Expr) -> Expr {
            self.binary(MatMul {}, rhs)
        }

        fn mean(&self) -> Expr {
            self.unary(ReduceMean {
                axes: Some(vec![-1]),
                keep_dims: false,
            })
        }

        fn pow(&self, rhs: Expr) -> Expr {
            self.binary(Pow {}, rhs)
        }

        fn sigmoid(&self) -> Expr {
            self.unary(Sigmoid {})
        }

        fn square(&self) -> Expr {
            self.binary(Pow {}, Expr::constant(2.0))
        }

        fn sqrt(&self) -> Expr {
            self.unary(Sqrt {})
        }

        fn softmax(&self, axis: isize) -> Expr {
            self.unary(Softmax { axis })
        }

        fn tanh(&self) -> Expr {
            self.unary(Tanh {})
        }

        fn transpose(&self) -> Expr {
            self.unary(Transpose { perm: None })
        }
    }

    #[test]
    fn test_convert_captured_values_to_constants() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();

        // Add a ref-counted constant value that can be cheaply cloned.
        let const_tensor = arc_tensor_view(42.);
        graph.add_constant(Some("const_a"), const_tensor);

        // Capture the constant in the subgraph as a value.
        let mut subgraph = Graph::new();
        let sg_val = subgraph.add_value(Some("const_a"), None, None);
        subgraph.set_captures(&[sg_val]);
        subgraph.set_output_ids(&[sg_val]);

        // Run optimizations on the subgraph. This should replace the captured
        // value with a local constant that references the same data.
        let optimizer = GraphOptimizer::new();
        let capture_env = CaptureEnv::new(None, &graph, None, None, None);
        let optimized_subgraph = optimizer.optimize(subgraph, Some(&capture_env))?;

        let outputs = optimized_subgraph.output_ids();
        assert!(optimized_subgraph.captures().is_empty());
        assert_eq!(outputs.len(), 1);
        let node = optimized_subgraph.get_node(outputs[0]).unwrap();
        assert_eq!(node.name(), Some("const_a"));
        assert!(matches!(node, Node::Constant(_)));

        Ok(())
    }

    #[test]
    fn test_constant_propagation() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();

        // Add an operator with constant inputs.
        let const_a = graph.add_constant(Some("const_a"), Tensor::from([1, 2, 3]));
        let const_b = graph.add_constant(Some("const_b"), Tensor::from([4, 5, 6]));
        let (_, add_out) = graph.add_simple_op("add_1", Add {}, &[const_a, const_b]);

        // Add an operator with a dynamic input and the output of the previous operator.
        let input = graph.add_value(Some("input"), None, None);
        let (add_op_2, add_2_out) = graph.add_simple_op("add_2", Add {}, &[add_out, input]);
        graph.set_input_ids(&[input]);
        graph.set_output_ids(&[add_out, add_2_out]);

        // Optimize the graph. This should replace the first operator's output
        // with a constant value.
        let optimizer = GraphOptimizer::new();
        let optimized_graph = optimizer.optimize(graph, None)?;

        // Check that we got the expected inputs and outputs. The optimizer
        // does not promise to preserve IDs for unmodified parts of the graph,
        // but the current implementation does.
        assert_eq!(optimized_graph.input_ids(), &[input]);
        assert_ne!(optimized_graph.output_ids()[0], add_out);
        assert_eq!(optimized_graph.output_ids()[1], add_2_out);

        // Check first output was replaced with constant.
        let replaced_node = optimized_graph
            .get_node(optimized_graph.output_ids()[0])
            .and_then(|n| match &n {
                Node::Constant(c) => Some(c),
                _ => None,
            })
            .unwrap();
        let Constant::Int32(const_int) = replaced_node else {
            return Err("constant not an int".into());
        };
        assert_eq!(const_int.view(), Tensor::from([5, 7, 9]));

        // Check input to second operator was replaced with constant.
        let op = optimized_graph
            .get_node(add_op_2)
            .and_then(|n| match &n {
                Node::Operator(op) => Some(op),
                _ => None,
            })
            .unwrap();
        let input_ids: Vec<_> = op.input_ids().iter().map(|id| id.unwrap()).collect();
        assert_eq!(input_ids.len(), 2);
        assert_ne!(input_ids[0], add_out);
        assert_eq!(input_ids[0], optimized_graph.output_ids()[0]);
        assert_eq!(input_ids[1], input);

        Ok(())
    }

    #[test]
    fn test_fuse_op_with_duplicate_inputs() {
        // We use MatMul + Add fusion for this test, but any fused op that
        // takes multiple non-constant inputs would work.
        let graph = {
            let x = Expr::value("x");
            let bias = [1., 2., 3.];
            let expr = x.matmul(x.clone()) + bias;
            expr.build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedMatMul");
    }

    #[test]
    fn test_fuse_transpose() {
        let graph = {
            let x = Expr::value("x");
            let y = Expr::value("y");
            x.transpose().matmul(y.transpose()).build_graph(["x", "y"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "TransformInputs(MatMul)");
        assert_eq!(
            op.input_ids(),
            graph
                .input_ids()
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_fuse_silu() {
        let graph = {
            let x = Expr::value("x");
            let expr = x.clone() * x.sigmoid();
            expr.build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "Silu");
    }

    #[test]
    fn test_fuse_swish() {
        let graph = {
            let x = Expr::value("x");
            let beta = 1.7;
            let expr = x.clone() * (x.clone() * beta).sigmoid();
            expr.build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        let swish_op = op.operator().downcast_ref::<Swish>().unwrap();
        assert_eq!(swish_op.beta, 1.7);
    }

    #[test]
    fn test_fuse_matmul_add() {
        let graph = {
            let a = Expr::value("a");
            let b = Expr::value("b");
            let bias = [1., 2., 3.];
            let expr = a.matmul(b) + bias;
            expr.build_graph(["a", "b"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedMatMul");
    }

    #[test]
    fn test_fuse_matmul_scaled() {
        // Pattern 1: MatMul(Mul(A, c), Mul(B, d)). This has scale applied to
        // inputs via `Mul` ops.
        let graph = {
            let a = Expr::value("a");
            let b = Expr::value("b");
            let expr = (a * 0.5).matmul(b * 0.3);
            expr.build_graph(["a", "b"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedMatMul");
        let fused_matmul_op = op.operator().downcast_ref::<FusedMatMul>().unwrap();
        assert_eq!(fused_matmul_op.alpha, Some(0.5 * 0.3));

        // Pattern 2: Div(MatMul(A, B), c). This has scale applied to outputs
        // via `Div` ops.
        let graph = {
            let a = Expr::value("a");
            let b = Expr::value("b");
            let expr = a.matmul(b) / 0.5;
            expr.build_graph(["a", "b"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedMatMul");
        let fused_matmul_op = op.operator().downcast_ref::<FusedMatMul>().unwrap();
        assert_eq!(fused_matmul_op.alpha, Some(1. / 0.5));
    }

    #[test]
    fn test_chained_fused_ops() {
        // Two consecutive decomposed Silu operations
        let graph = {
            let x = Expr::value("x");
            let y = x.clone() * x.sigmoid();
            let z = y.clone() * y.sigmoid();
            z.build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        // Check that both ops were fused. This requires that the inputs to the
        // second group of fused nodes are updated after fusing the first.
        let (_, fused_op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(fused_op.operator().name(), "Silu");
        let (_, fused_op_2) = graph
            .get_source_node(fused_op.input_ids()[0].unwrap())
            .unwrap();
        assert_eq!(fused_op_2.operator().name(), "Silu");
    }

    #[test]
    fn test_fuse_gelu() {
        let graph = {
            let x = Expr::value("x");
            let sqrt_2 = (2.0f32).sqrt();
            let expr = x.clone() * ((x / sqrt_2).erf() + 1.0) * 0.5;
            expr.build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        let gelu = op.operator().downcast_ref::<Gelu>().unwrap();
        assert_eq!(gelu.approximate, false);
    }

    #[test]
    fn test_fuse_approx_gelu() {
        let graph = {
            let x = Expr::value("x");
            let sqrt_2_pi = Expr::constant((2.0f32 / std::f32::consts::PI).sqrt());
            let expr = x.clone()
                * 0.5
                * (Expr::constant(1.)
                    + (sqrt_2_pi * (x.clone() + x.pow(Expr::constant(3.0)) * 0.044715)).tanh());
            expr.build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        let gelu = op.operator().downcast_ref::<Gelu>().unwrap();
        assert_eq!(gelu.approximate, true);
    }

    fn layer_norm_graph(with_bias: bool) -> Graph {
        // Center mean
        let epsilon = 1e-6;
        let x = Expr::value("x");
        let x_mean = x.mean();
        let x_sub_mean = x.clone() - x_mean;

        // Normalize variance
        let normalized = x_sub_mean.clone() / (x_sub_mean.square().mean() + epsilon).sqrt();

        // Shift and scale result
        let scale = [3., 4., 5.];
        let expr = if with_bias {
            let bias = [1., 2., 3.];
            normalized * scale + bias
        } else {
            normalized * scale
        };
        expr.build_graph(["x"])
    }

    #[test]
    fn test_fuse_layer_norm() {
        #[derive(Debug)]
        struct Case {
            with_bias: bool,
        }

        let cases = [Case { with_bias: true }, Case { with_bias: false }];

        cases.test_each(|&Case { with_bias }| {
            let graph = layer_norm_graph(with_bias);
            let graph = optimize_graph(graph).unwrap();
            let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
            assert_eq!(op.operator().name(), "LayerNormalization");

            let layer_norm = op.operator().downcast_ref::<LayerNormalization>().unwrap();
            assert_eq!(layer_norm.epsilon, Some(1e-6));
            let bias_input = op.input_ids().get(2).copied().flatten();
            assert_eq!(bias_input.is_some(), with_bias);
        })
    }

    #[test]
    fn test_fuse_rms_norm() {
        // See https://arxiv.org/pdf/1910.07467
        let graph = {
            let x = Expr::value("x");
            let epsilon = 1e-6;
            let rms = (x.square().mean() + epsilon).sqrt();
            let scale = [3., 4., 5.];
            let expr = x * (Expr::constant(1.) / rms) * scale;
            expr.build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        let rms_norm = op.operator().downcast_ref::<RmsNormalization>().unwrap();
        assert_eq!(rms_norm.epsilon, Some(1e-6));
    }

    #[test]
    fn test_fuse_add_softmax() {
        let graph = {
            let qk = Expr::value("qk");
            let m = Expr::value("m");
            let expr = (qk + m).softmax(-1);
            expr.build_graph(["qk", "m"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "AddSoftmax");
    }

    #[test]
    fn test_optimize_preserves_input_output_nodes() {
        // Fuse-able Transpose + MatMul
        let graph = {
            let x = Expr::value("x");
            let y = Expr::value("y");
            x.transpose().matmul(y).build_graph(["x", "y"])
        };
        let orig_input_ids = graph.input_ids().to_vec();
        let orig_output_ids = graph.output_ids().to_vec();

        let graph = optimize_graph(graph).unwrap();

        // Verify that optimizer did change the graph
        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "TransformInputs(MatMul)");

        // The IDs of the input and output nodes should be the same after
        // optimization.
        //
        // The optimizer could have created new output nodes instead, but it
        // would need to ensure that the new outputs preserved value node
        // metadata (name, shape) from the original outputs.
        assert_eq!(graph.input_ids(), orig_input_ids);
        assert_eq!(graph.output_ids(), orig_output_ids);
    }

    #[test]
    fn test_optimize_error() {
        let mut graph = Graph::new();
        let optimizer = GraphOptimizer::new();
        let invalid_id = NodeId::from_u32(123);
        graph.set_input_ids(&[invalid_id]);
        graph.set_output_ids(&[invalid_id]);
        let result = optimizer.optimize(graph, None);
        assert!(matches!(result, Err(OptimizeError::RunError(_))));
    }

    #[test]
    fn test_optimize_removes_unfused_ops() {
        let graph = {
            let x = Expr::value("x");
            let sqrt_2 = (2.0f32).sqrt();
            let expr = x.clone() * ((x / sqrt_2).erf() + 1.0) * 0.5;
            expr.build_graph(["x"])
        };

        // Intermediate nodes of Gelu op. These should be removed after fusion.
        let ops = ["Mul", "Erf", "Div", "Add"];
        for op in ops {
            assert!(graph.get_node_id(op).is_some());
        }

        let optimized = optimize_graph(graph).unwrap();
        for op in ops {
            assert!(optimized.get_node_id(op).is_none());
        }
        let fused_op = optimized
            .get_node_id("Mul_1")
            .and_then(|id| optimized.get_node(id))
            .and_then(|n| n.as_operator())
            .unwrap();
        assert_eq!(fused_op.operator().name(), "Gelu");
    }

    #[test]
    fn test_optimize_does_not_fuse_if_intermediate_outputs_reused() {
        let mut graph = {
            let x = Expr::value("x");
            let sqrt_2 = (2.0f32).sqrt();
            let expr = x.clone() * ((x / sqrt_2).erf() + 1.0) * 0.5;
            expr.build_graph(["x"])
        };

        // Add an operator which reuses an intermediate value from the Gelu
        // subgraph. This should prevent fusion.
        let erf_out = graph.get_node_id("Erf_out").unwrap();
        graph.add_simple_op("neg", Neg {}, &[erf_out]);

        let optimized = optimize_graph(graph).unwrap();
        let fused_op = optimized
            .get_node_id("Mul_1")
            .and_then(|id| optimized.get_node(id))
            .and_then(|n| n.as_operator())
            .unwrap();
        assert_eq!(fused_op.operator().name(), "Mul");
    }

    #[test]
    fn test_optimize_does_not_fuse_if_intermediate_output_is_graph_output() {
        let mut graph = {
            let x = Expr::value("x");
            let sqrt_2 = (2.0f32).sqrt();
            let expr = x.clone() * ((x / sqrt_2).erf() + 1.0) * 0.5;
            expr.build_graph(["x"])
        };

        // Add intermediate output as an output of the graph. This should
        // prevent fusion.
        let erf_out = graph.get_node_id("Erf_out").unwrap();
        let mut output_ids = graph.output_ids().to_vec();
        output_ids.push(erf_out);
        graph.set_output_ids(&output_ids);

        let optimized = optimize_graph(graph).unwrap();

        let fused_op = optimized
            .get_node_id("Mul_1")
            .and_then(|id| optimized.get_node(id))
            .and_then(|n| n.as_operator())
            .unwrap();
        assert_eq!(fused_op.operator().name(), "Mul");
    }
}
