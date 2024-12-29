use std::error::Error;
use std::fmt::{Display, Formatter};

use rten_tensor::Tensor;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::downcast::DowncastDyn;
use crate::graph::{
    CaptureEnv, Constant, ConstantNode, Graph, Node, NodeId, OperatorNode, RunError, TypedConstant,
};
use crate::ops::fused::FusedTranspose;
use crate::ops::{
    FusedMatMul, Gelu, LayerNormalization, Operator, ReduceMean, RmsNormalization, Silu, Swish,
    Transpose,
};
use crate::Output;

mod pattern_matcher;

use pattern_matcher::{binary_op, const_symbol, symbol, unary_op, unary_op_key};

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

    /// Add a new operator to the graph with a single output node.
    ///
    /// `op_output_id` specifies the ID of the output node. If not specified,
    /// a new value node is created.
    ///
    /// Returns the ID of the output node.
    fn add_operator(
        &mut self,
        name: Option<&str>,
        op: Box<dyn Operator + Send + Sync>,
        inputs: &[Option<NodeId>],
        op_output_id: Option<NodeId>,
    ) -> NodeId {
        let op_output_id = op_output_id.unwrap_or(self.graph.add_value(None, None, None));
        let op_id = self.graph.add_op(name, op, inputs, &[Some(op_output_id)]);

        for input_id in inputs.iter().filter_map(|id| *id) {
            if let Some(op_ids) = self.edges.get_mut(&input_id) {
                op_ids.push(op_id);
            } else {
                self.edges.insert(input_id, vec![op_id]);
            }
        }

        op_output_id
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
        let fusions: Vec<_> = self
            .iter_operators()
            .filter_map(|(op_node_id, op_node)| create_fusion(self, op_node_id, op_node))
            .collect();

        for Fusion {
            name,
            fused_op,
            input_ids,
            output_id,
        } in fusions
        {
            self.add_operator(name.as_deref(), fused_op, &input_ids, Some(output_id));
        }
    }

    fn output_ids(&self) -> &[NodeId] {
        &self.output_ids
    }

    /// Return the operator node in `graph` that has an incoming edge from a
    /// value node.
    ///
    /// Returns `None` if there are zero or many such operators.
    fn find_operator_with_input(&self, value_node_id: NodeId) -> Option<&OperatorNode> {
        let targets = self.edges.get(&value_node_id).map(|v| v.as_slice())?;
        let target = match targets {
            &[op_id] => Some(op_id),
            _ => None,
        };
        target.map(|id| match self.graph.get_node(id) {
            Some(Node::Operator(op_node)) => op_node,
            _ => panic!("expected operator"),
        })
    }

    /// Get the scalar value of a constant node, or `None` if the node is
    /// not a constant or the value is not a scalar.
    fn get_scalar(&self, node_id: NodeId) -> Option<f32> {
        self.graph.get_node(node_id).and_then(|node| match node {
            Node::Constant(const_node) => const_node.as_scalar(),
            _ => None,
        })
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

/// Defines a fused operator which replaces a subgraph.
struct Fusion {
    name: Option<String>,
    fused_op: Box<dyn Operator + Send + Sync>,
    input_ids: Vec<Option<NodeId>>,
    output_id: NodeId,
}

impl Fusion {
    /// Create a fusion with a given operator, name and input nodes.
    ///
    /// `output_id` specifies the output ID of the subgraph that this fusion
    /// replaces.
    fn from_op<Op: Operator + Send + Sync>(
        name: Option<&str>,
        op: Op,
        input_ids: Vec<Option<NodeId>>,
        output_id: NodeId,
    ) -> Fusion {
        Fusion {
            name: name.map(|s| s.to_string()),
            fused_op: Box::new(op),
            input_ids,
            output_id,
        }
    }
}

/// Utilities for matching patterns in a graph.
trait OperatorMatch {
    /// Test if an operator node matches a given operator and has N inputs and
    /// M outputs.
    fn match_type<Op: Operator, const N: usize, const M: usize>(
        &self,
    ) -> Option<(&Op, [NodeId; N], [NodeId; M])>;
}

impl OperatorMatch for OperatorNode {
    fn match_type<Op: Operator, const N: usize, const M: usize>(
        &self,
    ) -> Option<(&Op, [NodeId; N], [NodeId; M])> {
        let op = self.operator().downcast_ref::<Op>()?;

        let input_ids = self.input_ids();
        if input_ids.len() != N || input_ids.iter().any(|n| n.is_none()) {
            return None;
        }

        let output_ids = self.output_ids();
        if output_ids.len() != M || output_ids.iter().any(|n| n.is_none()) {
            return None;
        }

        let input_ids: [NodeId; N] = std::array::from_fn(|i| input_ids[i].unwrap());
        let output_ids: [NodeId; M] = std::array::from_fn(|i| output_ids[i].unwrap());

        Some((op, input_ids, output_ids))
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

        self.fuse_silu(&mut graph_mut)?;
        self.fuse_swish(&mut graph_mut)?;
        self.fuse_gelu(&mut graph_mut)?;
        self.fuse_layer_norm(&mut graph_mut)?;
        self.fuse_rms_norm(&mut graph_mut)?;
        self.fuse_matmul_add(&mut graph_mut)?;
        self.fuse_matmul_scaled(&mut graph_mut)?;
        self.fuse_transpose(&mut graph_mut)?;

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
                Output::FloatTensor(tensor) => graph.add_constant(const_name, tensor),
                Output::Int32Tensor(tensor) => graph.add_constant(const_name, tensor),
                Output::Int8Tensor(tensor) => graph.add_constant(const_name, tensor),
                Output::UInt8Tensor(tensor) => graph.add_constant(const_name, tensor),
            };
            graph.replace_value(value_node_id, const_id);
        }

        Ok(())
    }

    /// Fuse `Op(Transpose(X), Y, ...) -> Z` into `FusedTranspose<Op>(X, Y, ...) -> Z`.
    ///
    /// This avoids materializing the transposed input for operators which can
    /// efficiently handle the input being non-contiguous.
    fn fuse_transpose(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        graph.apply_fusion(|edges, _op_node_id, op_node| {
            let (transpose_op, [transpose_input], [transpose_output]) =
                op_node.match_type::<Transpose, 1, 1>()?;

            let transpose_target = edges.find_operator_with_input(transpose_output)?;

            // Filter against a set of operators which are known to efficiently
            // handle transposed inputs.
            if !["MatMul", "FusedMatMul"].contains(&transpose_target.operator().name()) {
                return None;
            }

            // Only single-output operators are currently supported.
            let target_output = match transpose_target.output_ids() {
                [Some(output)] => Some(*output),
                _ => None,
            }?;

            // Replace transpose output with transpose input in the fused
            // operator's inputs.
            let fused_input_idx = transpose_target
                .input_ids()
                .iter()
                .position(|&input_id| input_id == Some(transpose_output))
                .expect("fused input missing");
            let mut fused_input = transpose_target.input_ids().to_vec();
            fused_input[fused_input_idx] = Some(transpose_input);

            let fused_op = FusedTranspose::wrap(
                transpose_target.clone_operator(),
                fused_input_idx,
                transpose_op.perm.as_deref(),
            );

            Some(Fusion::from_op(
                transpose_target.name(),
                fused_op,
                fused_input,
                target_output,
            ))
        });

        Ok(())
    }

    /// Fuse `x * Sigmoid(x)` into `Silu(x)`.
    fn fuse_silu(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        let x = symbol("x");
        let silu_pattern = x.clone() * unary_op("Sigmoid", x.clone());

        graph.apply_fusion(|graph, op_node_id, op_node| {
            let silu_match = silu_pattern.test(op_node_id, graph.graph())?;
            let silu_input = silu_match.resolved_symbol("x").expect("missing symbol");
            let op_output = op_node.output_id()?;

            Some(Fusion::from_op(
                op_node.name(),
                Silu {},
                vec![Some(silu_input)],
                op_output,
            ))
        });

        Ok(())
    }

    /// Fuse `x * Sigmoid(beta * x)` into `Swish(x, beta)`.
    fn fuse_swish(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        let x = symbol("x");
        let beta = const_symbol("beta");
        let swish_pattern = x.clone() * unary_op("Sigmoid", beta * x.clone());

        graph.apply_fusion(|graph, op_node_id, op_node| {
            let swish_match = swish_pattern.test(op_node_id, graph.graph())?;
            let swish_input = swish_match.resolved_symbol("x").expect("missing symbol");
            let beta_input = swish_match.resolved_symbol("beta").expect("missing symbol");
            let beta = graph.get_scalar(beta_input)?;
            let op_output = op_node.output_id()?;

            Some(Fusion::from_op(
                op_node.name(),
                Swish { beta },
                [Some(swish_input)].into(),
                op_output,
            ))
        });

        Ok(())
    }

    /// Fuse `Add(MatMul(a, b), bias)` into `MatMulAdd(a, b, bias)`.
    fn fuse_matmul_add(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        let a = symbol("a");
        let b = symbol("b");
        let bias = const_symbol("bias");
        let matmul_add_pat = binary_op(
            "Add",
            binary_op("MatMul", a.clone(), b.clone()),
            bias.clone(),
        );

        graph.apply_fusion(|graph, op_node_id, op_node| {
            let matmul_add_match = matmul_add_pat.test(op_node_id, graph.graph())?;

            let a_input = matmul_add_match.resolved_symbol("a").unwrap();
            let b_input = matmul_add_match.resolved_symbol("b").unwrap();
            let bias_input = matmul_add_match.resolved_symbol("bias").unwrap();
            let op_output = op_node.output_id()?;

            let is_bias_a_vector = match graph.graph().get_node(bias_input) {
                Some(Node::Constant(const_node)) => const_node.shape().len() == 1,
                _ => false,
            };

            if !is_bias_a_vector {
                return None;
            }

            Some(Fusion::from_op(
                op_node.name(),
                FusedMatMul { alpha: None },
                [Some(a_input), Some(b_input), Some(bias_input)].into(),
                op_output,
            ))
        });

        Ok(())
    }

    /// Fuse multiplication or division of MatMul inputs and outputs by
    /// scalars.
    ///
    /// A subgraph of the form `Mul(MatMul(Mul(X, c), Mul(Y, d)), e)` where c, d
    /// and e are constants can be rewritten as `FusedMatMul(X, Y, alpha=c * d *
    /// e)`. Each `Mul(X, c)` can also be expressed as `Div(X, 1/c)`.
    fn fuse_matmul_scaled(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
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
        let get_scale_factor =
            |graph: &GraphMutator, op_node: &OperatorNode| -> Option<(f32, NodeId)> {
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

        graph.apply_fusion(|graph, _op_node_id, op_node| {
            // Accumulated scale factor from scalings applied to MatMul inputs
            // and outputs.
            let mut alpha = 1.0;

            let op_output = op_node.output_id()?;

            // Check if this is a Mul/Div node scaling the output of a MatMul.
            let matmul_node = if ["Mul", "Div"].contains(&op_node.operator().name()) {
                let (output_scale, scale_input) = get_scale_factor(graph, op_node)?;
                alpha *= output_scale;
                let (_, scale_input_op) = graph.graph().get_source_node(scale_input)?;
                scale_input_op
            } else {
                op_node
            };

            if matmul_node.operator().name() != "MatMul" {
                return None;
            }

            let [matmul_lhs, matmul_rhs] = binary_op_input_ids(matmul_node)?;
            let lhs_input =
                if let Some((_, lhs_source_op)) = graph.graph().get_source_node(matmul_lhs) {
                    let (lhs_scale, lhs_input) =
                        get_scale_factor(graph, lhs_source_op).unwrap_or((1.0, matmul_lhs));
                    alpha *= lhs_scale;
                    lhs_input
                } else {
                    // MatMul LHS is not computed by an upstream operator.
                    matmul_lhs
                };

            let rhs_input =
                if let Some((_, rhs_source_op)) = graph.graph().get_source_node(matmul_rhs) {
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
                FusedMatMul { alpha: Some(alpha) },
                [Some(lhs_input), Some(rhs_input)].into(),
                op_output,
            ))
        });

        Ok(())
    }

    /// Fuse `0.5 * X * (1 + Erf(X / Sqrt(2)))` into `Gelu(X)`.
    fn fuse_gelu(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        // The expression for GELU is usually written as `x * 0.5 * (...)`
        // instead of `x * (...) * 0.5`. Ideally our graph pattern matcher
        // would be smart enough to let us write one pattern and have it match
        // either structure. However it isn't. The pattern used matches PyTorch's
        // `nn.GELU`.
        let x = symbol("x");
        let gelu_pattern = x.clone() * (unary_op("Erf", x.clone() / (2.0f32).sqrt()) + 1.0) * 0.5;

        graph.apply_fusion(|graph, op_node_id, op_node| {
            let gelu_match = gelu_pattern.test(op_node_id, graph.graph())?;
            let gelu_input = gelu_match.resolved_symbol("x").expect("missing symbol");
            let op_output = op_node.output_id()?;

            Some(Fusion::from_op(
                op_node.name(),
                Gelu {},
                vec![Some(gelu_input)],
                op_output,
            ))
        });

        Ok(())
    }

    /// Fuse `RMSNormalization(x)`.
    ///
    /// See https://pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html.
    fn fuse_rms_norm(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        let x = symbol("x");
        let scale = const_symbol("scale");
        let epsilon = const_symbol("epsilon");

        // The scaling of the input is canonically written as `x / (sqrt(rms) + epsilon)`.
        //
        // Here we test for `x * 1/(sqrt(rms) + epsilon)` because that is the
        // observed pattern in models like T5. Ideally we would recognize both.
        let rms_pat = x.clone()
            * (1.
                / unary_op(
                    "Sqrt",
                    epsilon
                        + unary_op_key("ReduceMean", binary_op("Pow", x.clone(), 2.0), "norm_mean"),
                ))
            * scale;

        graph.apply_fusion(|graph, op_node_id, op_node| {
            let rms_match = rms_pat.test(op_node_id, graph.graph())?;
            let x_input = rms_match.resolved_symbol("x").unwrap();
            let epsilon_input = rms_match.resolved_symbol("epsilon").unwrap();
            let epsilon = graph.get_scalar(epsilon_input)?;
            let scale_input = rms_match.resolved_symbol("scale").unwrap();
            let norm_mean = rms_match.resolved_symbol("norm_mean").unwrap();
            let op_output = op_node.output_id()?;

            if !mean_op_reduces_last_axis(graph.graph(), norm_mean) {
                return None;
            }

            Some(Fusion::from_op(
                op_node.name(),
                RmsNormalization {
                    axis: -1,
                    epsilon: Some(epsilon),
                },
                [Some(x_input), Some(scale_input)].into(),
                op_output,
            ))
        });

        Ok(())
    }

    /// Identify and fuse common patterns for `LayerNormalization(X)`.
    fn fuse_layer_norm(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        let x = symbol("x");

        // LayerNormalization has three steps. Pattern matching only supports a
        // single expression, so we use three patterns and match them in reverse
        // order (ie. starting from the output of the final step).

        // First step: Center values
        let center_pat = x.clone() - unary_op_key("ReduceMean", x.clone(), "center_mean");

        // Middle step: Normalize variance
        let epsilon = const_symbol("epsilon");
        let normalize_variance_pat = x.clone()
            / unary_op(
                "Sqrt",
                epsilon + unary_op_key("ReduceMean", binary_op("Pow", x.clone(), 2.0), "norm_mean"),
            );

        // Final step: Scale, and optionally shift, the normalized values
        let bias = const_symbol("bias");
        let scale = const_symbol("scale");
        let shift_scale_pat = (x.clone() * scale.clone()) + bias;
        let scale_pat = x.clone() * scale;

        graph.apply_fusion(|graph, op_node_id, op_node| {
            let (shift_scale_input, bias_input, scale_input) =
                if let Some(shift_scale_match) = shift_scale_pat.test(op_node_id, graph.graph()) {
                    // Found match for scale + bias.
                    let shift_scale_input = shift_scale_match.resolved_symbol("x").unwrap();
                    let bias_input = shift_scale_match.resolved_symbol("bias").unwrap();
                    let scale_input = shift_scale_match.resolved_symbol("scale").unwrap();
                    (shift_scale_input, Some(bias_input), scale_input)
                } else if let Some(scale_match) = scale_pat.test(op_node_id, graph.graph()) {
                    // Found match for scale only.
                    let x_input = scale_match.resolved_symbol("x").unwrap();
                    let scale_input = scale_match.resolved_symbol("scale").unwrap();
                    (x_input, None, scale_input)
                } else {
                    return None;
                };

            let norm_match = normalize_variance_pat.test(shift_scale_input, graph.graph())?;
            let norm_input = norm_match.resolved_symbol("x").unwrap();
            let epsilon_input = norm_match.resolved_symbol("epsilon").unwrap();
            let norm_mean = norm_match.resolved_symbol("norm_mean").unwrap();
            if !mean_op_reduces_last_axis(graph.graph(), norm_mean) {
                // The LayerNormalization operator supports taking the mean over
                // multiple trailing axes. However this fusion only supports the
                // common case of taking the mean over one axis.
                return None;
            }

            let center_match = center_pat.test(norm_input, graph.graph())?;
            let center_input = center_match.resolved_symbol("x").unwrap();
            let center_mean = center_match.resolved_symbol("center_mean").unwrap();
            if !mean_op_reduces_last_axis(graph.graph(), center_mean) {
                return None;
            }

            let op_output = op_node.output_id()?;

            let epsilon = match graph.graph().get_node(epsilon_input) {
                Some(Node::Constant(val)) => val.as_scalar(),
                _ => None,
            }?;

            Some(Fusion::from_op(
                op_node.name(),
                LayerNormalization {
                    axis: -1,
                    epsilon: Some(epsilon),
                },
                vec![Some(center_input), Some(scale_input), bias_input],
                op_output,
            ))
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

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::Arc;

    use rten_tensor::Tensor;

    use super::{GraphOptimizer, OptimizeError};
    use crate::constant_storage::{ArcSlice, ArcTensorView, ConstantStorage};
    use crate::downcast::DowncastDyn;
    use crate::graph::{CaptureEnv, Constant, Graph, Node, NodeId};
    use crate::ops::{
        Add, Div, Erf, FusedMatMul, LayerNormalization, MatMul, Mul, Pow, ReduceMean,
        RmsNormalization, Sigmoid, Sqrt, Sub, Swish, Transpose,
    };

    fn optimize_graph(graph: Graph) -> Result<Graph, OptimizeError> {
        let optimizer = GraphOptimizer::new();
        optimizer.optimize(graph, None)
    }

    fn arc_tensor_view(val: f32) -> ArcTensorView<f32> {
        let const_data = Vec::from(val.to_le_bytes());
        let const_storage = Arc::new(ConstantStorage::Buffer(const_data));
        let slice = ArcSlice::new(
            const_storage.clone(),
            // Safety: We are transmuting a `u8` slice created from an `f32` slice
            // back to an `f32` slice.
            unsafe {
                std::slice::from_raw_parts(
                    const_storage.data().as_ptr() as *const f32,
                    const_storage.data().len() / std::mem::size_of::<f32>(),
                )
            },
        )
        .unwrap();
        ArcTensorView::from_data(&[], slice)
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
    fn test_fuse_transpose() {
        let mut graph = Graph::new();

        let input_1 = graph.add_value(None, None, None);
        let input_2 = graph.add_value(None, None, None);

        let (_, transpose_out) =
            graph.add_simple_op("transpose", Transpose { perm: None }, &[input_1]);
        let (_, matmul_out) = graph.add_simple_op("matmul", MatMul {}, &[transpose_out, input_2]);
        graph.set_input_ids(&[input_1, input_2]);
        graph.set_output_ids(&[matmul_out]);

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedTranspose(MatMul)");
        assert_eq!(op.name(), Some("matmul"));
    }

    #[test]
    fn test_fuse_silu() {
        let mut graph = Graph::new();

        let input = graph.add_value(None, None, None);
        let (_, sigmoid_out) = graph.add_simple_op("sigmoid", Sigmoid {}, &[input]);
        let (_, mul_out) = graph.add_simple_op("mul", Mul {}, &[input, sigmoid_out]);
        graph.set_input_ids(&[input]);
        graph.set_output_ids(&[mul_out]);

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "Silu");
        assert_eq!(op.name(), Some("mul"));
    }

    #[test]
    fn test_fuse_swish() {
        let mut graph = Graph::new();

        let input = graph.add_value(None, None, None);
        let beta = graph.add_constant(None, Tensor::from(1.7));
        let (_, mul_beta_out) = graph.add_simple_op("mul_beta", Mul {}, &[input, beta]);
        let (_, sigmoid_out) = graph.add_simple_op("sigmoid", Sigmoid {}, &[mul_beta_out]);
        let (_, mul_out) = graph.add_simple_op("mul", Mul {}, &[input, sigmoid_out]);
        graph.set_input_ids(&[input]);
        graph.set_output_ids(&[mul_out]);

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        let swish_op = op.operator().downcast_ref::<Swish>().unwrap();
        assert_eq!(swish_op.beta, 1.7);
        assert_eq!(op.name(), Some("mul"));
    }

    #[test]
    fn test_fuse_matmul_add() {
        let mut graph = Graph::new();

        let a = graph.add_value(None, None, None);
        let b = graph.add_value(None, None, None);
        let bias = graph.add_constant(None, Tensor::from([1., 2., 3.]));

        let (_, matmul_out) = graph.add_simple_op("matmul", MatMul {}, &[a, b]);
        let (_, add_out) = graph.add_simple_op("add", Add {}, &[matmul_out, bias]);
        graph.set_input_ids(&[a, b]);
        graph.set_output_ids(&[add_out]);

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedMatMul");
        assert_eq!(op.name(), Some("add"));
    }

    #[test]
    fn test_fuse_matmul_scaled() {
        // Pattern 1: MatMul(Mul(A, c), Mul(B, d)). This has scale applied to
        // inputs via `Mul` ops.
        let mut graph = Graph::new();
        let a = graph.add_value(None, None, None);
        let b = graph.add_value(None, None, None);
        let c = graph.add_constant(None, Tensor::from(0.5));
        let d = graph.add_constant(None, Tensor::from(0.3));
        let (_, mul_a_out) = graph.add_simple_op("scale-a", Mul {}, &[a, c]);
        let (_, mul_b_out) = graph.add_simple_op("scale-b", Mul {}, &[b, d]);
        let (_, matmul_out) = graph.add_simple_op("matmul", MatMul {}, &[mul_a_out, mul_b_out]);
        graph.set_input_ids(&[a, b]);
        graph.set_output_ids(&[matmul_out]);

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedMatMul");
        assert_eq!(op.name(), Some("matmul"));
        let fused_matmul_op = op.operator().downcast_ref::<FusedMatMul>().unwrap();
        assert_eq!(fused_matmul_op.alpha, Some(0.5 * 0.3));

        // Pattern 2: Div(MatMul(A, B), c). This has scale applied to outputs
        // via `Div` ops.
        let mut graph = Graph::new();
        let a = graph.add_value(None, None, None);
        let b = graph.add_value(None, None, None);
        let c = graph.add_constant(None, Tensor::from(0.5));
        let (_, matmul_out) = graph.add_simple_op("matmul", MatMul {}, &[a, b]);
        let (_, div_out) = graph.add_simple_op("div", Div {}, &[matmul_out, c]);
        graph.set_input_ids(&[a, b]);
        graph.set_output_ids(&[div_out]);

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedMatMul");
        assert_eq!(op.name(), Some("matmul"));
        let fused_matmul_op = op.operator().downcast_ref::<FusedMatMul>().unwrap();
        assert_eq!(fused_matmul_op.alpha, Some(1. / 0.5));
    }

    #[test]
    fn test_chained_fused_ops() {
        let mut graph = Graph::new();

        // Add two consecutive decomposed Silu operations
        let input = graph.add_value(None, None, None);
        let (_, sigmoid_out) = graph.add_simple_op("sigmoid", Sigmoid {}, &[input]);
        let (_, mul_out) = graph.add_simple_op("mul", Mul {}, &[input, sigmoid_out]);
        let (_, sigmoid_2_out) = graph.add_simple_op("sigmoid", Sigmoid {}, &[mul_out]);
        let (_, mul_2_out) = graph.add_simple_op("mul", Mul {}, &[mul_out, sigmoid_2_out]);
        graph.set_input_ids(&[input]);
        graph.set_output_ids(&[mul_2_out]);

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
        let mut graph = Graph::new();

        let sqrt_2 = graph.add_constant(None, Tensor::from((2.0f32).sqrt()));
        let one = graph.add_constant(None, Tensor::from(1.0));
        let half = graph.add_constant(None, Tensor::from(0.5));

        let input = graph.add_value(None, None, None);
        let (_, div_out) = graph.add_simple_op("div", Div {}, &[input, sqrt_2]);
        let (_, erf_out) = graph.add_simple_op("erf", Erf {}, &[div_out]);
        let (_, add_out) = graph.add_simple_op("add", Add {}, &[erf_out, one]);
        let (_, mul_out) = graph.add_simple_op("mul", Mul {}, &[input, add_out]);
        let (_, mul_half_out) = graph.add_simple_op("mul_half", Mul {}, &[mul_out, half]);
        graph.set_input_ids(&[input]);
        graph.set_output_ids(&[mul_half_out]);

        let graph = optimize_graph(graph).unwrap();
        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "Gelu");
        assert_eq!(op.name(), Some("mul_half"));
    }

    fn layer_norm_graph(with_bias: bool) -> Graph {
        let mut graph = Graph::new();
        let input = graph.add_value(None, None, None);

        // Center values
        let (_, mean_out) = graph.add_simple_op(
            "mean",
            ReduceMean {
                axes: Some(vec![-1]),
                keep_dims: false,
            },
            &[input],
        );
        let (_, sub_out) = graph.add_simple_op("sub", Sub {}, &[input, mean_out]);

        // Normalize variance
        let two = graph.add_constant(None, Tensor::from(2.));
        let (_, pow_out) = graph.add_simple_op("pow", Pow {}, &[sub_out, two]);
        let (_, var_mean_out) = graph.add_simple_op(
            "var_mean",
            ReduceMean {
                axes: Some(vec![-1]),
                keep_dims: false,
            },
            &[pow_out],
        );
        let epsilon = graph.add_constant(None, Tensor::from(1e-6));
        let (_, add_eps_out) = graph.add_simple_op("add_eps", Add {}, &[epsilon, var_mean_out]);
        let (_, sqrt_out) = graph.add_simple_op("sqrt", Sqrt {}, &[add_eps_out]);
        let (_, div_out) = graph.add_simple_op("div", Div {}, &[sub_out, sqrt_out]);

        // Shift and scale
        let scale = graph.add_constant(None, Tensor::from([3., 4., 5.]));
        let (_, mul_out) = graph.add_simple_op("mul", Mul {}, &[div_out, scale]);

        if with_bias {
            let bias = graph.add_constant(None, Tensor::from([1., 2., 3.]));
            let (_, add_out) = graph.add_simple_op("final_add", Add {}, &[mul_out, bias]);
            graph.set_output_ids(&[add_out]);
        } else {
            graph.set_output_ids(&[mul_out]);
        }

        graph.set_input_ids(&[input]);
        graph
    }

    #[test]
    fn test_fuse_layer_norm() {
        struct Case<'a> {
            with_bias: bool,
            output_name: &'a str,
        }

        let cases = [
            Case {
                with_bias: true,
                output_name: "final_add",
            },
            Case {
                with_bias: false,
                output_name: "mul",
            },
        ];

        for Case {
            with_bias,
            output_name,
        } in cases
        {
            let graph = layer_norm_graph(with_bias);
            let graph = optimize_graph(graph).unwrap();
            let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
            assert_eq!(op.operator().name(), "LayerNormalization");
            assert_eq!(op.name(), Some(output_name));

            let layer_norm = op.operator().downcast_ref::<LayerNormalization>().unwrap();
            assert_eq!(layer_norm.epsilon, Some(1e-6));
        }
    }

    fn rms_norm_graph() -> Graph {
        let mut graph = Graph::new();
        let input = graph.add_value(None, None, None);

        // Divide input by root mean squared to normalize scale.
        let two = graph.add_constant(None, Tensor::from(2.));
        let (_, pow_out) = graph.add_simple_op("pow", Pow {}, &[input, two]);
        let (_, var_mean_out) = graph.add_simple_op(
            "var_mean",
            ReduceMean {
                axes: Some(vec![-1]),
                keep_dims: false,
            },
            &[pow_out],
        );
        let epsilon = graph.add_constant(None, Tensor::from(1e-6));
        let (_, add_eps_out) = graph.add_simple_op("add_eps", Add {}, &[epsilon, var_mean_out]);
        let (_, sqrt_out) = graph.add_simple_op("sqrt", Sqrt {}, &[add_eps_out]);

        let one = graph.add_constant(None, Tensor::from(1.));
        let (_, reciprocal_out) = graph.add_simple_op("div", Div {}, &[one, sqrt_out]);
        let (_, mul_rcp_out) = graph.add_simple_op("mul", Mul {}, &[input, reciprocal_out]);

        // Apply constant scale
        let scale = graph.add_constant(None, Tensor::from([3., 4., 5.]));
        let (_, mul_out) = graph.add_simple_op("mul", Mul {}, &[mul_rcp_out, scale]);

        graph.set_input_ids(&[input]);
        graph.set_output_ids(&[mul_out]);

        graph
    }

    #[test]
    fn test_fuse_rms_norm() {
        let graph = rms_norm_graph();

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        let rms_norm = op.operator().downcast_ref::<RmsNormalization>().unwrap();
        assert_eq!(rms_norm.epsilon, Some(1e-6));
    }

    #[test]
    fn test_optimize_preserves_input_output_nodes() {
        let mut graph = Graph::new();

        let input_1 = graph.add_value(None, None, None);
        let input_2 = graph.add_value(None, None, None);

        // Add fuse-able Transpose + MatMul
        let (_, transpose_out) =
            graph.add_simple_op("transpose", Transpose { perm: None }, &[input_1]);
        let (_, matmul_out) = graph.add_simple_op("matmul", MatMul {}, &[transpose_out, input_2]);
        graph.set_input_ids(&[input_1, input_2]);
        graph.set_output_ids(&[matmul_out]);

        let graph = optimize_graph(graph).unwrap();

        // Verify that optimizer did change the graph
        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedTranspose(MatMul)");

        // The IDs of the input and output nodes should be the same after
        // optimization.
        //
        // The optimizer could have created new output nodes instead, but it
        // would need to ensure that the new outputs preserved value node
        // metadata (name, shape) from the original outputs.
        assert_eq!(graph.input_ids(), &[input_1, input_2]);
        assert_eq!(graph.output_ids(), &[matmul_out]);
        assert_eq!(graph.node_name(matmul_out), "matmul_out");
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
}
