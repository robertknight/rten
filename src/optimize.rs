use std::any::Any;
use std::error::Error;
use std::fmt::{Display, Formatter};

use rten_tensor::Tensor;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::{
    CaptureEnv, Constant, ConstantNode, Graph, Node, NodeId, OperatorNode, RunError,
};
use crate::ops::Operator;
use crate::Value;

mod fusions;
mod pattern_matcher;

use fusions::{
    AddSoftmaxFusion, ApproxGeluFusion, Fusion, FusionVisitor, GeluFusion,
    LayerNormalizationFusion, MatMulAddFusion, MatMulScaleFusion, PatternFusion,
    RmsNormalizationFusion, SiluFusion, SwishFusion, TransposeFusion,
};

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
    ///
    /// Returns the number of applied fusions.
    fn apply_fusion<F: Fn(&Self, NodeId, &OperatorNode) -> Option<Fusion>>(
        &mut self,
        create_fusion: F,
    ) -> usize {
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

        let n_fusions = fusions.len();

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

        n_fusions
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
        let fusions: &[&dyn DynFusionVisitor] = &[
            &DynFusion(SiluFusion {}.into_visitor()),
            &DynFusion(SwishFusion {}.into_visitor()),
            &DynFusion(GeluFusion {}.into_visitor()),
            &DynFusion(ApproxGeluFusion {}.into_visitor()),
            &DynFusion(LayerNormalizationFusion {}),
            &DynFusion(RmsNormalizationFusion {}.into_visitor()),
            &DynFusion(MatMulAddFusion {}.into_visitor()),
            &DynFusion(MatMulScaleFusion {}),
            &DynFusion(AddSoftmaxFusion {}.into_visitor()),
            &DynFusion(TransposeFusion {}),
        ];

        let max_iters = 3;
        for _ in 0..max_iters {
            let n_fused_ops = self.apply_fusions(&mut graph_mut, fusions)?;
            if n_fused_ops == 0 {
                // We reached a fixed point.
                break;
            }
        }

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
    ///
    /// Returns the number of fusions that were applied.
    fn apply_fusions(
        &self,
        graph: &mut GraphMutator,
        visitors: &[&dyn DynFusionVisitor],
    ) -> Result<usize, OptimizeError> {
        // Create the prepared state once and then re-use it for each operator
        // visited.
        let prepared_state: Vec<_> = visitors.iter().map(|f| f.prepare(graph.graph())).collect();

        let n_fusions = graph.apply_fusion(|graph, op_node_id, op_node| {
            for (visitor, state) in visitors.iter().zip(&prepared_state) {
                if let Some(fusion) =
                    visitor.maybe_fuse(state.as_ref(), graph.graph(), op_node_id, op_node)
                {
                    return Some(fusion);
                }
            }
            None
        });

        Ok(n_fusions)
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// A dyn-compatible version of [`FusionVisitor`].
///
/// This replaces the associated `State` associated type with `dyn Any`.
trait DynFusionVisitor {
    fn prepare(&self, graph: &Graph) -> Box<dyn Any>;
    fn maybe_fuse(
        &self,
        state: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion>;
}

/// Wraps a fusion visitor to implement [`DynFusionVisitor`].
struct DynFusion<F: FusionVisitor>(F);

impl<F: FusionVisitor> DynFusionVisitor for DynFusion<F>
where
    F::State: Any,
{
    fn prepare(&self, graph: &Graph) -> Box<dyn Any> {
        Box::new(self.0.prepare(graph))
    }

    fn maybe_fuse(
        &self,
        state: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let state = state.downcast_ref().unwrap();
        self.0.maybe_fuse(state, graph, op_node_id, op_node)
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
