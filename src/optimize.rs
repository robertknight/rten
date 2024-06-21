use std::error::Error;
use std::fmt::{Display, Formatter};

use rten_tensor::Tensor;
use rustc_hash::FxHashMap;

use crate::downcast::DowncastDyn;
use crate::graph::{Constant, ConstantNode, Graph, Node, NodeId, OperatorNode, RunError};
use crate::ops::fused::FusedTranspose;
use crate::ops::{MatMul, Mul, Operator, Sigmoid, Silu, Transpose};
use crate::Output;

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

/// Optimized graph produced by [`GraphOptimizer`].
pub struct OptimizedGraph {
    /// The optimized graph.
    pub graph: Graph,

    /// IDs of input nodes. These correspond to the input IDs passed to
    /// [`GraphOptimizer::optimize`].
    pub input_ids: Vec<NodeId>,

    /// IDs of output nodes. These correspond to the output IDs passed to
    /// [`GraphOptimizer::optimize`].
    pub output_ids: Vec<NodeId>,
}

/// Holds a [`Graph`] and associated data structures while it is being mutated
/// by an optimizer, and provides operations to update the graph.
struct GraphMutator {
    /// Map of (value_node_id, operator_node_ids) for each value node that
    /// is an input to one or more operators.
    edges: FxHashMap<NodeId, Vec<NodeId>>,
    graph: Graph,
    output_ids: Vec<usize>,
}

impl GraphMutator {
    fn from_graph(graph: Graph, output_ids: &[usize]) -> GraphMutator {
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
            edges,
            graph,
            output_ids: output_ids.to_vec(),
        }
    }

    /// Add a new constant value to the graph.
    fn add_constant<T>(&mut self, name: Option<&str>, value: Tensor<T>) -> NodeId
    where
        Constant: From<ConstantNode<T>>,
    {
        self.graph.add_constant(name, value)
    }

    /// Add a new operator to the graph with a single output node.
    ///
    /// Returns the ID of the output node.
    fn add_operator(
        &mut self,
        name: Option<&str>,
        op: Box<dyn Operator + Send + Sync>,
        inputs: &[Option<NodeId>],
    ) -> NodeId {
        let op_output_id = self.graph.add_value(None, None);
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

    fn into_graph_and_output_ids(self) -> (Graph, Vec<NodeId>) {
        (self.graph, self.output_ids)
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
    fn apply_fusion<F: Fn(&Self, &OperatorNode) -> Option<Fusion>>(&mut self, create_fusion: F) {
        let mut fusions = Vec::new();

        for (_, op_node) in self.iter_operators() {
            if let Some(fusion) = create_fusion(self, op_node) {
                fusions.push(fusion);
            }
        }

        for fusion in fusions {
            fusion.apply(self);
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
    old_output_id: NodeId,
}

impl Fusion {
    /// Create a fusion with a given operator, name and input nodes.
    ///
    /// `old_output_id` specifies the output ID of the subgraph that this fusion
    /// replaces.
    fn from_op<Op: Operator + Send + Sync>(
        name: Option<&str>,
        op: Op,
        input_ids: &[NodeId],
        old_output_id: NodeId,
    ) -> Fusion {
        Fusion {
            name: name.map(|s| s.to_string()),
            fused_op: Box::new(op),
            input_ids: input_ids.iter().copied().map(Some).collect(),
            old_output_id,
        }
    }

    /// Apply the fusion to the graph.
    ///
    /// This adds the fused operator to the graph and replaces references to
    /// the original output nodes with the fused operator's outputs.
    fn apply(self, graph: &mut GraphMutator) {
        let Fusion {
            name,
            fused_op,
            input_ids,
            old_output_id,
        } = self;

        let fused_op_output_id = graph.add_operator(name.as_deref(), fused_op, &input_ids);
        graph.replace_value(old_output_id, fused_op_output_id);
    }
}

/// Utilities for matching patterns in a graph.
trait OperatorMatch {
    /// Test if an operator node matches a given operator and has N inputs and
    /// M outputs.
    fn match_type<Op: Operator, const N: usize, const M: usize>(
        &self,
    ) -> Option<(&Op, [usize; N], [usize; M])>;
}

impl OperatorMatch for OperatorNode {
    fn match_type<Op: Operator, const N: usize, const M: usize>(
        &self,
    ) -> Option<(&Op, [usize; N], [usize; M])> {
        let op = self.operator().downcast_ref::<Op>()?;

        let input_ids = self.input_ids();
        if input_ids.len() != N || input_ids.iter().any(|n| n.is_none()) {
            return None;
        }

        let output_ids = self.output_ids();
        if output_ids.len() != M || output_ids.iter().any(|n| n.is_none()) {
            return None;
        }

        let input_ids: [usize; N] = std::array::from_fn(|i| input_ids[i].unwrap());
        let output_ids: [usize; M] = std::array::from_fn(|i| output_ids[i].unwrap());

        Some((op, input_ids, output_ids))
    }
}

/// Return true if `a` and `b`, viewed as sets, contain the same elements.
fn array_sets_equal<T: PartialEq + Ord, const N: usize>(mut a: [T; N], mut b: [T; N]) -> bool {
    a.sort_unstable();
    b.sort_unstable();
    a == b
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
    /// The input and output nodes specified by `input_ids` and `output_ids`
    /// will be preserved, but their IDs may change. Other nodes in the graph
    /// may be modified, removed or replaced by optimization.
    ///
    /// This method returns the new graph along with the node IDs in the new
    /// graph that correspond to `input_ids` and `output_ids`.
    pub fn optimize(
        &self,
        graph: Graph,
        input_ids: &[NodeId],
        output_ids: &[NodeId],
    ) -> Result<OptimizedGraph, OptimizeError> {
        let mut graph_mut = GraphMutator::from_graph(graph, output_ids);

        self.propagate_constants(&mut graph_mut)?;

        self.fuse_transpose_matmul(&mut graph_mut)?;
        self.fuse_silu(&mut graph_mut)?;

        let (graph, output_ids) = graph_mut.into_graph_and_output_ids();

        Ok(OptimizedGraph {
            graph,
            input_ids: input_ids.to_vec(),
            output_ids,
        })
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
            let const_id = match output {
                Output::FloatTensor(tensor) => graph.add_constant(const_name.as_deref(), tensor),
                Output::IntTensor(tensor) => graph.add_constant(const_name.as_deref(), tensor),
            };
            graph.replace_value(value_node_id, const_id);
        }

        Ok(())
    }

    /// Fuse `MatMul(Transpose(X), Y)`.
    ///
    /// The `MatMul` inputs can be in either order.
    fn fuse_transpose_matmul(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        graph.apply_fusion(|edges, op_node| {
            let (transpose_op, [transpose_input], [transpose_output]) =
                op_node.match_type::<Transpose, 1, 1>()?;

            let transpose_target = edges.find_operator_with_input(transpose_output)?;

            let (matmul_op, matmul_inputs, [matmul_output]) =
                transpose_target.match_type::<MatMul, 2, 1>()?;
            let [matmul_input_a, matmul_input_b] = matmul_inputs;

            let fused_input = if matmul_input_a == transpose_output {
                [transpose_input, matmul_input_b]
            } else {
                [matmul_input_a, transpose_input]
            };

            let fused_op = FusedTranspose::wrap(
                Box::new(matmul_op.clone()),
                if matmul_input_a == transpose_output {
                    0
                } else {
                    1
                },
                transpose_op.perm.as_deref(),
            );

            Some(Fusion::from_op(
                transpose_target.name(),
                fused_op,
                &fused_input,
                matmul_output,
            ))
        });

        Ok(())
    }

    /// Fuse `x * Sigmoid(x)` into `Silu(x)`.
    fn fuse_silu(&self, graph: &mut GraphMutator) -> Result<(), OptimizeError> {
        graph.apply_fusion(|graph, op_node| {
            let (_sigmoid_op, [sigmoid_input], [sigmoid_output]) =
                op_node.match_type::<Sigmoid, 1, 1>()?;
            let sigmoid_target = graph.find_operator_with_input(sigmoid_output)?;

            let (_mul_op, mul_inputs, [mul_output]) = sigmoid_target.match_type::<Mul, 2, 1>()?;

            array_sets_equal(mul_inputs, [sigmoid_input, sigmoid_output]).then(|| {
                Fusion::from_op(sigmoid_target.name(), Silu {}, &[sigmoid_input], mul_output)
            })
        });

        Ok(())
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::Tensor;

    use super::{GraphOptimizer, OptimizeError, OptimizedGraph};
    use crate::graph::{Constant, Graph, Node, NodeId, OperatorNode};
    use crate::ops::{Add, MatMul, Mul, Operator, Sigmoid, Transpose};

    /// Extensions to [`Graph`] to make tests easier to write.
    trait GraphTestUtils {
        /// Add a single-output operator to the graph and return a tuple of
        /// `(operator_node_id, output_node_id)`.
        fn add_simple_op<Op: Operator + Send + Sync + 'static>(
            &mut self,
            name: &str,
            op: Op,
            input_ids: &[NodeId],
        ) -> (NodeId, NodeId);

        fn get_operator(&self, node_id: NodeId) -> Option<&OperatorNode>;
        fn get_constant(&self, node_id: NodeId) -> Option<&Constant>;
    }

    impl GraphTestUtils for Graph {
        fn add_simple_op<Op: Operator + Send + Sync + 'static>(
            &mut self,
            name: &str,
            op: Op,
            input_ids: &[NodeId],
        ) -> (NodeId, NodeId) {
            let op_out_name = format!("{}_out", name);
            let op_out_id = self.add_value(Some(&op_out_name), None);
            let input_ids: Vec<_> = input_ids.iter().copied().map(Some).collect();
            let op_node_id =
                self.add_op(Some(name), Box::new(op), &input_ids, &[op_out_id].map(Some));
            (op_node_id, op_out_id)
        }

        fn get_operator(&self, node_id: NodeId) -> Option<&OperatorNode> {
            match self.get_node(node_id) {
                Some(Node::Operator(op)) => Some(op),
                _ => None,
            }
        }

        fn get_constant(&self, node_id: NodeId) -> Option<&Constant> {
            match self.get_node(node_id) {
                Some(Node::Constant(constant)) => Some(constant),
                _ => None,
            }
        }
    }

    fn optimize_graph(
        graph: Graph,
        input_ids: &[NodeId],
        output_ids: &[NodeId],
    ) -> Result<(Graph, Vec<NodeId>), OptimizeError> {
        let optimizer = GraphOptimizer::new();
        let OptimizedGraph {
            graph, output_ids, ..
        } = optimizer.optimize(graph, input_ids, output_ids)?;
        Ok((graph, output_ids))
    }

    /// Return the operator node which produces a given output value node.
    fn source_operator(graph: &Graph, output_id: NodeId) -> Option<&OperatorNode> {
        graph.iter().find_map(|(_, node)| match node {
            Node::Operator(op) => (op.output_ids() == &[Some(output_id)]).then_some(op),
            _ => None,
        })
    }

    #[test]
    fn test_constant_propagation() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();

        // Add an operator with constant inputs.
        let const_a = graph.add_constant(Some("const_a"), Tensor::from([1, 2, 3]));
        let const_b = graph.add_constant(Some("const_b"), Tensor::from([4, 5, 6]));
        let (_, add_out) = graph.add_simple_op("add_1", Add {}, &[const_a, const_b]);

        // Add an operator with a dynamic input and the output of the previous operator.
        let input = graph.add_value(Some("input"), None);
        let (add_op_2, add_2_out) = graph.add_simple_op("add_2", Add {}, &[add_out, input]);

        // Optimize the graph. This should replace the first operator's output
        // with a constant value.
        let optimizer = GraphOptimizer::new();
        let OptimizedGraph {
            graph: optimized_graph,
            input_ids: optimized_graph_input_ids,
            output_ids: optimized_graph_output_ids,
        } = optimizer.optimize(graph, &[input], &[add_out, add_2_out])?;

        // Check that we got the expected inputs and outputs. The optimizer
        // does not promise to preserve IDs for unmodified parts of the graph,
        // but the current implementation does.
        assert_eq!(optimized_graph_input_ids, &[input]);
        assert_ne!(optimized_graph_output_ids[0], add_out);
        assert_eq!(optimized_graph_output_ids[1], add_2_out);

        // Check first output was replaced with constant.
        let replaced_node = optimized_graph
            .get_constant(optimized_graph_output_ids[0])
            .unwrap();
        let Constant::Int(const_int) = replaced_node else {
            return Err("constant not an int".into());
        };
        assert_eq!(const_int.view(), Tensor::from([5, 7, 9]));

        // Check input to second operator was replaced with constant.
        let op = optimized_graph.get_operator(add_op_2).unwrap();
        let input_ids: Vec<_> = op.input_ids().iter().map(|id| id.unwrap()).collect();
        assert_eq!(input_ids.len(), 2);
        assert_ne!(input_ids[0], add_out);
        assert_eq!(input_ids[0], optimized_graph_output_ids[0]);
        assert_eq!(input_ids[1], input);

        Ok(())
    }

    #[test]
    fn test_fuse_transpose_matmul() {
        let mut graph = Graph::new();

        let input_1 = graph.add_value(None, None);
        let input_2 = graph.add_value(None, None);

        let (_, transpose_out) =
            graph.add_simple_op("transpose", Transpose { perm: None }, &[input_1]);
        let (_, matmul_out) = graph.add_simple_op("matmul", MatMul {}, &[transpose_out, input_2]);

        let (graph, new_output_ids) =
            optimize_graph(graph, &[input_1, input_2], &[matmul_out]).unwrap();

        let op = source_operator(&graph, new_output_ids[0]).unwrap();
        assert_eq!(op.operator().name(), "FusedTranspose(MatMul)");
        assert_eq!(op.name(), Some("matmul"));
    }

    #[test]
    fn test_fuse_silu() {
        let mut graph = Graph::new();

        let input = graph.add_value(None, None);
        let (_, sigmoid_out) = graph.add_simple_op("sigmoid", Sigmoid {}, &[input]);
        let (_, mul_out) = graph.add_simple_op("mul", Mul {}, &[input, sigmoid_out]);

        let (graph, new_output_ids) = optimize_graph(graph, &[input], &[mul_out]).unwrap();

        let op = source_operator(&graph, new_output_ids[0]).unwrap();
        assert_eq!(op.operator().name(), "Silu");
        assert_eq!(op.name(), Some("mul"));
    }

    #[test]
    fn test_optimize_error() {
        let graph = Graph::new();
        let optimizer = GraphOptimizer::new();
        let invalid_id = 123;
        let result = optimizer.optimize(graph, &[invalid_id], &[invalid_id]);
        assert!(matches!(result, Err(OptimizeError::RunError(_))));
    }
}
