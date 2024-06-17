use std::error::Error;
use std::fmt::{Display, Formatter};

use rustc_hash::FxHashMap;

use crate::graph::{Graph, Node, NodeId, RunError};
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
        mut graph: Graph,
        input_ids: &[NodeId],
        output_ids: &[NodeId],
    ) -> Result<OptimizedGraph, OptimizeError> {
        let mut output_ids = output_ids.to_vec();
        self.propagate_constants(&mut graph, &mut output_ids)?;

        Ok(OptimizedGraph {
            graph,
            input_ids: input_ids.to_vec(),
            output_ids,
        })
    }

    /// Apply constant propagation to replace parts of the graph which depend
    /// only on constant values with a pre-computed constant.
    fn propagate_constants(
        &self,
        graph: &mut Graph,
        output_ids: &mut [NodeId],
    ) -> Result<(), OptimizeError> {
        // Map of (value_node_id, operator_node_ids) for each value node that is
        // an input to one or more operators.
        let edges: FxHashMap<NodeId, Vec<NodeId>> = graph.iter().fold(
            FxHashMap::default(),
            |mut edges, (node_id, node)| match node {
                Node::Operator(op_node) => {
                    for edge_start in op_node.input_ids().flatten() {
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

        // Do a partial run with no inputs. This evaluates all nodes that
        // transitively depend only on constants.
        let leaves = graph
            .partial_run(vec![], output_ids, None)
            .map_err(OptimizeError::RunError)?;

        // Take the resulting (value_node_id, value) list, create new constant
        // nodes in the graph with the value and replace references to
        // `value_node_id` in operator inputs and model outputs with the new
        // constant.
        for (node_id, output) in leaves {
            let const_name = graph
                .get_node(node_id)
                .and_then(|n| n.name())
                .map(|name| name.to_string());
            let const_id = match output {
                Output::FloatTensor(tensor) => graph.add_constant(const_name.as_deref(), tensor),
                Output::IntTensor(tensor) => graph.add_constant(const_name.as_deref(), tensor),
            };

            if let Some(operator_ids) = edges.get(&node_id) {
                for &op_id in operator_ids {
                    let Some(Node::Operator(op_node)) = graph.get_node_mut(op_id) else {
                        panic!("operator node not found");
                    };
                    op_node.replace_input(node_id, const_id);
                }
            };

            for output_id in output_ids.iter_mut().filter(|id| **id == node_id) {
                *output_id = const_id;
            }
        }

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
    use crate::ops::{Add, Operator};

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
        let input_ids: Vec<_> = op.input_ids().map(|id| id.unwrap()).collect();
        assert_eq!(input_ids.len(), 2);
        assert_ne!(input_ids[0], add_out);
        assert_eq!(input_ids[0], optimized_graph_output_ids[0]);
        assert_eq!(input_ids[1], input);

        Ok(())
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
