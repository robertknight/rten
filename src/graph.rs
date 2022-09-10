use std::collections::{HashMap, HashSet};

use crate::ops::Operator;
use crate::tensor::Tensor;

struct OperatorNode {
    inputs: Vec<NodeId>,
    output: NodeId,
    operator: Box<dyn Operator>,
}

enum Node {
    Operator(OperatorNode),
    Constant(Tensor),
    Value,
}

pub type NodeId = usize;

pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    /// Create a new empty dataflow graph.
    pub fn new() -> Graph {
        Graph { nodes: Vec::new() }
    }

    /// Add an operator node to the graph.
    ///
    /// `inputs` specifies which other nodes in the graph should be used as
    /// inputs to this operation when the graph is executed. These other nodes
    /// can be inputs, constants (for weights and biases) or outputs of other
    /// operators.
    ///
    /// Returns the ID of the added operator's output node.
    pub fn add_op(&mut self, op: Box<dyn Operator>, inputs: &[NodeId]) -> NodeId {
        let output_id = self.add_value();
        self.nodes.push(Node::Operator(OperatorNode {
            inputs: Vec::from(inputs),
            output: output_id,
            operator: op,
        }));
        output_id
    }

    /// Add a constant node to the graph.
    ///
    /// Returns the ID of the added node.
    pub fn add_constant(&mut self, value: Tensor) -> NodeId {
        self.nodes.push(Node::Constant(value));
        self.nodes.len() - 1
    }

    /// Add a value node to the graph.
    ///
    /// This serves as a placeholder for a value which is available only when
    /// the graph is executed, such as an input or operator output.
    ///
    /// Returns the ID of the added node.
    pub fn add_value(&mut self) -> NodeId {
        self.nodes.push(Node::Value);
        self.nodes.len() - 1
    }

    /// Compute a set of output values given a set of inputs, using the
    /// processing steps and constant values defined by the graph.
    pub fn run(&self, inputs: &[(NodeId, &Tensor)], outputs: &[NodeId]) -> Vec<Tensor> {
        let plan = self.create_plan(inputs, outputs);

        // Collect operator inputs
        let mut values: HashMap<NodeId, &Tensor> = inputs.iter().map(|x| *x).collect();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let Node::Constant(tensor) = node {
                values.insert(node_id, tensor);
            }
        }

        // Execute the plan
        let mut temp_values: HashMap<NodeId, Tensor> = HashMap::new();
        for op_node in plan.iter() {
            let mut op_inputs = Vec::new();
            for node_id in op_node.inputs.iter() {
                if let Some(value) = values.get(&node_id) {
                    op_inputs.push(*value);
                } else if let Some(value) = temp_values.get(&node_id) {
                    op_inputs.push(value);
                } else {
                    panic!("Unable to find operator input {}", node_id);
                }
            }
            let output = op_node.operator.run(&op_inputs[..]);
            temp_values.insert(op_node.output, output);
            // TODO - Remove temporary inputs that are no longer needed
        }

        // Return the requested outputs
        outputs
            .iter()
            .map(|output_id| {
                if let Some(&value) = values.get(output_id) {
                    value.clone()
                } else if let Some(value) = temp_values.remove(output_id) {
                    value
                } else {
                    unreachable!()
                }
            })
            .collect()
    }

    /// Create an execution plan for a sequence of computation steps that begin
    /// with `inputs` and eventually produces `outputs`.
    fn create_plan(&self, inputs: &[(NodeId, &Tensor)], outputs: &[NodeId]) -> Vec<&OperatorNode> {
        // Map of output node to source operator
        let operator_nodes: HashMap<NodeId, &OperatorNode> = self
            .nodes
            .iter()
            .filter_map(|node| match node {
                Node::Operator(op_node) => Some((op_node.output, op_node)),
                _ => None,
            })
            .collect();

        // Set of values that are available after executing the plan
        let mut resolved_values: HashSet<NodeId> =
            inputs.iter().map(|(node_id, _)| *node_id).collect();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let Node::Constant(tensor) = node {
                resolved_values.insert(node_id);
            }
        }

        // Needed values that are not yet produced by the plan
        let mut missing_values: Vec<NodeId> = outputs
            .iter()
            .filter(|it| !resolved_values.contains(it))
            .map(|x| *x)
            .collect();

        // Build execution plan in reverse order
        let mut plan: Vec<&OperatorNode> = Vec::new();
        while let Some(missing_value) = missing_values.pop() {
            if let Some(op_node) = operator_nodes.get(&missing_value) {
                resolved_values.insert(missing_value);
                plan.push(&op_node);
                for input in op_node.inputs.iter() {
                    if !resolved_values.contains(&input) {
                        missing_values.push(*input);
                    }
                }
            } else {
                panic!(
                    "Unable to generate execution plan. Missing value {}",
                    missing_value
                );
            }
        }
        plan.reverse();

        plan
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::Graph;
    use crate::ops::{Conv2d, Operator, ReLU};
    use crate::tensor::{from_data, Tensor};

    /// Check that the shapes of two tensors are equal and that their contents
    /// are approximately equal.
    fn expect_equal(x: &Tensor, y: &Tensor) -> Result<(), String> {
        if x.shape != y.shape {
            return Err(format!(
                "Tensors have different shapes. {:?} vs. {:?}",
                &x.shape, &y.shape
            ));
        }

        let eps = 0.001;
        for i in 0..x.data.len() {
            let xi = x.data[i];
            let yi = y.data[i];

            if (xi - yi).abs() > eps {
                return Err(format!(
                    "Tensor values differ at index {}: {} vs {}",
                    i, xi, yi
                ));
            }
        }

        return Ok(());
    }

    // Test of a very simple graph with a typical structure (one input, one
    // output, Conv2d + ReLU operation).
    #[test]
    fn test_graph_run() -> Result<(), String> {
        let mut g = Graph::new();

        let weights = from_data(
            vec![3, 3, 1, 1],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );
        let weights_id = g.add_constant(weights);
        let input_id = g.add_value();

        let conv_out = g.add_op(
            Box::new(Conv2d {
                padding: (1, 1),
                groups: 1,
            }),
            &[input_id, weights_id],
        );
        let relu_out = g.add_op(Box::new(ReLU {}), &[conv_out]);

        let input = from_data(
            vec![3, 3, 1],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let results = g.run(&[(input_id, &input)], &[relu_out]);

        let expected = from_data(
            vec![3, 3, 1],
            vec![
                1.5202, 1.5592, 0.9939, 1.7475, 2.6358, 1.3428, 1.0165, 1.1806, 0.8685,
            ],
        );
        assert_eq!(results.len(), 1);
        expect_equal(&results[0], &expected)
    }

    struct AddOne {}
    impl Operator for AddOne {
        fn run(&self, inputs: &[&Tensor]) -> Tensor {
            let input = inputs[0];
            let output_data = input.data.iter().map(|x| x + 1.0).collect();
            Tensor {
                shape: input.shape.clone(),
                data: output_data,
            }
        }
    }

    #[test]
    fn test_graph_many_steps() -> Result<(), String> {
        let mut g = Graph::new();

        let input = from_data(vec![5], vec![1., 2., 3., 4., 5.]);
        let input_id = g.add_value();

        let mut prev_output = input_id;
        for i in 0..100 {
            prev_output = g.add_op(Box::new(AddOne {}), &[prev_output]);
        }

        let results = g.run(&[(input_id, &input)], &[prev_output]);

        let expected = from_data(vec![5], vec![101., 102., 103., 104., 105.]);
        expect_equal(&results[0], &expected)
    }
}
