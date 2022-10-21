use std::collections::{HashMap, HashSet};

use crate::ops::{Input, Operator};
use crate::tensor::Tensor;
use crate::timer::Timer;

struct OperatorNode {
    inputs: Vec<NodeId>,
    output: NodeId,
    operator: Box<dyn Operator>,
}

enum Constant {
    Float(Tensor<f32>),
    Int(Tensor<i32>),
}

impl From<Tensor<f32>> for Constant {
    fn from(t: Tensor<f32>) -> Constant {
        Constant::Float(t)
    }
}

impl From<Tensor<i32>> for Constant {
    fn from(t: Tensor<i32>) -> Constant {
        Constant::Int(t)
    }
}

enum Node {
    Operator(OperatorNode),
    Constant(Constant),
    Value,
}

pub type NodeId = usize;

pub struct Graph {
    nodes: Vec<Node>,
}

pub struct RunOptions {
    /// Whether to log operator timing as the graph executes
    pub timing: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions { timing: false }
    }
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
    pub fn add_constant<T: Into<Constant>>(&mut self, value: T) -> NodeId {
        let constant: Constant = value.into();
        self.nodes.push(Node::Constant(constant));
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
    pub fn run(
        &self,
        inputs: &[(NodeId, &Tensor)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Vec<Tensor> {
        let plan = self.create_plan(inputs, outputs);
        let opts = opts.unwrap_or_default();

        let mut run_timer = Timer::new();
        if opts.timing {
            run_timer.start();
        }

        // Collect operator inputs
        let mut values: HashMap<NodeId, Input> = inputs
            .iter()
            .copied()
            .map(|(id, t)| (id, t.into()))
            .collect();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let Node::Constant(constant) = node {
                let input = match constant {
                    Constant::Float(t) => t.into(),
                    Constant::Int(t) => t.into(),
                };
                values.insert(node_id, input);
            }
        }

        // Count how often each temporary input is used, so we can free them
        // when no longer needed.
        let mut usage_counts: HashMap<NodeId, usize> = HashMap::new();
        for (_, op_node) in plan.iter() {
            for node_id in op_node.inputs.iter() {
                usage_counts
                    .entry(*node_id)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        // Execute the plan
        let mut temp_values: HashMap<NodeId, Tensor> = HashMap::new();
        let mut op_elapsed: HashMap<&str, f32> = HashMap::new();

        for (op_node_id, op_node) in plan.iter() {
            let mut op_timer = Timer::new();
            if opts.timing {
                op_timer.start();
            }

            let mut input_shapes = Vec::new();

            // Test if the operator can be run in-place to save allocations.
            // This requires that the inputs are temporary values produced by
            // earlier ops, and that they are not going to be needed by other
            // ops in future.
            let can_run_in_place = op_node.inputs.len() == 1
                && op_node.operator.can_run_in_place()
                && temp_values.contains_key(&op_node.inputs[0])
                && usage_counts.get(&op_node.inputs[0]) == Some(&1);

            let output = if can_run_in_place {
                let mut input = temp_values.remove(&op_node.inputs[0]).unwrap();
                op_node.operator.run_in_place(&mut input);
                input
            } else {
                let mut op_inputs: Vec<Input> = Vec::new();
                for node_id in op_node.inputs.iter() {
                    if let Some(&value) = values.get(node_id) {
                        op_inputs.push(value);
                    } else if let Some(value) = temp_values.get(node_id) {
                        op_inputs.push(value.into());
                    } else {
                        // If this is reached, there was a bug in plan creation.
                        panic!(
                            "Invalid plan did not produce input value {} for operator {}",
                            node_id, op_node_id
                        );
                    }
                }
                input_shapes = op_inputs.iter().map(|x| x.shape()).collect();
                op_node.operator.run(&op_inputs[..])
            };

            if opts.timing {
                op_timer.end();

                if let Some(elapsed) = op_elapsed.get_mut(op_node.operator.name()) {
                    *elapsed += op_timer.elapsed();
                } else {
                    op_elapsed.insert(op_node.operator.name(), op_timer.elapsed());
                }

                println!(
                    "#{} {:?} with {:?} in {}ms",
                    op_node_id,
                    op_node.operator,
                    input_shapes,
                    op_timer.elapsed()
                );
            }

            temp_values.insert(op_node.output, output);

            // Remove temporary values that are no longer needed
            for node_id in op_node.inputs.iter() {
                let usage = *usage_counts.get(node_id).unwrap();
                if usage == 1 {
                    temp_values.remove(node_id);
                } else {
                    usage_counts.insert(*node_id, usage - 1);
                }
            }
        }

        if opts.timing {
            run_timer.end();
            println!(
                "Graph run of {} ops finished in {}ms",
                plan.len(),
                run_timer.elapsed()
            );

            // Display cumulative times for each op type, sorted by op name
            let mut op_timings: Vec<_> = op_elapsed.iter().collect();
            op_timings.sort_by(|a, b| a.0.cmp(b.0));
            for (op_name, total_time) in op_timings.iter() {
                println!("  {} {}ms", op_name, total_time);
            }
        }

        // Return the requested outputs
        outputs
            .iter()
            .map(|output_id| {
                if let Some(&value) = values.get(output_id) {
                    value.as_float().unwrap().clone()
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
    ///
    /// Any node IDs in `outputs` which reference constant or input values are
    /// omitted from the plan.
    fn create_plan(
        &self,
        inputs: &[(NodeId, &Tensor)],
        outputs: &[NodeId],
    ) -> Vec<(NodeId, &OperatorNode)> {
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
            if let Node::Constant(_) = node {
                resolved_values.insert(node_id);
            }
        }

        // Build an execution plan via a depth first traversal of the graph
        // starting at the output nodes. A helper struct is used as recursive
        // closures are not supported in Rust.
        struct PlanBuilder<'a> {
            resolved_values: HashSet<NodeId>,
            plan: Vec<(NodeId, &'a OperatorNode)>,
            operator_nodes: HashMap<NodeId, &'a OperatorNode>,
        }
        impl<'a> PlanBuilder<'a> {
            fn visit(&mut self, node_id: NodeId, op_node: &'a OperatorNode) {
                for input in op_node.inputs.iter() {
                    if self.resolved_values.contains(input) {
                        continue;
                    }
                    if let Some(input_op_node) = self.operator_nodes.get(input) {
                        self.visit(*input, input_op_node);
                    } else {
                        panic!("Unable to generate execution plan. Missing value {}", input)
                    }
                }
                self.resolved_values.insert(node_id);
                self.plan.push((node_id, op_node));
            }

            fn plan(mut self, outputs: &[NodeId]) -> Vec<(NodeId, &'a OperatorNode)> {
                for output_id in outputs.iter() {
                    if let Some(op_node) = self.operator_nodes.get(output_id) {
                        self.visit(*output_id, op_node);
                    } else if !self.resolved_values.contains(output_id) {
                        panic!(
                            "Unable to generate execution plan. Missing value {}",
                            output_id
                        )
                    }
                }
                self.plan
            }
        }

        let builder = PlanBuilder {
            resolved_values,
            plan: Vec::new(),
            operator_nodes,
        };
        builder.plan(outputs)
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::Graph;
    use crate::ops::{Concat, Conv2d, Input, Operator, ReLU};
    use crate::tensor::{from_data, zero_tensor, Tensor};
    use crate::test_util::expect_equal;

    // Test of a very simple graph with a typical structure (one input, one
    // output, Conv2d + ReLU operation).
    #[test]
    fn test_graph_run() -> Result<(), String> {
        let mut g = Graph::new();

        let weights = from_data(
            vec![1, 1, 3, 3],
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
            vec![1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let results = g.run(&[(input_id, &input)], &[relu_out], None);

        let expected = from_data(
            vec![1, 1, 3, 3],
            vec![
                1.5202, 1.5592, 0.9939, 1.7475, 2.6358, 1.3428, 1.0165, 1.1806, 0.8685,
            ],
        );
        assert_eq!(results.len(), 1);
        expect_equal(&results[0], &expected)
    }

    #[derive(Debug)]
    struct AddOne {}
    impl Operator for AddOne {
        fn name(&self) -> &str {
            "AddOne"
        }

        fn run(&self, inputs: &[Input]) -> Tensor {
            let input = inputs[0].as_float().unwrap();
            let output_data = input.elements().map(|x| x + 1.0).collect();
            from_data(input.shape().into(), output_data)
        }
    }

    #[test]
    fn test_graph_planning_order() -> Result<(), String> {
        let mut g = Graph::new();

        let input_id = g.add_value();

        let op_a = g.add_op(Box::new(AddOne {}), &[input_id]);
        let op_b = g.add_op(Box::new(AddOne {}), &[op_a]);

        // op_c has both op_a and op_b as inputs. Since op_b depends on op_a,
        // execution must run op_a, then op_b, then op_c.
        let op_c = g.add_op(Box::new(Concat { dim: 0 }), &[op_a, op_b]);

        // op_d is the same as op_c, but input order is reversed
        let op_d = g.add_op(Box::new(Concat { dim: 0 }), &[op_b, op_a]);

        let input = from_data(vec![1], vec![1.]);

        let results = g.run(&[(input_id, &input)], &[op_c], None);
        let expected = from_data(vec![2], vec![2., 3.]);
        expect_equal(&results[0], &expected)?;

        let results = g.run(&[(input_id, &input)], &[op_d], None);
        let expected = from_data(vec![2], vec![3., 2.]);
        expect_equal(&results[0], &expected)
    }

    #[test]
    fn test_graph_many_steps() -> Result<(), String> {
        let mut g = Graph::new();

        let input = from_data(vec![5], vec![1., 2., 3., 4., 5.]);
        let input_id = g.add_value();

        let mut prev_output = input_id;
        for _ in 0..100 {
            prev_output = g.add_op(Box::new(AddOne {}), &[prev_output]);
        }

        let results = g.run(&[(input_id, &input)], &[prev_output], None);

        let expected = from_data(vec![5], vec![101., 102., 103., 104., 105.]);
        expect_equal(&results[0], &expected)
    }

    #[test]
    fn test_noop_graph() -> Result<(), String> {
        let mut g = Graph::new();

        let input = from_data(vec![5], vec![1., 2., 3., 4., 5.]);
        let input_id = g.add_value();

        let results = g.run(&[(input_id, &input)], &[input_id], None);

        expect_equal(&results[0], &input)
    }

    #[test]
    fn test_constant_graph() -> Result<(), String> {
        let mut g = Graph::new();

        let value = from_data(vec![5], vec![1., 2., 3., 4., 5.]);
        let const_id = g.add_constant(value.clone());

        let results = g.run(&[], &[const_id], None);

        expect_equal(&results[0], &value)
    }

    #[test]
    fn test_no_outputs() {
        let g = Graph::new();
        let results = g.run(&[], &[], None);
        assert_eq!(results.len(), 0);
    }

    #[test]
    #[should_panic(expected = "Unable to generate execution plan. Missing value 123")]
    fn test_panic_if_invalid_output() {
        let g = Graph::new();
        g.run(&[], &[123], None);
    }

    #[test]
    #[should_panic(expected = "Unable to generate execution plan. Missing value 42")]
    fn test_panic_if_missing_operator_input() {
        let mut g = Graph::new();
        let output = g.add_op(Box::new(ReLU {}), &[42]);
        g.run(&[], &[output], None);
    }

    #[derive(Debug)]
    struct AddOneInPlace {}
    impl Operator for AddOneInPlace {
        fn name(&self) -> &str {
            "AddOneInPlace"
        }

        fn can_run_in_place(&self) -> bool {
            true
        }

        fn run(&self, inputs: &[Input]) -> Tensor {
            // An operator should normally have the same behavior in `run`
            // and `run_in_place`. Here we use different behavior to make it
            // possible to distinguish which path was used.
            inputs[0].as_float().unwrap().clone()
        }

        fn run_in_place(&self, input: &mut Tensor) {
            for x in input.data_mut().iter_mut() {
                *x = *x + 1.0;
            }
        }
    }

    #[test]
    fn test_runs_op_in_place() {
        let mut g = Graph::new();
        let input_id = g.add_value();

        let op1_id = g.add_op(Box::new(AddOneInPlace {}), &[input_id]);
        let op2_id = g.add_op(Box::new(AddOneInPlace {}), &[op1_id]);
        let op3_id = g.add_op(Box::new(AddOneInPlace {}), &[op2_id]);
        let op4_id = g.add_op(Box::new(AddOneInPlace {}), &[op2_id]);
        let input = zero_tensor(&[1, 1]);

        // First operator should not be run in-place, since it has an
        // immutable input. The result should be the same as the input.
        let results = g.run(&[(input_id, &input)], &[op1_id], None);
        assert_eq!(results[0][[0, 0]], 0.0);

        // Second operator should be run in-place, as it meets all the
        // requirements for this optimization.
        let results = g.run(&[(input_id, &input)], &[op2_id], None);
        assert_eq!(results[0][[0, 0]], 1.0);

        // Third op should not be run in place, because its input is re-used
        // for fourth op. Fourth op can run in place as by then, it is the
        // only consumer of its input.
        let results = g.run(&[(input_id, &input)], &[op3_id, op4_id], None);
        assert_eq!(results[0][[0, 0]], 1.0);
        assert_eq!(results[1][[0, 0]], 2.0);
    }
}
