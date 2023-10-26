use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::iter::zip;

use wasnn_tensor::prelude::*;
use wasnn_tensor::Tensor;

use crate::ops::{Input, InputList, OpError, Operator, Output};
use crate::timer::Timer;

/// Represents the size of a dimension of a runtime-provided value, such as
/// an operator input, output or intermediate value.
#[derive(Clone, Debug, PartialEq)]
pub enum Dimension {
    /// A dimension whose expected size is fixed and specified as part of the
    /// model.
    Fixed(usize),

    /// A dimension whose size is determined at runtime. The symbol provides
    /// a name to identify when different values share a size.
    Symbolic(String),
}

pub struct OperatorNode {
    name: Option<String>,
    inputs: Vec<Option<NodeId>>,
    outputs: Vec<Option<NodeId>>,
    operator: Box<dyn Operator + Sync>,
}

pub struct ValueNode {
    name: Option<String>,
    shape: Option<Vec<Dimension>>,
}

pub struct ConstantNode<T> {
    name: Option<String>,
    data: Tensor<T>,
}

pub enum Constant {
    Float(ConstantNode<f32>),
    Int(ConstantNode<i32>),
}

impl From<ConstantNode<f32>> for Constant {
    fn from(node: ConstantNode<f32>) -> Constant {
        Constant::Float(node)
    }
}

impl From<ConstantNode<i32>> for Constant {
    fn from(node: ConstantNode<i32>) -> Constant {
        Constant::Int(node)
    }
}

pub enum Node {
    Operator(OperatorNode),
    Constant(Constant),
    Value(ValueNode),
}

impl Node {
    /// Return the debug name of this node
    pub fn name(&self) -> Option<&str> {
        let maybe_name = match self {
            Node::Operator(node) => &node.name,
            Node::Constant(constant) => match constant {
                Constant::Float(node) => &node.name,
                Constant::Int(node) => &node.name,
            },
            Node::Value(node) => &node.name,
        };
        maybe_name.as_ref().map(|s| s.as_str())
    }

    /// Return the tensor shape associated with this node.
    ///
    /// For constants this is the shape of the tensor. Operator nodes have no
    /// shape. For values (eg. inputs/outputs) this is the expected shape.
    pub fn shape(&self) -> Option<Vec<Dimension>> {
        let dims_from_fixed_shape =
            |shape: &[usize]| shape.iter().copied().map(Dimension::Fixed).collect();

        match self {
            Node::Operator(_) => None,
            Node::Constant(constant) => match constant {
                Constant::Float(node) => Some(dims_from_fixed_shape(node.data.shape())),
                Constant::Int(node) => Some(dims_from_fixed_shape(node.data.shape())),
            },
            Node::Value(node) => node.shape.clone(),
        }
    }
}

pub type NodeId = usize;

/// A graph defines how to produce output values from a set of dynamic input
/// values and constants, by flowing the inputs through a series of computation
/// steps (operators).
///
/// Graphs consists of three types of node, each of which has a numeric ID and a
/// unique string name. A node in the graph is either a constant value such as
/// weights produced during training, a dynamically supplied or produced input
/// or output value, or a computation step.
pub struct Graph {
    nodes: Vec<Node>,
}

/// Reasons why a graph execution failed
#[derive(Eq, PartialEq, Debug)]
pub enum RunError {
    /// An input or output node ID is invalid
    InvalidNodeId,

    /// A plan could not be constructed that would generate the requested output
    /// from the input.
    PlanningError(String),

    /// Execution of an operator failed
    OperatorError { name: String, error: OpError },

    /// The output of a graph operator did not match expectations (eg. the
    /// count, types or shapes of outputs did not match what was expected.)
    OutputMismatch(&'static str),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::InvalidNodeId => write!(f, "node ID is invalid"),
            RunError::PlanningError(ref err) => write!(f, "planning error {:?}", err),
            RunError::OperatorError {
                name,
                error: ref err,
            } => write!(f, "operator \"{}\" failed: {:?}", name, err),
            RunError::OutputMismatch(err) => write!(f, "output mismatch {:?}", err),
        }
    }
}

/// Return true if all elements in `xs` are unique according to the comparison
/// function `eq`.
///
/// `xs` is assumed to be small enough that comparing all pairs is still fast.
fn all_unique<T, F: Fn(&T, &T) -> bool>(xs: &[T], eq: F) -> bool {
    xs.iter()
        .all(|x| xs.iter().filter(|y| eq(x, y)).count() == 1)
}

/// Counter that tracks the remaining usage count of a graph node value.
///
/// This is used to keep intermediate graph outputs alive until they are no
/// longer needed.
struct NodeRefCount {
    rc: HashMap<NodeId, usize>,
}

impl NodeRefCount {
    fn new() -> NodeRefCount {
        NodeRefCount { rc: HashMap::new() }
    }

    /// Increment ref count of node
    fn inc(&mut self, id: NodeId) {
        self.rc
            .entry(id)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    /// Decrement ref count of node and return new count
    fn dec(&mut self, id: NodeId) -> usize {
        let Some(rc) = self.rc.get_mut(&id) else {
            return 0;
        };
        *rc = rc.saturating_sub(1);
        if *rc == 0 {
            self.rc.remove(&id);
            0
        } else {
            *rc
        }
    }

    fn count(&self, id: NodeId) -> usize {
        *self.rc.get(&id).unwrap_or(&0)
    }
}

impl Error for RunError {}

#[derive(Default)]
pub struct RunOptions {
    /// Whether to log times spent in different operators when run completes.
    pub timing: bool,

    /// Whether to log information about each graph operation as it is executed,
    /// including input shapes and execution time. This will slow down
    /// execution.
    pub verbose: bool,
}

impl Graph {
    /// Create a new empty dataflow graph.
    pub fn new() -> Graph {
        Graph { nodes: Vec::new() }
    }

    /// Add an operator node to the graph.
    ///
    /// `name` is an identifier for this node that is used in debug messages etc.
    ///
    /// `inputs` specifies which other nodes in the graph should be used as
    /// inputs to this operation when the graph is executed. These other nodes
    /// can be inputs, constants (for weights and biases) or outputs of other
    /// operators.
    ///
    /// `outputs` specifies which value nodes the operator's outputs should be
    /// written to.
    ///
    /// Returns the ID of the operator node.
    pub fn add_op(
        &mut self,
        name: Option<&str>,
        op: Box<dyn Operator + Sync>,
        inputs: &[Option<NodeId>],
        outputs: &[Option<NodeId>],
    ) -> NodeId {
        self.nodes.push(Node::Operator(OperatorNode {
            name: name.map(|s| s.to_owned()),
            inputs: Vec::from(inputs),
            outputs: Vec::from(outputs),
            operator: op,
        }));
        self.nodes.len() - 1
    }

    /// Add a constant node to the graph.
    ///
    /// `name` is an identifier for this node that is used in debug messages etc.
    ///
    /// Returns the ID of the added node.
    pub fn add_constant<T>(&mut self, name: Option<&str>, value: Tensor<T>) -> NodeId
    where
        ConstantNode<T>: Into<Constant>,
    {
        let node = ConstantNode {
            name: name.map(|s| s.to_owned()),
            data: value,
        };
        self.nodes.push(Node::Constant(node.into()));
        self.nodes.len() - 1
    }

    /// Add a value node to the graph.
    ///
    /// `name` is an identifier for this node that is used in debug messages etc.
    /// `shape` is the expected shape of the value at runtime, or None if not
    /// known.
    ///
    /// This serves as a placeholder for a value which is available only when
    /// the graph is executed, such as an input or operator output.
    ///
    /// Returns the ID of the added node.
    pub fn add_value(&mut self, name: Option<&str>, shape: Option<Vec<Dimension>>) -> NodeId {
        self.nodes.push(Node::Value(ValueNode {
            name: name.map(|s| s.to_owned()),
            shape,
        }));
        self.nodes.len() - 1
    }

    /// Return the debug name for a node.
    pub fn node_name(&self, id: NodeId) -> String {
        self.get_node(id)
            .and_then(|node| node.name())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("[ID: {}]", id))
    }

    /// Retrieve a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Compute a set of output values given a set of inputs, using the
    /// processing steps and constant values defined by the graph.
    pub fn run(
        &self,
        inputs: &[(NodeId, Input)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let plan = self.create_plan(inputs, outputs)?;
        let opts = opts.unwrap_or_default();

        let mut run_timer = Timer::new();
        if opts.timing {
            run_timer.start();
        }

        // Collect operator inputs
        let mut values: HashMap<NodeId, Input> = inputs.iter().copied().collect();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let Node::Constant(constant) = node {
                let input = match constant {
                    Constant::Float(node) => Input::FloatTensor(&node.data),
                    Constant::Int(node) => Input::IntTensor(&node.data),
                };
                values.insert(node_id, input);
            }
        }

        // Count how often each temporary output is used, so we can free them
        // when no longer needed.
        let mut temp_value_refcount = NodeRefCount::new();
        for (_, op_node) in plan.iter() {
            for node_id in op_node.inputs.iter().filter_map(|node| *node) {
                temp_value_refcount.inc(node_id);
            }
        }

        // Increment usage count of all output nodes, so we retain them after
        // the operator has run.
        for node_id in outputs {
            temp_value_refcount.inc(*node_id);
        }

        // Execute the plan
        let mut temp_values: HashMap<NodeId, Output> = HashMap::new();
        let mut op_elapsed: HashMap<&str, f32> = HashMap::new();
        let record_timing = opts.timing || opts.verbose;

        for (step, (op_node_id, op_node)) in plan.iter().enumerate() {
            let mut op_timer = Timer::new();
            if record_timing {
                op_timer.start();
            }

            let first_input = op_node.inputs.get(0).copied().flatten();

            // Test if the operator can be run in-place to save allocations.
            //
            // This requires that the first input is a temporary value produced
            // by earlier ops, and this value is not going to be needed by other
            // ops in future.
            //
            // If we can run in-place, extract the first input to re-use as the
            // output.
            let in_place_input = first_input.and_then(|first_input| {
                if op_node.operator.can_run_in_place()
                    && temp_values.contains_key(&first_input)
                    && temp_value_refcount.count(first_input) == 1
                {
                    temp_value_refcount.dec(first_input);
                    Some(temp_values.remove(&first_input).unwrap())
                } else {
                    None
                }
            });

            // If logging is enabled, save the shape at the start of execution
            // so we can report it later.
            let in_place_input_shape = match (opts.verbose, &in_place_input) {
                (true, Some(output)) => Some(output.shape().to_vec()),
                _ => None,
            };

            // Collect all or remaining inputs for the operator
            let mut op_inputs: Vec<Option<Input>> = Vec::new();
            let immutable_inputs = if in_place_input.is_some() {
                &op_node.inputs[1..]
            } else {
                &op_node.inputs[..]
            };
            for node_id in immutable_inputs.iter() {
                if let Some(node_id) = node_id {
                    if let Some(&value) = values.get(node_id) {
                        op_inputs.push(Some(value));
                    } else if let Some(value) = temp_values.get(node_id) {
                        let input = match value {
                            Output::IntTensor(t) => Input::IntTensor(t),
                            Output::FloatTensor(t) => Input::FloatTensor(t),
                        };
                        op_inputs.push(Some(input));
                    } else {
                        // If this is reached, there was a bug in plan creation.
                        panic!(
                            "Invalid plan did not produce input value {} for operator {}",
                            self.node_name(*node_id),
                            self.node_name(*op_node_id),
                        );
                    }
                } else {
                    op_inputs.push(None);
                }
            }

            let op_result = if let Some(input) = in_place_input {
                op_node
                    .operator
                    .run_in_place(input, InputList::from_optional(&op_inputs))
                    .map(|out| [out].into())
            } else {
                op_node
                    .operator
                    .run(InputList::from_optional(&op_inputs[..]))
            };

            if record_timing {
                op_timer.end();

                if let Some(elapsed) = op_elapsed.get_mut(op_node.operator.name()) {
                    *elapsed += op_timer.elapsed_ms();
                } else {
                    op_elapsed.insert(op_node.operator.name(), op_timer.elapsed_ms());
                }
            }

            // Log verbose info if enabled. This is done before we check the
            // result so that in the event of an error, the verbose log includes
            // the failing operator's inputs.
            if opts.verbose {
                let mut input_shapes: Vec<_> = op_inputs
                    .iter()
                    .map(|x| x.as_ref().map(|input| input.shape()))
                    .collect();
                if let Some(first_input_shape) = in_place_input_shape.as_ref() {
                    input_shapes.insert(0, Some(first_input_shape));
                }

                println!(
                    "#{} {} ({})",
                    step,
                    op_node.operator.name(),
                    op_node.name.as_ref().unwrap_or(&String::new())
                );
                for (index, (id, shape)) in
                    zip(op_node.inputs.iter(), input_shapes.iter()).enumerate()
                {
                    if let (Some(id), Some(shape)) = (id, shape) {
                        let name = self.node_name(*id);
                        println!("  input {}: {} ({:?})", index, name, shape);
                    }
                }

                if let Ok(outputs) = op_result.as_ref() {
                    for (index, (id, output)) in
                        zip(op_node.outputs.iter(), outputs.iter()).enumerate()
                    {
                        let name = id.map(|id| self.node_name(id)).unwrap_or(String::new());
                        println!("  output {}: {} ({:?})", index, name, output.shape());
                    }
                }

                println!("  time: {}ms", op_timer.elapsed_ms());
            }

            let outputs = match op_result {
                Ok(outputs) => outputs,
                Err(op_error) => {
                    let err = RunError::OperatorError {
                        name: op_node.name.as_deref().unwrap_or("").to_string(),
                        error: op_error,
                    };
                    return Err(err);
                }
            };

            if op_node.outputs.len() != outputs.len() {
                return Err(RunError::OutputMismatch(
                    "operator output count did not match expected count",
                ));
            }

            for (&output_id, output) in zip(op_node.outputs.iter(), outputs.into_iter()) {
                if let Some(output_id) = output_id {
                    temp_values.insert(output_id, output);
                }
            }

            // Remove temporary values that are no longer needed
            for node_id in op_node.inputs.iter().filter_map(|node| *node) {
                let rc = temp_value_refcount.dec(node_id);
                if rc == 0 {
                    temp_values.remove(&node_id);
                }
            }
        }

        if opts.timing {
            run_timer.end();
            println!(
                "Graph run of {} ops finished in {}ms",
                plan.len(),
                run_timer.elapsed_ms()
            );
            self.print_timings(&op_elapsed, run_timer.elapsed_ms());
        }

        // Return the requested outputs
        let result = outputs
            .iter()
            .map(|output_id| {
                if let Some(&value) = values.get(output_id) {
                    match value {
                        Input::IntTensor(t) => Output::IntTensor(t.clone()),
                        Input::FloatTensor(t) => Output::FloatTensor(t.clone()),
                    }
                } else {
                    // During execution planning we verified that each output
                    // ID is valid and unique, so this should always succeed.
                    temp_values.remove(output_id).expect("missing output value")
                }
            })
            .collect();
        Ok(result)
    }

    /// Print a table of operator timings from a graph run.
    fn print_timings(&self, op_elapsed: &HashMap<&str, f32>, run_time: f32) {
        // Display cumulative times for each op type, sorted by op name
        let total_op_time: f32 = op_elapsed.values().sum();
        let mut op_timings: Vec<_> = op_elapsed
            .iter()
            .map(|(name, time)| (*name, *time))
            .collect();
        op_timings.sort_by(|a, b| a.0.cmp(b.0));

        // Show time taken by non-operator processing, such as any memory
        // allocation / freeing that is done outside of ops.
        op_timings.push(("[Other]", run_time - total_op_time));

        let rows: Vec<_> = op_timings
            .iter()
            .map(|(op_name, op_total_time)| {
                let op_percent = (*op_total_time / total_op_time) * 100.;
                [
                    op_name.to_string(),
                    format!("{:.2}ms", op_total_time),
                    format!("({:.2}%)", op_percent),
                ]
            })
            .collect();
        let col_widths: Vec<usize> = (0..3)
            .map(|col| rows.iter().fold(0, |width, row| row[col].len().max(width)))
            .collect();

        for row in rows {
            println!(
                "{0:1$} {2:3$} {4:5$}",
                row[0], col_widths[0], row[1], col_widths[1], row[2], col_widths[2]
            );
        }
    }

    /// Create an execution plan for a sequence of computation steps that begin
    /// with `inputs` and eventually produces `outputs`.
    ///
    /// The set of input and output node IDs must be unique.
    ///
    /// Any node IDs in `outputs` which reference constant or input values are
    /// omitted from the plan.
    fn create_plan(
        &self,
        inputs: &[(NodeId, Input)],
        outputs: &[NodeId],
    ) -> Result<Vec<(NodeId, &OperatorNode)>, RunError> {
        if !all_unique(outputs, |x, y| x == y) {
            return Err(RunError::PlanningError("output IDs are not unique".into()));
        }

        if !all_unique(inputs, |(x_id, _), (y_id, _)| x_id == y_id) {
            return Err(RunError::PlanningError("input IDs are not unique".into()));
        }

        // Map of output node to source operator
        let mut operator_nodes = HashMap::new();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let Node::Operator(op_node) = node {
                for output_id in op_node.outputs.iter().filter_map(|node| *node) {
                    operator_nodes.insert(output_id, (node_id, op_node));
                }
            }
        }

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
            graph: &'a Graph,
            resolved_values: HashSet<NodeId>,
            plan: Vec<(NodeId, &'a OperatorNode)>,

            // Map of output ID to (op node ID, op)
            operator_nodes: HashMap<NodeId, (NodeId, &'a OperatorNode)>,
        }
        impl<'a> PlanBuilder<'a> {
            /// Add all the transitive dependencies of `op_node` to the plan,
            /// followed by `op_node`.
            fn visit(
                &mut self,
                op_node_id: NodeId,
                op_node: &'a OperatorNode,
            ) -> Result<(), RunError> {
                for input in op_node.inputs.iter().filter_map(|node| *node) {
                    if self.resolved_values.contains(&input) {
                        continue;
                    }
                    if let Some((input_op_id, input_op_node)) =
                        self.operator_nodes.get(&input).copied()
                    {
                        self.visit(input_op_id, input_op_node)?;
                    } else {
                        let msg = format!(
                            "Missing input \"{}\" for op \"{}\"",
                            self.graph.node_name(input),
                            self.graph.node_name(op_node_id)
                        );
                        return Err(RunError::PlanningError(msg));
                    }
                }
                for output_id in op_node.outputs.iter().filter_map(|node| *node) {
                    self.resolved_values.insert(output_id);
                }
                self.plan.push((op_node_id, op_node));
                Ok(())
            }

            /// Return a sequential plan to generate `outputs`. The plan is
            /// a vec of `(op_node_id, operator)` tuples.
            fn plan(
                mut self,
                outputs: &[NodeId],
            ) -> Result<Vec<(NodeId, &'a OperatorNode)>, RunError> {
                for output_id in outputs.iter() {
                    if self.resolved_values.contains(output_id) {
                        // Value is either a constant node or is produced by
                        // an operator that is already in the plan.
                        continue;
                    }

                    if let Some((op_node_id, op_node)) = self.operator_nodes.get(output_id).copied()
                    {
                        self.visit(op_node_id, op_node)?;
                    } else {
                        let msg = format!("Missing output {}", output_id);
                        return Err(RunError::PlanningError(msg));
                    }
                }
                Ok(self.plan)
            }
        }

        let builder = PlanBuilder {
            graph: self,
            resolved_values,
            plan: Vec::new(),
            operator_nodes,
        };
        builder.plan(outputs)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use wasnn_tensor::prelude::*;
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{tensor, Tensor};

    use crate::graph::{Dimension, Graph, RunError};
    use crate::ops::{
        Concat, Conv, InputList, IntoOpResult, OpError, Operator, Output, Relu, Shape,
    };

    // Test of a very simple graph with a typical structure (one input, one
    // output, Conv + Relu operation).
    #[test]
    fn test_graph_run() -> Result<(), String> {
        let mut g = Graph::new();

        let weights = Tensor::from_data(
            &[1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );
        let weights_id = g.add_constant(Some("weight"), weights);
        let input_id = g.add_value(Some("input"), None);

        let conv_out = g.add_value(Some("conv_out"), None);
        g.add_op(
            Some("conv"),
            Box::new(Conv {
                dilations: vec![1, 1],
                groups: 1,
                padding: [1, 1, 1, 1].into(),
                strides: vec![1, 1],
            }),
            &[input_id, weights_id].map(Some),
            &[conv_out].map(Some),
        );
        let relu_out = g.add_value(Some("relu_out"), None);
        g.add_op(
            Some("relu"),
            Box::new(Relu {}),
            &[conv_out].map(Some),
            &[relu_out].map(Some),
        );

        let input = Tensor::from_data(
            &[1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let results = g
            .run(&[(input_id, (&input).into())], &[relu_out], None)
            .unwrap();

        let expected = Tensor::from_data(
            &[1, 1, 3, 3],
            vec![
                1.5202, 1.5592, 0.9939, 1.7475, 2.6358, 1.3428, 1.0165, 1.1806, 0.8685,
            ],
        );
        assert_eq!(results.len(), 1);
        expect_equal(results[0].as_float_ref().unwrap(), &expected)
    }

    #[test]
    fn test_graph_node_debug_names() {
        let mut g = Graph::new();

        let weights = Tensor::from_data(&[1], vec![0.3230]);
        let weights_id = g.add_constant(Some("weights"), weights.clone());
        let input_id = g.add_value(Some("input"), None);
        let relu_out_id = g.add_value(Some("relu_out"), None);
        let relu_op_id = g.add_op(
            Some("relu"),
            Box::new(Relu {}),
            &[Some(input_id)],
            &[Some(relu_out_id)],
        );

        assert_eq!(g.node_name(weights_id), "weights");
        assert_eq!(g.node_name(input_id), "input");
        assert_eq!(g.node_name(relu_op_id), "relu");

        let anon_weights_id = g.add_constant(None, weights);
        let anon_input_id = g.add_value(None, None);
        let anon_out_id = g.add_value(None, None);
        let anon_op_id = g.add_op(
            None,
            Box::new(Relu {}),
            &[Some(input_id)],
            &[Some(anon_out_id)],
        );

        assert_eq!(
            g.node_name(anon_weights_id),
            format!("[ID: {}]", anon_weights_id)
        );
        assert_eq!(
            g.node_name(anon_input_id),
            format!("[ID: {}]", anon_input_id)
        );
        assert_eq!(g.node_name(anon_op_id), format!("[ID: {}]", anon_op_id));
    }

    #[test]
    fn test_graph_node_shapes() {
        let mut g = Graph::new();

        let weights = Tensor::from_data(&[1, 1, 2], vec![0.3230, 0.5]);
        let weights_id = g.add_constant(Some("weights"), weights.clone());
        let input_id = g.add_value(
            Some("input"),
            Some(
                [
                    Dimension::Symbolic("batch".to_string()),
                    Dimension::Fixed(3),
                    Dimension::Fixed(5),
                    Dimension::Fixed(5),
                ]
                .to_vec(),
            ),
        );
        let relu_out_id = g.add_value(Some("relu_out"), None);
        let relu_op_id = g.add_op(
            Some("relu"),
            Box::new(Relu {}),
            &[Some(input_id)],
            &[Some(relu_out_id)],
        );

        assert_eq!(
            g.get_node(weights_id).and_then(|n| n.shape()),
            Some([1, 1, 2].map(Dimension::Fixed).to_vec())
        );
        assert_eq!(
            g.get_node(input_id).and_then(|n| n.shape()),
            Some(
                [
                    Dimension::Symbolic("batch".to_string()),
                    Dimension::Fixed(3),
                    Dimension::Fixed(5),
                    Dimension::Fixed(5),
                ]
                .to_vec()
            )
        );
        assert_eq!(g.get_node(relu_op_id).and_then(|n| n.shape()), None);
    }

    #[derive(Debug)]
    struct AddOne {}
    impl Operator for AddOne {
        fn name(&self) -> &str {
            "AddOne"
        }

        fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
            let input: &Tensor<f32> = inputs.require_as(0)?;
            let output_data: Vec<f32> = input.iter().map(|x| x + 1.0).collect();
            Tensor::<f32>::from_data(input.shape().into(), output_data).into_op_result()
        }
    }

    #[test]
    fn test_graph_planning_order() -> Result<(), String> {
        let mut g = Graph::new();

        let input_id = g.add_value(Some("input"), None);

        let op_a_out = g.add_value(Some("op_a_out"), None);
        g.add_op(
            Some("op_a"),
            Box::new(AddOne {}),
            &[Some(input_id)],
            &[Some(op_a_out)],
        );
        let op_b_out = g.add_value(Some("op_b_out"), None);
        g.add_op(
            Some("op_b"),
            Box::new(AddOne {}),
            &[Some(op_a_out)],
            &[Some(op_b_out)],
        );

        // op_c has both op_a and op_b as inputs. Since op_b depends on op_a,
        // execution must run op_a, then op_b, then op_c.
        let op_c_out = g.add_value(Some("op_c_out"), None);
        g.add_op(
            Some("op_c"),
            Box::new(Concat { axis: 0 }),
            &[op_a_out, op_b_out].map(Some),
            &[Some(op_c_out)],
        );

        // op_d is the same as op_c, but input order is reversed
        let op_d_out = g.add_value(Some("op_d_out"), None);
        g.add_op(
            Some("op_d"),
            Box::new(Concat { axis: 0 }),
            &[op_b_out, op_a_out].map(Some),
            &[Some(op_d_out)],
        );

        let input = Tensor::from_data(&[1], vec![1.]);

        let results = g
            .run(&[(input_id, (&input).into())], &[op_c_out], None)
            .unwrap();
        let expected = Tensor::from_data(&[2], vec![2., 3.]);
        expect_equal(results[0].as_float_ref().unwrap(), &expected)?;

        let results = g
            .run(&[(input_id, (&input).into())], &[op_d_out], None)
            .unwrap();
        let expected = Tensor::from_data(&[2], vec![3., 2.]);
        expect_equal(results[0].as_float_ref().unwrap(), &expected)
    }

    // Perform a graph run where one of the outputs is also an input for other
    // steps of the run.
    #[test]
    fn test_graph_intermediate_output() {
        let mut g = Graph::new();

        let input_id = g.add_value(Some("input"), None);
        let op_a_out = g.add_value(Some("op_a_out"), None);
        g.add_op(
            Some("op_a"),
            Box::new(AddOne {}),
            &[Some(input_id)],
            &[Some(op_a_out)],
        );
        let op_b_out = g.add_value(Some("op_b_out"), None);
        g.add_op(
            Some("op_b"),
            Box::new(AddOne {}),
            &[Some(op_a_out)],
            &[Some(op_b_out)],
        );

        let input = tensor!(0.);
        let results = g
            .run(&[(input_id, (&input).into())], &[op_a_out, op_b_out], None)
            .unwrap();
        assert_eq!(results[0].as_float_ref().unwrap(), &tensor!(1.));
        assert_eq!(results[1].as_float_ref().unwrap(), &tensor!(2.));
    }

    #[test]
    fn test_graph_many_steps() -> Result<(), String> {
        let mut g = Graph::new();

        let input = Tensor::from_data(&[5], vec![1., 2., 3., 4., 5.]);
        let input_id = g.add_value(Some("input"), None);

        let mut prev_output = input_id;
        for _ in 0..100 {
            let next_output = g.add_value(None, None);
            g.add_op(
                None,
                Box::new(AddOne {}),
                &[Some(prev_output)],
                &[Some(next_output)],
            );
            prev_output = next_output;
        }

        let results = g
            .run(&[(input_id, (&input).into())], &[prev_output], None)
            .unwrap();

        let expected = Tensor::from_data(&[5], vec![101., 102., 103., 104., 105.]);
        expect_equal(results[0].as_float_ref().unwrap(), &expected)
    }

    #[test]
    fn test_noop_graph() -> Result<(), String> {
        let mut g = Graph::new();

        let input = Tensor::from_data(&[5], vec![1., 2., 3., 4., 5.]);
        let input_id = g.add_value(Some("input"), None);

        let results = g
            .run(&[(input_id, (&input).into())], &[input_id], None)
            .unwrap();

        expect_equal(results[0].as_float_ref().unwrap(), &input)
    }

    #[test]
    fn test_constant_graph() -> Result<(), String> {
        let mut g = Graph::new();

        let value = Tensor::from_data(&[5], vec![1., 2., 3., 4., 5.]);
        let const_id = g.add_constant(Some("weight"), value.clone());

        let results = g.run(&[], &[const_id], None).unwrap();

        expect_equal(results[0].as_float_ref().unwrap(), &value)
    }

    #[test]
    fn test_no_outputs() {
        let g = Graph::new();
        let results = g.run(&[], &[], None).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_duplicate_inputs() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None);
        let input = tensor!([1.]);
        let result = g.run(
            &[(input_id, (&input).into()), (input_id, (&input).into())],
            &[input_id],
            None,
        );
        assert_eq!(
            result,
            Err(RunError::PlanningError("input IDs are not unique".into()))
        );
    }

    #[test]
    fn test_duplicate_outputs() {
        let mut g = Graph::new();

        let input_id = g.add_value(Some("input"), None);
        let op_a_out = g.add_value(Some("op_a_out"), None);
        g.add_op(
            Some("op_a"),
            Box::new(AddOne {}),
            &[Some(input_id)],
            &[Some(op_a_out)],
        );

        let input = tensor!([1.]);

        let result = g.run(&[(input_id, (&input).into())], &[op_a_out, op_a_out], None);

        assert_eq!(
            result,
            Err(RunError::PlanningError("output IDs are not unique".into()))
        );
    }

    #[test]
    fn test_call_op_with_missing_input() {
        let mut g = Graph::new();
        let output = g.add_value(None, None);

        // Call an operator with an input omitted by setting it to `None`,
        // as opposed to passing a shorter input list. This enables omitting
        // an input but still providing subsequent ones.
        g.add_op(Some("shape"), Box::new(Shape {}), &[None], &[Some(output)]);

        let results = g.run(&[], &[output], None);

        assert_eq!(
            results.err(),
            Some(RunError::OperatorError {
                name: "shape".to_string(),
                error: OpError::MissingInputs
            })
        );
    }

    #[test]
    fn test_err_if_invalid_output() {
        let g = Graph::new();
        let result = g.run(&[], &[123], None);
        assert_eq!(
            result.err(),
            Some(RunError::PlanningError("Missing output 123".to_string()))
        );
    }

    #[test]
    fn test_err_if_missing_operator_input() {
        let mut g = Graph::new();
        let output = g.add_value(None, None);
        g.add_op(Some("op"), Box::new(Relu {}), &[Some(42)], &[Some(output)]);
        let result = g.run(&[], &[output], None);
        assert_eq!(
            result.err(),
            Some(RunError::PlanningError(
                "Missing input \"[ID: 42]\" for op \"op\"".to_string()
            ))
        );
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

        fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
            // An operator should normally have the same behavior in `run`
            // and `run_in_place`. Here we use different behavior to make it
            // possible to distinguish which path was used.
            let input: &Tensor<f32> = inputs.require_as(0)?;
            input.clone().into_op_result()
        }

        fn run_in_place(&self, input: Output, _other: InputList) -> Result<Output, OpError> {
            let mut output = input.into_float().unwrap();
            for x in output.data_mut().iter_mut() {
                *x = *x + 1.0;
            }
            Ok(output.into())
        }
    }

    #[test]
    fn test_runs_op_in_place() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None);

        let op1_out = g.add_value(Some("op1_out"), None);
        g.add_op(
            Some("op1"),
            Box::new(AddOneInPlace {}),
            &[Some(input_id)],
            &[Some(op1_out)],
        );
        let op2_out = g.add_value(Some("op2_out"), None);
        g.add_op(
            Some("op2"),
            Box::new(AddOneInPlace {}),
            &[Some(op1_out)],
            &[Some(op2_out)],
        );
        let op3_out = g.add_value(Some("op3_out"), None);
        g.add_op(
            Some("op3"),
            Box::new(AddOneInPlace {}),
            &[Some(op2_out)],
            &[Some(op3_out)],
        );
        let op4_out = g.add_value(Some("op4_out"), None);
        g.add_op(
            Some("op4"),
            Box::new(AddOneInPlace {}),
            &[Some(op2_out)],
            &[Some(op4_out)],
        );
        let input = Tensor::<f32>::zeros(&[1, 1]);

        // First operator should not be run in-place, since it has an
        // immutable input. The result should be the same as the input.
        let results = g
            .run(&[(input_id, (&input).into())], &[op1_out], None)
            .unwrap();
        assert_eq!(results[0].as_float_ref().unwrap()[[0, 0]], 0.0);

        // Second operator should be run in-place, as it meets all the
        // requirements for this optimization.
        let results = g
            .run(&[(input_id, (&input).into())], &[op2_out], None)
            .unwrap();
        assert_eq!(results[0].as_float_ref().unwrap()[[0, 0]], 1.0);

        // Third op should not be run in place, because its input is re-used
        // for fourth op. Fourth op can run in place as by then, it is the
        // only consumer of its input.
        let results = g
            .run(&[(input_id, (&input).into())], &[op3_out, op4_out], None)
            .unwrap();
        assert_eq!(results[0].as_float_ref().unwrap()[[0, 0]], 1.0);
        assert_eq!(results[1].as_float_ref().unwrap()[[0, 0]], 2.0);
    }

    /// Test operator that produces multiple outputs
    #[derive(Debug)]
    struct Split {
        run_count: Arc<Mutex<u32>>,
    }

    impl Split {
        fn new() -> Split {
            Split {
                run_count: Arc::new(Mutex::new(0)),
            }
        }
    }

    impl Operator for Split {
        fn name(&self) -> &str {
            "Split"
        }

        fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
            {
                let mut rc = self.run_count.lock().unwrap();
                *rc += 1;
            }

            let input: &Tensor<f32> = inputs.require_as(0)?;
            let left_split_len = input.len() / 2;
            let left_split = Tensor::from_vec(input.iter().take(left_split_len).copied().collect());
            let right_split =
                Tensor::from_vec(input.iter().skip(left_split_len).copied().collect());
            Ok([left_split.into(), right_split.into()].into())
        }
    }

    #[test]
    fn test_multiple_outputs() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None);
        let left_split_out = g.add_value(Some("left_split"), None);
        let right_split_out = g.add_value(Some("right_split"), None);

        let split_op = Box::new(Split::new());
        let run_count = split_op.run_count.clone();

        g.add_op(
            Some("split"),
            split_op,
            &[Some(input_id)],
            &[left_split_out, right_split_out].map(Some),
        );

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut results = g
            .run(
                &[(input_id, (&input).into())],
                &[left_split_out, right_split_out],
                None,
            )
            .unwrap();

        assert_eq!(*run_count.lock().unwrap(), 1);

        assert_eq!(results.len(), 2);
        let left_split = results.remove(0).into_float().unwrap();
        let right_split = results.remove(0).into_float().unwrap();
        assert_eq!(left_split.to_vec(), &[1.0, 2.0]);
        assert_eq!(right_split.to_vec(), &[3.0, 4.0, 5.0]);
    }
}
