use std::env;
use std::error::Error;
use std::fmt;
use std::iter::zip;

use rten_tensor::prelude::*;
use rten_tensor::Tensor;

// The std HashMap/HashSet provide DOS resistance. In this module hash keys are
// mostly `NodeId`s which we allocate ourselves, so this is not a concern.
// Instead we want faster hashing.
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ops::{Input, InputList, OpError, Operator, Output};
use crate::tensor_pool::TensorPool;
use crate::timer::Timer;
use crate::timing::{InputShape, RunTiming, TimingRecord, TimingSort};

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
    operator: Box<dyn Operator + Send + Sync>,
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

impl Constant {
    fn len(&self) -> usize {
        match self {
            Constant::Float(f) => f.data.len(),
            Constant::Int(i) => i.data.len(),
        }
    }
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

/// ID of a node in a [Model](crate::Model) graph.
pub type NodeId = usize;

/// Reasons why a graph execution failed
#[derive(Eq, PartialEq, Debug)]
pub enum RunError {
    /// An input or output node ID is invalid
    InvalidNodeId,

    /// No node with a given name could be found
    InvalidNodeName(String),

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
            RunError::InvalidNodeName(ref name) => write!(f, "no node found with name {}", name),
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

/// Options for creating a graph execution plan.
#[derive(Default)]
struct PlanOptions {
    /// Whether a plan can be successfully created if certain inputs are
    /// missing. If true, the planner will create the plan as if those inputs
    /// would be provided later.
    allow_missing_inputs: bool,
}

/// Counter that tracks the remaining usage count of a graph node value.
///
/// This is used to keep intermediate graph outputs alive until they are no
/// longer needed.
struct NodeRefCount {
    rc: FxHashMap<NodeId, usize>,
}

impl NodeRefCount {
    fn new() -> NodeRefCount {
        NodeRefCount {
            rc: FxHashMap::default(),
        }
    }

    /// Increment ref count of node
    fn inc(&mut self, id: NodeId) {
        self.rc
            .entry(id)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    /// Decrement ref count of node and return new count, removing the entry
    /// if it reaches zero.
    ///
    /// Returns `None` if there was no entry for this node.
    fn dec(&mut self, id: NodeId) -> Option<usize> {
        let rc = self.rc.get_mut(&id)?;
        *rc = rc.saturating_sub(1);
        if *rc == 0 {
            self.rc.remove(&id);
            Some(0)
        } else {
            Some(*rc)
        }
    }

    fn count(&self, id: NodeId) -> usize {
        *self.rc.get(&id).unwrap_or(&0)
    }
}

impl Error for RunError {}

/// Options that control logging and other behaviors when executing a
/// [Model](crate::Model).
#[derive(Default)]
pub struct RunOptions {
    /// Whether to log times spent in different operators when run completes.
    pub timing: bool,

    /// Order in which timings should be sorted. Defaults to sorting in
    /// descending order by time.
    pub timing_sort: TimingSort,

    /// Whether to include a breakdown of execution time by input shape, in
    /// timing reports.
    pub timing_by_shape: bool,

    /// Whether to log information about each graph operation as it is executed,
    /// including input shapes and execution time. This will slow down
    /// execution.
    pub verbose: bool,
}

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
        op: Box<dyn Operator + Send + Sync>,
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

    /// Return the total number of parameters in all constant nodes in the graph.
    pub fn total_params(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| match node {
                Node::Operator(_) => 0,
                Node::Value(_) => 0,
                Node::Constant(constant) => constant.len(),
            })
            .sum()
    }

    /// Compute a set of output values given a set of inputs, using the
    /// processing steps and constant values defined by the graph.
    pub fn run(
        &self,
        inputs: &[(NodeId, Input)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let plan = self.create_plan(
            inputs,
            outputs,
            PlanOptions {
                allow_missing_inputs: false,
            },
        )?;
        self.run_plan(inputs, &plan, outputs, opts)
    }

    fn run_plan(
        &self,
        inputs: &[(NodeId, Input)],
        plan: &[(NodeId, &OperatorNode)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let opts = opts.unwrap_or_default();

        let mut run_timer = Timer::new();
        if opts.timing {
            run_timer.start();
        }

        let inputs_by_id: FxHashMap<NodeId, Input> = inputs.iter().cloned().collect();
        let get_value_from_constant_or_input = |node_id: NodeId| -> Option<Input> {
            if let Some(Node::Constant(constant)) = self.nodes.get(node_id) {
                let value = match constant {
                    Constant::Float(node) => Input::FloatTensor(node.data.view()),
                    Constant::Int(node) => Input::IntTensor(node.data.view()),
                };
                Some(value)
            } else {
                inputs_by_id.get(&node_id).cloned()
            }
        };

        // Count how often each temporary output is used, so we can free them
        // when no longer needed.
        let mut temp_value_refcount = NodeRefCount::new();
        for (_, op_node) in plan.iter() {
            for node_id in op_node.inputs.iter().filter_map(|node| *node) {
                if let Some(Node::Value(_)) = self.nodes.get(node_id) {
                    temp_value_refcount.inc(node_id);
                }
            }
        }

        // Increment usage count of all output nodes, so we retain them after
        // the operator has run.
        for node_id in outputs {
            temp_value_refcount.inc(*node_id);
        }

        // Create a pool to re-use buffers across execution steps.
        //
        // If the feature flag is off, we still create the pool, but never
        // release buffers back into it, so all allocations use the system
        // allocator.
        let pool = TensorPool::new();
        let use_pool = env::var_os("RTEN_USE_POOL").is_some();

        // Execute the plan
        let mut temp_values: FxHashMap<NodeId, Output> = FxHashMap::default();
        let mut op_elapsed: Vec<TimingRecord> = Vec::new();
        let record_timing = opts.timing || opts.verbose;
        let mut alloc_timer = Timer::new();

        for (step, (op_node_id, op_node)) in plan.iter().enumerate() {
            let mut op_timer = Timer::new();
            if record_timing {
                op_timer.start();
            }

            // Choose the input that we'll try to modify in-place to avoid
            // allocating a new buffer for the output. This will be passed as
            // the first input to `Operator::run_in_place`.
            //
            // For non-commutative ops we have to use the first input. For
            // commutative ops we can swap inputs around if that enables us to
            // run an op in place.
            let in_place_input_id = if op_node.operator.can_run_in_place() {
                if op_node.operator.is_commutative() {
                    // Pick the largest input by number of elements. This
                    // assumes that commutative op outputs will have a shape
                    // that matches their largest input (eg. consider a
                    // binary op that broadcasts inputs to a common shape).
                    op_node
                        .inputs
                        .iter()
                        .max_by_key(|input_id| {
                            input_id
                                .and_then(|id| temp_values.get(&id))
                                .map(|val| val.len())
                                .unwrap_or(0)
                        })
                        .copied()
                        .flatten()
                } else {
                    op_node.inputs.first().copied().flatten()
                }
            } else {
                None
            };

            // If the operator can run in place, check if we have a tensor
            // that can be used as the output. This requires that the tensor
            // is not a constant (eg. weights) and is not going to be used by
            // other ops in future.
            let in_place_input = in_place_input_id.and_then(|first_input| {
                if temp_values.contains_key(&first_input)
                    && temp_value_refcount.count(first_input) == 1
                {
                    temp_value_refcount.dec(first_input);
                    Some(temp_values.remove(&first_input).unwrap())
                } else {
                    None
                }
            });

            // Collect all or remaining inputs for the operator
            let mut op_inputs: Vec<Option<Input>> = Vec::with_capacity(op_node.inputs.len());
            for node_id in op_node.inputs.iter() {
                if in_place_input.is_some() && *node_id == in_place_input_id {
                    continue;
                }

                if let Some(node_id) = node_id {
                    if let Some(value) = get_value_from_constant_or_input(*node_id) {
                        op_inputs.push(Some(value));
                    } else if let Some(value) = temp_values.get(node_id) {
                        let input = match value {
                            Output::IntTensor(t) => Input::IntTensor(t.view()),
                            Output::FloatTensor(t) => Input::FloatTensor(t.view()),
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

            // Collect input shapes if we'll need them for timing or logging.
            let input_shapes = if opts.timing_by_shape || opts.verbose {
                let mut shapes: Vec<InputShape> = Vec::new();
                if let Some(ref input) = in_place_input {
                    shapes.push(Some(input.shape().into()));
                }
                for input in &op_inputs {
                    shapes.push(input.as_ref().map(|i| i.shape().into()))
                }
                shapes
            } else {
                Vec::new()
            };

            let op_result = if let Some(input) = in_place_input {
                op_node
                    .operator
                    .run_in_place(&pool, input, InputList::from_optional(op_inputs))
                    .map(|out| [out].into())
            } else {
                op_node
                    .operator
                    .run(&pool, InputList::from_optional(op_inputs))
            };

            if record_timing {
                op_timer.end();

                op_elapsed.push(TimingRecord {
                    name: op_node.operator.name().to_string(),
                    input_shapes: input_shapes.clone(),
                    elapsed_micros: op_timer.elapsed_micros(),
                    node_name: op_node.name.clone().unwrap_or(String::new()),
                });
            }

            // Log verbose info if enabled. This is done before we check the
            // result so that in the event of an error, the verbose log includes
            // the failing operator's inputs.
            if opts.verbose {
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
            record_timing.then(|| alloc_timer.start());
            for node_id in op_node.inputs.iter().filter_map(|node| *node) {
                let rc = temp_value_refcount.dec(node_id);
                if rc == Some(0) {
                    if let (true, Some(tensor)) = (use_pool, temp_values.remove(&node_id)) {
                        match tensor {
                            Output::FloatTensor(t) => pool.add_tensor(t),
                            Output::IntTensor(t) => pool.add_tensor(t),
                        }
                    }
                }
            }
            record_timing.then(|| alloc_timer.end());
        }

        if opts.timing {
            run_timer.end();
            println!(
                "Graph run of {} ops finished in {}ms",
                plan.len(),
                run_timer.elapsed_ms()
            );
            println!(
                "Pool allocs {} hits {}",
                pool.alloc_count(),
                pool.hit_count()
            );
            let timing = RunTiming {
                records: &op_elapsed,
                alloc_time: alloc_timer.elapsed_ms(),
                total_time: run_timer.elapsed_ms(),
            };
            print!("{}", timing.display(opts.timing_sort, opts.timing_by_shape));
        }

        // Return the requested outputs
        let result = outputs
            .iter()
            .map(|output_id| {
                if let Some(value) = get_value_from_constant_or_input(*output_id) {
                    match value {
                        Input::IntTensor(t) => Output::IntTensor(t.to_tensor()),
                        Input::FloatTensor(t) => Output::FloatTensor(t.to_tensor()),
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

    /// Run part of the graph required to produce `outputs`, given an
    /// incomplete set of `inputs`.
    ///
    /// It is expected that `inputs` is missing some values which are required
    /// to produce `outputs`. This method will nevertheless produce an
    /// evaluation plan and evaluate as many intermediate nodes as possible,
    /// stopping when an operator is reached that transitively depends on a
    /// missing input. The result is the list of IDs and values of the leaf
    /// nodes of the subgraph that was evaluated. These intermediate values can
    /// later be passed to calls to `run` when the missing values are available.
    pub fn partial_run(
        &self,
        inputs: &[(NodeId, Input)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<(NodeId, Output)>, RunError> {
        let plan = self.create_plan(
            inputs,
            outputs,
            PlanOptions {
                allow_missing_inputs: true,
            },
        )?;
        let input_ids: Vec<_> = inputs.iter().map(|(id, _)| id).copied().collect();
        let (pruned_plan, pruned_plan_output_ids) = self.prune_plan(&plan, &input_ids, outputs);
        let outputs = self.run_plan(inputs, &pruned_plan, &pruned_plan_output_ids, opts)?;
        let output_ids_and_values: Vec<_> =
            pruned_plan_output_ids.into_iter().zip(outputs).collect();
        Ok(output_ids_and_values)
    }

    /// Prune a plan so that it contains only operators which can be executed
    /// given an initial set of inputs.
    ///
    /// Returns a tuple of `(pruned_plan, new_outputs)` where `new_outputs`
    /// contains the IDs of leaf nodes in the pruned plan. These are the values
    /// that can still be generated by the reduced plan, and are either in
    /// the original `outputs` list or are inputs to parts of the plan that
    /// were pruned away.
    fn prune_plan<'a>(
        &self,
        plan: &[(NodeId, &'a OperatorNode)],
        inputs: &[NodeId],
        outputs: &[NodeId],
    ) -> (Vec<(NodeId, &'a OperatorNode)>, Vec<NodeId>) {
        let mut resolved_values = self.init_resolved_values(inputs.iter().copied());
        let mut pruned_plan = Vec::new();

        // IDs of input nodes for pruned operators that we can still generate
        // with the pruned plan.
        let mut pruned_ops_resolved_inputs = FxHashSet::<NodeId>::default();

        // Walk forwards through the plan and prune away steps that cannot be
        // computed due to missing inputs.
        for (node_id, op_node) in plan {
            let all_inputs_available = op_node
                .inputs
                .iter()
                .filter_map(|id_opt| *id_opt)
                .all(|input_id| resolved_values.contains(&input_id));
            if !all_inputs_available {
                for input_id in op_node.inputs.iter().filter_map(|id_opt| *id_opt) {
                    if resolved_values.contains(&input_id) {
                        pruned_ops_resolved_inputs.insert(input_id);
                    }
                }
                continue;
            }
            resolved_values.extend(op_node.outputs.iter().filter_map(|id_opt| *id_opt));
            pruned_plan.push((*node_id, *op_node));
        }

        // Get IDs of values produced by the pruned plan which are either in the
        // originally requested set of outputs, or are inputs to steps of the
        // original plan that were pruned away.
        let new_outputs: Vec<NodeId> = pruned_plan
            .iter()
            .flat_map(|(_, op_node)| op_node.outputs.iter())
            .filter_map(|id_opt| *id_opt)
            .filter(|output| {
                outputs.contains(output) || pruned_ops_resolved_inputs.contains(output)
            })
            .collect();

        (pruned_plan, new_outputs)
    }

    /// Return the node IDs whose values are available at the start of graph
    /// execution, given a collection of initial inputs.
    fn init_resolved_values<I: Iterator<Item = NodeId>>(&self, inputs: I) -> FxHashSet<NodeId> {
        inputs
            .chain(
                self.nodes.iter().enumerate().filter_map(|(node_id, node)| {
                    matches!(node, Node::Constant(_)).then_some(node_id)
                }),
            )
            .collect()
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
        options: PlanOptions,
    ) -> Result<Vec<(NodeId, &OperatorNode)>, RunError> {
        if !all_unique(outputs, |x, y| x == y) {
            return Err(RunError::PlanningError("output IDs are not unique".into()));
        }

        if !all_unique(inputs, |(x_id, _), (y_id, _)| x_id == y_id) {
            return Err(RunError::PlanningError("input IDs are not unique".into()));
        }

        // Map of output node to source operator
        let mut operator_nodes = FxHashMap::default();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let Node::Operator(op_node) = node {
                for output_id in op_node.outputs.iter().filter_map(|node| *node) {
                    operator_nodes.insert(output_id, (node_id, op_node));
                }
            }
        }

        // Build an execution plan via a depth first traversal of the graph
        // starting at the output nodes. A helper struct is used as recursive
        // closures are not supported in Rust.
        struct PlanBuilder<'a> {
            graph: &'a Graph,
            resolved_values: FxHashSet<NodeId>,
            plan: Vec<(NodeId, &'a OperatorNode)>,

            // Map of output ID to (op node ID, op)
            operator_nodes: FxHashMap<NodeId, (NodeId, &'a OperatorNode)>,

            options: PlanOptions,
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
                    } else if self.options.allow_missing_inputs {
                        continue;
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

        // Set of values that are available after executing the plan
        let resolved_values: FxHashSet<NodeId> =
            self.init_resolved_values(inputs.iter().map(|(node_id, _)| *node_id));

        let builder = PlanBuilder {
            graph: self,
            resolved_values,
            plan: Vec::new(),
            operator_nodes,
            options,
        };
        builder.plan(outputs)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::{Arc, Mutex};

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::{expect_equal, expect_equal_with_tolerance};
    use rten_tensor::{tensor, Tensor, TensorView};

    use crate::graph::{Dimension, Graph, RunError};
    use crate::ops::{
        Add, Concat, Conv, InputList, IntoOpResult, OpError, Operator, Output, Relu, Shape,
    };
    use crate::tensor_pool::TensorPool;

    #[derive(Clone, Debug, Default)]
    struct Metrics {
        run_count: u32,
        run_in_place_count: u32,
    }

    /// Operator adapter that wraps an underlying operator in order to track
    /// uses of it.
    #[derive(Debug)]
    struct TrackUsage<Op: Operator> {
        inner: Op,
        metrics: Arc<Mutex<Metrics>>,
    }

    impl<Op: Operator> TrackUsage<Op> {
        /// Construct a new adapter that wraps `inner`.
        fn new(inner: Op) -> Self {
            TrackUsage {
                inner,
                metrics: Default::default(),
            }
        }

        /// Return a shared reference to the operator's usage counters.
        fn metrics(&self) -> Arc<Mutex<Metrics>> {
            self.metrics.clone()
        }
    }

    impl<Op: Operator> Operator for TrackUsage<Op> {
        fn name(&self) -> &str {
            self.inner.name()
        }

        fn can_run_in_place(&self) -> bool {
            self.inner.can_run_in_place()
        }

        fn is_commutative(&self) -> bool {
            self.inner.is_commutative()
        }

        fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
            {
                let mut m = self.metrics.lock().unwrap();
                m.run_count += 1;
            }
            self.inner.run(pool, inputs)
        }

        fn run_in_place(
            &self,
            pool: &TensorPool,
            output: Output,
            inputs: InputList,
        ) -> Result<Output, OpError> {
            {
                let mut m = self.metrics.lock().unwrap();
                m.run_in_place_count += 1;
            }
            self.inner.run_in_place(pool, output, inputs)
        }
    }

    // Test of a very simple graph with a typical structure (one input, one
    // output, Conv + Relu operation).
    #[test]
    fn test_graph_run() -> Result<(), Box<dyn Error>> {
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
        expect_equal_with_tolerance(results[0].as_float_ref().unwrap(), &expected, 1e-4, 0.)?;

        Ok(())
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

        fn run(&self, _pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
            let input: TensorView<f32> = inputs.require_as(0)?;
            let output_data: Vec<f32> = input.iter().map(|x| x + 1.0).collect();
            Tensor::<f32>::from_data(input.shape().into(), output_data).into_op_result()
        }
    }

    #[test]
    fn test_graph_planning_order() -> Result<(), Box<dyn Error>> {
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
        expect_equal(results[0].as_float_ref().unwrap(), &expected)?;

        Ok(())
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
    fn test_graph_many_steps() -> Result<(), Box<dyn Error>> {
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
        expect_equal(results[0].as_float_ref().unwrap(), &expected)?;

        Ok(())
    }

    #[test]
    fn test_noop_graph() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let input = Tensor::from_data(&[5], vec![1., 2., 3., 4., 5.]);
        let input_id = g.add_value(Some("input"), None);

        let results = g
            .run(&[(input_id, (&input).into())], &[input_id], None)
            .unwrap();

        expect_equal(results[0].as_float_ref().unwrap(), &input)?;

        Ok(())
    }

    #[test]
    fn test_constant_graph() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let value = Tensor::from_data(&[5], vec![1., 2., 3., 4., 5.]);
        let const_id = g.add_constant(Some("weight"), value.clone());

        let results = g.run(&[], &[const_id], None).unwrap();

        expect_equal(results[0].as_float_ref().unwrap(), &value)?;

        Ok(())
    }

    #[test]
    fn test_total_params() {
        let mut g = Graph::new();
        g.add_constant(Some("floats"), Tensor::<f32>::zeros(&[10, 10]));
        g.add_constant(Some("ints"), Tensor::<i32>::zeros(&[10, 10]));
        assert_eq!(g.total_params(), 200);
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

        fn run(&self, _pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
            // An operator should normally have the same behavior in `run`
            // and `run_in_place`. Here we use different behavior to make it
            // possible to distinguish which path was used.
            let input: TensorView<f32> = inputs.require_as(0)?;
            input.to_tensor().into_op_result()
        }

        fn run_in_place(
            &self,
            _pool: &TensorPool,
            input: Output,
            _other: InputList,
        ) -> Result<Output, OpError> {
            let mut output = input.into_float().unwrap();
            for x in output.iter_mut() {
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

    // Test that the graph executor will swap inputs to commutative ops if
    // necessary to enable running in-place.
    #[test]
    fn test_runs_commutative_op_in_place() {
        use crate::ops::Add; // A commutative operator

        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None);
        let bias_id = g.add_value(Some("bias"), None);

        let op1 = TrackUsage::new(Add {});
        let op1_metrics = op1.metrics();

        let op2 = TrackUsage::new(Add {});
        let op2_metrics = op2.metrics();

        let op1_out = g.add_value(Some("op1_out"), None);
        g.add_op(
            Some("op1"),
            Box::new(op1),
            &[Some(input_id), Some(bias_id)],
            &[Some(op1_out)],
        );
        let op2_out = g.add_value(Some("op2_out"), None);
        g.add_op(
            Some("op2"),
            Box::new(op2),
            // Note here the input ordering. The bias value is smaller, but
            // is the first argument. This operator can run in place, but only
            // if the inputs are swapped.
            &[Some(bias_id), Some(op1_out)],
            &[Some(op2_out)],
        );
        let input = Tensor::<f32>::zeros(&[2, 2]);
        let bias = tensor!(1.5);

        let results = g
            .run(
                &[(input_id, (&input).into()), (bias_id, (&bias).into())],
                &[op2_out],
                None,
            )
            .unwrap();

        // Bias value should be added twice to every input.
        assert_eq!(
            results[0]
                .as_float_ref()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            &[3., 3., 3., 3.]
        );

        // The first operator in a graph run must always copy its input.
        let op1_metrics = op1_metrics.lock().unwrap();
        assert_eq!(op1_metrics.run_count, 1);
        assert_eq!(op1_metrics.run_in_place_count, 0);

        // The second operator should run in-place.
        let op2_metrics = op2_metrics.lock().unwrap();
        assert_eq!(op2_metrics.run_count, 0);
        assert_eq!(op2_metrics.run_in_place_count, 1);
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

        fn run(&self, _pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
            {
                let mut rc = self.run_count.lock().unwrap();
                *rc += 1;
            }

            let input: TensorView<f32> = inputs.require_as(0)?;
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

    #[test]
    fn test_partial_run() -> Result<(), Box<dyn Error>> {
        // Set up graph like:
        //
        // C0, V0 --> Op0 --> Op2 --> [Output]
        // C1, V1 --> Op1 --^
        //
        // Where `Cn` are constants, `Vn` are input values and `OpN` are
        // operators.
        let mut g = Graph::new();
        let const_0 = g.add_constant(Some("c0"), tensor!(3.));
        let val_0 = g.add_value(Some("i0"), None);
        let const_1 = g.add_constant(Some("c1"), tensor!(4.));
        let val_1 = g.add_value(Some("i1"), None);

        let add_op_0 = Box::new(Add {});
        let op_0_out = g.add_value(Some("out0"), None);
        g.add_op(
            Some("Add_0"),
            add_op_0,
            &[Some(const_0), Some(val_0)],
            &[op_0_out].map(Some),
        );

        let add_op_1 = Box::new(Add {});
        let op_1_out = g.add_value(Some("out1"), None);
        g.add_op(
            Some("Add_1"),
            add_op_1,
            &[Some(const_1), Some(val_1)],
            &[op_1_out].map(Some),
        );

        let add_op_2 = Box::new(Add {});
        let op_2_out = g.add_value(Some("out2"), None);
        g.add_op(
            Some("Add_2"),
            add_op_2,
            &[Some(op_0_out), Some(op_1_out)],
            &[op_2_out].map(Some),
        );

        // Run graph with no inputs. This is equivalent to constant evaluation.
        // In this case no operators can be evaluated with graph constants
        // alone, so the output is empty.
        let partial_outs = g.partial_run(&[], &[op_2_out], None)?;
        assert_eq!(partial_outs.len(), 0);

        // Run graph with just the `V0` input. This will compute the result of
        // `Op0` but not other nodes which depend on `V1`.
        let input = tensor!(2.);
        let partial_outs = g.partial_run(&[(val_0, input.view().into())], &[op_2_out], None)?;
        assert_eq!(partial_outs.len(), 1);
        assert_eq!(partial_outs[0].0, op_0_out);
        assert_eq!(partial_outs[0].1, Output::FloatTensor(tensor!(5.)));

        // Run graph with just the `V1` input. This will compute the result of
        // `Op1` but not other nodes which depend on `V0`.
        let input = tensor!(2.);
        let partial_outs = g.partial_run(&[(val_1, input.view().into())], &[op_2_out], None)?;
        assert_eq!(partial_outs.len(), 1);
        assert_eq!(partial_outs[0].0, op_1_out);
        assert_eq!(partial_outs[0].1, Output::FloatTensor(tensor!(6.)));

        // Run graph with all inputs. This should behave like `Graph::run`.
        let partial_outs = g.partial_run(
            &[(val_1, input.view().into()), (val_0, input.view().into())],
            &[op_2_out],
            None,
        )?;
        assert_eq!(partial_outs.len(), 1);
        assert_eq!(partial_outs[0].0, op_2_out);
        assert_eq!(partial_outs[0].1, Output::FloatTensor(tensor!(11.)));

        Ok(())
    }
}
