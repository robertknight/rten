use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::iter::zip;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rten_tensor::prelude::*;
use rten_tensor::{DynLayout, Tensor, TensorView};

// The std HashMap/HashSet provide DOS resistance. In this module hash keys are
// mostly `NodeId`s which we allocate ourselves, so this is not a concern.
// Instead we want faster hashing.
use rustc_hash::{FxHashMap, FxHashSet};

use smallvec::SmallVec;

use crate::constant_storage::ArcTensorView;
use crate::env::env_flag;
use crate::ops::{Input, InputList, InputOrOutput, OpError, Operator, Output, OutputList};
use crate::tensor_pool::TensorPool;
use crate::threading;
use crate::timing::{InputShape, Instant, RunTiming, TimingRecord, TimingSort};

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
    operator: Arc<dyn Operator + Send + Sync>,
}

impl OperatorNode {
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn input_ids(&self) -> &[Option<NodeId>] {
        &self.inputs
    }

    pub fn output_ids(&self) -> &[Option<NodeId>] {
        &self.outputs
    }

    pub fn output_id(&self) -> Option<NodeId> {
        match &self.outputs[..] {
            [Some(id)] => Some(*id),
            _ => None,
        }
    }

    pub fn operator(&self) -> &dyn Operator {
        self.operator.as_ref()
    }

    /// Return a new `Arc` reference to this node's operator.
    ///
    /// Since operators are stateless and immutable once added to a graph, they
    /// can be "cloned" just be creating a new reference.
    pub fn clone_operator(&self) -> Arc<dyn Operator + Send + Sync> {
        self.operator.clone()
    }

    pub fn replace_input(&mut self, old_id: NodeId, new_id: NodeId) {
        for input_id in self.inputs.iter_mut() {
            if *input_id == Some(old_id) {
                *input_id = Some(new_id);
            }
        }
    }
}

pub struct ValueNode {
    name: Option<String>,
    shape: Option<Vec<Dimension>>,
}

impl ValueNode {
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Data for a constant node (ie. model weights) in a [Graph].
pub enum ConstantNodeData<T> {
    Owned(Tensor<T>),
    Arc(ArcTensorView<T>),
}

impl<T> From<Tensor<T>> for ConstantNodeData<T> {
    fn from(val: Tensor<T>) -> ConstantNodeData<T> {
        ConstantNodeData::Owned(val)
    }
}

impl<T> From<ArcTensorView<T>> for ConstantNodeData<T> {
    fn from(val: ArcTensorView<T>) -> ConstantNodeData<T> {
        ConstantNodeData::Arc(val)
    }
}

pub struct ConstantNode<T> {
    name: Option<String>,
    data: ConstantNodeData<T>,
}

impl<T> ConstantNode<T> {
    pub fn view(&self) -> TensorView<T> {
        match &self.data {
            ConstantNodeData::Owned(data) => data.view(),
            ConstantNodeData::Arc(data) => data.view(),
        }
    }

    fn layout(&self) -> &DynLayout {
        match &self.data {
            ConstantNodeData::Owned(data) => data.layout(),
            ConstantNodeData::Arc(data) => data.layout(),
        }
    }
}

pub enum Constant {
    Float(ConstantNode<f32>),
    Int(ConstantNode<i32>),
}

impl Constant {
    pub fn name(&self) -> Option<&str> {
        match self {
            Constant::Float(f) => f.name.as_deref(),
            Constant::Int(i) => i.name.as_deref(),
        }
    }

    fn layout(&self) -> &DynLayout {
        match self {
            Constant::Float(f) => f.layout(),
            Constant::Int(i) => i.layout(),
        }
    }

    /// Return the data for this constant as a tensor view.
    pub fn as_input(&self) -> Input {
        match self {
            Constant::Float(f) => Input::FloatTensor(f.view()),
            Constant::Int(i) => Input::IntTensor(i.view()),
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

/// Extract typed data from a [`Constant`].
pub trait TypedConstant<T> {
    fn as_view(&self) -> Option<TensorView<T>>;
    fn as_scalar(&self) -> Option<T>;
    fn as_vector(&self) -> Option<&[T]>;
}

macro_rules! impl_typed_constant {
    ($type:ty, $variant:ident) => {
        impl TypedConstant<$type> for Constant {
            fn as_view(&self) -> Option<TensorView<$type>> {
                match self {
                    Constant::$variant(tensor) => Some(tensor.view()),
                    _ => None,
                }
            }

            fn as_scalar(&self) -> Option<$type> {
                self.as_view().and_then(|view| view.item().copied())
            }

            fn as_vector(&self) -> Option<&[$type]> {
                self.as_view()
                    .and_then(|view| match (view.ndim(), view.data()) {
                        (1, Some(vec_data)) => Some(vec_data),
                        _ => None,
                    })
            }
        }
    };
}

impl_typed_constant!(f32, Float);
impl_typed_constant!(i32, Int);

pub enum Node {
    Operator(OperatorNode),
    Constant(Constant),
    Value(ValueNode),
}

impl Node {
    /// Return the debug name of this node
    pub fn name(&self) -> Option<&str> {
        match self {
            Node::Operator(node) => node.name(),
            Node::Constant(constant) => constant.name(),
            Node::Value(node) => node.name(),
        }
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
            Node::Constant(node) => Some(dims_from_fixed_shape(node.layout().shape())),
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
    OperatorError {
        /// Name of the operator node.
        name: String,
        error: OpError,
    },

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
    rc: Vec<u8>,
}

impl NodeRefCount {
    /// Create a new ref count array with a maximum node ID of `n_nodes - 1`.
    fn with_capacity(n_nodes: usize) -> NodeRefCount {
        NodeRefCount {
            rc: vec![0; n_nodes],
        }
    }

    /// Increment ref count of node. If the refcount reaches `u8::MAX` it
    /// will become "sticky" and never decrement.
    fn inc(&mut self, id: NodeId) {
        let rc = &mut self.rc[id];
        *rc = rc.saturating_add(1);
    }

    /// Decrement ref count of node and return new count, or `None` if the
    /// ref count was already zero.
    fn dec(&mut self, id: NodeId) -> Option<usize> {
        let rc = &mut self.rc[id];

        // If the refcount reaches the max value, it becomes sticky.
        if *rc == u8::MAX {
            return Some(*rc as usize);
        } else if *rc == 0 {
            return None;
        }

        *rc = rc.saturating_sub(1);
        Some(*rc as usize)
    }

    fn count(&self, id: NodeId) -> usize {
        self.rc[id] as usize
    }
}

impl Error for RunError {}

/// An execution plan specifying the operations to perform to derive a set of
/// output nodes given a set of input nodes.
struct CachedPlan {
    /// Sorted list of value nodes that are provided at the start of execution.
    inputs: Vec<NodeId>,

    /// Sorted list of value nodes produced after the plan has executed.
    outputs: Vec<NodeId>,

    /// List of operator nodes to execute to produce `outputs` given `inputs`.
    plan: Vec<NodeId>,
}

impl CachedPlan {
    fn new(inputs: &[NodeId], outputs: &[NodeId], plan: Vec<NodeId>) -> CachedPlan {
        let mut inputs = inputs.to_vec();
        let mut outputs = outputs.to_vec();

        inputs.sort();
        outputs.sort();

        CachedPlan {
            inputs,
            outputs,
            plan,
        }
    }

    /// Return true if a set of input and output nodes matches those used to
    /// create the plan.
    fn matches(&self, inputs: &[NodeId], outputs: &[NodeId]) -> bool {
        let input_match = inputs.len() == self.inputs.len()
            && inputs
                .iter()
                .all(|node_id| self.inputs.binary_search(node_id).is_ok());
        let output_match = outputs.len() == self.outputs.len()
            && outputs
                .iter()
                .all(|node_id| self.outputs.binary_search(node_id).is_ok());
        input_match && output_match
    }

    /// Return the set of operator node IDs for this plan.
    fn plan(&self) -> &[NodeId] {
        &self.plan
    }
}

/// Options that control logging and other behaviors when executing a
/// [Model](crate::Model).
#[derive(Clone, Default, PartialEq)]
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
/// weights produced during training, a dynamic value passed or computed at
/// runtime, or an operator.
///
/// A subset of the nodes are designated as the default inputs and outputs.
/// These constitute the "public API" of the graph and will be preserved after
/// any optimizations applied to the graph structure at runtime. Other
/// "internal" nodes may be replaced or removed.
pub struct Graph {
    nodes: Vec<Node>,

    /// The plan that was used for the most recent execution of the graph.
    cached_plan: Mutex<Option<Arc<CachedPlan>>>,

    /// Map of value node ID => source operator ID. This enables traversing the
    /// graph from outputs to inputs.
    source_ids: FxHashMap<NodeId, NodeId>,

    /// Default inputs for a graph run.
    input_ids: Vec<NodeId>,

    /// Default outputs for a graph run.
    output_ids: Vec<NodeId>,

    node_id_from_name: HashMap<String, NodeId>,
}

impl Graph {
    /// Create a new empty graph.
    pub fn new() -> Graph {
        Self::with_capacity(0)
    }

    /// Create a new graph with pre-allocated storage space for nodes.
    pub fn with_capacity(n_nodes: usize) -> Graph {
        Graph {
            nodes: Vec::with_capacity(n_nodes),
            cached_plan: Mutex::new(None),
            source_ids: FxHashMap::default(),
            input_ids: Vec::with_capacity(n_nodes),
            output_ids: Vec::with_capacity(n_nodes),
            node_id_from_name: HashMap::with_capacity(n_nodes),
        }
    }

    /// Set which nodes are the default inputs for this graph.
    pub fn set_input_ids(&mut self, node_ids: &[NodeId]) {
        self.input_ids = node_ids.to_vec();
    }

    /// Return the nodes which are the default inputs for this graph.
    pub fn input_ids(&self) -> &[NodeId] {
        &self.input_ids
    }

    /// Set which nodes are the default outputs for this graph.
    pub fn set_output_ids(&mut self, node_ids: &[NodeId]) {
        self.output_ids = node_ids.to_vec();
    }

    /// Return the nodes which are the default outputs for this graph.
    pub fn output_ids(&self) -> &[NodeId] {
        &self.output_ids
    }

    fn add_node(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        let node_id = self.nodes.len() - 1;

        if let Some(name) = self.nodes[node_id].name() {
            self.node_id_from_name.insert(name.to_string(), node_id);
        }

        node_id
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
        let op_id = self.add_node(Node::Operator(OperatorNode {
            name: name.map(|s| s.to_owned()),
            inputs: Vec::from(inputs),
            outputs: Vec::from(outputs),
            operator: Arc::from(op),
        }));

        for output_id in outputs.iter().flatten() {
            self.source_ids.insert(*output_id, op_id);
        }

        op_id
    }

    /// Add an operator and output value node to the graph.
    ///
    /// This is a simplified variant of [`add_op`](Self::add_op) for the
    /// common case of an operator with a single output and no missing inputs.
    ///
    /// Returns an `(operator_node_id, output_node_id)` tuple.
    #[cfg(test)]
    pub fn add_simple_op<Op: Operator + Send + Sync>(
        &mut self,
        name: &str,
        op: Op,
        input_ids: &[NodeId],
    ) -> (NodeId, NodeId) {
        let op_out_name = format!("{}_out", name);
        let op_out_id = self.add_value(Some(&op_out_name), None);
        let input_ids: Vec<_> = input_ids.iter().copied().map(Some).collect();
        let op_node_id = self.add_op(Some(name), Box::new(op), &input_ids, &[op_out_id].map(Some));
        (op_node_id, op_out_id)
    }

    /// Add a constant node to the graph.
    ///
    /// `name` is an identifier for this node that is used in debug messages etc.
    ///
    /// Returns the ID of the added node.
    pub fn add_constant<T, V>(&mut self, name: Option<&str>, value: V) -> NodeId
    where
        V: Into<ConstantNodeData<T>>,
        ConstantNode<T>: Into<Constant>,
    {
        let node = ConstantNode {
            name: name.map(|s| s.to_owned()),
            data: value.into(),
        };

        self.add_node(Node::Constant(node.into()))
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
        self.add_node(Node::Value(ValueNode {
            name: name.map(|s| s.to_owned()),
            shape,
        }))
    }

    /// Return an iterator over nodes in the graph.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().enumerate()
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

    /// Look up a node ID given its unique name
    pub fn get_node_id(&self, name: &str) -> Option<NodeId> {
        self.node_id_from_name.get(name).copied()
    }

    /// Look up the operator node which produced a given value node.
    pub fn get_source_node(&self, id: NodeId) -> Option<(NodeId, &OperatorNode)> {
        self.source_ids
            .get(&id)
            .and_then(|&id| match self.get_node(id) {
                Some(Node::Operator(op_node)) => Some((id, op_node)),
                _ => None,
            })
    }

    /// Retrieve a node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    /// Return the total number of parameters in all constant nodes in the graph.
    pub fn total_params(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| match node {
                Node::Operator(_) => 0,
                Node::Value(_) => 0,
                Node::Constant(constant) => constant.layout().len(),
            })
            .sum()
    }

    /// Compute a set of output values given a set of inputs, using the
    /// processing steps and constant values defined by the graph.
    pub fn run(
        &self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let plan = {
            // Reuse the plan from the previous run if the input and output IDs
            // match, otherwise create a new one.
            //
            // Note that we only hold the plan lock while creating the plan,
            // not while executing the model.
            let mut cached_plan = self.cached_plan.lock().unwrap();
            let input_ids: Vec<_> = inputs.iter().map(|(node_id, _)| *node_id).collect();
            match cached_plan.as_ref() {
                Some(plan) if plan.matches(&input_ids, outputs) => plan.clone(),
                _ => {
                    let plan = self.create_plan(
                        &inputs,
                        outputs,
                        PlanOptions {
                            allow_missing_inputs: false,
                        },
                    )?;
                    *cached_plan = Some(Arc::new(CachedPlan::new(&input_ids, outputs, plan)));
                    cached_plan.clone().unwrap()
                }
            }
        };

        threading::thread_pool().run(|| self.run_plan(inputs, plan.plan(), outputs, opts))
    }

    fn run_plan(
        &self,
        mut inputs: Vec<(NodeId, InputOrOutput)>,
        plan: &[NodeId],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let opts = opts.unwrap_or_default();

        let mut temp_values: FxHashMap<NodeId, Output> = FxHashMap::default();

        // Extract all the owned tensor inputs into the temp value map.
        //
        // This enables these inputs to be used for in-place operations or
        // returned directly as outputs.
        let mut idx = 0;
        while idx < inputs.len() {
            if matches!(inputs[idx], (_, InputOrOutput::Output(_))) {
                let (node_id, InputOrOutput::Output(outp)) = inputs.remove(idx) else {
                    unreachable!();
                };
                temp_values.insert(node_id, outp);
            } else {
                idx += 1;
            }
        }

        let inputs_by_id: FxHashMap<NodeId, InputOrOutput> = inputs.iter().cloned().collect();
        let get_value_from_constant_or_input = |node_id: NodeId| -> Option<Input> {
            match self.nodes.get(node_id) {
                Some(Node::Constant(constant)) => Some(constant.as_input()),
                Some(Node::Value(_)) => inputs_by_id.get(&node_id).map(|input| input.as_input()),
                Some(Node::Operator(_)) | None => {
                    panic!("node is not a value or constant");
                }
            }
        };

        // Count how often each temporary output is used, so we can free them
        // when no longer needed.
        let mut temp_value_refcount = NodeRefCount::with_capacity(self.nodes.len());
        for &op_node_id in plan.iter() {
            let Some(Node::Operator(op_node)) = self.nodes.get(op_node_id) else {
                return Err(RunError::PlanningError(
                    "operator node not found".to_string(),
                ));
            };
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
        let use_pool = env_flag("RTEN_USE_POOL", true);

        // Execute the plan
        let record_timing = opts.timing || opts.verbose;
        let mut op_timing_records: Vec<TimingRecord> = if record_timing {
            Vec::with_capacity(plan.len())
        } else {
            Vec::new()
        };

        let mut op_start = Instant::now();

        for (step, &op_node_id) in plan.iter().enumerate() {
            let Some(Node::Operator(op_node)) = self.nodes.get(op_node_id) else {
                return Err(RunError::PlanningError(
                    "operator node not found".to_string(),
                ));
            };

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
            let in_place_input = in_place_input_id.and_then(|input| {
                if temp_value_refcount.count(input) == 1 {
                    temp_values.remove(&input)
                } else {
                    None
                }
            });

            // Collect all or remaining inputs for the operator
            let mut op_inputs: SmallVec<[Option<Input>; 4]> =
                SmallVec::with_capacity(op_node.inputs.len());
            for node_id in op_node.inputs.iter() {
                if in_place_input.is_some() && *node_id == in_place_input_id {
                    continue;
                }

                if let Some(node_id) = node_id {
                    if let Some(value) = get_value_from_constant_or_input(*node_id) {
                        op_inputs.push(Some(value));
                    } else if let Some(value) = temp_values.get(node_id) {
                        op_inputs.push(Some(value.as_input()));
                    } else {
                        // If this is reached, there was a bug in plan creation.
                        panic!(
                            "Invalid plan did not produce input value {} for operator {}",
                            self.node_name(*node_id),
                            self.node_name(op_node_id),
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

            // Run the operation.
            let op_result = if let Some(input) = in_place_input {
                op_node
                    .operator
                    .run_in_place(&pool, input, InputList::from_optional(&op_inputs))
                    .map(|out| [out].into())
            } else {
                op_node
                    .operator
                    .run(&pool, InputList::from_optional(&op_inputs))
            };
            std::mem::drop(op_inputs);

            // Print verbose logs if enabled. This is done before checking the
            // op's result, so logs will contain details of the failed operation
            // in the event of an error.
            if opts.verbose {
                let op_duration = Instant::now() - op_start;
                self.print_op_timing(step, op_node, &op_result, op_duration, &input_shapes);
            }

            // Extract outputs or fail if an error occurred.
            let outputs = op_result.map_err(|op_error| RunError::OperatorError {
                name: op_node.name.as_deref().unwrap_or("").to_string(),
                error: op_error,
            })?;
            if op_node.outputs.len() != outputs.len() {
                return Err(RunError::OutputMismatch(
                    "operator output count did not match expected count",
                ));
            }

            // Save outputs for future steps.
            temp_values.extend(
                op_node
                    .outputs
                    .iter()
                    .zip(outputs.into_iter())
                    .filter_map(|(output_id, output)| output_id.map(|id| (id, output))),
            );

            // Remove temporary values that are no longer needed
            for node_id in op_node.inputs.iter().filter_map(|node| *node) {
                let rc = temp_value_refcount.dec(node_id);
                if rc == Some(0) {
                    if let (true, Some(tensor)) = (use_pool, temp_values.remove(&node_id)) {
                        tensor.add_to_pool(&pool)
                    }
                }
            }

            if record_timing {
                let op_end = Instant::now();
                let op_duration = op_end - op_start;
                op_start = op_end;

                op_timing_records.push(TimingRecord {
                    name: op_node.operator.name(),
                    input_shapes,
                    elapsed: op_duration,
                    node_name: op_node.name.as_deref().unwrap_or(""),
                });
            }
        }

        if opts.timing {
            self.print_run_timing(plan, &pool, &op_timing_records, &opts);
        }

        // Return the requested outputs
        let result = outputs
            .iter()
            .map(|output_id| {
                if let Some(value) = get_value_from_constant_or_input(*output_id) {
                    value.to_output()
                } else {
                    // During execution planning we verified that each output
                    // ID is valid and unique, so this should always succeed.
                    temp_values.remove(output_id).expect("missing output value")
                }
            })
            .collect();
        Ok(result)
    }

    /// Print detailed information about an operation just after it has run.
    fn print_op_timing(
        &self,
        step: usize,
        op_node: &OperatorNode,
        op_result: &Result<OutputList, OpError>,
        op_duration: Duration,
        input_shapes: &[InputShape],
    ) {
        println!(
            "#{} {} ({})",
            step,
            op_node.operator.name(),
            op_node.name.as_ref().unwrap_or(&String::new())
        );
        for (index, (id, shape)) in zip(op_node.inputs.iter(), input_shapes.iter()).enumerate() {
            if let (Some(id), Some(shape)) = (id, shape) {
                let name = self.node_name(*id);
                println!("  input {}: {} ({:?})", index, name, shape);
            }
        }

        if let Ok(outputs) = op_result.as_ref() {
            for (index, (id, output)) in zip(op_node.outputs.iter(), outputs.iter()).enumerate() {
                let name = id.map(|id| self.node_name(id)).unwrap_or(String::new());
                println!("  output {}: {} ({:?})", index, name, output.shape());
            }
        }

        println!("  time: {}ms", op_duration.as_secs_f64() * 1000.0);
    }

    /// Print a profiling summary at the end of the run.
    fn print_run_timing(
        &self,
        plan: &[NodeId],
        pool: &TensorPool,
        op_timing_records: &[TimingRecord],
        opts: &RunOptions,
    ) {
        let run_duration: Duration = op_timing_records.iter().map(|r| r.elapsed).sum();
        let run_duration_ms = run_duration.as_secs_f64() * 1000.0;
        println!(
            "Graph run of {} ops finished in {:.3}ms",
            plan.len(),
            run_duration_ms,
        );
        println!(
            "Pool allocs {} hits {}",
            pool.alloc_count(),
            pool.hit_count()
        );
        let timing = RunTiming {
            records: op_timing_records,
            total_time: run_duration,
        };
        print!(
            "{}",
            timing.display(opts.timing_sort.clone(), opts.timing_by_shape)
        );
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
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<(NodeId, Output)>, RunError> {
        let plan = self.create_plan(
            &inputs,
            outputs,
            PlanOptions {
                allow_missing_inputs: true,
            },
        )?;
        let input_ids: Vec<_> = inputs.iter().map(|(id, _)| id).copied().collect();
        let (pruned_plan, pruned_plan_output_ids) = self.prune_plan(&plan, &input_ids, outputs);
        let outputs = threading::thread_pool()
            .run(|| self.run_plan(inputs, &pruned_plan, &pruned_plan_output_ids, opts))?;
        let output_ids_and_values: Vec<_> =
            pruned_plan_output_ids.into_iter().zip(outputs).collect();
        Ok(output_ids_and_values)
    }

    /// Prune a plan so that it contains only operators which can be executed
    /// given a subset of the inputs.
    ///
    /// Returns a tuple of `(pruned_plan, new_outputs)` where `new_outputs`
    /// contains the IDs of leaf nodes in the pruned plan. These are the values
    /// that can still be generated by the reduced plan, and are either in
    /// the original `outputs` list or are inputs to parts of the plan that
    /// were pruned away.
    fn prune_plan(
        &self,
        plan: &[NodeId],
        inputs: &[NodeId],
        outputs: &[NodeId],
    ) -> (Vec<NodeId>, Vec<NodeId>) {
        let mut resolved_values = self.init_resolved_values(inputs.iter().copied());
        let mut pruned_plan = Vec::new();
        let mut candidate_outputs = Vec::new();

        // IDs of input nodes for pruned operators that we can still generate
        // with the pruned plan.
        let mut pruned_ops_resolved_inputs = FxHashSet::<NodeId>::default();

        // Walk forwards through the plan and prune away steps that cannot be
        // computed due to missing inputs.
        for &node_id in plan {
            let Some(Node::Operator(op_node)) = self.nodes.get(node_id) else {
                continue;
            };
            let all_inputs_available = op_node
                .inputs
                .iter()
                .filter_map(|id_opt| *id_opt)
                .all(|input_id| resolved_values.contains(&input_id));
            if !op_node.operator.is_deterministic() || !all_inputs_available {
                for input_id in op_node.inputs.iter().filter_map(|id_opt| *id_opt) {
                    if resolved_values.contains(&input_id) {
                        pruned_ops_resolved_inputs.insert(input_id);
                    }
                }
                continue;
            }
            resolved_values.extend(op_node.outputs.iter().filter_map(|id_opt| *id_opt));
            pruned_plan.push(node_id);
            candidate_outputs.extend(op_node.outputs.iter().filter_map(|id_opt| *id_opt));
        }

        // Get IDs of values produced by the pruned plan which are either in the
        // originally requested set of outputs, or are inputs to steps of the
        // original plan that were pruned away.
        let new_outputs: Vec<NodeId> = candidate_outputs
            .into_iter()
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
        inputs: &[(NodeId, InputOrOutput)],
        outputs: &[NodeId],
        options: PlanOptions,
    ) -> Result<Vec<NodeId>, RunError> {
        if !all_unique(outputs, |x, y| x == y) {
            return Err(RunError::PlanningError("output IDs are not unique".into()));
        }

        if !all_unique(inputs, |(x_id, _), (y_id, _)| x_id == y_id) {
            return Err(RunError::PlanningError("input IDs are not unique".into()));
        }

        // Build an execution plan via a depth first traversal of the graph
        // starting at the output nodes. A helper struct is used as recursive
        // closures are not supported in Rust.
        struct PlanBuilder<'a> {
            graph: &'a Graph,
            resolved_values: FxHashSet<NodeId>,
            plan: Vec<(NodeId, &'a OperatorNode)>,
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
                    if let Some((input_op_id, input_op_node)) = self.graph.get_source_node(input) {
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
            fn plan(mut self, outputs: &[NodeId]) -> Result<Vec<NodeId>, RunError> {
                for output_id in outputs.iter() {
                    if self.resolved_values.contains(output_id) {
                        // Value is either a constant node or is produced by
                        // an operator that is already in the plan.
                        continue;
                    }

                    if let Some((op_node_id, op_node)) = self.graph.get_source_node(*output_id) {
                        self.visit(op_node_id, op_node)?;
                    } else {
                        let msg = format!("Missing output {}", output_id);
                        return Err(RunError::PlanningError(msg));
                    }
                }
                Ok(self.plan.into_iter().map(|(node_id, _)| node_id).collect())
            }
        }

        // Set of values that are available after executing the plan
        let resolved_values: FxHashSet<NodeId> =
            self.init_resolved_values(inputs.iter().map(|(node_id, _)| *node_id));

        let builder = PlanBuilder {
            graph: self,
            resolved_values,
            plan: Vec::new(),
            options,
        };
        builder.plan(outputs)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::{Arc, Mutex};

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::{expect_equal, expect_equal_with_tolerance};
    use rten_tensor::{Tensor, TensorView};

    use smallvec::smallvec;

    use super::CachedPlan;
    use crate::graph::{Dimension, Graph, Node, RunError, TypedConstant};
    use crate::ops::{
        Add, Concat, Conv, InputList, IntoOpResult, OpError, Operator, Output, OutputList, Relu,
        Shape,
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

        fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
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

        let (_, conv_out) = g.add_simple_op(
            "conv",
            Conv {
                dilations: vec![1, 1],
                groups: 1,
                padding: [1, 1, 1, 1].into(),
                strides: vec![1, 1],
            },
            &[input_id, weights_id],
        );
        let (_, relu_out) = g.add_simple_op("relu", Relu {}, &[conv_out]);

        let input = Tensor::from_data(
            &[1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let results = g
            .run(vec![(input_id, input.into())], &[relu_out], None)
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
        let (relu_op_id, _) = g.add_simple_op("relu", Relu {}, &[input_id]);

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

        fn run(&self, _pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
            let input: TensorView<f32> = inputs.require_as(0)?;
            let output_data: Vec<f32> = input.iter().map(|x| x + 1.0).collect();
            Tensor::<f32>::from_data(input.shape().into(), output_data).into_op_result()
        }
    }

    #[test]
    fn test_graph_planning_order() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let input_id = g.add_value(Some("input"), None);

        let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);
        let (_, op_b_out) = g.add_simple_op("op_b", AddOne {}, &[op_a_out]);

        // op_c has both op_a and op_b as inputs. Since op_b depends on op_a,
        // execution must run op_a, then op_b, then op_c.
        let (_, op_c_out) = g.add_simple_op("op_c", Concat { axis: 0 }, &[op_a_out, op_b_out]);

        // op_d is the same as op_c, but input order is reversed
        let (_, op_d_out) = g.add_simple_op("op_d", Concat { axis: 0 }, &[op_b_out, op_a_out]);

        let input = Tensor::from_data(&[1], vec![1.]);

        let results = g
            .run(vec![(input_id, input.view().into())], &[op_c_out], None)
            .unwrap();
        let expected = Tensor::from_data(&[2], vec![2., 3.]);
        expect_equal(results[0].as_float_ref().unwrap(), &expected)?;

        let results = g
            .run(vec![(input_id, input.into())], &[op_d_out], None)
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
        let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);
        let (_, op_b_out) = g.add_simple_op("op_b", AddOne {}, &[op_a_out]);

        let input = Tensor::from(0.);
        let results = g
            .run(vec![(input_id, input.into())], &[op_a_out, op_b_out], None)
            .unwrap();
        assert_eq!(results[0].as_float_ref().unwrap(), &Tensor::from(1.));
        assert_eq!(results[1].as_float_ref().unwrap(), &Tensor::from(2.));
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
            .run(vec![(input_id, input.into())], &[prev_output], None)
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
            .run(vec![(input_id, input.view().into())], &[input_id], None)
            .unwrap();

        expect_equal(results[0].as_float_ref().unwrap(), &input)?;

        Ok(())
    }

    #[test]
    fn test_constant_graph() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let value = Tensor::from_data(&[5], vec![1., 2., 3., 4., 5.]);
        let const_id = g.add_constant(Some("weight"), value.clone());

        let results = g.run(vec![], &[const_id], None).unwrap();

        expect_equal(results[0].as_float_ref().unwrap(), &value)?;

        Ok(())
    }

    #[test]
    fn test_typed_constant() {
        let mut g = Graph::new();
        let scalar_id = g.add_constant(None, Tensor::from(42.));
        let vec_id = g.add_constant(None, Tensor::from([1, 2, 3]));

        let scalar_node = match g.get_node(scalar_id) {
            Some(Node::Constant(c)) => Some(c),
            _ => None,
        }
        .unwrap();
        let vec_node = match g.get_node(vec_id) {
            Some(Node::Constant(c)) => Some(c),
            _ => None,
        }
        .unwrap();

        assert_eq!(scalar_node.as_scalar(), Some(42.0));
        assert_ne!(scalar_node.as_scalar(), Some(42));
        assert_eq!(vec_node.as_scalar(), None::<i32>);

        assert_eq!(vec_node.as_vector(), Some([1, 2, 3].as_slice()));
        assert_eq!(vec_node.as_scalar(), None::<f32>);
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
        let results = g.run(vec![], &[], None).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_duplicate_inputs() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None);
        let input = Tensor::from([1.]);
        let result = g.run(
            vec![
                (input_id, input.view().into()),
                (input_id, input.view().into()),
            ],
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
        let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);

        let input = Tensor::from([1.]);

        let result = g.run(vec![(input_id, input.into())], &[op_a_out, op_a_out], None);

        assert_eq!(
            result,
            Err(RunError::PlanningError("output IDs are not unique".into()))
        );
    }

    #[test]
    fn test_call_op_with_missing_input() {
        let mut g = Graph::new();

        // Call an operator with an input omitted by setting it to `None`,
        // as opposed to passing a shorter input list. This enables omitting
        // an input but still providing subsequent ones.
        let output = g.add_value(None, None);
        g.add_op(Some("shape"), Box::new(Shape {}), &[None], &[Some(output)]);

        let results = g.run(vec![], &[output], None);

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
        let result = g.run(vec![], &[123], None);
        assert_eq!(
            result.err(),
            Some(RunError::PlanningError("Missing output 123".to_string()))
        );
    }

    #[test]
    fn test_err_if_missing_operator_input() {
        let mut g = Graph::new();
        let (_, output) = g.add_simple_op("op", Relu {}, &[42]);
        let result = g.run(vec![], &[output], None);
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

        fn run(&self, _pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
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

        let (_, op1_out) = g.add_simple_op("op1", AddOneInPlace {}, &[input_id]);
        let (_, op2_out) = g.add_simple_op("op2", AddOneInPlace {}, &[op1_out]);
        let (_, op3_out) = g.add_simple_op("op3", AddOneInPlace {}, &[op2_out]);
        let (_, op4_out) = g.add_simple_op("op4", AddOneInPlace {}, &[op2_out]);
        let input = Tensor::<f32>::zeros(&[1, 1]);

        // First operator should not be run in-place, since it has an
        // immutable input. The result should be the same as the input.
        let results = g
            .run(vec![(input_id, input.view().into())], &[op1_out], None)
            .unwrap();
        assert_eq!(results[0].as_float_ref().unwrap()[[0, 0]], 0.0);

        // Second operator should be run in-place, as it meets all the
        // requirements for this optimization.
        let results = g
            .run(vec![(input_id, input.view().into())], &[op2_out], None)
            .unwrap();
        assert_eq!(results[0].as_float_ref().unwrap()[[0, 0]], 1.0);

        // Third op should not be run in place, because its input is re-used
        // for fourth op. Fourth op can run in place as by then, it is the
        // only consumer of its input.
        let results = g
            .run(
                vec![(input_id, input.view().into())],
                &[op3_out, op4_out],
                None,
            )
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

        let (_, op1_out) = g.add_simple_op("op1", op1, &[input_id, bias_id]);
        let (_, op2_out) = g.add_simple_op(
            "op2",
            op2,
            // Note here the input ordering. The bias value is smaller, but
            // is the first argument. This operator can run in place, but only
            // if the inputs are swapped.
            &[bias_id, op1_out],
        );
        let input = Tensor::<f32>::zeros(&[2, 2]);
        let bias = Tensor::from(1.5);

        let results = g
            .run(
                vec![(input_id, input.view().into()), (bias_id, bias.into())],
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

        // The first operator must copy its input because it is a view.
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

        fn run(&self, _pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
            {
                let mut rc = self.run_count.lock().unwrap();
                *rc += 1;
            }

            let input: TensorView<f32> = inputs.require_as(0)?;
            let left_split_len = input.len() / 2;
            let left_split = Tensor::from_vec(input.iter().take(left_split_len).copied().collect());
            let right_split =
                Tensor::from_vec(input.iter().skip(left_split_len).copied().collect());
            Ok(smallvec![left_split.into(), right_split.into()])
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

        let input = Tensor::from([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut results = g
            .run(
                vec![(input_id, input.into())],
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
        let const_0 = g.add_constant(Some("c0"), Tensor::from(3.));
        let val_0 = g.add_value(Some("i0"), None);
        let const_1 = g.add_constant(Some("c1"), Tensor::from(4.));
        let val_1 = g.add_value(Some("i1"), None);

        let (_, op_0_out) = g.add_simple_op("Add_0", Add {}, &[const_0, val_0]);
        let (_, op_1_out) = g.add_simple_op("Add_1", Add {}, &[const_1, val_1]);
        let (_, op_2_out) = g.add_simple_op("Add_2", Add {}, &[op_0_out, op_1_out]);

        // Run graph with no inputs. This is equivalent to constant evaluation.
        // In this case no operators can be evaluated with graph constants
        // alone, so the output is empty.
        let partial_outs = g.partial_run(vec![], &[op_2_out], None)?;
        assert_eq!(partial_outs.len(), 0);

        // Run graph with just the `V0` input. This will compute the result of
        // `Op0` but not other nodes which depend on `V1`.
        let input = Tensor::from(2.);
        let partial_outs = g.partial_run(vec![(val_0, input.view().into())], &[op_2_out], None)?;
        assert_eq!(partial_outs.len(), 1);
        assert_eq!(partial_outs[0].0, op_0_out);
        assert_eq!(partial_outs[0].1, Output::FloatTensor(Tensor::from(5.)));

        // Run graph with just the `V1` input. This will compute the result of
        // `Op1` but not other nodes which depend on `V0`.
        let input = Tensor::from(2.);
        let partial_outs = g.partial_run(vec![(val_1, input.view().into())], &[op_2_out], None)?;
        assert_eq!(partial_outs.len(), 1);
        assert_eq!(partial_outs[0].0, op_1_out);
        assert_eq!(partial_outs[0].1, Output::FloatTensor(Tensor::from(6.)));

        // Run graph with all inputs. This should behave like `Graph::run`.
        let partial_outs = g.partial_run(
            vec![(val_1, input.view().into()), (val_0, input.view().into())],
            &[op_2_out],
            None,
        )?;
        assert_eq!(partial_outs.len(), 1);
        assert_eq!(partial_outs[0].0, op_2_out);
        assert_eq!(partial_outs[0].1, Output::FloatTensor(Tensor::from(11.)));

        Ok(())
    }

    #[derive(Debug)]
    struct Counter {
        count: AtomicI32,
    }

    impl Operator for Counter {
        fn name(&self) -> &str {
            "Counter"
        }

        fn is_deterministic(&self) -> bool {
            false
        }

        fn run(&self, _pool: &TensorPool, _inputs: InputList) -> Result<OutputList, OpError> {
            let count = self.count.fetch_add(1, Ordering::SeqCst);
            Ok([Tensor::from(count).into()].into())
        }
    }

    #[test]
    fn test_partial_run_non_deterministic_ops() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();
        let const_val = g.add_constant(Some("c0"), Tensor::from(3));

        // Add deterministic op with constant inputs.
        let (_, add_op_0_out) = g.add_simple_op("Add_0", Add {}, &[const_val, const_val]);

        // Add non-deterministic op.
        let (_, count_op_out) = g.add_simple_op(
            "Count",
            Counter {
                count: AtomicI32::new(0),
            },
            &[],
        );

        // Add final op that combines outputs from other ops.
        let (_, add_op_1_out) = g.add_simple_op("Add_1", Add {}, &[add_op_0_out, count_op_out]);

        // Do a partial run with no inputs. This should propagate constants
        // though all the deterministic operators, but skip any
        // non-deterministic ops.
        let partial_outs = g.partial_run(vec![], &[add_op_1_out], None)?;
        assert_eq!(partial_outs.len(), 1);
        assert_eq!(partial_outs[0].0, add_op_0_out);

        Ok(())
    }

    #[test]
    fn test_cached_plan_matches() {
        let input_ids = &[3, 1, 2];
        let output_ids = &[6, 4, 5];
        let op_ids = &[10, 11, 12];

        let plan = CachedPlan::new(input_ids, output_ids, op_ids.to_vec());

        assert!(plan.matches(input_ids, output_ids));

        // Same input and output IDs, different orders.
        assert!(plan.matches(&[1, 2, 3], &[4, 5, 6]));
        assert!(plan.matches(&[3, 2, 1], &[6, 5, 4]));

        // Different input and output IDs
        assert!(!plan.matches(&[20, 21, 22], output_ids));
        assert!(!plan.matches(input_ids, &[20, 21, 22]));
    }
}
