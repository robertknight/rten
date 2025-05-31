use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::hash::BuildHasherDefault;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rten_tensor::prelude::*;

// The std HashMap/HashSet provide DOS resistance. In this module hash keys are
// mostly `NodeId`s which we allocate ourselves, so this is not a concern.
// Instead we want faster hashing.
use rustc_hash::FxHashMap;

use smallvec::SmallVec;

use crate::env::env_flag;
use crate::ops::{
    DataType, Input, InputList, InputMeta, InputOrOutput, OpError, OpRunContext, Operator, Output,
    OutputList, PrepackedInput,
};
use crate::tensor_pool::TensorPool;
use crate::threading;
use crate::timing::{Instant, ProfileFormat, Profiler, TimingRecord, TimingSort};
use crate::weight_cache::WeightCache;

#[cfg(test)]
pub mod builder;

mod capture_env;
pub use capture_env::CaptureEnv;
mod node;
use node::ValueNode;
pub use node::{
    Constant, ConstantNode, ConstantNodeData, Dimension, Node, OperatorNode, TypedConstant,
};
mod noop_hash;
use noop_hash::NoopHashMap;
mod planner;
use planner::{CachedPlan, PlanOptions, Planner};

mod node_id;
pub use node_id::NodeId;

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

        /// Shape and dtype of operator inputs.
        ///
        /// This can be `None` if input metadata was not captured. Individual
        /// inputs can be `None` if it is a missing optional input.
        inputs: Option<Vec<Option<InputMeta>>>,
    },

    /// The output of a graph operator did not match expectations (eg. the
    /// count, types or shapes of outputs did not match what was expected.)
    OutputMismatch(&'static str),
}

impl RunError {
    pub(crate) fn op_error(name: &str, error: OpError, ctx: Option<&OpRunContext>) -> Self {
        RunError::OperatorError {
            name: name.to_string(),
            error,
            inputs: ctx.map(|ctx| {
                ctx.inputs()
                    .iter()
                    .map(|inp| inp.map(|inp| inp.to_meta()))
                    .collect()
            }),
        }
    }
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
                inputs,
            } => write!(
                f,
                "operator \"{}\" failed: {:?}. Inputs {:?}",
                name, err, inputs
            ),
            RunError::OutputMismatch(err) => write!(f, "output mismatch {:?}", err),
        }
    }
}

impl Error for RunError {}

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
        let rc = &mut self.rc[id.as_usize()];
        *rc = rc.saturating_add(1);
    }

    /// Decrement ref count of node and return new count, or `None` if the
    /// ref count was already zero.
    fn dec(&mut self, id: NodeId) -> Option<usize> {
        let rc = &mut self.rc[id.as_usize()];

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
        self.rc[id.as_usize()] as usize
    }
}

/// Options that control logging and other behaviors when executing a
/// [`Model`](crate::Model).
#[derive(Clone, Default)]
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

    /// The thread pool to execute the model on. By default the model is
    /// executed on the global thread pool.
    pub thread_pool: Option<Arc<threading::ThreadPool>>,
}

impl RunOptions {
    /// Return the thread pool to use for inference.
    ///
    /// This is either the pool specified via the `thread_pool` field or the
    /// global thread pool.
    pub(crate) fn thread_pool(&self) -> &threading::ThreadPool {
        self.thread_pool
            .as_deref()
            .unwrap_or(threading::thread_pool())
    }
}

impl std::fmt::Debug for RunOptions {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("RunOptions")
            .field("timing", &self.timing)
            .field("timing_sort", &self.timing_sort)
            .field("timing_by_shape", &self.timing_by_shape)
            .field("verbose", &self.verbose)
            .finish()
    }
}

// PartialEq impl that ignores non-comparable fields.
impl PartialEq<Self> for RunOptions {
    fn eq(&self, other: &Self) -> bool {
        self.timing == other.timing
            && self.timing_sort == other.timing_sort
            && self.timing_by_shape == other.timing_by_shape
            && self.verbose == other.verbose
    }
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
/// ## Input and output nodes
///
/// A subset of the nodes are designated as the default inputs and outputs.
/// These constitute the "public API" of the graph and will be preserved after
/// any optimizations applied to the graph structure at runtime. Other
/// "internal" nodes may be replaced or removed.
///
/// ## Captured nodes
///
/// Control flow operators such as `If` and `Loop` execute subgraphs. Operators
/// in these subgraphs may reference inputs which are not computed by the
/// subgraph but are instead looked up by name in parent graphs. Nodes are
/// created in the subgraph to represent these captured values, and their IDs
/// are referenced in operator input lists. The IDs of all these nodes are
/// returned by [`captures`](Graph::captures).
pub struct Graph {
    /// Nodes that make up the graph. The graph's edges are stored as part of
    /// operator nodes.
    nodes: NoopHashMap<NodeId, Node>,

    next_node_id: u32,

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

    /// IDs of nodes that represent values captured from the parent scope.
    captures: Vec<NodeId>,
}

impl Graph {
    /// Create a new empty graph.
    pub fn new() -> Graph {
        Self::with_capacity(0)
    }

    /// Create a new graph with pre-allocated storage space for nodes.
    pub fn with_capacity(n_nodes: usize) -> Graph {
        Graph {
            nodes: HashMap::with_capacity_and_hasher(n_nodes, BuildHasherDefault::new()),
            next_node_id: 0,
            cached_plan: Mutex::new(None),
            source_ids: FxHashMap::default(),
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            captures: Vec::new(),
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

    /// Set the IDs of nodes whose values are captured from the enclosing scope.
    pub fn set_captures(&mut self, captures: &[NodeId]) {
        self.captures = captures.to_vec()
    }

    /// Return the IDs of nodes whose values are captured from the enclosing
    /// scope.
    ///
    /// This does not include transitive captures in subgraphs.
    pub fn captures(&self) -> &[NodeId] {
        &self.captures
    }

    /// Remove nodes from the graph.
    ///
    /// This method accepts a list of node IDs as it is more efficient to
    /// remove nodes in batches.
    pub fn remove_nodes(&mut self, node_ids: &[NodeId]) {
        self.clear_cached_plan();
        self.input_ids.retain(|id| !node_ids.contains(id));
        self.output_ids.retain(|id| !node_ids.contains(id));
        self.captures.retain(|id| !node_ids.contains(id));
        self.source_ids
            .retain(|val_id, op_id| !node_ids.contains(val_id) && !node_ids.contains(op_id));
        for node_id in node_ids {
            if let Some(name) = self.nodes.get(node_id).and_then(|n| n.name()) {
                self.node_id_from_name.remove(name);
            }
            self.nodes.remove(node_id);
        }
    }

    /// Return an iterator over the names of nodes whose values are captured
    /// from the parent graph.
    ///
    /// This does include transitive captures from subgraphs.
    fn capture_names(&self) -> Vec<&str> {
        let mut captures: Vec<&str> = self
            .captures()
            .iter()
            .filter_map(|&cap_node_id| self.get_node(cap_node_id).and_then(|n| n.name()))
            .collect();

        for node in self.nodes.values() {
            if let Node::Operator(op) = node {
                for subgraph in op.operator().subgraphs() {
                    captures.extend(subgraph.capture_names())
                }
            }
        }

        captures
    }

    /// Add a node to the graph and return its ID.
    ///
    /// This contains the common logic for adding different types of node to
    /// the graph.
    fn add_node(&mut self, node: Node) -> NodeId {
        let node_id = NodeId::from_u32(self.next_node_id);
        self.nodes.insert(node_id, node);
        self.next_node_id += 1;

        if let Some(name) = self.nodes.get(&node_id).unwrap().name() {
            self.node_id_from_name.insert(name.to_string(), node_id);
        }

        node_id
    }

    /// Invalidate cached execution plans.
    fn clear_cached_plan(&mut self) {
        if let Ok(plan) = self.cached_plan.get_mut() {
            plan.take();
        }
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
    /// written to. If there is already an existing operator which uses the
    /// same output, the new operator will become the source for this output
    /// value. This enables replacing an operator while preserving metadata
    /// of the output value (name, shape etc.).
    ///
    /// Returns the ID of the operator node.
    pub fn add_op(
        &mut self,
        name: Option<&str>,
        op: Box<dyn Operator + Send + Sync>,
        inputs: &[Option<NodeId>],
        outputs: &[Option<NodeId>],
    ) -> NodeId {
        let op_node = Node::Operator(OperatorNode::new(name, inputs, outputs, op));
        let op_id = self.add_node(op_node);

        for output_id in outputs.iter().flatten() {
            self.source_ids.insert(*output_id, op_id);
        }

        // Clear cached plan in case we just replaced the source operator for
        // one of the output IDs.
        self.clear_cached_plan();

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
        let op_out_id = self.add_value(Some(&op_out_name), None, None);
        let input_ids: Vec<_> = input_ids.iter().copied().map(Some).collect();
        let op_node_id = self.add_op(Some(name), Box::new(op), &input_ids, &[op_out_id].map(Some));
        (op_node_id, op_out_id)
    }

    /// Convert `value` to a constant node and add it to the graph.
    ///
    /// `name` is an identifier for this node that is used in debug messages etc.
    ///
    /// Returns the ID of the added node.
    pub fn add_constant<T, V>(&mut self, name: Option<&str>, value: V) -> NodeId
    where
        V: Into<ConstantNodeData<T>>,
        ConstantNode<T>: Into<Constant>,
    {
        let const_node: Constant = ConstantNode::new(name, value.into()).into();
        self.add_constant_node(const_node)
    }

    /// Pre-pack constant inputs (ie. weights) to operators.
    ///
    /// When loading models, prepacking should be performed after graph
    /// optimization. There may be other nodes in between the weight constant
    /// and the compute node, which would prevent prepacking. Graph optimization
    /// can eliminate these. A common example is when weights are transposed.
    pub fn prepack_weights(&self, cache: &mut WeightCache) {
        for (op_node_id, op_node) in self.iter().filter_map(|(node_id, node)| match node {
            Node::Operator(op) => Some((node_id, op)),
            _ => None,
        }) {
            for input_index in op_node.operator().prepack_inputs() {
                let Some(input_id) = op_node.input_ids().get(input_index).copied().flatten() else {
                    continue;
                };

                if cache.contains(input_id) {
                    // Input was already pre-packed. This might happen if the
                    // input is used by multiple operators.
                    continue;
                }

                let Some(Node::Constant(const_node)) = self.get_node(input_id) else {
                    // Input is a value computed during inference, so we don't have it to prepack.
                    continue;
                };

                let Some(packed) = op_node
                    .operator()
                    .prepack(input_index, const_node.as_input())
                else {
                    // Operator doesn't support or decided not to prepack this value.
                    continue;
                };

                cache.insert(input_id, packed);
            }

            let subgraph_caches: Vec<_> = op_node
                .operator()
                .subgraphs()
                .into_iter()
                .map(|subgraph| {
                    let mut subgraph_cache = WeightCache::new();
                    subgraph.prepack_weights(&mut subgraph_cache);
                    subgraph_cache
                })
                .collect();
            cache.insert_subgraph_caches(op_node_id, subgraph_caches);
        }
    }

    /// Add a constant node to the graph.
    ///
    /// Returns the ID of the added node.
    pub fn add_constant_node(&mut self, node: Constant) -> NodeId {
        self.add_node(Node::Constant(node))
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
    pub fn add_value(
        &mut self,
        name: Option<&str>,
        shape: Option<Vec<Dimension>>,
        dtype: Option<DataType>,
    ) -> NodeId {
        let value_node = Node::Value(ValueNode::new(name, shape, dtype));
        self.add_node(value_node)
    }

    /// Return an iterator over nodes in the graph.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().map(|(id, node)| (*id, node))
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
        self.nodes.get(&id)
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
        self.nodes.get_mut(&id)
    }

    /// Return the total number of parameters in all constant nodes in this
    /// graph and subgraphs.
    pub fn total_params(&self) -> usize {
        self.nodes
            .values()
            .map(|node| match node {
                Node::Operator(op_node) => op_node
                    .operator()
                    .subgraphs()
                    .iter()
                    .map(|sg| sg.total_params())
                    .sum(),
                Node::Value(_) => 0,
                Node::Constant(constant) => constant.layout().len(),
            })
            .sum()
    }

    /// Return the sequence of operators from the current graph that would be
    /// executed in order to compute `outputs` given `inputs`, without actually
    /// running the model.
    ///
    /// The result does not include nodes from any subgraphs that an operator
    /// may run.
    pub fn execution_plan(
        &self,
        inputs: &[NodeId],
        outputs: &[NodeId],
    ) -> Result<Vec<NodeId>, RunError> {
        self.create_plan(inputs, outputs, false /* is_subgraph */)
    }

    /// Compute a set of output values given a set of inputs, using the
    /// processing steps and constant values defined by the graph.
    pub fn run(
        &self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        weight_cache: Option<&WeightCache>,
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let input_ids: Vec<_> = inputs.iter().map(|(node_id, _)| *node_id).collect();
        let plan = self.get_cached_plan(&input_ids, outputs, false /* is_subgraph */)?;
        let opts = opts.unwrap_or_default();
        opts.thread_pool().run(|| {
            let mut profiler =
                (opts.timing || opts.verbose).then(|| Profiler::with_capacity(plan.plan().len()));

            let result = self.run_plan(
                inputs,
                plan.plan(),
                outputs,
                None, /* captures */
                None, /* pool */
                weight_cache,
                profiler.as_mut(),
                &opts,
            );

            if let Some(profiler) = &profiler {
                let print_opts = ProfileFormat {
                    timing_sort: opts.timing_sort.clone(),
                    timing_by_shape: opts.timing_by_shape,
                };
                profiler.print(print_opts);
            }

            result
        })
    }

    /// Compute output values from a subgraph.
    ///
    /// This method is like [`run`](Self::run) but has a `captures` argument
    /// which allows the subgraph to access values in the parent scope.
    pub fn run_subgraph<'a>(
        &'a self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        captures: CaptureEnv,
        pool: Option<&TensorPool>,
        weight_cache: Option<&WeightCache>,
        profiler: Option<&mut Profiler<'a>>,
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let input_ids: Vec<_> = inputs.iter().map(|(node_id, _)| *node_id).collect();
        let plan = self.get_cached_plan(&input_ids, outputs, true /* is_subgraph */)?;
        let opts = opts.unwrap_or_default();
        self.run_plan(
            inputs,
            plan.plan(),
            outputs,
            Some(captures),
            pool,
            weight_cache,
            profiler,
            &opts,
        )
    }

    fn get_cached_plan(
        &self,
        inputs: &[NodeId],
        outputs: &[NodeId],
        is_subgraph: bool,
    ) -> Result<Arc<CachedPlan>, RunError> {
        // Reuse the plan from the previous run if the input and output IDs
        // match, otherwise create a new one.
        //
        // Note that we only hold the plan lock while creating the plan,
        // not while executing the model.
        let mut cached_plan = self.cached_plan.lock().unwrap();
        let plan = match cached_plan.as_ref() {
            Some(plan) if plan.matches(inputs, outputs) => plan.clone(),
            _ => {
                let plan = self.create_plan(inputs, outputs, is_subgraph)?;
                *cached_plan = Some(Arc::new(CachedPlan::new(inputs, outputs, plan)));
                cached_plan.clone().unwrap()
            }
        };
        Ok(plan)
    }

    fn create_plan(
        &self,
        inputs: &[NodeId],
        outputs: &[NodeId],
        is_subgraph: bool,
    ) -> Result<Vec<NodeId>, RunError> {
        let planner = Planner::with_graph(self);
        planner.create_plan(
            inputs,
            outputs,
            PlanOptions {
                allow_missing_inputs: false,
                captures_available: is_subgraph,
            },
        )
    }

    fn run_plan<'a>(
        &'a self,
        mut inputs: Vec<(NodeId, InputOrOutput)>,
        plan: &[NodeId],
        outputs: &[NodeId],
        mut captures: Option<CaptureEnv>,
        pool: Option<&TensorPool>,
        weight_cache: Option<&WeightCache>,
        mut profiler: Option<&mut Profiler<'a>>,
        opts: &RunOptions,
    ) -> Result<Vec<Output>, RunError> {
        let mut temp_values: FxHashMap<NodeId, Output> = FxHashMap::default();

        // Extract all owned tensor inputs into the owned value map.
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
            match self.nodes.get(&node_id) {
                Some(Node::Constant(constant)) => Some(constant.as_input()),
                Some(Node::Value(_)) => inputs_by_id.get(&node_id).map(|input| input.as_input()),
                _ => {
                    panic!("node {} is not a value or constant", node_id);
                }
            }
        };

        fn get_value_from_capture<'a>(
            nodes: &NoopHashMap<NodeId, Node>,
            captures: Option<&'a CaptureEnv>,
            node_id: NodeId,
        ) -> Option<Input<'a>> {
            let name = nodes.get(&node_id).and_then(|n| n.name())?;
            captures.and_then(|cap| cap.get_input(name))
        }

        // Count how often each temporary output is used, so we can free them
        // when no longer needed.
        let mut temp_value_refcount = NodeRefCount::with_capacity(self.next_node_id as usize);
        for &op_node_id in plan.iter() {
            let Some(Node::Operator(op_node)) = self.nodes.get(&op_node_id) else {
                return Err(RunError::PlanningError(
                    "operator node not found".to_string(),
                ));
            };
            for node_id in self.operator_dependencies(op_node) {
                if let Some(Node::Value(_)) = self.nodes.get(&node_id) {
                    temp_value_refcount.inc(node_id);
                }
            }
        }

        // Increment usage count of all output nodes, so we retain them after
        // the operator has run.
        for node_id in outputs {
            temp_value_refcount.inc(*node_id);
        }

        // Create or re-use pool for buffer allocations.
        //
        // If the feature flag is off, we still create the pool, but never
        // release buffers back into it, so all allocations use the system
        // allocator.
        let new_pool = TensorPool::new();
        let pool = pool.unwrap_or(&new_pool);
        let use_pool = env_flag("RTEN_USE_POOL", true);

        // Execute the plan
        let mut op_start = Instant::now();

        for (step, &op_node_id) in plan.iter().enumerate() {
            let Some(Node::Operator(op_node)) = self.nodes.get(&op_node_id) else {
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
            let in_place_input_id = if op_node.operator().can_run_in_place() {
                if op_node.operator().is_commutative() {
                    // Pick the largest input by number of elements. This
                    // assumes that commutative op outputs will have a shape
                    // that matches their largest input (eg. consider a
                    // binary op that broadcasts inputs to a common shape).
                    op_node
                        .input_ids()
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
                    op_node.input_ids().first().copied().flatten()
                }
            } else {
                None
            };

            // Take a value for passing to an operator as an owned value, if
            // it won't be needed by other operators in future.
            let mut take_value = |node_id| {
                if temp_value_refcount.count(node_id) == 1 {
                    if let Some(value) = temp_values.remove(&node_id) {
                        Some(value)
                    } else if self.captures.contains(&node_id) {
                        let name = self.nodes.get(&node_id).and_then(|n| n.name())?;
                        captures.as_mut().and_then(|cap| cap.take_input(name))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            // If the operator can run in place, check if we have a tensor
            // that can be used as the output. This requires that the tensor
            // is not a constant (eg. weights) and is not going to be used by
            // other ops in future.
            let in_place_input = in_place_input_id.and_then(&mut take_value);

            // Extract values used by the operator's subgraphs which can be
            // passed by value.
            let has_subgraph = op_node.operator().has_subgraph();
            let by_value_captures = has_subgraph.then(|| {
                let mut by_value_captures = FxHashMap::default();
                for node_id in self.operator_dependencies(op_node) {
                    if op_node.input_ids().contains(&Some(node_id)) {
                        continue;
                    }
                    if let Some(tensor) = take_value(node_id) {
                        by_value_captures.insert(node_id, tensor);
                    }
                }
                by_value_captures
            });

            // Collect all or remaining inputs for the operator
            let mut op_inputs: SmallVec<[Option<Input>; 4]> =
                SmallVec::with_capacity(op_node.input_ids().len());
            for node_id in op_node.input_ids().iter() {
                if in_place_input.is_some() && *node_id == in_place_input_id {
                    continue;
                }

                if let Some(node_id) = node_id {
                    if let Some(value) = get_value_from_constant_or_input(*node_id) {
                        op_inputs.push(Some(value));
                    } else if let Some(value) = temp_values.get(node_id) {
                        op_inputs.push(Some(value.as_input()));
                    } else if let Some(value) =
                        get_value_from_capture(&self.nodes, captures.as_ref(), *node_id)
                    {
                        op_inputs.push(Some(value))
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

            // Collect input metadata if we'll need it for timing or logging.
            let input_meta = if opts.timing_by_shape || opts.verbose {
                let mut meta: Vec<Option<InputMeta>> = Vec::new();
                if let Some(ref input) = in_place_input {
                    meta.push(Some(input.to_meta()));
                }
                for input in &op_inputs {
                    meta.push(input.as_ref().map(|i| i.to_meta()))
                }
                meta
            } else {
                Vec::new()
            };

            // Run the operation.
            let op_result = if let Some(input) = in_place_input {
                let inputs = InputList::from_optional(&op_inputs);
                let ctx = OpRunContext::new(pool, &inputs);
                op_node
                    .operator()
                    .run_in_place(input, &ctx)
                    .map(|out| [out].into())
                    .map_err(|e| {
                        // The error here is currently missing information about operator inputs.
                        RunError::op_error(op_node.name().unwrap_or_default(), e, None)
                    })
            } else if has_subgraph {
                let capture_env = CaptureEnv::new(
                    captures.as_ref(),
                    self,
                    Some(&inputs_by_id),
                    Some(&temp_values),
                    by_value_captures,
                );
                let inputs = InputList::from_optional(&op_inputs);
                let ctx = OpRunContext::new(pool, &inputs);
                op_node.operator().run_subgraph(
                    &ctx,
                    capture_env,
                    weight_cache.and_then(|wc| wc.get_subgraph_caches(op_node_id)),
                    profiler.as_deref_mut(),
                    Some(opts.clone()),
                )
            } else {
                let get_prepacked = |input_index: usize| -> Option<&PrepackedInput> {
                    op_node
                        .input_ids()
                        .get(input_index)
                        .copied()
                        .flatten()
                        .and_then(|node_id| weight_cache.and_then(|wc| wc.get(node_id)))
                };
                let inputs = InputList::from_optional(&op_inputs).with_prepacked(&get_prepacked);
                let mut ctx = OpRunContext::new(pool, &inputs);
                ctx.set_num_outputs(op_node.output_ids().len() as u32);
                op_node.operator().run(&ctx).map_err(|e| {
                    RunError::op_error(op_node.name().unwrap_or_default(), e, Some(&ctx))
                })
            };
            std::mem::drop(op_inputs);

            // Print verbose logs if enabled. This is done before checking the
            // op's result, so logs will contain details of the failed operation
            // in the event of an error.
            if opts.verbose {
                let op_duration = Instant::now() - op_start;
                self.print_op_timing(step, op_node, &op_result, op_duration, &input_meta);
            }

            // Extract outputs or fail if an error occurred.
            let outputs = op_result?;
            if op_node.output_ids().len() != outputs.len() {
                return Err(RunError::OutputMismatch(
                    "operator output count did not match expected count",
                ));
            }

            // Save outputs for future steps.
            temp_values.extend(
                op_node
                    .output_ids()
                    .iter()
                    .zip(outputs.into_iter())
                    .filter_map(|(output_id, output)| output_id.map(|id| (id, output))),
            );

            // Remove temporary values that are no longer needed
            for node_id in self.operator_dependencies(op_node) {
                let rc = temp_value_refcount.dec(node_id);
                if rc == Some(0) {
                    if let (true, Some(tensor)) = (use_pool, temp_values.remove(&node_id)) {
                        tensor.add_to_pool(pool)
                    }
                }
            }

            if let Some(profiler) = &mut profiler {
                let op_end = Instant::now();
                let op_duration = op_end - op_start;
                op_start = op_end;

                profiler.add_record(TimingRecord {
                    name: op_node.operator().name(),
                    input_meta,
                    elapsed: op_duration,
                    node_name: op_node.name().unwrap_or(""),
                });
            }
        }

        // Record memory allocation metrics
        if let Some(profiler) = &mut profiler {
            profiler.add_pool_metrics(pool.alloc_count(), pool.hit_count());
        }

        // Return the requested outputs
        let result = outputs
            .iter()
            .map(|output_id| {
                if let Some(value) = get_value_from_constant_or_input(*output_id) {
                    value.to_output()
                } else if let Some(value) =
                    get_value_from_capture(&self.nodes, captures.as_ref(), *output_id)
                {
                    value.to_output()
                } else {
                    // During execution planning we verified that each output
                    // ID is valid and unique, so this should always succeed.
                    temp_values.remove(output_id).expect("missing output value")
                }
            })
            .collect();

        // Release any unused captured values back into the pool for use by
        // parent graphs.
        if let Some(values) = captures.and_then(|mut cap| cap.take_all_inputs()) {
            for (_, value) in values {
                value.add_to_pool(pool);
            }
        }

        Ok(result)
    }

    /// Print detailed information about an operation just after it has run.
    fn print_op_timing(
        &self,
        step: usize,
        op_node: &OperatorNode,
        op_result: &Result<OutputList, RunError>,
        op_duration: Duration,
        input_meta: &[Option<InputMeta>],
    ) {
        println!(
            "#{} {} ({})",
            step,
            op_node.operator().name(),
            op_node.name().unwrap_or("")
        );
        for (index, (id, meta)) in op_node.input_ids().iter().zip(input_meta).enumerate() {
            if let (Some(id), Some(meta)) = (id, meta) {
                let name = self.node_name(*id);
                println!(
                    "  input {}: {} ({} {:?})",
                    index, name, meta.dtype, meta.shape
                );
            }
        }

        if let Ok(outputs) = op_result.as_ref() {
            for (index, (id, output)) in op_node.output_ids().iter().zip(outputs).enumerate() {
                let name = id.map(|id| self.node_name(id)).unwrap_or(String::new());
                println!("  output {}: {} ({:?})", index, name, output.shape());
            }
        }

        println!("  time: {:.3}ms", op_duration.as_secs_f64() * 1000.0);
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
        let input_ids: Vec<_> = inputs.iter().map(|(id, _)| id).copied().collect();
        let planner = Planner::with_graph(self);
        let plan = planner.create_plan(
            &input_ids,
            outputs,
            PlanOptions {
                allow_missing_inputs: true,
                captures_available: false,
            },
        )?;
        let (pruned_plan, pruned_plan_output_ids) = planner.prune_plan(&plan, &input_ids, outputs);
        let opts = opts.unwrap_or_default();
        let outputs = opts.thread_pool().run(|| {
            let mut profiler =
                (opts.timing || opts.verbose).then(|| Profiler::with_capacity(pruned_plan.len()));

            let result = self.run_plan(
                inputs,
                &pruned_plan,
                &pruned_plan_output_ids,
                None, /* captures */
                None, /* pool */
                None, /* weight cache */
                profiler.as_mut(),
                &opts,
            );

            if let Some(profiler) = &profiler {
                let print_opts = ProfileFormat {
                    timing_sort: opts.timing_sort.clone(),
                    timing_by_shape: opts.timing_by_shape,
                };
                profiler.print(print_opts);
            }

            result
        })?;
        let output_ids_and_values: Vec<_> =
            pruned_plan_output_ids.into_iter().zip(outputs).collect();
        Ok(output_ids_and_values)
    }

    /// Return the IDs of all nodes in the current graph that an operator
    /// depends on.
    ///
    /// This includes nodes used as input as well as captures if the operator
    /// has subgraphs.
    fn operator_dependencies<'a>(
        &'a self,
        op_node: &'a OperatorNode,
    ) -> impl Iterator<Item = NodeId> + Clone + 'a {
        op_node.input_ids().iter().filter_map(|id| *id).chain(
            op_node
                .operator()
                .subgraphs()
                .into_iter()
                .flat_map(|sg| sg.capture_names())
                .filter_map(move |cap_name| {
                    let cap_id = self.get_node_id(cap_name)?;
                    if !op_node.input_ids().contains(&Some(cap_id)) {
                        Some(cap_id)
                    } else {
                        // If the captured node is also used as an input,
                        // only yield it once in the output.
                        None
                    }
                }),
        )
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

    use smallvec::{smallvec, SmallVec};

    use super::{CachedPlan, CaptureEnv};
    use crate::graph::{Dimension, Graph, Node, NodeId, RunError, RunOptions, TypedConstant};
    use crate::ops::{
        Add, Concat, Conv, DataType, Identity, If, Input, IntoOpResult, MatMul, Mul, OpError,
        OpRunContext, Operator, Output, OutputList, PrepackedInput, Relu, Shape,
    };
    use crate::timing::Profiler;
    use crate::weight_cache::WeightCache;

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

        fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
            {
                let mut m = self.metrics.lock().unwrap();
                m.run_count += 1;
            }
            self.inner.run(ctx)
        }

        fn run_in_place(&self, input: Output, ctx: &OpRunContext) -> Result<Output, OpError> {
            {
                let mut m = self.metrics.lock().unwrap();
                m.run_in_place_count += 1;
            }
            self.inner.run_in_place(input, ctx)
        }
    }

    /// Operator that wraps a function.
    ///
    /// Useful for tests that want to inspect operator inputs.
    struct RunFn<F: Fn(&OpRunContext) -> Result<OutputList, OpError> + 'static> {
        run: F,
    }

    impl<F: Fn(&OpRunContext) -> Result<OutputList, OpError>> RunFn<F> {
        fn new(run: F) -> Self {
            Self { run }
        }
    }

    impl<F: Fn(&OpRunContext) -> Result<OutputList, OpError>> std::fmt::Debug for RunFn<F> {
        fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(fmt, "RunFn")
        }
    }

    impl<F: Fn(&OpRunContext) -> Result<OutputList, OpError>> Operator for RunFn<F> {
        fn name(&self) -> &str {
            "RunFn"
        }

        fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
            (self.run)(ctx)
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
        let input_id = g.add_value(Some("input"), None, None);

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
            .run(vec![(input_id, input.into())], &[relu_out], None, None)
            .unwrap();

        let expected = Tensor::from_data(
            &[1, 1, 3, 3],
            vec![
                1.5202, 1.5592, 0.9939, 1.7475, 2.6358, 1.3428, 1.0165, 1.1806, 0.8685,
            ],
        );
        assert_eq!(results.len(), 1);
        expect_equal_with_tolerance(
            &results[0].as_tensor_view().unwrap(),
            &expected.view(),
            1e-4,
            0.,
        )?;

        Ok(())
    }

    #[test]
    fn test_graph_node_debug_names() {
        let mut g = Graph::new();

        let weights = Tensor::from([0.3230]);
        let weights_id = g.add_constant(Some("weights"), weights.clone());
        let input_id = g.add_value(Some("input"), None, None);
        let relu_out_id = g.add_value(Some("relu_out"), None, None);
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
        let anon_input_id = g.add_value(None, None, None);
        let anon_out_id = g.add_value(None, None, None);
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
            None,
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

    #[test]
    fn test_graph_value_dtype() {
        let mut g = Graph::new();
        for dtype in [
            DataType::Float,
            DataType::Int32,
            DataType::UInt8,
            DataType::Int8,
        ] {
            let input_id = g.add_value(None, None, Some(dtype));
            let input_dtype = g.get_node(input_id).and_then(|n| n.dtype());
            assert_eq!(input_dtype, Some(dtype));
        }
    }

    #[derive(Debug)]
    struct AddOne {}
    impl Operator for AddOne {
        fn name(&self) -> &str {
            "AddOne"
        }

        fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
            let input: TensorView<f32> = ctx.inputs().require_as(0)?;
            let output_data: Vec<f32> = input.iter().map(|x| x + 1.0).collect();
            Tensor::<f32>::from_data(input.shape().into(), output_data).into_op_result()
        }
    }

    #[test]
    fn test_graph_planning_order() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let input_id = g.add_value(Some("input"), None, None);

        let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);
        let (_, op_b_out) = g.add_simple_op("op_b", AddOne {}, &[op_a_out]);

        // op_c has both op_a and op_b as inputs. Since op_b depends on op_a,
        // execution must run op_a, then op_b, then op_c.
        let (_, op_c_out) = g.add_simple_op("op_c", Concat { axis: 0 }, &[op_a_out, op_b_out]);

        // op_d is the same as op_c, but input order is reversed
        let (_, op_d_out) = g.add_simple_op("op_d", Concat { axis: 0 }, &[op_b_out, op_a_out]);

        let input = Tensor::from([1.]);

        let results = g
            .run(
                vec![(input_id, input.view().into())],
                &[op_c_out],
                None,
                None,
            )
            .unwrap();
        let expected = Tensor::from([2., 3.]);
        expect_equal(&results[0].as_tensor_view().unwrap(), &expected.view())?;

        let results = g
            .run(vec![(input_id, input.into())], &[op_d_out], None, None)
            .unwrap();
        let expected = Tensor::from([3., 2.]);
        expect_equal(&results[0].as_tensor_view().unwrap(), &expected.view())?;

        Ok(())
    }

    #[test]
    fn test_runs_non_in_place_ops_first() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let input_a_id = g.add_value(Some("input_a"), None, None);
        let input_b_id = g.add_value(Some("input_b"), None, None);

        let (add_op, add_out) = g.add_simple_op("add", Add {}, &[input_a_id, input_b_id]);
        let (shape_op, shape_out) = g.add_simple_op("shape", Shape::default(), &[input_a_id]);

        // The execution plan could run operators in either order and produce
        // the correct output. Since the `Add` op has the _potential_ to run in
        // place (if the input is passed as an owned value) and the `Shape` op
        // does not, the Shape op should be run first.
        let plan = g.execution_plan(&[input_a_id, input_b_id], &[add_out, shape_out])?;
        assert_eq!(plan, &[shape_op, add_op]);

        // Make sure the results are the same if the order of outputs is
        // swapped.
        let plan = g.execution_plan(&[input_a_id, input_b_id], &[shape_out, add_out])?;
        assert_eq!(plan, &[shape_op, add_op]);

        Ok(())
    }

    // Perform a graph run where one of the outputs is also an input for other
    // steps of the run.
    #[test]
    fn test_graph_intermediate_output() {
        let mut g = Graph::new();

        let input_id = g.add_value(Some("input"), None, None);
        let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);
        let (_, op_b_out) = g.add_simple_op("op_b", AddOne {}, &[op_a_out]);

        let input = Tensor::from(0.);
        let results = g
            .run(
                vec![(input_id, input.into())],
                &[op_a_out, op_b_out],
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            &results[0].as_tensor_view().unwrap(),
            &Tensor::from(1.).view()
        );
        assert_eq!(
            &results[1].as_tensor_view().unwrap(),
            &Tensor::from(2.).view()
        );
    }

    #[test]
    fn test_graph_many_steps() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let input = Tensor::from([1., 2., 3., 4., 5.]);
        let input_id = g.add_value(Some("input"), None, None);

        let mut prev_output = input_id;
        for _ in 0..100 {
            let next_output = g.add_value(None, None, None);
            g.add_op(
                None,
                Box::new(AddOne {}),
                &[Some(prev_output)],
                &[Some(next_output)],
            );
            prev_output = next_output;
        }

        let results = g
            .run(vec![(input_id, input.into())], &[prev_output], None, None)
            .unwrap();

        let expected = Tensor::from([101., 102., 103., 104., 105.]);
        expect_equal(&results[0].as_tensor_view().unwrap(), &expected.view())?;

        Ok(())
    }

    #[test]
    fn test_noop_graph() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let input = Tensor::from([1., 2., 3., 4., 5.]);
        let input_id = g.add_value(Some("input"), None, None);

        let results = g
            .run(
                vec![(input_id, input.view().into())],
                &[input_id],
                None,
                None,
            )
            .unwrap();

        expect_equal(&results[0].as_tensor_view().unwrap(), &input.view())?;

        Ok(())
    }

    #[test]
    fn test_constant_graph() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::new();

        let value = Tensor::from([1., 2., 3., 4., 5.]);
        let const_id = g.add_constant(Some("weight"), value.clone());

        let results = g.run(vec![], &[const_id], None, None).unwrap();

        expect_equal(&results[0].as_tensor_view().unwrap(), &value.view())?;

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

        let mut subgraph = Graph::new();
        subgraph.add_constant(Some("sg_floats"), Tensor::<f32>::zeros(&[10, 10]));
        g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

        assert_eq!(g.total_params(), 300);
    }

    #[test]
    fn test_no_outputs() {
        let g = Graph::new();
        let results = g.run(vec![], &[], None, None).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_duplicate_inputs() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);
        let input = Tensor::from([1.]);
        let result = g.run(
            vec![
                (input_id, input.view().into()),
                (input_id, input.view().into()),
            ],
            &[input_id],
            None,
            None,
        );
        assert_eq!(
            result,
            Err(RunError::PlanningError(
                "Inputs are not unique. Input \"input\" is duplicated.".into()
            ))
        );
    }

    #[test]
    fn test_duplicate_outputs() {
        let mut g = Graph::new();

        let input_id = g.add_value(Some("input"), None, None);
        let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);

        let input = Tensor::from([1.]);

        let result = g.run(
            vec![(input_id, input.into())],
            &[op_a_out, op_a_out],
            None,
            None,
        );

        assert_eq!(
            result,
            Err(RunError::PlanningError(
                "Outputs are not unique. Output \"op_a_out\" is duplicated.".into()
            ))
        );
    }

    #[test]
    fn test_no_source_for_output() {
        let mut g = Graph::new();
        let output_id = g.add_value(Some("output"), None, None);
        let err = g.run(vec![], &[output_id], None, None);
        assert_eq!(
            err,
            Err(RunError::PlanningError(
                "Source node not found for output \"output\"".into()
            ))
        );
    }

    #[test]
    fn test_invalid_input_id() {
        let mut g = Graph::new();

        let (op_id, op_out) = g.add_simple_op("op", AddOne {}, &[]);
        let input = Tensor::from([1.]);
        let invalid_id = NodeId::from_u32(1234);

        for wrong_input_id in [op_id, invalid_id] {
            let result = g.run(
                [(wrong_input_id, input.view().into())].into(),
                &[op_out],
                None,
                None,
            );
            let name = g.node_name(wrong_input_id);
            assert_eq!(
                result,
                Err(RunError::PlanningError(format!(
                    "Input 0 (\"{}\") is not a value node in the graph.",
                    name
                ),))
            );
        }
    }

    #[test]
    fn test_invalid_output_id() {
        let mut g = Graph::new();

        let input_id = g.add_value(None, None, None);
        let (op_id, _op_out) = g.add_simple_op("op", AddOne {}, &[input_id]);
        let input = Tensor::from([1.]);
        let invalid_id = NodeId::from_u32(1234);

        for wrong_output_id in [op_id, invalid_id] {
            let result = g.run(
                [(input_id, input.view().into())].into(),
                &[wrong_output_id],
                None,
                None,
            );
            let name = g.node_name(wrong_output_id);
            assert_eq!(
                result,
                Err(RunError::PlanningError(format!(
                    "Output 0 (\"{}\") is not a value node in the graph.",
                    name
                )))
            );
        }
    }

    #[test]
    fn test_call_op_with_missing_input() {
        let mut g = Graph::new();

        // Call an operator with an input omitted by setting it to `None`,
        // as opposed to passing a shorter input list. This enables omitting
        // an input but still providing subsequent ones.
        let output = g.add_value(None, None, None);
        g.add_op(
            Some("shape"),
            Box::new(Shape::default()),
            &[None],
            &[Some(output)],
        );

        let results = g.run(vec![], &[output], None, None);

        assert_eq!(
            results.err(),
            Some(RunError::OperatorError {
                name: "shape".to_string(),
                error: OpError::MissingInputs,
                inputs: Some([None].into()),
            })
        );
    }

    #[test]
    fn test_err_if_missing_operator_input() {
        let mut g = Graph::new();
        let (_, output) = g.add_simple_op("op", Relu {}, &[NodeId::from_u32(42)]);
        let result = g.run(vec![], &[output], None, None);
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

        fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
            // An operator should normally have the same behavior in `run`
            // and `run_in_place`. Here we use different behavior to make it
            // possible to distinguish which path was used.
            let input: TensorView<f32> = ctx.inputs().require_as(0)?;
            input.to_tensor().into_op_result()
        }

        fn run_in_place(&self, input: Output, _ctx: &OpRunContext) -> Result<Output, OpError> {
            let mut output = input.into_tensor::<f32>().unwrap();
            for x in output.iter_mut() {
                *x = *x + 1.0;
            }
            Ok(output.into())
        }
    }

    #[test]
    fn test_runs_op_in_place() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);

        let (_, op1_out) = g.add_simple_op("op1", AddOneInPlace {}, &[input_id]);
        let (_, op2_out) = g.add_simple_op("op2", AddOneInPlace {}, &[op1_out]);
        let (_, op3_out) = g.add_simple_op("op3", AddOneInPlace {}, &[op2_out]);
        let (_, op4_out) = g.add_simple_op("op4", AddOneInPlace {}, &[op2_out]);
        let input = Tensor::<f32>::zeros(&[1, 1]);

        // First operator should not be run in-place, since it has an
        // immutable input. The result should be the same as the input.
        let results = g
            .run(
                vec![(input_id, input.view().into())],
                &[op1_out],
                None,
                None,
            )
            .unwrap();
        assert_eq!(results[0].as_tensor_view::<f32>().unwrap()[[0, 0]], 0.0);

        // Second operator should be run in-place, as it meets all the
        // requirements for this optimization.
        let results = g
            .run(
                vec![(input_id, input.view().into())],
                &[op2_out],
                None,
                None,
            )
            .unwrap();
        assert_eq!(results[0].as_tensor_view::<f32>().unwrap()[[0, 0]], 1.0);

        // Third op should not be run in place, because its input is re-used
        // for fourth op. Fourth op can run in place as by then, it is the
        // only consumer of its input.
        let results = g
            .run(
                vec![(input_id, input.view().into())],
                &[op3_out, op4_out],
                None,
                None,
            )
            .unwrap();
        assert_eq!(results[0].as_tensor_view::<f32>().unwrap()[[0, 0]], 1.0);
        assert_eq!(results[1].as_tensor_view::<f32>().unwrap()[[0, 0]], 2.0);
    }

    // Test that the graph executor will swap inputs to commutative ops if
    // necessary to enable running in-place.
    #[test]
    fn test_runs_commutative_op_in_place() {
        use crate::ops::Add; // A commutative operator

        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);
        let bias_id = g.add_value(Some("bias"), None, None);

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
                None,
            )
            .unwrap();

        // Bias value should be added twice to every input.
        assert_eq!(
            results[0]
                .as_tensor_view::<f32>()
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

        fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
            {
                let mut rc = self.run_count.lock().unwrap();
                *rc += 1;
            }

            let input: TensorView<f32> = ctx.inputs().require_as(0)?;
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
        let input_id = g.add_value(Some("input"), None, None);
        let left_split_out = g.add_value(Some("left_split"), None, None);
        let right_split_out = g.add_value(Some("right_split"), None, None);

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
                None,
            )
            .unwrap();

        assert_eq!(*run_count.lock().unwrap(), 1);

        assert_eq!(results.len(), 2);
        let left_split = results.remove(0).into_tensor::<f32>().unwrap();
        let right_split = results.remove(0).into_tensor::<f32>().unwrap();
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
        let val_0 = g.add_value(Some("i0"), None, None);
        let const_1 = g.add_constant(Some("c1"), Tensor::from(4.));
        let val_1 = g.add_value(Some("i1"), None, None);

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

        fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
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
        let input_ids = &[3, 1, 2].map(NodeId::from_u32);
        let output_ids = &[6, 4, 5].map(NodeId::from_u32);
        let op_ids = &[10, 11, 12].map(NodeId::from_u32);

        let plan = CachedPlan::new(input_ids, output_ids, op_ids.to_vec());

        assert!(plan.matches(input_ids, output_ids));

        // Same input and output IDs, different orders.
        assert!(plan.matches(
            &[1, 2, 3].map(NodeId::from_u32),
            &[4, 5, 6].map(NodeId::from_u32)
        ));
        assert!(plan.matches(
            &[3, 2, 1].map(NodeId::from_u32),
            &[6, 5, 4].map(NodeId::from_u32)
        ));

        // Different input and output IDs
        assert!(!plan.matches(&[20, 21, 22].map(NodeId::from_u32), output_ids));
        assert!(!plan.matches(input_ids, &[20, 21, 22].map(NodeId::from_u32)));
    }

    /// A trivial control flow operator which just forwards inputs to a subgraph
    /// and returns its outputs.
    struct Subgraph {
        graph: Graph,
    }

    impl std::fmt::Debug for Subgraph {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(f, "Subgraph {{ ... }}")
        }
    }

    impl Operator for Subgraph {
        fn name(&self) -> &str {
            "Subgraph"
        }

        fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
            Err(OpError::InvalidValue(
                "operator must be run with `run_subgraph`",
            ))
        }

        fn subgraphs(&self) -> SmallVec<[&Graph; 2]> {
            SmallVec::from_slice(&[&self.graph])
        }

        fn run_subgraph<'a>(
            &'a self,
            ctx: &OpRunContext,
            captures: CaptureEnv,
            weight_caches: Option<&[WeightCache]>,
            profiler: Option<&mut Profiler<'a>>,
            options: Option<RunOptions>,
        ) -> Result<OutputList, RunError> {
            let inputs = self
                .graph
                .input_ids()
                .iter()
                .copied()
                .zip(ctx.inputs().iter().flatten().map(|i| i.into()))
                .collect();
            self.graph
                .run_subgraph(
                    inputs,
                    self.graph.output_ids(),
                    captures,
                    Some(ctx.pool()),
                    weight_caches.map(|wcs| &wcs[0]),
                    profiler,
                    options,
                )
                .map(|xs| xs.into_iter().collect())
        }
    }

    #[test]
    fn test_subgraph() {
        let mut g = Graph::new();
        let input = g.add_value(Some("input"), None, None);

        // Add subgraphs for `If` operation. These capture `input`.
        let mut then_branch = Graph::new();
        let tb_input = then_branch.add_value(Some("input"), None, None);
        let two = then_branch.add_constant(None, Tensor::from(2.));
        let (_, tb_output) = then_branch.add_simple_op("Mul", Mul {}, &[tb_input, two]);
        then_branch.set_captures(&[tb_input]);
        then_branch.set_output_ids(&[tb_output]);

        let mut else_branch = Graph::new();
        let eb_input = else_branch.add_value(Some("input"), None, None);
        let three = else_branch.add_constant(None, Tensor::from(3.));
        let (_, eb_output) = else_branch.add_simple_op("Mul", Mul {}, &[eb_input, three]);
        else_branch.set_captures(&[eb_input]);
        else_branch.set_output_ids(&[eb_output]);

        // Add `If` operator that runs one of two subgraphs.
        let cond = g.add_value(Some("cond"), None, None);
        let branch = If {
            then_branch,
            else_branch,
        };
        let (_, if_out) = g.add_simple_op("If", branch, &[cond]);

        // Evaluate `then` branch
        let mut result = g
            .run(
                vec![
                    (input, Tensor::from(2.).into()),
                    (cond, Tensor::from(1).into()),
                ],
                &[if_out],
                None,
                None,
            )
            .unwrap();
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result, Tensor::from(4.));

        // Evaluate `else` branch
        let mut result = g
            .run(
                vec![
                    (input, Tensor::from(2.).into()),
                    (cond, Tensor::from(0).into()),
                ],
                &[if_out],
                None,
                None,
            )
            .unwrap();
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result, Tensor::from(6.));
    }

    #[test]
    fn test_nested_subgraph() {
        let mut g = Graph::new();
        let input = g.add_value(Some("input"), None, None);

        let mut subgraph = Graph::new();

        let mut nested_subgraph = Graph::new();
        let ns_input = nested_subgraph.add_value(Some("input"), None, None);
        nested_subgraph.set_captures(&[ns_input]);
        nested_subgraph.set_output_ids(&[ns_input]);

        let (_, ns_out) = subgraph.add_simple_op(
            "Subgraph",
            Subgraph {
                graph: nested_subgraph,
            },
            &[],
        );
        subgraph.set_output_ids(&[ns_out]);

        let (_, sg_out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

        let mut result = g
            .run(
                vec![(input, Tensor::from(2.).into())],
                &[sg_out],
                None,
                None,
            )
            .unwrap();
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result, Tensor::from(2.));
    }

    #[test]
    fn test_captures_not_available_when_subgraph_is_run_directly() {
        let mut subgraph = Graph::new();
        let sg_input = subgraph.add_value(Some("input"), None, None);
        subgraph.set_captures(&[sg_input]);
        let (_, sg_add) = subgraph.add_simple_op("Id", Identity {}, &[sg_input]);
        subgraph.set_output_ids(&[sg_add]);

        // When a subgraph is run via `run_subgraph` the planner will assume
        // that captured values are available. If the graph is run directly
        // however, this is not the case.
        //
        // Cases where subgraphs are run directly include the constant
        // propagation pass of graph optimization.

        let result = subgraph.partial_run(Vec::new(), &[sg_add], None).unwrap();
        assert_eq!(result.len(), 0);

        let result = subgraph.run(Vec::new(), &[sg_add], None, None);
        assert_eq!(
            result,
            Err(RunError::PlanningError(
                "Missing input \"input\" for op \"Id\"".to_string()
            ))
        );
    }

    #[test]
    fn test_partial_run_considers_subgraph_captures() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);

        let mut subgraph = Graph::new();
        let sg_input = subgraph.add_value(Some("input"), None, None);
        subgraph.set_captures(&[sg_input]);
        let (_, sg_add) = subgraph.add_simple_op("Id", Identity {}, &[sg_input]);
        subgraph.set_output_ids(&[sg_add]);

        let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

        // `partial_run` should skip operators that can't be evaluated due to
        // missing captures.
        let result = g.partial_run(Vec::new(), &[out], None).unwrap();
        assert_eq!(result.len(), 0);

        // When the captures are available, `partial_run` should evaluate the
        // operator as normal.
        let result = g
            .partial_run([(input_id, Tensor::from(4.).into())].into(), &[out], None)
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_plan_considers_capture_dependencies() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);

        let (_, _) = g.add_simple_op("Add", Add {}, &[input_id, input_id]);

        // Add a subgraph with a captured value that is the output of an
        // operation in the parent graph.
        let mut subgraph = Graph::new();
        let sg_input = subgraph.add_value(Some("Add_out"), None, None);
        subgraph.set_captures(&[sg_input]);
        let (_, sg_out) = subgraph.add_simple_op("Id", Identity {}, &[sg_input]);
        subgraph.set_output_ids(&[sg_out]);

        let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

        // Run the graph. The planner must account for captured dependencies
        // in the `Subgraph` op.
        let input = Tensor::from(3.);
        let mut result = g
            .run(vec![(input_id, input.into())], &[out], None, None)
            .unwrap();
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result.item(), Some(&6.));
    }

    #[test]
    fn test_plan_considers_transitive_capture_dependencies() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);

        let (_, _) = g.add_simple_op("Add", Add {}, &[input_id, input_id]);

        // Add nested subgraphs where an operation in the innermost graph has
        // a dependency on an operator output in the top-level graph.
        let mut subgraph = Graph::new();
        let mut nested_subgraph = Graph::new();
        let ns_input = nested_subgraph.add_value(Some("Add_out"), None, None);
        nested_subgraph.set_captures(&[ns_input]);
        let (_, ns_out) = nested_subgraph.add_simple_op("Id", Identity {}, &[ns_input]);
        nested_subgraph.set_output_ids(&[ns_out]);

        let (_, sg_out) = subgraph.add_simple_op(
            "Subgraph",
            Subgraph {
                graph: nested_subgraph,
            },
            &[],
        );
        subgraph.set_output_ids(&[sg_out]);

        let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

        // Run the graph. The planner must account for captured dependencies
        // from the innermost graph in the `Subgraph` op.
        let input = Tensor::from(3.);
        let mut result = g
            .run(vec![(input_id, input.into())], &[out], None, None)
            .unwrap();
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result.item(), Some(&6.));
    }

    #[test]
    fn test_keeps_temp_value_needed_as_subgraph_capture() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);

        // Compute a temporary `id_out` value and use it in the main graph.
        let (_, id_out) = g.add_simple_op("Id", Identity {}, &[input_id]);
        let (_, mul_out) = g.add_simple_op("Mul", Mul {}, &[id_out, id_out]);

        // Add a subgraph which depends on the temporary `id_out` value via a
        // capture. Graph execution must keep the `id_out` value around until
        // this has run, even though no ops in the main graph need it as inputs.
        let mut subgraph = Graph::new();
        let sg_input = subgraph.add_value(Some("Id_out"), None, None);
        subgraph.set_captures(&[sg_input]);
        let (_, sg_out) = subgraph.add_simple_op("Id", Identity {}, &[sg_input]);
        subgraph.set_output_ids(&[sg_out]);

        // Add op to main graph which runs the subgraph. This has a dummy
        // dependency on `mul_out`.
        let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[mul_out]);

        let input = Tensor::from(3.);
        let mut result = g
            .run(vec![(input_id, input.into())], &[out], None, None)
            .unwrap();
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result.item(), Some(&3.));
    }

    #[test]
    fn test_captures_by_value_if_possible() {
        // Set up a graph that runs a subgraph and passes captures by value,
        // if the value is passed to the graph as an owned value.
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);

        let mut subgraph = Graph::new();
        let sg_input = subgraph.add_value(Some("input"), None, None);
        subgraph.set_captures(&[sg_input]);

        let id_op = TrackUsage::new(Identity {});
        let id_op_metrics = id_op.metrics();
        let (_, id_out) = subgraph.add_simple_op("Id", id_op, &[sg_input]);
        subgraph.set_output_ids(&[id_out]);
        let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

        // Run graph with an owned value as input.
        let input = Tensor::from(42.);
        let mut result = g
            .run(vec![(input_id, input.into())], &[out], None, None)
            .unwrap();

        // Check result and that Identity operation was run in-place.
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result.item(), Some(&42.));

        {
            let id_op_metrics = id_op_metrics.lock().unwrap();
            assert_eq!(id_op_metrics.run_count, 0);
            assert_eq!(id_op_metrics.run_in_place_count, 1);
        }

        // Run graph with view as input.
        let input = Tensor::from(42.);
        let mut result = g
            .run(vec![(input_id, input.view().into())], &[out], None, None)
            .unwrap();

        // Check result and that Identity operation was not run in-place.
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result.item(), Some(&42.));

        {
            let id_op_metrics = id_op_metrics.lock().unwrap();
            assert_eq!(id_op_metrics.run_count, 1);
            assert_eq!(id_op_metrics.run_in_place_count, 1);
        }
    }

    // MatMul wrapper that verifies its B input (ie. the weights) are prepacked.
    #[derive(Debug)]
    struct MatMulExpectPacked {
        inner: MatMul,
    }

    impl MatMulExpectPacked {
        fn new() -> Self {
            MatMulExpectPacked { inner: MatMul {} }
        }
    }

    impl Operator for MatMulExpectPacked {
        fn name(&self) -> &str {
            "MatMulExpectPacked"
        }

        fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
            [1].into()
        }

        fn prepack(&self, index: usize, input: Input) -> Option<PrepackedInput> {
            self.inner.prepack(index, input)
        }

        fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
            let prepacked = ctx.inputs().get_prepacked(1);
            assert!(prepacked.is_some());
            self.inner.run(ctx)
        }
    }

    #[test]
    fn test_prepack_weights() {
        // Create a graph and a subgraph, both with operators that can
        // use prepacked weights.
        let mut graph = Graph::new();
        let mut cache = WeightCache::new();

        let input = graph.add_value(Some("input"), None, None);
        let weights = graph.add_constant(None, Tensor::<f32>::zeros(&[10, 7]));
        let (_, matmul_out) =
            graph.add_simple_op("MatMul", MatMulExpectPacked::new(), &[input, weights]);

        let mut subgraph = Graph::new();
        let sg_input = subgraph.add_value(Some("sg-input"), None, None);
        let sg_weights = subgraph.add_constant(None, Tensor::<f32>::zeros(&[7, 5]));
        let (_, sg_matmul_out) = subgraph.add_simple_op(
            "sg-MatMul",
            MatMulExpectPacked::new(),
            &[sg_input, sg_weights],
        );
        subgraph.set_input_ids(&[sg_input]);
        subgraph.set_output_ids(&[sg_matmul_out]);

        let (subgraph_op, subgraph_out) =
            graph.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[matmul_out]);
        graph.set_input_ids(&[input]);
        graph.set_output_ids(&[subgraph_out]);

        // Prepack weights and verify that the cache was populated.
        graph.prepack_weights(&mut cache);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(weights).is_some());

        let sg_cache = cache
            .get_subgraph_caches(subgraph_op)
            .map(|caches| &caches[0])
            .unwrap();
        assert!(sg_cache.get(sg_weights).is_some());

        // Run the graph, passing the cache. The MatMul wrapper will verify
        // that the B / RHS inputs were passed from the cache.
        let input_value = Tensor::<f32>::zeros(&[3, 10]);
        graph
            .run(
                [(input, input_value.into())].into(),
                &[subgraph_out],
                Some(&cache),
                None,
            )
            .unwrap();
    }

    #[test]
    fn test_run_context_num_outputs() {
        let mut g = Graph::new();
        let input_id = g.add_value(Some("input"), None, None);
        let (_, op_out) = g.add_simple_op(
            "test_op",
            RunFn::new(|ctx| {
                assert_eq!(ctx.num_outputs(), Some(1));
                let output: Output = Tensor::from_scalar(0.).into();
                Ok([output].into())
            }),
            &[input_id],
        );
        let input = Tensor::from([1, 2, 3]);
        g.run(vec![(input_id, input.into())], &[op_out], None, None)
            .unwrap();
    }

    #[test]
    fn test_remove_nodes() {
        let mut g = Graph::new();
        let val_id = g.add_value(Some("value"), None, None);
        g.set_input_ids(&[val_id]);
        g.set_output_ids(&[val_id]);

        assert!(g.get_node(val_id).is_some());
        assert!(g.get_node_id("value").is_some());

        g.remove_nodes(&[val_id]);

        assert!(g.get_node(val_id).is_none());
        assert!(g.get_node_id("value").is_none());
        assert!(g.input_ids().is_empty());
        assert!(g.output_ids().is_empty());

        // Removing an operator should remove it as the source node for its outputs.
        let val_id = g.add_value(Some("value2"), None, None);
        let (op_id, out_id) = g.add_simple_op("Mul", Mul {}, &[val_id, val_id]);
        assert!(g.get_source_node(out_id).is_some());

        g.remove_nodes(&[op_id]);

        assert!(g.get_source_node(out_id).is_none());
    }
}
