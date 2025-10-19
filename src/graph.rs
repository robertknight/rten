use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rayon::prelude::*;
use rten_tensor::prelude::*;

// The std HashMap/HashSet provide DOS resistance. In this module hash keys are
// mostly `NodeId`s which we allocate ourselves, so this is not a concern.
// Instead we want faster hashing.
use rustc_hash::{FxHashMap, FxHashSet};

use smallvec::SmallVec;

use crate::buffer_pool::BufferPool;
use crate::env::env_flag;
use crate::ops::{InputList, OpRunContext, Operator, OutputList, PrepackedInput};
use crate::threading;
use crate::timing::{Instant, ProfileFormat, Profiler, TimingFilter, TimingRecord, TimingSort};
use crate::value::{DataType, Value, ValueMeta, ValueOrView, ValueView};
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
use planner::{CachedPlan, Planner};
mod run_error;
pub(crate) use run_error::RunErrorImpl;
pub use run_error::{RunError, RunErrorKind};

pub use planner::PlanOptions;

mod node_id;
pub use node_id::NodeId;

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
///
/// ```
/// use rten::RunOptions;
///
/// let opts = RunOptions::default()
///     .with_timing(true)
///     .with_verbose(false)
///     .with_thread_pool(None);
/// ```
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct RunOptions {
    /// Whether to log times spent in different operators when run completes.
    pub timing: bool,

    /// Filter which operator nodes are included in the timing report.
    ///
    /// Nodes matching any of the filters are included.
    pub timing_filter: Vec<TimingFilter>,

    /// Whether to include a breakdown of execution time by input shape, in
    /// timing reports.
    pub timing_by_shape: bool,

    /// Order in which timings should be sorted. Defaults to sorting in
    /// descending order by time.
    pub timing_sort: TimingSort,

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

    pub fn with_timing(mut self, timing: bool) -> Self {
        self.timing = timing;
        self
    }

    pub fn with_timing_filter(mut self, filter: Vec<TimingFilter>) -> Self {
        self.timing_filter = filter;
        self
    }

    pub fn with_timing_by_shape(mut self, timing_by_shape: bool) -> Self {
        self.timing_by_shape = timing_by_shape;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_thread_pool(mut self, pool: Option<Arc<threading::ThreadPool>>) -> Self {
        self.thread_pool = pool;
        self
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
            && self.timing_filter == other.timing_filter
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

    /// Map of value node to operator nodes that consume the value as an input.
    consumer_ids: FxHashMap<NodeId, SmallVec<[NodeId; 1]>>,

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
            consumer_ids: FxHashMap::default(),
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
        if node_ids.is_empty() {
            return;
        }

        self.clear_cached_plan();

        // Use a set for faster lookup in case we are removing many nodes.
        let node_ids: FxHashSet<NodeId> = node_ids.iter().copied().collect();

        // Remove nodes from graph inputs and outputs.
        self.input_ids.retain(|id| !node_ids.contains(id));
        self.output_ids.retain(|id| !node_ids.contains(id));
        self.captures.retain(|id| !node_ids.contains(id));

        // Remove nodes from output value -> source operator edges.
        self.source_ids
            .retain(|val_id, op_id| !node_ids.contains(val_id) && !node_ids.contains(op_id));

        // Remove nodes from input value -> consumer operator edges.
        for consumer_ops in self.consumer_ids.values_mut() {
            consumer_ops.retain(|op_id| !node_ids.contains(op_id));
        }

        // Finally remove nodes from the graph.
        for node_id in node_ids {
            self.consumer_ids.remove(&node_id);
            if let Some(name) = self.nodes.get(&node_id).and_then(|n| n.name()) {
                self.node_id_from_name.remove(name);
            }
            self.nodes.remove(&node_id);
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
            if let Node::Operator(op) = node
                && let Some(subgraph_op) = op.operator().as_subgraph_op()
            {
                for subgraph in subgraph_op.subgraphs() {
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
        op: Arc<dyn Operator + Send + Sync>,
        inputs: &[Option<NodeId>],
        outputs: &[Option<NodeId>],
    ) -> NodeId {
        let op_node = Node::Operator(OperatorNode::new(name, inputs, outputs, op));
        let op_id = self.add_node(op_node);

        for output_id in outputs.iter().flatten() {
            self.source_ids.insert(*output_id, op_id);
        }
        let consumer_entry = SmallVec::from([op_id]);
        for input_id in inputs.iter().flatten() {
            self.consumer_ids
                .entry(*input_id)
                .and_modify(|vec| vec.push(op_id))
                .or_insert(consumer_entry.clone());
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
        let op_node_id = self.add_op(Some(name), Arc::new(op), &input_ids, &[op_out_id].map(Some));
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
        enum Entry {
            Cache((NodeId, PrepackedInput)),
            SubgraphCache((NodeId, Vec<WeightCache>)),
        }

        // Traverse operators and prepack in parallel.
        let entries: Vec<Entry> = threading::thread_pool().run(|| {
            self.nodes
                .par_iter()
                .filter_map(|(node_id, node)| match node {
                    Node::Operator(op) => Some((*node_id, op)),
                    _ => None,
                })
                .flat_map(|(op_node_id, op_node)| {
                    let mut entries = Vec::new();

                    for input_index in op_node.operator().prepack_inputs() {
                        let Some(input_id) =
                            op_node.input_ids().get(input_index).copied().flatten()
                        else {
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
                            .prepack(input_index, const_node.as_view())
                        else {
                            // Operator doesn't support or decided not to prepack this value.
                            continue;
                        };

                        entries.push(Entry::Cache((input_id, packed)));
                    }

                    let mut subgraph_caches = Vec::new();

                    if let Some(sg_op) = op_node.operator().as_subgraph_op() {
                        subgraph_caches.extend(sg_op.subgraphs().into_iter().map(|subgraph| {
                            let mut subgraph_cache = WeightCache::new();
                            subgraph.prepack_weights(&mut subgraph_cache);
                            subgraph_cache
                        }));
                    }

                    if !subgraph_caches.is_empty() {
                        entries.push(Entry::SubgraphCache((op_node_id, subgraph_caches)));
                    }

                    entries
                })
                .collect()
        });

        // Move the entries into the output cache in serial.
        for entry in entries {
            match entry {
                Entry::Cache((input_id, packed)) => cache.insert(input_id, packed),
                Entry::SubgraphCache((op_id, subgraph_cache)) => {
                    cache.insert_subgraph_caches(op_id, subgraph_cache)
                }
            }
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

    /// Look up the operator nodes which consume a given value or constant node.
    pub fn get_consumers(&self, id: NodeId) -> Option<&[NodeId]> {
        self.consumer_ids.get(&id).map(|v| &**v)
    }

    /// Retrieve a node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Replace an operator input with a different value or constant.
    pub fn replace_input(&mut self, op_id: NodeId, old_input_id: NodeId, new_input_id: NodeId) {
        let Some(Node::Operator(op_node)) = self.get_node_mut(op_id) else {
            panic!("operator node not found");
        };
        op_node.replace_input(old_input_id, new_input_id);

        // Remove operator as consumer of old input ID.
        if let Some(ops) = self.consumer_ids.get_mut(&old_input_id) {
            ops.retain(|op| *op != op_id);
        }

        // Add operator as consumer of new input ID.
        self.consumer_ids
            .entry(new_input_id)
            .and_modify(|ops| ops.push(op_id))
            .or_insert([op_id].into());
    }

    /// Return the total number of parameters in all constant nodes in this
    /// graph and subgraphs.
    pub fn total_params(&self) -> usize {
        self.nodes
            .values()
            .map(|node| match node {
                Node::Operator(op_node) => op_node
                    .operator()
                    .as_subgraph_op()
                    .map(|sg| sg.subgraphs().iter().map(|sg| sg.total_params()).sum())
                    .unwrap_or(0),
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
        opts: PlanOptions,
    ) -> Result<Vec<NodeId>, RunError> {
        self.create_plan(inputs, outputs, opts)
    }

    /// Compute a set of output values given a set of inputs, using the
    /// processing steps and constant values defined by the graph.
    pub fn run(
        &self,
        inputs: Vec<(NodeId, ValueOrView)>,
        outputs: &[NodeId],
        weight_cache: Option<&WeightCache>,
        opts: Option<RunOptions>,
    ) -> Result<Vec<Value>, RunError> {
        let input_ids: Vec<_> = inputs.iter().map(|(node_id, _)| *node_id).collect();
        let plan = self.get_cached_plan(&input_ids, outputs, false /* is_subgraph */)?;
        let opts = opts.unwrap_or_default();
        opts.thread_pool().run(|| {
            let mut profiler =
                (opts.timing || opts.verbose).then(|| Profiler::with_capacity(plan.plan().len()));

            let pool = BufferPool::new();

            let result = self.run_plan(
                inputs,
                plan.plan(),
                outputs,
                None, /* captures */
                &pool,
                weight_cache,
                profiler.as_mut(),
                &opts,
            );

            if let Some(profiler) = &profiler {
                let print_opts = ProfileFormat {
                    sort: opts.timing_sort.clone(),
                    filter: opts.timing_filter.clone(),
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
        inputs: Vec<(NodeId, ValueOrView)>,
        outputs: &[NodeId],
        captures: CaptureEnv,
        pool: &BufferPool,
        weight_cache: Option<&WeightCache>,
        profiler: Option<&mut Profiler<'a>>,
        opts: Option<RunOptions>,
    ) -> Result<Vec<Value>, RunError> {
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
                let plan = self.create_plan(
                    inputs,
                    outputs,
                    PlanOptions {
                        allow_missing_inputs: false,
                        captures_available: is_subgraph,
                    },
                )?;
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
        opts: PlanOptions,
    ) -> Result<Vec<NodeId>, RunError> {
        Planner::with_graph(self).create_plan(inputs, outputs, opts)
    }

    fn run_plan<'a>(
        &'a self,
        mut inputs: Vec<(NodeId, ValueOrView)>,
        plan: &[NodeId],
        outputs: &[NodeId],
        mut captures: Option<CaptureEnv>,
        pool: &BufferPool,
        weight_cache: Option<&WeightCache>,
        mut profiler: Option<&mut Profiler<'a>>,
        opts: &RunOptions,
    ) -> Result<Vec<Value>, RunError> {
        let mut temp_values: FxHashMap<NodeId, Value> = FxHashMap::default();

        // Extract all owned tensor inputs into the owned value map.
        //
        // This enables these inputs to be used for in-place operations or
        // returned directly as outputs.
        let mut idx = 0;
        while idx < inputs.len() {
            if matches!(inputs[idx], (_, ValueOrView::Value(_))) {
                let (node_id, ValueOrView::Value(outp)) = inputs.remove(idx) else {
                    unreachable!();
                };
                temp_values.insert(node_id, outp);
            } else {
                idx += 1;
            }
        }

        let inputs_by_id: FxHashMap<NodeId, ValueOrView> = inputs.iter().cloned().collect();
        let get_value_from_constant_or_input = |node_id: NodeId| -> Option<ValueView> {
            match self.nodes.get(&node_id) {
                Some(Node::Constant(constant)) => Some(constant.as_view()),
                Some(Node::Value(_)) => inputs_by_id.get(&node_id).map(|input| input.as_view()),
                _ => {
                    panic!("node {} is not a value or constant", node_id);
                }
            }
        };

        fn get_value_from_capture<'a>(
            nodes: &NoopHashMap<NodeId, Node>,
            captures: Option<&'a CaptureEnv>,
            node_id: NodeId,
        ) -> Option<ValueView<'a>> {
            let name = nodes.get(&node_id).and_then(|n| n.name())?;
            captures.and_then(|cap| cap.get_input(name))
        }

        // Count how often each temporary output is used, so we can free them
        // when no longer needed.
        let mut temp_value_refcount = NodeRefCount::with_capacity(self.next_node_id as usize);
        for &op_node_id in plan.iter() {
            let Some(Node::Operator(op_node)) = self.nodes.get(&op_node_id) else {
                return Err(
                    RunErrorImpl::PlanningError("operator node not found".to_string()).into(),
                );
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

        // Choose whether to use tensor pool. If disabled, buffers are still
        // allocated from the pool but never released to it, so allocations will
        // still come from the system allocator.
        let use_pool = env_flag("RTEN_USE_POOL", true);

        // Execute the plan
        let mut op_start = Instant::now();

        for (step, &op_node_id) in plan.iter().enumerate() {
            let Some(Node::Operator(op_node)) = self.nodes.get(&op_node_id) else {
                return Err(
                    RunErrorImpl::PlanningError("operator node not found".to_string()).into(),
                );
            };

            // Choose the input that we'll try to modify in-place to avoid
            // allocating a new buffer for the output. This will be passed as
            // the first input to `Operator::run_in_place`.
            //
            // For non-commutative ops we have to use the first input. For
            // commutative ops we can swap inputs around if that enables us to
            // run an op in place.
            let try_in_place_input_id = if op_node.operator().can_run_in_place() {
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

            // If the operator can run in place, try to get the owned tensor to
            // use as an output. This requires that the tensor is not a constant
            // (eg. weights) and is not going to be used by other ops in future.
            let in_place_input: Option<(NodeId, Value)> = if let Some(id) = try_in_place_input_id
                && let Some(value) = take_value(id)
            {
                Some((id, value))
            } else {
                None
            };

            // Extract values used by the operator's subgraphs which can be
            // passed by value.
            let subgraph_op = op_node.operator().as_subgraph_op();
            let by_value_captures = subgraph_op.is_some().then(|| {
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
            let mut op_inputs: SmallVec<[Option<ValueView>; 4]> =
                SmallVec::with_capacity(op_node.input_ids().len());
            for node_id in op_node.input_ids().iter() {
                if let Some(node_id) = node_id {
                    if let Some((id, _value)) = &in_place_input
                        && node_id == id
                    {
                        // This input is being passed separately as a mutable
                        // value.
                        continue;
                    }

                    if let Some(value) = get_value_from_constant_or_input(*node_id) {
                        op_inputs.push(Some(value));
                    } else if let Some(value) = temp_values.get(node_id) {
                        op_inputs.push(Some(value.as_view()));
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
                // Record the input value IDs and metadata together here because
                // inputs may be reordered if the operator is commutative.
                let mut meta: Vec<(Option<NodeId>, Option<ValueMeta>)> = Vec::new();
                if let Some((id, value)) = &in_place_input {
                    meta.push((Some(*id), Some(value.to_meta())));
                }
                for (id, input) in op_node
                    .input_ids()
                    .iter()
                    .copied()
                    .filter(|id| *id != in_place_input.as_ref().map(|(id, _value)| *id))
                    .zip(&op_inputs)
                {
                    meta.push((id, input.as_ref().map(|i| i.to_meta())))
                }
                meta
            } else {
                Vec::new()
            };

            // Run the operation.
            let get_prepacked = |input_index: usize| -> Option<&PrepackedInput> {
                op_node
                    .input_ids()
                    .get(input_index)
                    .copied()
                    .flatten()
                    .and_then(|node_id| weight_cache.and_then(|wc| wc.get(node_id)))
            };
            let inputs = InputList::from_optional(&op_inputs)
                .with_prepacked(&get_prepacked)
                .with_first_input_omitted(in_place_input.is_some());
            let mut ctx = OpRunContext::new(pool, &inputs);
            ctx.set_num_outputs(op_node.output_ids().len() as u32);

            let op_result = if let Some((_id, value)) = in_place_input {
                let input_dtype = value.dtype();
                let input_shape = value.shape();
                op_node
                    .operator()
                    .run_in_place(value, &ctx)
                    .map(|out| [out].into())
                    .map_err(|e| {
                        // The error here is currently missing information about operator inputs.
                        RunError::in_place_op_error(
                            op_node.name().unwrap_or_default(),
                            e,
                            &ctx,
                            input_dtype,
                            &input_shape,
                        )
                    })
            } else if let Some(subgraph_op) = subgraph_op {
                ctx.set_name(op_node.name());

                let capture_env = CaptureEnv::new(
                    captures.as_ref(),
                    self,
                    Some(&inputs_by_id),
                    Some(&temp_values),
                    by_value_captures,
                );
                subgraph_op
                    .run_subgraph(
                        &ctx,
                        capture_env,
                        weight_cache.and_then(|wc| wc.get_subgraph_caches(op_node_id)),
                        profiler.as_deref_mut(),
                        Some(opts.clone()),
                    )
                    .map_err(|err| RunError::subgraph_error(op_node.name(), err))
            } else {
                op_node
                    .operator()
                    .run(&ctx)
                    .map_err(|e| RunError::op_error(op_node.name().unwrap_or_default(), e, &ctx))
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
            let expected_num_outputs = op_node.output_ids().len();
            if expected_num_outputs > outputs.len() {
                return Err(RunErrorImpl::OutputMismatch {
                    name: op_node.name().unwrap_or_default().to_string(),
                    error: format!(
                        "operator returned {} outputs but expected {}",
                        outputs.len(),
                        expected_num_outputs,
                    ),
                }
                .into());
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
                if rc == Some(0)
                    && use_pool
                    && let Some(tensor) = temp_values.remove(&node_id)
                {
                    tensor.add_to_pool(pool)
                }
            }

            if let Some(profiler) = &mut profiler {
                let op_end = Instant::now();
                let op_duration = op_end - op_start;
                op_start = op_end;

                // Skip control flow ops to avoid double-counting the time from
                // ops inside the subgraph.
                if subgraph_op.is_none() {
                    profiler.add_record(TimingRecord {
                        name: op_node.operator().name(),
                        input_meta,
                        elapsed: op_duration,
                        node_name: op_node.name().unwrap_or(""),
                    });
                }
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
                    value.to_owned()
                } else if let Some(value) =
                    get_value_from_capture(&self.nodes, captures.as_ref(), *output_id)
                {
                    value.to_owned()
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
        input_meta: &[(Option<NodeId>, Option<ValueMeta>)],
    ) {
        println!(
            "#{} {} ({})",
            step,
            op_node.operator().name(),
            op_node.name().unwrap_or("")
        );
        for (index, (id, meta)) in input_meta.iter().enumerate() {
            if let Some(id) = id
                && let Some(meta) = meta
            {
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
        inputs: Vec<(NodeId, ValueOrView)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<(NodeId, Value)>, RunError> {
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

            let pool = BufferPool::new();
            let result = self.run_plan(
                inputs,
                &pruned_plan,
                &pruned_plan_output_ids,
                None, /* captures */
                &pool,
                None, /* weight cache */
                profiler.as_mut(),
                &opts,
            );

            if let Some(profiler) = &profiler {
                let print_opts = ProfileFormat {
                    sort: opts.timing_sort.clone(),
                    filter: opts.timing_filter.clone(),
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
    ) -> impl Iterator<Item = NodeId> + 'a {
        op_node
            .input_ids()
            .iter()
            .filter_map(|id| *id)
            .chain(op_node.capture_names().filter_map(move |cap_name| {
                let cap_id = self.get_node_id(cap_name)?;
                if !op_node.input_ids().contains(&Some(cap_id)) {
                    Some(cap_id)
                } else {
                    // If the captured node is also used as an input,
                    // only yield it once in the output.
                    None
                }
            }))
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
