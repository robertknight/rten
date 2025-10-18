use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::{Graph, Node, NodeId, OperatorNode, RunError, RunErrorImpl};

/// Options for creating a graph execution plan using [`Planner`].
#[derive(Clone)]
pub struct PlanOptions {
    /// Whether a plan can be successfully created if certain inputs are
    /// missing. If true, the planner will create the plan as if those inputs
    /// would be provided later.
    pub allow_missing_inputs: bool,

    /// Whether to treat a graph's captured values as available during planning.
    ///
    /// This should be true when generating a plan in the context of a normal
    /// run, but false if a plan is being generated that will run a subgraph
    /// on its own.
    pub captures_available: bool,
}

impl Default for PlanOptions {
    fn default() -> Self {
        PlanOptions {
            allow_missing_inputs: false,

            // Assume subgraphs will usually be run as part of their parent
            // graph, with captures available.
            //
            // An exception is when we're doing partial evaluation of the graph
            // as part of constant propagation.
            captures_available: true,
        }
    }
}

/// An execution plan specifying the sequence of operations to run from a graph
/// to derive a set of output values given a set of input values.
pub struct CachedPlan {
    /// Sorted list of value nodes that are provided at the start of execution.
    inputs: Vec<NodeId>,

    /// Sorted list of value nodes produced after the plan has executed.
    outputs: Vec<NodeId>,

    /// List of operator nodes to execute to produce `outputs` given `inputs`.
    plan: Vec<NodeId>,
}

impl CachedPlan {
    pub fn new(inputs: &[NodeId], outputs: &[NodeId], plan: Vec<NodeId>) -> CachedPlan {
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
    pub fn matches(&self, inputs: &[NodeId], outputs: &[NodeId]) -> bool {
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

    /// Return the IDs of the sequence of operators to run.
    pub fn plan(&self) -> &[NodeId] {
        &self.plan
    }
}

/// Return the first element in `xs` which is a duplicate of an earlier element.
fn first_duplicate_by<T, F: Fn(&T, &T) -> bool>(xs: &[T], eq: F) -> Option<&T> {
    for (i, x) in xs.iter().enumerate() {
        for y in &xs[i + 1..] {
            if eq(x, y) {
                return Some(y);
            }
        }
    }
    None
}

/// Planner creates execution plans for graph runs.
///
/// An execution plan is a sequence of operator nodes to evaluate in order to
/// produces values for a set of output nodes in the graph, given values for
/// a set of input nodes.
pub struct Planner<'a> {
    graph: &'a Graph,
}

impl<'a> Planner<'a> {
    /// Create an execution planner for a graph.
    pub fn with_graph(graph: &'a Graph) -> Self {
        Planner { graph }
    }

    /// Create an execution plan for a sequence of computation steps that begin
    /// with `inputs` and eventually produces `outputs`.
    ///
    /// The set of input and output node IDs must be unique.
    ///
    /// Any node IDs in `outputs` which reference constant or input values are
    /// omitted from the plan.
    pub fn create_plan(
        &self,
        inputs: &[NodeId],
        outputs: &[NodeId],
        options: PlanOptions,
    ) -> Result<Vec<NodeId>, RunError> {
        if let Some(dupe_id) = first_duplicate_by(outputs, |x, y| x == y) {
            let name = self.graph.node_name(*dupe_id);
            return Err(RunErrorImpl::PlanningError(format!(
                "Outputs are not unique. Output \"{}\" is duplicated.",
                name
            ))
            .into());
        }
        for (output_index, output_id) in outputs.iter().enumerate() {
            match self.graph.get_node(*output_id) {
                Some(Node::Value(_) | Node::Constant(_)) => {}
                _ => {
                    let name = self.graph.node_name(*output_id);
                    return Err(RunErrorImpl::PlanningError(format!(
                        "Output {} (\"{}\") is not a value node in the graph.",
                        output_index, name
                    ))
                    .into());
                }
            }
        }

        if let Some(dupe_id) = first_duplicate_by(inputs, |x, y| x == y) {
            let name = self.graph.node_name(*dupe_id);
            return Err(RunErrorImpl::PlanningError(format!(
                "Inputs are not unique. Input \"{}\" is duplicated.",
                name
            ))
            .into());
        }
        for (input_index, input_id) in inputs.iter().enumerate() {
            match self.graph.get_node(*input_id) {
                Some(Node::Value(_) | Node::Constant(_)) => {}
                _ => {
                    let name = self.graph.node_name(*input_id);
                    return Err(RunErrorImpl::PlanningError(format!(
                        "Input {} (\"{}\") is not a value node in the graph.",
                        input_index, name
                    ))
                    .into());
                }
            }
        }

        let builder = PlanBuilder::new(self.graph, inputs, options);
        builder.plan(outputs)
    }

    /// Prune a plan so that it contains only operators which can be executed
    /// given a subset of the inputs.
    ///
    /// `inputs` should be a subset of the inputs that were used to create
    /// `plan` originally.
    ///
    /// Returns a tuple of `(pruned_plan, new_outputs)` where `new_outputs`
    /// contains the IDs of leaf nodes in the pruned plan. These are the values
    /// that can still be generated by the reduced plan, and are either in
    /// the original `outputs` list or are inputs to parts of the plan that
    /// were pruned away.
    pub fn prune_plan(
        &self,
        plan: &[NodeId],
        inputs: &[NodeId],
        outputs: &[NodeId],
    ) -> (Vec<NodeId>, Vec<NodeId>) {
        let mut resolved_values = ResolvedValueSet::new(
            self.graph,
            inputs.iter().copied(),
            false, /* include_captures */
        );
        let mut pruned_plan = Vec::new();
        let mut candidate_outputs = inputs.to_vec();

        // IDs of input nodes for pruned operators that we can still generate
        // with the pruned plan.
        let mut pruned_ops_resolved_inputs = FxHashSet::<NodeId>::default();

        // Walk forwards through the plan and prune away steps that cannot be
        // computed due to missing inputs.
        for &node_id in plan {
            let Some(Node::Operator(op_node)) = self.graph.get_node(node_id) else {
                continue;
            };

            let all_inputs: SmallVec<[NodeId; 4]> =
                self.graph.operator_dependencies(op_node).collect();

            let all_inputs_available = all_inputs
                .iter()
                .all(|input_id| resolved_values.contains(*input_id));

            // Prune op if:
            //
            // - The output varies on each run (`Random*`)
            // - We are missing a required input
            let prune_op = !op_node.operator().is_deterministic() || !all_inputs_available;

            if prune_op {
                for input_id in all_inputs {
                    if resolved_values.contains(input_id) {
                        pruned_ops_resolved_inputs.insert(input_id);
                    }
                }
                continue;
            }
            resolved_values.extend(op_node.output_ids().iter().filter_map(|id_opt| *id_opt));
            pruned_plan.push(node_id);
            candidate_outputs.extend(op_node.output_ids().iter().filter_map(|id_opt| *id_opt));
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
}

/// Helper that builds an execution plan via a depth first traversal of the graph
/// starting at the output nodes.
struct PlanBuilder<'a> {
    graph: &'a Graph,
    resolved_values: ResolvedValueSet<'a>,
    plan: Vec<(NodeId, &'a OperatorNode)>,
    options: PlanOptions,
}

impl<'a> PlanBuilder<'a> {
    fn new(graph: &'a Graph, inputs: &[NodeId], options: PlanOptions) -> PlanBuilder<'a> {
        // Set of values that are available after executing the plan
        let resolved_values =
            ResolvedValueSet::new(graph, inputs.iter().copied(), options.captures_available);

        PlanBuilder {
            graph,
            resolved_values,
            plan: Vec::new(),
            options,
        }
    }

    /// Add all the transitive dependencies of `op_node` to the plan,
    /// followed by `op_node`.
    ///
    /// `active_set` tracks the IDs of operator nodes that are currently being
    /// visited.
    fn visit(
        &mut self,
        op_node_id: NodeId,
        op_node: &'a OperatorNode,
        active_set: &mut FxHashSet<NodeId>,
    ) -> Result<(), RunError> {
        active_set.insert(op_node_id);

        for input in self.graph.operator_dependencies(op_node) {
            if self.resolved_values.contains(input) {
                continue;
            }
            if let Some((input_op_id, input_op_node)) = self.graph.get_source_node(input) {
                if active_set.contains(&input_op_id) {
                    let msg = format!(
                        "Encountered cycle visiting dependency \"{}\" of operator \"{}\"",
                        self.graph.node_name(input),
                        self.graph.node_name(op_node_id)
                    );
                    return Err(RunErrorImpl::PlanningError(msg).into());
                }
                self.visit(input_op_id, input_op_node, active_set)?;
            } else if self.options.allow_missing_inputs {
                continue;
            } else {
                let msg = format!(
                    "Missing input \"{}\" for op \"{}\"",
                    self.graph.node_name(input),
                    self.graph.node_name(op_node_id)
                );
                return Err(RunErrorImpl::PlanningError(msg).into());
            }
        }
        for output_id in op_node.output_ids().iter().filter_map(|node| *node) {
            self.resolved_values.insert(output_id);
        }
        self.plan.push((op_node_id, op_node));

        active_set.remove(&op_node_id);

        Ok(())
    }

    /// Take the current execution plan and re-order it for more
    /// efficient execution.
    fn sort_plan(self, mut resolved_values: ResolvedValueSet) -> Vec<NodeId> {
        // Build map of value node to operators that depend on the value.
        let mut dependent_ops: FxHashMap<NodeId, Vec<(NodeId, &OperatorNode)>> =
            FxHashMap::default();
        for (op_node_id, op_node) in &self.plan {
            for input_id in self.graph.operator_dependencies(op_node) {
                if let Some(deps) = dependent_ops.get_mut(&input_id) {
                    deps.push((*op_node_id, op_node));
                } else {
                    dependent_ops.insert(input_id, [(*op_node_id, *op_node)].into());
                }
            }
        }

        let mut output_plan = Vec::with_capacity(self.plan.len());

        // Initialize frontier with all operators that can be executed
        // from initially-available values.
        let mut frontier: Vec<(NodeId, &OperatorNode)> = Vec::new();
        for (op_node_id, op_node) in &self.plan {
            if self
                .graph
                .operator_dependencies(op_node)
                .all(|id| resolved_values.contains(id))
            {
                frontier.push((*op_node_id, op_node));
            }
        }

        debug_assert!(!frontier.is_empty(), "initial frontier is empty");

        // Loop while we still have operators to compute.
        while !frontier.is_empty() {
            // Choose an operator to execute next and add it to the plan.
            //
            // We run non-in-place operators first, so that operators
            // which can run in-place are more likely to have their
            // inputs available for in-place execution.
            let op_pos = frontier
                .iter()
                .position(|(_id, op)| !op.operator().can_run_in_place())
                .unwrap_or(0);
            let (next_op_id, op_node) = frontier.remove(op_pos);
            output_plan.push(next_op_id);

            // Mark the operator's outputs as computed.
            resolved_values.extend(op_node.output_ids().iter().filter_map(|id| *id));

            // Visit operators that depend on current op outputs. Add
            // to frontier set if all dependencies have been resolved.
            for output_id in op_node.output_ids() {
                let Some(output_id) = output_id else {
                    continue;
                };
                let Some(deps) = dependent_ops.get(output_id) else {
                    continue;
                };
                for (candidate_op_id, candidate_op) in deps {
                    if frontier.iter().any(|(op_id, _)| op_id == candidate_op_id) {
                        continue;
                    }

                    if self
                        .graph
                        .operator_dependencies(candidate_op)
                        .all(|id| resolved_values.contains(id))
                    {
                        frontier.push((*candidate_op_id, candidate_op));
                    }
                }
            }
        }

        output_plan
    }

    /// Return a sequential plan to generate `outputs`.
    fn plan(mut self, outputs: &[NodeId]) -> Result<Vec<NodeId>, RunError> {
        let initial_resolved_values = self.resolved_values.clone();

        // Set of operator nodes currently being visited.
        let mut active_set = FxHashSet::default();

        // Build initial plan by traversing graph backwards from outputs.
        for output_id in outputs.iter() {
            if self.resolved_values.contains(*output_id) {
                // Value is either a constant node or is produced by
                // an operator that is already in the plan.
                continue;
            }

            if let Some((op_node_id, op_node)) = self.graph.get_source_node(*output_id) {
                self.visit(op_node_id, op_node, &mut active_set)?;
            } else {
                let output_name = self.graph.node_name(*output_id);
                let msg = format!("Source node not found for output \"{}\"", output_name);
                return Err(RunErrorImpl::PlanningError(msg).into());
            }
        }

        // When doing partial evaluation, just return the initial plan.
        // This avoids having to handle missing inputs when sorting the
        // plan.
        if self.options.allow_missing_inputs || self.plan.is_empty() {
            return Ok(self.plan.into_iter().map(|(op_id, _)| op_id).collect());
        }

        // Re-order initial plan to get a more efficient execution
        // order.
        let sorted_plan = self.sort_plan(initial_resolved_values);

        Ok(sorted_plan)
    }
}

/// Keeps track of which nodes' values are available at different points during
/// execution.
#[derive(Clone)]
struct ResolvedValueSet<'a> {
    graph: &'a Graph,

    // Resolved nodes, excluding constants.
    resolved: FxHashSet<NodeId>,
}

impl<'a> ResolvedValueSet<'a> {
    /// Create a new set where all nodes in `inputs` and constants are included.
    ///
    /// If `include_captures` are true then nodes which capture values from a
    /// parent graph are also available.
    fn new(graph: &'a Graph, inputs: impl Iterator<Item = NodeId>, include_captures: bool) -> Self {
        let mut resolved: FxHashSet<NodeId> = inputs.collect();
        if include_captures {
            resolved.extend(graph.captures().iter().copied());
        }
        Self { graph, resolved }
    }

    fn contains(&self, id: NodeId) -> bool {
        self.resolved.contains(&id) || matches!(self.graph.get_node(id), Some(Node::Constant(_)))
    }

    fn extend(&mut self, ids: impl Iterator<Item = NodeId>) {
        self.resolved.extend(ids)
    }

    fn insert(&mut self, id: NodeId) {
        self.resolved.insert(id);
    }
}
