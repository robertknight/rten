use std::any::Any;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::Value;
use crate::graph::{
    CaptureEnv, Constant, ConstantNode, ConstantNodeData, Graph, Node, NodeId, OperatorNode,
    PlanOptions, RunError,
};
use crate::ops::{Identity, Operator};

mod fusions;
mod pattern_matcher;

use fusions::{
    AddSoftmaxFusion, ApproxGeluFusion, CastElimination, Fusion, FusionVisitor, GeluFusion,
    IdentityFusion, LayerNormalizationFusion, MatMulAddFusion, MatMulIntegerToFloatFusion,
    MatMulScaleFusion, PatternFusion, ReciprocalFusion, ReduceMeanAxesFusion,
    RmsNormalizationFusion, ShapeSliceToConstant, SiluFusion, SwishFusion, TransposeFusion,
};

/// Errors that occur while applying graph optimizations.
#[derive(Debug)]
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

/// Holds a [`Graph`] and associated data structures while it is being mutated
/// by an optimizer, and provides operations to update the graph.
struct GraphMutator {
    graph: Graph,
    output_ids: Vec<NodeId>,
}

impl GraphMutator {
    fn from_graph(graph: Graph) -> GraphMutator {
        GraphMutator {
            output_ids: graph.output_ids().to_vec(),
            graph,
        }
    }

    /// Add a new constant value to the graph.
    fn add_constant<T>(
        &mut self,
        name: Option<&str>,
        value: impl Into<ConstantNodeData<T>>,
    ) -> NodeId
    where
        Constant: From<ConstantNode<T>>,
    {
        self.graph.add_constant(name, value)
    }

    /// Add a new constant value to the graph.
    fn add_constant_node(&mut self, const_node: Constant) -> NodeId {
        self.graph.add_constant_node(const_node)
    }

    /// Add a new operator to the graph.
    fn add_operator(
        &mut self,
        name: Option<&str>,
        op: Arc<dyn Operator + Send + Sync>,
        inputs: &[Option<NodeId>],
        outputs: &[Option<NodeId>],
    ) {
        self.graph.add_op(name, op, inputs, outputs);
    }

    /// Return a reference to the graph.
    ///
    /// Note there is no mutable variant of this method. All graph updates must
    /// be done via methods of this struct.
    fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Update the output IDs of the graph and return it.
    fn finalize_graph(mut self) -> Graph {
        self.graph.set_output_ids(&self.output_ids);
        self.graph
    }

    /// Iterate over each operator node in the graph and potentially apply a
    /// fusion which combines this node and adjacent nodes.
    ///
    /// Returns the number of applied fusions.
    fn apply_fusion<F: Fn(&Self, NodeId, &OperatorNode) -> Option<Fusion>>(
        &mut self,
        create_fusion: F,
    ) -> usize {
        struct Replacement {
            unfused_ops: Vec<NodeId>,
            fusion: Fusion,
        }

        let mut ops_pending_removal = FxHashSet::default();

        // Get all operators, in the order they will be run.
        let Ok(mut operators) = self.graph.execution_plan(
            self.graph.input_ids(),
            self.graph.output_ids(),
            PlanOptions::default(),
        ) else {
            return 0;
        };

        // Reverse the operator list so that we traverse from outputs towards
        // inputs. We do this because fusion matchers start from the output node
        // of the subgraph they match. Sometimes fusions have optional output
        // steps which mean they can match multiple nodes. For example the layer
        // normalization fusion can match with and without an `Add(normalized,
        // bias)` step at the end. In this case we want to visit the optional
        // operators first, if present, so that it gets included in the fusion.
        operators.reverse();

        let fusions: Vec<_> = operators
            .into_iter()
            .filter_map(|op_node_id| {
                let op_node = self
                    .graph
                    .get_node(op_node_id)
                    .and_then(|op| op.as_operator())
                    .unwrap();

                // Don't try and fuse operators that were removed when fusing a
                // node we visited earlier.
                if ops_pending_removal.contains(&op_node_id) {
                    return None;
                }

                let fusion = create_fusion(self, op_node_id, op_node)?;

                // Check for outputs of intermediate steps in the fused subgraph
                // used by operators outside of the subgraph. If any are found,
                // we can't fuse the subgraph as the intermediate value would no
                // longer be available.
                let mut input_ids = fusion.input_ids();

                // Execution planning disallows duplicate input IDs. An operator
                // however is allowed to use the same value for multiple inputs.
                input_ids.sort();
                input_ids.dedup();

                let output_ids = fusion.output_ids();
                let unfused_ops = self
                    .graph
                    .execution_plan(&input_ids, &output_ids, PlanOptions::default())
                    .unwrap();

                for unfused_op in &unfused_ops {
                    // Skip this fusion if it includes an operator which was
                    // fused when visiting an earlier operator.
                    if ops_pending_removal.contains(unfused_op) {
                        return None;
                    }
                    ops_pending_removal.insert(*unfused_op);
                }

                let reused_output = find_operator_output_used_outside_subgraph(
                    &self.graph,
                    &unfused_ops,
                    &output_ids,
                );
                if reused_output.is_some() {
                    return None;
                }

                Some(Replacement {
                    fusion,
                    unfused_ops,
                })
            })
            .collect();

        let n_fusions = fusions.len();

        // Remove all the nodes from the unfused subgraph.
        //
        // We do this before adding the new nodes to avoid node name conflicts.
        // Also the cost of `remove_nodes` is O(N) in the number of nodes in
        // the graph, so we want to remove nodes in as few calls as possible.
        let removed_nodes: Vec<NodeId> = fusions
            .iter()
            .flat_map(|f| &f.unfused_ops)
            .copied()
            .collect();
        self.graph.remove_nodes(&removed_nodes);

        // Add the fused operators.
        for Replacement {
            fusion,
            unfused_ops: _,
        } in fusions
        {
            match fusion {
                Fusion::Op(fusion) => {
                    self.add_operator(
                        fusion.name.as_deref(),
                        fusion.fused_op,
                        &fusion.input_ids,
                        &fusion.output_ids,
                    );
                }
                Fusion::Identity {
                    input_id,
                    output_id,
                } => {
                    // Optimization must preserve input/output IDs, so if the
                    // identity output is a graph output, replace with an
                    // `Identity` operator. Otherwise we can remove the operator
                    // entirely.
                    if self.graph.output_ids().contains(&output_id) {
                        self.add_operator(
                            None,
                            Arc::new(Identity {}),
                            &[Some(input_id)],
                            &[Some(output_id)],
                        )
                    } else {
                        self.replace_value(output_id, input_id);
                    }
                }
                Fusion::Constant {
                    input_ids: _,
                    output_id,
                    value,
                } => {
                    let const_id = self.graph.add_constant_node(value);
                    self.replace_value(output_id, const_id);
                }
            }
        }

        n_fusions
    }

    fn output_ids(&self) -> &[NodeId] {
        &self.output_ids
    }

    fn set_captures(&mut self, captures: &[NodeId]) {
        self.graph.set_captures(captures)
    }

    /// Replace `old_value_id` with `new_value_id` in operator inputs and graph
    /// outputs.
    fn replace_value(&mut self, old_value_id: NodeId, new_value_id: NodeId) {
        // Replace `old_value_id` in graph outputs.
        for output_id in self.output_ids.iter_mut().filter(|id| **id == old_value_id) {
            *output_id = new_value_id;
        }

        // Replace `old_value_id` in operator inputs.
        let Some(consumer_ids) = self.graph.get_consumers(old_value_id) else {
            return;
        };
        let consumer_ids: SmallVec<[NodeId; 1]> = SmallVec::from_slice(consumer_ids);

        for op_id in consumer_ids {
            self.graph.replace_input(op_id, old_value_id, new_value_id);
        }
    }
}

/// Find an operator output in a subgraph which is used outside the subgraph,
/// excluding outputs listed in `output_ids`, which are the final outputs of the
/// subgraph.
fn find_operator_output_used_outside_subgraph(
    graph: &Graph,
    subgraph_ops: &[NodeId],
    output_ids: &[NodeId],
) -> Option<NodeId> {
    for op_id in subgraph_ops {
        let op = graph
            .get_node(*op_id)
            .and_then(|n| n.as_operator())
            .expect("node ID should be a valid operator ID");

        for output in op.output_ids().iter().flatten() {
            if output_ids.contains(output) {
                continue;
            }

            // Check for intermediate output used as graph output.
            if graph.output_ids().contains(output) {
                return Some(*output);
            }

            // Check for intermediate output used as input to operator node
            // outside subgraph.
            let Some(consumers) = graph.get_consumers(*output) else {
                continue;
            };
            for consumer in consumers {
                if !subgraph_ops.contains(consumer) {
                    return Some(*output);
                }
            }
        }
    }
    None
}

/// Configuration for [`GraphOptimizer::optimize`].
#[derive(Clone, Default)]
pub struct OptimizeOptions {}

/// Applies optimizations to a [`Graph`] to enable faster inference.
pub struct GraphOptimizer {}

impl GraphOptimizer {
    /// Create a new optimizer with the default set of optimizations enabled.
    pub fn new() -> Self {
        GraphOptimizer {}
    }

    /// Apply optimizations to a graph.
    ///
    /// The graph's input and output nodes, identified by
    /// [`input_ids`](Graph::input_ids) and [`output_ids`](Graph::output_ids)
    /// will be preserved. Other nodes may be modified, removed or replaced.
    ///
    /// Returns the optimized graph.
    pub fn optimize(
        &self,
        graph: Graph,
        capture_env: Option<&CaptureEnv>,
        _options: OptimizeOptions,
    ) -> Result<Graph, OptimizeError> {
        let mut graph_mut = GraphMutator::from_graph(graph);

        // "Early" fusions. These are fusions which can benefit constant
        // propagation by enabling it to eliminate more nodes, or by avoiding
        // unnecessary work during constant propagation.
        let mut early_fusions = FusionList::new();

        // Fusions which replace dynamic values with constants, extending the
        // reach of constant prop.
        early_fusions.push(ShapeSliceToConstant {});

        // Fusions which eliminate no-op nodes, saving work during constant prop
        // and inference.
        early_fusions.push(CastElimination {});
        early_fusions.push(IdentityFusion {});

        self.apply_fusions(&mut graph_mut, early_fusions.visitors())?;

        // Constant propagation.
        //
        // This is done before fusion passes since various fusions require
        // certain nodes to be constant.
        if let Some(capture_env) = capture_env {
            self.convert_captured_values_to_constants(&mut graph_mut, capture_env)?;
        }
        self.propagate_constants(&mut graph_mut)?;

        // Fuse operators.
        //
        // The ordering is significant as fusions are tried in turn until a
        // match is found.
        let mut fusions = FusionList::new();

        // Another identity elimination pass. This can eliminate patterns which
        // are discovered to be identity ops (eg. `x + 0`) after constant prop.
        fusions.push(IdentityFusion {});

        // Canonicalizations to make other fusions support a wider range of
        // patterns.
        fusions.push(ReciprocalFusion {}.into_visitor());
        fusions.push(ReduceMeanAxesFusion {}.into_visitor());

        // Activation fusions
        fusions.push(SiluFusion {}.into_visitor());
        fusions.push(SwishFusion {}.into_visitor());
        fusions.push(GeluFusion {}.into_visitor());
        fusions.push(ApproxGeluFusion {}.into_visitor());

        // Normalization fusions
        fusions.push(LayerNormalizationFusion {}.into_visitor());
        fusions.push(RmsNormalizationFusion {}.into_visitor());

        // Matmul fusions
        fusions.push(MatMulAddFusion {}.into_visitor());
        fusions.push(MatMulScaleFusion {});
        fusions.push(MatMulIntegerToFloatFusion {}.into_visitor());

        // Attention fusions
        fusions.push(AddSoftmaxFusion {}.into_visitor());

        // Layout fusions
        fusions.push(TransposeFusion {});

        let max_iters = 3;
        for _ in 0..max_iters {
            let n_fused_ops = self.apply_fusions(&mut graph_mut, fusions.visitors())?;
            if n_fused_ops == 0 {
                // We reached a fixed point.
                break;
            }
        }

        Ok(graph_mut.finalize_graph())
    }

    /// Replace captured values in a graph with constants if the captured value
    /// resolves to a constant node in the parent graph.
    ///
    /// This pass should be performed before other transformations which
    /// require certain nodes to be constants.
    fn convert_captured_values_to_constants(
        &self,
        graph: &mut GraphMutator,
        capture_env: &CaptureEnv,
    ) -> Result<(), OptimizeError> {
        let captured_constants: Vec<(NodeId, Constant)> = graph
            .graph()
            .captures()
            .iter()
            .filter_map(|&capture_id| {
                let cap_name = graph.graph().get_node(capture_id).and_then(|n| n.name())?;
                let Some(Node::Constant(const_node)) = capture_env.get_node(cap_name) else {
                    return None;
                };
                const_node
                    .clone_ref()
                    .map(|local_const| (capture_id, local_const))
            })
            .collect();

        let mut new_captures: FxHashSet<_> = graph.graph().captures().iter().copied().collect();

        for (capture_id, local_const) in captured_constants {
            let const_id = graph.add_constant_node(local_const);
            new_captures.remove(&capture_id);
            graph.replace_value(capture_id, const_id);
        }

        let new_captures: Vec<_> = new_captures.into_iter().collect();
        graph.set_captures(&new_captures);

        Ok(())
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
            let const_name = const_name.as_deref();

            let const_id = match output {
                Value::FloatTensor(tensor) => {
                    Some(graph.add_constant(const_name, tensor.into_arc()))
                }
                Value::Int32Tensor(tensor) => {
                    Some(graph.add_constant(const_name, tensor.into_arc()))
                }
                Value::Int8Tensor(tensor) => {
                    Some(graph.add_constant(const_name, tensor.into_arc()))
                }
                Value::UInt8Tensor(tensor) => {
                    Some(graph.add_constant(const_name, tensor.into_arc()))
                }

                // Sequence constants are not yet supported, but we could add
                // them in future. For now, the sequence generated by constant
                // propagation is just discarded.
                Value::Sequence(_) => None,
            };
            if let Some(const_id) = const_id {
                graph.replace_value(value_node_id, const_id);
            }
        }

        Ok(())
    }

    /// Traverse the graph and test each operator node against a sequence of
    /// fusion visitors, applying the first returned fusion if any.
    ///
    /// This function visits each operator only once, so won't combine multiple
    /// fusions. Fusions that can be combined must be applied in separate
    /// passes.
    ///
    /// Returns the number of fusions that were applied.
    fn apply_fusions(
        &self,
        graph: &mut GraphMutator,
        visitors: &[Box<dyn DynFusionVisitor>],
    ) -> Result<usize, OptimizeError> {
        // Create the prepared state once and then re-use it for each operator
        // visited.
        let prepared_state: Vec<_> = visitors.iter().map(|f| f.prepare(graph.graph())).collect();

        let n_fusions = graph.apply_fusion(|graph, op_node_id, op_node| {
            for (visitor, state) in visitors.iter().zip(&prepared_state) {
                if let Some(fusion) =
                    visitor.maybe_fuse(state.as_ref(), graph.graph(), op_node_id, op_node)
                {
                    return Some(fusion);
                }
            }
            None
        });

        Ok(n_fusions)
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// A dyn-compatible version of [`FusionVisitor`].
///
/// This replaces the associated `State` associated type with `dyn Any`.
trait DynFusionVisitor {
    fn prepare(&self, graph: &Graph) -> Box<dyn Any>;
    fn maybe_fuse(
        &self,
        state: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion>;
}

/// Wraps a fusion visitor to implement [`DynFusionVisitor`].
struct DynFusion<F: FusionVisitor>(F);

impl<F: FusionVisitor> DynFusionVisitor for DynFusion<F>
where
    F::State: Any,
{
    fn prepare(&self, graph: &Graph) -> Box<dyn Any> {
        Box::new(self.0.prepare(graph))
    }

    fn maybe_fuse(
        &self,
        state: &dyn Any,
        graph: &Graph,
        op_node_id: NodeId,
        op_node: &OperatorNode,
    ) -> Option<Fusion> {
        let state = state.downcast_ref().unwrap();
        self.0.maybe_fuse(state, graph, op_node_id, op_node)
    }
}

/// A list of fusion passes to apply to a model.
struct FusionList {
    fusions: Vec<Box<dyn DynFusionVisitor>>,
}

impl FusionList {
    /// Create an empty fusion list.
    fn new() -> Self {
        Self {
            fusions: Vec::new(),
        }
    }

    /// Add a new fusion pass.
    fn push<F: FusionVisitor + 'static>(&mut self, fusion: F) {
        self.fusions.push(Box::new(DynFusion(fusion)))
    }

    /// Return visitors for the registered fusions.
    fn visitors(&self) -> &[Box<dyn DynFusionVisitor>] {
        &self.fusions
    }
}

#[cfg(test)]
mod tests;
