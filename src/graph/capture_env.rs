use rustc_hash::FxHashMap;

use crate::value::{Value, ValueOrView, ValueView};

use super::{Graph, Node, NodeId};

/// An environment from which subgraphs can resolve captured values.
///
/// Subgraphs used by control flow operators (`If`, `Loop` etc.) may contain
/// value nodes that capture their values from parent graphs, like a captured
/// value in a Rust closure. A `CaptureEnv` is passed to the subgraph when
/// it is executed and used to resolve these values.
///
/// `CaptureEnv`s are arranged in a hierarchy. Value lookups will attempt to
/// look up the value in the environment's associated graph. If no such node
/// exists, the value will be looked up in the parent environment and so on.
///
/// Values can be captured either by reference or
/// by value. Values that are captured by-value can potentially be used as
/// [`in-place inputs`](Operator::run_in_place).
#[derive(Clone)]
pub struct CaptureEnv<'a> {
    // The parent environment to search if a node name is not found in this
    // environment.
    parent: Option<&'a CaptureEnv<'a>>,

    // The "local" graph for this environment. Node names are looked up in
    // this graph first and if found, values are resolved from `inputs` or
    // `temp_values`.
    graph: Option<&'a Graph>,

    // Values passed as inputs to the graph run.
    inputs: Option<&'a FxHashMap<NodeId, ValueOrView<'a>>>,

    // Values computed during the graph run, captured by reference.
    temp_values_by_ref: Option<&'a FxHashMap<NodeId, Value>>,

    // Values computed during the graph run, captured by value.
    temp_values: Option<FxHashMap<NodeId, Value>>,
}

impl<'a> CaptureEnv<'a> {
    /// Create a new capture environment.
    ///
    /// Lookups will first match nodes in `graph` and then try the `parent`
    /// environment if that fails. Lookups that match constant nodes will be
    /// resolved from the node directly. Lookups that match value nodes will
    /// be resolved from the captured values first or the captured inputs
    /// otherwise.
    pub fn new(
        parent: Option<&'a CaptureEnv<'a>>,
        graph: &'a Graph,
        inputs: Option<&'a FxHashMap<NodeId, ValueOrView<'a>>>,
        temp_values_by_ref: Option<&'a FxHashMap<NodeId, Value>>,
        temp_values: Option<FxHashMap<NodeId, Value>>,
    ) -> CaptureEnv<'a> {
        CaptureEnv {
            parent,
            graph: Some(graph),
            inputs,
            temp_values_by_ref,
            temp_values,
        }
    }

    /// Create an empty capture environment which will return `None` for all
    /// lookups.
    #[allow(unused)]
    pub fn empty() -> Self {
        // This could be generated via a derived Default impl, but the meaning
        // of a "default" CaptureEnv may be unclear.
        CaptureEnv {
            parent: None,
            graph: None,
            inputs: None,
            temp_values_by_ref: None,
            temp_values: None,
        }
    }

    /// Simplified constructor for capture environments associated with top
    /// level graphs.
    ///
    /// All of the sources of runtime-provided inputs are set to `None`, so
    /// only captured constants are available.
    #[cfg(test)]
    pub fn top_level_static(graph: &'a Graph) -> Self {
        CaptureEnv {
            parent: None,
            graph: Some(graph),
            inputs: None,
            temp_values_by_ref: None,
            temp_values: None,
        }
    }

    /// Return a new capture environment which has `self` as a parent.
    ///
    /// The child `CaptureEnv` will have no captures of its own. This is useful
    /// in loop operators which need to create a new capture environment to pass
    /// to each iteration of a loop.
    pub fn child(&self) -> CaptureEnv<'_> {
        CaptureEnv {
            parent: Some(self),
            graph: None,
            inputs: None,
            temp_values_by_ref: None,
            temp_values: None,
        }
    }

    /// Look up a node by name in this environment.
    pub fn get_node(&self, name: &str) -> Option<&'a Node> {
        if let Some(graph) = self.graph {
            if let Some(node_id) = graph.get_node_id(name) {
                // If a node by this name exists in this graph, but is a placeholder
                // for a value captured from a parent graph, then ignore it.
                if !graph.captures().contains(&node_id) {
                    return graph.get_node(node_id);
                }
            }
        }

        self.parent.and_then(|parent| parent.get_node(name))
    }

    /// Look up an operator input value by name in this environment.
    pub fn get_input(&self, name: &str) -> Option<ValueView<'_>> {
        if let Some(graph) = self.graph {
            if let Some(node_id) = graph.get_node_id(name) {
                // If a node by this name exists in this graph, but is a placeholder
                // for a value captured from a parent graph, then ignore it.
                if !graph.captures().contains(&node_id) {
                    // Otherwise, get the value from this scope.
                    return match graph.get_node(node_id) {
                        Some(Node::Constant(c)) => Some(c.as_view()),
                        Some(Node::Value(_)) => self
                            .temp_values_by_ref
                            .and_then(|tv| tv.get(&node_id))
                            .map(|i| i.as_view())
                            .or_else(|| {
                                self.temp_values
                                    .as_ref()
                                    .and_then(|tv| tv.get(&node_id))
                                    .map(|o| o.as_view())
                            })
                            .or_else(|| {
                                self.inputs
                                    .and_then(|i| i.get(&node_id))
                                    .map(|i| i.as_view())
                            }),
                        _ => None,
                    };
                }
            }
        }

        self.parent.and_then(|parent| parent.get_input(name))
    }

    /// Remove and return a value from the capture environment's map of by-value
    /// captures.
    pub fn take_input(&mut self, name: &str) -> Option<Value> {
        let node_id = self.graph.and_then(|g| g.get_node_id(name))?;
        self.temp_values.as_mut()?.remove(&node_id)
    }

    /// Remove and return all by-value captures.
    pub fn take_all_inputs(&mut self) -> Option<FxHashMap<NodeId, Value>> {
        self.temp_values.take()
    }
}
