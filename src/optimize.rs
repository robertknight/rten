use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::graph::{Graph, NodeId};

#[derive(Clone, Debug, PartialEq)]
pub enum OptimizeError {
    UnknownError,
}

impl Display for OptimizeError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Self::UnknownError => write!(f, "optimizing graph failed"),
        }
    }
}

impl Error for OptimizeError {}

/// Optimized graph produced by [`GraphOptimizer`].
pub struct OptimizedGraph {
    /// The optimized graph.
    pub graph: Graph,

    /// IDs of input nodes. These correspond to the input IDs passed to
    /// [`GraphOptimizer::optimize`].
    pub input_ids: Vec<NodeId>,

    /// IDs of output nodes. These correspond to the output IDs passed to
    /// [`GraphOptimizer::optimize`].
    pub output_ids: Vec<NodeId>,
}

/// Applies optimizations to a [`Graph`] to enable faster inference.
pub struct GraphOptimizer {}

impl GraphOptimizer {
    /// Create a new optimizer with the default set of optimizations enabled.
    pub fn new() -> Self {
        GraphOptimizer {}
    }

    /// Apply optimizations to a graph and return the new graph.
    ///
    /// The input and output nodes specified by `input_ids` and `output_ids`
    /// will be preserved, but their IDs may change.
    ///
    /// Other nodes in the graph
    pub fn optimize(
        &self,
        graph: Graph,
        input_ids: &[NodeId],
        output_ids: &[NodeId],
    ) -> Result<OptimizedGraph, OptimizeError> {
        let x = OptimizeError::UnknownError;

        Ok(OptimizedGraph {
            graph,
            input_ids: input_ids.to_vec(),
            output_ids: output_ids.to_vec(),
        })
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
