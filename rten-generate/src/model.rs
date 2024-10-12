//! Abstraction over [`rten::Model`] for querying and executing ML models.

use std::error::Error;

use rten::{Dimension, InputOrOutput, NodeId, Output, RunOptions};

/// Describes the name and shape of a model input or output.
///
/// This is similar to [`rten::NodeInfo`] but the name and shape are required.
#[derive(Clone)]
pub struct NodeInfo {
    name: String,
    shape: Vec<Dimension>,
}

impl NodeInfo {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[Dimension] {
        &self.shape
    }

    pub fn from_name_shape(name: &str, shape: &[Dimension]) -> NodeInfo {
        NodeInfo {
            name: name.to_string(),
            shape: shape.to_vec(),
        }
    }
}

/// Abstraction over [`rten::Model`] used by [`Generator`](crate::Generator) to
/// query and execute a machine learning model.
///
/// This is implemented by [`rten::Model`] and the trait's methods correspond
/// to methods of the same name in that type.
pub trait Model {
    /// Get the ID of an input or output node.
    fn find_node(&self, name: &str) -> Option<NodeId>;

    /// Get the name and shape of an input or output node.
    ///
    /// Returns `None` if the node does not exist, or name or shape information
    /// is not available.
    fn node_info(&self, id: NodeId) -> Option<NodeInfo>;

    /// Return the node IDs of the model's inputs.
    fn input_ids(&self) -> &[NodeId];

    /// Run the model with the provided inputs and return the results.
    fn run(
        &self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, Box<dyn Error>>;

    /// Run as much of the model as possible given the provided inputs and
    /// return the leaves of the evaluation where execution stopped.
    fn partial_run(
        &self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<(NodeId, Output)>, Box<dyn Error>>;
}

impl Model for rten::Model {
    fn find_node(&self, name: &str) -> Option<NodeId> {
        self.find_node(name)
    }

    fn node_info(&self, id: NodeId) -> Option<NodeInfo> {
        self.node_info(id).and_then(|info| {
            let name = info.name()?;
            let dims = info.shape()?;

            Some(NodeInfo {
                name: name.to_string(),
                shape: dims,
            })
        })
    }

    fn input_ids(&self) -> &[NodeId] {
        self.input_ids()
    }

    fn run(
        &self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, Box<dyn Error>> {
        self.run(inputs, outputs, opts).map_err(|e| e.into())
    }

    fn partial_run(
        &self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<(NodeId, Output)>, Box<dyn Error>> {
        self.partial_run(inputs, outputs, opts)
            .map_err(|e| e.into())
    }
}
