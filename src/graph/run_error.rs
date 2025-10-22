use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::operator::{OpError, OpRunContext};
use crate::value::{ValueMeta, ValueType};

/// Errors that occur when running a model.
#[derive(Debug)]
pub struct RunError(RunErrorImpl);

impl RunError {
    /// Name hierarchy of the graph nodes that this error relates to.
    ///
    /// In a graph with no subgraphs, this will contain one entry with
    /// the name of the node, or `None` if the error does not relate to a
    /// particular node or the node has no name.
    ///
    /// When an error occurs in a subgraph, the last entry contains the name
    /// of the node in the inner-most subgraph and the previous entries contain
    /// the names of the operators in the ancestor graphs.
    pub fn node_path(&self) -> Vec<Option<&str>> {
        self.0.node_path()
    }

    /// Return the general category of error.
    pub fn kind(&self) -> RunErrorKind {
        self.0.kind()
    }

    pub(crate) fn op_error(name: &str, error: OpError, ctx: &OpRunContext) -> Self {
        RunErrorImpl::OperatorError {
            name: name.to_string(),
            error,
            inputs: ctx
                .inputs()
                .iter()
                .map(|inp| inp.map(|inp| inp.to_meta()))
                .collect(),
        }
        .into()
    }

    pub(crate) fn in_place_op_error(
        name: &str,
        error: OpError,
        ctx: &OpRunContext,
        main_input_dtype: ValueType,
        main_input_shape: &[usize],
    ) -> Self {
        let meta = ValueMeta {
            dtype: main_input_dtype,
            shape: main_input_shape.to_vec(),
        };
        let mut inputs: Vec<_> = [Some(meta)].into();
        inputs.extend(ctx.inputs().iter().map(|inp| inp.map(|inp| inp.to_meta())));
        RunErrorImpl::OperatorError {
            name: name.to_string(),
            error,
            inputs,
        }
        .into()
    }

    pub(crate) fn subgraph_error(name: Option<&str>, error: Self) -> Self {
        RunErrorImpl::SubgraphError {
            name: name.unwrap_or_default().to_string(),
            error: Box::new(error),
        }
        .into()
    }
}

impl Display for RunError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for RunError {}

impl From<RunErrorImpl> for RunError {
    fn from(inner: RunErrorImpl) -> Self {
        Self(inner)
    }
}

/// The category of model execution error. See [`RunError::kind`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RunErrorKind {
    /// An input or output node was not found.
    NodeNotFound,
    /// Failed to construct an execution plan that would generate the requested
    /// outputs from the inputs.
    PlanningError,
    /// An error occurred when running an operator.
    OperatorError,
}

/// Internal implementation of [`RunError`].
#[derive(Debug)]
pub(crate) enum RunErrorImpl {
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
        /// Inputs can be `None` if they are optional inputs which were not
        /// provided.
        inputs: Vec<Option<ValueMeta>>,
    },

    /// The output of a graph operator did not match expectations (eg. the
    /// count, types or shapes of outputs did not match what was expected.)
    OutputMismatch {
        /// Name of the operator node.
        name: String,

        /// Error details.
        error: String,
    },

    /// An error occurred while running a subgraph.
    SubgraphError {
        /// Name of the operator which ran the subgraph.
        name: String,

        /// Error that occurred while running the subgraph.
        error: Box<RunError>,
    },
}

impl RunErrorImpl {
    fn kind(&self) -> RunErrorKind {
        type Kind = RunErrorKind;

        match self {
            Self::InvalidNodeId | Self::InvalidNodeName(_) => Kind::NodeNotFound,
            Self::PlanningError(_) => Kind::PlanningError,
            Self::OperatorError { .. } | Self::OutputMismatch { .. } => Kind::OperatorError,
            Self::SubgraphError { error, .. } => error.kind(),
        }
    }

    fn node_path(&self) -> Vec<Option<&str>> {
        match self {
            Self::InvalidNodeId => [None].into(),
            Self::InvalidNodeName(name) => [Some(name.as_str())].into(),
            Self::PlanningError(_) => [None].into(),
            Self::OperatorError { name, .. } => [Some(name.as_str())].into(),
            Self::OutputMismatch { name, .. } => [Some(name.as_str())].into(),
            Self::SubgraphError { name, error } => {
                let mut path = vec![Some(name.as_str())];
                path.extend(error.node_path());
                path
            }
        }
    }
}

impl Display for RunErrorImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNodeId => write!(f, "node ID is invalid"),
            Self::InvalidNodeName(name) => write!(f, "no node found with name {}", name),
            Self::PlanningError(err) => write!(f, "planning error: {}", err),
            Self::OperatorError {
                name,
                error: err,
                inputs,
            } => {
                write!(f, "operator \"{}\" failed: {}. Inputs were (", name, err,)?;
                for (i, input) in inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if let Some(meta) = input {
                        write!(f, "{}", meta)?;
                    } else {
                        write!(f, "-")?;
                    }
                }
                write!(f, ")")
            }
            Self::OutputMismatch { name, error } => {
                write!(f, "operator \"{}\" output mismatch: {}", name, error)
            }
            Self::SubgraphError { name, error } => {
                write!(f, "operator \"{}\" subgraph error: {}", name, error)
            }
        }
    }
}
