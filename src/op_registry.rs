use std::error::Error;
use std::fmt::{Display, Formatter};

pub mod rten_registry;
use rten_registry::RtenOpRegistry;

pub mod onnx_registry;
use onnx_registry::OnnxOpRegistry;

/// Registry used to deserialize operators when loading a model.
///
/// New registries have no operators registered by default. To create a registry
/// with all built-in operators pre-registered, use
/// [`OpRegistry::with_all_ops`]. Alternatively create a new registry and
/// selectively register the required operators using
/// [`OpRegistry::register_op`]. This can be useful to reduce binary size, as
/// the linker will remove code for unused operators.
#[derive(Default)]
pub struct OpRegistry {
    /// Registry for deserializing operators from .rten model files.
    rten_registry: RtenOpRegistry,

    /// Registry for deserializing operators from .onnx model files.
    onnx_registry: OnnxOpRegistry,
}

impl OpRegistry {
    /// Create a new empty registry.
    pub fn new() -> OpRegistry {
        OpRegistry {
            rten_registry: RtenOpRegistry::new(),
            onnx_registry: OnnxOpRegistry::new(),
        }
    }

    /// Register the default/built-in implementation of an operator.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use rten::ops::{Conv, Gemm, MaxPool, ReduceMean, Relu};
    /// use rten::{Model, ModelOptions, OpRegistry};
    ///
    /// // Register only the ops needed for the model.
    /// let mut reg = OpRegistry::new();
    /// reg.register_op::<Conv>();
    /// reg.register_op::<Relu>();
    /// reg.register_op::<MaxPool>();
    /// reg.register_op::<ReduceMean>();
    /// reg.register_op::<Gemm>();
    ///
    /// let model = ModelOptions::with_ops(reg).load_file("mnist.onnx")?;
    /// # Ok(()) }
    /// ```
    pub fn register_op<Op: rten_registry::ReadOp + onnx_registry::ReadOp + 'static>(&mut self) {
        self.rten_registry.register_op::<Op>();
        self.onnx_registry.register_op::<Op>();
    }

    /// Return the inner registry for deserializing operators from .rten models.
    pub(crate) fn rten_registry(&self) -> &rten_registry::RtenOpRegistry {
        &self.rten_registry
    }

    /// Return the inner registry for deserializing operators from .onnx models.
    pub(crate) fn onnx_registry(&self) -> &onnx_registry::OnnxOpRegistry {
        &self.onnx_registry
    }

    /// Create a new registry with all built-in operators registered.
    pub fn with_all_ops() -> OpRegistry {
        OpRegistry {
            rten_registry: RtenOpRegistry::with_all_ops(),
            onnx_registry: OnnxOpRegistry::with_all_ops(),
        }
    }
}

/// Error type for errors that occur when de-serializing an operator.
#[derive(Debug)]
pub enum ReadOpError {
    /// The `attrs` field for this operator is not set or has the wrong type.
    AttrsMissingError,
    /// An attribute has an unsupported or invalid value.
    AttrError {
        /// Name of the attribute.
        attr: String,
        /// Description of the attribute error.
        error: String,
    },
    /// The operator is either unrecognized or not available.
    OperatorUnavailable {
        /// Name of the operator, if recognized but not enabled.
        name: Option<String>,
    },
    /// The operator requires a crate feature that was not enabled.
    FeatureNotEnabled {
        /// Name of the operator.
        name: String,

        /// Crate feature needed to enable it.
        feature: String,
    },
    /// An error occurred deserializing a subgraph.
    SubgraphError(Box<dyn Error + Send + Sync>),
}

impl ReadOpError {
    fn attr_error(attr: impl AsRef<str>, error: impl AsRef<str>) -> Self {
        Self::AttrError {
            attr: attr.as_ref().to_string(),
            error: error.as_ref().to_string(),
        }
    }
}

impl Display for ReadOpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadOpError::AttrsMissingError => write!(f, "attributes are missing"),
            ReadOpError::AttrError { attr, error } => {
                write!(f, "error in attribute \"{}\": {}", attr, error)
            }
            ReadOpError::SubgraphError(err) => write!(f, "subgraph error: {}", err),
            ReadOpError::OperatorUnavailable { name } => {
                if let Some(name) = name {
                    write!(f, "{name} operator not enabled")
                } else {
                    write!(f, "operator not supported")
                }
            }
            ReadOpError::FeatureNotEnabled { name, feature } => {
                write!(
                    f,
                    "{name} operator not enabled because rten was compiled without the \"{feature}\" feature"
                )
            }
        }
    }
}

impl Error for ReadOpError {}
