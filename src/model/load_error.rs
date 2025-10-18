use std::error::Error;
use std::fmt::{Display, Formatter};

/// Errors that occur when loading a model.
#[derive(Debug)]
pub struct LoadError {
    inner: LoadErrorImpl,
    node: Option<String>,
}

impl LoadError {
    pub(crate) fn new(kind: LoadErrorImpl) -> Self {
        Self {
            inner: kind,
            node: None,
        }
    }

    pub(crate) fn for_node(node: Option<&str>, kind: LoadErrorImpl) -> Self {
        Self {
            inner: kind,
            node: node.map(|n| n.to_string()),
        }
    }

    /// The name of the graph node that this error relates to.
    ///
    /// This can be `None` if the error is not about a specific node, or if that
    /// node doesn't have a name.
    pub fn node(&self) -> Option<&str> {
        self.node.as_deref()
    }

    /// Return the category of error.
    pub fn kind(&self) -> LoadErrorKind {
        self.inner.kind()
    }
}

impl Display for LoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(node) = self.node.as_deref() {
            write!(f, "in node \"{}\": {}", node, self.inner)
        } else {
            self.inner.fmt(f)
        }
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.inner.source()
    }
}

impl From<LoadErrorImpl> for LoadError {
    fn from(val: LoadErrorImpl) -> Self {
        Self::new(val)
    }
}

/// Categories of error when loading a model.
///
/// See [`LoadError::kind`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LoadErrorKind {
    /// An I/O error occurred reading the model file.
    IoError,

    /// An error occurred parsing the model file.
    ParseError,

    /// Failed to deserialize an operator.
    OperatorInvalid,

    /// There was a problem with the graph structure.
    GraphError,

    /// A problem occurred while optimizing the model.
    OptimizeError,

    /// The model file type was unrecognized.
    UnknownFileType,

    /// There was a problem loading weights from an external file.
    ExternalDataError,

    /// The model file type is recognized, but support for this format was not
    /// enabled when the `rten` crate was built.
    FormatNotEnabled,
}

/// The internal implementation of [`LoadError`].
#[derive(Debug)]
pub(crate) enum LoadErrorImpl {
    /// The FlatBuffers data describing the model is not supported by this
    /// version of RTen.
    SchemaVersionUnsupported,

    /// An error occurred reading the file from disk.
    ReadFailed(std::io::Error),

    /// An error occurred parsing the data describing the model structure.
    ParseFailed(Box<dyn Error + Send + Sync>),

    /// An error occurred deserializing an operator.
    OperatorInvalid(Box<dyn Error + Send + Sync>),

    /// An error occurred while traversing the model's graph to instantiate
    /// nodes and connections.
    GraphError(Box<dyn Error + Send + Sync>),

    /// An error occurred while optimizing the graph.
    OptimizeError(Box<dyn Error + Send + Sync>),

    /// The file's header is invalid.
    InvalidHeader(Box<dyn Error + Send + Sync>),

    /// The file type of the model could not be determined.
    UnknownFileType,

    /// An error occurred reading tensor data stored externally.
    ExternalDataError(Box<dyn Error + Send + Sync>),

    /// The model file type is supported by RTen, but the necessary crate
    /// features were not enabled.
    #[allow(unused)]
    FormatNotEnabled,
}

impl LoadErrorImpl {
    fn kind(&self) -> LoadErrorKind {
        type Kind = LoadErrorKind;

        match self {
            Self::SchemaVersionUnsupported => Kind::ParseError,
            Self::ReadFailed(_) => Kind::IoError,
            Self::ParseFailed(_) => Kind::ParseError,
            Self::OperatorInvalid(_) => Kind::OperatorInvalid,
            Self::GraphError(_) => Kind::GraphError,
            Self::OptimizeError(_) => Kind::OptimizeError,
            Self::InvalidHeader(_) => Kind::ParseError,
            Self::UnknownFileType => Kind::UnknownFileType,
            Self::ExternalDataError(_) => Kind::ExternalDataError,
            Self::FormatNotEnabled => Kind::FormatNotEnabled,
        }
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SchemaVersionUnsupported => None,
            Self::ReadFailed(err) => Some(err),
            Self::ParseFailed(err) => Some(err.as_ref()),
            Self::OperatorInvalid(err) => Some(err.as_ref()),
            Self::GraphError(err) => Some(err.as_ref()),
            Self::OptimizeError(err) => Some(err.as_ref()),
            Self::InvalidHeader(err) => Some(err.as_ref()),
            Self::UnknownFileType => None,
            Self::ExternalDataError(err) => Some(err.as_ref()),
            Self::FormatNotEnabled => None,
        }
    }
}

impl Display for LoadErrorImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaVersionUnsupported => write!(f, "unsupported schema version"),
            Self::ReadFailed(e) => write!(f, "read error: {e}"),
            Self::ParseFailed(e) => write!(f, "parse error: {e}"),
            Self::OperatorInvalid(e) => write!(f, "operator error: {e}"),
            Self::GraphError(e) => write!(f, "graph error: {e}"),
            Self::OptimizeError(e) => write!(f, "graph optimization error: {e}"),
            Self::InvalidHeader(e) => write!(f, "invalid header: {e}"),
            Self::UnknownFileType => write!(f, "unknown model file type"),
            Self::ExternalDataError(e) => write!(f, "external data error: {e}"),
            Self::FormatNotEnabled => {
                write!(f, "rten was built without support for this model format")
            }
        }
    }
}

/// Create a [`LoadError`] that relates to a specific graph node.
macro_rules! load_error {
    ($kind:ident, $node_name:expr, $format_str:literal, $($arg:tt)*) => {{
        let err = format!($format_str, $($arg)*);
        LoadError::for_node($node_name, LoadErrorImpl::$kind(err.into()))
    }};

    ($kind:ident, $node_name:expr, $err:expr) => {{
        LoadError::for_node($node_name, LoadErrorImpl::$kind($err.into()))
    }}
}

pub(crate) use load_error;
