use std::error::Error;
use std::fmt::{Display, Formatter};

#[cfg(feature = "rten_format")]
pub mod rten_registry;
#[cfg(feature = "rten_format")]
use rten_registry::RtenOpRegistry;

#[cfg(feature = "onnx_format")]
pub mod onnx_registry;
#[cfg(feature = "onnx_format")]
use onnx_registry::OnnxOpRegistry;

/// Registry used to deserialize operators when loading a model.
///
/// A registry is created automatically when using APIs such as
/// [`Model::load_file`](crate::Model::load_file) or
/// [`ModelOptions::with_all_ops`](crate::ModelOptions::with_all_ops). These
/// registries have all operators enabled.
///
/// Enabling only a subset of operators can reduce binary size, as the linker
/// will remove code for unused operators. To do this, create a custom registry
/// using the [`op_registry`](crate::op_registry) macro and pass it to
/// [`ModelOptions::with_ops`](crate::ModelOptions::with_ops).
pub struct OpRegistry {
    /// Registry for deserializing operators from .rten model files.
    #[cfg(feature = "rten_format")]
    rten_registry: RtenOpRegistry,

    /// Registry for deserializing operators from .onnx model files.
    #[cfg(feature = "onnx_format")]
    onnx_registry: OnnxOpRegistry,
}

impl OpRegistry {
    /// Create a new registry with selected operators enabled.
    pub fn with_ops(ops: &[&dyn RegisterOp]) -> OpRegistry {
        let mut reg = OpRegistry {
            #[cfg(feature = "rten_format")]
            rten_registry: RtenOpRegistry::new(),
            #[cfg(feature = "onnx_format")]
            onnx_registry: OnnxOpRegistry::new(),
        };

        for op in ops {
            op.register(&mut reg);
        }
        reg
    }

    /// Create a new registry with all operators enabled.
    pub fn with_all_ops() -> OpRegistry {
        OpRegistry {
            #[cfg(feature = "rten_format")]
            rten_registry: RtenOpRegistry::with_all_ops(),
            #[cfg(feature = "onnx_format")]
            onnx_registry: OnnxOpRegistry::with_all_ops(),
        }
    }

    /// Return the inner registry for deserializing operators from .rten models.
    #[cfg(feature = "rten_format")]
    pub(crate) fn rten_registry(&self) -> &rten_registry::RtenOpRegistry {
        &self.rten_registry
    }

    /// Return the inner registry for deserializing operators from .onnx models.
    #[cfg(feature = "onnx_format")]
    pub(crate) fn onnx_registry(&self) -> &onnx_registry::OnnxOpRegistry {
        &self.onnx_registry
    }
}

/// Error type for errors that occur when de-serializing an operator.
#[derive(Debug)]
pub enum ReadOpError {
    /// The `attrs` field for this operator is not set or has the wrong type.
    #[cfg(feature = "rten_format")]
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
    #[allow(unused)]
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
            #[cfg(feature = "rten_format")]
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

/// Construct an [`OpRegistry`] with a given set of operators enabled.
///
/// ```no_run
/// use rten::{ModelOptions, op_registry};
///
/// // Create a registry with four operators enabled.
/// let registry = op_registry!(Conv, Add, MatMul, Relu);
/// let model = ModelOptions::with_ops(registry).load_file("model.onnx");
/// ```
#[macro_export]
macro_rules! op_registry {
    ($($op:ident),*) => {{
        #[allow(unused_mut)]
        let ops: &[&dyn $crate::RegisterOp] = &[$(
            &$crate::op_types::$op,
        )*];
        $crate::OpRegistry::with_ops(ops)
    }}
}

/// Enable an operator in a registry. See [`OpRegistry`].
pub trait RegisterOp {
    fn register(&self, registry: &mut OpRegistry);
}

/// Types that can be used with [`OpRegistry::with_ops`].
///
/// The types in this module act as "keys" that enable an operator to be
/// deserialized when loading a model.
pub mod op_types {
    use super::{OpRegistry, RegisterOp};
    use crate::ops;

    macro_rules! declare_op {
        ($op:ident) => {
            pub struct $op;

            impl RegisterOp for $op {
                fn register(&self, registry: &mut OpRegistry) {
                    #[cfg(feature = "rten_format")]
                    registry.rten_registry.register_op::<ops::$op>();
                    #[cfg(feature = "onnx_format")]
                    registry.onnx_registry.register_op::<ops::$op>();
                }
            }
        };

        ($op:ident, feature=$feature:literal) => {
            #[cfg(feature = $feature)]
            pub struct $op;

            #[cfg(feature = $feature)]
            impl RegisterOp for $op {
                fn register(&self, registry: &mut OpRegistry) {
                    #[cfg(feature = "rten_format")]
                    registry.rten_registry.register_op::<ops::$op>();
                    #[cfg(feature = "onnx_format")]
                    registry.onnx_registry.register_op::<ops::$op>();
                }
            }
        };
    }

    declare_op!(Abs);
    declare_op!(Acos);
    declare_op!(Add);
    declare_op!(And);
    declare_op!(ArgMax);
    declare_op!(ArgMin);
    declare_op!(Asin);
    declare_op!(Atan);
    declare_op!(AveragePool);
    declare_op!(BatchNormalization);
    declare_op!(Cast);
    declare_op!(CastLike);
    declare_op!(Ceil);
    declare_op!(Clip);
    declare_op!(Concat);
    declare_op!(ConcatFromSequence);
    declare_op!(Conv);
    declare_op!(ConvInteger);
    declare_op!(ConstantOfShape);
    declare_op!(ConvTranspose);
    declare_op!(Cos);
    declare_op!(CumSum);
    declare_op!(DequantizeLinear);
    declare_op!(DepthToSpace);
    declare_op!(Div);
    declare_op!(Dropout, feature = "random");
    declare_op!(DynamicQuantizeLinear);
    declare_op!(Einsum);
    declare_op!(Elu);
    declare_op!(Equal);
    declare_op!(Erf);
    declare_op!(Exp);
    declare_op!(Expand);
    declare_op!(EyeLike);
    declare_op!(Flatten);
    declare_op!(Floor);
    declare_op!(Gather);
    declare_op!(GatherElements);
    declare_op!(GatherND);
    declare_op!(Gelu);
    declare_op!(Gemm);
    declare_op!(GlobalAveragePool);
    declare_op!(Greater);
    declare_op!(GreaterOrEqual);
    declare_op!(GridSample);
    declare_op!(GRU);
    declare_op!(HardSigmoid);
    declare_op!(HardSwish);
    declare_op!(Identity);
    declare_op!(If);
    declare_op!(InstanceNormalization);
    declare_op!(IsInf);
    declare_op!(IsNaN);
    declare_op!(LayerNormalization);
    declare_op!(LeakyRelu);
    declare_op!(Less);
    declare_op!(LessOrEqual);
    declare_op!(Log);
    declare_op!(LogSoftmax);
    declare_op!(Loop);
    declare_op!(LSTM);
    declare_op!(MatMul);
    declare_op!(MatMulInteger);
    declare_op!(Max);
    declare_op!(MaxPool);
    declare_op!(Mean);
    declare_op!(Min);
    declare_op!(Mod);
    declare_op!(Mul);
    declare_op!(Neg);
    declare_op!(NonMaxSuppression);
    declare_op!(NonZero);
    declare_op!(Not);
    declare_op!(OneHot);
    declare_op!(Or);
    declare_op!(Pad);
    declare_op!(Pow);
    declare_op!(PRelu);
    declare_op!(QuantizeLinear);
    declare_op!(RandomNormal, feature = "random");
    declare_op!(RandomNormalLike, feature = "random");
    declare_op!(RandomUniform, feature = "random");
    declare_op!(RandomUniformLike, feature = "random");
    declare_op!(Range);
    declare_op!(Reciprocal);
    declare_op!(ReduceL2);
    declare_op!(ReduceMax);
    declare_op!(ReduceMean);
    declare_op!(ReduceMin);
    declare_op!(ReduceProd);
    declare_op!(ReduceSum);
    declare_op!(ReduceSumSquare);
    declare_op!(Relu);
    declare_op!(Reshape);
    declare_op!(Resize);
    declare_op!(Round);
    declare_op!(ScatterElements);
    declare_op!(ScatterND);
    declare_op!(SequenceAt);
    declare_op!(SequenceEmpty);
    declare_op!(SequenceErase);
    declare_op!(SequenceConstruct);
    declare_op!(SequenceInsert);
    declare_op!(SequenceLength);
    declare_op!(Shape);
    declare_op!(Sigmoid);
    declare_op!(Sign);
    declare_op!(Sin);
    declare_op!(Size);
    declare_op!(Slice);
    declare_op!(Softmax);
    declare_op!(Softplus);
    declare_op!(Split);
    declare_op!(SplitToSequence);
    declare_op!(Sqrt);
    declare_op!(Squeeze);
    declare_op!(STFT, feature = "fft");
    declare_op!(Sub);
    declare_op!(Sum);
    declare_op!(Tan);
    declare_op!(Tanh);
    declare_op!(Tile);
    declare_op!(TopK);
    declare_op!(Transpose);
    declare_op!(Trilu);
    declare_op!(Unsqueeze);
    declare_op!(Where);
    declare_op!(Xor);
}
