//! Deserialization of operators from FlatBuffers-based .rten files.

use std::collections::HashMap;
use std::sync::Arc;

use rten_model_file::schema as sg;
use smallvec::{SmallVec, smallvec};

use super::ReadOpError;
use crate::graph::Graph;
use crate::ops;
use crate::ops::{
    BoxOrder, CoordTransformMode, DepthToSpaceMode, Direction, NearestMode, Operator, PadMode,
    Padding, ResizeMode, ScatterReduction,
};
use crate::value::{DataType, Scalar};

/// Context object passed to [`ReadOp::read`] implementations.
pub trait OpLoadContext {
    /// Deserialize a graph definition.
    fn load_graph(&self, graph: sg::Graph) -> Result<Graph, ReadOpError>;
}

/// Deserialize operators from .rten model files.
#[derive(Default)]
pub struct RtenOpRegistry {
    ops: HashMap<sg::OperatorType, Box<ReadOpFunction>>,
}

impl RtenOpRegistry {
    /// Create a new empty registry.
    pub fn new() -> RtenOpRegistry {
        RtenOpRegistry {
            ops: HashMap::new(),
        }
    }

    /// Register the default/built-in implementation of an operator.
    pub fn register_op<Op: ReadOp + 'static>(&mut self) {
        self.register_op_with_factory(
            Op::op_type(),
            Box::new(|op: &sg::OperatorNode, ctx: &dyn OpLoadContext| Op::read_boxed(op, ctx)),
        );
    }

    /// Register a stub implementation of an operator.
    ///
    /// This registers stubs for an operator that is not available because
    /// necessary crate features were not enabled. The purpose of the stub is
    /// to generate a more helpful error message.
    #[allow(unused)]
    pub(crate) fn register_stub_op(&mut self, op_type: sg::OperatorType, feature: &'static str) {
        self.register_op_with_factory(
            op_type,
            Box::new(move |_op, _ctx| {
                Err(ReadOpError::FeatureNotEnabled {
                    name: op_type.variant_name().unwrap_or(""),
                    feature,
                })
            }),
        );
    }

    /// Deserialize an operator from a model file using the operators in the
    /// registry.
    pub(crate) fn read_op(&self, op: &sg::OperatorNode, ctx: &dyn OpLoadContext) -> ReadOpResult {
        self.ops
            .get(&op.type_())
            .ok_or_else(|| ReadOpError::OperatorUnavailable {
                name: op.type_().variant_name(),
            })
            .and_then(|read_fn| read_fn(op, ctx))
    }

    /// Register an operator with a custom factory to deserialize it from a
    /// model file.
    fn register_op_with_factory(
        &mut self,
        op_type: sg::OperatorType,
        factory: Box<ReadOpFunction>,
    ) {
        self.ops.insert(op_type, factory);
    }

    /// Create a new registry with all built-in operators registered.
    pub fn with_all_ops() -> RtenOpRegistry {
        let mut reg = RtenOpRegistry::new();

        macro_rules! register_op {
            ($op:ident) => {
                reg.register_op::<ops::$op>()
            };

            ($op:ident, feature=$feature:literal) => {
                #[cfg(feature = $feature)]
                reg.register_op::<ops::$op>();
                #[cfg(not(feature = $feature))]
                reg.register_stub_op(sg::OperatorType::$op, $feature);
            };
        }

        register_op!(Abs);
        register_op!(Acos);
        register_op!(Add);
        register_op!(And);
        register_op!(ArgMax);
        register_op!(ArgMin);
        register_op!(Asin);
        register_op!(Atan);
        register_op!(AveragePool);
        register_op!(BatchNormalization);
        register_op!(Cast);
        register_op!(CastLike);
        register_op!(Ceil);
        register_op!(Clip);
        register_op!(Concat);
        register_op!(ConcatFromSequence);
        register_op!(Conv);
        register_op!(ConvInteger);
        register_op!(ConstantOfShape);
        register_op!(ConvTranspose);
        register_op!(Cos);
        register_op!(CumSum);
        register_op!(DequantizeLinear);
        register_op!(DepthToSpace);
        register_op!(Div);
        register_op!(Dropout, feature = "random");
        register_op!(DynamicQuantizeLinear);
        register_op!(Einsum);
        register_op!(Elu);
        register_op!(Equal);
        register_op!(Erf);
        register_op!(Exp);
        register_op!(Expand);
        register_op!(EyeLike);
        register_op!(Flatten);
        register_op!(Floor);
        register_op!(Gather);
        register_op!(GatherElements);
        register_op!(GatherND);
        register_op!(Gelu);
        register_op!(Gemm);
        register_op!(GlobalAveragePool);
        register_op!(Greater);
        register_op!(GreaterOrEqual);
        register_op!(GridSample);
        register_op!(GRU);
        register_op!(HardSigmoid);
        register_op!(HardSwish);
        register_op!(Identity);
        register_op!(If);
        register_op!(InstanceNormalization);
        register_op!(IsInf);
        register_op!(IsNaN);
        register_op!(LayerNormalization);
        register_op!(LeakyRelu);
        register_op!(Less);
        register_op!(LessOrEqual);
        register_op!(Log);
        register_op!(LogSoftmax);
        register_op!(Loop);
        register_op!(LSTM);
        register_op!(MatMul);
        register_op!(MatMulInteger);
        register_op!(Max);
        register_op!(MaxPool);
        register_op!(Mean);
        register_op!(Min);
        register_op!(Mod);
        register_op!(Mul);
        register_op!(Neg);
        register_op!(NonMaxSuppression);
        register_op!(NonZero);
        register_op!(Not);
        register_op!(OneHot);
        register_op!(Or);
        register_op!(Pad);
        register_op!(Pow);
        register_op!(PRelu);
        register_op!(QuantizeLinear);
        register_op!(RandomNormal, feature = "random");
        register_op!(RandomNormalLike, feature = "random");
        register_op!(RandomUniform, feature = "random");
        register_op!(RandomUniformLike, feature = "random");
        register_op!(Range);
        register_op!(Reciprocal);
        register_op!(ReduceL2);
        register_op!(ReduceMax);
        register_op!(ReduceMean);
        register_op!(ReduceMin);
        register_op!(ReduceProd);
        register_op!(ReduceSum);
        register_op!(ReduceSumSquare);
        register_op!(Relu);
        register_op!(Reshape);
        register_op!(Resize);
        register_op!(Round);
        register_op!(ScatterElements);
        register_op!(ScatterND);
        register_op!(SequenceAt);
        register_op!(SequenceEmpty);
        register_op!(SequenceErase);
        register_op!(SequenceConstruct);
        register_op!(SequenceInsert);
        register_op!(SequenceLength);
        register_op!(Shape);
        register_op!(Sigmoid);
        register_op!(Sign);
        register_op!(Sin);
        register_op!(Size);
        register_op!(Slice);
        register_op!(Softmax);
        register_op!(Softplus);
        register_op!(Split);
        register_op!(SplitToSequence);
        register_op!(Sqrt);
        register_op!(Squeeze);
        register_op!(STFT, feature = "fft");
        register_op!(Sub);
        register_op!(Sum);
        register_op!(Tan);
        register_op!(Tanh);
        register_op!(Tile);
        register_op!(TopK);
        register_op!(Transpose);
        register_op!(Trilu);
        register_op!(Unsqueeze);
        register_op!(Where);
        register_op!(Xor);

        reg
    }
}

pub fn convert_dtype(attr: &'static str, dtype: sg::DataType) -> Result<DataType, ReadOpError> {
    match dtype {
        sg::DataType::Int32 => Ok(DataType::Int32),
        sg::DataType::Float => Ok(DataType::Float),
        sg::DataType::UInt8 => Ok(DataType::UInt8),
        sg::DataType::Int8 => Ok(DataType::Int8),
        _ => Err(ReadOpError::AttrError {
            attr,
            error: "unknown value",
        }),
    }
}

fn convert_reduction(
    attr: &'static str,
    r: sg::ScatterReduction,
) -> Result<Option<ScatterReduction>, ReadOpError> {
    let reduction = match r {
        sg::ScatterReduction::None => None,
        sg::ScatterReduction::Add => Some(ScatterReduction::Add),
        sg::ScatterReduction::Mul => Some(ScatterReduction::Mul),
        sg::ScatterReduction::Min => Some(ScatterReduction::Min),
        sg::ScatterReduction::Max => Some(ScatterReduction::Max),
        _ => {
            return Err(ReadOpError::AttrError {
                attr,
                error: "unknown value",
            });
        }
    };
    Ok(reduction)
}

fn padding_from_attrs(
    auto_pad: sg::AutoPad,
    pads: Option<flatbuffers::Vector<'_, u32>>,
) -> Padding {
    match (auto_pad, pads) {
        (sg::AutoPad::Same, _) => Padding::Same,
        (sg::AutoPad::NotSet, Some(pads)) => {
            Padding::Fixed(pads.iter().map(|p| p as usize).collect())
        }
        _ => Padding::Fixed(smallvec!(0; 4)),
    }
}

fn vec_from_attr(attr: Option<flatbuffers::Vector<u32>>, default: &[usize]) -> Vec<usize> {
    attr.map(|val| val.iter().map(|x| x as usize).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn opt_vec_from_attr(attr: Option<flatbuffers::Vector<u32>>) -> Option<Vec<usize>> {
    attr.map(|val| val.iter().map(|x| x as usize).collect())
}

/// Result of deserializing an operator node from a model file.
type ReadOpResult = Result<Arc<dyn Operator + Send + Sync>, ReadOpError>;

/// A function that deserializes an operator node.
type ReadOpFunction = dyn Fn(&sg::OperatorNode, &dyn OpLoadContext) -> ReadOpResult;

/// Trait that deserializes an operator from a `.rten` file into an [`Operator`]
/// implementation.
///
/// This trait is implemented for all operators in [`crate::ops`].
pub trait ReadOp: Operator + Sized + Send + Sync {
    /// Return the type enum value for this operator.
    fn op_type() -> sg::OperatorType;

    /// Deserialize an operator.
    ///
    /// The node's type must correspond to the result of `op_type`.
    fn read(op: &sg::OperatorNode, ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError>;

    /// Deserialize an operator into a boxed `dyn Operator`.
    ///
    /// The node's type must correspond to the result of `op_type`.
    fn read_boxed(op: &sg::OperatorNode, ctx: &dyn OpLoadContext) -> ReadOpResult
    where
        Self: 'static,
    {
        let op = Self::read(op, ctx)?;
        Ok(Arc::new(op))
    }
}

/// Convenience macro to simplify implementing [`ReadOp`].
///
/// The syntax is `impl_read_op!(BarOp, attrs_as_bar_attrs, constructor)` where
/// `BarOp` is the type implementing [`Operator`], `attr_as_bar_attrs` is the
/// method on `sg::OperatorNode` to get the associated attributes table and
/// `constructor` is a function that takes the operator attributes and
/// constructs a `BarOp` struct. There are a few tokens that can be used in
/// place of `constructor` to reduce boilerplate when constructing different
/// operators that take the same set of attributes.
macro_rules! impl_read_op {
    ($op:ident) => {
        impl ReadOp for ops::$op {
            fn op_type() -> sg::OperatorType {
                sg::OperatorType::$op
            }

            fn read(_op: &sg::OperatorNode, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
                Ok(ops::$op {})
            }
        }
    };

    ($op:ident, $attrs_method:ident, axis) => {
        impl ReadOp for ops::$op {
            fn op_type() -> sg::OperatorType {
                sg::OperatorType::$op
            }

            fn read(op: &sg::OperatorNode, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
                let attrs = op.$attrs_method().ok_or(ReadOpError::AttrsMissingError)?;
                let op = ops::$op {
                    axis: attrs.axis() as isize,
                };
                Ok(op)
            }
        }
    };

    ($op:ident, $attrs_method:ident, reduce_axis) => {
        impl ReadOp for ops::$op {
            fn op_type() -> sg::OperatorType {
                sg::OperatorType::$op
            }

            fn read(op: &sg::OperatorNode, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
                let attrs = op.$attrs_method().ok_or(ReadOpError::AttrsMissingError)?;
                let op = ops::$op {
                    axis: attrs.axis() as isize,
                    keep_dims: attrs.keep_dims(),
                };
                Ok(op)
            }
        }
    };

    ($op:ident, $attrs_method:ident, reduce_axes) => {
        impl ReadOp for ops::$op {
            fn op_type() -> sg::OperatorType {
                sg::OperatorType::$op
            }

            fn read(op: &sg::OperatorNode, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
                let attrs = op.$attrs_method().ok_or(ReadOpError::AttrsMissingError)?;
                let axes = attrs.axes().map(|axes| axes.iter().collect());
                let op = ops::$op {
                    axes,
                    keep_dims: attrs.keep_dims(),
                };
                Ok(op)
            }
        }
    };

    ($op:ident, $attrs_method:ident, $read_op:expr) => {
        impl ReadOp for ops::$op {
            fn op_type() -> sg::OperatorType {
                sg::OperatorType::$op
            }

            fn read(op: &sg::OperatorNode, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
                let attrs = op.$attrs_method().ok_or(ReadOpError::AttrsMissingError)?;
                #[allow(clippy::redundant_closure_call)]
                let op = { $read_op(attrs)? };
                Ok(op)
            }
        }
    };
}

impl_read_op!(Abs);
impl_read_op!(Acos);
impl_read_op!(Add);
impl_read_op!(And);
impl_read_op!(ArgMax, attrs_as_arg_max_attrs, reduce_axis);
impl_read_op!(ArgMin, attrs_as_arg_max_attrs, reduce_axis);
impl_read_op!(Asin);
impl_read_op!(Atan);
impl_read_op!(
    AveragePool,
    attrs_as_average_pool_attrs,
    |attrs: sg::AveragePoolAttrs| {
        let kernel_size: SmallVec<_> = attrs.kernel_size().iter().map(|x| x as usize).collect();
        let padding = padding_from_attrs(attrs.auto_pad(), attrs.pads());
        let strides = attrs
            .strides()
            .map(|stride| stride.iter().map(|x| x as usize).collect())
            .unwrap_or(std::iter::repeat(1).take(kernel_size.len()).collect());

        Ok(ops::AveragePool {
            kernel_size,
            padding,
            count_include_pad: attrs.count_include_pad(),
            strides,
            ceil_mode: attrs.ceil_mode(),
        })
    }
);
impl_read_op!(
    BatchNormalization,
    attrs_as_batch_normalization_attrs,
    |attrs: sg::BatchNormalizationAttrs| {
        Ok(ops::BatchNormalization {
            epsilon: attrs.epsilon(),
        })
    }
);
impl_read_op!(Cast, attrs_as_cast_attrs, |attrs: sg::CastAttrs| {
    let to = convert_dtype("to", attrs.to())?;
    Ok(ops::Cast { to })
});
impl_read_op!(
    CastLike,
    attrs_as_cast_like_attrs,
    |_attrs: sg::CastLikeAttrs| { Ok(ops::CastLike {}) }
);
impl_read_op!(Ceil);
impl_read_op!(Clip);
impl_read_op!(Concat, attrs_as_concat_attrs, axis);
impl_read_op!(
    ConcatFromSequence,
    attrs_as_concat_from_sequence_attrs,
    |attrs: sg::ConcatFromSequenceAttrs| {
        let axis = attrs.axis();
        let new_axis = attrs.new_axis();
        Ok(ops::ConcatFromSequence { axis, new_axis })
    }
);
impl_read_op!(Conv, attrs_as_conv_attrs, |attrs: sg::ConvAttrs| {
    let groups = attrs.groups() as usize;
    let padding = padding_from_attrs(attrs.auto_pad(), attrs.pads());
    let strides = vec_from_attr(attrs.strides(), &[1, 1]);
    let dilations = vec_from_attr(attrs.dilations(), &[1, 1]);
    Ok(ops::Conv {
        groups,
        padding,
        strides,
        dilations,
    })
});
impl_read_op!(ConvInteger, attrs_as_conv_attrs, |attrs: sg::ConvAttrs| {
    let groups = attrs.groups() as usize;
    let padding = padding_from_attrs(attrs.auto_pad(), attrs.pads());
    let strides = vec_from_attr(attrs.strides(), &[1, 1]);
    let dilations = vec_from_attr(attrs.dilations(), &[1, 1]);
    Ok(ops::ConvInteger {
        groups,
        padding,
        strides,
        dilations,
    })
});
impl_read_op!(
    ConstantOfShape,
    attrs_as_constant_of_shape_attrs,
    |attrs: sg::ConstantOfShapeAttrs| {
        let value = if let Some(int_val) = attrs.value_as_int_scalar() {
            Scalar::Int(int_val.value())
        } else if let Some(float_val) = attrs.value_as_float_scalar() {
            Scalar::Float(float_val.value())
        } else {
            Scalar::Int(0)
        };
        Ok(ops::ConstantOfShape { value })
    }
);
impl_read_op!(
    ConvTranspose,
    attrs_as_conv_transpose_attrs,
    |attrs: sg::ConvTransposeAttrs| {
        let padding = padding_from_attrs(attrs.auto_pad(), attrs.pads());
        let strides = vec_from_attr(attrs.strides(), &[1, 1]);
        let output_padding = opt_vec_from_attr(attrs.output_padding());
        let groups = attrs.groups() as usize;
        Ok(ops::ConvTranspose {
            padding,
            strides,
            groups,
            output_padding,
        })
    }
);
impl_read_op!(Cos);
impl_read_op!(CumSum);
impl_read_op!(DequantizeLinear, attrs_as_dequantize_linear_attrs, axis);
impl_read_op!(
    DepthToSpace,
    attrs_as_depth_to_space_attrs,
    |attrs: sg::DepthToSpaceAttrs| {
        let mode = match attrs.mode() {
            sg::DepthToSpaceMode::DCR => DepthToSpaceMode::DepthColumnRow,
            sg::DepthToSpaceMode::CRD => DepthToSpaceMode::ColumnRowDepth,
            _ => {
                return Err(ReadOpError::AttrError {
                    attr: "mode",
                    error: "unknown value",
                })?;
            }
        };
        let block_size = attrs.block_size();
        Ok(ops::DepthToSpace { mode, block_size })
    }
);
impl_read_op!(Div);

#[cfg(feature = "random")]
impl_read_op!(
    Dropout,
    attrs_as_dropout_attrs,
    |attrs: sg::DropoutAttrs| { Ok(ops::Dropout { seed: attrs.seed() }) }
);

impl_read_op!(DynamicQuantizeLinear);
impl_read_op!(Einsum, attrs_as_einsum_attrs, |attrs: sg::EinsumAttrs| {
    Ok(ops::Einsum {
        equation: attrs.equation().unwrap_or("").to_string(),
    })
});
impl_read_op!(Elu, attrs_as_elu_attrs, |attrs: sg::EluAttrs| {
    Ok(ops::Elu {
        alpha: attrs.alpha(),
    })
});
impl_read_op!(Equal);
impl_read_op!(Erf);
impl_read_op!(Exp);
impl_read_op!(Expand);
impl_read_op!(
    EyeLike,
    attrs_as_eye_like_attrs,
    |attrs: sg::EyeLikeAttrs| {
        Ok(ops::EyeLike {
            dtype: attrs
                .dtype()
                .map(|dt| convert_dtype("dtype", dt))
                .transpose()?,
            k: attrs.k(),
        })
    }
);
impl_read_op!(Flatten, attrs_as_flatten_attrs, axis);
impl_read_op!(Floor);
impl_read_op!(Gather, attrs_as_gather_attrs, axis);
impl_read_op!(GatherElements, attrs_as_gather_attrs, axis);
impl_read_op!(
    GatherND,
    attrs_as_gather_ndattrs,
    |attrs: sg::GatherNDAttrs| {
        Ok(ops::GatherND {
            batch_dims: attrs.batch_dims() as usize,
        })
    }
);
impl_read_op!(Gelu, attrs_as_gelu_attrs, |attrs: sg::GeluAttrs| {
    let approximate = match attrs.approximate() {
        sg::GeluApproximation::None => false,
        sg::GeluApproximation::Tanh => true,
        _ => {
            return Err(ReadOpError::AttrError {
                attr: "approximate",
                error: "unsupported gelu approximation",
            });
        }
    };
    Ok(ops::Gelu { approximate })
});
impl_read_op!(Gemm, attrs_as_gemm_attrs, |attrs: sg::GemmAttrs| {
    Ok(ops::Gemm {
        alpha: attrs.alpha(),
        beta: attrs.beta(),
        transpose_a: attrs.transpose_a(),
        transpose_b: attrs.transpose_b(),
    })
});
impl_read_op!(GlobalAveragePool);
impl_read_op!(Greater);
impl_read_op!(GreaterOrEqual);
impl_read_op!(
    GridSample,
    attrs_as_grid_sample_attrs,
    |attrs: sg::GridSampleAttrs| {
        Ok(ops::GridSample {
            align_corners: attrs.align_corners(),
        })
    }
);
impl_read_op!(GRU, attrs_as_gruattrs, |attrs: sg::GRUAttrs| {
    let hidden_size = attrs.hidden_size() as usize;
    let direction = match attrs.direction() {
        sg::RNNDirection::Forward => Direction::Forward,
        sg::RNNDirection::Reverse => Direction::Reverse,
        sg::RNNDirection::Bidirectional => Direction::Bidirectional,
        _ => Direction::Forward,
    };

    Ok(ops::GRU {
        direction,
        hidden_size,
        linear_before_reset: attrs.linear_before_reset(),
    })
});
impl_read_op!(
    HardSigmoid,
    attrs_as_hard_sigmoid_attrs,
    |attrs: sg::HardSigmoidAttrs| {
        Ok(ops::HardSigmoid {
            alpha: attrs.alpha(),
            beta: attrs.beta(),
        })
    }
);
impl_read_op!(HardSwish);
impl_read_op!(Identity);

impl ReadOp for ops::If {
    fn op_type() -> sg::OperatorType {
        sg::OperatorType::If
    }

    fn read(op: &sg::OperatorNode, ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
        let attrs = op
            .attrs_as_if_attrs()
            .ok_or(ReadOpError::AttrsMissingError)?;
        let then_branch = ctx.load_graph(attrs.then_branch().ok_or(ReadOpError::AttrError {
            attr: "then",
            error: "missing branch",
        })?)?;
        let else_branch = ctx.load_graph(attrs.else_branch().ok_or(ReadOpError::AttrError {
            attr: "else",
            error: "missing branch",
        })?)?;

        Ok(ops::If {
            then_branch,
            else_branch,
        })
    }
}

impl_read_op!(
    InstanceNormalization,
    attrs_as_batch_normalization_attrs,
    |attrs: sg::BatchNormalizationAttrs| {
        Ok(ops::InstanceNormalization {
            epsilon: Some(attrs.epsilon()),
        })
    }
);
impl_read_op!(IsInf, attrs_as_is_inf_attrs, |_attrs: sg::IsInfAttrs| {
    Ok(ops::IsInf {})
});
impl_read_op!(IsNaN);
impl_read_op!(
    LayerNormalization,
    attrs_as_layer_normalization_attrs,
    |attrs: sg::LayerNormalizationAttrs| {
        Ok(ops::LayerNormalization {
            axis: attrs.axis() as isize,
            epsilon: Some(attrs.epsilon()),
        })
    }
);
impl_read_op!(
    LeakyRelu,
    attrs_as_leaky_relu_attrs,
    |attrs: sg::LeakyReluAttrs| {
        Ok(ops::LeakyRelu {
            alpha: attrs.alpha(),
        })
    }
);
impl_read_op!(Less);
impl_read_op!(LessOrEqual);
impl_read_op!(Log);
impl_read_op!(LogSoftmax, attrs_as_softmax_attrs, axis);

impl ReadOp for ops::Loop {
    fn op_type() -> sg::OperatorType {
        sg::OperatorType::Loop
    }

    fn read(op: &sg::OperatorNode, ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
        let attrs = op
            .attrs_as_loop_attrs()
            .ok_or(ReadOpError::AttrsMissingError)?;
        let body = ctx.load_graph(attrs.body().ok_or(ReadOpError::AttrError {
            attr: "loop",
            error: "missing body",
        })?)?;

        Ok(ops::Loop { body })
    }
}

impl_read_op!(LSTM, attrs_as_lstmattrs, |attrs: sg::LSTMAttrs| {
    let hidden_size = attrs.hidden_size() as usize;
    let direction = match attrs.direction() {
        sg::RNNDirection::Forward => Direction::Forward,
        sg::RNNDirection::Reverse => Direction::Reverse,
        sg::RNNDirection::Bidirectional => Direction::Bidirectional,
        _ => Direction::Forward,
    };
    Ok(ops::LSTM {
        direction,
        hidden_size,
    })
});
impl_read_op!(MatMul);
impl_read_op!(MatMulInteger);
impl_read_op!(Max);
impl_read_op!(
    MaxPool,
    attrs_as_max_pool_attrs,
    |attrs: sg::MaxPoolAttrs| {
        let kernel_size: SmallVec<_> = attrs.kernel_size().iter().map(|x| x as usize).collect();
        let padding = padding_from_attrs(attrs.auto_pad(), attrs.pads());
        let strides = attrs
            .strides()
            .map(|stride| stride.iter().map(|x| x as usize).collect())
            .unwrap_or(std::iter::repeat(1).take(kernel_size.len()).collect());

        Ok(ops::MaxPool {
            kernel_size,
            padding,
            strides,
            ceil_mode: attrs.ceil_mode(),
        })
    }
);
impl_read_op!(Mean);
impl_read_op!(Min);
impl_read_op!(Mod, attrs_as_mod_attrs, |attrs: sg::ModAttrs| {
    Ok(ops::Mod { fmod: attrs.fmod() })
});
impl_read_op!(Mul);
impl_read_op!(Neg);
impl_read_op!(
    NonMaxSuppression,
    attrs_as_non_max_suppression_attrs,
    |attrs: sg::NonMaxSuppressionAttrs| {
        let box_order = match attrs.box_order() {
            sg::NMSBoxOrder::CenterWidthHeight => BoxOrder::CenterWidthHeight,
            sg::NMSBoxOrder::TopLeftBottomRight => BoxOrder::TopLeftBottomRight,
            _ => BoxOrder::TopLeftBottomRight,
        };
        Ok(ops::NonMaxSuppression { box_order })
    }
);
impl_read_op!(NonZero);
impl_read_op!(Not);
impl_read_op!(OneHot, attrs_as_one_hot_attrs, axis);
impl_read_op!(Or);

impl ReadOp for ops::Pad {
    fn op_type() -> sg::OperatorType {
        sg::OperatorType::Pad
    }

    fn read(op: &sg::OperatorNode, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
        // Pad attributes are optional for backwards compatibility.
        let attrs = op.attrs_as_pad_attrs();
        let mode = match attrs.map(|a| a.mode()).unwrap_or(sg::PadMode::Constant) {
            sg::PadMode::Constant => PadMode::Constant,
            sg::PadMode::Reflect => PadMode::Reflect,
            sg::PadMode::Edge => PadMode::Edge,
            sg::PadMode::Wrap => PadMode::Wrap,
            _ => {
                return Err(ReadOpError::AttrError {
                    attr: "mode",
                    error: "unknown value",
                });
            }
        };
        Ok(ops::Pad { mode })
    }
}

impl_read_op!(Pow);
impl_read_op!(PRelu);

impl_read_op!(
    QuantizeLinear,
    attrs_as_quantize_linear_attrs,
    |attrs: sg::QuantizeLinearAttrs| {
        let output_dtype = attrs
            .output_dtype()
            .map(|dtype| convert_dtype("output_dtype", dtype))
            .transpose()?;
        Ok(ops::QuantizeLinear {
            axis: attrs.axis() as isize,
            output_dtype,
        })
    }
);

#[cfg(feature = "random")]
impl_read_op!(
    RandomNormal,
    attrs_as_random_normal_attrs,
    |attrs: sg::RandomNormalAttrs| {
        let shape = attrs
            .shape()
            .map(|shape| shape.iter().map(|size| size as usize).collect())
            .unwrap_or_default();

        Ok(ops::RandomNormal {
            shape,
            mean: attrs.mean(),
            scale: attrs.scale(),
            seed: attrs.seed(),
        })
    }
);
#[cfg(feature = "random")]
impl_read_op!(
    RandomNormalLike,
    attrs_as_random_normal_like_attrs,
    |attrs: sg::RandomNormalLikeAttrs| {
        Ok(ops::RandomNormalLike {
            mean: attrs.mean(),
            scale: attrs.scale(),
            seed: attrs.seed(),
        })
    }
);
#[cfg(feature = "random")]
impl_read_op!(
    RandomUniform,
    attrs_as_random_uniform_attrs,
    |attrs: sg::RandomUniformAttrs| {
        let shape = attrs
            .shape()
            .map(|shape| shape.iter().map(|size| size as usize).collect())
            .unwrap_or_default();

        Ok(ops::RandomUniform {
            shape,
            high: attrs.high(),
            low: attrs.low(),
            seed: attrs.seed(),
        })
    }
);
#[cfg(feature = "random")]
impl_read_op!(
    RandomUniformLike,
    attrs_as_random_uniform_like_attrs,
    |attrs: sg::RandomUniformLikeAttrs| {
        Ok(ops::RandomUniformLike {
            high: attrs.high(),
            low: attrs.low(),
            seed: attrs.seed(),
        })
    }
);

impl_read_op!(Range);
impl_read_op!(Reciprocal);
impl_read_op!(ReduceL2, attrs_as_reduce_mean_attrs, reduce_axes);
impl_read_op!(ReduceMax, attrs_as_reduce_mean_attrs, reduce_axes);
impl_read_op!(ReduceMean, attrs_as_reduce_mean_attrs, reduce_axes);
impl_read_op!(ReduceMin, attrs_as_reduce_mean_attrs, reduce_axes);
impl_read_op!(ReduceProd, attrs_as_reduce_mean_attrs, reduce_axes);
impl_read_op!(ReduceSum, attrs_as_reduce_mean_attrs, reduce_axes);
impl_read_op!(ReduceSumSquare, attrs_as_reduce_mean_attrs, reduce_axes);
impl_read_op!(Relu);
impl_read_op!(
    Reshape,
    attrs_as_reshape_attrs,
    |attrs: sg::ReshapeAttrs| {
        Ok(ops::Reshape {
            allow_zero: attrs.allow_zero(),
        })
    }
);
impl_read_op!(Resize, attrs_as_resize_attrs, |attrs: sg::ResizeAttrs| {
    let mode = match attrs.mode() {
        sg::ResizeMode::Nearest => ResizeMode::Nearest,
        sg::ResizeMode::Linear => ResizeMode::Linear,
        _ => ResizeMode::Nearest,
    };
    let nearest_mode = match attrs.nearest_mode() {
        sg::NearestMode::Floor => NearestMode::Floor,
        sg::NearestMode::Ceil => NearestMode::Ceil,
        sg::NearestMode::RoundPreferFloor => NearestMode::RoundPreferFloor,
        sg::NearestMode::RoundPreferCeil => NearestMode::RoundPreferCeil,
        _ => NearestMode::default(),
    };

    let coord_mode = match attrs.coord_mode() {
        sg::CoordTransformMode::Asymmetric => CoordTransformMode::Asymmetric,
        sg::CoordTransformMode::HalfPixel => CoordTransformMode::HalfPixel,
        sg::CoordTransformMode::AlignCorners => CoordTransformMode::AlignCorners,
        sg::CoordTransformMode::PytorchHalfPixel => CoordTransformMode::PytorchHalfPixel,
        _ => CoordTransformMode::default(),
    };

    Ok(ops::Resize {
        mode,
        coord_mode,
        nearest_mode,
    })
});
impl_read_op!(Round);
impl_read_op!(
    ScatterElements,
    attrs_as_scatter_elements_attrs,
    |attrs: sg::ScatterElementsAttrs| {
        Ok(ops::ScatterElements {
            axis: attrs.axis() as isize,
            reduction: convert_reduction("reduction", attrs.reduction())?,
        })
    }
);
impl_read_op!(
    ScatterND,
    attrs_as_scatter_ndattrs,
    |attrs: sg::ScatterNDAttrs| {
        Ok(ops::ScatterND {
            reduction: convert_reduction("reduction", attrs.reduction())?,
        })
    }
);

impl_read_op!(SequenceAt);
impl_read_op!(SequenceConstruct);
impl_read_op!(
    SequenceEmpty,
    attrs_as_sequence_empty_attrs,
    |attrs: sg::SequenceEmptyAttrs| {
        let dtype = attrs
            .dtype()
            .map(|dtype| convert_dtype("dtype", dtype))
            .transpose()?;
        Ok(ops::SequenceEmpty { dtype })
    }
);
impl_read_op!(SequenceErase);
impl_read_op!(SequenceInsert);
impl_read_op!(SequenceLength);

impl ReadOp for ops::Shape {
    fn op_type() -> sg::OperatorType {
        sg::OperatorType::Shape
    }

    fn read(op: &sg::OperatorNode, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
        // Shape attributes are optional for backwards compatibility
        let attrs = op.attrs_as_shape_attrs();
        let start = attrs.and_then(|a| a.start());
        let end = attrs.and_then(|a| a.end());
        Ok(ops::Shape { start, end })
    }
}

impl_read_op!(Sigmoid);
impl_read_op!(Sign);
impl_read_op!(Sin);
impl_read_op!(Size);
impl_read_op!(Slice);
impl_read_op!(Softmax, attrs_as_softmax_attrs, axis);
impl_read_op!(Softplus);
impl_read_op!(Split, attrs_as_split_attrs, |attrs: sg::SplitAttrs| {
    let axis = attrs.axis() as isize;
    let num_outputs = attrs.num_outputs().map(|n| n as u32);
    Ok(ops::Split { axis, num_outputs })
});
impl_read_op!(
    SplitToSequence,
    attrs_as_split_to_sequence_attrs,
    |attrs: sg::SplitToSequenceAttrs| {
        Ok(ops::SplitToSequence {
            axis: attrs.axis(),
            keep_dims: attrs.keep_dims(),
        })
    }
);
impl_read_op!(Sqrt);
impl_read_op!(Squeeze);

#[cfg(feature = "fft")]
impl_read_op!(STFT, attrs_as_stftattrs, |attrs: sg::STFTAttrs| {
    Ok(ops::STFT {
        onesided: attrs.onesided(),
    })
});

impl_read_op!(Sub);
impl_read_op!(Sum);
impl_read_op!(Tan);
impl_read_op!(Tanh);
impl_read_op!(Tile);
impl_read_op!(TopK, attrs_as_top_kattrs, |attrs: sg::TopKAttrs| {
    let largest = attrs.largest();
    let sorted = attrs.sorted();
    let axis = attrs.axis();
    Ok(ops::TopK {
        axis: Some(axis as isize),
        largest,
        sorted,
    })
});
impl_read_op!(
    Transpose,
    attrs_as_transpose_attrs,
    |attrs: sg::TransposeAttrs| {
        let perm = attrs
            .perm()
            .map(|perm| perm.iter().map(|dim| dim as usize).collect());
        Ok(ops::Transpose { perm })
    }
);
impl_read_op!(Trilu, attrs_as_trilu_attrs, |attrs: sg::TriluAttrs| {
    Ok(ops::Trilu {
        upper: attrs.upper(),
    })
});
impl_read_op!(Unsqueeze);
impl_read_op!(Where);
impl_read_op!(Xor);

#[cfg(test)]
mod tests {
    // See `test_all_op_types` in model.rs for a test that exercises
    // deserialization of all operators.
}
