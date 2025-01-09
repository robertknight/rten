use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, Vector, WIPOffset};
use rten_tensor::prelude::*;
use rten_tensor::TensorView;

use crate::graph::{Dimension, NodeId};
use crate::header::Header;
use crate::number::LeBytes;
use crate::ops::{
    ArgMax, ArgMin, AveragePool, BatchNormalization, BoxOrder, Cast, Concat, ConstantOfShape, Conv,
    ConvTranspose, CoordTransformMode, DataType, DepthToSpace, DepthToSpaceMode, DequantizeLinear,
    Einsum, Elu, Flatten, Gather, GatherElements, GatherND, Gelu, Gemm, HardSigmoid,
    InstanceNormalization, LayerNormalization, LeakyRelu, LogSoftmax, MaxPool, Mod, NearestMode,
    NonMaxSuppression, OneHot, Padding, QuantizeLinear, ReduceMax, ReduceMean, ReduceMin,
    ReduceProd, ReduceSum, ReduceSumSquare, Reshape, Resize, ResizeMode, Scalar, ScatterElements,
    ScatterReduction, Softmax, Split, TopK, Transpose, Trilu,
};
use crate::schema_generated as sg;

#[cfg(feature = "random")]
use crate::ops::{RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike};

/// Struct like `crate::ops::If` with subgraph attributes replaced by
/// pre-serialized graphs.
pub struct IfArgs<'a> {
    pub then_branch: WIPOffset<sg::Graph<'a>>,
    pub else_branch: WIPOffset<sg::Graph<'a>>,
}

/// Enum of all the built-in operators
pub enum OpType<'a> {
    Abs,
    Acos,
    Add,
    And,
    ArgMax(ArgMax),
    ArgMin(ArgMin),
    Asin,
    Atan,
    AveragePool(AveragePool),
    BatchNormalization(BatchNormalization),
    Cast(Cast),
    Ceil,
    Clip,
    Concat(Concat),
    ConstantOfShape(ConstantOfShape),
    Conv(Conv),
    ConvTranspose(ConvTranspose),
    Cos,
    DequantizeLinear(DequantizeLinear),
    DepthToSpace(DepthToSpace),
    Div,
    DynamicQuantizeLinear,
    Einsum(Einsum),
    Elu(Elu),
    Equal,
    Erf,
    Exp,
    Expand,
    Flatten(Flatten),
    Floor,
    Gather(Gather),
    GatherElements(GatherElements),
    GatherND(GatherND),
    Gelu(Gelu),
    Gemm(Gemm),
    GlobalAveragePool,
    Greater,
    GreaterOrEqual,
    HardSigmoid(HardSigmoid),
    HardSwish,
    Identity,
    If(IfArgs<'a>),
    InstanceNormalization(InstanceNormalization),
    LayerNormalization(LayerNormalization),
    LeakyRelu(LeakyRelu),
    Less,
    LessOrEqual,
    Log,
    LogSoftmax(LogSoftmax),
    MatMul,
    MatMulInteger,
    Max,
    MaxPool(MaxPool),
    Mean,
    Min,
    Mod(Mod),
    Mul,
    Neg,
    NonMaxSuppression(NonMaxSuppression),
    NonZero,
    Not,
    OneHot(OneHot),
    Or,
    Pad,
    Pow,

    #[cfg(feature = "random")]
    RandomNormal(RandomNormal),
    #[cfg(feature = "random")]
    RandomNormalLike(RandomNormalLike),
    #[cfg(feature = "random")]
    RandomUniform(RandomUniform),
    #[cfg(feature = "random")]
    RandomUniformLike(RandomUniformLike),

    Range,
    Reciprocal,
    ReduceMax(ReduceMax),
    ReduceMean(ReduceMean),
    ReduceMin(ReduceMin),
    ReduceProd(ReduceProd),
    ReduceSum(ReduceSum),
    ReduceSumSquare(ReduceSumSquare),
    Relu,
    Reshape(Reshape),
    Resize(Resize),
    Round,
    QuantizeLinear(QuantizeLinear),
    ScatterElements(ScatterElements),
    Shape,
    Sigmoid,
    Sign,
    Sin,
    Size,
    Slice,
    Softmax(Softmax),
    Softplus,
    Split(Split),
    Sqrt,
    Squeeze,
    Sub,
    Sum,
    Tan,
    Tanh,
    Tile,
    TopK(TopK),
    Transpose(Transpose),
    Trilu(Trilu),
    Unsqueeze,
    Where,
    Xor,
}

/// Specifies which version of the model format to generate.
pub enum ModelFormat {
    /// Generate the V1 format. This consists of just FlatBuffers data
    /// containing both the model structure and tensor data.
    V1,

    /// Generate the V2 format, which consists of a header, the FlatBuffers
    /// data containing the model structure and a tensor data segment.
    V2,
}

/// Helpers for converting tensors to constant node data.
pub trait ToConstantData: Sized {
    /// Return the data type identifier for this type.
    fn dtype() -> sg::ConstantDataType;

    /// Create a `ConstantData` union value that stores the tensor data in the
    /// model buffer.
    fn create_inline_data(
        builder: &mut FlatBufferBuilder,
        data: &[Self],
    ) -> (sg::ConstantData, WIPOffset<UnionWIPOffset>);
}

macro_rules! impl_to_constant_data {
    ($type:ty, $dtype:ident, $inline_union_type:ident, $inline_args:ident) => {
        impl ToConstantData for $type {
            fn dtype() -> sg::ConstantDataType {
                sg::ConstantDataType::$dtype
            }

            fn create_inline_data(
                builder: &mut FlatBufferBuilder<'_>,
                data: &[Self],
            ) -> (sg::ConstantData, WIPOffset<UnionWIPOffset>) {
                let data_vec = builder.create_vector(data);
                let data = sg::$inline_union_type::create(
                    builder,
                    &sg::$inline_args {
                        data: Some(data_vec),
                    },
                )
                .as_union_value();
                (sg::ConstantData::$inline_union_type, data)
            }
        }
    };
}

impl_to_constant_data!(f32, Float32, FloatData, FloatDataArgs);
impl_to_constant_data!(i32, Int32, Int32Data, Int32DataArgs);
impl_to_constant_data!(u8, UInt8, UInt8Data, UInt8DataArgs);
impl_to_constant_data!(i8, Int8, Int8Data, Int8DataArgs);

enum NodeData<'a> {
    Constant(WIPOffset<sg::ConstantNode<'a>>),
    Value(WIPOffset<sg::ValueNode<'a>>),
    Operator(WIPOffset<sg::OperatorNode<'a>>),
}

/// Arguments for [`ModelBuilder::add_metadata`].
pub struct MetadataArgs {
    pub onnx_hash: Option<String>,
}

struct PadArgs {
    auto_pad: sg::AutoPad,
    pads: Option<Vec<usize>>,
}

fn pad_args_from_padding(padding: Padding) -> PadArgs {
    match padding {
        Padding::Same => PadArgs {
            auto_pad: sg::AutoPad::Same,
            pads: None,
        },
        Padding::Fixed(pads) => PadArgs {
            auto_pad: sg::AutoPad::NotSet,
            pads: Some(pads.iter().copied().collect()),
        },
    }
}

fn convert_dtype(dtype: DataType) -> sg::DataType {
    match dtype {
        DataType::Int32 => sg::DataType::Int32,
        DataType::Float => sg::DataType::Float,
        DataType::Int8 => sg::DataType::Int8,
        DataType::UInt8 => sg::DataType::UInt8,
    }
}

/// Builder for serializing a graph or subgraph to FlatBuffers.
pub struct GraphBuilder<'mb, 'a> {
    builder: &'mb mut FlatBufferBuilder<'a>,
    tensor_data_builder: Option<&'mb mut TensorDataBuilder>,

    nodes: Vec<WIPOffset<sg::Node<'a>>>,
    input_ids: Vec<NodeId>,
    output_ids: Vec<NodeId>,
}

impl<'mb, 'a> GraphBuilder<'mb, 'a> {
    fn new(
        builder: &'mb mut FlatBufferBuilder<'a>,
        tensor_data_builder: Option<&'mb mut TensorDataBuilder>,
    ) -> GraphBuilder<'mb, 'a> {
        GraphBuilder {
            builder,
            tensor_data_builder,

            nodes: Vec::new(),
            input_ids: Vec::new(),
            output_ids: Vec::new(),
        }
    }

    fn add_node(&mut self, name: Option<&str>, data: NodeData) -> NodeId {
        let (data_type, union_val) = match data {
            NodeData::Constant(offset) => (sg::NodeKind::ConstantNode, offset.as_union_value()),
            NodeData::Value(offset) => (sg::NodeKind::ValueNode, offset.as_union_value()),
            NodeData::Operator(offset) => (sg::NodeKind::OperatorNode, offset.as_union_value()),
        };
        let args = sg::NodeArgs {
            name: name.map(|x| self.builder.create_string(x)),
            data_type,
            data: Some(union_val),
        };
        let node = sg::Node::create(self.builder, &args);
        self.nodes.push(node);
        NodeId::from_u32((self.nodes.len() - 1) as u32)
    }

    /// Return a graph builder for a subgraph.
    pub fn subgraph_builder(&mut self) -> GraphBuilder<'_, 'a> {
        GraphBuilder::new(
            self.builder,
            if let Some(tdb) = self.tensor_data_builder.as_mut() {
                Some(*tdb)
            } else {
                None
            },
        )
    }

    /// Add a constant node (eg. weights, biases) to the model
    pub fn add_constant<T: Copy + LeBytes + ToConstantData>(
        &mut self,
        input: TensorView<T>,
    ) -> NodeId {
        let shape: Vec<u32> = input.shape().iter().map(|&x| x as u32).collect();
        let shape_vec = self.builder.create_vector(&shape[..]);
        let dtype = <T as ToConstantData>::dtype();

        let elts: Vec<T> = input.to_vec();
        let args: sg::ConstantNodeArgs = if let Some(tdb) = self.tensor_data_builder.as_mut() {
            let offset = tdb.add_tensor(&elts) as u64;

            sg::ConstantNodeArgs {
                shape: Some(shape_vec),
                strides: None,
                data_type: sg::ConstantData::NONE,
                data: None,
                data_offset: Some(offset),
                dtype: Some(dtype),
            }
        } else {
            let (inline_dtype, data) =
                <T as ToConstantData>::create_inline_data(self.builder, &elts);

            sg::ConstantNodeArgs {
                shape: Some(shape_vec),
                strides: None,
                data_type: inline_dtype,
                data: Some(data),
                data_offset: None,
                dtype: Some(dtype),
            }
        };

        let const_node = sg::ConstantNode::create(self.builder, &args);
        self.add_node(None, NodeData::Constant(const_node))
    }

    /// Add a value node to the model
    pub fn add_value(
        &mut self,
        id: &str,
        shape: Option<&[Dimension]>,
        dtype: Option<DataType>,
    ) -> NodeId {
        let shape = shape.map(|shape| {
            let dim_vec: Vec<_> = shape
                .iter()
                .map(|dim| match dim {
                    Dimension::Fixed(value) => sg::Dim::create(
                        self.builder,
                        &sg::DimArgs {
                            name: None,
                            value: *value as u32,
                        },
                    ),
                    Dimension::Symbolic(name) => {
                        let name_offset = self.builder.create_string(name);
                        sg::Dim::create(
                            self.builder,
                            &sg::DimArgs {
                                name: Some(name_offset),
                                value: 0,
                            },
                        )
                    }
                })
                .collect();
            self.builder.create_vector(&dim_vec[..])
        });
        let dtype = dtype.map(convert_dtype);
        let value_node = sg::ValueNode::create(self.builder, &sg::ValueNodeArgs { shape, dtype });
        self.add_node(Some(id), NodeData::Value(value_node))
    }

    /// Add an operator node to the model
    pub fn add_operator(
        &mut self,
        id: &str,
        op_info: OpType,
        inputs: &[Option<NodeId>],
        outputs: &[NodeId],
    ) -> NodeId {
        // Generate an (op_type, attr_type, attrs) tuple for an operator with
        // no attributes.
        macro_rules! op {
            ($op_name:ident) => {
                (sg::OperatorType::$op_name, sg::OperatorAttrs::NONE, None)
            };
        }

        /// Generate an (op_type, attr_type, attrs) tuple for an operator with
        /// attributes.
        macro_rules! op_with_attrs {
            ($op_name:ident, $attr_type:ident, $args: expr) => {{
                let args = ($args);
                let attrs = sg::$attr_type::create(self.builder, &args).as_union_value();
                (
                    sg::OperatorType::$op_name,
                    sg::OperatorAttrs::$attr_type,
                    Some(attrs),
                )
            }};
        }

        macro_rules! reduce_attrs {
            ($args:expr) => {{
                let axes = self.create_vec($args.axes, |axis| axis);
                sg::ReduceMeanAttrsArgs {
                    axes,
                    keep_dims: $args.keep_dims,
                }
            }};
        }

        // Convert internal operator and attribute types to corresponding
        // FlatBuffers types, and write attribute data into buffer.
        let (op_type, attrs_type, attrs) = match op_info {
            OpType::Abs => op!(Abs),
            OpType::Acos => op!(Acos),
            OpType::Add => op!(Add),
            OpType::And => op!(And),
            OpType::ArgMax(args) => op_with_attrs!(ArgMax, ArgMaxAttrs, {
                sg::ArgMaxAttrsArgs {
                    axis: args.axis as i32,
                    keep_dims: args.keep_dims,
                }
            }),
            OpType::ArgMin(args) => op_with_attrs!(ArgMin, ArgMaxAttrs, {
                sg::ArgMaxAttrsArgs {
                    axis: args.axis as i32,
                    keep_dims: args.keep_dims,
                }
            }),
            OpType::Asin => op!(Asin),
            OpType::Atan => op!(Atan),
            OpType::AveragePool(args) => op_with_attrs!(AveragePool, AveragePoolAttrs, {
                let pad_args = pad_args_from_padding(args.padding);
                let pads = self.create_vec(pad_args.pads, |pad| pad as u32);
                let kernel_size = self.create_vec(Some(args.kernel_size.into()), |sz| sz as u32);
                let strides = self.create_vec(Some(args.strides.into()), |s| s as u32);
                sg::AveragePoolAttrsArgs {
                    kernel_size,
                    auto_pad: pad_args.auto_pad,
                    pads,
                    strides,
                    count_include_pad: args.count_include_pad,
                }
            }),
            OpType::BatchNormalization(args) => op_with_attrs!(
                BatchNormalization,
                BatchNormalizationAttrs,
                sg::BatchNormalizationAttrsArgs {
                    epsilon: args.epsilon
                }
            ),
            OpType::Cast(args) => op_with_attrs!(
                Cast,
                CastAttrs,
                sg::CastAttrsArgs {
                    to: convert_dtype(args.to),
                }
            ),
            OpType::Ceil => op!(Ceil),
            OpType::Clip => op!(Clip),
            OpType::Concat(args) => op_with_attrs!(
                Concat,
                ConcatAttrs,
                sg::ConcatAttrsArgs {
                    axis: args.axis as i32,
                }
            ),
            OpType::ConstantOfShape(args) => {
                op_with_attrs!(ConstantOfShape, ConstantOfShapeAttrs, {
                    match args.value {
                        Scalar::Int(int_value) => sg::ConstantOfShapeAttrsArgs {
                            value_type: sg::Scalar::IntScalar,
                            value: Some(
                                sg::IntScalar::create(
                                    self.builder,
                                    &sg::IntScalarArgs { value: int_value },
                                )
                                .as_union_value(),
                            ),
                        },
                        Scalar::Float(float_value) => sg::ConstantOfShapeAttrsArgs {
                            value_type: sg::Scalar::FloatScalar,
                            value: Some(
                                sg::FloatScalar::create(
                                    self.builder,
                                    &sg::FloatScalarArgs { value: float_value },
                                )
                                .as_union_value(),
                            ),
                        },
                    }
                })
            }
            OpType::Conv(args) => op_with_attrs!(Conv, ConvAttrs, {
                let pad_args = pad_args_from_padding(args.padding);
                let pads = self.create_vec(pad_args.pads, |pad| pad as u32);
                let dilations = self.create_vec(Some(args.dilations), |d| d as u32);
                let strides = self.create_vec(Some(args.strides), |s| s as u32);

                sg::ConvAttrsArgs {
                    dilations,
                    groups: args.groups as u32,
                    auto_pad: pad_args.auto_pad,
                    pads,
                    strides,
                }
            }),
            OpType::ConvTranspose(args) => op_with_attrs!(ConvTranspose, ConvTransposeAttrs, {
                let pad_args = pad_args_from_padding(args.padding);
                let pads = self.create_vec(pad_args.pads, |pad| pad as u32);
                let strides = self.create_vec(Some(args.strides), |s| s as u32);
                sg::ConvTransposeAttrsArgs {
                    strides,
                    auto_pad: pad_args.auto_pad,
                    pads,
                }
            }),
            OpType::Cos => op!(Cos),
            OpType::DequantizeLinear(args) => op_with_attrs!(
                DequantizeLinear,
                DequantizeLinearAttrs,
                sg::DequantizeLinearAttrsArgs {
                    axis: args.axis as i32,
                }
            ),
            OpType::DepthToSpace(args) => op_with_attrs!(
                DepthToSpace,
                DepthToSpaceAttrs,
                sg::DepthToSpaceAttrsArgs {
                    block_size: args.block_size,
                    mode: match args.mode {
                        DepthToSpaceMode::DepthColumnRow => sg::DepthToSpaceMode::DCR,
                        DepthToSpaceMode::ColumnRowDepth => sg::DepthToSpaceMode::CRD,
                    }
                }
            ),
            OpType::Div => op!(Div),
            OpType::DynamicQuantizeLinear => op!(DynamicQuantizeLinear),
            OpType::Einsum(args) => {
                let equation = self.builder.create_string(&args.equation);
                op_with_attrs!(
                    Einsum,
                    EinsumAttrs,
                    sg::EinsumAttrsArgs {
                        equation: Some(equation)
                    }
                )
            }
            OpType::Elu(args) => {
                op_with_attrs!(Elu, EluAttrs, sg::EluAttrsArgs { alpha: args.alpha })
            }
            OpType::Equal => op!(Equal),
            OpType::Erf => op!(Erf),
            OpType::Exp => op!(Exp),
            OpType::Expand => op!(Expand),
            OpType::Flatten(args) => op_with_attrs!(
                Flatten,
                FlattenAttrs,
                sg::FlattenAttrsArgs {
                    axis: args.axis as i32,
                }
            ),
            OpType::Floor => op!(Floor),
            OpType::Gather(args) => op_with_attrs!(
                Gather,
                GatherAttrs,
                sg::GatherAttrsArgs {
                    axis: args.axis as i32,
                }
            ),
            OpType::GatherElements(args) => op_with_attrs!(
                GatherElements,
                GatherAttrs,
                sg::GatherAttrsArgs {
                    axis: args.axis as i32,
                }
            ),
            OpType::GatherND(args) => op_with_attrs!(
                GatherND,
                GatherNDAttrs,
                sg::GatherNDAttrsArgs {
                    batch_dims: args.batch_dims as i32,
                }
            ),
            OpType::Gelu(_args) => op_with_attrs!(Gelu, GeluAttrs, sg::GeluAttrsArgs {}),
            OpType::Gemm(args) => op_with_attrs!(
                Gemm,
                GemmAttrs,
                sg::GemmAttrsArgs {
                    alpha: args.alpha,
                    beta: args.beta,
                    transpose_a: args.transpose_a,
                    transpose_b: args.transpose_b,
                }
            ),
            OpType::GlobalAveragePool => op!(GlobalAveragePool),
            OpType::Greater => op!(Greater),
            OpType::GreaterOrEqual => op!(GreaterOrEqual),
            OpType::HardSigmoid(args) => op_with_attrs!(
                HardSigmoid,
                HardSigmoidAttrs,
                sg::HardSigmoidAttrsArgs {
                    alpha: args.alpha,
                    beta: args.beta
                }
            ),
            OpType::HardSwish => op!(HardSwish),
            OpType::Identity => op!(Identity),
            OpType::If(args) => op_with_attrs!(
                If,
                IfAttrs,
                sg::IfAttrsArgs {
                    then_branch: Some(args.then_branch),
                    else_branch: Some(args.else_branch),
                }
            ),
            OpType::InstanceNormalization(args) => op_with_attrs!(
                InstanceNormalization,
                BatchNormalizationAttrs,
                sg::BatchNormalizationAttrsArgs {
                    epsilon: args.epsilon.unwrap_or(1e-5)
                }
            ),
            OpType::LayerNormalization(args) => op_with_attrs!(
                LayerNormalization,
                LayerNormalizationAttrs,
                sg::LayerNormalizationAttrsArgs {
                    axis: args.axis as i32,
                    epsilon: args.epsilon.unwrap_or(1e-5)
                }
            ),
            OpType::LeakyRelu(args) => op_with_attrs!(
                LeakyRelu,
                LeakyReluAttrs,
                sg::LeakyReluAttrsArgs { alpha: args.alpha }
            ),
            OpType::Less => op!(Less),
            OpType::LessOrEqual => op!(LessOrEqual),
            OpType::Log => op!(Log),
            OpType::LogSoftmax(args) => op_with_attrs!(
                LogSoftmax,
                SoftmaxAttrs,
                sg::SoftmaxAttrsArgs {
                    axis: args.axis as i32,
                }
            ),
            OpType::MatMul => op!(MatMul),
            OpType::MatMulInteger => op!(MatMulInteger),
            OpType::Max => op!(Max),
            OpType::MaxPool(args) => op_with_attrs!(MaxPool, MaxPoolAttrs, {
                let pad_args = pad_args_from_padding(args.padding);
                let pads = self.create_vec(pad_args.pads, |pad| pad as u32);
                let kernel_size = self.create_vec(Some(args.kernel_size.into()), |sz| sz as u32);
                let strides = self.create_vec(Some(args.strides.into()), |s| s as u32);
                sg::MaxPoolAttrsArgs {
                    kernel_size,
                    auto_pad: pad_args.auto_pad,
                    pads,
                    strides,
                }
            }),
            OpType::Mean => op!(Mean),
            OpType::Min => op!(Min),
            OpType::Mod(args) => {
                op_with_attrs!(Mod, ModAttrs, sg::ModAttrsArgs { fmod: args.fmod })
            }
            OpType::Mul => op!(Mul),
            OpType::Neg => op!(Neg),
            OpType::NonMaxSuppression(args) => {
                op_with_attrs!(
                    NonMaxSuppression,
                    NonMaxSuppressionAttrs,
                    sg::NonMaxSuppressionAttrsArgs {
                        box_order: match args.box_order {
                            BoxOrder::TopLeftBottomRight => sg::NMSBoxOrder::TopLeftBottomRight,
                            BoxOrder::CenterWidthHeight => sg::NMSBoxOrder::CenterWidthHeight,
                        }
                    }
                )
            }
            OpType::NonZero => op!(NonZero),
            OpType::Not => op!(Not),
            OpType::Or => op!(Or),
            OpType::OneHot(args) => {
                op_with_attrs!(
                    OneHot,
                    OneHotAttrs,
                    sg::OneHotAttrsArgs {
                        axis: args.axis as i32
                    }
                )
            }
            OpType::Pad => op!(Pad),
            OpType::Pow => op!(Pow),

            OpType::QuantizeLinear(args) => op_with_attrs!(
                QuantizeLinear,
                QuantizeLinearAttrs,
                sg::QuantizeLinearAttrsArgs {
                    axis: args.axis as i32,
                    output_dtype: None, // Not yet implemented
                }
            ),

            #[cfg(feature = "random")]
            OpType::RandomNormal(args) => {
                let shape = self.create_vec(Some(args.shape), |size| size as u32);
                op_with_attrs!(RandomNormal, RandomNormalAttrs, {
                    sg::RandomNormalAttrsArgs {
                        mean: args.mean,
                        scale: args.scale,
                        seed: args.seed,
                        shape,
                    }
                })
            }

            #[cfg(feature = "random")]
            OpType::RandomNormalLike(args) => {
                op_with_attrs!(RandomNormalLike, RandomNormalLikeAttrs, {
                    sg::RandomNormalLikeAttrsArgs {
                        mean: args.mean,
                        scale: args.scale,
                        seed: args.seed,
                    }
                })
            }

            #[cfg(feature = "random")]
            OpType::RandomUniform(args) => {
                let shape = self.create_vec(Some(args.shape), |size| size as u32);
                op_with_attrs!(RandomUniform, RandomUniformAttrs, {
                    sg::RandomUniformAttrsArgs {
                        high: args.high,
                        low: args.low,
                        seed: args.seed,
                        shape,
                    }
                })
            }

            #[cfg(feature = "random")]
            OpType::RandomUniformLike(args) => {
                op_with_attrs!(RandomUniformLike, RandomUniformLikeAttrs, {
                    sg::RandomUniformLikeAttrsArgs {
                        high: args.high,
                        low: args.low,
                        seed: args.seed,
                    }
                })
            }

            OpType::Range => op!(Range),
            OpType::Reciprocal => op!(Reciprocal),
            OpType::ReduceMax(args) => {
                op_with_attrs!(ReduceMax, ReduceMeanAttrs, reduce_attrs!(args))
            }
            OpType::ReduceMean(args) => {
                op_with_attrs!(ReduceMean, ReduceMeanAttrs, reduce_attrs!(args))
            }
            OpType::ReduceMin(args) => {
                op_with_attrs!(ReduceMin, ReduceMeanAttrs, reduce_attrs!(args))
            }
            OpType::ReduceProd(args) => {
                op_with_attrs!(ReduceProd, ReduceMeanAttrs, reduce_attrs!(args))
            }
            OpType::ReduceSum(args) => {
                op_with_attrs!(ReduceSum, ReduceMeanAttrs, reduce_attrs!(args))
            }
            OpType::ReduceSumSquare(args) => {
                op_with_attrs!(ReduceSumSquare, ReduceMeanAttrs, reduce_attrs!(args))
            }
            OpType::Relu => op!(Relu),
            OpType::Reshape(args) => op_with_attrs!(Reshape, ReshapeAttrs, {
                sg::ReshapeAttrsArgs {
                    allow_zero: args.allow_zero,
                }
            }),
            OpType::Resize(args) => op_with_attrs!(Resize, ResizeAttrs, {
                let mode = match args.mode {
                    ResizeMode::Nearest => sg::ResizeMode::Nearest,
                    ResizeMode::Linear => sg::ResizeMode::Linear,
                };
                let coord_mode = match args.coord_mode {
                    CoordTransformMode::Asymmetric => sg::CoordTransformMode::Asymmetric,
                    CoordTransformMode::HalfPixel => sg::CoordTransformMode::HalfPixel,
                    CoordTransformMode::AlignCorners => sg::CoordTransformMode::AlignCorners,
                };
                let nearest_mode = match args.nearest_mode {
                    NearestMode::Ceil => sg::NearestMode::Ceil,
                    NearestMode::Floor => sg::NearestMode::Floor,
                    NearestMode::RoundPreferCeil => sg::NearestMode::RoundPreferCeil,
                    NearestMode::RoundPreferFloor => sg::NearestMode::RoundPreferFloor,
                };
                sg::ResizeAttrsArgs {
                    mode,
                    coord_mode,
                    nearest_mode,
                }
            }),
            OpType::Round => op!(Round),
            OpType::ScatterElements(args) => {
                op_with_attrs!(ScatterElements, ScatterElementsAttrs, {
                    let reduction = match args.reduction {
                        None => sg::ScatterReduction::None,
                        Some(ScatterReduction::Add) => sg::ScatterReduction::Add,
                        Some(ScatterReduction::Mul) => sg::ScatterReduction::Mul,
                        Some(ScatterReduction::Min) => sg::ScatterReduction::Min,
                        Some(ScatterReduction::Max) => sg::ScatterReduction::Max,
                    };
                    sg::ScatterElementsAttrsArgs {
                        axis: args.axis as i32,
                        reduction,
                    }
                })
            }
            OpType::Shape => op!(Shape),
            OpType::Sigmoid => op!(Sigmoid),
            OpType::Slice => op!(Slice),
            OpType::Sin => op!(Sin),
            OpType::Sign => op!(Sign),
            OpType::Size => op!(Size),
            OpType::Softmax(args) => op_with_attrs!(
                Softmax,
                SoftmaxAttrs,
                sg::SoftmaxAttrsArgs {
                    axis: args.axis as i32,
                }
            ),
            OpType::Softplus => op!(Softplus),
            OpType::Split(args) => op_with_attrs!(Split, SplitAttrs, {
                sg::SplitAttrsArgs {
                    axis: args.axis as i32,
                }
            }),
            OpType::Sqrt => op!(Sqrt),
            OpType::Squeeze => op!(Squeeze),
            OpType::Sub => op!(Sub),
            OpType::Sum => op!(Sum),
            OpType::Tan => op!(Tan),
            OpType::Tanh => op!(Tanh),
            OpType::Tile => op!(Tile),
            OpType::TopK(args) => op_with_attrs!(TopK, TopKAttrs, {
                sg::TopKAttrsArgs {
                    axis: args.axis.unwrap_or(-1) as i32,
                    largest: args.largest,
                    sorted: args.sorted,
                }
            }),
            OpType::Transpose(args) => op_with_attrs!(Transpose, TransposeAttrs, {
                let perm = self.create_vec(args.perm, |dim| dim as u32);
                sg::TransposeAttrsArgs { perm }
            }),
            OpType::Trilu(args) => op_with_attrs!(Trilu, TriluAttrs, {
                sg::TriluAttrsArgs { upper: args.upper }
            }),
            OpType::Unsqueeze => op!(Unsqueeze),
            OpType::Where => op!(Where),
            OpType::Xor => op!(Xor),
        };

        let input_ids: Vec<i32> = inputs
            .iter()
            .map(|&id| match id {
                Some(id) => id.as_u32() as i32,
                None => -1,
            })
            .collect();
        let output_ids: Vec<i32> = outputs.iter().map(|&id| id.as_u32() as i32).collect();

        let input_vec = self.builder.create_vector(&input_ids);
        let output_vec = self.builder.create_vector(&output_ids);
        let op_node = sg::OperatorNode::create(
            self.builder,
            &sg::OperatorNodeArgs {
                type_: op_type,
                attrs_type,
                attrs,
                inputs: Some(input_vec),
                outputs: Some(output_vec),
            },
        );
        self.add_node(Some(id), NodeData::Operator(op_node))
    }

    /// Mark a node in the graph as an input.
    pub fn add_input(&mut self, node_id: NodeId) {
        self.input_ids.push(node_id);
    }

    /// Mark a node in the graph as an output.
    pub fn add_output(&mut self, node_id: NodeId) {
        self.output_ids.push(node_id);
    }

    /// Convert a `Vec<T>` of elements to a `Vec<U>` and add them to the model buffer
    fn create_vec<T: Copy, U: flatbuffers::Push + Copy, F: Fn(T) -> U>(
        &mut self,
        data: Option<Vec<T>>,
        map: F,
    ) -> Option<WIPOffset<Vector<'a, U::Output>>> {
        data.map(|vec| {
            let converted_vec: Vec<U> = vec.iter().copied().map(map).collect();
            self.builder.create_vector(&converted_vec)
        })
    }

    /// Finish writing this graph to the FlatBuffers buffer.
    pub fn finish(self) -> WIPOffset<sg::Graph<'a>> {
        let input_ids: Vec<_> = self.input_ids.iter().map(|id| id.as_u32()).collect();
        let output_ids: Vec<_> = self.output_ids.iter().map(|id| id.as_u32()).collect();

        let inputs_vec = self.builder.create_vector(&input_ids);
        let outputs_vec = self.builder.create_vector(&output_ids);
        let nodes_vec = self.builder.create_vector(&self.nodes[..]);

        sg::Graph::create(
            self.builder,
            &sg::GraphArgs {
                nodes: Some(nodes_vec),
                inputs: Some(inputs_vec),
                outputs: Some(outputs_vec),
                captures: None,
            },
        )
    }
}

/// Serializes models to the RTen model format.
///
/// This exists for use in model-loading tests. Models for deployment are
/// normally built by converting ONNX models using the Python scripts.
pub struct ModelBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    graph: Option<WIPOffset<sg::Graph<'a>>>,
    metadata: Option<WIPOffset<sg::Metadata<'a>>>,

    // Builder for the buffer containing tensor data stored outside the model.
    // `None` if building the V1 format which stores all data inline.
    tensor_data_builder: Option<TensorDataBuilder>,
}

impl<'a> ModelBuilder<'a> {
    pub fn new(format: ModelFormat) -> ModelBuilder<'a> {
        let builder = FlatBufferBuilder::with_capacity(1024);
        ModelBuilder {
            builder,
            graph: None,
            metadata: None,
            tensor_data_builder: match format {
                ModelFormat::V1 => None,
                ModelFormat::V2 => Some(TensorDataBuilder::new()),
            },
        }
    }

    /// Return a builder that can be used to serialize the main graph for the
    /// model.
    ///
    /// Call [`GraphBuilder::finish`] to finish serialization and pass the
    /// result to [`set_graph`](ModelBuilder::set_graph).
    pub fn graph_builder<'mb>(&'mb mut self) -> GraphBuilder<'mb, 'a> {
        GraphBuilder::new(&mut self.builder, self.tensor_data_builder.as_mut())
    }

    /// Set the main graph for this model.
    ///
    /// To construct the graph, use [`graph_builder`](ModelBuilder::graph_builder).
    pub fn set_graph(&mut self, graph: WIPOffset<sg::Graph<'a>>) {
        self.graph = Some(graph);
    }

    /// Add model metadata
    pub fn add_metadata(&mut self, metadata: MetadataArgs) {
        let hash = metadata
            .onnx_hash
            .as_ref()
            .map(|hash| self.builder.create_string(hash));
        let mut meta_builder = sg::MetadataBuilder::new(&mut self.builder);
        if let Some(hash) = hash {
            meta_builder.add_onnx_hash(hash);
        }
        self.metadata = Some(meta_builder.finish());
    }

    /// Finish writing the model data to the buffer and return the buffer's contents.
    pub fn finish(mut self) -> Vec<u8> {
        let model = sg::Model::create(
            &mut self.builder,
            &sg::ModelArgs {
                schema_version: 1,
                graph: self.graph,
                metadata: self.metadata,
            },
        );

        self.builder.finish(model, None);
        let model_data = self.builder.finished_data().to_vec();

        // If we are storing tensor data externally, then generate a file in the
        // V2 format. Otherwise use the V1 format which contains just the
        // FlatBuffers data.
        if let Some(tensor_data) = self.tensor_data_builder.take() {
            let mut file_buf = Vec::new();
            let tensor_data = tensor_data.into_vec();
            let header = Header {
                version: 2,
                model_len: model_data.len() as u64,
                model_offset: Header::LEN as u64,
                tensor_data_offset: Header::LEN as u64 + model_data.len() as u64,
            };
            file_buf.extend(header.to_buf());
            file_buf.extend(model_data);
            file_buf.extend(tensor_data);
            file_buf
        } else {
            model_data
        }
    }
}

impl Default for ModelBuilder<'_> {
    fn default() -> Self {
        Self::new(ModelFormat::V2)
    }
}

/// Builds the buffer used for storing tensor data outside the FlatBuffers
/// model.
struct TensorDataBuilder {
    data: Vec<u8>,
}

impl TensorDataBuilder {
    fn new() -> TensorDataBuilder {
        TensorDataBuilder { data: Vec::new() }
    }

    /// Append data for a tensor to the end of the buffer.
    ///
    /// Returns the offset within the buffer of the start of the data for the
    /// tensor.
    fn add_tensor<T: Copy + LeBytes>(&mut self, data: &[T]) -> usize {
        let offset = self.data.len();

        // This currently uses the minimum required alignment for the type.
        //
        // In the real models we might choose a larger alignment so that rows
        // of matrices start on a cache line boundary.
        let align = std::mem::align_of::<T>();
        let padding = offset.next_multiple_of(align) - offset;
        self.data.extend(std::iter::repeat(0).take(padding));

        let start_offset = self.data.len();

        for x in data {
            let bytes = x.to_le_bytes();
            self.data.extend(bytes.as_ref());
        }

        start_offset
    }

    /// Consume the builder and return the finalized tensor data buffer.
    fn into_vec(self) -> Vec<u8> {
        self.data
    }
}
