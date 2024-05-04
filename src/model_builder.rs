use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, Vector, WIPOffset};
use rten_tensor::prelude::*;
use rten_tensor::Tensor;

use crate::graph::Dimension;
use crate::ops::{
    ArgMax, ArgMin, AveragePool, BatchNormalization, BoxOrder, Cast, Concat, ConstantOfShape, Conv,
    ConvTranspose, CoordTransformMode, DataType, Elu, Flatten, Gather, GatherElements, Gemm,
    HardSigmoid, InstanceNormalization, LayerNormalization, LeakyRelu, LogSoftmax, MaxPool, Mod,
    NearestMode, NonMaxSuppression, OneHot, Padding, ReduceMax, ReduceMean, ReduceMin, ReduceProd,
    ReduceSum, ReduceSumSquare, Reshape, Resize, ResizeMode, Scalar, ScatterElements,
    ScatterReduction, Softmax, Split, TopK, Transpose, Trilu,
};
use crate::schema_generated as sg;

#[cfg(feature = "random")]
use crate::ops::{RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike};

/// Enum of all the built-in operators
pub enum OpType {
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
    Div,
    Elu(Elu),
    Equal,
    Erf,
    Exp,
    Expand,
    Flatten(Flatten),
    Floor,
    Gather(Gather),
    GatherElements(GatherElements),
    Gemm(Gemm),
    GlobalAveragePool,
    Greater,
    GreaterOrEqual,
    HardSigmoid(HardSigmoid),
    HardSwish,
    Identity,
    InstanceNormalization(InstanceNormalization),
    LayerNormalization(LayerNormalization),
    LeakyRelu(LeakyRelu),
    Less,
    LessOrEqual,
    Log,
    LogSoftmax(LogSoftmax),
    MatMul,
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

/// Builds a serialized FlatBuffers representation of a model using the schema
/// defined in schema.fbs.
///
/// This exists for use in model-loading tests. Models for deployment are
/// normally built by converting ONNX models using the Python scripts.
pub struct ModelBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    nodes: Vec<WIPOffset<sg::Node<'a>>>,
    input_ids: Vec<u32>,
    output_ids: Vec<u32>,
    metadata: Option<WIPOffset<sg::Metadata<'a>>>,
}

enum NodeData<'a> {
    Constant(WIPOffset<sg::ConstantNode<'a>>),
    Value(WIPOffset<sg::ValueNode<'a>>),
    Operator(WIPOffset<sg::OperatorNode<'a>>),
}

/// Arguments for [ModelBuilder::add_metadata].
pub struct MetadataArgs {
    pub onnx_hash: Option<String>,
}

struct PadArgs {
    pad_mode: sg::PadMode,
    pads: Option<Vec<usize>>,
}

fn pad_args_from_padding(padding: Padding) -> PadArgs {
    match padding {
        Padding::Same => PadArgs {
            pad_mode: sg::PadMode::Same,
            pads: None,
        },
        Padding::Fixed(pads) => PadArgs {
            pad_mode: sg::PadMode::Fixed,
            pads: Some(pads.iter().copied().collect()),
        },
    }
}

impl<'a> ModelBuilder<'a> {
    pub fn new() -> ModelBuilder<'a> {
        let builder = FlatBufferBuilder::with_capacity(1024);
        ModelBuilder {
            builder,
            nodes: Vec::new(),
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            metadata: None,
        }
    }

    fn add_node(&mut self, name: Option<&str>, data: NodeData) -> u32 {
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
        let node = sg::Node::create(&mut self.builder, &args);
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    /// Add a constant node (eg. weights, biases) to the model
    pub fn add_float_constant(&mut self, input: &Tensor) -> u32 {
        let elts: Vec<f32> = input.to_vec();
        let data_vec = self.builder.create_vector(&elts);

        let float_data = sg::FloatData::create(
            &mut self.builder,
            &sg::FloatDataArgs {
                data: Some(data_vec),
            },
        );

        self.add_constant_node(
            input.shape(),
            sg::ConstantData::FloatData,
            float_data.as_union_value(),
        )
    }

    /// Add a constant node (eg. weights, biases) to the model
    pub fn add_int_constant(&mut self, input: &Tensor<i32>) -> u32 {
        let elts: Vec<i32> = input.to_vec();
        let data_vec = self.builder.create_vector(&elts);

        let int_data = sg::IntData::create(
            &mut self.builder,
            &sg::IntDataArgs {
                data: Some(data_vec),
            },
        );

        self.add_constant_node(
            input.shape(),
            sg::ConstantData::IntData,
            int_data.as_union_value(),
        )
    }

    fn add_constant_node(
        &mut self,
        shape: &[usize],
        data_type: sg::ConstantData,
        data: WIPOffset<UnionWIPOffset>,
    ) -> u32 {
        let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
        let shape_vec = self.builder.create_vector(&shape[..]);

        let const_node = sg::ConstantNode::create(
            &mut self.builder,
            &sg::ConstantNodeArgs {
                shape: Some(shape_vec),
                data_type,
                data: Some(data),
            },
        );
        self.add_node(None, NodeData::Constant(const_node))
    }

    /// Add a value node to the model
    pub fn add_value(&mut self, id: &str, shape: Option<&[Dimension]>) -> u32 {
        let shape = shape.map(|shape| {
            let dim_vec: Vec<_> = shape
                .iter()
                .map(|dim| match dim {
                    Dimension::Fixed(value) => sg::Dim::create(
                        &mut self.builder,
                        &sg::DimArgs {
                            name: None,
                            value: *value as u32,
                        },
                    ),
                    Dimension::Symbolic(name) => {
                        let name_offset = self.builder.create_string(name);
                        sg::Dim::create(
                            &mut self.builder,
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
        let value_node = sg::ValueNode::create(&mut self.builder, &sg::ValueNodeArgs { shape });
        self.add_node(Some(id), NodeData::Value(value_node))
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

    /// Add an operator node to the model
    pub fn add_operator(
        &mut self,
        id: &str,
        op_info: OpType,
        inputs: &[Option<u32>],
        outputs: &[u32],
    ) -> u32 {
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
                let attrs = sg::$attr_type::create(&mut self.builder, &args).as_union_value();
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
                    pad_mode: pad_args.pad_mode,
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
                    to: match args.to {
                        DataType::Int32 => sg::DataType::Int32,
                        DataType::Float => sg::DataType::Float,
                    },
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
                                    &mut self.builder,
                                    &sg::IntScalarArgs { value: int_value },
                                )
                                .as_union_value(),
                            ),
                        },
                        Scalar::Float(float_value) => sg::ConstantOfShapeAttrsArgs {
                            value_type: sg::Scalar::FloatScalar,
                            value: Some(
                                sg::FloatScalar::create(
                                    &mut self.builder,
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
                    pad_mode: pad_args.pad_mode,
                    pads,
                    strides,
                }
            }),
            OpType::ConvTranspose(args) => op_with_attrs!(ConvTranspose, ConvTransposeAttrs, {
                let strides = self.create_vec(Some(args.strides.into()), |s| s as u32);
                sg::ConvTransposeAttrsArgs { strides }
            }),
            OpType::Cos => op!(Cos),
            OpType::Div => op!(Div),
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
            OpType::Max => op!(Max),
            OpType::MaxPool(args) => op_with_attrs!(MaxPool, MaxPoolAttrs, {
                let pad_args = pad_args_from_padding(args.padding);
                let pads = self.create_vec(pad_args.pads, |pad| pad as u32);
                let kernel_size = self.create_vec(Some(args.kernel_size.into()), |sz| sz as u32);
                let strides = self.create_vec(Some(args.strides.into()), |s| s as u32);
                sg::MaxPoolAttrsArgs {
                    kernel_size,
                    pad_mode: pad_args.pad_mode,
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
                Some(id) => id as i32,
                None => -1,
            })
            .collect();
        let output_ids: Vec<i32> = outputs.iter().map(|&id| id as i32).collect();

        let input_vec = self.builder.create_vector(&input_ids);
        let output_vec = self.builder.create_vector(&output_ids);
        let op_node = sg::OperatorNode::create(
            &mut self.builder,
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
    pub fn add_input(&mut self, node_id: u32) {
        self.input_ids.push(node_id);
    }

    /// Mark a node in the graph as an output.
    pub fn add_output(&mut self, node_id: u32) {
        self.output_ids.push(node_id);
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
        let inputs_vec = self.builder.create_vector(&self.input_ids[..]);
        let outputs_vec = self.builder.create_vector(&self.output_ids[..]);
        let nodes_vec = self.builder.create_vector(&self.nodes[..]);

        let graph = sg::Graph::create(
            &mut self.builder,
            &sg::GraphArgs {
                nodes: Some(nodes_vec),
                inputs: Some(inputs_vec),
                outputs: Some(outputs_vec),
            },
        );

        let model = sg::Model::create(
            &mut self.builder,
            &sg::ModelArgs {
                schema_version: 1,
                graph: Some(graph),
                metadata: self.metadata,
            },
        );

        self.builder.finish(model, None);
        self.builder.finished_data().to_vec()
    }
}

impl<'a> Default for ModelBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}
