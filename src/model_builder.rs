extern crate flatbuffers;

use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, Vector, WIPOffset};

use crate::ops::{OpType, Padding};
use crate::schema_generated as sg;
use crate::tensor::Tensor;

/// Builds a serialized FlatBuffers representation of a model using the schema
/// defined in schema.fbs.
///
/// This exists for use in model-loading tests. Models for deployment are
/// normally built by converting ONNX models using the Python scripts.
pub struct ModelBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    nodes: Vec<WIPOffset<sg::Node<'a>>>,
}

enum NodeData<'a> {
    Constant(WIPOffset<sg::ConstantNode<'a>>),
    Value(WIPOffset<sg::ValueNode<'a>>),
    Operator(WIPOffset<sg::OperatorNode<'a>>),
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
            pads: Some(pads.into()),
        },
    }
}

impl<'a> ModelBuilder<'a> {
    pub fn new() -> ModelBuilder<'a> {
        let builder = FlatBufferBuilder::with_capacity(1024);
        ModelBuilder {
            builder,
            nodes: Vec::new(),
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
        let elts: Vec<f32> = input.elements().collect();
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
        let elts: Vec<i32> = input.elements().collect();
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
    pub fn add_value(&mut self, id: &str) -> u32 {
        let value_node = sg::ValueNode::create(&mut self.builder, &sg::ValueNodeArgs {});
        self.add_node(Some(id), NodeData::Value(value_node))
    }

    fn create_u32_vec<'fbb>(
        &mut self,
        data: Option<Vec<usize>>,
    ) -> Option<WIPOffset<Vector<'a, u32>>> {
        let vec_u32: Option<Vec<u32>> =
            data.map(|vec| vec.iter().map(|&item| item as u32).collect());
        vec_u32.map(|v| self.builder.create_vector(&v))
    }

    /// Add an operator node to the model
    pub fn add_operator(&mut self, id: &str, op_info: OpType, inputs: &[u32]) -> u32 {
        type OT = sg::OperatorType;
        type OA = sg::OperatorAttrs;
        let no_attrs = sg::OperatorAttrs::NONE;

        // Translate internal operator info to the types in the schema.
        // There is unfortunately a lot of boilerplate here.
        let (op_type, attrs_type, attrs) = match op_info {
            OpType::Add => (OT::Add, no_attrs, None),
            OpType::AveragePool2d(args) => (
                OT::AveragePool2d,
                OA::AveragePool2dAttrs,
                Some({
                    let pad_args = pad_args_from_padding(args.padding);
                    let pads = self.create_u32_vec(pad_args.pads);
                    sg::AveragePool2dAttrs::create(&mut self.builder, {
                        &sg::AveragePool2dAttrsArgs {
                            kernel_size: args.kernel_size as u32,
                            pad_mode: pad_args.pad_mode,
                            pads,
                            stride: args.stride as u32,
                        }
                    })
                    .as_union_value()
                }),
            ),
            OpType::BatchNormalization(args) => (
                OT::BatchNormalization,
                OA::BatchNormalizationAttrs,
                Some(
                    sg::BatchNormalizationAttrs::create(
                        &mut self.builder,
                        &sg::BatchNormalizationAttrsArgs {
                            epsilon: args.epsilon,
                        },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::Clip(args) => (
                OT::Clip,
                OA::ClipAttrs,
                Some(
                    sg::ClipAttrs::create(
                        &mut self.builder,
                        &sg::ClipAttrsArgs {
                            min: args.min,
                            max: args.max,
                        },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::Concat(args) => (
                OT::Concat,
                OA::ConcatAttrs,
                Some(
                    sg::ConcatAttrs::create(
                        &mut self.builder,
                        &sg::ConcatAttrsArgs {
                            dim: args.dim as u32,
                        },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::Conv2d(args) => (
                OT::Conv2d,
                OA::Conv2dAttrs,
                Some({
                    let pad_args = pad_args_from_padding(args.padding);
                    let pads = self.create_u32_vec(pad_args.pads);
                    sg::Conv2dAttrs::create(&mut self.builder, {
                        &sg::Conv2dAttrsArgs {
                            groups: args.groups as u32,
                            pad_mode: pad_args.pad_mode,
                            pads,
                            stride: args.stride as u32,
                        }
                    })
                    .as_union_value()
                }),
            ),
            OpType::ConvTranspose2d(args) => (
                OT::ConvTranspose2d,
                OA::ConvTranspose2dAttrs,
                Some(
                    sg::ConvTranspose2dAttrs::create(
                        &mut self.builder,
                        &sg::ConvTranspose2dAttrsArgs {
                            stride: args.stride as u32,
                        },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::Div => (OT::Div, no_attrs, None),
            OpType::Gather(args) => (
                OT::Gather,
                OA::GatherAttrs,
                Some(
                    sg::GatherAttrs::create(
                        &mut self.builder,
                        &sg::GatherAttrsArgs {
                            axis: args.axis as u32,
                        },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::Gemm(args) => (
                OT::Gemm,
                OA::GemmAttrs,
                Some(
                    sg::GemmAttrs::create(
                        &mut self.builder,
                        &sg::GemmAttrsArgs {
                            alpha: args.alpha,
                            beta: args.beta,
                            transpose_a: args.transpose_a,
                            transpose_b: args.transpose_b,
                        },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::GlobalAveragePool => (OT::GlobalAveragePool, no_attrs, None),
            OpType::Identity => (OT::Identity, no_attrs, None),
            OpType::LeakyRelu(args) => (
                OT::LeakyRelu,
                OA::LeakyReluAttrs,
                Some(
                    sg::LeakyReluAttrs::create(
                        &mut self.builder,
                        &sg::LeakyReluAttrsArgs { alpha: args.alpha },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::MatMul => (OT::MatMul, no_attrs, None),
            OpType::MaxPool2d(args) => (
                OT::MaxPool2d,
                OA::MaxPool2dAttrs,
                Some({
                    let pad_args = pad_args_from_padding(args.padding);
                    let pads = self.create_u32_vec(pad_args.pads);
                    sg::MaxPool2dAttrs::create(&mut self.builder, {
                        &sg::MaxPool2dAttrsArgs {
                            kernel_size: args.kernel_size as u32,
                            pad_mode: pad_args.pad_mode,
                            pads,
                            stride: args.stride as u32,
                        }
                    })
                    .as_union_value()
                }),
            ),
            OpType::Mul => (OT::Mul, no_attrs, None),
            OpType::Pad2d(args) => (
                OT::Pad2d,
                OA::Pad2dAttrs,
                Some(
                    sg::Pad2dAttrs::create(
                        &mut self.builder,
                        &sg::Pad2dAttrsArgs {
                            pad_left: args.padding[0] as u32,
                            pad_top: args.padding[1] as u32,
                            pad_right: args.padding[2] as u32,
                            pad_bottom: args.padding[3] as u32,
                        },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::Relu => (OT::Relu, no_attrs, None),
            OpType::Reshape => (OT::Reshape, no_attrs, None),
            OpType::Shape => (OT::Shape, no_attrs, None),
            OpType::Sigmoid => (OT::Sigmoid, no_attrs, None),
            OpType::Slice => (OT::Slice, no_attrs, None),
            OpType::Softmax(args) => (
                OT::Softmax,
                OA::SoftmaxAttrs,
                Some(
                    sg::SoftmaxAttrs::create(
                        &mut self.builder,
                        &sg::SoftmaxAttrsArgs {
                            axis: args.axis as u32,
                        },
                    )
                    .as_union_value(),
                ),
            ),
            OpType::Squeeze(args) => {
                let axes = self.create_u32_vec(args.axes);
                (
                    OT::Squeeze,
                    OA::SqueezeAttrs,
                    Some(
                        sg::SqueezeAttrs::create(&mut self.builder, &sg::SqueezeAttrsArgs { axes })
                            .as_union_value(),
                    ),
                )
            }
            OpType::Sub => (OT::Sub, no_attrs, None),
            OpType::Transpose(args) => {
                let perm = self.create_u32_vec(args.perm);
                (
                    OT::Transpose,
                    OA::TransposeAttrs,
                    Some(
                        sg::TransposeAttrs::create(
                            &mut self.builder,
                            &sg::TransposeAttrsArgs { perm },
                        )
                        .as_union_value(),
                    ),
                )
            }
            OpType::Unsqueeze(args) => {
                let axes_u32: Vec<u32> = args.axes.iter().map(|&axis| axis as u32).collect();
                let axes = self.builder.create_vector(&axes_u32);
                (
                    OT::Unsqueeze,
                    OA::UnsqueezeAttrs,
                    Some(
                        sg::UnsqueezeAttrs::create(
                            &mut self.builder,
                            &sg::UnsqueezeAttrsArgs { axes: Some(axes) },
                        )
                        .as_union_value(),
                    ),
                )
            }
        };

        let input_vec = self.builder.create_vector(inputs);
        let op_node = sg::OperatorNode::create(
            &mut self.builder,
            &sg::OperatorNodeArgs {
                type_: op_type,
                attrs_type,
                attrs,
                inputs: Some(input_vec),
            },
        );
        self.add_node(Some(id), NodeData::Operator(op_node))
    }

    /// Finish writing the model data to the buffer and return the buffer's contents.
    pub fn finish(mut self) -> Vec<u8> {
        let nodes_vec = self.builder.create_vector(&self.nodes[..]);

        let graph = sg::Graph::create(
            &mut self.builder,
            &sg::GraphArgs {
                nodes: Some(nodes_vec),
            },
        );

        let model = sg::Model::create(
            &mut self.builder,
            &sg::ModelArgs {
                schema_version: 1,
                graph: Some(graph),
            },
        );

        self.builder.finish(model, None);
        self.builder.finished_data().to_vec()
    }
}
