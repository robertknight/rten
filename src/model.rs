extern crate flatbuffers;

use std::collections::HashMap;

use crate::graph::{Graph, NodeId, RunOptions};
use crate::ops;
use crate::ops::{Operator, Output, Padding};
use crate::schema_generated::{root_as_model, OperatorNode, OperatorType, PadMode};
use crate::tensor::{from_data, Tensor};

pub struct Model {
    node_ids: HashMap<String, NodeId>,
    graph: Graph,
}

impl Model {
    /// Find a node in the model's graph given its string ID.
    pub fn find_node(&self, id: &str) -> Option<NodeId> {
        self.node_ids.get(id).copied()
    }

    /// Execute the model.
    ///
    /// The input and output nodes are specified via IDs looked up via `find_node`.
    pub fn run(
        &self,
        inputs: &[(NodeId, &Tensor)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Vec<Output> {
        self.graph.run(inputs, outputs, opts)
    }
}

fn read_add_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Add {})
}

fn read_average_pool_2d_op(node: &OperatorNode) -> Box<dyn Operator> {
    let kernel_size;
    let padding;
    let stride;

    if let Some(attrs) = node.attrs_as_average_pool_2d_attrs() {
        kernel_size = attrs.kernel_size() as usize;
        padding = match attrs.pad_mode() {
            PadMode::Same => Padding::Same,
            PadMode::Fixed => Padding::Fixed((
                attrs.pad_vertical() as usize,
                attrs.pad_horizontal() as usize,
            )),
            _ => Padding::Fixed((0, 0)),
        };
        stride = attrs.stride() as usize;
    } else {
        kernel_size = 1;
        padding = Padding::Fixed((0, 0));
        stride = 1;
    }

    Box::new(ops::AveragePool2d {
        kernel_size,
        padding,
        stride,
    })
}

fn read_batch_normalization_op(node: &OperatorNode) -> Box<dyn Operator> {
    let epsilon = match node.attrs_as_batch_normalization_attrs() {
        Some(attrs) => attrs.epsilon(),
        None => 1e-5,
    };
    Box::new(ops::BatchNormalization { epsilon })
}

fn read_clip_op(node: &OperatorNode) -> Box<dyn Operator> {
    let min;
    let max;

    if let Some(attrs) = node.attrs_as_clip_attrs() {
        min = attrs.min();
        max = attrs.max();
    } else {
        min = f32::NEG_INFINITY;
        max = f32::INFINITY;
    }

    Box::new(ops::Clip { min, max })
}

fn read_concat_op(node: &OperatorNode) -> Box<dyn Operator> {
    let dim = match node.attrs_as_concat_attrs() {
        Some(concat_attrs) => concat_attrs.dim() as usize,
        None => 0,
    };
    Box::new(ops::Concat { dim })
}

fn read_conv_2d_op(node: &OperatorNode) -> Box<dyn Operator> {
    let groups;
    let padding;
    let stride;

    if let Some(attrs) = node.attrs_as_conv_2d_attrs() {
        groups = attrs.groups() as usize;
        padding = match attrs.pad_mode() {
            PadMode::Same => Padding::Same,
            PadMode::Fixed => Padding::Fixed((
                attrs.pad_vertical() as usize,
                attrs.pad_horizontal() as usize,
            )),
            _ => Padding::Fixed((0, 0)),
        };
        stride = attrs.stride() as usize;
    } else {
        groups = 1;
        padding = Padding::Fixed((0, 0));
        stride = 1;
    }

    Box::new(ops::Conv2d {
        groups,
        padding,
        stride,
    })
}

fn read_conv_transpose_2d_op(node: &OperatorNode) -> Box<dyn Operator> {
    let stride = match node.attrs_as_conv_transpose_2d_attrs() {
        Some(attrs) => attrs.stride() as usize,
        None => 2,
    };
    Box::new(ops::ConvTranspose2d { stride })
}

fn read_gather_op(node: &OperatorNode) -> Box<dyn Operator> {
    let axis = match node.attrs_as_gather_attrs() {
        Some(attrs) => attrs.axis() as usize,
        None => 0,
    };
    Box::new(ops::Gather { axis })
}

fn read_gemm_op(node: &OperatorNode) -> Box<dyn Operator> {
    let alpha;
    let beta;
    let transpose_a;
    let transpose_b;

    if let Some(attrs) = node.attrs_as_gemm_attrs() {
        alpha = attrs.alpha();
        beta = attrs.beta();
        transpose_a = attrs.transpose_a();
        transpose_b = attrs.transpose_b();
    } else {
        alpha = 1.0;
        beta = 1.0;
        transpose_a = false;
        transpose_b = false;
    }

    Box::new(ops::Gemm {
        alpha,
        beta,
        transpose_a,
        transpose_b,
    })
}

fn read_global_average_pool_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::GlobalAveragePool {})
}

fn read_leaky_relu_op(node: &OperatorNode) -> Box<dyn Operator> {
    let alpha = match node.attrs_as_leaky_relu_attrs() {
        Some(attrs) => attrs.alpha(),
        None => 0.0,
    };
    Box::new(ops::LeakyRelu { alpha })
}

fn read_max_pool_2d_op(node: &OperatorNode) -> Box<dyn Operator> {
    let kernel_size;
    let padding;
    let stride;

    if let Some(attrs) = node.attrs_as_max_pool_2d_attrs() {
        kernel_size = attrs.kernel_size() as usize;
        padding = match attrs.pad_mode() {
            PadMode::Same => Padding::Same,
            PadMode::Fixed => Padding::Fixed((
                attrs.pad_vertical() as usize,
                attrs.pad_horizontal() as usize,
            )),
            _ => Padding::Fixed((0, 0)),
        };
        stride = attrs.stride() as usize;
    } else {
        kernel_size = 1;
        padding = Padding::Fixed((0, 0));
        stride = 1;
    }

    Box::new(ops::MaxPool2d {
        kernel_size,
        padding,
        stride,
    })
}

fn read_matmul_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::MatMul {})
}

fn read_mul_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Mul {})
}

fn read_pad_2d_op(node: &OperatorNode) -> Box<dyn Operator> {
    let padding = match node.attrs_as_pad_2d_attrs() {
        Some(attrs) => [
            attrs.pad_left() as usize,
            attrs.pad_top() as usize,
            attrs.pad_right() as usize,
            attrs.pad_bottom() as usize,
        ],
        None => [0, 0, 0, 0],
    };
    Box::new(ops::Pad2d { padding })
}

fn read_relu_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Relu {})
}

fn read_reshape_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Reshape {})
}

fn read_shape_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Shape {})
}

fn read_sigmoid_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Sigmoid {})
}

fn read_slice_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Slice {})
}

fn read_softmax_op(node: &OperatorNode) -> Box<dyn Operator> {
    let axis = match node.attrs_as_softmax_attrs() {
        Some(attrs) => attrs.axis() as usize,
        None => 0,
    };
    Box::new(ops::Softmax { axis })
}

fn read_squeeze_op(node: &OperatorNode) -> Box<dyn Operator> {
    let mut axes: Option<Vec<usize>> = None;
    if let Some(attrs) = node.attrs_as_squeeze_attrs() {
        if let Some(axes_vec) = attrs.axes() {
            axes = Some(axes_vec.iter().map(|axis| axis as usize).collect());
        }
    }
    Box::new(ops::Squeeze { axes })
}

fn read_transpose_op(node: &OperatorNode) -> Box<dyn Operator> {
    let mut perm: Option<Vec<usize>> = None;
    if let Some(attrs) = node.attrs_as_transpose_attrs() {
        if let Some(perm_vec) = attrs.perm() {
            perm = Some(perm_vec.iter().map(|dim| dim as usize).collect());
        }
    }
    Box::new(ops::Transpose { perm })
}

fn read_unsqueeze_op(node: &OperatorNode) -> Box<dyn Operator> {
    let mut axes: Vec<usize>;
    if let Some(attrs) = node.attrs_as_unsqueeze_attrs() {
        axes = attrs.axes().iter().map(|axis| axis as usize).collect();
        axes.sort();
    } else {
        axes = Vec::new();
    }
    Box::new(ops::Unsqueeze { axes })
}

fn read_operator(node: &OperatorNode) -> Result<Box<dyn Operator>, String> {
    let op: Box<dyn Operator> = match node.type_() {
        OperatorType::Add => read_add_op(node),
        OperatorType::AveragePool2d => read_average_pool_2d_op(node),
        OperatorType::BatchNormalization => read_batch_normalization_op(node),
        OperatorType::Clip => read_clip_op(node),
        OperatorType::Concat => read_concat_op(node),
        OperatorType::Conv2d => read_conv_2d_op(node),
        OperatorType::ConvTranspose2d => read_conv_transpose_2d_op(node),
        OperatorType::Gather => read_gather_op(node),
        OperatorType::Gemm => read_gemm_op(node),
        OperatorType::GlobalAveragePool => read_global_average_pool_op(node),
        OperatorType::LeakyRelu => read_leaky_relu_op(node),
        OperatorType::MatMul => read_matmul_op(node),
        OperatorType::MaxPool2d => read_max_pool_2d_op(node),
        OperatorType::Mul => read_mul_op(node),
        OperatorType::Pad2d => read_pad_2d_op(node),
        OperatorType::Relu => read_relu_op(node),
        OperatorType::Reshape => read_reshape_op(node),
        OperatorType::Shape => read_shape_op(node),
        OperatorType::Sigmoid => read_sigmoid_op(node),
        OperatorType::Slice => read_slice_op(node),
        OperatorType::Softmax => read_softmax_op(node),
        OperatorType::Squeeze => read_squeeze_op(node),
        OperatorType::Transpose => read_transpose_op(node),
        OperatorType::Unsqueeze => read_unsqueeze_op(node),
        _ => return Err("Unknown operator type".to_string()),
    };
    Ok(op)
}

/// Load a serialized model.
pub fn load_model(data: &[u8]) -> Result<Model, String> {
    let model = root_as_model(data).map_err(|e| format!("Error parsing flatbuffer {:?}", e))?;

    if model.schema_version() != 1 {
        return Err("Unsupported schema version".to_string());
    }

    let mut graph = Graph::new();

    // Map of model node name to graph node ID
    let mut node_id_from_name: HashMap<String, NodeId> = HashMap::new();

    // Map of model node index to graph node ID
    let mut node_id_from_index: HashMap<usize, NodeId> = HashMap::new();

    let mut add_node_id = |name: Option<&str>, graph_node| {
        if let Some(name) = name {
            node_id_from_name.insert(name.to_string(), graph_node);
        }
    };

    if let Some(nodes) = model.graph().nodes() {
        for (node_index, node) in nodes.iter().enumerate() {
            if let Some(operator) = node.data_as_operator_node() {
                let op = read_operator(&operator)?;

                let mut inputs: Vec<NodeId> = Vec::new();
                if let Some(model_inputs) = operator.inputs() {
                    for model_node_index in model_inputs.iter() {
                        let index_usize = model_node_index as usize;
                        if let Some(node_id) = node_id_from_index.get(&index_usize) {
                            inputs.push(*node_id)
                        } else {
                            return Err("Operator input is invalid".to_string());
                        }
                    }
                }

                let graph_node = graph.add_op(node.name(), op, &inputs);

                add_node_id(node.name(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else if node.data_as_value_node().is_some() {
                let graph_node = graph.add_value(node.name());

                add_node_id(node.name(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else if let Some(constant) = node.data_as_constant_node() {
                let shape: Vec<usize> = constant.shape().iter().map(|x| x as usize).collect();
                let graph_node = if let Some(float_data) = constant.data_as_float_data() {
                    let data: Vec<f32> = float_data.data().iter().collect();
                    let tensor = from_data(shape, data);
                    graph.add_constant(node.name(), tensor)
                } else if let Some(int_data) = constant.data_as_int_data() {
                    let data: Vec<i32> = int_data.data().iter().collect();
                    let tensor = from_data(shape, data);
                    graph.add_constant(node.name(), tensor)
                } else {
                    panic!("Unsupported constant data type");
                };

                add_node_id(node.name(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else {
                return Err("Unknown node type".to_string());
            }
        }
    }

    let model = Model {
        node_ids: node_id_from_name,
        graph,
    };
    Ok(model)
}

#[cfg(test)]
mod tests {
    extern crate flatbuffers;

    use crate::model::load_model;
    use crate::model_builder::ModelBuilder;
    use crate::ops;
    use crate::ops::{OpType, Padding};
    use crate::tensor::{from_data, from_vec};

    fn generate_model_buffer() -> Vec<u8> {
        let mut builder = ModelBuilder::new();

        let const_val = from_data(vec![1, 2, 2], vec![0.5, -0.5, 0.1, -0.1]);
        let const_node = builder.add_float_constant(&const_val);
        let input_node = builder.add_value("input");

        let concat_node = builder.add_operator(
            "concat",
            OpType::Concat(ops::Concat { dim: 0 }),
            &[const_node, input_node],
        );
        builder.add_operator("output", OpType::Relu, &[concat_node]);

        builder.finish()
    }

    #[test]
    fn test_load_and_run_model() {
        let buffer = generate_model_buffer();

        let model = load_model(&buffer).unwrap();
        let input_id = model.find_node("input").unwrap();
        let output_id = model.find_node("output").unwrap();

        let input = from_data(vec![1, 2, 2], vec![1., 2., -1., -2.]);
        let result = model.run(&[(input_id, &input)], &[output_id], None);

        assert_eq!(result.len(), 1);

        let result_tensor = result[0].as_float_ref().unwrap();

        assert_eq!(result_tensor.shape(), vec![2, 2, 2]);
        assert_eq!(result_tensor.data(), vec![0.5, 0., 0.1, 0., 1., 2., 0., 0.]);
    }

    // This test exercises basic execution of all operators. It doesn't check
    // the results of operators, it just sure they can be deserialized and
    // executed with valid inputs.
    #[test]
    fn test_all_op_types() {
        let mut builder = ModelBuilder::new();

        let input_node = builder.add_value("input");
        let input_2d = builder.add_value("input.2d");

        let kernel_val = from_data(vec![1, 1, 1, 1], vec![0.5]);
        let kernel = builder.add_float_constant(&kernel_val);

        let indices_val = from_data(vec![1], vec![1]);
        let indices = builder.add_int_constant(&indices_val);

        builder.add_operator("add", OpType::Add, &[input_node, input_node]);

        builder.add_operator(
            "average_pool_2d",
            OpType::AveragePool2d(ops::AveragePool2d {
                kernel_size: 2,
                stride: 2,
                padding: Padding::Fixed((0, 0)),
            }),
            &[input_node],
        );

        // Dummy value for BatchNormalization inputs which are vectors with
        // per-channel values.
        let batch_norm_param_val = from_vec(vec![1.0]);
        let batch_norm_param = builder.add_float_constant(&batch_norm_param_val);

        builder.add_operator(
            "batch_normalization",
            OpType::BatchNormalization(ops::BatchNormalization { epsilon: 1e-5 }),
            &[
                input_node,
                batch_norm_param, /* scale */
                batch_norm_param, /* bias */
                batch_norm_param, /* mean */
                batch_norm_param, /* variance */
            ],
        );

        builder.add_operator(
            "clip",
            OpType::Clip(ops::Clip { min: 1.0, max: 5.0 }),
            &[input_node],
        );
        builder.add_operator(
            "concat",
            OpType::Concat(ops::Concat { dim: 0 }),
            &[input_node, input_node],
        );
        builder.add_operator(
            "conv_2d",
            OpType::Conv2d(ops::Conv2d {
                padding: Padding::Fixed((1, 1)),
                groups: 1,
                stride: 1,
            }),
            &[input_node, kernel],
        );
        builder.add_operator(
            "conv_transpose_2d",
            OpType::ConvTranspose2d(ops::ConvTranspose2d { stride: 2 }),
            &[input_node, kernel],
        );
        builder.add_operator(
            "gather",
            OpType::Gather(ops::Gather { axis: 0 }),
            &[input_node, indices],
        );
        builder.add_operator(
            "gemm",
            OpType::Gemm(ops::Gemm {
                alpha: 1.0,
                beta: 1.0,
                transpose_a: false,
                transpose_b: false,
            }),
            &[input_2d, input_2d],
        );
        builder.add_operator(
            "global_average_pool",
            OpType::GlobalAveragePool,
            &[input_node],
        );
        builder.add_operator(
            "leaky_relu",
            OpType::LeakyRelu(ops::LeakyRelu { alpha: 0.01 }),
            &[input_node],
        );
        builder.add_operator("matmul", OpType::MatMul, &[input_2d, input_2d]);
        builder.add_operator(
            "max_pool_2d",
            OpType::MaxPool2d(ops::MaxPool2d {
                kernel_size: 2,
                stride: 2,
                padding: Padding::Fixed((0, 0)),
            }),
            &[input_node],
        );
        builder.add_operator("mul", OpType::Mul, &[input_node, input_node]);
        builder.add_operator(
            "pad_2d",
            OpType::Pad2d(ops::Pad2d {
                padding: [1, 1, 1, 1],
            }),
            &[input_node],
        );
        builder.add_operator("relu", OpType::Relu, &[input_node]);

        let new_shape = builder.add_int_constant(&from_data(vec![1], vec![9]));
        builder.add_operator("reshape", OpType::Reshape, &[input_node, new_shape]);
        builder.add_operator("shape", OpType::Shape, &[input_node]);
        builder.add_operator("sigmoid", OpType::Sigmoid, &[input_node]);

        let const_0 = builder.add_int_constant(&from_data(vec![1], vec![0]));
        let const_1 = builder.add_int_constant(&from_data(vec![1], vec![1]));
        builder.add_operator(
            "slice",
            OpType::Slice,
            &[input_node, const_0, const_1, const_0],
        );
        builder.add_operator(
            "softmax",
            OpType::Softmax(ops::Softmax { axis: 1 }),
            &[input_node],
        );
        builder.add_operator(
            "squeeze",
            OpType::Squeeze(ops::Squeeze { axes: None }),
            &[input_node],
        );
        builder.add_operator(
            "transpose",
            OpType::Transpose(ops::Transpose { perm: None }),
            &[input_node],
        );
        builder.add_operator(
            "unsqueeze",
            OpType::Unsqueeze(ops::Unsqueeze { axes: vec![0, 4] }),
            &[input_node],
        );

        let buffer = builder.finish();

        let model = load_model(&buffer).unwrap();

        // Operators that accept a 4D input (eg. NCHW).
        let outputs = vec![
            "add",
            "average_pool_2d",
            "batch_normalization",
            "clip",
            "concat",
            "conv_2d",
            "conv_transpose_2d",
            "global_average_pool",
            "leaky_relu",
            "max_pool_2d",
            "mul",
            "pad_2d",
            "relu",
            "reshape",
            "shape",
            "sigmoid",
            "slice",
            "softmax",
            "squeeze",
            "transpose",
            "unsqueeze",
        ];
        let input = from_data(vec![1, 1, 3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model.run(&[(input_node as usize, &input)], &[output_id], None);
            assert_eq!(result.len(), 1);
        }

        // Operators that accept a 2D input.
        let outputs = vec!["matmul"];
        let input = from_data(vec![3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model.run(&[(input_2d as usize, &input)], &[output_id], None);
            assert_eq!(result.len(), 1);
        }
    }
}
