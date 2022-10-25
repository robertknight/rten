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

fn read_global_average_pool_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::GlobalAveragePool {})
}

fn read_max_pool_2d_op(node: &OperatorNode) -> Box<dyn Operator> {
    let kernel_size = match node.attrs_as_max_pool_2d_attrs() {
        Some(attrs) => attrs.kernel_size() as usize,
        None => 2,
    };
    Box::new(ops::MaxPool2d { kernel_size })
}

fn read_matmul_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::MatMul {})
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
    Box::new(ops::ReLU {})
}

fn read_reshape_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Reshape {})
}

fn read_sigmoid_op(_: &OperatorNode) -> Box<dyn Operator> {
    Box::new(ops::Sigmoid {})
}

fn read_slice_op(node: &OperatorNode) -> Box<dyn Operator> {
    let dim;
    let start;
    let end;

    if let Some(attrs) = node.attrs_as_slice_attrs() {
        dim = attrs.dim() as usize;
        start = attrs.start() as usize;
        end = attrs.end() as usize;
    } else {
        dim = 0;
        start = 0;
        end = 0;
    }
    Box::new(ops::Slice { dim, start, end })
}

fn read_operator(node: &OperatorNode) -> Result<Box<dyn Operator>, String> {
    let op: Box<dyn Operator> = match node.type_() {
        OperatorType::Add => read_add_op(node),
        OperatorType::Clip => read_clip_op(node),
        OperatorType::Concat => read_concat_op(node),
        OperatorType::Conv2d => read_conv_2d_op(node),
        OperatorType::ConvTranspose2d => read_conv_transpose_2d_op(node),
        OperatorType::GlobalAveragePool => read_global_average_pool_op(node),
        OperatorType::MatMul => read_matmul_op(node),
        OperatorType::MaxPool2d => read_max_pool_2d_op(node),
        OperatorType::Pad2d => read_pad_2d_op(node),
        OperatorType::ReLU => read_relu_op(node),
        OperatorType::Reshape => read_reshape_op(node),
        OperatorType::Sigmoid => read_sigmoid_op(node),
        OperatorType::Slice => read_slice_op(node),
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

    // Map of model node ID to graph node ID
    let mut node_id_from_id: HashMap<String, NodeId> = HashMap::new();

    // Map of model node index to graph node ID
    let mut node_id_from_index: HashMap<usize, NodeId> = HashMap::new();

    let mut add_node_id = |id: Option<&str>, graph_node| {
        if let Some(id) = id {
            node_id_from_id.insert(id.to_string(), graph_node);
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

                let graph_node = graph.add_op(op, &inputs);

                add_node_id(node.id(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else if node.data_as_value_node().is_some() {
                let graph_node = graph.add_value();

                add_node_id(node.id(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else if let Some(constant) = node.data_as_constant_node() {
                let shape: Vec<usize> = constant.shape().iter().map(|x| x as usize).collect();
                let graph_node = if let Some(float_data) = constant.data_as_float_data() {
                    let data: Vec<f32> = float_data.data().iter().collect();
                    let tensor = from_data(shape, data);
                    graph.add_constant(tensor)
                } else if let Some(int_data) = constant.data_as_int_data() {
                    let data: Vec<i32> = int_data.data().iter().collect();
                    let tensor = from_data(shape, data);
                    graph.add_constant(tensor)
                } else {
                    panic!("Unsupported constant data type");
                };

                add_node_id(node.id(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else {
                return Err("Unknown node type".to_string());
            }
        }
    }

    let model = Model {
        node_ids: node_id_from_id,
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
    use crate::tensor::from_data;

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
        builder.add_operator("output", OpType::ReLU, &[concat_node]);

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

    #[test]
    fn test_all_op_types() {
        let mut builder = ModelBuilder::new();

        let input_node = builder.add_value("input");
        let input_2d = builder.add_value("input.2d");

        let kernel_val = from_data(vec![1, 1, 1, 1], vec![0.5]);
        let kernel = builder.add_float_constant(&kernel_val);

        builder.add_operator("add", OpType::Add, &[input_node, input_node]);
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
            "global_average_pool",
            OpType::GlobalAveragePool,
            &[input_node],
        );
        builder.add_operator("matmul", OpType::MatMul, &[input_2d, input_2d]);
        builder.add_operator(
            "max_pool_2d",
            OpType::MaxPool2d(ops::MaxPool2d { kernel_size: 2 }),
            &[input_node],
        );
        builder.add_operator(
            "pad_2d",
            OpType::Pad2d(ops::Pad2d {
                padding: [1, 1, 1, 1],
            }),
            &[input_node],
        );
        builder.add_operator("relu", OpType::ReLU, &[input_node]);

        let new_shape = builder.add_int_constant(&from_data(vec![1], vec![9]));
        builder.add_operator("reshape", OpType::Reshape, &[input_node, new_shape]);

        builder.add_operator("sigmoid", OpType::Sigmoid, &[input_node]);
        builder.add_operator(
            "slice",
            OpType::Slice(ops::Slice {
                dim: 0,
                start: 0,
                end: 1,
            }),
            &[input_node],
        );

        let buffer = builder.finish();

        let model = load_model(&buffer).unwrap();

        // Test cases that accept a 4D input (eg. NCHW).
        let outputs = vec![
            "add",
            "clip",
            "concat",
            "conv_2d",
            "conv_transpose_2d",
            "global_average_pool",
            "max_pool_2d",
            "pad_2d",
            "relu",
            "reshape",
            "sigmoid",
            "slice",
        ];
        let input = from_data(vec![1, 1, 3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model.run(&[(input_node as usize, &input)], &[output_id], None);
            assert_eq!(result.len(), 1);
        }

        // Test cases that accept a 2D input.
        let outputs = vec!["matmul"];
        let input = from_data(vec![3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model.run(&[(input_2d as usize, &input)], &[output_id], None);
            assert_eq!(result.len(), 1);
        }
    }
}
