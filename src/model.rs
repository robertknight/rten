extern crate flatbuffers;

use std::collections::HashMap;

use crate::graph::{Graph, NodeId};
use crate::ops;
use crate::ops::Operator;
use crate::schema_generated::{root_as_model, OperatorNode, OperatorType};
use crate::tensor::{from_data, Tensor};

pub struct Model {
    node_ids: HashMap<String, NodeId>,
    graph: Graph,
}

impl Model {
    /// Find a node in the model's graph given its string ID.
    pub fn find_node(&self, id: &str) -> Option<NodeId> {
        self.node_ids.get(id).map(|x| *x)
    }

    /// Execute the model.
    ///
    /// The input and output nodes are specified via IDs looked up via `find_node`.
    pub fn run(&self, inputs: &[(NodeId, &Tensor)], outputs: &[NodeId]) -> Vec<Tensor> {
        self.graph.run(inputs, outputs)
    }
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

    if let Some(attrs) = node.attrs_as_conv_2d_attrs() {
        groups = attrs.groups() as usize;
        padding = (
            attrs.pad_horizontal() as usize,
            attrs.pad_vertical() as usize,
        );
    } else {
        groups = 1;
        padding = (0, 0);
    }

    Box::new(ops::Conv2d { groups, padding })
}

fn read_conv_transpose_2d_op(node: &OperatorNode) -> Box<dyn Operator> {
    let stride = match node.attrs_as_conv_transpose_2d_attrs() {
        Some(attrs) => attrs.stride() as usize,
        None => 2,
    };
    Box::new(ops::ConvTranspose2d { stride })
}

fn read_max_pool_2d_op(node: &OperatorNode) -> Box<dyn Operator> {
    let kernel_size = match node.attrs_as_max_pool_2d_attrs() {
        Some(attrs) => attrs.kernel_size() as usize,
        None => 2,
    };
    Box::new(ops::MaxPool2d { kernel_size })
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
        OperatorType::Concat => read_concat_op(&node),
        OperatorType::Conv2d => read_conv_2d_op(&node),
        OperatorType::ConvTranspose2d => read_conv_transpose_2d_op(&node),
        OperatorType::MaxPool2d => read_max_pool_2d_op(&node),
        OperatorType::Pad2d => read_pad_2d_op(&node),
        OperatorType::ReLU => read_relu_op(&node),
        OperatorType::Sigmoid => read_sigmoid_op(&node),
        OperatorType::Slice => read_slice_op(&node),
        _ => return Err(format!("Unknown operator type")),
    };
    Ok(op)
}

/// Load a serialized model.
pub fn load_model(data: &[u8]) -> Result<Model, String> {
    let model = root_as_model(data).map_err(|e| format!("Error parsing flatbuffer {:?}", e))?;

    if model.schema_version() != 1 {
        return Err(format!("Unsupported schema version"));
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
                            return Err(format!("Operator input is invalid"));
                        }
                    }
                }

                let graph_node = graph.add_op(op, &inputs);

                add_node_id(node.id(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else if let Some(_) = node.data_as_value_node() {
                let graph_node = graph.add_value();

                add_node_id(node.id(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else if let Some(constant) = node.data_as_constant_node() {
                let shape: Vec<usize> = constant.shape().iter().map(|x| x as usize).collect();
                let data: Vec<f32> = constant.data().iter().collect();
                let tensor = from_data(shape, data);
                let graph_node = graph.add_constant(tensor);

                add_node_id(node.id(), graph_node);
                node_id_from_index.insert(node_index, graph_node);
            } else {
                return Err(format!("Unknown node type"));
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
    use crate::ops::OpType;
    use crate::tensor::from_data;

    fn generate_model_buffer() -> Vec<u8> {
        let mut builder = ModelBuilder::new();

        let const_val = from_data(vec![1, 2, 2], vec![0.5, -0.5, 0.1, -0.1]);
        let const_node = builder.add_constant(&const_val);
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
        let result = model.run(&[(input_id, &input)], &[output_id]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), vec![2, 2, 2]);
        assert_eq!(result[0].data(), vec![0.5, 0., 0.1, 0., 1., 2., 0., 0.]);
    }

    #[test]
    fn test_all_op_types() {
        let mut builder = ModelBuilder::new();

        let input_node = builder.add_value("input");
        let kernel_val = from_data(vec![1, 1, 1, 1], vec![0.5]);
        let kernel = builder.add_constant(&kernel_val);

        builder.add_operator(
            "concat",
            OpType::Concat(ops::Concat { dim: 0 }),
            &[input_node, input_node],
        );
        builder.add_operator(
            "conv_2d",
            OpType::Conv2d(ops::Conv2d {
                padding: (1, 1),
                groups: 1,
            }),
            &[input_node, kernel],
        );
        builder.add_operator(
            "conv_transpose_2d",
            OpType::ConvTranspose2d(ops::ConvTranspose2d { stride: 2 }),
            &[input_node, kernel],
        );
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

        let outputs = vec![
            "concat",
            "conv_2d",
            "conv_transpose_2d",
            "max_pool_2d",
            "pad_2d",
            "relu",
            "sigmoid",
            "slice",
        ];
        let input = from_data(vec![1, 1, 3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model.run(&[(input_node as usize, &input)], &[output_id]);
            assert_eq!(result.len(), 1);
        }
    }
}
