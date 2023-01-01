extern crate flatbuffers;

use std::collections::HashMap;

use crate::graph::{Graph, NodeId, RunError, RunOptions};
use crate::ops;
use crate::ops::{DataType, Input, Operator, Output, Padding, ResizeMode, Scalar};
use crate::schema_generated as sg;
use crate::schema_generated::{root_as_model, OperatorNode, OperatorType, PadMode};
use crate::tensor::from_data;

pub struct Model {
    node_ids: HashMap<String, NodeId>,
    input_ids: Vec<NodeId>,
    output_ids: Vec<NodeId>,
    graph: Graph,
}

impl Model {
    /// Load a serialized model.
    pub fn load(data: &[u8]) -> Result<Model, String> {
        load_model(data)
    }

    /// Find a node in the model's graph given its string ID.
    pub fn find_node(&self, id: &str) -> Option<NodeId> {
        self.node_ids.get(id).copied()
    }

    /// Return the IDs of input nodes.
    pub fn input_ids(&self) -> &[NodeId] {
        &self.input_ids
    }

    /// Return the IDs of output nodes.
    pub fn output_ids(&self) -> &[NodeId] {
        &self.output_ids
    }

    /// Execute the model.
    ///
    /// The input and output nodes are specified via IDs looked up via `find_node`.
    pub fn run(
        &self,
        inputs: &[(NodeId, Input)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        self.graph.run(inputs, outputs, opts)
    }
}

fn padding_from_attrs(mode: PadMode, pads: Option<flatbuffers::Vector<'_, u32>>) -> Padding {
    match (mode, pads) {
        (PadMode::Same, _) => Padding::Same,
        (PadMode::Fixed, Some(pads)) => Padding::Fixed([
            pads.get(0) as usize,
            pads.get(1) as usize,
            pads.get(2) as usize,
            pads.get(3) as usize,
        ]),
        _ => Padding::Fixed([0, 0, 0, 0]),
    }
}

/// Read the first N items from `iter` into an array.
///
/// Panics if the iterator yields fewer than N items.
fn array_from_iter<const N: usize, T: Default + Copy, I: Iterator<Item = T>>(
    mut iter: I,
) -> [T; N] {
    let mut result = [T::default(); N];
    for i in 0..N {
        result[i] = iter.next().expect("incorrect array size");
    }
    result
}

/// Error type for errors that occur when de-serializing an operator.
enum ReadOpError {
    /// The operator attributes were missing or of the wrong type.
    AttrError,
    /// The operator type is incorrect or unsupported.
    UnsupportedOperator,
}

type ReadOpResult = Result<Box<dyn Operator>, ReadOpError>;

fn read_average_pool_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_average_pool_attrs()
        .ok_or(ReadOpError::AttrError)?;

    let kernel_size = array_from_iter(attrs.kernel_size().iter().map(|x| x as usize));
    let padding = padding_from_attrs(attrs.pad_mode(), attrs.pads());
    let strides = attrs
        .strides()
        .map(|stride| array_from_iter(stride.iter().map(|x| x as usize)))
        .unwrap_or([1, 1]);

    Ok(Box::new(ops::AveragePool {
        kernel_size,
        padding,
        strides,
    }))
}

fn read_batch_normalization_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_batch_normalization_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::BatchNormalization {
        epsilon: attrs.epsilon(),
    }))
}

fn read_cast_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_cast_attrs().ok_or(ReadOpError::AttrError)?;
    let to = match attrs.to() {
        sg::DataType::Int32 => DataType::Int32,
        sg::DataType::Float => DataType::Float,
        _ => DataType::Float,
    };
    Ok(Box::new(ops::Cast { to }))
}

fn read_clip_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_clip_attrs().ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Clip {
        min: attrs.min(),
        max: attrs.max(),
    }))
}

fn read_concat_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_concat_attrs().ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Concat {
        dim: attrs.dim() as usize,
    }))
}

fn read_conv_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_conv_attrs().ok_or(ReadOpError::AttrError)?;

    let groups = attrs.groups() as usize;
    let padding = padding_from_attrs(attrs.pad_mode(), attrs.pads());
    let strides = attrs
        .strides()
        .map(|stride| array_from_iter(stride.iter().map(|x| x as usize)))
        .unwrap_or([1, 1]);

    Ok(Box::new(ops::Conv {
        groups,
        padding,
        strides,
    }))
}

fn read_constant_of_shape_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_constant_of_shape_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let value = if let Some(int_val) = attrs.value_as_int_scalar() {
        Scalar::Int(int_val.value())
    } else if let Some(float_val) = attrs.value_as_float_scalar() {
        Scalar::Float(float_val.value())
    } else {
        Scalar::Int(0)
    };
    Ok(Box::new(ops::ConstantOfShape { value }))
}

fn read_conv_transpose_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_conv_transpose_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let strides = attrs
        .strides()
        .map(|stride| array_from_iter(stride.iter().map(|x| x as usize)))
        .unwrap_or([1, 1]);
    Ok(Box::new(ops::ConvTranspose { strides }))
}

fn read_gather_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_gather_attrs().ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Gather {
        axis: attrs.axis() as usize,
    }))
}

fn read_gemm_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_gemm_attrs().ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Gemm {
        alpha: attrs.alpha(),
        beta: attrs.beta(),
        transpose_a: attrs.transpose_a(),
        transpose_b: attrs.transpose_b(),
    }))
}

fn read_leaky_relu_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_leaky_relu_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::LeakyRelu {
        alpha: attrs.alpha(),
    }))
}

fn read_max_pool_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_max_pool_attrs()
        .ok_or(ReadOpError::AttrError)?;

    let kernel_size = array_from_iter(attrs.kernel_size().iter().map(|x| x as usize));
    let padding = padding_from_attrs(attrs.pad_mode(), attrs.pads());
    let strides = attrs
        .strides()
        .map(|stride| array_from_iter(stride.iter().map(|x| x as usize)))
        .unwrap_or([1, 1]);

    Ok(Box::new(ops::MaxPool {
        kernel_size,
        padding,
        strides,
    }))
}

fn read_reduce_mean_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_reduce_mean_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let axes = attrs.axes().map(|axes| axes.iter().collect());
    let keep_dims = attrs.keep_dims();
    Ok(Box::new(ops::ReduceMean { axes, keep_dims }))
}

fn read_resize_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_resize_attrs().ok_or(ReadOpError::AttrError)?;
    let mode = match attrs.mode() {
        sg::ResizeMode::Nearest => ResizeMode::Nearest,
        sg::ResizeMode::Linear => ResizeMode::Linear,
        _ => ResizeMode::Nearest,
    };
    Ok(Box::new(ops::Resize { mode }))
}

fn read_softmax_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_softmax_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Softmax {
        axis: attrs.axis() as usize,
    }))
}

fn read_split_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_split_attrs().ok_or(ReadOpError::AttrError)?;

    let axis = attrs.axis() as isize;
    let split = if let Some(split_vec) = attrs.split() {
        split_vec.iter().map(|size| size as usize).collect()
    } else {
        Vec::new()
    };

    Ok(Box::new(ops::Split { axis, split }))
}

fn read_squeeze_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_squeeze_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let axes = attrs
        .axes()
        .map(|axes| axes.iter().map(|axis| axis as usize).collect());
    Ok(Box::new(ops::Squeeze { axes }))
}

fn read_transpose_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_transpose_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let perm = attrs
        .perm()
        .map(|perm| perm.iter().map(|dim| dim as usize).collect());
    Ok(Box::new(ops::Transpose { perm }))
}

fn read_unsqueeze_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_unsqueeze_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let mut axes: Vec<usize> = attrs.axes().iter().map(|axis| axis as usize).collect();
    axes.sort();
    Ok(Box::new(ops::Unsqueeze { axes }))
}

/// Create a `Box<dyn Operator>` for an operator that has no attributes.
macro_rules! op {
    ($op_name:ident) => {
        Ok(Box::new(ops::$op_name {}))
    };
}

fn read_operator(node: &OperatorNode) -> ReadOpResult {
    match node.type_() {
        OperatorType::Add => op!(Add),
        OperatorType::AveragePool => read_average_pool_op(node),
        OperatorType::BatchNormalization => read_batch_normalization_op(node),
        OperatorType::Cast => read_cast_op(node),
        OperatorType::Clip => read_clip_op(node),
        OperatorType::Concat => read_concat_op(node),
        OperatorType::Conv => read_conv_op(node),
        OperatorType::ConstantOfShape => read_constant_of_shape_op(node),
        OperatorType::ConvTranspose => read_conv_transpose_op(node),
        OperatorType::Cos => op!(Cos),
        OperatorType::Div => op!(Div),
        OperatorType::Equal => op!(Equal),
        OperatorType::Erf => op!(Erf),
        OperatorType::Expand => op!(Expand),
        OperatorType::Gather => read_gather_op(node),
        OperatorType::Gemm => read_gemm_op(node),
        OperatorType::GlobalAveragePool => op!(GlobalAveragePool),
        OperatorType::Identity => op!(Identity),
        OperatorType::LeakyRelu => read_leaky_relu_op(node),
        OperatorType::Less => op!(Less),
        OperatorType::MatMul => op!(MatMul),
        OperatorType::MaxPool => read_max_pool_op(node),
        OperatorType::Mul => op!(Mul),
        OperatorType::Pad => op!(Pad),
        OperatorType::Pow => op!(Pow),
        OperatorType::Range => op!(Range),
        OperatorType::ReduceMean => read_reduce_mean_op(node),
        OperatorType::Relu => op!(Relu),
        OperatorType::Reshape => op!(Reshape),
        OperatorType::Resize => read_resize_op(node),
        OperatorType::Shape => op!(Shape),
        OperatorType::Sigmoid => op!(Sigmoid),
        OperatorType::Sin => op!(Sin),
        OperatorType::Slice => op!(Slice),
        OperatorType::Softmax => read_softmax_op(node),
        OperatorType::Split => read_split_op(node),
        OperatorType::Sqrt => op!(Sqrt),
        OperatorType::Squeeze => read_squeeze_op(node),
        OperatorType::Sub => op!(Sub),
        OperatorType::Tanh => op!(Tanh),
        OperatorType::Transpose => read_transpose_op(node),
        OperatorType::Unsqueeze => read_unsqueeze_op(node),
        OperatorType::Where => op!(Where),
        _ => Err(ReadOpError::UnsupportedOperator),
    }
}

fn load_model(data: &[u8]) -> Result<Model, String> {
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

    let input_ids = model
        .graph()
        .inputs()
        .map(|ids| ids.iter().map(|id| id as NodeId).collect())
        .unwrap_or_default();

    let output_ids = model
        .graph()
        .outputs()
        .map(|ids| ids.iter().map(|id| id as NodeId).collect())
        .unwrap_or_default();

    if let Some(nodes) = model.graph().nodes() {
        for (node_index, node) in nodes.iter().enumerate() {
            if let Some(operator) = node.data_as_operator_node() {
                let op = read_operator(&operator).map_err(|err| match err {
                    ReadOpError::UnsupportedOperator => "unsupported operator".to_string(),
                    ReadOpError::AttrError => "incorrect or missing attributes".to_string(),
                })?;

                let mut inputs: Vec<NodeId> = Vec::new();
                if let Some(op_input_ids) = operator.inputs() {
                    for node_index in op_input_ids.iter() {
                        let index_usize = node_index as usize;
                        if let Some(node_id) = node_id_from_index.get(&index_usize) {
                            inputs.push(*node_id)
                        } else {
                            return Err("Operator input is invalid".to_string());
                        }
                    }
                }

                let mut outputs: Vec<NodeId> = Vec::new();
                if let Some(op_output_ids) = operator.outputs() {
                    for node_index in op_output_ids.iter() {
                        let index_usize = node_index as usize;
                        if let Some(node_id) = node_id_from_index.get(&index_usize) {
                            outputs.push(*node_id)
                        } else {
                            return Err("Operator output is invalid".to_string());
                        }
                    }
                }

                let graph_node = graph.add_op(node.name(), op, &inputs, &outputs);

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
        input_ids,
        output_ids,
        graph,
    };
    Ok(model)
}

#[cfg(test)]
mod tests {
    extern crate flatbuffers;

    use crate::model::Model;
    use crate::model_builder::{ModelBuilder, OpType};
    use crate::ops;
    use crate::ops::{Padding, ResizeMode, Scalar};
    use crate::tensor::{from_data, from_scalar, from_vec};

    fn generate_model_buffer() -> Vec<u8> {
        let mut builder = ModelBuilder::new();

        let const_val = from_data(vec![1, 2, 2], vec![0.5, -0.5, 0.1, -0.1]);
        let const_node = builder.add_float_constant(&const_val);
        let input_node = builder.add_value("input");
        let output_node = builder.add_value("output");

        builder.add_input(input_node);
        builder.add_output(output_node);

        let concat_out = builder.add_value("concat_out");
        builder.add_operator(
            "concat",
            OpType::Concat(ops::Concat { dim: 0 }),
            &[const_node, input_node],
            &[concat_out],
        );
        builder.add_operator("relu", OpType::Relu, &[concat_out], &[output_node]);

        builder.finish()
    }

    #[test]
    fn test_model_input_output_ids() {
        let buffer = generate_model_buffer();

        let model = Model::load(&buffer).unwrap();
        let input_id = model.find_node("input").unwrap();
        let output_id = model.find_node("output").unwrap();

        assert_eq!(model.input_ids(), &[input_id]);
        assert_eq!(model.output_ids(), &[output_id]);
    }

    #[test]
    fn test_load_and_run_model() {
        let buffer = generate_model_buffer();

        let model = Model::load(&buffer).unwrap();
        let input_id = model.input_ids()[0];
        let output_id = model.output_ids()[0];

        let input = from_data(vec![1, 2, 2], vec![1., 2., -1., -2.]);
        let result = model
            .run(&[(input_id, (&input).into())], &[output_id], None)
            .unwrap();

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

        let add_out = builder.add_value("add_out");
        builder.add_operator("add", OpType::Add, &[input_node, input_node], &[add_out]);

        let average_pool_out = builder.add_value("average_pool_out");
        builder.add_operator(
            "average_pool",
            OpType::AveragePool(ops::AveragePool {
                kernel_size: [2, 2],
                strides: [2, 2],
                padding: Padding::Fixed([0, 0, 0, 0]),
            }),
            &[input_node],
            &[average_pool_out],
        );

        // Dummy value for BatchNormalization inputs which are vectors with
        // per-channel values.
        let batch_norm_param_val = from_vec(vec![1.0]);
        let batch_norm_param = builder.add_float_constant(&batch_norm_param_val);

        let batch_normalization_out = builder.add_value("batch_normalization_out");
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
            &[batch_normalization_out],
        );

        let cast_out = builder.add_value("cast_out");
        builder.add_operator(
            "cast",
            OpType::Cast(ops::Cast {
                to: ops::DataType::Float,
            }),
            &[input_node],
            &[cast_out],
        );

        let clip_out = builder.add_value("clip_out");
        builder.add_operator(
            "clip",
            OpType::Clip(ops::Clip { min: 1.0, max: 5.0 }),
            &[input_node],
            &[clip_out],
        );

        let concat_out = builder.add_value("concat_out");
        builder.add_operator(
            "concat",
            OpType::Concat(ops::Concat { dim: 0 }),
            &[input_node, input_node],
            &[concat_out],
        );

        let shape = builder.add_int_constant(&from_data(vec![3], vec![1, 5, 10]));
        let constant_of_shape_out = builder.add_value("constant_of_shape_out");
        builder.add_operator(
            "constant_of_shape",
            OpType::ConstantOfShape(ops::ConstantOfShape {
                value: Scalar::Int(42),
            }),
            &[shape],
            &[constant_of_shape_out],
        );

        let conv_out = builder.add_value("conv_out");
        builder.add_operator(
            "conv",
            OpType::Conv(ops::Conv {
                padding: Padding::Fixed([1, 1, 1, 1]),
                groups: 1,
                strides: [1, 1],
            }),
            &[input_node, kernel],
            &[conv_out],
        );

        let conv_transpose_out = builder.add_value("conv_transpose_out");
        builder.add_operator(
            "conv_transpose",
            OpType::ConvTranspose(ops::ConvTranspose { strides: [2, 2] }),
            &[input_node, kernel],
            &[conv_transpose_out],
        );

        let cos_out = builder.add_value("cos_out");
        builder.add_operator("cos", OpType::Cos, &[input_node], &[cos_out]);

        let div_out = builder.add_value("div_out");
        builder.add_operator("div", OpType::Div, &[input_node, input_node], &[div_out]);

        let erf_out = builder.add_value("erf_out");
        builder.add_operator("erf", OpType::Erf, &[input_node], &[erf_out]);

        let expand_shape_val = from_vec(vec![2, 2, 3, 3]);
        let expand_shape = builder.add_int_constant(&expand_shape_val);
        let expand_out = builder.add_value("expand_out");
        builder.add_operator(
            "expand",
            OpType::Expand,
            &[input_node, expand_shape],
            &[expand_out],
        );

        let equal_out = builder.add_value("equal_out");
        builder.add_operator(
            "equal",
            OpType::Equal,
            &[input_node, input_node],
            &[equal_out],
        );

        let gather_out = builder.add_value("gather_out");
        builder.add_operator(
            "gather",
            OpType::Gather(ops::Gather { axis: 0 }),
            &[input_node, indices],
            &[gather_out],
        );

        let gemm_out = builder.add_value("gemm_out");
        builder.add_operator(
            "gemm",
            OpType::Gemm(ops::Gemm {
                alpha: 1.0,
                beta: 1.0,
                transpose_a: false,
                transpose_b: false,
            }),
            &[input_2d, input_2d],
            &[gemm_out],
        );

        let global_average_pool_out = builder.add_value("global_average_pool_out");
        builder.add_operator(
            "global_average_pool",
            OpType::GlobalAveragePool,
            &[input_node],
            &[global_average_pool_out],
        );

        let identity_out = builder.add_value("identity_out");
        builder.add_operator("identity", OpType::Identity, &[input_node], &[identity_out]);

        let leaky_relu_out = builder.add_value("leaky_relu_out");
        builder.add_operator(
            "leaky_relu",
            OpType::LeakyRelu(ops::LeakyRelu { alpha: 0.01 }),
            &[input_node],
            &[leaky_relu_out],
        );

        let less_out = builder.add_value("less_out");
        builder.add_operator("less", OpType::Less, &[input_node, input_node], &[less_out]);

        let matmul_out = builder.add_value("matmul_out");
        builder.add_operator(
            "matmul",
            OpType::MatMul,
            &[input_2d, input_2d],
            &[matmul_out],
        );

        let max_pool_out = builder.add_value("max_pool_out");
        builder.add_operator(
            "max_pool",
            OpType::MaxPool(ops::MaxPool {
                kernel_size: [2, 2],
                strides: [2, 2],
                padding: Padding::Fixed([0, 0, 0, 0]),
            }),
            &[input_node],
            &[max_pool_out],
        );

        let mul_out = builder.add_value("mul_out");
        builder.add_operator("mul", OpType::Mul, &[input_node, input_node], &[mul_out]);

        let pads = builder.add_int_constant(&from_data(vec![8], vec![0, 0, 1, 1, 0, 0, 1, 1]));
        let pad_out = builder.add_value("pad_out");
        builder.add_operator("pad", OpType::Pad, &[input_node, pads], &[pad_out]);

        let pow_out = builder.add_value("pow_out");
        builder.add_operator("pow", OpType::Pow, &[input_node, input_node], &[pow_out]);

        let range_start_node = builder.add_value("range_start");
        let range_limit_node = builder.add_value("range_limit");
        let range_delta_node = builder.add_value("range_delta");
        let range_out = builder.add_value("range");
        builder.add_operator(
            "range",
            OpType::Range,
            &[range_start_node, range_limit_node, range_delta_node],
            &[range_out],
        );

        let reduce_mean_out = builder.add_value("reduce_mean_out");
        builder.add_operator(
            "reduce_mean",
            OpType::ReduceMean(ops::ReduceMean {
                axes: None,
                keep_dims: false,
            }),
            &[input_node],
            &[reduce_mean_out],
        );

        let relu_out = builder.add_value("relu_out");
        builder.add_operator("relu", OpType::Relu, &[input_node], &[relu_out]);

        let new_shape = builder.add_int_constant(&from_data(vec![1], vec![9]));
        let reshape_out = builder.add_value("reshape_out");
        builder.add_operator(
            "reshape",
            OpType::Reshape,
            &[input_node, new_shape],
            &[reshape_out],
        );

        let resize_out = builder.add_value("resize_out");
        let resize_roi_val = from_vec(vec![0., 0., 0., 0., 1., 1., 1., 1.]);
        let resize_scales_val = from_vec(vec![1., 1., 2., 2.]);
        let resize_roi = builder.add_float_constant(&resize_roi_val);
        let resize_scales = builder.add_float_constant(&resize_scales_val);
        builder.add_operator(
            "resize",
            OpType::Resize(ops::Resize {
                mode: ResizeMode::Nearest,
            }),
            &[input_node, resize_roi, resize_scales],
            &[resize_out],
        );

        let shape_out = builder.add_value("shape_out");
        builder.add_operator("shape", OpType::Shape, &[input_node], &[shape_out]);

        let sigmoid_out = builder.add_value("sigmoid_out");
        builder.add_operator("sigmoid", OpType::Sigmoid, &[input_node], &[sigmoid_out]);

        let sin_out = builder.add_value("sin_out");
        builder.add_operator("sin", OpType::Sin, &[input_node], &[sin_out]);

        let const_0 = builder.add_int_constant(&from_data(vec![1], vec![0]));
        let const_1 = builder.add_int_constant(&from_data(vec![1], vec![1]));
        let slice_out = builder.add_value("slice_out");
        builder.add_operator(
            "slice",
            OpType::Slice,
            &[input_node, const_0, const_1, const_0],
            &[slice_out],
        );

        let softmax_out = builder.add_value("softmax_out");
        builder.add_operator(
            "softmax",
            OpType::Softmax(ops::Softmax { axis: 1 }),
            &[input_node],
            &[softmax_out],
        );

        let sqrt_out = builder.add_value("sqrt_out");
        builder.add_operator("sqrt", OpType::Sqrt, &[input_node], &[sqrt_out]);

        let squeeze_out = builder.add_value("squeeze_out");
        builder.add_operator(
            "squeeze",
            OpType::Squeeze(ops::Squeeze { axes: None }),
            &[input_node],
            &[squeeze_out],
        );

        let split_out_1 = builder.add_value("split_out_1");
        let split_out_2 = builder.add_value("split_out_2");
        builder.add_operator(
            "split",
            OpType::Split(ops::Split {
                axis: 1,
                split: vec![1, 2],
            }),
            &[input_2d],
            &[split_out_1, split_out_2],
        );

        let sub_out = builder.add_value("sub_out");
        builder.add_operator("sub", OpType::Sub, &[input_node, input_node], &[sub_out]);

        let tanh_out = builder.add_value("tanh_out");
        builder.add_operator("tanh", OpType::Tanh, &[input_node], &[tanh_out]);

        let transpose_out = builder.add_value("transpose_out");
        builder.add_operator(
            "transpose",
            OpType::Transpose(ops::Transpose { perm: None }),
            &[input_node],
            &[transpose_out],
        );

        let unsqueeze_out = builder.add_value("unsqueeze_out");
        builder.add_operator(
            "unsqueeze",
            OpType::Unsqueeze(ops::Unsqueeze { axes: vec![0, 4] }),
            &[input_node],
            &[unsqueeze_out],
        );

        let where_out = builder.add_value("where_out");
        let where_cond = builder.add_value("where_cond");
        let where_x = builder.add_value("where_x");
        let where_y = builder.add_value("where_y");
        builder.add_operator(
            "where",
            OpType::Where,
            &[where_cond, where_x, where_y],
            &[where_out],
        );

        let buffer = builder.finish();

        let model = Model::load(&buffer).unwrap();

        // Outputs of ops tested with a 4D input (eg. NCHW image).
        let outputs = vec![
            "add_out",
            "average_pool_out",
            "batch_normalization_out",
            "cast_out",
            "clip_out",
            "concat_out",
            "constant_of_shape_out",
            "conv_out",
            "conv_transpose_out",
            "cos_out",
            "div_out",
            "equal_out",
            "erf_out",
            "expand_out",
            "identity_out",
            "global_average_pool_out",
            "leaky_relu_out",
            "less_out",
            "max_pool_out",
            "mul_out",
            "pad_out",
            "pow_out",
            "reduce_mean_out",
            "relu_out",
            "reshape_out",
            "resize_out",
            "shape_out",
            "sigmoid_out",
            "slice_out",
            "softmax_out",
            "sqrt_out",
            "squeeze_out",
            "sin_out",
            "sub_out",
            "tanh_out",
            "transpose_out",
            "unsqueeze_out",
        ];
        let input = from_data(vec![1, 1, 3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model
                .run(
                    &[(input_node as usize, (&input).into())],
                    &[output_id],
                    None,
                )
                .unwrap();
            assert_eq!(result.len(), 1);
        }

        // Outputs of ops tested with a 2D input.
        let outputs = vec!["matmul_out", "split_out_1", "split_out_2"];
        let input = from_data(vec![3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model
                .run(&[(input_2d as usize, (&input).into())], &[output_id], None)
                .unwrap();
            assert_eq!(result.len(), 1);
        }

        // Range op
        let start = from_scalar(0.);
        let limit = from_scalar(5.);
        let delta = from_scalar(1.);
        let result = model
            .run(
                &[
                    (range_start_node as usize, (&start).into()),
                    (range_limit_node as usize, (&limit).into()),
                    (range_delta_node as usize, (&delta).into()),
                ],
                &[range_out as usize],
                None,
            )
            .unwrap();
        assert_eq!(result.len(), 1);

        // Where op
        let cond = from_scalar(1);
        let x = from_vec(vec![1, 2, 3]);
        let y = from_vec(vec![4, 5, 6]);
        let result = model
            .run(
                &[
                    (where_cond as usize, (&cond).into()),
                    (where_x as usize, (&x).into()),
                    (where_y as usize, (&y).into()),
                ],
                &[where_out as usize],
                None,
            )
            .unwrap();
        assert_eq!(result.len(), 1);
    }
}
