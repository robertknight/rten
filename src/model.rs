extern crate flatbuffers;

use std::collections::HashMap;

use crate::graph::{Graph, NodeId, RunError, RunOptions};
use crate::ops;
use crate::ops::{
    CoordTransformMode, DataType, Input, LSTMDirection, NearestMode, Operator, Output, Padding,
    ResizeMode, Scalar,
};
use crate::schema_generated as sg;
use crate::schema_generated::{root_as_model, OperatorNode, OperatorType, PadMode};
use crate::tensor::Tensor;

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

fn read_arg_max_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_arg_max_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::ArgMax {
        axis: attrs.axis() as isize,
        keep_dims: attrs.keep_dims(),
    }))
}

fn read_arg_min_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_arg_max_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::ArgMin {
        axis: attrs.axis() as isize,
        keep_dims: attrs.keep_dims(),
    }))
}

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

fn read_flatten_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_flatten_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Flatten {
        axis: attrs.axis() as isize,
    }))
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

fn read_lstm_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_lstmattrs().ok_or(ReadOpError::AttrError)?;

    let hidden_size = attrs.hidden_size() as usize;
    let direction = match attrs.direction() {
        sg::LSTMDirection::Forwards => LSTMDirection::Forwards,
        sg::LSTMDirection::Reverse => LSTMDirection::Reverse,
        sg::LSTMDirection::Bidirectional => LSTMDirection::Bidirectional,
        _ => LSTMDirection::Forwards,
    };

    Ok(Box::new(ops::LSTM {
        direction,
        hidden_size,
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

fn read_reduce_l2_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_reduce_mean_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let axes = attrs.axes().map(|axes| axes.iter().collect());
    let keep_dims = attrs.keep_dims();
    Ok(Box::new(ops::ReduceL2 { axes, keep_dims }))
}

fn read_reshape_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_reshape_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let allow_zero = attrs.allow_zero();
    Ok(Box::new(ops::Reshape { allow_zero }))
}

fn read_resize_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_resize_attrs().ok_or(ReadOpError::AttrError)?;
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
        _ => CoordTransformMode::default(),
    };

    Ok(Box::new(ops::Resize {
        mode,
        coord_mode,
        nearest_mode,
    }))
}

fn read_softmax_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_softmax_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Softmax {
        axis: attrs.axis() as isize,
    }))
}

fn read_split_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_split_attrs().ok_or(ReadOpError::AttrError)?;
    let axis = attrs.axis() as isize;
    Ok(Box::new(ops::Split { axis }))
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

/// Create a `Box<dyn Operator>` for an operator that has no attributes.
macro_rules! op {
    ($op_name:ident) => {
        Ok(Box::new(ops::$op_name {}))
    };
}

fn read_operator(node: &OperatorNode) -> ReadOpResult {
    match node.type_() {
        OperatorType::Add => op!(Add),
        OperatorType::ArgMax => read_arg_max_op(node),
        OperatorType::ArgMin => read_arg_min_op(node),
        OperatorType::AveragePool => read_average_pool_op(node),
        OperatorType::BatchNormalization => read_batch_normalization_op(node),
        OperatorType::Cast => read_cast_op(node),
        OperatorType::Clip => op!(Clip),
        OperatorType::Concat => read_concat_op(node),
        OperatorType::Conv => read_conv_op(node),
        OperatorType::ConstantOfShape => read_constant_of_shape_op(node),
        OperatorType::ConvTranspose => read_conv_transpose_op(node),
        OperatorType::Cos => op!(Cos),
        OperatorType::CumSum => op!(CumSum),
        OperatorType::Div => op!(Div),
        OperatorType::Equal => op!(Equal),
        OperatorType::Erf => op!(Erf),
        OperatorType::Expand => op!(Expand),
        OperatorType::Flatten => read_flatten_op(node),
        OperatorType::Gather => read_gather_op(node),
        OperatorType::Gemm => read_gemm_op(node),
        OperatorType::GlobalAveragePool => op!(GlobalAveragePool),
        OperatorType::Greater => op!(Greater),
        OperatorType::Identity => op!(Identity),
        OperatorType::LeakyRelu => read_leaky_relu_op(node),
        OperatorType::Less => op!(Less),
        OperatorType::LessOrEqual => op!(LessOrEqual),
        OperatorType::Log => op!(Log),
        OperatorType::LSTM => read_lstm_op(node),
        OperatorType::MatMul => op!(MatMul),
        OperatorType::MaxPool => read_max_pool_op(node),
        OperatorType::Mul => op!(Mul),
        OperatorType::Pad => op!(Pad),
        OperatorType::Pow => op!(Pow),
        OperatorType::Range => op!(Range),
        OperatorType::ReduceL2 => read_reduce_l2_op(node),
        OperatorType::ReduceMean => read_reduce_mean_op(node),
        OperatorType::Relu => op!(Relu),
        OperatorType::Reshape => read_reshape_op(node),
        OperatorType::Resize => read_resize_op(node),
        OperatorType::Shape => op!(Shape),
        OperatorType::Sigmoid => op!(Sigmoid),
        OperatorType::Sin => op!(Sin),
        OperatorType::Slice => op!(Slice),
        OperatorType::Softmax => read_softmax_op(node),
        OperatorType::Split => read_split_op(node),
        OperatorType::Sqrt => op!(Sqrt),
        OperatorType::Squeeze => op!(Squeeze),
        OperatorType::Sub => op!(Sub),
        OperatorType::Tanh => op!(Tanh),
        OperatorType::Transpose => read_transpose_op(node),
        OperatorType::Unsqueeze => op!(Unsqueeze),
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

                let mut inputs: Vec<Option<NodeId>> = Vec::new();
                if let Some(op_input_ids) = operator.inputs() {
                    for node_index in op_input_ids.iter() {
                        if node_index < 0 {
                            inputs.push(None);
                            continue;
                        }
                        let index_usize = node_index as usize;
                        if let Some(node_id) = node_id_from_index.get(&index_usize) {
                            inputs.push(Some(*node_id))
                        } else {
                            return Err("Operator input is invalid".to_string());
                        }
                    }
                }

                let mut outputs: Vec<Option<NodeId>> = Vec::new();
                if let Some(op_output_ids) = operator.outputs() {
                    for node_index in op_output_ids.iter() {
                        if node_index < 0 {
                            outputs.push(None);
                            continue;
                        }
                        let index_usize = node_index as usize;
                        if let Some(node_id) = node_id_from_index.get(&index_usize) {
                            outputs.push(Some(*node_id))
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
                    let tensor = Tensor::from_data(&shape, data);
                    graph.add_constant(node.name(), tensor)
                } else if let Some(int_data) = constant.data_as_int_data() {
                    let data: Vec<i32> = int_data.data().iter().collect();
                    let tensor = Tensor::from_data(&shape, data);
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

    use crate::graph::RunError;
    use crate::model::Model;
    use crate::model_builder::{ModelBuilder, OpType};
    use crate::ops;
    use crate::ops::{CoordTransformMode, NearestMode, OpError, Padding, ResizeMode, Scalar};
    use crate::tensor;
    use crate::tensor::{from_data, from_scalar, TensorLayout};

    fn generate_model_buffer() -> Vec<u8> {
        let mut builder = ModelBuilder::new();

        let const_val = from_data(&[1, 2, 2], vec![0.5, -0.5, 0.1, -0.1]);
        let const_node = builder.add_float_constant(&const_val);
        let input_node = builder.add_value("input");
        let output_node = builder.add_value("output");

        builder.add_input(input_node);
        builder.add_output(output_node);

        let concat_out = builder.add_value("concat_out");
        builder.add_operator(
            "concat",
            OpType::Concat(ops::Concat { dim: 0 }),
            &[const_node, input_node].map(Some),
            &[concat_out],
        );
        builder.add_operator("relu", OpType::Relu, &[Some(concat_out)], &[output_node]);

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

        let input = from_data(&[1, 2, 2], vec![1., 2., -1., -2.]);
        let result = model
            .run(&[(input_id, (&input).into())], &[output_id], None)
            .unwrap();

        assert_eq!(result.len(), 1);

        let result_tensor = result[0].as_float_ref().unwrap();

        assert_eq!(result_tensor.shape(), vec![2, 2, 2]);
        assert_eq!(result_tensor.data(), vec![0.5, 0., 0.1, 0., 1., 2., 0., 0.]);
    }

    #[test]
    fn test_omitted_optional_inputs() {
        let mut builder = ModelBuilder::new();

        let output_node = builder.add_value("output");
        builder.add_output(output_node);
        builder.add_operator("shape", OpType::Shape, &[None], &[output_node]);

        let buffer = builder.finish();
        let model = Model::load(&buffer).unwrap();

        let result = model.run(&[], &[output_node as usize], None);

        assert_eq!(
            result.err(),
            Some(RunError::OperatorError {
                name: "shape".to_string(),
                error: OpError::MissingInputs
            })
        );
    }

    // This test exercises basic execution of all operators. It doesn't check
    // the results of operators, it just makes sure they can be deserialized and
    // executed successfully.
    #[test]
    fn test_all_op_types() {
        let mut builder = ModelBuilder::new();

        let input_node = builder.add_value("input");
        let input_2d = builder.add_value("input.2d");

        let kernel_val = from_data(&[1, 1, 1, 1], vec![0.5]);
        let kernel = builder.add_float_constant(&kernel_val);

        // Names of all operator output nodes.
        let mut op_outputs = Vec::new();

        let mut add_operator =
            |builder: &mut ModelBuilder, name: &str, op: OpType, input_nodes: &[Option<u32>]| {
                let output_name = format!("{}_out", name);
                let op_output_node = builder.add_value(&output_name);
                builder.add_operator(name, op, input_nodes, &[op_output_node]);
                op_outputs.push(output_name);
                op_output_node
            };

        // Add a new operator node and associated output value node to the model.
        //
        // Returns the node ID of the output node.
        macro_rules! add_operator {
            ($op_name:ident, $op_inputs:expr) => {
                add_operator(
                    &mut builder,
                    stringify!($op_name),
                    OpType::$op_name,
                    &$op_inputs.map(Some),
                )
            };

            ($op_name:ident, $op_inputs:expr, $attrs: tt) => {
                add_operator(
                    &mut builder,
                    stringify!($op_name),
                    OpType::$op_name(ops::$op_name $attrs),
                    &$op_inputs.map(Some),
                )
            };
        }

        add_operator!(Add, [input_node, input_node]);
        add_operator!(ArgMax, [input_node], { axis: 3, keep_dims: false });
        add_operator!(ArgMin, [input_node], { axis: 3, keep_dims: false });
        add_operator!(AveragePool, [input_node], {
            kernel_size: [2, 2],
            strides: [2, 2],
            padding: Padding::Fixed([0, 0, 0, 0]),
        });

        // Dummy value for BatchNormalization inputs which are vectors with
        // per-channel values.
        let batch_norm_param_val = tensor!([1.0]);
        let batch_norm_param = builder.add_float_constant(&batch_norm_param_val);
        add_operator!(
            BatchNormalization,
            [
                input_node,
                batch_norm_param, /* scale */
                batch_norm_param, /* bias */
                batch_norm_param, /* mean */
                batch_norm_param, /* variance */
            ],
            { epsilon: 1e-5 }
        );

        add_operator!(Cast, [input_node], { to: ops::DataType::Float });

        let clip_min = builder.add_float_constant(&tensor!(1.));
        let clip_max = builder.add_float_constant(&tensor!(5.));
        add_operator!(Clip, [input_node, clip_min, clip_max]);
        add_operator!(Concat, [input_node, input_node], { dim: 0 });

        let shape = builder.add_int_constant(&from_data(&[3], vec![1, 5, 10]));
        add_operator!(ConstantOfShape, [shape], { value: Scalar::Int(42) });

        add_operator!(Conv, [input_node, kernel], {
            padding: Padding::Fixed([1, 1, 1, 1]),
            groups: 1,
            strides: [1, 1],
        });

        add_operator!(ConvTranspose, [input_node, kernel], { strides: [2, 2] });
        add_operator!(Cos, [input_node]);
        add_operator!(Div, [input_node, input_node]);
        add_operator!(Equal, [input_node, input_node]);
        add_operator!(Erf, [input_node]);

        let expand_shape_val = tensor!([2, 2, 3, 3]);
        let expand_shape = builder.add_int_constant(&expand_shape_val);
        add_operator!(Expand, [input_node, expand_shape]);

        add_operator!(Flatten, [input_node], { axis: 1 });

        let gather_indices_val = from_data(&[1], vec![0]);
        let gather_indices = builder.add_int_constant(&gather_indices_val);
        add_operator!(Gather, [input_node, gather_indices], { axis: 0 });

        add_operator!(Gemm, [input_2d, input_2d], {
            alpha: 1.0,
            beta: 1.0,
            transpose_a: false,
            transpose_b: false,
        });
        add_operator!(GlobalAveragePool, [input_node]);
        add_operator!(Greater, [input_node, input_node]);
        add_operator!(Identity, [input_node]);
        add_operator!(LeakyRelu, [input_node], { alpha: 0.01 });
        add_operator!(Less, [input_node, input_node]);
        add_operator!(LessOrEqual, [input_node, input_node]);
        add_operator!(Log, [input_node]);
        add_operator!(MatMul, [input_2d, input_2d]);
        add_operator!(MaxPool, [input_node], {
            kernel_size: [2, 2],
            strides: [2, 2],
            padding: Padding::Fixed([0, 0, 0, 0]),
        });
        add_operator!(Mul, [input_node, input_node]);

        let pads = builder.add_int_constant(&from_data(&[8], vec![0, 0, 1, 1, 0, 0, 1, 1]));
        add_operator!(Pad, [input_node, pads]);
        add_operator!(Pow, [input_node, input_node]);

        let range_start_node = builder.add_value("range_start");
        let range_limit_node = builder.add_value("range_limit");
        let range_delta_node = builder.add_value("range_delta");
        let range_out = add_operator!(
            Range,
            [range_start_node, range_limit_node, range_delta_node]
        );

        add_operator!(ReduceMean, [input_node], {
            axes: None,
            keep_dims: false,
        });
        add_operator!(Relu, [input_node]);

        let new_shape = builder.add_int_constant(&from_data(&[1], vec![9]));
        add_operator!(Reshape, [input_node, new_shape], {
            allow_zero: false,
        });

        let resize_roi_val = tensor!([0., 0., 0., 0., 1., 1., 1., 1.]);
        let resize_scales_val = tensor!([1., 1., 2., 2.]);
        let resize_roi = builder.add_float_constant(&resize_roi_val);
        let resize_scales = builder.add_float_constant(&resize_scales_val);
        add_operator!(Resize, [input_node, resize_roi, resize_scales], {
            mode: ResizeMode::Nearest,
            nearest_mode: NearestMode::default(),
            coord_mode: CoordTransformMode::default()
        });

        add_operator!(Shape, [input_node]);
        add_operator!(Sigmoid, [input_node]);
        add_operator!(Sin, [input_node]);

        let const_0 = builder.add_int_constant(&from_data(&[1], vec![0]));
        let const_1 = builder.add_int_constant(&from_data(&[1], vec![1]));
        add_operator!(Slice, [input_node, const_0, const_1, const_0]);

        add_operator!(Softmax, [input_node], { axis: 1 });
        add_operator!(Sqrt, [input_node]);
        add_operator!(Squeeze, [input_node]);

        let split_splits = builder.add_int_constant(&tensor!([1, 2]));
        let split_out_1 = builder.add_value("Split_out_1");
        let split_out_2 = builder.add_value("Split_out_2");
        builder.add_operator(
            "Split",
            OpType::Split(ops::Split { axis: 1 }),
            &[input_2d, split_splits].map(Some),
            &[split_out_1, split_out_2],
        );

        add_operator!(Sub, [input_node, input_node]);
        add_operator!(Tanh, [input_node]);
        add_operator!(Transpose, [input_node], { perm: None });

        let unsqueeze_axes = builder.add_int_constant(&tensor!([0, 4]));
        add_operator!(Unsqueeze, [input_node, unsqueeze_axes]);

        let where_cond = builder.add_value("where_cond");
        let where_x = builder.add_value("where_x");
        let where_y = builder.add_value("where_y");
        let where_out = add_operator!(Where, [where_cond, where_x, where_y]);

        let buffer = builder.finish();

        let model = Model::load(&buffer).unwrap();

        // Most ops are tested with a 4D input (eg. NCHW image). A few require
        // different shapes are tested separately.
        let input = from_data(&[1, 1, 3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        for output in op_outputs {
            if [
                "Gemm_out",
                "MatMul_out",
                "Split_out_1",
                "Split_out_2",
                "Range_out",
                "Where_out",
            ]
            .contains(&output.as_str())
            {
                // This op requires special handling. See below.
                continue;
            }

            let output_id = model.find_node(&output).unwrap();
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
        let outputs = vec!["Gemm_out", "MatMul_out", "Split_out_1", "Split_out_2"];
        let input = from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

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
        let x = tensor!([1, 2, 3]);
        let y = tensor!([4, 5, 6]);
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
