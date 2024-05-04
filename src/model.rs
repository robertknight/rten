extern crate flatbuffers;

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};

use rten_tensor::Tensor;
use smallvec::smallvec;

use crate::env::str_as_bool;
use crate::graph::{Dimension, Graph, Node, NodeId, RunError, RunOptions};
use crate::model_metadata::ModelMetadata;
use crate::ops;
use crate::ops::{
    BoxOrder, CoordTransformMode, DataType, Direction, Input, NearestMode, Operator, Output,
    Padding, ResizeMode, Scalar, ScatterReduction,
};
use crate::schema_generated as sg;
use crate::schema_generated::{root_as_model, OperatorNode, OperatorType, PadMode};
use crate::timing::TimingSort;

/// The central type used to execute RTen machine learning models.
///
/// Models are loaded from `.rten` format model files using [Model::load] and
/// executed using [Model::run] or one of the other `run_*` methods. They
/// take a list of tensor views as inputs, perform a series of computations and
/// return one or more output tensors. `.rten` models use
/// [FlatBuffers](https://github.com/google/flatbuffers) and are conceptually
/// similar to the `.ort` format used by ONNX Runtime and `.tflite` used by
/// TensorFlow Lite.
///
/// RTen models are logically graphs consisting of three types of nodes:
///
///  - Values which are supplied or generated at runtime
///  - Constants which are the weights, biases and other parameters of the
///    model. Their values are determined when the model is trained.
///  - Operators which combine the values and constants using operations such
///    as matrix multiplication, convolution etc.
///
/// Some of these nodes are designated as inputs and outputs. The IDs of these
/// nodes can be obtained using [Model::input_ids] and [Model::output_ids].
/// These IDs are then used when calling [Model::run]. Model execution consists
/// of generating a plan which starts with the input nodes, and executes the
/// necessary operators to generate the requested outputs.
///
/// ## Partial evaluation
///
/// Some models, such as transformer decoders, are evaluated repeatedly in a
/// loop. If such models have inputs which are constant in each iteration of the
/// loop, execution can be sped up by using partial evaluation. This involves
/// evaluating the part of the graph that depends only on the constant inputs
/// once, outside the loop. To do this use [Model::partial_run].
///
/// ## Custom operator registries
///
/// By default all supported ONNX operators are available for use by the model.
/// You can reduce binary size and compilation time by loading a model with
/// only a subset of operators enabled. See [Model::load_with_ops].
pub struct Model {
    node_ids: HashMap<String, NodeId>,
    input_ids: Vec<NodeId>,
    output_ids: Vec<NodeId>,
    graph: Graph,
    metadata: ModelMetadata,
}

/// Provides access to metadata about a graph node.
pub struct NodeInfo<'a> {
    node: &'a Node,
}

impl<'a> NodeInfo<'a> {
    /// Return the unique name associated with the node, if present.
    pub fn name(&self) -> Option<&str> {
        self.node.name()
    }

    /// Return the tensor shape associated with a node.
    ///
    /// The shape can be a combination of fixed values and symbolic names.
    pub fn shape(&self) -> Option<Vec<Dimension>> {
        self.node.shape()
    }
}

/// Parse profiling flags from the `RTEN_TIMING` environment variable and
/// update the graph run configuration `opts`.
///
/// This env var is a space-separated sequence of `key=value` pairs.
fn parse_timing_config(config: &str, opts: &mut RunOptions) {
    opts.timing = true;

    for token in config.split_ascii_whitespace() {
        if let Some((key, val)) = token.split_once('=') {
            let (key, val) = (key.trim(), val.trim());

            match key {
                "by-shape" => opts.timing_by_shape = str_as_bool(val),
                "sort" => match val {
                    "name" => opts.timing_sort = TimingSort::ByName,
                    "time" => opts.timing_sort = TimingSort::ByTime,
                    _ => eprintln!("Unrecognized sort order \"{}\"", val),
                },
                _ => {
                    eprintln!("Unrecognized timing option \"{}\"", key);
                }
            }
        }
    }
}

impl Model {
    /// Load a serialized model.
    ///
    /// The model will have all of the built-in operators available to it (see
    /// [OpRegistry::with_all_ops]).
    pub fn load(data: &[u8]) -> Result<Model, ModelLoadError> {
        let registry = OpRegistry::with_all_ops();
        Self::load_with_ops(data, &registry)
    }

    /// Load a serialized model with a custom operator registry.
    pub fn load_with_ops(data: &[u8], registry: &OpRegistry) -> Result<Model, ModelLoadError> {
        let model = root_as_model(data).map_err(ModelLoadError::ParseFailed)?;

        if model.schema_version() != 1 {
            return Err(ModelLoadError::SchemaVersionUnsupported);
        }

        let mut graph = Graph::new();

        let node_count = model.graph().nodes().map(|ns| ns.len()).unwrap_or(0);

        // Map of model node name to graph node ID
        let mut node_id_from_name: HashMap<String, NodeId> = HashMap::with_capacity(node_count);

        // Map of model node index to graph node ID
        let mut node_id_from_index: HashMap<usize, NodeId> = HashMap::with_capacity(node_count);

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
                    let op = registry
                        .read_op(&operator)
                        .map_err(ModelLoadError::OperatorInvalid)?;

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
                                return Err(ModelLoadError::GraphError(
                                    "operator input is invalid".to_string(),
                                ));
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
                                return Err(ModelLoadError::GraphError(
                                    "operator output is invalid".to_string(),
                                ));
                            }
                        }
                    }

                    let graph_node = graph.add_op(node.name(), op, &inputs, &outputs);

                    add_node_id(node.name(), graph_node);
                    node_id_from_index.insert(node_index, graph_node);
                } else if let Some(value_node) = node.data_as_value_node() {
                    let shape: Option<Vec<Dimension>> = value_node.shape().map(|shape| {
                        shape
                            .iter()
                            .map(|dim| {
                                if let Some(name) = dim.name() {
                                    Dimension::Symbolic(name.to_string())
                                } else {
                                    Dimension::Fixed(dim.value() as usize)
                                }
                            })
                            .collect()
                    });
                    let graph_node = graph.add_value(node.name(), shape);

                    add_node_id(node.name(), graph_node);
                    node_id_from_index.insert(node_index, graph_node);
                } else if let Some(constant) = node.data_as_constant_node() {
                    let shape: Vec<usize> = constant.shape().iter().map(|x| x as usize).collect();
                    let graph_node = if let Some(float_data) = constant.data_as_float_data() {
                        let data: Vec<f32> = vec_from_flatbuffers_vec(float_data.data());
                        let tensor = Tensor::from_data(&shape, data);
                        graph.add_constant(node.name(), tensor)
                    } else if let Some(int_data) = constant.data_as_int_data() {
                        let data: Vec<i32> = vec_from_flatbuffers_vec(int_data.data());
                        let tensor = Tensor::from_data(&shape, data);
                        graph.add_constant(node.name(), tensor)
                    } else {
                        return Err(ModelLoadError::GraphError(
                            "unsupported constant data type".to_string(),
                        ));
                    };

                    add_node_id(node.name(), graph_node);
                    node_id_from_index.insert(node_index, graph_node);
                } else {
                    return Err(ModelLoadError::GraphError("unknown node type".to_string()));
                }
            }
        }

        let metadata = model
            .metadata()
            .map(ModelMetadata::deserialize)
            .unwrap_or_default();

        let model = Model {
            node_ids: node_id_from_name,
            input_ids,
            output_ids,
            graph,
            metadata,
        };
        Ok(model)
    }

    /// Find a node in the model's graph given its string name.
    pub fn find_node(&self, id: &str) -> Option<NodeId> {
        self.node_ids.get(id).copied()
    }

    /// Find a node in the model's graph given its string name.
    ///
    /// This is a convenience method which is like [Model::find_node] but
    /// returns an error that includes the node's name if the node is not found.
    pub fn node_id(&self, id: &str) -> Result<NodeId, RunError> {
        self.node_ids
            .get(id)
            .copied()
            .ok_or_else(|| RunError::InvalidNodeName(id.to_string()))
    }

    /// Return metadata about a node in the model's graph.
    pub fn node_info(&self, id: NodeId) -> Option<NodeInfo> {
        self.graph.get_node(id).map(|node| NodeInfo { node })
    }

    /// Return metadata about the model.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Return the IDs of input nodes.
    pub fn input_ids(&self) -> &[NodeId] {
        &self.input_ids
    }

    /// Return the IDs of output nodes.
    pub fn output_ids(&self) -> &[NodeId] {
        &self.output_ids
    }

    /// Return the total number of parameters in the model's weights.
    pub fn total_params(&self) -> usize {
        self.graph.total_params()
    }

    /// Convenience method that returns the expected input shape for the index'th input.
    ///
    /// The shape may contain a mix of fixed and symbolic dimensions.
    pub fn input_shape(&self, index: usize) -> Option<Vec<Dimension>> {
        let input_id = self.input_ids.get(index)?;
        let node_info = self.node_info(*input_id)?;
        node_info.shape()
    }

    /// Execute the model and return the outputs specified by `outputs`.
    ///
    /// This method allows for running a model with a variable number of inputs
    /// and outputs of different types. See [Model::run_one] or [Model::run_n]
    /// for the common case of running a model with a single or statically
    /// known number of inputs and outputs.
    ///
    /// The input and output nodes are specified via IDs looked up via `find_node`.
    pub fn run(
        &self,
        inputs: &[(NodeId, Input)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let mut opts = opts.unwrap_or_default();
        if let Some(timing_var) = env::var_os("RTEN_TIMING") {
            let timing_var = timing_var.to_string_lossy();
            parse_timing_config(&timing_var, &mut opts);
        }
        self.graph.run(inputs, outputs, Some(opts))
    }

    /// Run a model and retrieve `N` outputs.
    ///
    /// This is a simplified version of [Model::run] for the common case of
    /// executing a model with a statically known number of outputs.
    pub fn run_n<const N: usize>(
        &self,
        inputs: &[(NodeId, Input)],
        outputs: [NodeId; N],
        opts: Option<RunOptions>,
    ) -> Result<[Output; N], RunError> {
        let result = self.run(inputs, &outputs, opts)?;
        Ok(result.try_into().expect("wrong output count"))
    }

    /// Run a model with a single input and output.
    ///
    /// This is a simplified version of [Model::run] for the common case of
    /// executing a model with a single input and output.
    pub fn run_one(&self, input: Input, opts: Option<RunOptions>) -> Result<Output, RunError> {
        let &input_id = self.input_ids().first().ok_or(RunError::InvalidNodeId)?;
        let &output_id = self.output_ids().first().ok_or(RunError::InvalidNodeId)?;
        self.run_n(&[(input_id, input)], [output_id], opts)
            .map(|[result]| result)
    }

    /// Run the model using an incomplete set of inputs.
    ///
    /// Unlike [`run`](Model::run) this will not fail if some values required to
    /// compute `outputs` are missing. Instead it will compute as many
    /// intermediate values as possible using the provided inputs and return the
    /// leaf values of the subgraph that was executed. These intermediate
    /// outputs can then be passed to future calls to [`run`](Model::run) when
    /// the other inputs are available.
    ///
    /// This method can speed up autoregressive / recurrent models where the
    /// model is run in a loop during inference, but some inputs are constant
    /// across each iteration of the loop. In such cases, execution times can be
    /// reduced by performing a `partial_run` once outside the loop, providing
    /// the constant inputs, and the results can be provided together with the
    /// the remaining inputs to `run` calls inside the loop.
    pub fn partial_run(
        &self,
        inputs: &[(NodeId, Input)],
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<(NodeId, Output)>, RunError> {
        self.graph.partial_run(inputs, outputs, opts)
    }
}

fn padding_from_attrs(mode: PadMode, pads: Option<flatbuffers::Vector<'_, u32>>) -> Padding {
    match (mode, pads) {
        (PadMode::Same, _) => Padding::Same,
        (PadMode::Fixed, Some(pads)) => Padding::Fixed(pads.iter().map(|p| p as usize).collect()),
        _ => Padding::Fixed(smallvec!(0; 4)),
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

/// Result of deserializing an operator node from a model file.
pub type ReadOpResult = Result<Box<dyn Operator + Send + Sync>, ReadOpError>;

/// A function that deserializes an operator node.
pub type ReadOpFunction = dyn Fn(&OperatorNode) -> ReadOpResult;

/// Trait that that creates the default/built-in implementation of an operator,
/// for use with [OpRegistry::register_op].
///
/// This trait is implemented for all operators in [crate::ops].
pub trait DefaultOperatorFactory {
    /// Return the type enum value for this operator.
    fn op_type() -> sg::OperatorType;

    /// Function which reads an `OperatorNode` struct from a model file and
    /// creates an instance of the operator as a `Box<dyn Operator>`.
    fn factory() -> Box<ReadOpFunction>;
}

/// Implement `DefaultOperatorFactory` for a built-in operator factory.
macro_rules! impl_default_factory {
    ($op:ident) => {
        impl DefaultOperatorFactory for ops::$op {
            fn op_type() -> OperatorType {
                OperatorType::$op
            }

            fn factory() -> Box<ReadOpFunction> {
                Box::new(move |_| Ok(Box::new(ops::$op {})))
            }
        }
    };

    ($op:ident, $factory:ident) => {
        impl DefaultOperatorFactory for ops::$op {
            fn op_type() -> OperatorType {
                OperatorType::$op
            }

            fn factory() -> Box<ReadOpFunction> {
                Box::new($factory)
            }
        }
    };
}

impl_default_factory!(Abs);
impl_default_factory!(Acos);
impl_default_factory!(Add);
impl_default_factory!(And);
impl_default_factory!(ArgMax, read_arg_max_op);
impl_default_factory!(ArgMin, read_arg_min_op);
impl_default_factory!(Asin);
impl_default_factory!(Atan);
impl_default_factory!(AveragePool, read_average_pool_op);
impl_default_factory!(BatchNormalization, read_batch_normalization_op);
impl_default_factory!(Cast, read_cast_op);
impl_default_factory!(Ceil);
impl_default_factory!(Clip);
impl_default_factory!(Concat, read_concat_op);
impl_default_factory!(Conv, read_conv_op);
impl_default_factory!(ConstantOfShape, read_constant_of_shape_op);
impl_default_factory!(ConvTranspose, read_conv_transpose_op);
impl_default_factory!(Cos);
impl_default_factory!(CumSum);
impl_default_factory!(Div);
impl_default_factory!(Elu, read_elu_op);
impl_default_factory!(Equal);
impl_default_factory!(Erf);
impl_default_factory!(Exp);
impl_default_factory!(Expand);
impl_default_factory!(Flatten, read_flatten_op);
impl_default_factory!(Floor);
impl_default_factory!(Gather, read_gather_op);
impl_default_factory!(GatherElements, read_gather_elements_op);
impl_default_factory!(Gemm, read_gemm_op);
impl_default_factory!(GlobalAveragePool);
impl_default_factory!(Greater);
impl_default_factory!(GreaterOrEqual);
impl_default_factory!(GRU, read_gru_op);
impl_default_factory!(HardSigmoid, read_hard_sigmoid_op);
impl_default_factory!(HardSwish);
impl_default_factory!(Identity);
impl_default_factory!(InstanceNormalization, read_instance_normalization_op);
impl_default_factory!(LayerNormalization, read_layer_normalization_op);
impl_default_factory!(LeakyRelu, read_leaky_relu_op);
impl_default_factory!(Less);
impl_default_factory!(LessOrEqual);
impl_default_factory!(Log);
impl_default_factory!(LogSoftmax, read_log_softmax_op);
impl_default_factory!(LSTM, read_lstm_op);
impl_default_factory!(MatMul);
impl_default_factory!(Max);
impl_default_factory!(MaxPool, read_max_pool_op);
impl_default_factory!(Mean);
impl_default_factory!(Min);
impl_default_factory!(Mod, read_mod_op);
impl_default_factory!(Mul);
impl_default_factory!(Neg);
impl_default_factory!(NonMaxSuppression, read_non_max_suppression_op);
impl_default_factory!(NonZero);
impl_default_factory!(Not);
impl_default_factory!(OneHot, read_onehot_op);
impl_default_factory!(Or);
impl_default_factory!(Pad);
impl_default_factory!(Pow);

#[cfg(feature = "random")]
impl_default_factory!(RandomNormal, read_random_normal_op);
#[cfg(feature = "random")]
impl_default_factory!(RandomNormalLike, read_random_normal_like_op);
#[cfg(feature = "random")]
impl_default_factory!(RandomUniform, read_random_uniform_op);
#[cfg(feature = "random")]
impl_default_factory!(RandomUniformLike, read_random_uniform_like_op);

impl_default_factory!(Range);
impl_default_factory!(Reciprocal);
impl_default_factory!(ReduceL2, read_reduce_l2_op);
impl_default_factory!(ReduceMax, read_reduce_max_op);
impl_default_factory!(ReduceMean, read_reduce_mean_op);
impl_default_factory!(ReduceMin, read_reduce_min_op);
impl_default_factory!(ReduceProd, read_reduce_prod_op);
impl_default_factory!(ReduceSum, read_reduce_sum_op);
impl_default_factory!(ReduceSumSquare, read_reduce_sum_square_op);
impl_default_factory!(Relu);
impl_default_factory!(Reshape, read_reshape_op);
impl_default_factory!(Resize, read_resize_op);
impl_default_factory!(Round);
impl_default_factory!(ScatterElements, read_scatter_elements_op);
impl_default_factory!(ScatterND, read_scatter_nd_op);
impl_default_factory!(Shape);
impl_default_factory!(Sigmoid);
impl_default_factory!(Sign);
impl_default_factory!(Sin);
impl_default_factory!(Size);
impl_default_factory!(Slice);
impl_default_factory!(Softmax, read_softmax_op);
impl_default_factory!(Split, read_split_op);
impl_default_factory!(Sqrt);
impl_default_factory!(Squeeze);
impl_default_factory!(Sub);
impl_default_factory!(Sum);
impl_default_factory!(Tan);
impl_default_factory!(Tanh);
impl_default_factory!(Tile);
impl_default_factory!(TopK, read_topk_op);
impl_default_factory!(Transpose, read_transpose_op);
impl_default_factory!(Trilu, read_trilu_op);
impl_default_factory!(Unsqueeze);
impl_default_factory!(Where);
impl_default_factory!(Xor);

/// Registry used to instantiate operators when loading a model file.
///
/// New registries have no operators registered by default. If you want to get
/// one that has all the built-in operators pre-registered, use
/// [OpRegistry::with_all_ops]. Alternatively you can create a new registry and
/// just selectively register the operators you need using
/// [OpRegistry::register_op]. This can be useful to reduce binary sizes or
/// customize the implementation of an operator.
#[derive(Default)]
pub struct OpRegistry {
    ops: HashMap<sg::OperatorType, Box<ReadOpFunction>>,
}

impl OpRegistry {
    /// Create a new empty registry.
    pub fn new() -> OpRegistry {
        OpRegistry {
            ops: HashMap::new(),
        }
    }

    /// Register the default/built-in implementation of an operator.
    pub fn register_op<Op: DefaultOperatorFactory>(&mut self) {
        self.register_op_with_factory(Op::op_type(), Op::factory());
    }

    /// Deserialize an operator from a model file using the operators in the
    /// registry.
    fn read_op(&self, op: &OperatorNode) -> ReadOpResult {
        self.ops
            .get(&op.type_())
            .ok_or_else(|| {
                ReadOpError::UnsupportedOperator(
                    op.type_().variant_name().unwrap_or("(unknown)").to_string(),
                )
            })
            .and_then(|read_fn| read_fn(op))
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
    pub fn with_all_ops() -> OpRegistry {
        let mut reg = OpRegistry::new();

        macro_rules! register_op {
            ($op:ident) => {
                reg.register_op::<ops::$op>()
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
        register_op!(Ceil);
        register_op!(Clip);
        register_op!(Concat);
        register_op!(Conv);
        register_op!(ConstantOfShape);
        register_op!(ConvTranspose);
        register_op!(Cos);
        register_op!(CumSum);
        register_op!(Div);
        register_op!(Elu);
        register_op!(Equal);
        register_op!(Erf);
        register_op!(Exp);
        register_op!(Expand);
        register_op!(Flatten);
        register_op!(Floor);
        register_op!(Gather);
        register_op!(GatherElements);
        register_op!(Gemm);
        register_op!(GlobalAveragePool);
        register_op!(Greater);
        register_op!(GreaterOrEqual);
        register_op!(GRU);
        register_op!(HardSigmoid);
        register_op!(HardSwish);
        register_op!(Identity);
        register_op!(InstanceNormalization);
        register_op!(LayerNormalization);
        register_op!(LeakyRelu);
        register_op!(Less);
        register_op!(LessOrEqual);
        register_op!(Log);
        register_op!(LogSoftmax);
        register_op!(LSTM);
        register_op!(MatMul);
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

        #[cfg(feature = "random")]
        register_op!(RandomNormal);
        #[cfg(feature = "random")]
        register_op!(RandomNormalLike);
        #[cfg(feature = "random")]
        register_op!(RandomUniform);
        #[cfg(feature = "random")]
        register_op!(RandomUniformLike);

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
        register_op!(Shape);
        register_op!(Sigmoid);
        register_op!(Sign);
        register_op!(Sin);
        register_op!(Size);
        register_op!(Slice);
        register_op!(Softmax);
        register_op!(Split);
        register_op!(Sqrt);
        register_op!(Squeeze);
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

/// Error type for errors that occur when de-serializing an operator.
#[derive(Debug, PartialEq)]
pub enum ReadOpError {
    /// The operator attributes were missing or of the wrong type.
    AttrError,
    /// The operator type is incorrect or unsupported.
    UnsupportedOperator(String),
}

impl Display for ReadOpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadOpError::AttrError => write!(f, "invalid attributes for operator"),
            ReadOpError::UnsupportedOperator(name) => {
                write!(f, "operator {name} is not supported or not enabled")
            }
        }
    }
}

impl Error for ReadOpError {}

/// Define a function that reads an operator with one attribute, `axis`.
macro_rules! read_axis_op {
    ($func_name:ident, $attr_method:ident, $op:ident) => {
        fn $func_name(node: &OperatorNode) -> ReadOpResult {
            let attrs = node.$attr_method().ok_or(ReadOpError::AttrError)?;
            Ok(Box::new(ops::$op {
                axis: attrs.axis() as isize,
            }))
        }
    };
}

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
        count_include_pad: attrs.count_include_pad(),
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

read_axis_op!(read_concat_op, attrs_as_concat_attrs, Concat);

fn read_conv_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_conv_attrs().ok_or(ReadOpError::AttrError)?;

    let groups = attrs.groups() as usize;
    let padding = padding_from_attrs(attrs.pad_mode(), attrs.pads());
    let strides: Vec<usize> = attrs
        .strides()
        .map(|stride| stride.iter().map(|x| x as usize).collect())
        .unwrap_or(vec![1, 1]);
    let dilations: Vec<usize> = attrs
        .dilations()
        .map(|dilation| dilation.iter().map(|x| x as usize).collect())
        .unwrap_or(vec![1, 1]);

    Ok(Box::new(ops::Conv {
        groups,
        padding,
        strides,
        dilations,
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

fn read_elu_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_elu_attrs().ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Elu {
        alpha: attrs.alpha(),
    }))
}

read_axis_op!(read_flatten_op, attrs_as_flatten_attrs, Flatten);
read_axis_op!(read_gather_op, attrs_as_gather_attrs, Gather);
read_axis_op!(
    read_gather_elements_op,
    attrs_as_gather_attrs,
    GatherElements
);

fn read_gemm_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_gemm_attrs().ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Gemm {
        alpha: attrs.alpha(),
        beta: attrs.beta(),
        transpose_a: attrs.transpose_a(),
        transpose_b: attrs.transpose_b(),
    }))
}

fn read_gru_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_gruattrs().ok_or(ReadOpError::AttrError)?;

    let hidden_size = attrs.hidden_size() as usize;
    let direction = match attrs.direction() {
        sg::RNNDirection::Forward => Direction::Forward,
        sg::RNNDirection::Reverse => Direction::Reverse,
        sg::RNNDirection::Bidirectional => Direction::Bidirectional,
        _ => Direction::Forward,
    };

    Ok(Box::new(ops::GRU {
        direction,
        hidden_size,
        linear_before_reset: attrs.linear_before_reset(),
    }))
}

fn read_hard_sigmoid_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_hard_sigmoid_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::HardSigmoid {
        alpha: attrs.alpha(),
        beta: attrs.beta(),
    }))
}

fn read_instance_normalization_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_batch_normalization_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::InstanceNormalization {
        epsilon: Some(attrs.epsilon()),
    }))
}

fn read_layer_normalization_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_layer_normalization_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::LayerNormalization {
        axis: attrs.axis() as isize,
        epsilon: Some(attrs.epsilon()),
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

read_axis_op!(read_log_softmax_op, attrs_as_softmax_attrs, LogSoftmax);

fn read_lstm_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_lstmattrs().ok_or(ReadOpError::AttrError)?;

    let hidden_size = attrs.hidden_size() as usize;
    let direction = match attrs.direction() {
        sg::RNNDirection::Forward => Direction::Forward,
        sg::RNNDirection::Reverse => Direction::Reverse,
        sg::RNNDirection::Bidirectional => Direction::Bidirectional,
        _ => Direction::Forward,
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

fn read_mod_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_mod_attrs().ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Mod { fmod: attrs.fmod() }))
}

fn read_non_max_suppression_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_non_max_suppression_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let box_order = match attrs.box_order() {
        sg::NMSBoxOrder::CenterWidthHeight => BoxOrder::CenterWidthHeight,
        sg::NMSBoxOrder::TopLeftBottomRight => BoxOrder::TopLeftBottomRight,
        _ => BoxOrder::TopLeftBottomRight,
    };
    Ok(Box::new(ops::NonMaxSuppression { box_order }))
}

read_axis_op!(read_onehot_op, attrs_as_one_hot_attrs, OneHot);

#[cfg(feature = "random")]
fn read_random_normal_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_random_normal_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let shape = attrs
        .shape()
        .map(|shape| shape.iter().map(|size| size as usize).collect())
        .unwrap_or(vec![]);

    Ok(Box::new(ops::RandomNormal {
        shape,
        mean: attrs.mean(),
        scale: attrs.scale(),
        seed: attrs.seed(),
    }))
}

#[cfg(feature = "random")]
fn read_random_normal_like_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_random_normal_like_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::RandomNormalLike {
        mean: attrs.mean(),
        scale: attrs.scale(),
        seed: attrs.seed(),
    }))
}

#[cfg(feature = "random")]
fn read_random_uniform_like_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_random_uniform_like_attrs()
        .ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::RandomUniformLike {
        high: attrs.high(),
        low: attrs.low(),
        seed: attrs.seed(),
    }))
}

#[cfg(feature = "random")]
fn read_random_uniform_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_random_uniform_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let shape = attrs
        .shape()
        .map(|shape| shape.iter().map(|size| size as usize).collect())
        .unwrap_or(vec![]);

    Ok(Box::new(ops::RandomUniform {
        shape,
        high: attrs.high(),
        low: attrs.low(),
        seed: attrs.seed(),
    }))
}

fn read_reduce_attrs(node: &OperatorNode) -> Result<(Option<Vec<i32>>, bool), ReadOpError> {
    let attrs = node
        .attrs_as_reduce_mean_attrs()
        .ok_or(ReadOpError::AttrError)?;
    let axes = attrs.axes().map(|axes| axes.iter().collect());
    let keep_dims = attrs.keep_dims();
    Ok((axes, keep_dims))
}

/// Define a function that reads `Reduce*` operators.
macro_rules! read_reduce_op {
    ($func_name:ident, $op:ident) => {
        fn $func_name(node: &OperatorNode) -> ReadOpResult {
            let (axes, keep_dims) = read_reduce_attrs(node)?;
            Ok(Box::new(ops::$op { axes, keep_dims }))
        }
    };
}

read_reduce_op!(read_reduce_l2_op, ReduceL2);
read_reduce_op!(read_reduce_max_op, ReduceMax);
read_reduce_op!(read_reduce_mean_op, ReduceMean);
read_reduce_op!(read_reduce_min_op, ReduceMin);
read_reduce_op!(read_reduce_prod_op, ReduceProd);
read_reduce_op!(read_reduce_sum_op, ReduceSum);
read_reduce_op!(read_reduce_sum_square_op, ReduceSumSquare);

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
        sg::CoordTransformMode::AlignCorners => CoordTransformMode::AlignCorners,
        _ => CoordTransformMode::default(),
    };

    Ok(Box::new(ops::Resize {
        mode,
        coord_mode,
        nearest_mode,
    }))
}

fn convert_reduction(r: sg::ScatterReduction) -> Result<Option<ScatterReduction>, ReadOpError> {
    let reduction = match r {
        sg::ScatterReduction::None => None,
        sg::ScatterReduction::Add => Some(ScatterReduction::Add),
        sg::ScatterReduction::Mul => Some(ScatterReduction::Mul),
        sg::ScatterReduction::Min => Some(ScatterReduction::Min),
        sg::ScatterReduction::Max => Some(ScatterReduction::Max),
        _ => {
            return Err(ReadOpError::AttrError);
        }
    };
    Ok(reduction)
}

fn read_scatter_elements_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_scatter_elements_attrs()
        .ok_or(ReadOpError::AttrError)?;

    Ok(Box::new(ops::ScatterElements {
        axis: attrs.axis() as isize,
        reduction: convert_reduction(attrs.reduction())?,
    }))
}

fn read_scatter_nd_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node
        .attrs_as_scatter_ndattrs()
        .ok_or(ReadOpError::AttrError)?;

    Ok(Box::new(ops::ScatterND {
        reduction: convert_reduction(attrs.reduction())?,
    }))
}

read_axis_op!(read_softmax_op, attrs_as_softmax_attrs, Softmax);
read_axis_op!(read_split_op, attrs_as_split_attrs, Split);

fn read_topk_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_top_kattrs().ok_or(ReadOpError::AttrError)?;
    let largest = attrs.largest();
    let sorted = attrs.sorted();
    let axis = attrs.axis();
    Ok(Box::new(ops::TopK {
        axis: Some(axis as isize),
        largest,
        sorted,
    }))
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

fn read_trilu_op(node: &OperatorNode) -> ReadOpResult {
    let attrs = node.attrs_as_trilu_attrs().ok_or(ReadOpError::AttrError)?;
    Ok(Box::new(ops::Trilu {
        upper: attrs.upper(),
    }))
}

/// Errors reported by [Model::load].
#[derive(Debug, PartialEq)]
pub enum ModelLoadError {
    SchemaVersionUnsupported,

    /// An error occurred parsing the FlatBuffers file.
    ParseFailed(flatbuffers::InvalidFlatbuffer),

    /// An error occurred deserializing an operator.
    OperatorInvalid(ReadOpError),

    /// An error occurred while traversing the model's graph to instantiate
    /// nodes and connections.
    GraphError(String),
}

impl Display for ModelLoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelLoadError::SchemaVersionUnsupported => write!(f, "unsupported schema version"),
            ModelLoadError::ParseFailed(e) => write!(f, "parse error: {e}"),
            ModelLoadError::OperatorInvalid(e) => write!(f, "operator error: {e}"),
            ModelLoadError::GraphError(e) => write!(f, "graph error: {e}"),
        }
    }
}

impl Error for ModelLoadError {}

/// Optimized conversion of a `flatbuffers::Vector<T>` into a `Vec<T>` for
/// primitive types.
///
/// This relies on the fact that the underlying bytes are likely correctly
/// aligned for `T`. If so, we can transmute `&[u8] -> &[T]` and then benefit
/// from fast slice-to-Vec conversion.
fn vec_from_flatbuffers_vec<'a, T: Copy + flatbuffers::Follow<'a, Inner = T>>(
    fbv: flatbuffers::Vector<'a, T>,
) -> Vec<T> {
    let bytes = fbv.bytes();
    if bytes.as_ptr() as usize % std::mem::align_of::<T>() == 0 {
        // Safety: We checked that the data is correctly aligned, and we trust
        // `flatbuffers::Vector<T>` that its bytes contain `fbv.len()` Ts.
        let typed_slice = unsafe {
            let typed_slice = std::mem::transmute::<&[u8], &[T]>(bytes);
            &typed_slice[..fbv.len()]
        };
        typed_slice.to_vec()
    } else {
        fbv.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    extern crate flatbuffers;

    use rten_tensor::prelude::*;
    use rten_tensor::{tensor, Tensor};

    use crate::graph::{Dimension, RunError};
    use crate::model::Model;
    use crate::model_builder::{MetadataArgs, ModelBuilder, OpType};
    use crate::ops;
    use crate::ops::{
        BoxOrder, CoordTransformMode, NearestMode, OpError, Output, ResizeMode, Scalar,
    };
    use crate::{ModelLoadError, OpRegistry, ReadOpError};

    fn generate_model_buffer() -> Vec<u8> {
        let mut builder = ModelBuilder::new();

        let const_val = Tensor::from_data(&[1, 2, 2], vec![0.5, -0.5, 0.1, -0.1]);
        let const_node = builder.add_float_constant(&const_val);

        let input_shape: Vec<Dimension> = const_val
            .shape()
            .iter()
            .copied()
            .map(Dimension::Fixed)
            .collect();
        let input_node = builder.add_value("input", Some(&input_shape));
        let output_node = builder.add_value("output", None);

        builder.add_input(input_node);
        builder.add_output(output_node);

        let concat_out = builder.add_value("concat_out", None);
        builder.add_operator(
            "concat",
            OpType::Concat(ops::Concat { axis: 0 }),
            &[const_node, input_node].map(Some),
            &[concat_out],
        );
        builder.add_operator("relu", OpType::Relu, &[Some(concat_out)], &[output_node]);

        builder.add_metadata(MetadataArgs {
            onnx_hash: Some("abc".to_string()),
        });

        builder.finish()
    }

    #[test]
    fn test_model_input_output_ids() {
        let buffer = generate_model_buffer();

        let model = Model::load(&buffer).unwrap();

        // Valid model IDs
        let input_id = model.find_node("input").unwrap();
        let output_id = model.find_node("output").unwrap();

        assert_eq!(model.input_ids(), &[input_id]);
        assert_eq!(model.output_ids(), &[output_id]);

        // Get the same node ID via a convenience method which returns a
        // Result.
        assert_eq!(model.node_id("input"), Ok(input_id));

        // Invalid model ID
        assert_eq!(model.find_node("does_not_exist"), None);
        assert_eq!(
            model.node_id("does_not_exist"),
            Err(RunError::InvalidNodeName("does_not_exist".to_string()))
        );
    }

    #[test]
    fn test_unsupported_operator() {
        let buffer = generate_model_buffer();
        let registry = OpRegistry::new();
        let result = Model::load_with_ops(&buffer, &registry);
        assert_eq!(
            result.err(),
            Some(ModelLoadError::OperatorInvalid(
                ReadOpError::UnsupportedOperator("Concat".to_string())
            ))
        );
    }

    #[test]
    fn test_shape_info() {
        let buffer = generate_model_buffer();
        let model = Model::load(&buffer).unwrap();
        let input_id = model.input_ids()[0];

        let shape = model
            .node_info(input_id)
            .and_then(|ni| ni.shape())
            .expect("input shape missing");
        assert_eq!(shape, &[1, 2, 2].map(Dimension::Fixed));
    }

    #[test]
    fn test_metadata() {
        let buffer = generate_model_buffer();
        let model = Model::load(&buffer).unwrap();
        assert_eq!(model.metadata().onnx_hash(), Some("abc"));
        assert_eq!(model.metadata().description(), None);
    }

    #[test]
    fn test_input_shape() {
        let buffer = generate_model_buffer();
        let model = Model::load(&buffer).unwrap();
        assert_eq!(
            model.input_shape(0),
            Some(vec![
                Dimension::Fixed(1),
                Dimension::Fixed(2),
                Dimension::Fixed(2),
            ])
        );
    }

    #[test]
    fn test_load_and_run_model() {
        let buffer = generate_model_buffer();

        let model = Model::load(&buffer).unwrap();
        let input_id = model.input_ids()[0];
        let output_id = model.output_ids()[0];

        let input = Tensor::from_data(&[1, 2, 2], vec![1., 2., -1., -2.]);

        // Test a normal model run.
        let mut result = model
            .run(&[(input_id, (&input).into())], &[output_id], None)
            .unwrap();

        assert_eq!(result.len(), 1);
        let result_tensor = result.remove(0).into_float().unwrap();
        assert_eq!(result_tensor.shape(), &[2, 2, 2]);
        assert_eq!(result_tensor.to_vec(), &[0.5, 0., 0.1, 0., 1., 2., 0., 0.]);

        // Test a partial run. Since we are providing all inputs, this works the
        // same as `Model::run`. See `Graph::partial_run` tests for other cases.
        let partial_run_result = model
            .partial_run(&[(input_id, (&input).into())], &[output_id], None)
            .unwrap();
        assert_eq!(
            partial_run_result,
            vec![(output_id, Output::FloatTensor(result_tensor))]
        );
    }

    #[test]
    fn test_run_one() {
        let buffer = generate_model_buffer();
        let model = Model::load(&buffer).unwrap();

        let input = tensor!((1, 2, 2); [1., 2., -1., -2.]);
        let result: Tensor<f32> = model
            .run_one((&input).into(), None)
            .unwrap()
            .try_into()
            .unwrap();

        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.to_vec(), &[0.5, 0., 0.1, 0., 1., 2., 0., 0.]);
    }

    #[test]
    fn test_omitted_optional_inputs() {
        let mut builder = ModelBuilder::new();

        let output_node = builder.add_value("output", None);
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

        let input_node = builder.add_value("input", None);
        let input_2d = builder.add_value("input.2d", None);
        let input_bool = builder.add_value("input.bool", None);

        // 4D shape used as the primary input to test most operators (eg. NCHW image). A few
        // require a different shape.
        let input_shape = [1, 1, 3, 3];

        let kernel_val = Tensor::from_data(&[1, 1, 1, 1], vec![0.5]);
        let kernel = builder.add_float_constant(&kernel_val);

        // Names of all operator output nodes.
        let mut op_outputs = Vec::new();

        let mut add_operator =
            |builder: &mut ModelBuilder, name: &str, op: OpType, input_nodes: &[Option<u32>]| {
                let output_name = format!("{}_out", name);
                let op_output_node = builder.add_value(&output_name, None);
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

        add_operator!(Abs, [input_node]);
        add_operator!(Acos, [input_node]);
        add_operator!(Add, [input_node, input_node]);
        add_operator!(And, [input_bool, input_bool]);
        add_operator!(ArgMax, [input_node], { axis: 3, keep_dims: false });
        add_operator!(ArgMin, [input_node], { axis: 3, keep_dims: false });
        add_operator!(Asin, [input_node]);
        add_operator!(Atan, [input_node]);
        add_operator!(AveragePool, [input_node], {
            kernel_size: [2, 2],
            strides: [2, 2],
            padding: [0, 0, 0, 0].into(),
            count_include_pad: false,
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
        add_operator!(Ceil, [input_node]);

        let clip_min = builder.add_float_constant(&tensor!(1.));
        let clip_max = builder.add_float_constant(&tensor!(6.));
        add_operator!(Clip, [input_node, clip_min, clip_max]);
        add_operator!(Concat, [input_node, input_node], { axis: 0 });

        let shape = builder.add_int_constant(&Tensor::from_data(&[3], vec![1, 5, 10]));
        add_operator!(ConstantOfShape, [shape], { value: Scalar::Int(42) });

        add_operator!(Conv, [input_node, kernel], {
            dilations: vec![1, 1],
            groups: 1,
            padding: [1, 1, 1, 1].into(),
            strides: vec![1, 1],
        });

        add_operator!(ConvTranspose, [input_node, kernel], { strides: [2, 2] });
        add_operator!(Cos, [input_node]);
        add_operator!(Div, [input_node, input_node]);
        add_operator!(Elu, [input_node], { alpha: 1.0 });
        add_operator!(Equal, [input_node, input_node]);
        add_operator!(Erf, [input_node]);
        add_operator!(Exp, [input_node]);

        let expand_shape_val = tensor!([2, 2, 3, 3]);
        let expand_shape = builder.add_int_constant(&expand_shape_val);
        add_operator!(Expand, [input_node, expand_shape]);

        add_operator!(Flatten, [input_node], { axis: 1 });
        add_operator!(Floor, [input_node]);

        let gather_indices_val = Tensor::from_data(&[1], vec![0]);
        let gather_indices = builder.add_int_constant(&gather_indices_val);
        add_operator!(Gather, [input_node, gather_indices], { axis: 0 });

        let gather_elements_indices_val = Tensor::zeros(&input_shape);
        let gather_elements_indices = builder.add_int_constant(&gather_elements_indices_val);
        add_operator!(GatherElements, [input_node, gather_elements_indices], { axis: 0 });

        add_operator!(Gemm, [input_2d, input_2d], {
            alpha: 1.0,
            beta: 1.0,
            transpose_a: false,
            transpose_b: false,
        });
        add_operator!(GlobalAveragePool, [input_node]);
        add_operator!(Greater, [input_node, input_node]);
        add_operator!(GreaterOrEqual, [input_node, input_node]);
        add_operator!(HardSigmoid, [input_node], {
            alpha: 0.2,
            beta: 0.5,
        });
        add_operator!(HardSwish, [input_node]);

        // TODO - Add GRU operator

        add_operator!(Identity, [input_node]);

        let instance_norm_scale_val = tensor!([1.0]);
        let instance_norm_scale = builder.add_float_constant(&instance_norm_scale_val);
        let instance_norm_bias_val = tensor!([1.0]);
        let instance_norm_bias = builder.add_float_constant(&instance_norm_bias_val);
        add_operator!(InstanceNormalization, [
            input_node, instance_norm_scale, instance_norm_bias
        ], { epsilon: Some(1e-5) });

        let layer_norm_scale_val = tensor!([1.0]);
        let layer_norm_scale = builder.add_float_constant(&layer_norm_scale_val);
        let layer_norm_bias_val = tensor!([1.0]);
        let layer_norm_bias = builder.add_float_constant(&layer_norm_bias_val);
        add_operator!(LayerNormalization, [
            input_node, layer_norm_scale, layer_norm_bias
        ], { axis: -1, epsilon: Some(1e-5) });

        add_operator!(LeakyRelu, [input_node], { alpha: 0.01 });
        add_operator!(Less, [input_node, input_node]);
        add_operator!(LessOrEqual, [input_node, input_node]);
        add_operator!(Log, [input_node]);
        add_operator!(LogSoftmax, [input_node], { axis: 1 });

        // TODO - Add LSTM operator

        add_operator!(MatMul, [input_2d, input_2d]);
        add_operator!(Max, [input_node, input_node]);
        add_operator!(MaxPool, [input_node], {
            kernel_size: [2, 2],
            strides: [2, 2],
            padding: [0, 0, 0, 0].into(),
        });
        add_operator!(Mean, [input_node, input_node]);
        add_operator!(Min, [input_node, input_node]);
        add_operator!(Mod, [input_node, input_node], {
            fmod: false,
        });
        add_operator!(Mul, [input_node, input_node]);
        add_operator!(Neg, [input_node]);

        let nms_n_boxes = 10;
        let nms_n_classes = 20;
        let nms_boxes = builder.add_float_constant(&Tensor::zeros(&[1, nms_n_boxes, 4]));
        let nms_scores =
            builder.add_float_constant(&Tensor::zeros(&[1, nms_n_classes, nms_n_boxes]));
        let nms_max_outputs_per_class = builder.add_int_constant(&tensor!(10));
        let nms_iou_threshold = builder.add_float_constant(&tensor!(0.45));
        let nms_score_threshold = builder.add_float_constant(&tensor!(0.2));

        add_operator!(NonMaxSuppression, [nms_boxes, nms_scores, nms_max_outputs_per_class, nms_iou_threshold, nms_score_threshold], {
            box_order: BoxOrder::CenterWidthHeight,
        });

        add_operator!(NonZero, [input_node]);
        add_operator!(Not, [input_bool]);

        let onehot_indices = builder.add_int_constant(&tensor!([0, 1, 2]));
        let onehot_depth = builder.add_int_constant(&tensor!(5));
        let onehot_values = builder.add_float_constant(&tensor!([1., 0.]));
        add_operator!(OneHot, [onehot_indices, onehot_depth, onehot_values], {
            axis: -1,
        });

        add_operator!(Or, [input_bool, input_bool]);

        let pads = builder.add_int_constant(&Tensor::from_data(&[8], vec![0, 0, 1, 1, 0, 0, 1, 1]));
        add_operator!(Pad, [input_node, pads]);
        add_operator!(Pow, [input_node, input_node]);

        add_operator!(RandomNormal, [], {
            shape: vec![50, 50],
            mean: 0.,
            scale: 1.,
            seed: None,
        });
        add_operator!(RandomNormalLike, [input_node], {
            mean: 0.,
            scale: 1.,
            seed: None,
        });
        add_operator!(RandomUniform, [], {
            shape: vec![50, 50],
            low: 0.,
            high: 1.,
            seed: None,
        });
        add_operator!(RandomUniformLike, [input_node], {
            low: 0.,
            high: 1.,
            seed: None,
        });

        let range_start_node = builder.add_value("range_start", None);
        let range_limit_node = builder.add_value("range_limit", None);
        let range_delta_node = builder.add_value("range_delta", None);
        let range_out = add_operator!(
            Range,
            [range_start_node, range_limit_node, range_delta_node]
        );

        add_operator!(Reciprocal, [input_node]);
        add_operator!(ReduceMean, [input_node], {
            axes: None,
            keep_dims: false,
        });
        add_operator!(ReduceMax, [input_node], {
            axes: None,
            keep_dims: false,
        });
        add_operator!(ReduceMin, [input_node], {
            axes: None,
            keep_dims: false,
        });
        add_operator!(ReduceProd, [input_node], {
            axes: None,
            keep_dims: false,
        });
        add_operator!(ReduceSum, [input_node], {
            axes: None,
            keep_dims: false,
        });
        add_operator!(ReduceSumSquare, [input_node], {
            axes: None,
            keep_dims: false,
        });
        add_operator!(Relu, [input_node]);

        let new_shape = builder.add_int_constant(&Tensor::from_data(&[1], vec![9]));
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

        add_operator!(Round, [input_node]);

        add_operator!(Shape, [input_node]);
        add_operator!(Sigmoid, [input_node]);
        add_operator!(Sign, [input_node]);
        add_operator!(Sin, [input_node]);
        add_operator!(Size, [input_node]);

        let scatter_elem_indices_val = Tensor::zeros(&input_shape);
        let scatter_elem_indices = builder.add_int_constant(&scatter_elem_indices_val);
        let scatter_elem_updates_val = Tensor::zeros(&input_shape);
        let scatter_elem_updates = builder.add_float_constant(&scatter_elem_updates_val);
        add_operator!(
            ScatterElements,
            [input_node, scatter_elem_indices, scatter_elem_updates],
            { axis: 0, reduction: None }
        );

        let const_0 = builder.add_int_constant(&Tensor::from_data(&[1], vec![0]));
        let const_1 = builder.add_int_constant(&Tensor::from_data(&[1], vec![1]));
        add_operator!(Slice, [input_node, const_0, const_1, const_0]);

        add_operator!(Softmax, [input_node], { axis: 1 });
        add_operator!(Sqrt, [input_node]);
        add_operator!(Squeeze, [input_node]);

        let split_splits = builder.add_int_constant(&tensor!([1, 2]));
        let split_out_1 = builder.add_value("Split_out_1", None);
        let split_out_2 = builder.add_value("Split_out_2", None);
        builder.add_operator(
            "Split",
            OpType::Split(ops::Split { axis: 1 }),
            &[input_2d, split_splits].map(Some),
            &[split_out_1, split_out_2],
        );

        add_operator!(Sub, [input_node, input_node]);
        add_operator!(Sum, [input_node, input_node]);
        add_operator!(Tan, [input_node]);
        add_operator!(Tanh, [input_node]);

        let tile_repeats = builder.add_int_constant(&tensor!([1, 2, 3, 4]));
        add_operator!(Tile, [input_node, tile_repeats]);

        let topk_k = builder.add_int_constant(&tensor!(3));
        let topk_out_values = builder.add_value("TopK_out_values", None);
        let topk_out_indices = builder.add_value("TopK_out_indices", None);
        builder.add_operator(
            "TopK",
            OpType::TopK(ops::TopK {
                largest: true,
                sorted: true,
                axis: Some(-1),
            }),
            &[input_2d, topk_k].map(Some),
            &[topk_out_values, topk_out_indices],
        );

        add_operator!(Transpose, [input_node], { perm: None });

        add_operator!(Trilu, [input_node], { upper: true });

        let unsqueeze_axes = builder.add_int_constant(&tensor!([0, 4]));
        add_operator!(Unsqueeze, [input_node, unsqueeze_axes]);

        let where_cond = builder.add_value("where_cond", None);
        let where_x = builder.add_value("where_x", None);
        let where_y = builder.add_value("where_y", None);
        let where_out = add_operator!(Where, [where_cond, where_x, where_y]);

        add_operator!(Xor, [input_bool, input_bool]);

        let buffer = builder.finish();

        let model = Model::load(&buffer).unwrap();

        // Most ops are tested with one of several standard inputs:
        //
        //  - 4D float tensor (like an NCHW image)
        //  - Bool-ish int tensor
        //
        // A few require different shapes are tested separately.
        let input = Tensor::from_data(&input_shape, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input_bool_data: Tensor<i32> = tensor!([0, 1, 1]);
        for output in op_outputs {
            if [
                "Gemm_out",
                "MatMul_out",
                "Range_out",
                "Split_out_1",
                "Split_out_2",
                "TopK_out_indices",
                "TopK_out_values",
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
                    &[
                        (input_node as usize, (&input).into()),
                        (input_bool as usize, (&input_bool_data).into()),
                    ],
                    &[output_id],
                    None,
                )
                .unwrap();
            assert_eq!(result.len(), 1);
        }

        // Outputs of ops tested with a 2D input.
        let outputs = vec![
            "Gemm_out",
            "MatMul_out",
            "Split_out_1",
            "Split_out_2",
            "TopK_out_indices",
            "TopK_out_values",
        ];
        let input = Tensor::from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model
                .run(&[(input_2d as usize, (&input).into())], &[output_id], None)
                .unwrap();
            assert_eq!(result.len(), 1);
        }

        // Range op
        let start = Tensor::from_scalar(0.);
        let limit = Tensor::from_scalar(5.);
        let delta = Tensor::from_scalar(1.);
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
        let cond = Tensor::from_scalar(1);
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
