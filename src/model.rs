use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "mmap")]
use std::fs::File;

#[cfg(feature = "mmap")]
use memmap2::Mmap;

use rten_tensor::Tensor;

use crate::constant_storage::{ArcSlice, ArcTensorView, ConstantStorage};
use crate::env::str_as_bool;
use crate::graph::{
    CaptureEnv, ConstantNodeData, Dimension, Graph, Node, NodeId, RunError, RunOptions,
};
use crate::header::{Header, HeaderError};
use crate::model_metadata::ModelMetadata;
use crate::number::LeBytes;
use crate::op_registry::{convert_dtype, OpLoadContext, OpRegistry, ReadOpError};
use crate::ops::{DataType, InputOrOutput, Output};
use crate::optimize::GraphOptimizer;
use crate::schema_generated as sg;
use crate::schema_generated::root_as_model;
use crate::slice_cast::{cast_pod_slice, Pod};
use crate::timing::TimingSort;
use crate::weight_cache::WeightCache;

/// The central type used to execute RTen machine learning models.
///
/// Models are loaded from `.rten` format model files and executed using
/// [`Model::run`]. They take a list of tensor views as inputs, perform a series
/// of computations and return one or more output tensors.
///
/// ## Example
///
/// ```no_run
/// use rten_tensor::prelude::*;
/// use rten_tensor::Tensor;
///
/// use rten::Model;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let model = Model::load_file("model.rten")?;
///     let input_id = model.node_id("input")?;
///     let output_id = model.node_id("output")?;
///
///     // Prepare inputs in format expected by model.
///     let input_data: Tensor<f32> = Tensor::zeros(&[1, 3, 224, 224]);
///
///     let mut outputs = model.run(vec![(input_id, input_data.into())], &[output_id], None)?;
///     let output: Tensor<f32> = outputs.remove(0).try_into()?;
///
///     // Post-process outputs.
///
///     Ok(())
/// }
/// ```
///
/// ## About RTen models
///
/// `.rten` models use the [FlatBuffers](https://github.com/google/flatbuffers)
/// format and are conceptually similar to the `.ort` format used by ONNX
/// Runtime and `.tflite` used by TensorFlow Lite.
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
/// nodes can be obtained using [`Model::input_ids`] and [`Model::output_ids`].
/// These IDs are then used when calling [`Model::run`]. Model execution consists
/// of generating a plan which starts with the input nodes, and executes the
/// necessary operators to generate the requested outputs.
///
/// ## Graph optimizations
///
/// By default RTen applies various optimizations to the model when it is loaded
/// to improve inference performance. These optimizations guarantee to preserve
/// the model's inputs and outputs, but other nodes may be replaced or
/// eliminated. To configure or disable optimizations, use [`ModelOptions`].
///
/// ## Weight prepacking
///
/// In addition to optimizing the structure of the graph, RTen can create copies
/// of the weights with an optimized ("packed") data layout at model load time.
/// Enabling this will increase model load time and memory usage but reduce the
/// time taken per inference. When this option is disabled, weights are packed
/// temporarily on-demand just before they are used for computation.
///
/// For generative transformer models (aka. "transformer decoders") prepacking
/// is generally only useful when processing multiple input tokens at a time.
///
/// Prepacking is disabled by default but can be enabled using [`ModelOptions`].
///
/// ## Partial evaluation
///
/// Some models, such as transformer decoders, are evaluated repeatedly in a
/// loop. If such models have inputs which are constant in each iteration of the
/// loop, execution can be sped up by using partial evaluation. This involves
/// evaluating the part of the graph that depends only on the constant inputs
/// once, outside the loop. To do this use [`Model::partial_run`].
///
/// ## Custom operator registries
///
/// By default all supported ONNX operators are available for use by the model.
/// You can reduce binary size and compilation time by loading a model with
/// only a subset of operators enabled. See [`ModelOptions::with_ops`].
pub struct Model {
    graph: Graph,
    metadata: ModelMetadata,
    weight_cache: WeightCache,
}

/// Provides access to metadata about a graph node.
pub struct NodeInfo<'a> {
    node: &'a Node,
}

impl NodeInfo<'_> {
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

    /// Return the expected data type for this node at runtime.
    ///
    /// For constants the data type is always known. For values the data type
    /// may be specified. For operators this always returns `None`.
    pub fn dtype(&self) -> Option<DataType> {
        self.node.dtype()
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

/// Configuration for loading subgraphs.
struct SubgraphOptions<'a> {
    /// Tensor data storage
    storage: Arc<ConstantStorage>,

    /// Offset of tensor data within the storage.
    tensor_data_offset: Option<u64>,

    /// Whether to apply optimizations when loading the subgraph.
    optimize: bool,

    /// Provides access to info about nodes captured from parent graphs.
    /// This is needed for some optimization passes.
    capture_env: Option<&'a CaptureEnv<'a>>,
}

/// Options which customize how a model is loaded.
///
/// This enables more advanced use cases such as loading a model with only
/// a subset of operators available, or with different sets of optimizations
/// applied.
pub struct ModelOptions {
    registry: OpRegistry,
    optimize: bool,
    prepack_weights: bool,
}

impl ModelOptions {
    /// Create a set of options with all operators enabled.
    pub fn with_all_ops() -> ModelOptions {
        Self::with_ops(OpRegistry::with_all_ops())
    }

    /// Create a set of options with a custom set of operators enabled.
    ///
    /// This can be used to reduce binary size by excluding operators that
    /// the model will not use, or use custom implementations of operators.
    pub fn with_ops(ops: OpRegistry) -> ModelOptions {
        ModelOptions {
            registry: ops,
            optimize: true,
            prepack_weights: false,
        }
    }

    /// Set whether graph optimizations are enabled.
    pub fn enable_optimization(&mut self, enable: bool) -> &mut Self {
        self.optimize = enable;
        self
    }

    /// Set whether weights are prepacked.
    ///
    /// Prepacking creates copies of the weights with an optimized data layout.
    /// Enabling this will increase model load time and memory usage but allow
    /// for faster inference.
    pub fn prepack_weights(&mut self, prepack: bool) -> &mut Self {
        self.prepack_weights = prepack;
        self
    }

    /// Load the model from a file. See [`Model::load_file`].
    pub fn load_file<P: AsRef<Path>>(&self, path: P) -> Result<Model, ModelLoadError> {
        let data = std::fs::read(path).map_err(ModelLoadError::ReadFailed)?;
        self.load(data)
    }

    /// Load the model from a data buffer. See [`Model::load`].
    pub fn load(&self, data: Vec<u8>) -> Result<Model, ModelLoadError> {
        let storage = Arc::new(ConstantStorage::Buffer(data));
        Model::load_impl(storage, self)
    }

    /// Load the model from a static slice of bytes. See [`Model::load_static_slice`].
    pub fn load_static_slice(&self, data: &'static [u8]) -> Result<Model, ModelLoadError> {
        let storage = Arc::new(ConstantStorage::StaticSlice(data));
        Model::load_impl(storage, self)
    }

    /// Load the model from a memory-mapped view of a file. See [`Model::load_mmap`].
    ///
    /// # Safety
    ///
    /// See notes in [`Model::load_mmap`].
    #[cfg(feature = "mmap")]
    pub unsafe fn load_mmap<P: AsRef<Path>>(&self, path: P) -> Result<Model, ModelLoadError> {
        let file = File::open(path).map_err(ModelLoadError::ReadFailed)?;
        let mmap = Mmap::map(&file).map_err(ModelLoadError::ReadFailed)?;
        let storage = Arc::new(ConstantStorage::Mmap(mmap));
        Model::load_impl(storage, self)
    }
}

impl Model {
    /// Load a serialized model from a `.rten` file.
    ///
    /// This method reads the entire file into memory. For large models (hundreds
    /// of MB or more), [`load_mmap`](Model::load_mmap) can be faster.
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Model, ModelLoadError> {
        ModelOptions::with_all_ops().load_file(path)
    }

    /// Load a serialized model from a byte buffer.
    pub fn load(data: Vec<u8>) -> Result<Model, ModelLoadError> {
        ModelOptions::with_all_ops().load(data)
    }

    /// Load a serialized model from a static byte slice.
    ///
    /// This is useful for loading models embedded in the binary via
    /// [`include_bytes`] for example.
    pub fn load_static_slice(data: &'static [u8]) -> Result<Model, ModelLoadError> {
        ModelOptions::with_all_ops().load_static_slice(data)
    }

    /// Load a serialized model by mapping a view of a file as memory.
    ///
    /// This method requires the `mmap` crate feature to be enabled.
    ///
    /// Loading a model via memory-mapping makes the initial load of the model
    /// faster for large models (hundreds of MB) and also allows sharing the
    /// memory with other processes. If a process uses `load_file`, its private
    /// memory usage will be the size of the model plus its working space. If a
    /// process uses `load_mmap`, its private memory usage will only be that
    /// needed for working space.
    ///
    /// The first _run_ of a memory-mapped model will be slower than if the file
    /// is read into memory first and then executed. Depending on the size of
    /// the model, the overall time taken for load + first run may be less or
    /// about the same.  Subsequent model executions should the same time.
    ///
    /// # Safety
    ///
    /// This method is marked unsafe because undefined behavior can be caused
    /// if the model file is modified on disk while it is being used by a
    /// `Model`. Callers will need to decide whether this is an acceptable risk
    /// for their context.
    ///
    /// # Platform support
    ///
    /// This function is not available on WebAssembly. Use [`load`](Self::load)
    /// or [`load_file`](Self::load_file) instead.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use rten::Model;
    ///
    /// let model = unsafe { Model::load_mmap("model.rten")? };
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "mmap")]
    #[cfg(not(target_arch = "wasm32"))]
    pub unsafe fn load_mmap<P: AsRef<Path>>(path: P) -> Result<Model, ModelLoadError> {
        ModelOptions::with_all_ops().load_mmap(path)
    }

    fn load_impl(
        storage: Arc<ConstantStorage>,
        options: &ModelOptions,
    ) -> Result<Model, ModelLoadError> {
        let registry = &options.registry;

        let file_data = storage.data();
        let header = match Header::from_buf(file_data) {
            Ok(header) => Some(header),
            Err(HeaderError::InvalidMagic) => None,
            Err(err) => {
                return Err(ModelLoadError::InvalidHeader(Box::new(err)));
            }
        };

        let model_data = if let Some(header) = header.as_ref() {
            let (offset, len) = (header.model_offset as usize, header.model_len as usize);
            &file_data[offset..offset + len]
        } else {
            file_data
        };

        let model = root_as_model(model_data).map_err(ModelLoadError::ParseFailed)?;

        if model.schema_version() != 1 {
            return Err(ModelLoadError::SchemaVersionUnsupported);
        }

        let tensor_data_offset = header.as_ref().map(|h| h.tensor_data_offset);
        let graph = Self::load_graph(
            model.graph(),
            registry,
            storage.clone(),
            tensor_data_offset,
            options.optimize,
            None, /* capture_env */
        )?;

        let mut weight_cache = WeightCache::new();
        if options.prepack_weights {
            graph.prepack_weights(&mut weight_cache);
        }

        let metadata = model
            .metadata()
            .map(ModelMetadata::deserialize)
            .unwrap_or_default();

        let model = Model {
            graph,
            metadata,
            weight_cache,
        };
        Ok(model)
    }

    fn load_graph(
        serialized_graph: sg::Graph,
        registry: &OpRegistry,
        storage: Arc<ConstantStorage>,
        tensor_data_offset: Option<u64>,
        optimize: bool,
        capture_env: Option<&CaptureEnv>,
    ) -> Result<Graph, ModelLoadError> {
        let node_count = serialized_graph.nodes().map(|ns| ns.len()).unwrap_or(0);

        // Map of model node index to graph node ID
        let mut node_id_from_index: HashMap<usize, NodeId> = HashMap::with_capacity(node_count);

        let input_ids: Vec<NodeId> = serialized_graph
            .inputs()
            .map(|ids| ids.iter().map(NodeId::from_u32).collect())
            .unwrap_or_default();

        let output_ids: Vec<NodeId> = serialized_graph
            .outputs()
            .map(|ids| ids.iter().map(NodeId::from_u32).collect())
            .unwrap_or_default();

        let mut graph = Graph::with_capacity(node_count);
        graph.set_input_ids(&input_ids);
        graph.set_output_ids(&output_ids);

        if let Some(captures) = serialized_graph.captures() {
            let captures: Vec<NodeId> = captures.iter().map(NodeId::from_u32).collect();
            graph.set_captures(&captures);
        }

        if let Some(nodes) = serialized_graph.nodes() {
            for (node_index, node) in nodes.iter().enumerate() {
                let graph_node = if let Some(operator) = node.data_as_operator_node() {
                    Self::add_graph_operator(
                        &mut graph,
                        node.name(),
                        operator,
                        registry,
                        &node_id_from_index,
                        SubgraphOptions {
                            storage: storage.clone(),
                            tensor_data_offset,
                            optimize,
                            capture_env,
                        },
                    )?
                } else if let Some(value) = node.data_as_value_node() {
                    Self::add_graph_value(&mut graph, node.name(), value)?
                } else if let Some(constant) = node.data_as_constant_node() {
                    Self::add_graph_constant(
                        &mut graph,
                        node.name(),
                        constant,
                        &storage,
                        tensor_data_offset,
                    )?
                } else {
                    return Err(ModelLoadError::GraphError(
                        NodeError::for_node(node.name(), "unknown node type").into(),
                    ));
                };
                node_id_from_index.insert(node_index, graph_node);
            }
        }

        if optimize {
            let optimizer = GraphOptimizer::new();
            optimizer
                .optimize(graph, capture_env)
                .map_err(|err| ModelLoadError::OptimizeError(Box::new(err)))
        } else {
            Ok(graph)
        }
    }

    fn add_graph_operator(
        graph: &mut Graph,
        name: Option<&str>,
        operator: sg::OperatorNode,
        registry: &OpRegistry,
        node_id_from_index: &HashMap<usize, NodeId>,
        subgraph_opts: SubgraphOptions,
    ) -> Result<NodeId, ModelLoadError> {
        let load_subgraph = |g: sg::Graph| -> Result<Graph, ModelLoadError> {
            let SubgraphOptions {
                storage,
                tensor_data_offset,
                optimize,
                capture_env,
            } = &subgraph_opts;
            let capture_env = CaptureEnv::new(*capture_env, graph, None, None, None);
            Self::load_graph(
                g,
                registry,
                storage.clone(),
                *tensor_data_offset,
                *optimize,
                Some(&capture_env),
            )
        };

        struct LoadContext<'a> {
            load_graph: &'a dyn Fn(sg::Graph) -> Result<Graph, ModelLoadError>,
        }

        impl OpLoadContext for LoadContext<'_> {
            fn load_graph(&self, graph: sg::Graph) -> Result<Graph, ReadOpError> {
                (self.load_graph)(graph).map_err(|err| ReadOpError::SubgraphError(err.into()))
            }
        }

        let ctx = LoadContext {
            load_graph: &load_subgraph,
        };
        let op = registry.read_op(&operator, &ctx).map_err(|err| {
            ModelLoadError::OperatorInvalid(NodeError::for_node(name, err).into())
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
                    return Err(ModelLoadError::GraphError(
                        NodeError::for_node(name, "operator input is invalid").into(),
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
                        NodeError::for_node(name, "operator output is invalid").into(),
                    ));
                }
            }
        }

        let graph_node = graph.add_op(name, op, &inputs, &outputs);
        Ok(graph_node)
    }

    fn add_graph_value(
        graph: &mut Graph,
        name: Option<&str>,
        value: sg::ValueNode,
    ) -> Result<NodeId, ModelLoadError> {
        let shape: Option<Vec<Dimension>> = value.shape().map(|shape| {
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
        let dtype = value
            .dtype()
            .map(|dtype| convert_dtype("", dtype))
            .transpose()
            .map_err(|err| {
                ModelLoadError::OperatorInvalid(NodeError::for_node(name, err).into())
            })?;
        let graph_node = graph.add_value(name, shape, dtype);
        Ok(graph_node)
    }

    fn add_graph_constant(
        graph: &mut Graph,
        name: Option<&str>,
        constant: sg::ConstantNode,
        storage: &Arc<ConstantStorage>,
        tensor_data_offset: Option<u64>,
    ) -> Result<NodeId, ModelLoadError> {
        let shape: Vec<usize> = constant.shape().iter().map(|x| x as usize).collect();

        if let Some(data_offset) = constant.data_offset() {
            // Constant data is stored outside the model buffer, in the same file.

            let Some(tensor_data_offset) = tensor_data_offset else {
                return Err(ModelLoadError::GraphError(
                    "tensor data section missing".into(),
                ));
            };
            let data_offset = (tensor_data_offset + data_offset) as usize;

            let graph_node = match constant.dtype() {
                Some(sg::ConstantDataType::Int32) => {
                    let const_data =
                        constant_data_from_storage_offset::<i32>(storage, &shape, data_offset)?;
                    graph.add_constant(name, const_data)
                }
                Some(sg::ConstantDataType::Float32) => {
                    let const_data =
                        constant_data_from_storage_offset::<f32>(storage, &shape, data_offset)?;
                    graph.add_constant(name, const_data)
                }
                Some(sg::ConstantDataType::Int8) => {
                    let const_data =
                        constant_data_from_storage_offset::<i8>(storage, &shape, data_offset)?;
                    graph.add_constant(name, const_data)
                }
                Some(sg::ConstantDataType::UInt8) => {
                    let const_data =
                        constant_data_from_storage_offset::<u8>(storage, &shape, data_offset)?;
                    graph.add_constant(name, const_data)
                }
                _ => {
                    return Err(ModelLoadError::GraphError(
                        NodeError::for_node(name, "unsupported data type for external constant")
                            .into(),
                    ));
                }
            };
            Ok(graph_node)
        } else {
            // Constant data is stored inline in model
            let graph_node = if let Some(float_data) = constant.data_as_float_data() {
                let const_data =
                    constant_data_from_flatbuffers_vec(storage, float_data.data(), &shape);
                graph.add_constant(name, const_data)
            } else if let Some(int_data) = constant.data_as_int_32_data() {
                let const_data =
                    constant_data_from_flatbuffers_vec(storage, int_data.data(), &shape);
                graph.add_constant(name, const_data)
            } else if let Some(int8_data) = constant.data_as_int_8_data() {
                let const_data =
                    constant_data_from_flatbuffers_vec(storage, int8_data.data(), &shape);
                graph.add_constant(name, const_data)
            } else if let Some(uint8_data) = constant.data_as_uint_8_data() {
                let const_data =
                    constant_data_from_flatbuffers_vec(storage, uint8_data.data(), &shape);
                graph.add_constant(name, const_data)
            } else {
                return Err(ModelLoadError::GraphError(
                    NodeError::for_node(name, "unsupported data type for inline constant").into(),
                ));
            };
            Ok(graph_node)
        }
    }

    /// Find a node in the model's graph given its string name.
    pub fn find_node(&self, id: &str) -> Option<NodeId> {
        self.graph.get_node_id(id)
    }

    /// Find a node in the model's graph given its string name.
    ///
    /// This is a convenience method which is like [`Model::find_node`] but
    /// returns an error that includes the node's name if the node is not found.
    pub fn node_id(&self, id: &str) -> Result<NodeId, RunError> {
        self.find_node(id)
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
        self.graph.input_ids()
    }

    /// Return the IDs of output nodes.
    pub fn output_ids(&self) -> &[NodeId] {
        self.graph.output_ids()
    }

    /// Return the total number of parameters in the model's weights.
    pub fn total_params(&self) -> usize {
        self.graph.total_params()
    }

    /// Convenience method that returns the expected input shape for the index'th input.
    ///
    /// The shape may contain a mix of fixed and symbolic dimensions.
    pub fn input_shape(&self, index: usize) -> Option<Vec<Dimension>> {
        let input_id = self.graph.input_ids().get(index)?;
        let node_info = self.node_info(*input_id)?;
        node_info.shape()
    }

    /// Execute the model and return the outputs specified by `outputs`.
    ///
    /// This method allows for running a model with a variable number of inputs
    /// and outputs of different types. See [`Model::run_one`] or [`Model::run_n`]
    /// for the common case of running a model with a single or statically
    /// known number of inputs and outputs.
    ///
    /// The input and output nodes are specified via IDs looked up via `find_node`.
    pub fn run(
        &self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Output>, RunError> {
        let mut opts = opts.unwrap_or_default();
        if let Some(timing_var) = env::var_os("RTEN_TIMING") {
            let timing_var = timing_var.to_string_lossy();
            parse_timing_config(&timing_var, &mut opts);
        }
        self.graph
            .run(inputs, outputs, Some(&self.weight_cache), Some(opts))
    }

    /// Run a model and retrieve `N` outputs.
    ///
    /// This is a simplified version of [`Model::run`] for the common case of
    /// executing a model with a statically known number of outputs. Use
    /// [`Model::run`] instead if the number of outputs is known only at runtime.
    pub fn run_n<const N: usize>(
        &self,
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: [NodeId; N],
        opts: Option<RunOptions>,
    ) -> Result<[Output; N], RunError> {
        let result = self.run(inputs, &outputs, opts)?;
        Ok(result.try_into().expect("wrong output count"))
    }

    /// Run a model with a single input and output.
    ///
    /// This is a simplified version of [`Model::run`] for the common case of
    /// executing a model with a single input and output.
    pub fn run_one(
        &self,
        input: InputOrOutput,
        opts: Option<RunOptions>,
    ) -> Result<Output, RunError> {
        let &input_id = self.input_ids().first().ok_or(RunError::InvalidNodeId)?;
        let &output_id = self.output_ids().first().ok_or(RunError::InvalidNodeId)?;
        self.run_n(vec![(input_id, input)], [output_id], opts)
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
        inputs: Vec<(NodeId, InputOrOutput)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<(NodeId, Output)>, RunError> {
        self.graph.partial_run(inputs, outputs, opts)
    }
}

/// Errors reported by [`Model::load`].
#[derive(Debug)]
pub enum ModelLoadError {
    /// The FlatBuffers data describing the model is not supported by this
    /// version of RTen.
    SchemaVersionUnsupported,

    /// An error occurred reading the file from disk.
    ReadFailed(std::io::Error),

    /// An error occurred parsing the FlatBuffers data describing the model
    /// structure.
    ParseFailed(flatbuffers::InvalidFlatbuffer),

    /// An error occurred deserializing an operator.
    OperatorInvalid(Box<dyn Error + Send + Sync>),

    /// An error occurred while traversing the model's graph to instantiate
    /// nodes and connections.
    GraphError(Box<dyn Error + Send + Sync>),

    /// An error occurred while optimizing the graph.
    OptimizeError(Box<dyn Error + Send + Sync>),

    /// The file's header is invalid.
    InvalidHeader(Box<dyn Error + Send + Sync>),
}

impl Display for ModelLoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelLoadError::SchemaVersionUnsupported => write!(f, "unsupported schema version"),
            ModelLoadError::ReadFailed(e) => write!(f, "read error: {e}"),
            ModelLoadError::ParseFailed(e) => write!(f, "parse error: {e}"),
            ModelLoadError::OperatorInvalid(e) => write!(f, "operator error: {e}"),
            ModelLoadError::GraphError(e) => write!(f, "graph error: {e}"),
            ModelLoadError::OptimizeError(e) => write!(f, "graph optimization error: {e}"),
            ModelLoadError::InvalidHeader(e) => write!(f, "invalid header: {e}"),
        }
    }
}

impl Error for ModelLoadError {}

/// A model error which pertains to a specific node.
#[derive(Debug)]
struct NodeError<E: Display> {
    name: Option<String>,
    inner: E,
}

impl<E: Display> NodeError<E> {
    fn for_node(name: Option<&str>, inner: E) -> Self {
        NodeError {
            name: name.map(|s| s.to_string()),
            inner,
        }
    }
}

impl<E: Display> Display for NodeError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "in node \"{}\": {}",
            self.name.as_deref().unwrap_or_default(),
            self.inner
        )
    }
}

impl<E: std::fmt::Debug + Display> Error for NodeError<E> {}

/// Transmute a `[u8]` to `[T]` provided it is correctly aligned and we're on
/// a little-endian system.
fn cast_le_bytes<T: Pod>(bytes: &[u8]) -> Option<&[T]> {
    if std::mem::size_of::<T>() != 1 && !cfg!(target_endian = "little") {
        return None;
    }
    cast_pod_slice(bytes)
}

/// Convert a range of bytes in storage into data for a graph constant.
///
/// If the data is correctly aligned and the system is little-endian, this will
/// return a view, otherwise it will copy the data into an owned tensor.
fn constant_data_from_storage_offset<T: LeBytes + Pod>(
    storage: &Arc<ConstantStorage>,
    shape: &[usize],
    offset: usize,
) -> Result<ConstantNodeData<T>, ModelLoadError> {
    let n_elements: usize = shape.iter().product();
    let byte_len = n_elements * std::mem::size_of::<T>();

    let Some(bytes) = storage.data().get(offset..offset + byte_len) else {
        return Err(ModelLoadError::GraphError(
            "invalid tensor data offset".into(),
        ));
    };

    if let Some(elements) = cast_le_bytes(bytes) {
        let storage =
            ArcSlice::new(storage.clone(), elements).expect("storage does not contain data");
        let const_data: ConstantNodeData<T> = ArcTensorView::from_data(shape, storage).into();
        Ok(const_data)
    } else {
        let data: Vec<T> = bytes
            .chunks(std::mem::size_of::<T>())
            .map(|chunk| T::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(Tensor::from_data(shape, data).into())
    }
}

/// Convert a vector from a FlatBuffers file into data for a graph constant node.
///
/// If the data is correctly aligned and the system is little-endian, this will
/// return a view, otherwise it will copy the data into an owned tensor.
fn constant_data_from_flatbuffers_vec<'a, T: Pod + flatbuffers::Follow<'a, Inner = T>>(
    storage: &Arc<ConstantStorage>,
    fb_vec: flatbuffers::Vector<'a, T>,
    shape: &[usize],
) -> ConstantNodeData<T> {
    if let Some(elements) = cast_le_bytes(fb_vec.bytes()) {
        let storage =
            ArcSlice::new(storage.clone(), elements).expect("storage does not contain data");
        ArcTensorView::from_data(shape, storage).into()
    } else {
        let storage: Vec<T> = fb_vec.iter().collect();
        Tensor::from_data(shape, storage).into()
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;

    use crate::graph::{Dimension, NodeId, RunError};
    use crate::model::{Model, ModelOptions};
    use crate::model_builder::{
        GraphBuilder, IfArgs, MetadataArgs, ModelBuilder, ModelFormat, OpType,
    };
    use crate::ops;
    use crate::ops::{
        BoxOrder, CoordTransformMode, DataType, DepthToSpaceMode, NearestMode, OpError, Output,
        ResizeMode, Scalar, Shape,
    };
    use crate::OpRegistry;

    fn generate_model_buffer(format: ModelFormat) -> Vec<u8> {
        let mut builder = ModelBuilder::new(format);
        let mut graph_builder = builder.graph_builder();

        let const_val = Tensor::from_data(&[1, 2, 2], vec![0.5, -0.5, 0.1, -0.1]);
        let const_node = graph_builder.add_constant(const_val.view());

        let input_shape: Vec<Dimension> = const_val
            .shape()
            .iter()
            .copied()
            .map(Dimension::Fixed)
            .collect();
        let input_node =
            graph_builder.add_value("input", Some(&input_shape), Some(DataType::Float));
        let output_node = graph_builder.add_value("output", None, Some(DataType::Float));

        graph_builder.add_input(input_node);
        graph_builder.add_output(output_node);

        let concat_out = graph_builder.add_value("concat_out", None, None);
        graph_builder.add_operator(
            "concat",
            OpType::Concat(ops::Concat { axis: 0 }),
            &[const_node, input_node].map(Some),
            &[concat_out],
        );
        graph_builder.add_operator("relu", OpType::Relu, &[Some(concat_out)], &[output_node]);

        let graph = graph_builder.finish();
        builder.set_graph(graph);
        builder.add_metadata(MetadataArgs {
            onnx_hash: Some("abc".to_string()),
        });

        builder.finish()
    }

    /// Generate input for the model created by `generate_model_buffer`.
    fn generate_input() -> Tensor<f32> {
        Tensor::from_data(&[1, 2, 2], vec![1., 2., -1., -2.])
    }

    /// Check the output of a model created by `generate_model_buffer`, using
    /// input created by `generate_input`.
    fn check_output(mut result: Vec<Output>) -> Tensor<f32> {
        assert_eq!(result.len(), 1);

        let tensor: Tensor<f32> = result.remove(0).into_tensor::<f32>().unwrap();
        assert_eq!(tensor.shape(), &[2, 2, 2]);
        assert_eq!(tensor.to_vec(), &[0.5, 0., 0.1, 0., 1., 2., 0., 0.]);

        tensor
    }

    #[test]
    fn test_model_input_output_ids() {
        let buffer = generate_model_buffer(ModelFormat::V2);

        let model = Model::load(buffer).unwrap();

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
        let buffer = generate_model_buffer(ModelFormat::V2);
        let registry = OpRegistry::new();
        let result = ModelOptions::with_ops(registry).load(buffer);
        assert_eq!(
            result.err().map(|err| err.to_string()).as_deref(),
            Some("operator error: in node \"concat\": operator Concat is not supported or not enabled")
        );
    }

    #[test]
    fn test_shape_info() {
        let buffer = generate_model_buffer(ModelFormat::V2);
        let model = Model::load(buffer).unwrap();
        let input_id = model.input_ids()[0];

        let shape = model
            .node_info(input_id)
            .and_then(|ni| ni.shape())
            .expect("input shape missing");
        assert_eq!(shape, &[1, 2, 2].map(Dimension::Fixed));
    }

    #[test]
    fn test_value_dtype_info() {
        let buffer = generate_model_buffer(ModelFormat::V2);
        let model = Model::load(buffer).unwrap();
        let input_id = model.input_ids()[0];

        let dtype = model
            .node_info(input_id)
            .and_then(|ni| ni.dtype())
            .expect("input dtype missing");
        assert_eq!(dtype, DataType::Float);
    }

    #[test]
    fn test_metadata() {
        let buffer = generate_model_buffer(ModelFormat::V2);
        let model = Model::load(buffer).unwrap();
        assert_eq!(model.metadata().onnx_hash(), Some("abc"));
        assert_eq!(model.metadata().description(), None);
    }

    #[test]
    fn test_input_shape() {
        let buffer = generate_model_buffer(ModelFormat::V2);
        let model = Model::load(buffer).unwrap();
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
        struct Case {
            format: ModelFormat,
            opts: Option<ModelOptions>,
        }

        let cases = [
            Case {
                format: ModelFormat::V1,
                opts: None,
            },
            Case {
                format: ModelFormat::V2,
                opts: None,
            },
            // Graph optimizations disabled
            Case {
                format: ModelFormat::V2,
                opts: Some({
                    let mut opts = ModelOptions::with_all_ops();
                    opts.enable_optimization(false);
                    opts
                }),
            },
            // Prepacking enabled
            Case {
                format: ModelFormat::V2,
                opts: Some({
                    let mut opts = ModelOptions::with_all_ops();
                    opts.prepack_weights(true);
                    opts
                }),
            },
        ];

        for Case { format, opts } in cases {
            let buffer = generate_model_buffer(format);

            let model = if let Some(opts) = opts {
                opts.load(buffer).unwrap()
            } else {
                Model::load(buffer).unwrap()
            };
            let input_id = model.input_ids()[0];
            let output_id = model.output_ids()[0];

            let input = generate_input();

            // Test a normal model run.
            let result = model
                .run(vec![(input_id, input.view().into())], &[output_id], None)
                .unwrap();
            let result_tensor = check_output(result);

            // Test a partial run. Since we are providing all inputs, this works the
            // same as `Model::run`. See `Graph::partial_run` tests for other cases.
            let partial_run_result = model
                .partial_run(vec![(input_id, input.into())], &[output_id], None)
                .unwrap();
            assert_eq!(
                partial_run_result,
                vec![(output_id, Output::FloatTensor(result_tensor))]
            );
        }
    }

    #[test]
    fn test_load_invalid_model() {
        struct Case {
            buf: Vec<u8>,
            expected_error: &'static str,
        }

        let buf = generate_model_buffer(ModelFormat::V2);

        let mut invalid_model = buf.clone();
        let header_size = 32;
        invalid_model.insert(header_size, 0); // Corrupt buffer after header

        let mut truncated_buf = buf.clone();
        truncated_buf.truncate(truncated_buf.len() - 1);

        let cases = [
            Case {
                buf: b"RTENabc".to_vec(),
                expected_error: "invalid header",
            },
            Case {
                buf: invalid_model,
                expected_error: "parse error:",
            },
            Case {
                buf: truncated_buf,
                expected_error: "graph error: invalid tensor data offset",
            },
        ];

        for Case {
            buf,
            expected_error,
        } in cases
        {
            let err = Model::load(buf).err().unwrap();
            assert!(err.to_string().contains(expected_error));
        }
    }

    #[test]
    fn test_load_static_slice() {
        let buffer = generate_model_buffer(ModelFormat::V2).leak();
        let model = Model::load_static_slice(buffer).unwrap();
        let input = generate_input();
        let input_id = model.input_ids()[0];
        let output_id = model.output_ids()[0];
        let result = model
            .run(vec![(input_id, input.into())], &[output_id], None)
            .unwrap();
        check_output(result);
    }

    #[test]
    fn test_load_file() {
        let buffer = generate_model_buffer(ModelFormat::V2);
        std::fs::write("model-load-file-test.rten", buffer).unwrap();

        let model = Model::load_file("model-load-file-test.rten").unwrap();
        let input_id = model.input_ids()[0];
        let output_id = model.output_ids()[0];

        let input = generate_input();
        let result = model
            .run(vec![(input_id, input.into())], &[output_id], None)
            .unwrap();
        check_output(result);
    }

    #[cfg(feature = "mmap")]
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_mmap() {
        let buffer = generate_model_buffer(ModelFormat::V2);
        std::fs::write("model-load-mmap-test.rten", buffer).unwrap();

        let model = unsafe { Model::load_mmap("model-load-mmap-test.rten").unwrap() };
        let input_id = model.input_ids()[0];
        let output_id = model.output_ids()[0];

        let input = generate_input();
        let result = model
            .run(vec![(input_id, input.into())], &[output_id], None)
            .unwrap();
        check_output(result);
    }

    #[test]
    fn test_run_one() {
        let buffer = generate_model_buffer(ModelFormat::V2);
        let model = Model::load(buffer).unwrap();

        let input = Tensor::from([[[1., 2.], [-1., -2.]]]);
        let result: Tensor<f32> = model
            .run_one(input.into(), None)
            .unwrap()
            .try_into()
            .unwrap();

        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.to_vec(), &[0.5, 0., 0.1, 0., 1., 2., 0., 0.]);
    }

    #[test]
    fn test_omitted_optional_inputs() {
        let mut builder = ModelBuilder::new(ModelFormat::V2);
        let mut graph_builder = builder.graph_builder();

        let output_node = graph_builder.add_value("output", None, None);
        graph_builder.add_output(output_node);
        graph_builder.add_operator(
            "shape",
            OpType::Shape(Shape::default()),
            &[None],
            &[output_node],
        );

        let graph = graph_builder.finish();
        builder.set_graph(graph);
        let buffer = builder.finish();

        // Load with optimizations disabled to prevent the optimizer from
        // running the graph as part of constant propagation.
        let model = ModelOptions::with_all_ops()
            .enable_optimization(false)
            .load(buffer)
            .unwrap();

        let result = model.run(vec![], &[output_node], None);

        assert_eq!(
            result.err(),
            Some(RunError::OperatorError {
                name: "shape".to_string(),
                error: OpError::MissingInputs,
                inputs: Some(Vec::new()),
            })
        );
    }

    // This test exercises basic execution of all operators. It doesn't check
    // the results of operators, it just makes sure they can be deserialized and
    // executed successfully.
    #[test]
    fn test_all_op_types() {
        let mut builder = ModelBuilder::new(ModelFormat::V2);
        let mut graph_builder = builder.graph_builder();

        let input_node = graph_builder.add_value("input", None, None);
        let input_2d = graph_builder.add_value("input.2d", None, None);
        let input_bool = graph_builder.add_value("input.bool", None, None);
        let input_u8 = graph_builder.add_value("input.u8", None, None);
        let input_2d_u8 = graph_builder.add_value("input.2d.u8", None, None);
        let input_2d_i8 = graph_builder.add_value("input.2d.i8", None, None);

        // 4D shape used as the primary input to test most operators (eg. NCHW image). A few
        // require a different shape.
        let input_shape = [1, 1, 3, 3];

        let kernel_val = Tensor::from_data(&[1, 1, 1, 1], vec![0.5]);
        let kernel = graph_builder.add_constant(kernel_val.view());

        let kernel_val_i8 = Tensor::from_data(&[1, 1, 1, 1], vec![0i8]);
        let kernel_i8 = graph_builder.add_constant(kernel_val_i8.view());

        // Names of all operator output nodes.
        let mut op_outputs = Vec::new();

        let mut add_operator =
            |builder: &mut GraphBuilder, name: &str, op: OpType, input_nodes: &[Option<NodeId>]| {
                let output_name = format!("{}_out", name);
                let op_output_node = builder.add_value(&output_name, None, None);
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
                    &mut graph_builder,
                    stringify!($op_name),
                    OpType::$op_name,
                    &$op_inputs.map(Some),
                )
            };

            ($op_name:ident, $op_inputs:expr, $attrs: tt) => {
                add_operator(
                    &mut graph_builder,
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
            kernel_size: [2, 2].into(),
            strides: [2, 2].into(),
            padding: [0, 0, 0, 0].into(),
            count_include_pad: false,
        });

        // Dummy value for BatchNormalization inputs which are vectors with
        // per-channel values.
        let batch_norm_param_val = Tensor::from([1.0]);
        let batch_norm_param = graph_builder.add_constant(batch_norm_param_val.view());
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
        add_operator!(CastLike, [input_node, input_node], {});
        add_operator!(Ceil, [input_node]);

        let clip_min = graph_builder.add_constant(Tensor::from(1.).view());
        let clip_max = graph_builder.add_constant(Tensor::from(6.).view());
        add_operator!(Clip, [input_node, clip_min, clip_max]);
        add_operator!(Concat, [input_node, input_node], { axis: 0 });

        let shape = graph_builder.add_constant(Tensor::from([1, 5, 10]).view());
        add_operator!(ConstantOfShape, [shape], { value: Scalar::Int(42) });

        add_operator!(Conv, [input_node, kernel], {
            dilations: vec![1, 1],
            groups: 1,
            padding: [1, 1, 1, 1].into(),
            strides: vec![1, 1],
        });
        add_operator!(ConvInteger, [input_u8, kernel_i8], {
            dilations: vec![1, 1],
            groups: 1,
            padding: [1, 1, 1, 1].into(),
            strides: vec![1, 1],
        });
        add_operator!(ConvTranspose, [input_node, kernel], {
            strides: vec![2, 2],
            padding: [0, 0, 0, 0].into(),
        });
        add_operator!(Cos, [input_node]);

        let const_u8_val = Tensor::from([0u8, 1, 2, 3, 4]);
        let const_u8 = graph_builder.add_constant(const_u8_val.view());

        let const_f32_val = const_u8_val.map(|x| *x as f32);
        let const_f32 = graph_builder.add_constant(const_f32_val.view());

        let scale_val = Tensor::from(1.);
        let scale = graph_builder.add_constant(scale_val.view());
        let zero_point_val = Tensor::from(0u8);
        let zero_point = graph_builder.add_constant(zero_point_val.view());
        add_operator!(DequantizeLinear, [const_u8, scale, zero_point], {
            axis: 0,
        });
        add_operator!(DepthToSpace, [input_node], {
            mode: DepthToSpaceMode::DepthColumnRow,
            block_size: 1,
        });
        add_operator!(QuantizeLinear, [const_f32, scale, zero_point], {
            axis: 0,
            output_dtype: None,
        });

        add_operator!(Div, [input_node, input_node]);
        #[cfg(feature = "random")]
        {
            let dropout_out = graph_builder.add_value("Dropout_out", None, None);
            let dropout_out_mask = graph_builder.add_value("Dropout_out_mask", None, None);
            graph_builder.add_operator(
                "Dropout",
                OpType::Dropout(ops::Dropout { seed: None }),
                &[input_2d].map(Some),
                &[dropout_out, dropout_out_mask],
            );
        }
        add_operator!(Elu, [input_node], { alpha: 1.0 });
        add_operator!(Equal, [input_node, input_node]);
        add_operator!(Erf, [input_node]);
        add_operator!(Exp, [input_node]);

        let expand_shape_val = Tensor::from([2, 2, 3, 3]);
        let expand_shape = graph_builder.add_constant(expand_shape_val.view());
        add_operator!(Expand, [input_node, expand_shape]);

        add_operator!(Flatten, [input_node], { axis: 1 });
        add_operator!(Floor, [input_node]);

        let gather_indices_val = Tensor::from([0]);
        let gather_indices = graph_builder.add_constant(gather_indices_val.view());
        add_operator!(Gather, [input_node, gather_indices], { axis: 0 });

        let gather_elements_indices_val = Tensor::<i32>::zeros(&input_shape);
        let gather_elements_indices =
            graph_builder.add_constant(gather_elements_indices_val.view());
        add_operator!(GatherElements, [input_node, gather_elements_indices], { axis: 0 });
        add_operator!(Gelu, [input_node], { approximate: false });
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

        // If operator
        let if_cond_val = Tensor::from(1);
        let if_cond = graph_builder.add_constant(if_cond_val.view());

        let mut then_branch_builder = graph_builder.subgraph_builder();
        let then_out_val = Tensor::from(2);
        let then_out = then_branch_builder.add_constant(then_out_val.view());
        then_branch_builder.add_output(then_out);
        let then_branch = then_branch_builder.finish();

        let mut else_branch_builder = graph_builder.subgraph_builder();
        let else_out_val = Tensor::from(3);
        let else_out = else_branch_builder.add_constant(else_out_val.view());
        else_branch_builder.add_output(else_out);
        let else_branch = else_branch_builder.finish();

        add_operator(
            &mut graph_builder,
            "If",
            OpType::If(IfArgs {
                then_branch,
                else_branch,
            }),
            &[Some(if_cond)],
        );

        let instance_norm_scale_val = Tensor::from([1.0]);
        let instance_norm_scale = graph_builder.add_constant(instance_norm_scale_val.view());
        let instance_norm_bias_val = Tensor::from([1.0]);
        let instance_norm_bias = graph_builder.add_constant(instance_norm_bias_val.view());
        add_operator!(InstanceNormalization, [
            input_node, instance_norm_scale, instance_norm_bias
        ], { epsilon: Some(1e-5) });

        let layer_norm_scale_val = Tensor::full(&[input_shape[input_shape.len() - 1]], 1.);
        let layer_norm_scale = graph_builder.add_constant(layer_norm_scale_val.view());
        let layer_norm_bias_val = layer_norm_scale_val.clone();
        let layer_norm_bias = graph_builder.add_constant(layer_norm_bias_val.view());
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
        add_operator!(MatMulInteger, [input_2d_u8, input_2d_i8]);

        add_operator!(Max, [input_node, input_node]);
        add_operator!(MaxPool, [input_node], {
            kernel_size: [2, 2].into(),
            strides: [2, 2].into(),
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
        let nms_boxes =
            graph_builder.add_constant(Tensor::<f32>::zeros(&[1, nms_n_boxes, 4]).view());
        let nms_scores = graph_builder
            .add_constant(Tensor::<f32>::zeros(&[1, nms_n_classes, nms_n_boxes]).view());
        let nms_max_outputs_per_class = graph_builder.add_constant(Tensor::from(10).view());
        let nms_iou_threshold = graph_builder.add_constant(Tensor::from(0.45).view());
        let nms_score_threshold = graph_builder.add_constant(Tensor::from(0.2).view());

        add_operator!(NonMaxSuppression, [nms_boxes, nms_scores, nms_max_outputs_per_class, nms_iou_threshold, nms_score_threshold], {
            box_order: BoxOrder::CenterWidthHeight,
        });

        add_operator!(NonZero, [input_node]);
        add_operator!(Not, [input_bool]);

        let onehot_indices = graph_builder.add_constant(Tensor::from([0, 1, 2]).view());
        let onehot_depth = graph_builder.add_constant(Tensor::from(5).view());
        let onehot_values = graph_builder.add_constant(Tensor::from([1., 0.]).view());
        add_operator!(OneHot, [onehot_indices, onehot_depth, onehot_values], {
            axis: -1,
        });

        add_operator!(Or, [input_bool, input_bool]);

        let pads = graph_builder.add_constant(Tensor::from([0, 0, 1, 1, 0, 0, 1, 1]).view());
        add_operator!(Pad, [input_node, pads]);
        add_operator!(Pow, [input_node, input_node]);

        #[cfg(feature = "random")]
        {
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
        }

        let range_start_node = graph_builder.add_value("range_start", None, None);
        let range_limit_node = graph_builder.add_value("range_limit", None, None);
        let range_delta_node = graph_builder.add_value("range_delta", None, None);
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

        let new_shape = graph_builder.add_constant(Tensor::from([9]).view());
        add_operator!(Reshape, [input_node, new_shape], {
            allow_zero: false,
        });

        let resize_roi_val = Tensor::from([0., 0., 0., 0., 1., 1., 1., 1.]);
        let resize_scales_val = Tensor::from([1., 1., 2., 2.]);
        let resize_roi = graph_builder.add_constant(resize_roi_val.view());
        let resize_scales = graph_builder.add_constant(resize_scales_val.view());
        add_operator!(Resize, [input_node, resize_roi, resize_scales], {
            mode: ResizeMode::Nearest,
            nearest_mode: NearestMode::default(),
            coord_mode: CoordTransformMode::default()
        });

        add_operator!(Round, [input_node]);

        add_operator!(Shape, [input_node], {
            start: Some(1),
            end: Some(-1),
        });
        add_operator!(Sigmoid, [input_node]);
        add_operator!(Sign, [input_node]);
        add_operator!(Sin, [input_node]);
        add_operator!(Size, [input_node]);

        let scatter_elem_indices_val = Tensor::<i32>::zeros(&input_shape);
        let scatter_elem_indices = graph_builder.add_constant(scatter_elem_indices_val.view());
        let scatter_elem_updates_val = Tensor::<f32>::zeros(&input_shape);
        let scatter_elem_updates = graph_builder.add_constant(scatter_elem_updates_val.view());
        add_operator!(
            ScatterElements,
            [input_node, scatter_elem_indices, scatter_elem_updates],
            { axis: 0, reduction: None }
        );

        let const_0 = graph_builder.add_constant(Tensor::from([0]).view());
        let const_1 = graph_builder.add_constant(Tensor::from([1]).view());
        add_operator!(Slice, [input_node, const_0, const_1, const_0]);

        add_operator!(Softplus, [input_node]);
        add_operator!(Softmax, [input_node], { axis: 1 });
        add_operator!(Sqrt, [input_node]);
        add_operator!(Squeeze, [input_node]);

        let split_splits = graph_builder.add_constant(Tensor::from([1, 2]).view());
        let split_out_1 = graph_builder.add_value("Split_out_1", None, None);
        let split_out_2 = graph_builder.add_value("Split_out_2", None, None);
        graph_builder.add_operator(
            "Split",
            OpType::Split(ops::Split {
                axis: 1,
                num_outputs: None,
            }),
            &[input_2d, split_splits].map(Some),
            &[split_out_1, split_out_2],
        );

        add_operator!(Sub, [input_node, input_node]);
        add_operator!(Sum, [input_node, input_node]);
        add_operator!(Tan, [input_node]);
        add_operator!(Tanh, [input_node]);

        let tile_repeats = graph_builder.add_constant(Tensor::from([1, 2, 3, 4]).view());
        add_operator!(Tile, [input_node, tile_repeats]);

        let topk_k = graph_builder.add_constant(Tensor::from(3).view());
        let topk_out_values = graph_builder.add_value("TopK_out_values", None, None);
        let topk_out_indices = graph_builder.add_value("TopK_out_indices", None, None);
        graph_builder.add_operator(
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

        let unsqueeze_axes = graph_builder.add_constant(Tensor::from([0, 4]).view());
        add_operator!(Unsqueeze, [input_node, unsqueeze_axes]);

        let where_cond = graph_builder.add_value("where_cond", None, None);
        let where_x = graph_builder.add_value("where_x", None, None);
        let where_y = graph_builder.add_value("where_y", None, None);
        let where_out = add_operator!(Where, [where_cond, where_x, where_y]);

        add_operator!(Xor, [input_bool, input_bool]);

        let graph = graph_builder.finish();
        builder.set_graph(graph);
        let buffer = builder.finish();

        let model = Model::load(buffer).unwrap();

        // Most ops are tested with one of several standard inputs:
        //
        //  - 4D float tensor (like an NCHW image)
        //  - Int8 NCHW tensor
        //  - Bool-ish int tensor
        //
        // A few require different shapes are tested separately.
        let input = Tensor::from_data(&input_shape, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input_bool_data: Tensor<i32> = Tensor::from([0, 1, 1]);
        let input_u8_data = input.map(|&x| x as u8);
        let input_2d_u8_data = Tensor::from([[1u8, 2], [3, 4]]);
        let input_2d_i8_data = Tensor::from([[1i8, 2], [3, 4]]);

        for output in op_outputs {
            if [
                "Dropout_out",
                "Dropout_out_mask",
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

            // Run with inputs as views.
            //
            // This will run the non-in-place implementation of the operator
            // (`Operator::run`).
            let output_id = model.find_node(&output).unwrap();
            let result = model
                .run(
                    vec![
                        (input_node, input.view().into()),
                        (input_bool, input_bool_data.view().into()),
                        (input_u8, input_u8_data.view().into()),
                        (input_2d_u8, input_2d_u8_data.view().into()),
                        (input_2d_i8, input_2d_i8_data.view().into()),
                    ],
                    &[output_id],
                    None,
                )
                .unwrap();
            assert_eq!(result.len(), 1);

            // Run with inputs as owned tensors.
            //
            // This will run the in-place implementation of the operator if
            // supported (`Operator::run_in_place`).
            let output_id = model.find_node(&output).unwrap();
            let result = model
                .run(
                    vec![
                        (input_node, input.clone().into()),
                        (input_bool, input_bool_data.clone().into()),
                        (input_u8, input_u8_data.clone().into()),
                        (input_2d_u8, input_2d_u8_data.view().into()),
                        (input_2d_i8, input_2d_i8_data.view().into()),
                    ],
                    &[output_id],
                    None,
                )
                .unwrap();
            assert_eq!(result.len(), 1);
        }

        // Outputs of ops which either have multiple outputs, or which are tested
        // with a 2D input.
        #[allow(unused_mut)]
        let mut outputs = vec![
            "Gemm_out",
            "MatMul_out",
            "Split_out_1",
            "Split_out_2",
            "TopK_out_indices",
            "TopK_out_values",
        ];

        #[cfg(feature = "random")]
        {
            outputs.extend(["Dropout_out", "Dropout_out_mask"]);
        }

        let input = Tensor::from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        for output in outputs {
            let output_id = model.find_node(output).unwrap();
            let result = model
                .run(vec![(input_2d, input.view().into())], &[output_id], None)
                .unwrap();
            assert_eq!(result.len(), 1);
        }

        // Range op
        let start = Tensor::from(0.);
        let limit = Tensor::from(5.);
        let delta = Tensor::from(1.);
        let result = model
            .run(
                vec![
                    (range_start_node, start.into()),
                    (range_limit_node, limit.into()),
                    (range_delta_node, delta.into()),
                ],
                &[range_out],
                None,
            )
            .unwrap();
        assert_eq!(result.len(), 1);

        // Where op
        let cond = Tensor::from(1);
        let x = Tensor::from([1, 2, 3]);
        let y = Tensor::from([4, 5, 6]);
        let result = model
            .run(
                vec![
                    (where_cond, cond.into()),
                    (where_x, x.into()),
                    (where_y, y.into()),
                ],
                &[where_out],
                None,
            )
            .unwrap();
        assert_eq!(result.len(), 1);
    }
}
