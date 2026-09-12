use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "mmap")]
use std::fs::File;

#[cfg(feature = "mmap")]
use memmap2::Mmap;

use crate::constant_storage::ConstantStorage;
use crate::env::str_as_bool;
use crate::graph::{Dimension, Graph, Node, NodeId, RunError, RunErrorImpl, RunOptions};
use crate::infer_shapes::InferShapeOptions;
use crate::op_registry::OpRegistry;
use crate::optimize::OptimizeOptions;
use crate::timing::{TimingFilter, TimingSort};
use crate::value::{Value, ValueOrView, ValueType};
use crate::weight_cache::WeightCache;

#[cfg(feature = "onnx_format")]
mod external_data;

mod file_type;
mod load_error;
mod metadata;

#[cfg(feature = "onnx_format")]
pub(crate) mod onnx_loader;

#[cfg(feature = "rten_format")]
mod rten_loader;

pub use load_error::{LoadError, LoadErrorKind};
pub use metadata::ModelMetadata;

use file_type::FileType;
use load_error::LoadErrorImpl;

#[cfg(test)]
pub mod rten_builder;

#[cfg(all(test, feature = "onnx_format"))]
pub mod onnx_builder;

/// The central type used to execute machine learning models.
///
/// Models are loaded from either `.onnx` or `.rten` format model files and
/// executed using [`Model::run`]. They take a list of tensor views as inputs,
/// perform a series of computations and return one or more output tensors.
///
/// ## Example
///
/// ```no_run
/// use rten::{Model, ValueView};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Load the model. If the model is large, using `load_mmap` can be faster.
///     let model = Model::load_file("model.onnx")?;
///
///     // Prepare inputs in format expected by model.
///     let input = ValueView::from_shape([4, 4], &[0.1, 0.2, 0.3, 0.4])?;
///
///     // Run the model.
///     //
///     // The inputs are a Vec of `(node_id, value)` tuples. The outputs are an
///     // array of node IDs.
///     let inputs = vec![
///         (model.node_id("input")?, input.into()),
///     ];
///     let outputs = [model.node_id("output")?];
///     let [output] = model.run_n(inputs, outputs, None)?;
///
///     // Extract outputs.
///     let (shape, data) = output.into_shape_vec::<f32, 2>()?;
///     let [height, width] = shape;
///
///     // Post-process outputs.
///
///     Ok(())
/// }
/// ```
///
/// ## About models
///
/// Machine learning models in RTen are logically graphs consisting of three
/// types of nodes:
///
///  - _Values_ which are supplied or generated at runtime
///  - _Constants_ which are the weights, biases and other parameters of the
///    model. Their values are determined when the model is trained.
///  - _Operators_ which combine the values and constants using operations such
///    as matrix multiplication, convolution etc.
///
/// Some of the value nodes are designated as inputs and outputs. The IDs of
/// these nodes can be obtained using [`Model::input_ids`] and
/// [`Model::output_ids`]. When a model is run, a plan is generated and executed
/// which starts with the provided inputs and runs the necessary operators to
/// generate the requested outputs.
///
/// ## Loading models
///
/// Models can be loaded from files using [`load_file`](Self::load_file) or
/// [`load_mmap`](Self::load_mmap), byte arrays using [`load`](Self::load) or
/// from static data embedded in the binary using
/// [`load_static_slice`](Self::load_static_slice). Additional configuration
/// options can be set by using [`ModelOptions`].
///
/// ## Inputs and outputs
///
/// ### Supported data types
///
/// Model inputs and outputs are tensors with `i32`, `f32`, `i8` or `u8`
/// elements. If an ONNX model expects an `i64` input (eg. for token IDs) or a
/// `bool` input (eg. for a mask), the input should be passed as `i32` instead.
/// If an ONNX model has an `i64` or `bool` output, these will be returned as
/// `i32`.
///
/// ### Querying input and output metadata
///
/// Node IDs for inputs and outputs can be looked up using
/// [`node_id`](Self::node_id). The shape and data type of an input or output
/// can be queried using [`node_info`](Self::node_info).
///
/// ### Creating inputs
///
/// Model inputs can be created from slices, `Vec`s or tensor types from
/// rten-tensor:
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use rten_tensor::NdTensor;
/// use rten::{Model, Value, ValueView};
///
/// let input_a = ValueView::from_shape([2, 2], &[1.0, 2.0, 3.0, 4.0])?;
/// let input_b = Value::from_shape([2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
/// let input_c = NdTensor::from_data([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
///
/// let model = Model::load_file("model.onnx")?;
/// let inputs = vec![
///   (model.node_id("input_a")?, input_a.into()),
///   (model.node_id("input_b")?, input_b.into()),
///   (model.node_id("input_c")?, input_c.into()),
/// ];
/// let outputs = [model.node_id("output")?];
/// let [output] = model.run_n(inputs, outputs, None)?;
/// # Ok(()) }
/// ```
///
/// ### Extracting outputs
///
/// The outputs returned by a model can be extracted into a `(shape, data)`
/// tuple using [`into_shape_vec`](Value::into_shape_vec) or a tensor using
/// `try_into`:
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use rten::Model;
/// use rten_tensor::NdTensor;
///
/// let model = Model::load_file("model.onnx?")?;
///
/// let inputs = vec![];
/// let outputs = [model.node_id("output_a")?, model.node_id("output_b")?];
/// let [output_a, output_b] = model.run_n(inputs, outputs, None)?;
///
/// let (shape, data) = output_a.into_shape_vec::<f32, 2>()?;
/// let tensor: NdTensor<f32, 2> = output_b.try_into()?;
/// # Ok(()) }
/// ```
///
/// ## Running models
///
/// Models are evaluated by calling one of the `run` methods. The most common is
/// [`run_n`](Self::run_n) which runs models with a fixed number of outputs. To
/// run a model with a variable number of outputs, use [`run`](Self::run). The
/// `run*` methods accept a [`RunOptions`] argument which configures the
/// thread pool to use and settings for execution and logging.
///
/// ### Generative models
///
/// Auto-regressive models (eg. LLMs) need to be run in a loop until some
/// termination condition is met. The companion
/// [rten-generate](https://docs.rs/rten-generate/) crate provides APIs to
/// simplify this process.
///
/// ## Performance
///
/// This section describes configuration that affects performance. Some options
/// are set when the model is loaded via [`ModelOptions`], while others are set
/// when the model is run via [`RunOptions`].
///
/// ### Parallelism
///
/// By default RTen will use multiple threads for inference, running models on
/// a global thread pool. The number of threads will be chosen to match the
/// number of physical cores. On platforms which have a mixture of performance
/// and efficiency cores, RTen may set the thread count to match the number of
/// performances cores.
///
/// You can configure the number of threads by creating a custom thread pool
/// and configuring inference to use it via [`RunOptions`]. This can also be
/// used to run a model with different inputs in parallel, by creating separate
/// thread pools.
///
/// ```
/// use std::sync::Arc;
/// use rten::{ThreadPool, RunOptions};
///
/// let pool = ThreadPool::with_num_threads(1);
/// let options = RunOptions::default()
///   .with_thread_pool(Some(Arc::new(pool)));
///
/// // Pass options to `Model::run`.
/// ```
///
/// The number of threads in the default thread pool can be customized by
/// setting the `RTEN_NUM_THREADS` environment variable.
///
/// ### Graph optimizations
///
/// By default RTen applies various optimizations to the model when it is loaded
/// to improve inference performance. These optimizations guarantee to preserve
/// the model's inputs and outputs, but other nodes may be replaced or
/// eliminated. To configure or disable optimizations, use [`ModelOptions`].
///
/// ```
/// use rten::ModelOptions;
///
/// let model = ModelOptions::with_all_ops()
///   .enable_optimization(false)
///   .load_file("model.onnx");
/// ```
///
/// Optimizations applied include:
///
/// - **Fusion**: Fusions combine operators to reduce the amount of data
///   movement required during inference.
/// - **Constant propagation**: Subgraphs which don't depend on
///   dynamic inputs are evaluated once at model load and replaced with the
///   result.
/// - **Identity elimination**: Operators which return their inputs unchanged
///   are removed.
///
/// ### Weight prepacking
///
/// In addition to optimizing the structure of the graph, RTen can create copies
/// of the weights with an optimized ("packed") data layout at model load time.
/// Enabling this will increase model load time and memory usage but may reduce
/// the time taken per inference. When this option is disabled, weights are
/// packed temporarily on-demand just before they are used for computation.
///
/// For generative transformer models (aka. "transformer decoders") prepacking
/// is generally only useful when processing multiple input tokens at a time.
///
/// Prepacking is disabled by default but can be enabled using [`ModelOptions`].
///
/// ### Partial evaluation
///
/// Some models, such as transformer decoders, are evaluated repeatedly in a
/// loop. If such models have inputs which are constant in each iteration of the
/// loop, execution can be sped up by using partial evaluation. This involves
/// evaluating the part of the graph that depends only on the constant inputs
/// once, outside the loop. To do this use [`Model::partial_run`].
///
/// ### Profiling
///
/// There is built-in support for reporting on the time taken for each operator,
/// which can optionally be broken down by input shape. These can be enabled via
/// fields of the [`RunOptions`] struct.
///
/// As a development convenience, setting the `RTEN_TIMING` environment variable
/// to "1" will cause timings for each operator to be reported after each
/// inference.
///
/// ## Compile time and binary size
///
/// This section describes model configuration that affects compile time and
/// binary size of projects using RTen.
///
/// ### Custom operator registries
///
/// By default all ONNX operators are available for use by models, except for
/// those which require enabling additional crate features.
///
/// You can reduce binary size and compilation time by loading a model with only
/// a subset of operators enabled. Operators that are not enabled will be
/// excluded from the binary during linking. To do this, create a custom
/// operator registry using the [`op_registry`](crate::op_registry) macro and
/// configure the model to use it using [`ModelOptions::with_ops`].
pub struct Model {
    graph: Graph,
    metadata: ModelMetadata,
    weight_cache: WeightCache,
}

impl Model {
    /// Load a serialized model from a `.onnx` or `.rten` file.
    ///
    /// This method reads the entire file into memory. For large models (hundreds
    /// of MB or more), [`load_mmap`](Model::load_mmap) can be faster.
    ///
    /// # External data
    ///
    /// When using this method, ONNX models with external data are supported.
    /// See the notes in [`load_mmap`](Self::load_mmap) for more details.
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Model, LoadError> {
        ModelOptions::with_all_ops().load_file(path)
    }

    /// Load a serialized model from a byte buffer.
    ///
    /// The model can be in either ONNX or RTen format. The model type is
    /// detected automatically.
    ///
    /// # External data
    ///
    /// To load ONNX models from a byte buffer that reference external data, use
    /// [`ModelOptions::external_data`] and [`ModelOptions::load`].
    pub fn load(data: Vec<u8>) -> Result<Model, LoadError> {
        ModelOptions::with_all_ops().load(data)
    }

    /// Load a serialized model from a static byte slice.
    ///
    /// This is useful for loading models embedded in the binary via
    /// [`include_bytes`] for example.
    ///
    /// The model can be in either ONNX or RTen format. The model type is
    /// detected automatically.
    ///
    /// # External data
    ///
    /// To load ONNX models from a static slice that reference external data,
    /// use [`ModelOptions::external_data_static`] and
    /// [`ModelOptions::load_static_slice`].
    pub fn load_static_slice(data: &'static [u8]) -> Result<Model, LoadError> {
        ModelOptions::with_all_ops().load_static_slice(data)
    }

    /// Load a serialized model by mapping a view of a file as memory.
    ///
    /// This method requires the `mmap` crate feature to be enabled.
    ///
    /// Loading a model via memory-mapping makes the initial load of the model
    /// faster for large models, **if the format supports memory-mapped data**
    /// (see section below), and also enables sharing the data with other
    /// processes.
    ///
    /// # Memory usage
    ///
    /// If a process uses `load_file`, its private
    /// memory usage will be the size of the model plus its working space. If a
    /// process uses `load_mmap`, its private memory usage will only be that
    /// needed for working space.
    ///
    /// The first _run_ of a memory-mapped model will be slower than if the file
    /// is read into memory first and then executed. Depending on the size of
    /// the model, the overall time taken for load + first run may be less or
    /// about the same.  Subsequent model executions should the same time.
    ///
    /// # Compatible formats
    ///
    /// Memory mapping is supported for:
    ///
    ///  - ONNX files with external data (eg. a `model.onnx` file with weights
    ///    stored in `model.onnx.data`)
    ///  - .rten format model files created via [rten-convert](https://pypi.org/project/rten-convert/)
    ///
    /// For ONNX files with embedded weights, `load_mmap` will fall back to
    /// copying the weights into private memory, the same as if `load_file` was
    /// used. The reason for this is that tensor data needs to be appropriately
    /// aligned and this is not the case for `.onnx` files with embedded
    /// weights.
    ///
    /// # External data
    ///
    /// Models in ONNX format may store data in an external file (eg.
    /// `model.onnx.data`). When weights are loaded from an external file, they
    /// are loaded via regular IO if the model is loaded with
    /// [`load_file`](Self::load_mmap) or memory-mapping if the model is loaded
    /// with [`load_mmap`](Self::load_mmap).
    ///
    /// # Safety
    ///
    /// This method is marked unsafe because undefined behavior can be caused if
    /// a memory-mapped model file is modified on disk while it is being used by
    /// a `Model`. Callers will need to decide whether this is an acceptable
    /// risk for their context. As a rule of thumb, this risk will be acceptable
    /// for most applications (see [this
    /// discussion](https://github.com/BurntSushi/ripgrep/issues/581) for
    /// example), but when writing a library, you will most likely want to defer
    /// the choice to the caller of the library.
    ///
    /// As a point of comparison, other machine learning
    /// runtimes like ONNX Runtime and llama.cpp do use memory mapping by
    /// default.
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
    pub unsafe fn load_mmap<P: AsRef<Path>>(path: P) -> Result<Model, LoadError> {
        let opts = ModelOptions::with_all_ops();
        unsafe { opts.load_mmap(path) }
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
            .ok_or_else(|| RunErrorImpl::InvalidNodeName(id.to_string()).into())
    }

    /// Return metadata about a node in the model's graph.
    pub fn node_info(&self, id: NodeId) -> Option<NodeInfo<'_>> {
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
    /// This method allows for a variable number of outputs. For the common
    /// case where the number of outputs is fixed, [`Model::run_n`] is preferred
    /// as it returns an array which can be destructured to extract individual
    /// outputs: `let [output_one, output_two] = model.run_n(...)`.
    ///
    /// The input and output nodes are specified via IDs looked up via
    /// [`node_id`](Model::node_id).
    ///
    /// Input values are validated against the shape and dtype specified in the
    /// model, which can be queried via [`Model::node_info`].
    pub fn run(
        &self,
        inputs: Vec<(NodeId, ValueOrView)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<Value>, RunError> {
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
    ///
    /// The input and output nodes are specified via IDs looked up via
    /// [`node_id`](Model::node_id).
    pub fn run_n<const N: usize>(
        &self,
        inputs: Vec<(NodeId, ValueOrView)>,
        outputs: [NodeId; N],
        opts: Option<RunOptions>,
    ) -> Result<[Value; N], RunError> {
        let result = self.run(inputs, &outputs, opts)?;
        Ok(result.try_into().expect("wrong output count"))
    }

    /// Run a model with a single input and output.
    ///
    /// This is a simplified version of [`Model::run`] for the common case of
    /// executing a model with a single input and output.
    pub fn run_one(&self, input: ValueOrView, opts: Option<RunOptions>) -> Result<Value, RunError> {
        let &input_id = self
            .input_ids()
            .first()
            .ok_or(RunErrorImpl::InvalidNodeId)?;
        let &output_id = self
            .output_ids()
            .first()
            .ok_or(RunErrorImpl::InvalidNodeId)?;
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
        inputs: Vec<(NodeId, ValueOrView)>,
        outputs: &[NodeId],
        opts: Option<RunOptions>,
    ) -> Result<Vec<(NodeId, Value)>, RunError> {
        self.graph.partial_run(inputs, outputs, opts)
    }

    // For model loader tests.
    #[cfg(test)]
    fn graph(&self) -> &Graph {
        &self.graph
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let node_names = |ids: &[NodeId]| -> Vec<&str> {
            ids.iter()
                .filter_map(|id| self.node_info(*id))
                .map(|info| info.name().unwrap_or(""))
                .collect()
        };

        let input_names = node_names(self.input_ids());
        let output_names = node_names(self.output_ids());

        f.debug_struct("Model")
            .field("inputs", &input_names)
            .field("outputs", &output_names)
            .finish()
    }
}

/// Provides access to metadata about a graph node.
pub struct NodeInfo<'a> {
    node: &'a Node,
}

impl<'a> NodeInfo<'a> {
    /// Return the unique name associated with the node, if present.
    pub fn name(&self) -> Option<&'a str> {
        self.node.name()
    }

    /// Return the tensor shape associated with a node.
    ///
    /// The shape can be a combination of fixed values and symbolic names.
    pub fn shape(&self) -> Option<Vec<Dimension>> {
        self.node.shape().map(|n| n.into_owned())
    }

    /// Return the expected data type for this node at runtime.
    ///
    /// For constants the data type is always known. For values the data type
    /// may be specified. For operators this always returns `None`.
    pub fn dtype(&self) -> Option<ValueType> {
        self.node.dtype()
    }
}

impl<'a> std::fmt::Debug for NodeInfo<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeInfo")
            .field("name", &self.name())
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
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
                "filter-op" => {
                    for op_name in val.split(',') {
                        opts.timing_filter
                            .push(TimingFilter::Operator(op_name.to_string()));
                    }
                }
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

/// Set whether shape and type inference is run when loading a model.
///
/// See [`ModelOptions::shape_inference`].
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeInferenceMode {
    /// Do not run shape inference
    Off,
    /// Run shape inference in best-effort mode.
    ///
    /// If shape inference is unsupported or fails for any operators, the
    /// model will still load but some optimizations might be missed.
    On,
    /// Run shape inference in strict mode.
    ///
    /// The model will fail to load if shape inference cannot infer the shapes
    /// or types of any values.
    Strict,
}

/// Options which customize how a model is loaded.
///
/// This enables more advanced use cases such as loading a model with only
/// a subset of operators available, or with different sets of optimizations
/// applied.
#[derive(Clone)]
pub struct ModelOptions {
    registry: Arc<OpRegistry>,
    optimize: bool,
    prepack_weights: bool,
    external_data: HashMap<String, Arc<ConstantStorage>>,
    infer_shapes: ShapeInferenceMode,
}

impl ModelOptions {
    /// Create a set of options with all operators enabled.
    pub fn with_all_ops() -> ModelOptions {
        Self::with_ops(OpRegistry::with_all_ops())
    }

    /// Create a set of options with a custom set of operators enabled.
    ///
    /// This can be used to reduce binary size by excluding operators that
    /// the model will not use.
    pub fn with_ops(ops: OpRegistry) -> ModelOptions {
        ModelOptions {
            registry: ops.into(),
            optimize: true,
            prepack_weights: false,
            external_data: HashMap::new(),
            infer_shapes: ShapeInferenceMode::On,
        }
    }

    /// Set whether graph optimizations are enabled.
    pub fn enable_optimization(&mut self, enable: bool) -> &mut Self {
        self.optimize = enable;
        self
    }

    /// Enable shape and type inference for values.
    ///
    /// This is equivalent to `self.shape_inference(ShapeInferenceMode::On)`.
    #[deprecated]
    pub fn enable_shape_inference(&mut self, enable: bool) -> &mut Self {
        self.infer_shapes = if enable {
            ShapeInferenceMode::On
        } else {
            ShapeInferenceMode::Off
        };
        self
    }

    /// Set whether shape and type inference is run as part of optimization.
    ///
    /// Shape inference is needed for some optimizations in order to verify that
    /// they are safe, by checking the shape and/or type of various values. By
    /// default shape inference is [enabled](ShapeInferenceMode::On) but will
    /// fail gracefully if the shapes of some values cannot be inferred. To
    /// enforce that shape inference is fully successful, [strict
    /// mode](ShapeInferenceMode::Strict) can be enabled.
    pub fn shape_inference(&mut self, mode: ShapeInferenceMode) -> &mut Self {
        self.infer_shapes = mode;
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

    /// Provide the content of an external data file as a buffer.
    ///
    /// This is used when an ONNX model loaded via [`load`](Self::load) or
    /// [`load_static_slice`](Self::load_static_slice) references data in an
    /// external file.
    pub fn external_data(&mut self, path: &str, buf: Vec<u8>) -> &mut Self {
        self.external_data_impl(path, ConstantStorage::Buffer(buf))
    }

    /// Provide the content of an external data file as a static slice.
    ///
    /// This can be used together with
    /// [`include_external_data`](crate::include_external_data) to embed a
    /// model's weights in the program.
    ///
    /// This is used when an ONNX model loaded via [`load`](Self::load) or
    /// [`load_static_slice`](Self::load_static_slice) references data in an
    /// external file.
    pub fn external_data_static(&mut self, path: &str, buf: &'static [u8]) -> &mut Self {
        self.external_data_impl(path, ConstantStorage::StaticSlice(buf))
    }

    fn external_data_impl(&mut self, path: &str, buf: ConstantStorage) -> &mut Self {
        self.external_data.insert(path.to_string(), Arc::new(buf));
        self
    }

    /// Load the model from a file. See [`Model::load_file`].
    pub fn load_file<P: AsRef<Path>>(&self, path: P) -> Result<Model, LoadError> {
        match FileType::from_path(path.as_ref()).ok_or(LoadErrorImpl::UnknownFileType)? {
            #[cfg(feature = "rten_format")]
            FileType::Rten => {
                use crate::constant_storage::ConstantStorage;

                let data = std::fs::read(&path).map_err(LoadErrorImpl::ReadFailed)?;
                let storage = Arc::new(ConstantStorage::Buffer(data));
                rten_loader::load(storage, self)
            }
            #[cfg(not(feature = "rten_format"))]
            FileType::Rten => Err(LoadErrorImpl::FormatNotEnabled.into()),
            #[cfg(feature = "onnx_format")]
            FileType::Onnx => {
                let loader = external_data::FileLoader::new(path.as_ref())?;
                onnx_loader::load(
                    onnx_loader::Source::Path(path.as_ref()),
                    Some(&loader),
                    self,
                )
            }
            #[cfg(not(feature = "onnx_format"))]
            FileType::Onnx => Err(LoadErrorImpl::FormatNotEnabled.into()),
        }
    }

    #[cfg(feature = "onnx_format")]
    fn mem_data_loader(&self) -> external_data::MemLoader {
        // This clones the map from path to reference-counted storage, but not
        // the storage itself.
        let external_data = self.external_data.clone();
        external_data::MemLoader::new(external_data)
    }

    /// Load the model from a data buffer. See [`Model::load`].
    pub fn load(&self, data: Vec<u8>) -> Result<Model, LoadError> {
        match FileType::from_buffer(&data).ok_or(LoadErrorImpl::UnknownFileType)? {
            #[cfg(feature = "rten_format")]
            FileType::Rten => {
                use crate::constant_storage::ConstantStorage;
                let storage = Arc::new(ConstantStorage::Buffer(data));
                rten_loader::load(storage, self)
            }
            #[cfg(not(feature = "rten_format"))]
            FileType::Rten => Err(LoadErrorImpl::FormatNotEnabled.into()),
            #[cfg(feature = "onnx_format")]
            FileType::Onnx => {
                let loader = self.mem_data_loader();
                onnx_loader::load(onnx_loader::Source::Buffer(&data), Some(&loader), self)
            }
            #[cfg(not(feature = "onnx_format"))]
            FileType::Onnx => Err(LoadErrorImpl::FormatNotEnabled.into()),
        }
    }

    /// Load the model from a static slice of bytes. See [`Model::load_static_slice`].
    pub fn load_static_slice(&self, data: &'static [u8]) -> Result<Model, LoadError> {
        match FileType::from_buffer(data).ok_or(LoadErrorImpl::UnknownFileType)? {
            #[cfg(feature = "rten_format")]
            FileType::Rten => {
                use crate::constant_storage::ConstantStorage;
                let storage = Arc::new(ConstantStorage::StaticSlice(data));
                rten_loader::load(storage, self)
            }
            #[cfg(not(feature = "rten_format"))]
            FileType::Rten => Err(LoadErrorImpl::FormatNotEnabled.into()),
            #[cfg(feature = "onnx_format")]
            FileType::Onnx => {
                let loader = self.mem_data_loader();
                onnx_loader::load(onnx_loader::Source::Buffer(data), Some(&loader), self)
            }
            #[cfg(not(feature = "onnx_format"))]
            FileType::Onnx => Err(LoadErrorImpl::FormatNotEnabled.into()),
        }
    }

    /// Load the model from a memory-mapped view of a file.
    ///
    /// This method is only efficient for `.rten` files and ONNX models with
    /// external weights. See [`Model::load_mmap`] for more details.
    ///
    /// To limit the scope of `unsafe` when using this API, you can construct
    /// a `ModelOptions` and clone it before calling `load_mmap`:
    ///
    /// ```no_run
    /// use rten::ModelOptions;
    ///
    /// let opts = ModelOptions::default().prepack_weights(true).clone();
    /// let model = unsafe { opts.load_mmap("model.rten") };
    /// ```
    ///
    /// If the model references tensor data in external files, that data will
    /// also be loaded via memory-mapping.
    ///
    /// # Safety
    ///
    /// See notes in [`Model::load_mmap`].
    #[cfg(feature = "mmap")]
    pub unsafe fn load_mmap<P: AsRef<Path>>(&self, path: P) -> Result<Model, LoadError> {
        let file = File::open(&path).map_err(LoadErrorImpl::ReadFailed)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(LoadErrorImpl::ReadFailed)?;
        match FileType::from_path(path.as_ref()).ok_or(LoadErrorImpl::UnknownFileType)? {
            #[cfg(feature = "rten_format")]
            FileType::Rten => {
                use crate::constant_storage::ConstantStorage;
                let storage = Arc::new(ConstantStorage::Mmap(mmap));
                rten_loader::load(storage, self)
            }
            #[cfg(not(feature = "rten_format"))]
            FileType::Rten => Err(LoadErrorImpl::FormatNotEnabled.into()),
            #[cfg(feature = "onnx_format")]
            FileType::Onnx => {
                // Safety: By calling `load_mmap` the caller has accepted the
                // associated risks, so we can also use mmap to load external
                // data files.
                let loader = unsafe { external_data::MmapLoader::new(path.as_ref()) }?;
                onnx_loader::load(onnx_loader::Source::Buffer(&mmap), Some(&loader), self)
            }
            #[cfg(not(feature = "onnx_format"))]
            FileType::Onnx => Err(LoadErrorImpl::FormatNotEnabled.into()),
        }
    }

    /// Convert optimization settings into the internal representation passed
    /// to the graph optimizer.
    fn optimize_mode(&self) -> OptimizeMode {
        if self.optimize {
            OptimizeMode::On(OptimizeOptions {
                infer_shapes: match self.infer_shapes {
                    ShapeInferenceMode::Off => None,
                    ShapeInferenceMode::On => Some(InferShapeOptions {
                        strict: false,
                        ..Default::default()
                    }),
                    ShapeInferenceMode::Strict => Some(InferShapeOptions {
                        strict: true,
                        ..Default::default()
                    }),
                },
            })
        } else {
            OptimizeMode::Off
        }
    }
}

impl std::fmt::Debug for ModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelOptions")
            .field("optimize", &self.optimize)
            .field("prepack_weights", &self.prepack_weights)
            .finish()
    }
}

/// Create model options using [`ModelOptions::with_all_ops`].
impl Default for ModelOptions {
    fn default() -> Self {
        ModelOptions::with_all_ops()
    }
}

#[derive(Clone)]
enum OptimizeMode {
    // Disable graph optimizations.
    Off,

    // Enable graph optimizations.
    On(OptimizeOptions),
}

/// Embed data in the program with sufficient alignment for use with
/// [`ModelOptions::external_data_static`].
///
/// This is like [`include_bytes`] but ensures the embedded data has the
/// necessary alignment for model weights.
#[macro_export]
macro_rules! include_external_data {
    ($path:literal) => {{
        // Based on https://users.rust-lang.org/t/can-i-conveniently-compile-bytes-into-a-rust-program-with-a-specific-alignment/24049/2
        #[repr(C)]
        pub struct AlignedAs<Align, Bytes: ?Sized> {
            pub _align: [Align; 0],
            pub bytes: Bytes,
        }
        // Use f64 as the alignment type because that's the largest supported
        // element type.
        static ALIGNED: &AlignedAs<f64, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };
        &ALIGNED.bytes
    }};
}

#[cfg(test)]
mod tests {
    use rten_onnx::onnx;
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, Tensor, TensorView};

    use crate::graph::{Dimension, RunErrorKind};
    use crate::model::onnx_builder::{
        GraphProtoExt, ModelProtoExt, NodeProtoExt, ToTensorProto, ValueInfoProtoExt, create_node,
        create_tensor_from_view, create_value_info,
    };
    use crate::model::rten_builder::{MetadataArgs, ModelBuilder, ModelFormat, OpType};
    use crate::model::{LoadErrorKind, Model, ModelOptions};
    use crate::op_registry;
    use crate::ops;
    use crate::value::{DataType, Value, ValueType};

    /// Create an ONNX model which concatenates a constant with the model input
    /// and applies a Relu operator to the result.
    fn generate_model_buffer() -> Vec<u8> {
        let const_val = Tensor::from_data(&[1, 2, 2], vec![0.5, -0.5, 0.1, -0.1]);
        let input_shape: Vec<Dimension> = const_val
            .shape()
            .iter()
            .copied()
            .map(Dimension::Fixed)
            .collect();

        let concat = create_node("Concat")
            .with_name("concat")
            .with_input("const")
            .with_input("input")
            .with_output("concat_out")
            .with_attr("axis", 0i64);
        let relu = create_node("Relu")
            .with_name("relu")
            .with_input("concat_out")
            .with_output("output");

        onnx::GraphProto::default()
            .with_initializer(create_tensor_from_view("const", const_val.view()))
            .with_input(
                create_value_info("input")
                    .with_dtype(onnx::DataType::FLOAT)
                    .with_shape(&input_shape),
            )
            .with_output(
                create_value_info("output")
                    .with_dtype(onnx::DataType::FLOAT)
                    .with_shape(&[
                        Dimension::Fixed(2),
                        Dimension::Fixed(2),
                        Dimension::Fixed(2),
                    ]),
            )
            .with_node(concat)
            .with_node(relu)
            .into_model()
            .with_opset("", 18)
            .with_producer("rten-test", "1.2.3")
            .with_metadata("description", "test model")
            .write_buf()
            .unwrap()
    }

    /// Version of [`generate_model_buffer`] which creates a model in the
    /// `.rten` format.
    fn generate_rten_model_buffer(format: ModelFormat) -> Vec<u8> {
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
    fn check_output(mut result: Vec<Value>) -> Tensor<f32> {
        assert_eq!(result.len(), 1);

        let tensor: Tensor<f32> = result.remove(0).into_tensor::<f32>().unwrap();
        assert_eq!(tensor.shape(), &[2, 2, 2]);
        assert_eq!(tensor.to_vec(), &[0.5, 0., 0.1, 0., 1., 2., 0., 0.]);

        tensor
    }

    #[test]
    fn test_model_input_output_ids() {
        let buffer = generate_model_buffer();

        let model = Model::load(buffer).unwrap();

        // Valid model IDs
        let input_id = model.find_node("input").unwrap();
        let output_id = model.find_node("output").unwrap();

        assert_eq!(model.input_ids(), &[input_id]);
        assert_eq!(model.output_ids(), &[output_id]);

        // Get the same node ID via a convenience method which returns a
        // Result.
        assert_eq!(model.node_id("input").ok(), Some(input_id));

        // Invalid model ID
        assert_eq!(model.find_node("does_not_exist"), None);

        let err = model.node_id("does_not_exist").err().unwrap();
        assert_eq!(err.node_path(), [Some("does_not_exist")]);
        assert_eq!(err.kind(), RunErrorKind::NodeNotFound);
    }

    #[test]
    fn test_unsupported_operator() {
        let buffer = generate_model_buffer();
        let registry = op_registry!();
        let result = ModelOptions::with_ops(registry).load(buffer);
        assert_eq!(
            result.err().map(|err| err.to_string()).as_deref(),
            Some(
                "in node \"concat\": operator error: Concat operator not supported or not enabled"
            )
        );
    }

    #[test]
    fn test_subset_of_ops_enabled() {
        let buffer = generate_model_buffer();
        let registry = op_registry!(Concat, Relu);
        let result = ModelOptions::with_ops(registry).load(buffer);
        assert!(result.is_ok());
    }

    #[test]
    fn test_shape_info() {
        let buffer = generate_model_buffer();
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
        let buffer = generate_model_buffer();
        let model = Model::load(buffer).unwrap();
        let input_id = model.input_ids()[0];

        let dtype = model
            .node_info(input_id)
            .and_then(|ni| ni.dtype())
            .expect("input dtype missing");
        assert_eq!(dtype, ValueType::Tensor(DataType::Float));
    }

    #[test]
    fn test_metadata() {
        let buffer = generate_model_buffer();
        let model = Model::load(buffer).unwrap();
        assert_eq!(model.metadata().producer_name(), Some("rten-test"));
        assert_eq!(model.metadata().producer_version(), Some("1.2.3"));
        assert_eq!(model.metadata().get("description"), Some("test model"));

        // Fields which are only set in `.rten` format models.
        assert_eq!(model.metadata().onnx_hash(), None);
    }

    #[test]
    fn test_input_shape() {
        let buffer = generate_model_buffer();
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
            buffer: Vec<u8>,
            opts: Option<ModelOptions>,
        }

        let cases = [
            Case {
                buffer: generate_model_buffer(),
                opts: None,
            },
            Case {
                buffer: generate_rten_model_buffer(ModelFormat::V1),
                opts: None,
            },
            Case {
                buffer: generate_rten_model_buffer(ModelFormat::V2),
                opts: None,
            },
            // Graph optimizations disabled
            Case {
                buffer: generate_model_buffer(),
                opts: Some({
                    let mut opts = ModelOptions::with_all_ops();
                    opts.enable_optimization(false);
                    opts
                }),
            },
            // Prepacking enabled
            Case {
                buffer: generate_model_buffer(),
                opts: Some({
                    let mut opts = ModelOptions::with_all_ops();
                    opts.prepack_weights(true);
                    opts
                }),
            },
        ];

        for Case { buffer, opts } in cases {
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
                vec![(output_id, Value::FloatTensor(result_tensor))]
            );
        }
    }

    #[test]
    fn test_model_debug() {
        let buffer = generate_model_buffer();
        let model = Model::load(buffer).unwrap();
        let debug_str = format!("{model:?}");
        assert_eq!(
            debug_str,
            "Model { inputs: [\"input\"], outputs: [\"output\"] }"
        );
    }

    #[test]
    fn test_load_invalid_model() {
        struct Case {
            buf: Vec<u8>,
            expected_error: &'static str,
        }

        // This test corrupts the model buffer in ways that are specific to the
        // `.rten` format.
        let buf = generate_rten_model_buffer(ModelFormat::V2);

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
            assert!(
                err.to_string().contains(expected_error),
                "expected \"{}\" to contain \"{}\"",
                err,
                expected_error
            );
        }
    }

    #[test]
    fn test_load_static_slice() {
        let buffer = generate_model_buffer().leak();
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
        let buffer = generate_model_buffer();
        std::fs::write("model-load-file-test.onnx", buffer).unwrap();

        let model = Model::load_file("model-load-file-test.onnx").unwrap();
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
        let buffer = generate_model_buffer();
        std::fs::write("model-load-mmap-test.onnx", buffer).unwrap();

        let model = unsafe { Model::load_mmap("model-load-mmap-test.onnx").unwrap() };
        let input_id = model.input_ids()[0];
        let output_id = model.output_ids()[0];

        let input = generate_input();
        let result = model
            .run(vec![(input_id, input.into())], &[output_id], None)
            .unwrap();
        check_output(result);
    }

    #[test]
    fn test_load_unknown_type() {
        let err = Model::load_file("README.md").err().unwrap();
        assert_eq!(err.kind(), LoadErrorKind::UnknownFileType);
    }

    #[cfg(feature = "onnx_format")]
    #[test]
    fn test_load_onnx() {
        let check_model = |model: Model| {
            assert_eq!(model.input_ids().len(), 1);
            let input_info = model.node_info(model.input_ids()[0]).unwrap();
            assert_eq!(input_info.name().unwrap(), "input");
            assert_eq!(
                input_info.shape().unwrap(),
                [1, 1, 28, 28].map(Dimension::Fixed)
            );

            assert_eq!(model.output_ids().len(), 1);
            let output_info = model.node_info(model.output_ids()[0]).unwrap();
            assert_eq!(output_info.name().unwrap(), "logits");
            assert_eq!(output_info.shape().unwrap(), [1, 10].map(Dimension::Fixed));

            let result = model
                .run_one(NdTensor::full([1, 1, 28, 28], 0.5).into(), None)
                .unwrap();
            assert_eq!(result.shape().as_slice(), &[1, 10]);
        };

        let model_path = "rten-onnx/test-data/mnist.onnx";
        let external_model_path = "rten-onnx/test-data/mnist-external/mnist.onnx";

        // Load from file path.
        let model = Model::load_file(model_path).unwrap();
        check_model(model);

        // Load file with external data.
        let model = Model::load_file(external_model_path).unwrap();
        check_model(model);

        // Load from buffer.
        let onnx_buf = std::fs::read(model_path).unwrap();
        let model = Model::load(onnx_buf).unwrap();
        check_model(model);

        // Load from buffer with external data.
        let onnx_buf = std::fs::read(external_model_path).unwrap();
        let data_buf = std::fs::read(format!("{}.data", external_model_path)).unwrap();
        let model = ModelOptions::with_all_ops()
            .external_data("mnist.onnx.data", data_buf)
            .load(onnx_buf)
            .unwrap();
        check_model(model);

        // Load from static slice with external data
        let model_static = include_bytes!("../rten-onnx/test-data/mnist-external/mnist.onnx");
        let model_data_static =
            crate::include_external_data!("../rten-onnx/test-data/mnist-external/mnist.onnx.data");
        let model = ModelOptions::with_all_ops()
            .external_data_static("mnist.onnx.data", model_data_static)
            .load_static_slice(model_static)
            .unwrap();
        check_model(model);

        // Load model file and external data using mmap.
        #[cfg(feature = "mmap")]
        {
            let model = unsafe { Model::load_mmap(external_model_path) }.unwrap();
            check_model(model);
        }
    }

    #[test]
    fn test_run_one() {
        let buffer = generate_model_buffer();
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
        // An empty input name represents an omitted optional input.
        let node = create_node("Shape")
            .with_name("shape")
            .with_input("")
            .with_output("output");

        let buffer = onnx::GraphProto::default()
            .with_output(create_value_info("output"))
            .with_node(node)
            .into_model()
            .with_opset("", 18)
            .write_buf()
            .unwrap();

        // Load with optimizations disabled to prevent the optimizer from
        // running the graph as part of constant propagation.
        let model = ModelOptions::with_all_ops()
            .enable_optimization(false)
            .load(buffer)
            .unwrap();

        let output_id = model.find_node("output").unwrap();
        let err = model.run(vec![], &[output_id], None).err().unwrap();
        assert_eq!(err.node_path(), [Some("shape")]);
        assert_eq!(err.kind(), RunErrorKind::OperatorError);
    }

    // This test exercises basic execution of all operators. It doesn't check
    // the results of operators, it just makes sure they can be deserialized and
    // executed successfully.
    #[test]
    fn test_all_op_types() {
        /// Create an operator node with a given type and inputs.
        fn node(op_type: &str, inputs: &[&str]) -> onnx::NodeProto {
            let mut node = create_node(op_type);
            for input in inputs {
                node = node.with_input(input);
            }
            node
        }

        /// Builds a graph in which the outputs of every operator are also
        /// outputs of the graph.
        struct GraphBuilder {
            graph: onnx::GraphProto,

            /// Names of all operator output values.
            op_outputs: Vec<String>,
        }

        impl GraphBuilder {
            fn add_input(&mut self, name: &str) {
                self.graph.input.push(create_value_info(name));
            }

            fn add_constant<T: ToTensorProto>(&mut self, name: &str, value: TensorView<T>) {
                self.graph
                    .initializer
                    .push(create_tensor_from_view(name, value));
            }

            /// Add an operator with a single output named `{op_type}_out` and
            /// return the output name.
            fn add_operator(&mut self, node: onnx::NodeProto) -> String {
                let output = format!("{}_out", node.op_type.as_deref().unwrap_or_default());
                self.add_operator_with_outputs(node, &[&output]);
                output
            }

            /// Add an operator with explicitly named outputs.
            ///
            /// This is used for operators which have more than one output.
            fn add_operator_with_outputs(&mut self, node: onnx::NodeProto, outputs: &[&str]) {
                // Set the operator node name to match the operator type. This
                // relies on each operator being used only once in the graph.
                let op_type = node.op_type.clone().unwrap_or_default();
                let mut node = node.with_name(&op_type);
                for output in outputs {
                    node = node.with_output(output);
                    self.graph.output.push(create_value_info(output));
                    self.op_outputs.push(output.to_string());
                }
                self.graph.node.push(node);
            }
        }

        let mut builder = GraphBuilder {
            graph: onnx::GraphProto::default(),
            op_outputs: Vec::new(),
        };

        for input in [
            "input",
            "input.2d",
            "input.bool",
            "input.u8",
            "input.2d.u8",
            "input.2d.i8",
        ] {
            builder.add_input(input);
        }

        // 4D shape used as the primary input to test most operators (eg. NCHW image). A few
        // require a different shape.
        let input_shape = [1, 1, 3, 3];

        builder.add_constant("kernel", Tensor::from_data(&[1, 1, 1, 1], vec![0.5]).view());
        builder.add_constant(
            "kernel.i8",
            Tensor::from_data(&[1, 1, 1, 1], vec![0i8]).view(),
        );

        builder.add_operator(node("Abs", &["input"]));
        builder.add_operator(node("Acos", &["input"]));
        builder.add_operator(node("Acosh", &["input"]));
        builder.add_operator(node("Add", &["input", "input"]));
        builder.add_operator(node("And", &["input.bool", "input.bool"]));
        builder.add_operator(
            node("ArgMax", &["input"])
                .with_attr("axis", 3i64)
                .with_attr("keepdims", false),
        );
        builder.add_operator(
            node("ArgMin", &["input"])
                .with_attr("axis", 3i64)
                .with_attr("keepdims", false),
        );
        builder.add_operator(node("Asin", &["input"]));
        builder.add_operator(node("Asinh", &["input"]));
        builder.add_operator(node("Atan", &["input"]));
        builder.add_operator(node("Atanh", &["input"]));
        builder.add_operator(
            node("AveragePool", &["input"])
                .with_attr("kernel_shape", vec![2i64, 2])
                .with_attr("strides", vec![2i64, 2])
                .with_attr("pads", vec![0i64, 0, 0, 0])
                .with_attr("count_include_pad", false)
                .with_attr("ceil_mode", false),
        );

        // Dummy value for BatchNormalization inputs which are vectors with
        // per-channel values.
        builder.add_constant("batch_norm_param", Tensor::from([1.0]).view());
        builder.add_operator(
            node(
                "BatchNormalization",
                &[
                    "input",
                    "batch_norm_param", /* scale */
                    "batch_norm_param", /* bias */
                    "batch_norm_param", /* mean */
                    "batch_norm_param", /* variance */
                ],
            )
            .with_attr("epsilon", 1e-5),
        );

        builder.add_operator(
            node("Cast", &["input"]).with_attr("to", i64::from(onnx::DataType::FLOAT.0)),
        );
        builder.add_operator(node("CastLike", &["input", "input"]));
        builder.add_operator(node("Ceil", &["input"]));

        builder.add_constant("clip_min", Tensor::from(1.).view());
        builder.add_constant("clip_max", Tensor::from(6.).view());
        builder.add_operator(node("Clip", &["input", "clip_min", "clip_max"]));
        builder.add_operator(node("Concat", &["input", "input"]).with_attr("axis", 0i64));

        builder.add_constant("shape", Tensor::from([1, 5, 10]).view());
        builder.add_operator(node("ConstantOfShape", &["shape"]).with_attr(
            "value",
            create_tensor_from_view("value", Tensor::from([42]).view()),
        ));

        builder.add_operator(
            node("Conv", &["input", "kernel"])
                .with_attr("kernel_shape", vec![1i64, 1])
                .with_attr("dilations", vec![1i64, 1])
                .with_attr("group", 1i64)
                .with_attr("pads", vec![1i64, 1, 1, 1])
                .with_attr("strides", vec![1i64, 1]),
        );
        builder.add_operator(
            node("ConvInteger", &["input.u8", "kernel.i8"])
                .with_attr("kernel_shape", vec![1i64, 1])
                .with_attr("dilations", vec![1i64, 1])
                .with_attr("group", 1i64)
                .with_attr("pads", vec![1i64, 1, 1, 1])
                .with_attr("strides", vec![1i64, 1]),
        );
        builder.add_operator(
            node("ConvTranspose", &["input", "kernel"])
                .with_attr("kernel_shape", vec![1i64, 1])
                .with_attr("dilations", vec![2i64, 2])
                .with_attr("group", 1i64)
                .with_attr("pads", vec![0i64, 0, 0, 0])
                .with_attr("strides", vec![2i64, 2]),
        );
        builder.add_operator(node("Cos", &["input"]));
        builder.add_operator(node("Cosh", &["input"]));

        builder.add_constant("cum_sum_axis", Tensor::from(0).view());
        builder.add_operator(
            node("CumSum", &["input", "cum_sum_axis"])
                .with_attr("exclusive", true)
                .with_attr("reverse", true),
        );

        let const_u8_val = Tensor::from([0u8, 1, 2, 3, 4]);
        builder.add_constant("const.u8", const_u8_val.view());

        let const_f32_val = const_u8_val.map(|x| *x as f32);
        builder.add_constant("const.f32", const_f32_val.view());

        builder.add_constant("scale", Tensor::from(1.).view());
        builder.add_constant("zero_point", Tensor::from(0u8).view());
        builder.add_operator(
            node("DequantizeLinear", &["const.u8", "scale", "zero_point"]).with_attr("axis", 0i64),
        );
        builder.add_operator(
            node("DepthToSpace", &["input"])
                .with_attr("mode", "DCR".to_string())
                .with_attr("blocksize", 1i64),
        );
        builder.add_operator(
            node("QuantizeLinear", &["const.f32", "scale", "zero_point"]).with_attr("axis", 0i64),
        );

        builder.add_operator(node("Div", &["input", "input"]));
        #[cfg(feature = "random")]
        builder.add_operator_with_outputs(
            node("Dropout", &["input.2d"]),
            &["Dropout_out", "Dropout_out_mask"],
        );
        builder.add_operator(node("Elu", &["input"]).with_attr("alpha", 1.0));
        builder.add_operator(node("Equal", &["input", "input"]));
        builder.add_operator(node("Erf", &["input"]));
        builder.add_operator(node("Exp", &["input"]));

        builder.add_constant("expand_shape", Tensor::from([2, 2, 3, 3]).view());
        builder.add_operator(node("Expand", &["input", "expand_shape"]));
        builder.add_operator(node("EyeLike", &["input.2d"]).with_attr("k", 2i64));

        builder.add_operator(node("Flatten", &["input"]).with_attr("axis", 1i64));
        builder.add_operator(node("Floor", &["input"]));

        builder.add_constant("gather_indices", Tensor::from([0]).view());
        builder.add_operator(node("Gather", &["input", "gather_indices"]).with_attr("axis", 0i64));

        builder.add_constant(
            "gather_elements_indices",
            Tensor::<i32>::zeros(&input_shape).view(),
        );
        builder.add_operator(
            node("GatherElements", &["input", "gather_elements_indices"]).with_attr("axis", 0i64),
        );
        builder.add_operator(node("Gelu", &["input"]).with_attr("approximate", "none".to_string()));
        builder.add_operator(
            node("Gemm", &["input.2d", "input.2d"])
                .with_attr("alpha", 1.0)
                .with_attr("beta", 1.0)
                .with_attr("transA", false)
                .with_attr("transB", false),
        );
        builder.add_operator(node("GlobalAveragePool", &["input"]));
        builder.add_operator(node("GlobalMaxPool", &["input"]));
        builder.add_operator(node("Greater", &["input", "input"]));
        builder.add_operator(node("GreaterOrEqual", &["input", "input"]));
        builder.add_operator(
            node("HardSigmoid", &["input"])
                .with_attr("alpha", 0.2)
                .with_attr("beta", 0.5),
        );
        builder.add_operator(node("HardSwish", &["input"]));

        // TODO - Add GRU operator

        builder.add_operator(node("Identity", &["input"]));

        // If operator. Each branch is a subgraph which returns a constant. The
        // constant is passed through an `Identity` operator so that the
        // subgraph's output has a different name than the constant.
        builder.add_constant("if_cond", Tensor::from(1).view());
        let if_branch = |name: &str, value: i32| {
            onnx::GraphProto::default()
                .with_initializer(create_tensor_from_view(
                    &format!("{name}_const"),
                    Tensor::from(value).view(),
                ))
                .with_node(
                    node("Identity", &[&format!("{name}_const")])
                        .with_name(name)
                        .with_output(name),
                )
                .with_output(create_value_info(name))
        };
        builder.add_operator(
            node("If", &["if_cond"])
                .with_attr("then_branch", if_branch("then_out", 2))
                .with_attr("else_branch", if_branch("else_out", 3)),
        );

        builder.add_constant("instance_norm_scale", Tensor::from([1.0]).view());
        builder.add_constant("instance_norm_bias", Tensor::from([1.0]).view());
        builder.add_operator(
            node(
                "InstanceNormalization",
                &["input", "instance_norm_scale", "instance_norm_bias"],
            )
            .with_attr("epsilon", 1e-5),
        );
        builder.add_operator(node("IsInf", &["input"]));
        builder.add_operator(node("IsNaN", &["input"]));

        let layer_norm_scale_val = Tensor::full(&[input_shape[input_shape.len() - 1]], 1.);
        builder.add_constant("layer_norm_scale", layer_norm_scale_val.view());
        builder.add_constant("layer_norm_bias", layer_norm_scale_val.view());
        builder.add_operator(
            node(
                "LayerNormalization",
                &["input", "layer_norm_scale", "layer_norm_bias"],
            )
            .with_attr("axis", -1i64)
            .with_attr("epsilon", 1e-5),
        );

        builder.add_operator(node("LeakyRelu", &["input"]).with_attr("alpha", 0.01));
        builder.add_operator(node("Less", &["input", "input"]));
        builder.add_operator(node("LessOrEqual", &["input", "input"]));
        builder.add_operator(node("Log", &["input"]));
        builder.add_operator(node("LogSoftmax", &["input"]).with_attr("axis", 1i64));

        // TODO - Add LSTM operator

        builder.add_operator(node("MatMul", &["input.2d", "input.2d"]));
        builder.add_operator(node("MatMulInteger", &["input.2d.u8", "input.2d.i8"]));

        builder.add_operator(node("Max", &["input", "input"]));
        builder.add_operator(
            node("MaxPool", &["input"])
                .with_attr("kernel_shape", vec![2i64, 2])
                .with_attr("strides", vec![2i64, 2])
                .with_attr("pads", vec![0i64, 0, 0, 0])
                .with_attr("ceil_mode", false),
        );
        builder.add_operator(node("Mean", &["input", "input"]));
        builder.add_operator(node("Min", &["input", "input"]));
        builder.add_operator(node("Mod", &["input", "input"]).with_attr("fmod", false));
        builder.add_operator(node("Mul", &["input", "input"]));
        builder.add_operator(node("Neg", &["input"]));

        let nms_n_boxes = 10;
        let nms_n_classes = 20;
        builder.add_constant(
            "nms_boxes",
            Tensor::<f32>::zeros(&[1, nms_n_boxes, 4]).view(),
        );
        builder.add_constant(
            "nms_scores",
            Tensor::<f32>::zeros(&[1, nms_n_classes, nms_n_boxes]).view(),
        );
        builder.add_constant("nms_max_outputs_per_class", Tensor::from(10).view());
        builder.add_constant("nms_iou_threshold", Tensor::from(0.45).view());
        builder.add_constant("nms_score_threshold", Tensor::from(0.2).view());
        builder.add_operator(
            node(
                "NonMaxSuppression",
                &[
                    "nms_boxes",
                    "nms_scores",
                    "nms_max_outputs_per_class",
                    "nms_iou_threshold",
                    "nms_score_threshold",
                ],
            )
            .with_attr("center_point_box", true),
        );

        builder.add_operator(node("NonZero", &["input"]));
        builder.add_operator(node("Not", &["input.bool"]));

        builder.add_constant("onehot_indices", Tensor::from([0, 1, 2]).view());
        builder.add_constant("onehot_depth", Tensor::from(5).view());
        builder.add_constant("onehot_values", Tensor::from([1., 0.]).view());
        builder.add_operator(
            node(
                "OneHot",
                &["onehot_indices", "onehot_depth", "onehot_values"],
            )
            .with_attr("axis", -1i64),
        );

        builder.add_operator(node("Or", &["input.bool", "input.bool"]));

        builder.add_constant("pads", Tensor::from([0, 0, 1, 1, 0, 0, 1, 1]).view());
        builder.add_operator(node("Pad", &["input", "pads"]));
        builder.add_operator(node("Pow", &["input", "input"]));

        #[cfg(feature = "random")]
        {
            builder.add_operator(
                node("RandomNormal", &[])
                    .with_attr("shape", vec![50i64, 50])
                    .with_attr("mean", 0.)
                    .with_attr("scale", 1.),
            );
            builder.add_operator(
                node("RandomNormalLike", &["input"])
                    .with_attr("mean", 0.)
                    .with_attr("scale", 1.),
            );
            builder.add_operator(
                node("RandomUniform", &[])
                    .with_attr("shape", vec![50i64, 50])
                    .with_attr("low", 0.)
                    .with_attr("high", 1.),
            );
            builder.add_operator(
                node("RandomUniformLike", &["input"])
                    .with_attr("low", 0.)
                    .with_attr("high", 1.),
            );
            builder.add_operator(node("Multinomial", &["input.2d"]).with_attr("sample_size", 4i64));
        }

        for input in ["range_start", "range_limit", "range_delta"] {
            builder.add_input(input);
        }
        let range_out = builder.add_operator(node(
            "Range",
            &["range_start", "range_limit", "range_delta"],
        ));

        builder.add_operator(node("Reciprocal", &["input"]));
        for reduce_op in [
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
            "ReduceL1",
            "ReduceL2",
        ] {
            builder.add_operator(node(reduce_op, &["input"]).with_attr("keepdims", false));
        }
        builder.add_operator(node("Relu", &["input"]));

        builder.add_constant("new_shape", Tensor::from([9]).view());
        builder
            .add_operator(node("Reshape", &["input", "new_shape"]).with_attr("allowzero", false));

        builder.add_constant(
            "resize_roi",
            Tensor::from([0., 0., 0., 0., 1., 1., 1., 1.]).view(),
        );
        builder.add_constant("resize_scales", Tensor::from([1., 1., 2., 2.]).view());
        builder.add_operator(
            node("Resize", &["input", "resize_roi", "resize_scales"])
                .with_attr("mode", "nearest".to_string())
                .with_attr("nearest_mode", "round_prefer_floor".to_string())
                .with_attr("coordinate_transformation_mode", "half_pixel".to_string()),
        );

        builder.add_operator(node("Round", &["input"]));

        builder.add_constant("upsample_scales", Tensor::from([1., 1., 2., 2.]).view());
        builder.add_operator(
            node("Upsample", &["input", "upsample_scales"])
                .with_attr("mode", "nearest".to_string()),
        );

        builder.add_operator(
            node("Shape", &["input"])
                .with_attr("start", 1i64)
                .with_attr("end", -1i64),
        );
        builder.add_operator(node("Sigmoid", &["input"]));
        builder.add_operator(node("Sign", &["input"]));
        builder.add_operator(node("Sin", &["input"]));
        builder.add_operator(node("Sinh", &["input"]));
        builder.add_operator(node("Size", &["input"]));

        builder.add_constant(
            "scatter_elem_indices",
            Tensor::<i32>::zeros(&input_shape).view(),
        );
        builder.add_constant(
            "scatter_elem_updates",
            Tensor::<f32>::zeros(&input_shape).view(),
        );
        builder.add_operator(
            node(
                "ScatterElements",
                &["input", "scatter_elem_indices", "scatter_elem_updates"],
            )
            .with_attr("axis", 0i64),
        );
        builder.add_operator(
            node(
                "Scatter",
                &["input", "scatter_elem_indices", "scatter_elem_updates"],
            )
            .with_attr("axis", 0i64),
        );

        // The standard 4D input has shape [batch=1, num_heads=1, seq=3,
        // head_size=3]. `rotary_embedding_dim` must be even and `<= head_size`,
        // so rotate the first 2 of the 3 head elements. The cos/sin caches have
        // shape [max_pos, rotary_embedding_dim / 2] and are gathered by
        // `position_ids`.
        builder.add_constant("rotary_cos", Tensor::<f32>::zeros(&[3, 1]).view());
        builder.add_constant("rotary_sin", Tensor::<f32>::zeros(&[3, 1]).view());
        builder.add_constant("rotary_pos", Tensor::from([[0i32, 1, 2]]).view());
        builder.add_operator(
            node(
                "RotaryEmbedding",
                &["input", "rotary_cos", "rotary_sin", "rotary_pos"],
            )
            .with_attr("interleaved", false)
            .with_attr("num_heads", 1i64)
            .with_attr("rotary_embedding_dim", 2i64),
        );

        builder.add_constant("const_0", Tensor::from([0]).view());
        builder.add_constant("const_1", Tensor::from([1]).view());
        builder.add_operator(node("Slice", &["input", "const_0", "const_1", "const_0"]));

        builder.add_operator(node("Softplus", &["input"]));
        builder.add_operator(node("Softmax", &["input"]).with_attr("axis", 1i64));
        builder.add_operator(node("Sqrt", &["input"]));
        builder.add_operator(node("Squeeze", &["input"]));

        builder.add_constant("split_splits", Tensor::from([1, 2]).view());
        builder.add_operator_with_outputs(
            node("Split", &["input.2d", "split_splits"]).with_attr("axis", 1i64),
            &["Split_out_1", "Split_out_2"],
        );

        builder.add_operator(node("Sub", &["input", "input"]));
        builder.add_operator(node("Sum", &["input", "input"]));
        builder.add_operator(node("Tan", &["input"]));
        builder.add_operator(node("Tanh", &["input"]));

        builder.add_constant("tile_repeats", Tensor::from([1, 2, 3, 4]).view());
        builder.add_operator(node("Tile", &["input", "tile_repeats"]));

        builder.add_constant("topk_k", Tensor::from(3).view());
        builder.add_operator_with_outputs(
            node("TopK", &["input.2d", "topk_k"])
                .with_attr("axis", -1i64)
                .with_attr("largest", true)
                .with_attr("sorted", true),
            &["TopK_out_values", "TopK_out_indices"],
        );

        builder.add_operator(node("Transpose", &["input"]));

        builder.add_operator(node("Trilu", &["input"]).with_attr("upper", true));

        builder.add_constant("unsqueeze_axes", Tensor::from([0, 4]).view());
        builder.add_operator(node("Unsqueeze", &["input", "unsqueeze_axes"]));

        for input in ["where_cond", "where_x", "where_y"] {
            builder.add_input(input);
        }
        let where_out = builder.add_operator(node("Where", &["where_cond", "where_x", "where_y"]));

        builder.add_operator(node("Xor", &["input.bool", "input.bool"]));

        let GraphBuilder { graph, op_outputs } = builder;
        let buffer = graph.into_model().with_opset("", 18).write_buf().unwrap();

        let model = Model::load(buffer).unwrap();

        let node_id = |name: &str| model.find_node(name).unwrap();
        let input_node = node_id("input");
        let input_2d = node_id("input.2d");
        let input_bool = node_id("input.bool");
        let input_u8 = node_id("input.u8");
        let input_2d_u8 = node_id("input.2d.u8");
        let input_2d_i8 = node_id("input.2d.i8");

        // Most ops are tested with one of several standard inputs:
        //
        //  - 4D float tensor (like an NCHW image)
        //  - Int8 NCHW tensor
        //  - Bool-ish int tensor
        //
        // A few require different shapes are tested separately.
        let input = Tensor::from_data(&input_shape, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input_2d_data = NdTensor::from([[1, 2, 3], [4, 5, 6]]);
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
                "Multinomial_out",
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
                        (input_2d, input_2d_data.view().into()),
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
                        (input_2d, input_2d_data.clone().into()),
                        (input_2d_u8, input_2d_u8_data.view().into()),
                        (input_2d_i8, input_2d_i8_data.view().into()),
                    ],
                    &[output_id],
                    None,
                )
                .unwrap();
            assert_eq!(result.len(), 1);
        }

        // Outputs of ops which require special handling.
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
            outputs.extend(["Dropout_out", "Dropout_out_mask", "Multinomial_out"]);
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
                    (node_id("range_start"), start.into()),
                    (node_id("range_limit"), limit.into()),
                    (node_id("range_delta"), delta.into()),
                ],
                &[node_id(&range_out)],
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
                    (node_id("where_cond"), cond.into()),
                    (node_id("where_x"), x.into()),
                    (node_id("where_y"), y.into()),
                ],
                &[node_id(&where_out)],
                None,
            )
            .unwrap();
        assert_eq!(result.len(), 1);
    }
}
