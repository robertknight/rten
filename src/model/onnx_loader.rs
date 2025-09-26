use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rten_base::byte_cast::{Pod, cast_pod_arc, cast_uninit_pod_mut_slice};
use rten_onnx::DecodeMessage;
use rten_onnx::onnx;
use rten_onnx::onnx::{GraphProto, ModelProto};
use rten_tensor::{ArcTensor, AssumeInit};

use super::NodeError;
use super::{Model, ModelLoadError, ModelOptions, OptimizeMode};
use crate::graph::{CaptureEnv, Constant, Dimension, Graph, NodeId};
use crate::model_metadata::ModelMetadata;
use crate::op_registry::onnx_registry::OpLoadContext;
use crate::op_registry::{OpRegistry, ReadOpError};
use crate::optimize::{GraphOptimizer, OptimizeOptions};
use crate::value::DataType;
use crate::weight_cache::WeightCache;

/// Load an RTen model from a serialized ONNX model.
///
/// An ONNX model is the serialized `ModelProto` Protocol Buffers message
/// defined in https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3.
pub fn load(
    data: &[u8],
    options: &ModelOptions,
    data_loader: &dyn ExternalDataLoader,
) -> Result<Model, ModelLoadError> {
    let model =
        ModelProto::decode(data).map_err(|err| ModelLoadError::ParseFailed(Box::new(err)))?;

    let optimize_opts = if options.optimize {
        OptimizeMode::On(OptimizeOptions::default())
    } else {
        OptimizeMode::Off
    };

    let graph = if let Some(onnx_graph) = model.graph {
        load_graph(
            &onnx_graph,
            &options.registry,
            optimize_opts,
            None,
            data_loader,
        )?
    } else {
        Graph::new()
    };

    let weight_cache = WeightCache::new();
    let metadata = ModelMetadata::default();
    Ok(Model {
        graph,
        weight_cache,
        metadata,
    })
}

/// Create a [`ModelLoadError`] that relates to a specific graph node.
macro_rules! load_error {
    ($kind:ident, $node_name:expr, $format_str:literal, $($arg:tt)*) => {{
        let err = format!($format_str, $($arg)*);
        ModelLoadError::$kind(NodeError::for_node($node_name, err).into())
    }};

    ($kind:ident, $node_name:expr, $err:expr) => {{
        ModelLoadError::$kind(NodeError::for_node($node_name, $err).into())
    }}
}

fn load_graph(
    onnx_graph: &GraphProto,
    registry: &OpRegistry,
    optimize: OptimizeMode,
    capture_env: Option<&CaptureEnv>,
    data_loader: &dyn ExternalDataLoader,
) -> Result<Graph, ModelLoadError> {
    let approx_node_count = onnx_graph.node.len() + onnx_graph.value_info.len();
    let mut graph = Graph::with_capacity(approx_node_count);

    // Create value nodes corresponding to `ValueInfoProto`s in the ONNX graph.
    for value in onnx_graph
        .value_info
        .iter()
        .chain(&onnx_graph.input)
        .chain(&onnx_graph.output)
    {
        let Some(name) = value.name else {
            continue;
        };
        let (dtype, shape) = load_value_info(value);
        graph.add_value(Some(name), shape, dtype);
    }

    let mut capture_ids = Vec::new();

    // Add constants from initializers.
    for initializer in &onnx_graph.initializer {
        let constant = load_constant(initializer, None, data_loader)?;
        graph.add_constant_node(constant);
    }

    // Add constants from "Constant" operators in the graph.
    for const_op in onnx_graph
        .node
        .iter()
        .filter(|op| op.op_type == Some("Constant"))
    {
        let constant = load_constant_from_constant_op(const_op, data_loader)?;
        graph.add_constant_node(constant);
    }

    // Create value nodes for operator inputs and outputs.
    for op in &onnx_graph.node {
        if op.op_type == Some("Constant") {
            // Constant operators are added to the graph as constants rather
            // than operators.
            continue;
        }
        for name in op.input.iter().chain(&op.output) {
            if name.is_empty() {
                // Empty names represent unused optional inputs or outputs.
                continue;
            }
            if graph.get_node_id(name).is_none() {
                let value_id = graph.add_value(Some(name), None, None);

                // If no value with this name was present in this graph, but
                // is available in a parent, mark it as a capture.
                //
                // FIXME - This relies on `onnx_graph.node` being sorted in
                // toplogical order so that if the value is available from the
                // current graph, it will be present in `graph` at this point.
                if let Some(capture_env) = capture_env
                    && capture_env.get_node(name).is_some()
                {
                    capture_ids.push(value_id);
                }
            }
        }
    }

    // Record which of the value nodes represent values coming from a parent graph.
    graph.set_captures(&capture_ids);

    // Set graph inputs and outputs.
    //
    // Value nodes should exist in the graph for all inputs and outputs at
    // this point.
    let input_ids: Vec<NodeId> = onnx_graph
        .input
        .iter()
        .filter_map(|value| value.name)
        .map(|name| {
            graph
                .get_node_id(name)
                .expect("input node should exist in graph")
        })
        .collect();
    graph.set_input_ids(&input_ids);

    let output_ids: Vec<NodeId> = onnx_graph
        .output
        .iter()
        .filter_map(|value| value.name)
        .map(|name| {
            graph
                .get_node_id(name)
                .expect("output node should exist in graph")
        })
        .collect();
    graph.set_output_ids(&output_ids);

    // Add model operators
    for onnx_op in &onnx_graph.node {
        if onnx_op.op_type == Some("Constant") {
            // Constant operators are added to the graph as constants rather
            // than operators.
            continue;
        }
        add_operator(
            &mut graph,
            onnx_op,
            registry,
            SubgraphOptions {
                optimize: optimize.clone(),
                capture_env,
                data_loader,
            },
        )?;
    }

    if let OptimizeMode::On(opts) = optimize {
        let optimizer = GraphOptimizer::new();
        optimizer
            .optimize(graph, capture_env, opts)
            .map_err(|err| ModelLoadError::OptimizeError(Box::new(err)))
    } else {
        Ok(graph)
    }
}

/// Convert data type and shape information from an ONNX value to RTen's
/// types.
fn load_value_info(value: &onnx::ValueInfoProto) -> (Option<DataType>, Option<Vec<Dimension>>) {
    let mut dtype = None;
    let mut shape = None;

    if let Some(type_info) = &value.r#type
        && let Some(tensor_type) = &type_info.tensor_type
    {
        if let Some(elem_type) = &tensor_type.elem_type {
            match *elem_type {
                onnx::DataType::FLOAT => dtype = Some(DataType::Float),
                onnx::DataType::INT64 | onnx::DataType::BOOL => dtype = Some(DataType::Int32),
                _ => {}
            }
        }
        if let Some(onnx_shape) = &tensor_type.shape {
            let mut dims = Vec::with_capacity(onnx_shape.dim.len());
            for dim in &onnx_shape.dim {
                if let Some(value) = dim.dim_value
                    && let Ok(size) = value.try_into()
                {
                    dims.push(Dimension::Fixed(size));
                } else if let Some(name) = dim.dim_param {
                    dims.push(Dimension::Symbolic(name.to_string()))
                }
            }
            shape = Some(dims)
        }
    }

    (dtype, shape)
}

/// Load data from an ONNX tensor.
///
/// Unlike when loading data from a .rten file, the data must be copied since
/// it may not have the required alignment.
fn load_constant(
    initializer: &onnx::TensorProto,
    name: Option<&str>,
    external_loader: &dyn ExternalDataLoader,
) -> Result<Constant, ModelLoadError> {
    let name = name.or(initializer.name);

    let shape: Result<Vec<usize>, _> = initializer.dims.iter().map(|&dim| dim.try_into()).collect();
    let shape =
        shape.map_err(|_| load_error!(GraphError, name, "initializer has invalid shape"))?;

    // Check if this tensor data is stored in the .onnx file or an external file.
    let data_location = initializer
        .data_location
        .unwrap_or(onnx::DataLocation::DEFAULT);
    let external_location = match data_location {
        onnx::DataLocation::DEFAULT => None,
        onnx::DataLocation::EXTERNAL => {
            Some(external_data_location(name, &initializer.external_data)?)
        }
        _ => {
            return Err(load_error!(GraphError, name, "unsupported data location"));
        }
    };

    // TensorProto can store data in one of several fields. Most use the
    // `raw_data` field.
    let raw_data = initializer.raw_data;

    let constant: Constant = match initializer.data_type {
        Some(onnx::DataType::FLOAT) => {
            let data = if let Some(data) = raw_data {
                load_raw_data(data, f32::from_le_bytes)
            } else if let Some(loc) = external_location {
                let u32s = external_loader.load_u32(loc)?;
                cast_pod_arc(u32s).unwrap()
            } else {
                initializer.float_data.as_slice().into()
            };
            let tensor = load_tensor(&shape, data, name)?;
            Constant::new(name, tensor)
        }
        Some(onnx::DataType::INT32) => {
            let data = if let Some(data) = raw_data {
                load_raw_data(data, i32::from_le_bytes)
            } else if let Some(loc) = external_location {
                let u32s = external_loader.load_u32(loc)?;
                cast_pod_arc(u32s).unwrap()
            } else {
                initializer.int32_data.as_slice().into()
            };
            let tensor = load_tensor(&shape, data, name)?;
            Constant::new(name, tensor)
        }
        Some(onnx::DataType::UINT8) => {
            let data = if let Some(data) = raw_data {
                data.into()
            } else if let Some(loc) = external_location {
                external_loader.load_u8(loc)?
            } else {
                initializer.int32_data.iter().map(|x| *x as u8).collect()
            };
            let tensor = load_tensor(&shape, data, name)?;
            Constant::new(name, tensor)
        }
        Some(onnx::DataType::INT8) => {
            let data = if let Some(data) = raw_data {
                let u8_to_i8 = |bytes: [u8; 1]| bytes[0] as i8;
                load_raw_data(data, u8_to_i8)
            } else if let Some(loc) = external_location {
                let u8s = external_loader.load_u8(loc)?;
                cast_pod_arc(u8s).unwrap()
            } else {
                initializer.int32_data.iter().map(|x| *x as i8).collect()
            };
            let tensor = load_tensor(&shape, data, name)?;
            Constant::new(name, tensor)
        }

        // RTen internally does not support i64 or bool tensors. Instead both
        // are converted to i32 at load time.
        Some(onnx::DataType::INT64) => {
            let data = if let Some(data) = raw_data {
                let i64_to_i32 =
                    |bytes: [u8; 8]| saturating_cast_i64_to_i32(i64::from_le_bytes(bytes));
                load_raw_data(data, i64_to_i32)
            } else if let Some(loc) = external_location {
                let u64s = external_loader.load_u64(loc)?;
                cast_pod_arc(u64s).unwrap()
            } else {
                initializer
                    .int64_data
                    .iter()
                    .copied()
                    .map(saturating_cast_i64_to_i32)
                    .collect()
            };
            let tensor = load_tensor(&shape, data, name)?;
            Constant::new(name, tensor)
        }
        Some(onnx::DataType::BOOL) => {
            let u8_to_i32 = |bytes: [u8; 1]| if bytes[0] != 0 { 1 } else { 0 };
            let data = if let Some(data) = raw_data {
                load_raw_data(data, u8_to_i32)
            } else if let Some(loc) = external_location {
                let u8s = external_loader.load_u8(loc)?;
                u8s.as_chunks().0.iter().copied().map(u8_to_i32).collect()
            } else {
                initializer
                    .int32_data
                    .iter()
                    .map(|x| if *x != 0 { 1 } else { 0 })
                    .collect()
            };
            let tensor = load_tensor(&shape, data, name)?;
            Constant::new(name, tensor)
        }

        Some(dtype) => {
            return Err(load_error!(
                GraphError,
                name,
                "initializer has unsupported data type {}",
                dtype.0
            ));
        }
        None => {
            return Err(load_error!(
                GraphError,
                name,
                "initializer is missing data type"
            ));
        }
    };

    Ok(constant)
}

/// Specifies the location of tensor data which is stored externally.
///
/// See https://github.com/onnx/onnx/blob/d2813e19cd9f0394f4b66fb392f0a09f231af77f/onnx/onnx.proto3#L743.
pub struct ExternalDataLocation<'a> {
    /// Path relative to the directory containing the ONNX protobuf model.
    path: &'a str,

    /// Offset of the start of the tensor data in the file.
    offset: u64,

    /// Length of the tensor data in bytes.
    length: u64,
}

/// Errors reading tensor data from an external file.
#[derive(Debug)]
pub enum ExternalDataError {
    /// An IO error occurred when accessing the external file.
    IoError(std::io::Error),

    /// The length of the external data is not a multiple of the element size
    /// (eg. if reading i32 elements, the length must be a multiple of 4).
    InvalidLength,

    /// External data is not supported in the current environment.
    NotSupported,
}

impl std::fmt::Display for ExternalDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(err) => write!(f, "io error: {}", err),
            Self::InvalidLength => write!(f, "length is not a multiple of element size"),
            Self::NotSupported => {
                write!(f, "external data not supported on this system")
            }
        }
    }
}

impl std::error::Error for ExternalDataError {}

impl From<ExternalDataError> for ModelLoadError {
    fn from(err: ExternalDataError) -> ModelLoadError {
        ModelLoadError::ExternalDataError(Box::new(err))
    }
}

/// Trait for loading data from an external file.
pub trait ExternalDataLoader {
    /// Load 8-bit elements from an external file.
    fn load_u8(&self, location: ExternalDataLocation) -> Result<Arc<[u8]>, ExternalDataError>;

    /// Load 32-bit elements from an external file.
    fn load_u32(&self, location: ExternalDataLocation) -> Result<Arc<[u32]>, ExternalDataError>;

    /// Load 64-bit elements from an external file.
    fn load_u64(&self, location: ExternalDataLocation) -> Result<Arc<[u64]>, ExternalDataError>;
}

/// External file loader that uses standard file IO.
pub struct ExternalFileLoader {
    /// Path to the directory external data files.
    base_path: PathBuf,

    /// Map of external data file name to open file.
    files: RefCell<HashMap<String, File>>,
}

impl ExternalFileLoader {
    pub fn new(model_path: &Path) -> Self {
        let base_path = model_path.parent().expect("should have a parent dir");
        Self {
            base_path: base_path.to_path_buf(),
            files: HashMap::new().into(),
        }
    }

    /// Read a chunk of a file specified by `location` into `buf`.
    ///
    /// The path specified by `location.path` is resolved relative to this
    /// loader's base path.
    ///
    /// The size of the file chunk specified by `location.length` must match
    /// the length of `buf`.
    fn read<T: Pod>(
        &self,
        location: ExternalDataLocation,
        mut buf: Arc<[MaybeUninit<T>]>,
    ) -> Result<Arc<[T]>, ExternalDataError> {
        // On a big-endian system we'd need to transmute bytes after reading
        // if we're reading multi-byte elements.
        if cfg!(target_endian = "big") {
            return Err(ExternalDataError::NotSupported);
        }

        let mut files = self.files.borrow_mut();
        let file = get_or_open_file(&mut files, &self.base_path, location.path)?;
        file.seek(SeekFrom::Start(location.offset))
            .map_err(ExternalDataError::IoError)?;
        let dst_slice = Arc::get_mut(&mut buf).unwrap();
        let dst_bytes = cast_uninit_pod_mut_slice::<T, u8>(dst_slice).unwrap();

        // FIXME: This should use `read_buf` to fill the buffer when that is
        // stabilized. Passing an uninitialized slice to `read_exact` is UB. We
        // are relying on the impl for `File` to follow the recommendations for
        // `read_exact` that impls should only write to the slice and not read
        // from it.
        file.read_exact(unsafe { dst_bytes.assume_init() })
            .map_err(ExternalDataError::IoError)?;

        // FIXME: We assume here that `read_exact` initialized the buffer,
        // although the contract does not guarantee this.
        Ok(unsafe { buf.assume_init() })
    }
}

impl ExternalDataLoader for ExternalFileLoader {
    fn load_u8(&self, location: ExternalDataLocation) -> Result<Arc<[u8]>, ExternalDataError> {
        let uninit_data = Arc::<[u8]>::new_uninit_slice(location.length as usize);
        let data = self.read(location, uninit_data)?;
        Ok(cast_pod_arc(data).unwrap())
    }

    fn load_u32(&self, location: ExternalDataLocation) -> Result<Arc<[u32]>, ExternalDataError> {
        let Some(len_u32) = location.length.checked_div(size_of::<u32>() as u64) else {
            return Err(ExternalDataError::InvalidLength);
        };
        let uninit_data = Arc::<[u32]>::new_uninit_slice(len_u32 as usize);
        let data = self.read(location, uninit_data)?;
        Ok(cast_pod_arc(data).unwrap())
    }

    fn load_u64(&self, location: ExternalDataLocation) -> Result<Arc<[u64]>, ExternalDataError> {
        let Some(len_u64) = location.length.checked_div(size_of::<u64>() as u64) else {
            return Err(ExternalDataError::InvalidLength);
        };
        let uninit_data = Arc::<[u64]>::new_uninit_slice(len_u64 as usize);
        let data = self.read(location, uninit_data)?;
        Ok(cast_pod_arc(data).unwrap())
    }
}

/// Get a handle to the open file named `path` in `files` or open the file for reading.
fn get_or_open_file<'a>(
    files: &'a mut HashMap<String, File>,
    base_path: &Path,
    path: &str,
) -> Result<&'a mut File, ExternalDataError> {
    if files.get(path).is_none() {
        // TODO - Disallow file paths that don't contain an `onnx_data` extension
        // or contain a directory path.
        let mut file_path = base_path.to_path_buf();
        file_path.push(path);
        let file = File::open(file_path).map_err(ExternalDataError::IoError)?;
        files.insert(path.to_string(), file);
    }
    Ok(files.get_mut(path).unwrap())
}

/// Parse the external location metadata from a `TensorProto.external_data` field.
fn external_data_location<'a>(
    name: Option<&str>,
    metadata: &[onnx::StringStringEntryProto<'a>],
) -> Result<ExternalDataLocation<'a>, ModelLoadError> {
    let mut location = None;
    let mut offset = None;
    let mut length = None;

    for metadata in metadata {
        let key = metadata.key.unwrap_or_default();
        let value = metadata.value.unwrap_or_default();

        match key {
            "location" => location = Some(value),
            "offset" => {
                offset =
                    Some(value.parse::<u64>().map_err(|_| {
                        load_error!(GraphError, name, "invalid external data offset")
                    })?);
            }
            "length" => {
                length =
                    Some(value.parse::<u64>().map_err(|_| {
                        load_error!(GraphError, name, "invalid external data length")
                    })?);
            }
            "checksum" => {}
            _ => {
                return Err(load_error!(
                    GraphError,
                    name,
                    "unsupported external data key {}",
                    key
                ));
            }
        }
    }

    let location =
        location.ok_or_else(|| load_error!(GraphError, name, "missing external data location"))?;
    let offset =
        offset.ok_or_else(|| load_error!(GraphError, name, "missing external data offset"))?;
    let length =
        length.ok_or_else(|| load_error!(GraphError, name, "missing external data length"))?;

    Ok(ExternalDataLocation {
        path: location,
        offset,
        length,
    })
}

/// Convert `x` to i32 with saturation.
///
/// RTen internally does not support i64 values so we convert to i32. We use a
/// saturating cast because there is a convention in ONNX models to use values
/// like `i64::{MIN, MAX}` to represent slicing to the end of a dimension in
/// Slice ops. This is handled by converting to `i32::{MIN, MAX}`.
fn saturating_cast_i64_to_i32(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Load a tensor from a "Constant" operator.
fn load_constant_from_constant_op(
    op: &onnx::NodeProto,
    data_loader: &dyn ExternalDataLoader,
) -> Result<Constant, ModelLoadError> {
    // The name of the constant node will be the name of its single output,
    // as that is the name that will be referenced by operator inputs.
    let [output] = &op.output[..] else {
        return Err(load_error!(OperatorInvalid, op.name, "missing output"));
    };
    let const_name = Some(*output);

    // Get constant value from attributes. The spec requires that exactly one
    // value attribute must be set.
    let mut constant = None;
    for attr in op.attribute.iter() {
        let Some(attr_name) = attr.name else {
            continue;
        };

        let attr_constant = match attr_name {
            "value" => {
                let Some(value) = &attr.t else {
                    return Err(load_error!(
                        OperatorInvalid,
                        op.name,
                        "invalid \"value\" attribute"
                    ));
                };
                load_constant(value, const_name, data_loader)?
            }
            "value_int" => {
                let value = attr.i.unwrap_or_default();
                let data = Arc::from([saturating_cast_i64_to_i32(value)]);
                let tensor = ArcTensor::from_data(&[], data);
                Constant::new(const_name, tensor)
            }
            "value_ints" => {
                let i32s: Arc<_> = attr
                    .ints
                    .iter()
                    .copied()
                    .map(saturating_cast_i64_to_i32)
                    .collect();
                let tensor = ArcTensor::from_data(&[i32s.len()], i32s);
                Constant::new(const_name, tensor)
            }
            "value_float" => {
                let data = Arc::from([attr.f.unwrap_or_default()]);
                let tensor = ArcTensor::from_data(&[], data);
                Constant::new(const_name, tensor)
            }
            "value_floats" => {
                let data = Arc::from(attr.floats.as_slice());
                let tensor = ArcTensor::from_data(&[attr.floats.len()], data);
                Constant::new(const_name, tensor)
            }
            _ => {
                // Known unsupported attributes: sparse_tensor, value_string,
                // value_strings.
                return Err(load_error!(
                    OperatorInvalid,
                    op.name,
                    "unsupported attribute {}",
                    attr_name
                ));
            }
        };

        if constant.is_some() {
            return Err(load_error!(
                OperatorInvalid,
                op.name,
                "multiple value attributes set"
            ));
        }
        constant = Some(attr_constant);
    }

    constant.ok_or_else(|| load_error!(OperatorInvalid, op.name, "value attribute not found"))
}

fn load_tensor<T>(
    shape: &[usize],
    data: Arc<[T]>,
    name: Option<&str>,
) -> Result<ArcTensor<T>, ModelLoadError> {
    let data_len = data.len();
    ArcTensor::try_from_data(shape, data).map_err(|_| {
        load_error!(
            GraphError,
            name,
            "length {} does not match shape {:?}",
            data_len,
            shape
        )
    })
}

fn load_raw_data<T, const SIZE_OF_T: usize>(
    data: &[u8],
    convert: impl Fn([u8; SIZE_OF_T]) -> T,
) -> Arc<[T]> {
    // TODO - Evaluate whether an explicit memcpy provides a performance benefit
    // for cases where the data is stored in the correct format except for
    // alignment.
    //
    // The code below already compiles to an efficient loop if `convert` is a
    // no-op (eg. `f32::from_le_bytes` on a little-endian system), but given
    // the size of tensor data, the system memcpy might be faster.
    data.as_chunks::<SIZE_OF_T>()
        .0
        .iter()
        .copied()
        .map(convert)
        .collect()
}

/// Configuration for loading subgraphs.
struct SubgraphOptions<'a> {
    /// Configuration for graph optimizer.
    optimize: OptimizeMode,

    /// Provides access to info about nodes captured from parent graphs.
    /// This is needed for some optimization passes.
    capture_env: Option<&'a CaptureEnv<'a>>,

    data_loader: &'a dyn ExternalDataLoader,
}

/// Load an ONNX operator and its subgraphs.
///
/// Value nodes must have been created in the graph for the operator's inputs
/// and outputs before this is called.
fn add_operator(
    graph: &mut Graph,
    onnx_op: &onnx::NodeProto,
    registry: &OpRegistry,
    subgraph_opts: SubgraphOptions,
) -> Result<(), ModelLoadError> {
    let load_subgraph = |g: &onnx::GraphProto| -> Result<Graph, ModelLoadError> {
        let SubgraphOptions {
            optimize,
            capture_env,
            data_loader,
        } = &subgraph_opts;
        let capture_env = CaptureEnv::new(*capture_env, graph, None, None, None);
        load_graph(
            g,
            registry,
            optimize.clone(),
            Some(&capture_env),
            *data_loader,
        )
    };

    struct LoadContext<'a> {
        load_graph: &'a dyn Fn(&onnx::GraphProto) -> Result<Graph, ModelLoadError>,
    }

    impl OpLoadContext for LoadContext<'_> {
        fn load_graph(&self, graph: &onnx::GraphProto) -> Result<Graph, ReadOpError> {
            (self.load_graph)(graph).map_err(|err| ReadOpError::SubgraphError(err.into()))
        }
    }

    let ctx = LoadContext {
        load_graph: &load_subgraph,
    };

    let inputs: Vec<Option<NodeId>> = onnx_op
        .input
        .iter()
        .map(|name| {
            if name.is_empty() {
                None
            } else {
                // nb. We expect graph nodes to be created for all inputs
                // before this method is called.
                Some(graph.get_node_id(name).unwrap())
            }
        })
        .collect();
    let outputs: Vec<Option<NodeId>> = onnx_op
        .output
        .iter()
        .map(|name| {
            if name.is_empty() {
                None
            } else {
                // nb. We expect graph nodes to be created for all inputs
                // before this method is called.
                Some(graph.get_node_id(name).unwrap())
            }
        })
        .collect();

    let op = registry
        .onnx_registry()
        .read_op(onnx_op, &ctx)
        .map_err(|err| load_error!(OperatorInvalid, onnx_op.name, err))?;

    graph.add_op(onnx_op.name, op, &inputs, &outputs);

    Ok(())
}
