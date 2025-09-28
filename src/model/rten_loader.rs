//! Load a model from a .rten model file.

use std::collections::HashMap;
use std::sync::Arc;

use rten_base::byte_cast::Pod;
use rten_model_file::header::{Header, HeaderError};
use rten_model_file::schema as sg;
use rten_model_file::schema::root_as_model;
use rten_tensor::ArcTensor;

use super::{
    ConstantStorage, Model, ModelLoadError, ModelMetadata, ModelOptions, NodeError, OptimizeMode,
    OptimizeOptions, SubgraphOptions, cast_le_bytes, constant_data_from_storage_offset,
};
use crate::constant_storage::{ArcSlice, ArcTensorView};
use crate::graph::{CaptureEnv, ConstantNodeData, Dimension, Graph, NodeId};
use crate::op_registry::rten_registry::{OpLoadContext, convert_dtype};
use crate::op_registry::{OpRegistry, ReadOpError};
use crate::optimize::GraphOptimizer;
use crate::weight_cache::WeightCache;

/// Load a model from a .rten model file.
pub fn load(
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

    let model =
        root_as_model(model_data).map_err(|err| ModelLoadError::ParseFailed(Box::new(err)))?;

    if model.schema_version() != 1 {
        return Err(ModelLoadError::SchemaVersionUnsupported);
    }

    let optimize_opts = if options.optimize {
        OptimizeMode::On(OptimizeOptions::default())
    } else {
        OptimizeMode::Off
    };

    let tensor_data_offset = header.as_ref().map(|h| h.tensor_data_offset);
    let graph = load_graph(
        model.graph(),
        registry,
        storage.clone(),
        tensor_data_offset,
        optimize_opts,
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
    optimize: OptimizeMode,
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
                add_graph_operator(
                    &mut graph,
                    node.name(),
                    operator,
                    registry,
                    &node_id_from_index,
                    SubgraphOptions {
                        storage: storage.clone(),
                        tensor_data_offset,
                        optimize: optimize.clone(),
                        capture_env,
                    },
                )?
            } else if let Some(value) = node.data_as_value_node() {
                add_graph_value(&mut graph, node.name(), value)?
            } else if let Some(constant) = node.data_as_constant_node() {
                add_graph_constant(
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

    if let OptimizeMode::On(opts) = optimize {
        let optimizer = GraphOptimizer::new();
        optimizer
            .optimize(graph, capture_env, opts)
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
        load_graph(
            g,
            registry,
            storage.clone(),
            *tensor_data_offset,
            optimize.clone(),
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
    let op = registry
        .rten_registry()
        .read_op(&operator, &ctx)
        .map_err(|err| ModelLoadError::OperatorInvalid(NodeError::for_node(name, err).into()))?;

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
        .map_err(|err| ModelLoadError::OperatorInvalid(NodeError::for_node(name, err).into()))?;
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
                    NodeError::for_node(name, "unsupported data type for external constant").into(),
                ));
            }
        };
        Ok(graph_node)
    } else {
        // Constant data is stored inline in model
        let graph_node = if let Some(float_data) = constant.data_as_float_data() {
            let const_data = constant_data_from_flatbuffers_vec(storage, float_data.data(), &shape);
            graph.add_constant(name, const_data)
        } else if let Some(int_data) = constant.data_as_int_32_data() {
            let const_data = constant_data_from_flatbuffers_vec(storage, int_data.data(), &shape);
            graph.add_constant(name, const_data)
        } else if let Some(int8_data) = constant.data_as_int_8_data() {
            let const_data = constant_data_from_flatbuffers_vec(storage, int8_data.data(), &shape);
            graph.add_constant(name, const_data)
        } else if let Some(uint8_data) = constant.data_as_uint_8_data() {
            let const_data = constant_data_from_flatbuffers_vec(storage, uint8_data.data(), &shape);
            graph.add_constant(name, const_data)
        } else {
            return Err(ModelLoadError::GraphError(
                NodeError::for_node(name, "unsupported data type for inline constant").into(),
            ));
        };
        Ok(graph_node)
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
        let data: Vec<T> = fb_vec.iter().collect();
        ArcTensor::from_data(shape, Arc::new(data)).into()
    }
}
