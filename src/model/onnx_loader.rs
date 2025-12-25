use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use rten_base::byte_cast::{Pod, cast_pod_slice};
use rten_base::half::f16_to_f32;
use rten_onnx::onnx;
use rten_tensor::{ArcTensor, Storage, Tensor};

use super::external_data::{DataLoader, DataLocation, DataSlice};
use super::load_error::{LoadError, LoadErrorImpl, load_error};
use super::metadata::{MetadataField, ModelMetadata};
use super::{Model, ModelOptions, OptimizeMode};
use crate::constant_storage::{ArcSlice, ArcTensorView};
use crate::graph::{
    CaptureEnv, Constant, ConstantNode, ConstantNodeData, Dimension, Graph, NodeId,
};
use crate::op_registry::onnx_registry::{ConstInput, DynParsedOp, OpLoadContext};
use crate::op_registry::{OpRegistry, ReadOpError};
use crate::optimize::{GraphOptimizer, OptimizeOptions};
use crate::value::{DataType, ValueType};
use crate::weight_cache::WeightCache;

/// Specifies where to load an ONNX model from.
pub enum Source<'a> {
    Path(&'a Path),
    Buffer(&'a [u8]),
    #[cfg(test)]
    Proto(onnx::ModelProto),
}

/// Load a serialized ONNX model from a file or buffer.
///
/// An ONNX model is the serialized `ModelProto` Protocol Buffers message
/// defined in https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3.
pub fn load(
    source: Source,
    loader: Option<&dyn DataLoader>,
    options: &ModelOptions,
) -> Result<Model, LoadError> {
    let model = match source {
        Source::Path(path) => {
            let file = File::open(path).map_err(LoadErrorImpl::ReadFailed)?;
            onnx::ModelProto::parse_file(file)
        }
        Source::Buffer(buf) => onnx::ModelProto::parse_buf(buf),
        #[cfg(test)]
        Source::Proto(proto) => Ok(proto),
    }
    .map_err(|err| LoadErrorImpl::ParseFailed(Box::new(err)))?;

    let optimize_opts = if options.optimize {
        OptimizeMode::On(OptimizeOptions {
            infer_shapes: options.infer_shapes,
        })
    } else {
        OptimizeMode::Off
    };

    let graph = if let Some(onnx_graph) = &model.graph {
        load_graph(onnx_graph, &options.registry, optimize_opts, None, loader)?
    } else {
        Graph::new()
    };

    let mut weight_cache = WeightCache::new();
    if options.prepack_weights {
        graph.prepack_weights(&mut weight_cache);
    }

    let metadata = load_metadata(&model);

    Ok(Model {
        metadata,
        graph,
        weight_cache,
    })
}

fn load_metadata(model: &onnx::ModelProto) -> ModelMetadata {
    let mut fields = Vec::new();
    if let Some(name) = &model.producer_name {
        fields.push((MetadataField::ProducerName, name.clone()));
    }
    if let Some(version) = &model.producer_version {
        fields.push((MetadataField::ProducerVersion, version.clone()));
    }
    for prop in &model.metadata_props {
        let Some(key) = &prop.key else {
            continue;
        };
        let value = prop.value.as_deref().unwrap_or_default();
        fields.push((MetadataField::Custom(key.clone()), value.to_string()));
    }
    ModelMetadata::from_fields(fields)
}

fn load_graph(
    onnx_graph: &onnx::GraphProto,
    registry: &OpRegistry,
    optimize: OptimizeMode,
    capture_env: Option<&CaptureEnv>,
    loader: Option<&dyn DataLoader>,
) -> Result<Graph, LoadError> {
    let approx_node_count = onnx_graph.node.len() + onnx_graph.value_info.len();
    let mut graph = Graph::with_capacity(approx_node_count);
    let mut unnamed_count = 0;

    let mut add_value = |graph: &mut Graph, name, value| {
        let (dtype, shape) = load_value_info(value, &mut unnamed_count);
        graph.add_value(Some(name), shape, dtype.map(ValueType::Tensor))
    };

    // Create value nodes corresponding to graph inputs and outputs.
    for value in &onnx_graph.input {
        let name = value.name.as_deref().unwrap_or_default();
        if name.is_empty() {
            return Err(LoadErrorImpl::GraphError(
                "graph input has missing or invalid name".into(),
            )
            .into());
        }
        add_value(&mut graph, name, value);
    }

    for value in &onnx_graph.output {
        let name = value.name.as_deref().unwrap_or_default();
        if name.is_empty() {
            return Err(LoadErrorImpl::GraphError(
                "graph output has missing or invalid name".into(),
            )
            .into());
        }
        add_value(&mut graph, name, value);
    }

    // Create map of value name to dtype and shape metadata.
    //
    // We don't actually create value nodes until we see the value being used as
    // an operator input or output.
    let mut name_to_value_info = HashMap::with_capacity(onnx_graph.value_info.len());
    for value in &onnx_graph.value_info {
        let name = match value.name.as_deref() {
            Some(name) if !name.is_empty() => name,
            _ => {
                // The name is optional in the protobuf schema, but required
                // in current ONNX IR versions.
                //
                // We ignore values with missing names here on the basis that
                // missing names, except for inputs/outputs, don't prevent
                // inference from working. It might prevent some graph
                // optimizations from being applied though.
                continue;
            }
        };
        name_to_value_info.insert(name, value);
    }

    // Add constants from initializers.
    for initializer in &onnx_graph.initializer {
        let constant = load_constant(initializer, loader, None)?;
        graph.add_constant_node(constant);
    }

    // Add constants from "Constant" operators in the graph.
    for const_op in onnx_graph
        .node
        .iter()
        .filter(|op| op.op_type.as_deref() == Some("Constant"))
    {
        let constant = load_constant_from_constant_op(const_op, loader)?;
        graph.add_constant_node(constant);
    }

    // Create value nodes for operator inputs and outputs.
    let mut capture_ids = Vec::new();
    for op in &onnx_graph.node {
        if op.op_type.as_deref() == Some("Constant") {
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
                let value_id = if let Some(value_info) = name_to_value_info.get(name.as_str()) {
                    add_value(&mut graph, name, value_info)
                } else {
                    // Add node without dtype or shape metadata.
                    graph.add_value(Some(name), None, None)
                };

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

    let node_ids_from_value_info =
        |graph: &Graph, values: &[onnx::ValueInfoProto]| -> Vec<NodeId> {
            values
                .iter()
                .map(|val| {
                    let name = val.name.as_deref().unwrap_or_default();
                    graph
                        .get_node_id(name)
                        .expect("value node should exist in graph")
                })
                .collect()
        };

    // Set graph inputs and outputs.
    //
    // Value nodes should exist in the graph for all inputs and outputs at
    // this point.
    let input_ids = node_ids_from_value_info(&graph, &onnx_graph.input);
    graph.set_input_ids(&input_ids);

    let output_ids = node_ids_from_value_info(&graph, &onnx_graph.output);
    graph.set_output_ids(&output_ids);

    // Add model operators
    for onnx_op in &onnx_graph.node {
        if onnx_op.op_type.as_deref() == Some("Constant") {
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
                loader,
            },
        )?;
    }

    if let OptimizeMode::On(opts) = optimize {
        let optimizer = GraphOptimizer::new();
        optimizer
            .optimize(graph, capture_env, opts)
            .map_err(|err| LoadErrorImpl::OptimizeError(Box::new(err)).into())
    } else {
        Ok(graph)
    }
}

/// Convert data type and shape information from an ONNX value to RTen's
/// types.
///
/// If a value has a dimension with a missing name, `unnamed_count` is used
/// to generate a name for it.
fn load_value_info(
    value: &onnx::ValueInfoProto,
    unnamed_count: &mut u32,
) -> (Option<DataType>, Option<Vec<Dimension>>) {
    let Some(type_info) = &value.r#type else {
        return (None, None);
    };

    // `ValueInfoProto`s can represent tensors, sequences and other types.
    // Only tensor types are supported here.
    let Some(tensor_type) = &type_info.tensor_type else {
        return (None, None);
    };

    let mut dtype = None;
    let mut shape = None;

    if let Some(elem_type) = &tensor_type.elem_type {
        dtype = match *elem_type {
            onnx::DataType::FLOAT => Some(DataType::Float),
            onnx::DataType::INT32 => Some(DataType::Int32),
            onnx::DataType::INT8 => Some(DataType::Int8),
            onnx::DataType::UINT8 => Some(DataType::UInt8),

            // RTen doesn't internally support i64 or bool tensors but converts
            // them to i32 tensors instead. Adjust the value type here to match.
            //
            // This does mean that when querying metadata for an input via
            // `Model::node_info`, the caller may get a type that doesn't
            // match the ONNX model. It will however match the type that RTen
            // expects for that input.
            onnx::DataType::INT64 | onnx::DataType::BOOL => Some(DataType::Int32),

            // RTen doesn't internally support f16 or f64 but converts to f32
            // tensors instead. Adjust the value type here to match.
            onnx::DataType::DOUBLE | onnx::DataType::FLOAT16 => Some(DataType::Float),

            _ => None,
        };
    }
    if let Some(onnx_shape) = &tensor_type.shape {
        shape = Some(
            onnx_shape
                .dim
                .iter()
                .map(|dim| {
                    if let Some(value) = dim.dim_value
                        && let Ok(size) = value.try_into()
                    {
                        Dimension::Fixed(size)
                    } else if let Some(name) = &dim.dim_param {
                        Dimension::Symbolic(name.to_string())
                    } else {
                        *unnamed_count += 1;
                        Dimension::Symbolic(format!("unnamed_{}", unnamed_count))
                    }
                })
                .collect(),
        );
    }

    (dtype, shape)
}

/// Create a constant graph node from an ONNX tensor.
///
/// If `name` is provided, it overrides the name from `initializer.name`.
fn load_constant(
    initializer: &onnx::TensorProto,
    loader: Option<&dyn DataLoader>,
    name: Option<&str>,
) -> Result<Constant, LoadError> {
    let name = name.or(initializer.name.as_deref());

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

    let external_data = if let Some(loc) = external_location {
        if let Some(loader) = &loader {
            let slice = loader
                .load(&loc)
                .map_err(|e| load_error!(ExternalDataError, name, e))?;
            Some(slice)
        } else {
            return Err(load_error!(
                ExternalDataError,
                name,
                "tensor has external data but model was loaded without external data source"
            ));
        }
    } else {
        None
    };

    // Tensor data can be stored in the `raw_data` field, one of several typed
    // fields, or externally.
    //
    // When data is not stored externally, most tensors use the `raw_data`
    // field, especially for large tensors. To make models load as fast as
    // possible, it is important to minimize copying of weights. Hence if the
    // data is stored in `raw_data`, we take and use that buffer rather than
    // copy here. If the data is stored in one of the typed fields
    // (`float_data`), we assume it is smaller and that copying them won't have
    // a significant impact.
    let raw_data = initializer.raw_data.as_ref().map(|data| data.take());

    let constant: Constant = match initializer.data_type {
        Some(onnx::DataType::FLOAT) => make_constant(
            name,
            &shape,
            raw_data,
            external_data,
            &initializer.float_data,
            |x| x,
        )?,
        Some(onnx::DataType::INT32) => make_constant(
            name,
            &shape,
            raw_data,
            external_data,
            &initializer.int32_data,
            |x| x,
        )?,
        Some(onnx::DataType::UINT8) => make_constant(
            name,
            &shape,
            raw_data,
            external_data,
            &initializer.int32_data,
            |x| x as u8,
        )?,
        Some(onnx::DataType::INT8) => make_constant(
            name,
            &shape,
            raw_data,
            external_data,
            &initializer.int32_data,
            |x| x as i8,
        )?,

        // RTen internally does not support i64 or bool tensors. Instead both
        // are converted to i32 at load time.
        Some(onnx::DataType::INT64) => {
            let i64_to_i32 = |bytes: [u8; 8]| saturating_cast_i64_to_i32(i64::from_le_bytes(bytes));
            let data = if let Some(data) = raw_data {
                elements_from_le_bytes(&data, i64_to_i32)
            } else if let Some(external_data) = external_data {
                elements_from_le_bytes(external_data.data(), i64_to_i32)
            } else {
                initializer
                    .int64_data
                    .iter()
                    .copied()
                    .map(saturating_cast_i64_to_i32)
                    .collect()
            };
            let tensor = tensor_from_elements(&shape, data, name)?;
            Constant::new(name, tensor)
        }
        Some(onnx::DataType::BOOL) => {
            let u8_to_i32 = |bytes: [u8; 1]| if bytes[0] != 0 { 1 } else { 0 };
            let data = if let Some(data) = raw_data {
                elements_from_le_bytes(&data, u8_to_i32)
            } else if let Some(external_data) = external_data {
                elements_from_le_bytes(external_data.data(), u8_to_i32)
            } else {
                initializer
                    .int32_data
                    .iter()
                    .map(|x| if *x != 0 { 1 } else { 0 })
                    .collect()
            };
            let tensor = tensor_from_elements(&shape, data, name)?;
            Constant::new(name, tensor)
        }

        // RTen does not natively support f64 tensors. Instead convert to f32
        // at load time.
        Some(onnx::DataType::DOUBLE) => {
            let data = if let Some(data) = raw_data {
                let f64_to_f32 = |bytes: [u8; 8]| f64::from_le_bytes(bytes) as f32;
                elements_from_le_bytes(&data, f64_to_f32)
            } else {
                initializer
                    .double_data
                    .iter()
                    .copied()
                    .map(|x| x as f32)
                    .collect()
            };
            let tensor = tensor_from_elements(&shape, data, name)?;
            Constant::new(name, tensor)
        }

        Some(onnx::DataType::FLOAT16) => {
            let data = if let Some(data) = raw_data {
                let f16_to_f32_bytes = |bytes: [u8; 2]| f16_to_f32(u16::from_le_bytes(bytes));
                elements_from_le_bytes(&data, f16_to_f32_bytes)
            } else {
                initializer
                    .int32_data
                    .iter()
                    .copied()
                    .map(|x| f16_to_f32(x as u16))
                    .collect()
            };
            let tensor = tensor_from_elements(&shape, data, name)?;
            Constant::new(name, tensor)
        }

        Some(dtype) => {
            return Err(load_error!(
                GraphError,
                name,
                "initializer has unsupported data type {}",
                dtype
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

/// Parse the external location metadata from a `TensorProto.external_data` field.
fn external_data_location(
    name: Option<&str>,
    metadata: &[onnx::StringStringEntryProto],
) -> Result<DataLocation, LoadError> {
    let mut location = None;
    let mut offset = None;
    let mut length = None;

    for metadata in metadata {
        let Some(key) = &metadata.key else {
            continue;
        };
        let Some(value) = &metadata.value else {
            continue;
        };

        match key.as_str() {
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

    Ok(DataLocation {
        path: location.to_string(),
        offset,
        length,
    })
}

/// Create a constant with elements of type `T`.
///
/// The tensor will use `raw_data` without copying if provided, otherwise the
/// data in `typed_data` will be copied and converted.
fn make_constant<T: Pod, U: Pod>(
    name: Option<&str>,
    shape: &[usize],
    raw_data: Option<Vec<u8>>,
    external_data: Option<DataSlice>,
    typed_data: &[U],
    convert: impl Fn(U) -> T,
) -> Result<Constant, LoadError>
where
    Constant: From<ConstantNode<T>>,
{
    let tensor: ConstantNodeData<T> = if let Some(data) = raw_data {
        tensor_from_bytes::<T>(shape, data, name)?.into()
    } else if let Some(external_data) = external_data {
        tensor_from_external_data::<T>(shape, &external_data, name)?.into()
    } else {
        let data = typed_data.iter().copied().map(convert).collect();
        tensor_from_elements(shape, data, name)?.into()
    };
    Ok(Constant::new(name, tensor))
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
    loader: Option<&dyn DataLoader>,
) -> Result<Constant, LoadError> {
    // The name of the constant node will be the name of its single output,
    // as that is the name that will be referenced by operator inputs.
    let [output] = &op.output[..] else {
        return Err(load_error!(
            OperatorInvalid,
            op.name.as_deref(),
            "missing output"
        ));
    };
    let const_name = Some(output.as_str());

    // Get constant value from attributes. The spec requires that exactly one
    // value attribute must be set.
    let mut constant = None;
    for attr in op.attribute.iter() {
        let Some(attr_name) = &attr.name else {
            continue;
        };

        let attr_constant = match attr_name.as_str() {
            "value" => {
                let Some(value) = &attr.t else {
                    return Err(load_error!(
                        OperatorInvalid,
                        op.name.as_deref(),
                        "invalid \"value\" attribute"
                    ));
                };
                load_constant(value, loader, const_name)?
            }
            "value_int" => {
                let value = attr.i.unwrap_or_default();
                let data = Vec::from([saturating_cast_i64_to_i32(value)]);
                let tensor = Tensor::from_data(&[], data);
                Constant::new(const_name, tensor.into_arc())
            }
            "value_ints" => {
                let i32s: Vec<_> = attr
                    .ints
                    .iter()
                    .copied()
                    .map(saturating_cast_i64_to_i32)
                    .collect();
                let tensor = Tensor::from_data(&[i32s.len()], i32s);
                Constant::new(const_name, tensor.into_arc())
            }
            "value_float" => {
                let data = Vec::from([attr.f.unwrap_or_default()]);
                let tensor = Tensor::from_data(&[], data);
                Constant::new(const_name, tensor.into_arc())
            }
            "value_floats" => {
                let data = attr.floats.clone();
                let tensor = Tensor::from_data(&[attr.floats.len()], data);
                Constant::new(const_name, tensor.into_arc())
            }
            _ => {
                // Known unsupported attributes: sparse_tensor, value_string,
                // value_strings.
                return Err(load_error!(
                    OperatorInvalid,
                    op.name.as_deref(),
                    "unsupported attribute {}",
                    attr_name
                ));
            }
        };

        if constant.is_some() {
            return Err(load_error!(
                OperatorInvalid,
                op.name.as_deref(),
                "multiple value attributes set"
            ));
        }
        constant = Some(attr_constant);
    }

    constant.ok_or_else(|| {
        load_error!(
            OperatorInvalid,
            op.name.as_deref(),
            "value attribute not found"
        )
    })
}

fn constant_from_attr_value(val: ConstInput) -> Constant {
    match val {
        ConstInput::Ints(vals) => {
            let vals: Vec<i32> = vals.into_iter().map(saturating_cast_i64_to_i32).collect();
            Constant::new(None, Tensor::from(vals).into_arc())
        }
        ConstInput::Float(float) => Constant::new(None, Tensor::from(float).into_arc()),
    }
}

fn tensor_from_elements<T>(
    shape: &[usize],
    data: Vec<T>,
    name: Option<&str>,
) -> Result<ArcTensor<T>, LoadError> {
    let data_len = data.len();
    let tensor = Tensor::try_from_data(shape, data)
        .map_err(|_| {
            load_error!(
                GraphError,
                name,
                "length {} does not match shape {:?}",
                data_len,
                shape
            )
        })?
        .into_arc();
    Ok(tensor)
}

/// Create a tensor by reinterpreting the little-endian bytes in `data` as type T.
fn tensor_from_bytes<T: Pod>(
    shape: &[usize],
    data: Vec<u8>,
    name: Option<&str>,
) -> Result<ArcTensorView<T>, LoadError> {
    // To support big-endian systems, this function would need to byte-swap
    // `T`-sized chunks of `data`.
    if !cfg!(target_endian = "little") {
        return Err(load_error!(
            GraphError,
            name,
            "ONNX model loading not supported on big-endian systems"
        ));
    }

    // We assume here that the allocator of `data` will always ensure some
    // minimum alignment regardless of type, and that alignment will be
    // sufficient for all the types of tensor `T` that we want to create using
    // this method. If that ever turns out not to be the case, we'll need to
    // copy the bytes into a new suitably-aligned buffer.
    let data = ArcSlice::<T>::from_bytes(data)
        .ok_or_else(|| load_error!(GraphError, name, "data has incorrect alignment"))?;
    let data_len = data.len();
    ArcTensorView::try_from_data(shape, data).map_err(|_| {
        load_error!(
            GraphError,
            name,
            "length {} does not match shape {:?}",
            data_len,
            shape
        )
    })
}

/// Create a tensor by reinterpreting bytes that have been loaded or
/// memory-mapped from an external file.
fn tensor_from_external_data<T: Pod>(
    shape: &[usize],
    data: &DataSlice,
    name: Option<&str>,
) -> Result<ArcTensorView<T>, LoadError> {
    let data: ArcSlice<T> = if let Some(elements) = cast_pod_slice(data.data()) {
        ArcSlice::new(data.storage.clone(), elements).unwrap()
    } else if data.data().is_empty() {
        // If `data.storage`'s backing storage is a zero-length `Vec<u8>` it
        // might have smaller alignment than required. Use
        // `ArcSlice::from_bytes` which has special handling of empty inputs.
        ArcSlice::from_bytes(Vec::new()).unwrap()
    } else {
        return Err(load_error!(
            GraphError,
            name,
            "data has incorrect alignment"
        ));
    };

    let data_len = data.len();
    ArcTensorView::try_from_data(shape, data).map_err(|_| {
        load_error!(
            GraphError,
            name,
            "length {} does not match shape {:?}",
            data_len,
            shape
        )
    })
}

/// Create a `Vec<T>` from a slice of little-endian bytes.
///
/// `convert` is used to convert each chunk of bytes into an element. There
/// may be unused bytes if `data.len()` is not a multiple of `ELEM_SIZE`.
fn elements_from_le_bytes<T, const ELEM_SIZE: usize>(
    data: &[u8],
    convert: impl Fn([u8; ELEM_SIZE]) -> T,
) -> Vec<T> {
    data.as_chunks::<ELEM_SIZE>()
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

    /// Data source for tensors with data stored outside model.
    loader: Option<&'a dyn DataLoader>,
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
) -> Result<(), LoadError> {
    let load_subgraph = |g: &onnx::GraphProto| -> Result<Graph, LoadError> {
        let SubgraphOptions {
            optimize,
            capture_env,
            loader,
        } = &subgraph_opts;
        let capture_env = CaptureEnv::new(*capture_env, graph, None, None, None);
        load_graph(g, registry, optimize.clone(), Some(&capture_env), *loader)
    };

    struct LoadContext<'a> {
        load_graph: &'a dyn Fn(&onnx::GraphProto) -> Result<Graph, LoadError>,
    }

    impl OpLoadContext for LoadContext<'_> {
        fn load_graph(&self, graph: &onnx::GraphProto) -> Result<Graph, ReadOpError> {
            (self.load_graph)(graph).map_err(|err| ReadOpError::SubgraphError(err.into()))
        }
    }

    let ctx = LoadContext {
        load_graph: &load_subgraph,
    };

    let node_ids_from_names = |names: &[String]| -> Vec<Option<NodeId>> {
        names
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
            .collect()
    };

    let DynParsedOp {
        op,
        const_inputs,
        unused_attrs,
    } = registry
        .onnx_registry()
        .read_op(onnx_op, &ctx)
        .map_err(|err| load_error!(OperatorInvalid, onnx_op.name.as_deref(), err))?;

    // Fail if any attributes were unused.
    if !unused_attrs.is_empty() {
        let names: Vec<_> = unused_attrs
            .iter()
            .map(|i| onnx_op.attribute[i].name.as_deref().unwrap_or_default())
            .collect();

        return Err(load_error!(
            OperatorInvalid,
            onnx_op.name.as_deref(),
            "unsupported or duplicated attributes: {}",
            names.join(", ")
        ));
    }

    // Map input and output names to graph node IDs.
    //
    // If there are attributes that need to be promoted to inputs, then create
    // constants for the attribute values and add those inputs.
    let mut inputs = node_ids_from_names(&onnx_op.input);
    let outputs = node_ids_from_names(&onnx_op.output);
    for (idx, value) in const_inputs {
        let constant = constant_from_attr_value(value);
        let const_id = graph.add_constant_node(constant);

        let idx = idx as usize;
        if inputs.len() <= idx {
            inputs.resize(idx + 1, None);
        }
        if inputs[idx].is_some() {
            return Err(load_error!(
                OperatorInvalid,
                onnx_op.name.as_deref(),
                "input {} specified as both attribute and input",
                idx
            ));
        }
        inputs[idx] = Some(const_id);
    }

    if let Some(max) = op.max_inputs()
        && inputs.len() > max
    {
        return Err(load_error!(
            OperatorInvalid,
            onnx_op.name.as_deref(),
            "operator has {} inputs but maximum is {}",
            inputs.len(),
            max
        ));
    }

    graph.add_op(onnx_op.name.as_deref(), op, &inputs, &outputs);

    Ok(())
}

#[cfg(test)]
mod tests {
    use rten_base::half::{f16_to_f32, f32_to_f16};
    use rten_onnx::onnx;
    use rten_tensor::{Tensor, TensorView};

    use super::{Source, load};
    use crate::graph::{Constant, Dimension, Graph, TypedConstant};
    use crate::model::external_data::{DataLoader, DataLocation, MemLoader};
    use crate::model::onnx_builder::{
        GraphProtoExt, NodeProtoExt, TensorData, create_node, create_tensor, create_value_info,
    };
    use crate::model::{LoadError, Model, ModelOptions};

    /// Load a model from a parsed `ModelProto` message.
    fn load_model(
        model: onnx::ModelProto,
        data_loader: Option<&dyn DataLoader>,
    ) -> Result<Model, LoadError> {
        load(
            Source::Proto(model),
            data_loader,
            // Disable optimization by default to test just the basic graph
            // creation.
            &ModelOptions::with_all_ops().enable_optimization(false),
        )
    }

    trait GetTensorByName {
        fn get_tensor_by_name<T>(&self, name: &str) -> Option<TensorView<'_, T>>
        where
            Constant: TypedConstant<T>;
    }

    impl GetTensorByName for Graph {
        fn get_tensor_by_name<T>(&self, name: &str) -> Option<TensorView<'_, T>>
        where
            Constant: TypedConstant<T>,
        {
            let id = self.get_node_id(name)?;
            self.get_node(id)?.as_constant()?.as_typed_view()
        }
    }

    impl GetTensorByName for Model {
        fn get_tensor_by_name<T>(&self, name: &str) -> Option<TensorView<'_, T>>
        where
            Constant: TypedConstant<T>,
        {
            self.graph.get_tensor_by_name(name)
        }
    }

    #[test]
    fn test_graph_invalid_input_name() {
        let model = onnx::GraphProto::default()
            .with_input(onnx::ValueInfoProto::default())
            .into_model();

        let err = load(Source::Proto(model), None, &ModelOptions::default())
            .err()
            .unwrap();

        assert_eq!(
            err.to_string(),
            "graph error: graph input has missing or invalid name"
        );
    }

    #[test]
    fn test_graph_invalid_output_name() {
        let model = onnx::GraphProto::default()
            .with_output(onnx::ValueInfoProto::default())
            .into_model();

        let err = load(Source::Proto(model), None, &ModelOptions::default())
            .err()
            .unwrap();

        assert_eq!(
            err.to_string(),
            "graph error: graph output has missing or invalid name"
        );
    }

    #[test]
    fn test_sub_graph_capture() {
        // Create subgraph with a capture.
        let id_op = create_node("Identity").with_input("x");
        let then_branch = onnx::GraphProto::default()
            .with_value(create_value_info("x"))
            .with_node(id_op);

        let if_node = create_node("If")
            .with_attr("then_branch", then_branch)
            .with_attr("else_branch", onnx::GraphProto::default())
            .with_name("if_op");

        let model_proto = onnx::GraphProto::default()
            .with_input(create_value_info("x"))
            .with_node(if_node)
            .into_model();

        let model = load_model(model_proto, None).unwrap();

        // Verify the capture list for the subgraph was populated correctly.
        let graph = model.graph();
        let if_node_id = graph.get_node_id("if_op").unwrap();
        let if_node = graph
            .get_node(if_node_id)
            .and_then(|n| n.as_operator())
            .unwrap();
        let then_branch = if_node.operator().as_subgraph_op().unwrap().subgraphs()[0];
        let captures = then_branch.captures();
        assert_eq!(captures.len(), 1);
        assert_eq!(then_branch.node_name(captures[0]), "x");
    }

    #[test]
    fn test_promote_attribute_to_input() {
        let node = create_node("Clip")
            .with_attr("min", -0.5)
            .with_attr("max", 0.5)
            .with_name("clip_op");

        let model_proto = onnx::GraphProto::default()
            .with_input(create_value_info("x"))
            .with_node(node)
            .into_model();

        let model = load_model(model_proto, None).unwrap();

        let graph = model.graph();
        let clip_op_id = graph.get_node_id("clip_op").unwrap();
        let clip_op = graph
            .get_node(clip_op_id)
            .and_then(|n| n.as_operator())
            .unwrap();
        assert_eq!(clip_op.input_ids().len(), 3);

        let min_val_id = clip_op.input_ids()[1].unwrap();
        let min_val: f32 = graph
            .get_node(min_val_id)
            .and_then(|n| n.as_constant())
            .and_then(|c| c.as_scalar())
            .unwrap();
        assert_eq!(min_val, -0.5);

        let max_val_id = clip_op.input_ids()[2].unwrap();
        let max_val: f32 = graph
            .get_node(max_val_id)
            .and_then(|n| n.as_constant())
            .and_then(|c| c.as_scalar())
            .unwrap();
        assert_eq!(max_val, 0.5);
    }

    #[test]
    fn test_load_f64_initializer() {
        // TensorProto using the `raw_data` field.
        let doubles_raw = create_tensor(
            "doubles_raw",
            &[],
            onnx::DataType::DOUBLE,
            TensorData::Raw((0.5f64).to_le_bytes().into()),
        );

        // TensorProto using the `double_data` field.
        let doubles_vec = create_tensor(
            "doubles_vec",
            &[3],
            onnx::DataType::DOUBLE,
            TensorData::Double(vec![0.1, 0.2, 0.3]),
        );

        let model_proto = onnx::GraphProto::default()
            .with_initializer(doubles_raw)
            .with_initializer(doubles_vec)
            .into_model();

        let model = load_model(model_proto, None).unwrap();

        let floats_raw = model.get_tensor_by_name::<f32>("doubles_raw").unwrap();
        assert_eq!(floats_raw, Tensor::from(0.5));

        let floats_vec = model.get_tensor_by_name::<f32>("doubles_vec").unwrap();
        assert_eq!(floats_vec, TensorView::from(&[0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_load_f16_initializer() {
        // "Round" an f32 by converting to f16 and back
        let round_f16 = |x: f32| f16_to_f32(f32_to_f16(x));

        // TensorProto using the `raw_data` field.
        let f16_raw = create_tensor(
            "f16_raw",
            &[],
            onnx::DataType::FLOAT16,
            TensorData::Raw(f32_to_f16(0.5).to_le_bytes().into()),
        );

        // TensorProto using the `int32_data` field.
        let f16_vec = create_tensor(
            "f16_vec",
            &[3],
            onnx::DataType::FLOAT16,
            TensorData::Int([0.1, 0.2, 0.3].map(|x| f32_to_f16(x) as i32).to_vec()),
        );

        let model_proto = onnx::GraphProto::default()
            .with_initializer(f16_raw)
            .with_initializer(f16_vec)
            .into_model();

        let model = load_model(model_proto, None).unwrap();

        let f16_raw = model.get_tensor_by_name::<f32>("f16_raw").unwrap();
        assert_eq!(f16_raw, Tensor::from(0.5));

        let f16_vec = model.get_tensor_by_name::<f32>("f16_vec").unwrap();
        assert_eq!(
            f16_vec,
            TensorView::from(&[round_f16(0.1), round_f16(0.2), round_f16(0.3)])
        );
    }

    #[test]
    fn test_initializer_with_unsupported_dtype() {
        let tensor = create_tensor(
            "init",
            &[],
            onnx::DataType::BFLOAT16,
            TensorData::Raw((0u16).to_le_bytes().into()),
        );

        let model_proto = onnx::GraphProto::default()
            .with_initializer(tensor)
            .into_model();

        let err = load_model(model_proto, None).err().unwrap();

        assert_eq!(
            err.to_string(),
            "in node \"init\": graph error: initializer has unsupported data type BFLOAT16"
        );
    }

    #[test]
    fn test_initializer_with_external_data() {
        let external_tensor = |name, dtype, offset, length| {
            create_tensor(
                name,
                &[],
                dtype,
                TensorData::External(DataLocation {
                    path: "test.onnx.data".to_string(),
                    offset,
                    length,
                }),
            )
        };

        let i64_tensor = external_tensor("i64_tensor", onnx::DataType::INT64, 8, 8);
        let f32_tensor = external_tensor("f32_tensor", onnx::DataType::FLOAT, 16, 4);
        let bool_tensor = external_tensor("bool_tensor", onnx::DataType::BOOL, 20, 1);

        let model_proto = onnx::GraphProto::default()
            .with_initializer(i64_tensor)
            .with_initializer(f32_tensor)
            .with_initializer(bool_tensor)
            .into_model();

        let mut buf = Vec::new();
        buf.extend(0i64.to_le_bytes());
        buf.extend(1i64.to_le_bytes());
        buf.extend((3.14f32).to_le_bytes());
        buf.push(1u8);
        let loader = MemLoader::from_entries([("test.onnx.data".to_string(), buf)]);

        let model = load_model(model_proto, Some(&loader)).unwrap();

        let tensor = model.get_tensor_by_name::<i32>("i64_tensor").unwrap();
        assert_eq!(tensor, Tensor::from(1i32));

        let tensor = model.get_tensor_by_name::<f32>("f32_tensor").unwrap();
        assert_eq!(tensor, Tensor::from(3.14));

        let tensor = model.get_tensor_by_name::<i32>("bool_tensor").unwrap();
        assert_eq!(tensor, Tensor::from(1i32));
    }

    #[test]
    fn test_unused_attributes() {
        let node = create_node("Clip")
            .with_attr("unused_attr", -0.5)
            .with_name("clip_op");
        let model_proto = onnx::GraphProto::default().with_node(node).into_model();

        let err = load_model(model_proto, None).err().unwrap();

        assert_eq!(
            err.to_string(),
            "in node \"clip_op\": operator error: unsupported or duplicated attributes: unused_attr"
        );
    }

    #[test]
    fn test_metadata() {
        let mut model_proto = onnx::GraphProto::default().into_model();
        model_proto.producer_name = Some("pytorch".into());
        model_proto.producer_version = Some("2.8.0".into());

        let mut custom_prop = onnx::StringStringEntryProto::default();
        custom_prop.key = Some("a_key".into());
        custom_prop.value = Some("a_value".into());
        model_proto.metadata_props.push(custom_prop);

        let model = load_model(model_proto, None).unwrap();

        assert_eq!(model.metadata().producer_name(), Some("pytorch"));
        assert_eq!(model.metadata().producer_version(), Some("2.8.0"));
        assert_eq!(model.metadata().get("a_key"), Some("a_value"));

        let mut fields: Vec<_> = model.metadata().fields().collect();
        fields.sort_by_key(|(field, _val)| *field);
        assert_eq!(
            fields,
            &[
                ("a_key", "a_value"),
                ("producer_name", "pytorch"),
                ("producer_version", "2.8.0"),
            ]
        );
    }

    #[test]
    fn test_too_many_inputs_for_operator() {
        let node = create_node("Relu")
            .with_input("x")
            .with_input("y")
            .with_name("relu_op");

        let model_proto = onnx::GraphProto::default()
            .with_input(create_value_info("x"))
            .with_input(create_value_info("y"))
            .with_node(node)
            .into_model();

        let err = load_model(model_proto, None).err().unwrap();

        assert_eq!(
            err.to_string(),
            "in node \"relu_op\": operator error: operator has 2 inputs but maximum is 1"
        );
    }

    #[test]
    fn test_unnamed_input_dimensions() {
        let mut value_info = create_value_info("x");
        let mut type_proto = onnx::TypeProto::default();
        let mut tensor_type = onnx::TypeProtoTensor::default();
        let mut shape = onnx::TensorShapeProto::default();

        // Add fixed dimension
        let mut dim1 = onnx::Dimension::default();
        dim1.dim_value = Some(3);
        shape.dim.push(dim1);

        // Add named dynamic dimension
        let mut dim2 = onnx::Dimension::default();
        dim2.dim_param = Some("batch".to_string());
        shape.dim.push(dim2);

        // Add unnamed dimension
        shape.dim.push(onnx::Dimension::default());

        tensor_type.shape = Some(shape);
        type_proto.tensor_type = Some(tensor_type);
        value_info.r#type = Some(type_proto);

        let model_proto = onnx::GraphProto::default()
            .with_input(value_info)
            .into_model();

        let model = load_model(model_proto, None).unwrap();

        let graph = model.graph();
        let input_id = graph.input_ids()[0];
        let input_node = graph.get_node(input_id).unwrap();
        let shape = input_node.shape().unwrap();

        assert_eq!(
            shape.as_ref(),
            &[
                Dimension::Fixed(3),
                Dimension::Symbolic("batch".to_string()),
                Dimension::Symbolic("unnamed_1".to_string()),
            ]
        );
    }
}
