//! Utilities for building ONNX protobuf messages.

use std::cell::RefCell;

use rten_onnx::onnx;

#[derive(Clone)]
pub enum AttrValue {
    Bool(bool),
    Float(f32),
    Int(i64),
    Ints(Vec<i64>),
    Tensor(onnx::TensorProto),
}

pub fn create_attr(name: &str, value: AttrValue) -> onnx::AttributeProto {
    let mut attr = onnx::AttributeProto::default();
    attr.name = Some(name.to_string());
    match value {
        AttrValue::Bool(val) => attr.i = Some(val as i64),
        AttrValue::Float(val) => attr.f = Some(val),
        AttrValue::Int(val) => attr.i = Some(val),
        AttrValue::Ints(val) => attr.ints = val,
        AttrValue::Tensor(val) => attr.t = Some(val),
    }
    attr
}

pub fn create_model(graph: onnx::GraphProto) -> onnx::ModelProto {
    let mut model = onnx::ModelProto::default();
    model.graph = Some(graph);
    model
}

pub fn create_node(op_type: &str, attrs: &[(&str, AttrValue)]) -> onnx::NodeProto {
    let mut node = onnx::NodeProto::default();
    node.op_type = Some(op_type.to_string());
    for (name, val) in attrs {
        node.attribute.push(create_attr(name, val.clone()));
    }
    node
}

#[derive(Clone, Debug)]
pub enum TensorData {
    /// Tensor elements as little-endian bytes.
    Raw(Vec<u8>),
    Double(Vec<f64>),
    Int(Vec<i32>),
}

pub fn create_tensor(
    name: &str,
    shape: &[usize],
    dtype: onnx::DataType,
    data: TensorData,
) -> onnx::TensorProto {
    let mut tensor = onnx::TensorProto::default();
    tensor.name = Some(name.to_string());
    tensor.dims = shape.iter().map(|size| *size as i64).collect();
    tensor.data_type = Some(dtype);

    match data {
        TensorData::Raw(raw) => tensor.raw_data = Some(RefCell::new(raw)),
        TensorData::Double(doubles) => tensor.double_data = doubles,
        TensorData::Int(ints) => tensor.int32_data = ints,
    }

    tensor
}

pub fn create_value_info(name: &str) -> onnx::ValueInfoProto {
    let mut val = onnx::ValueInfoProto::default();
    val.name = Some(name.into());
    val
}
