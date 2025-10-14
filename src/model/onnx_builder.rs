//! Utilities for building ONNX protobuf messages.

use std::cell::RefCell;

use rten_base::from::enum_from;
use rten_onnx::onnx;

#[derive(Clone)]
pub enum AttrValue {
    Bool(bool),
    Float(f32),
    Graph(onnx::GraphProto),
    Int(i64),
    Ints(Vec<i64>),
    String(String),
    Tensor(onnx::TensorProto),
}

enum_from!(AttrValue, Bool, bool);
enum_from!(AttrValue, Float, f32);
enum_from!(AttrValue, Graph, onnx::GraphProto);
enum_from!(AttrValue, Int, i64);
enum_from!(AttrValue, Ints, Vec<i64>);
enum_from!(AttrValue, String, String);
enum_from!(AttrValue, Tensor, onnx::TensorProto);

pub fn create_attr(name: &str, value: AttrValue) -> onnx::AttributeProto {
    let mut attr = onnx::AttributeProto::default();
    attr.name = Some(name.to_string());
    match value {
        AttrValue::Bool(val) => attr.i = Some(val as i64),
        AttrValue::Float(val) => attr.f = Some(val),
        AttrValue::Graph(val) => attr.g = Some(val),
        AttrValue::Int(val) => attr.i = Some(val),
        AttrValue::Ints(val) => attr.ints = val,
        AttrValue::String(val) => attr.s = Some(val),
        AttrValue::Tensor(val) => attr.t = Some(val),
    }
    attr
}

pub fn create_model(graph: onnx::GraphProto) -> onnx::ModelProto {
    let mut model = onnx::ModelProto::default();
    model.graph = Some(graph);
    model
}

pub fn create_node(op_type: &str) -> onnx::NodeProto {
    let mut node = onnx::NodeProto::default();
    node.op_type = Some(op_type.to_string());
    node
}

/// Fluent methods for building an [`onnx::NodeProto`].
pub trait NodeProtoExt {
    fn with_attr(self, name: &str, value: impl Into<AttrValue>) -> Self;
    fn with_name(self, name: &str) -> Self;
    fn with_input(self, name: &str) -> Self;
}

impl NodeProtoExt for onnx::NodeProto {
    fn with_attr(mut self, name: &str, value: impl Into<AttrValue>) -> Self {
        self.attribute.push(create_attr(name, value.into()));
        self
    }

    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn with_input(mut self, name: &str) -> Self {
        self.input.push(name.to_string());
        self
    }
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
