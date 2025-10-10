//! Utilities for building ONNX protobuf messages.

use rten_onnx::onnx;

#[derive(Clone)]
pub enum AttrValue {
    Bool(bool),
    Float(f32),
    Int(i64),
    Ints(Vec<i64>),
}

pub fn create_attr(name: &str, value: AttrValue) -> onnx::AttributeProto {
    let mut attr = onnx::AttributeProto::default();
    attr.name = Some(name.to_string());
    match value {
        AttrValue::Bool(val) => attr.i = Some(val as i64),
        AttrValue::Float(val) => attr.f = Some(val),
        AttrValue::Int(val) => attr.i = Some(val),
        AttrValue::Ints(val) => attr.ints = val,
    }
    attr
}

pub fn create_node(op_type: &str, attrs: &[(&str, AttrValue)]) -> onnx::NodeProto {
    let mut node = onnx::NodeProto::default();
    node.op_type = Some(op_type.to_string());
    for (name, val) in attrs {
        node.attribute.push(create_attr(name, val.clone()));
    }
    node
}

pub fn create_value_info(name: &str) -> onnx::ValueInfoProto {
    let mut val = onnx::ValueInfoProto::default();
    val.name = Some(name.into());
    val
}
