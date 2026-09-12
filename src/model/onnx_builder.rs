//! Utilities for building ONNX protobuf messages.

use std::cell::RefCell;

use rten_base::from::enum_from;
use rten_base::num::LeBytes;
use rten_onnx::onnx;
use rten_tensor::TensorView;
use rten_tensor::prelude::*;

use crate::graph::Dimension;
use crate::model::external_data::DataLocation;

#[derive(Clone)]
pub enum AttrValue {
    Bool(bool),
    Float(f32),
    Floats(Vec<f32>),
    Graph(onnx::GraphProto),
    Int(i64),
    Ints(Vec<i64>),
    String(String),
    Strings(Vec<String>),
    Tensor(onnx::TensorProto),
}

enum_from!(AttrValue, Bool, bool);
enum_from!(AttrValue, Float, f32);
enum_from!(AttrValue, Floats, Vec<f32>);
enum_from!(AttrValue, Graph, onnx::GraphProto);
enum_from!(AttrValue, Int, i64);
enum_from!(AttrValue, Ints, Vec<i64>);
enum_from!(AttrValue, String, String);
enum_from!(AttrValue, Strings, Vec<String>);
enum_from!(AttrValue, Tensor, onnx::TensorProto);

pub fn create_attr(name: &str, value: AttrValue) -> onnx::AttributeProto {
    use onnx::AttributeType;

    let mut attr = onnx::AttributeProto::default();
    attr.name = Some(name.to_string());
    attr.r#type = Some(match value {
        AttrValue::Bool(_) | AttrValue::Int(_) => AttributeType::INT,
        AttrValue::Float(_) => AttributeType::FLOAT,
        AttrValue::Floats(_) => AttributeType::FLOATS,
        AttrValue::Graph(_) => AttributeType::GRAPH,
        AttrValue::Ints(_) => AttributeType::INTS,
        AttrValue::String(_) => AttributeType::STRING,
        AttrValue::Strings(_) => AttributeType::STRINGS,
        AttrValue::Tensor(_) => AttributeType::TENSOR,
    });

    match value {
        AttrValue::Bool(val) => attr.i = Some(val as i64),
        AttrValue::Float(val) => attr.f = Some(val),
        AttrValue::Floats(val) => attr.floats = val,
        AttrValue::Graph(val) => attr.g = Some(val),
        AttrValue::Int(val) => attr.i = Some(val),
        AttrValue::Ints(val) => attr.ints = val,
        AttrValue::String(val) => attr.s = Some(val),
        AttrValue::Strings(val) => attr.strings = val,
        AttrValue::Tensor(val) => attr.t = Some(val),
    }

    attr
}

pub trait GraphProtoExt {
    fn into_model(self) -> onnx::ModelProto;
    fn with_initializer(self, tensor: onnx::TensorProto) -> Self;
    fn with_input(self, value: onnx::ValueInfoProto) -> Self;
    fn with_node(self, node: onnx::NodeProto) -> Self;
    fn with_output(self, value: onnx::ValueInfoProto) -> Self;
    fn with_value(self, value: onnx::ValueInfoProto) -> Self;
}

/// Fluent methods for building an [`onnx::GraphProto`].
impl GraphProtoExt for onnx::GraphProto {
    fn into_model(self) -> onnx::ModelProto {
        let mut model = onnx::ModelProto::default();
        model.ir_version = Some(10);
        model.graph = Some(self);
        model
    }

    fn with_initializer(mut self, tensor: onnx::TensorProto) -> Self {
        self.initializer.push(tensor);
        self
    }

    fn with_input(mut self, value: onnx::ValueInfoProto) -> Self {
        self.input.push(value);
        self
    }

    fn with_node(mut self, node: onnx::NodeProto) -> Self {
        self.node.push(node);
        self
    }

    fn with_output(mut self, value: onnx::ValueInfoProto) -> Self {
        self.output.push(value);
        self
    }

    fn with_value(mut self, value: onnx::ValueInfoProto) -> Self {
        self.value_info.push(value);
        self
    }
}

pub fn create_node(op_type: &str) -> onnx::NodeProto {
    let mut node = onnx::NodeProto::default();
    node.op_type = Some(op_type.to_string());
    node
}

/// Fluent methods for building an [`onnx::NodeProto`].
pub trait NodeProtoExt {
    fn with_attr(self, name: &str, value: impl Into<AttrValue>) -> Self;
    fn with_domain(self, domain: &str) -> Self;
    fn with_name(self, name: &str) -> Self;
    fn with_input(self, name: &str) -> Self;
    fn with_output(self, name: &str) -> Self;
}

impl NodeProtoExt for onnx::NodeProto {
    fn with_attr(mut self, name: &str, value: impl Into<AttrValue>) -> Self {
        self.attribute.push(create_attr(name, value.into()));
        self
    }

    fn with_domain(mut self, domain: &str) -> Self {
        self.domain = Some(domain.to_string());
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

    fn with_output(mut self, name: &str) -> Self {
        self.output.push(name.to_string());
        self
    }
}

#[derive(Clone, Debug)]
pub enum TensorData {
    /// Tensor elements as little-endian bytes.
    Raw(Vec<u8>),
    Double(Vec<f64>),
    Int(Vec<i32>),
    External(DataLocation),
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
        TensorData::External(location) => {
            tensor.data_location = Some(onnx::DataLocation::EXTERNAL);
            tensor.external_data = [
                onnx::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some(location.path.clone()),
                },
                onnx::StringStringEntryProto {
                    key: Some("offset".to_string()),
                    value: Some(location.offset.to_string()),
                },
                onnx::StringStringEntryProto {
                    key: Some("length".to_string()),
                    value: Some(location.length.to_string()),
                },
            ]
            .to_vec();
        }
    }

    tensor
}

pub fn create_value_info(name: &str) -> onnx::ValueInfoProto {
    let mut val = onnx::ValueInfoProto::default();
    val.name = Some(name.into());
    val
}

/// Fluent methods for building an [`onnx::ValueInfoProto`].
pub trait ValueInfoProtoExt {
    fn with_dtype(self, dtype: onnx::DataType) -> Self;
    fn with_shape(self, shape: &[Dimension]) -> Self;
}

impl ValueInfoProtoExt for onnx::ValueInfoProto {
    fn with_dtype(mut self, dtype: onnx::DataType) -> Self {
        tensor_type(&mut self).elem_type = Some(dtype);
        self
    }

    fn with_shape(mut self, shape: &[Dimension]) -> Self {
        let dim = shape
            .iter()
            .map(|dim| match dim {
                Dimension::Fixed(size) => onnx::Dimension {
                    dim_value: Some(*size as i64),
                    dim_param: None,
                },
                Dimension::Symbolic(name) => onnx::Dimension {
                    dim_value: None,
                    dim_param: Some(name.clone()),
                },
            })
            .collect();
        tensor_type(&mut self).shape = Some(onnx::TensorShapeProto { dim });
        self
    }
}

/// Return the tensor type of a value, creating it if not set.
fn tensor_type(value: &mut onnx::ValueInfoProto) -> &mut onnx::TypeProtoTensor {
    value
        .r#type
        .get_or_insert_default()
        .tensor_type
        .get_or_insert_default()
}

/// Fluent methods for building an [`onnx::ModelProto`].
pub trait ModelProtoExt {
    fn with_metadata(self, key: &str, value: &str) -> Self;
    fn with_opset(self, domain: &str, version: i64) -> Self;
    fn with_producer(self, name: &str, version: &str) -> Self;
}

impl ModelProtoExt for onnx::ModelProto {
    fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata_props.push(onnx::StringStringEntryProto {
            key: Some(key.to_string()),
            value: Some(value.to_string()),
        });
        self
    }

    fn with_opset(mut self, domain: &str, version: i64) -> Self {
        self.opset_import.push(onnx::OperatorSetIdProto {
            domain: Some(domain.to_string()),
            version: Some(version),
        });
        self
    }

    fn with_producer(mut self, name: &str, version: &str) -> Self {
        self.producer_name = Some(name.to_string());
        self.producer_version = Some(version.to_string());
        self
    }
}

/// Element types which can be stored in an [`onnx::TensorProto`].
pub trait ToTensorProto: Copy + LeBytes {
    /// ONNX data type which corresponds to this Rust type.
    fn dtype() -> onnx::DataType;
}

macro_rules! impl_to_tensor_proto {
    ($type:ty, $dtype:ident) => {
        impl ToTensorProto for $type {
            fn dtype() -> onnx::DataType {
                onnx::DataType::$dtype
            }
        }
    };
}

impl_to_tensor_proto!(f32, FLOAT);
impl_to_tensor_proto!(i32, INT32);
impl_to_tensor_proto!(i8, INT8);
impl_to_tensor_proto!(u8, UINT8);

/// Create an ONNX `TensorProto` from a tensor view.
pub fn create_tensor_from_view<T: ToTensorProto>(
    name: &str,
    view: TensorView<T>,
) -> onnx::TensorProto {
    let mut raw_data = Vec::with_capacity(view.len() * size_of::<T>());
    for elem in view.iter().copied() {
        raw_data.extend_from_slice(elem.to_le_bytes().as_ref());
    }
    create_tensor(name, view.shape(), T::dtype(), TensorData::Raw(raw_data))
}
