//! Types representing parsed ONNX protobuf messages.
//!
//! Each type corresponds to a message in
//! [onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3). See
//! the onnx.proto schema for documentation for individual fields and messages.
//! An ONNX model file is a serialized representation of the [`ModelProto`]
//! message.
//!
//! Enums are represented as i32 wrappers with associated constants for each
//! of the values. This is used instead of a Rust `enum` because protobuf
//! enums are open.

use protozero::field::{Field, FieldValue};

use crate::decode::{
    DecodeError, DecodeField, DecodeFrom, DecodeMessage, impl_decode_from_enum, unknown_field,
};

#[derive(Default)]
pub struct Dimension<'a> {
    pub dim_value: Option<i64>,
    pub dim_param: Option<&'a str>,
}

impl Dimension<'_> {
    const DIM_VALUE: u64 = 1; // int64
    const DIM_PARAM: u64 = 2; // string
}

impl<'a> DecodeField<'a> for Dimension<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::DIM_VALUE => self.dim_value.decode_from(field.value),
            Self::DIM_PARAM => self.dim_param.decode_from(field.value),
            _ => unknown_field("Dimension", field),
        }
    }
}

#[derive(Default)]
pub struct TensorShapeProto<'a> {
    pub dim: Vec<Dimension<'a>>,
}

impl TensorShapeProto<'_> {
    const DIM: u64 = 1; // repeated Dimension
}

impl<'a> DecodeField<'a> for TensorShapeProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::DIM => self.dim.decode_from(field.value),
            _ => unknown_field("TensorShapeProto", field),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct DataType(pub i32);

impl DataType {
    pub const FLOAT: Self = Self(1);
    pub const INT32: Self = Self(6);
    pub const INT64: Self = Self(7);
    pub const UINT8: Self = Self(2);
    pub const INT8: Self = Self(3);
    pub const BOOL: Self = Self(9);
}

impl_decode_from_enum!(DataType);

#[derive(Default)]
pub struct Tensor<'a> {
    pub elem_type: Option<DataType>,
    pub shape: Option<TensorShapeProto<'a>>,
}

impl Tensor<'_> {
    const ELEM_TYPE: u64 = 1; // DataType
    const SHAPE: u64 = 2; // TensorShapeProto
}

impl<'a> DecodeField<'a> for Tensor<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::ELEM_TYPE => {
                self.elem_type = Some(DataType(field.value.get_enum()?));
                Ok(())
            }
            Self::SHAPE => self.shape.decode_from(field.value),
            _ => unknown_field("Tensor", field),
        }
    }
}

#[derive(Default)]
pub struct Sequence<'a> {
    pub elem_type: Option<TypeProto<'a>>,
}

impl Sequence<'_> {
    const ELEM_TYPE: u64 = 1; // TypeProto
}

impl<'a> DecodeField<'a> for Sequence<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::ELEM_TYPE => self.elem_type.decode_from(field.value),
            _ => unknown_field("Sequence", field),
        }
    }
}

#[derive(Default)]
pub struct TypeProto<'a> {
    pub tensor_type: Option<Tensor<'a>>,
    pub sequence: Option<Box<Sequence<'a>>>,
}

impl TypeProto<'_> {
    const TENSOR_TYPE: u64 = 1; // TypeProto.Tensor
    const SEQUENCE: u64 = 4; // TypeProto.Sequence
}

impl<'a> DecodeField<'a> for TypeProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::TENSOR_TYPE => self.tensor_type.decode_from(field.value),
            Self::SEQUENCE => {
                let msg = field.value.get_bytes()?;
                self.sequence = Some(Box::new(Sequence::decode(msg)?));
                Ok(())
            }
            _ => unknown_field("TypeProto", field),
        }
    }
}

#[derive(Default)]
pub struct ValueInfoProto<'a> {
    pub name: Option<&'a str>,
    pub r#type: Option<TypeProto<'a>>,
}

impl ValueInfoProto<'_> {
    const NAME: u64 = 1; // string
    const TYPE: u64 = 2; // TypeProto
}

impl<'a> DecodeField<'a> for ValueInfoProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::NAME => self.name.decode_from(field.value),
            Self::TYPE => self.r#type.decode_from(field.value),
            _ => unknown_field("ValueInfoProto", field),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct AttributeType(pub i32);

impl AttributeType {
    pub const UNDEFINED: Self = Self(0);
    pub const FLOAT: Self = Self(1);
    pub const INT: Self = Self(2);
    pub const STRING: Self = Self(3);
    pub const GRAPH: Self = Self(5);
    pub const FLOATS: Self = Self(6);
    pub const INTS: Self = Self(7);
}

impl_decode_from_enum!(AttributeType);

#[derive(Default)]
pub struct AttributeProto<'a> {
    pub name: Option<&'a str>,
    pub f: Option<f32>,
    pub s: Option<&'a str>,
    pub i: Option<i64>,
    pub g: Option<GraphProto<'a>>,
    pub t: Option<TensorProto<'a>>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub r#type: Option<AttributeType>,
}

impl AttributeProto<'_> {
    const NAME: u64 = 1; // string
    const F: u64 = 2; // float
    const I: u64 = 3; // int64
    const S: u64 = 4; // bytes
    const T: u64 = 5; // TensorProto
    const G: u64 = 6; // GraphProto
    const FLOATS: u64 = 7; // repeated float
    const INTS: u64 = 8; // repeated int64
    const TYPE: u64 = 20; // AttributeType
}

impl<'a> DecodeField<'a> for AttributeProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::NAME => self.name.decode_from(field.value),
            Self::F => self.f.decode_from(field.value),
            Self::I => self.i.decode_from(field.value),
            Self::S => self.s.decode_from(field.value),
            Self::T => self.t.decode_from(field.value),
            Self::G => self.g.decode_from(field.value),
            Self::FLOATS => self.floats.decode_from(field.value),
            Self::INTS => self.ints.decode_from(field.value),
            Self::TYPE => self.r#type.decode_from(field.value),
            _ => unknown_field("AttributeProto", field),
        }
    }
}

#[derive(Default)]
pub struct NodeProto<'a> {
    pub name: Option<&'a str>,
    pub input: Vec<&'a str>,
    pub output: Vec<&'a str>,
    pub op_type: Option<&'a str>,
    pub attribute: Vec<AttributeProto<'a>>,
}

impl NodeProto<'_> {
    const INPUT: u64 = 1; // repeated string
    const OUTPUT: u64 = 2; // repeated string
    const NAME: u64 = 3; // string
    const OP_TYPE: u64 = 4; // string
    const ATTRIBUTE: u64 = 5; // repeated AttributeProto
    const DOC_STRING: u64 = 6; // string
    const DOMAIN: u64 = 7; // string
    const METADATA_PROPS: u64 = 9; // repeated StringStringEntryProto
}

impl<'a> DecodeField<'a> for NodeProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::INPUT => self.input.decode_from(field.value),
            Self::OUTPUT => self.output.decode_from(field.value),
            Self::NAME => self.name.decode_from(field.value),
            Self::OP_TYPE => self.op_type.decode_from(field.value),
            Self::ATTRIBUTE => self.attribute.decode_from(field.value),
            Self::DOC_STRING => Ok(()),
            Self::DOMAIN => Ok(()),
            Self::METADATA_PROPS => Ok(()),
            _ => unknown_field("NodeProto", field),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct DataLocation(pub i32);

impl DataLocation {
    pub const DEFAULT: Self = Self(0);
    pub const EXTERNAL: Self = Self(1);
}

impl_decode_from_enum!(DataLocation);

#[derive(Debug, Default)]
pub struct TensorProto<'a> {
    pub name: Option<&'a str>,
    pub data_type: Option<DataType>,
    pub dims: Vec<i64>,
    pub raw_data: Option<&'a [u8]>,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i32>,
    pub int64_data: Vec<i64>,
    pub external_data: Vec<StringStringEntryProto<'a>>,
    pub data_location: Option<DataLocation>,
}

impl TensorProto<'_> {
    const DIMS: u64 = 1; // repeated int64
    const DATA_TYPE: u64 = 2; // DataType
    const FLOAT_DATA: u64 = 4; // repeated float [packed]
    const INT32_DATA: u64 = 5; // repeated int32 [packed]
    const INT64_DATA: u64 = 7; // repeated int64 [packed]
    const NAME: u64 = 8; // string
    const RAW_DATA: u64 = 9; // bytes
    const EXTERNAL_DATA: u64 = 13; // repeated StringStringEntryProto
    const DATA_LOCATION: u64 = 14; // DataLocation
}

impl<'a> DecodeField<'a> for TensorProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::DIMS => self.dims.decode_from(field.value),
            Self::DATA_TYPE => self.data_type.decode_from(field.value),
            Self::FLOAT_DATA => self.float_data.decode_from(field.value),
            Self::INT32_DATA => self.int32_data.decode_from(field.value),
            Self::INT64_DATA => self.int64_data.decode_from(field.value),
            Self::NAME => self.name.decode_from(field.value),
            Self::RAW_DATA => self.raw_data.decode_from(field.value),
            Self::EXTERNAL_DATA => self.external_data.decode_from(field.value),
            Self::DATA_LOCATION => self.data_location.decode_from(field.value),
            _ => unknown_field("TensorProto", field),
        }
    }
}

#[derive(Debug, Default)]
pub struct StringStringEntryProto<'a> {
    pub key: Option<&'a str>,
    pub value: Option<&'a str>,
}

impl StringStringEntryProto<'_> {
    const KEY: u64 = 1; // string
    const VALUE: u64 = 2; // string
}

impl<'a> DecodeField<'a> for StringStringEntryProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::KEY => self.key.decode_from(field.value),
            Self::VALUE => self.value.decode_from(field.value),
            _ => unknown_field("StringStringEntryProto", field),
        }
    }
}

#[derive(Default)]
pub struct GraphProto<'a> {
    pub node: Vec<NodeProto<'a>>,
    pub initializer: Vec<TensorProto<'a>>,
    pub input: Vec<ValueInfoProto<'a>>,
    pub output: Vec<ValueInfoProto<'a>>,
    pub value_info: Vec<ValueInfoProto<'a>>,
}

impl GraphProto<'_> {
    const NODE: u64 = 1; // repeated NodeProto
    const NAME: u64 = 2; // string
    const INITIALIZER: u64 = 5; // repeated TensorProto
    const DOC_STRING: u64 = 10; // string
    const INPUT: u64 = 11; // repeated ValueInfoProto
    const OUTPUT: u64 = 12; // repeated ValueInfoProto
    const VALUE_INFO: u64 = 13; // repeated ValueInfoProto
    const METADATA_PROPS: u64 = 16; // repeated StringStringEntryProto
}

impl<'a> DecodeField<'a> for GraphProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::NODE => self.node.decode_from(field.value),
            Self::NAME => Ok(()),
            Self::INITIALIZER => self.initializer.decode_from(field.value),
            Self::DOC_STRING => Ok(()),
            Self::INPUT => self.input.decode_from(field.value),
            Self::OUTPUT => self.output.decode_from(field.value),
            Self::VALUE_INFO => self.value_info.decode_from(field.value),
            Self::METADATA_PROPS => Ok(()),
            _ => unknown_field("GraphProto", field),
        }
    }
}

#[derive(Default)]
pub struct ModelProto<'a> {
    pub graph: Option<GraphProto<'a>>,
}

impl ModelProto<'_> {
    const IR_VERSION: u64 = 1; // int64
    const PRODUCER_NAME: u64 = 2; // string
    const PRODUCER_VERSION: u64 = 3; // string
    const DOC_STRING: u64 = 6; // string
    const GRAPH: u64 = 7; // GraphProto
    const OPSET_IMPORT: u64 = 8; // repeated OperatorSetIdProto
    const METADATA_PROPS: u64 = 14; // repeated StringStringEntryProto
}

impl<'a> DecodeField<'a> for ModelProto<'a> {
    fn decode_field(&mut self, field: Field<'a>) -> Result<(), DecodeError> {
        match field.number {
            Self::IR_VERSION => Ok(()),
            Self::PRODUCER_NAME => Ok(()),
            Self::PRODUCER_VERSION => Ok(()),
            Self::DOC_STRING => Ok(()),
            Self::GRAPH => self.graph.decode_from(field.value),
            Self::OPSET_IMPORT => Ok(()),
            Self::METADATA_PROPS => Ok(()),
            _ => unknown_field("ModelProto", field),
        }
    }
}
