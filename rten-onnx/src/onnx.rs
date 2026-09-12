//! ONNX model Protocol Buffers types.
//!
//! The types in this module correspond to Protocol Buffers messages defined
//! in [onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3).
//! See the `.proto` file for detailed information on each type and field.
//!
//! These types are not complete. They only contain messages and fields which
//! are used by RTen or its associated tools.

use std::cell::RefCell;
use std::fmt;
use std::fs::File;

use crate::protobuf::{
    DecodeMessage, EncodeMessage, Fields, MessageWriter, OwnedValues, ProtobufError, ReadValue,
    ValueReader, ValueWriter, WriteValue,
};

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct AttributeType(pub i32);

impl AttributeType {
    pub const UNDEFINED: Self = Self(0);
    pub const FLOAT: Self = Self(1);
    pub const INT: Self = Self(2);
    pub const STRING: Self = Self(3);
    pub const TENSOR: Self = Self(4);
    pub const GRAPH: Self = Self(5);
    pub const FLOATS: Self = Self(6);
    pub const INTS: Self = Self(7);
    pub const STRINGS: Self = Self(8);
}

#[derive(Clone, Debug, Default)]
pub struct AttributeProto {
    pub name: Option<String>,
    pub f: Option<f32>,
    pub s: Option<String>,
    pub i: Option<i64>,
    pub g: Option<GraphProto>,
    pub t: Option<TensorProto>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<String>,
    pub r#type: Option<AttributeType>,
}

impl AttributeProto {
    const NAME: u64 = 1;
    const F: u64 = 2;
    const I: u64 = 3;
    const S: u64 = 4;
    const T: u64 = 5;
    const G: u64 = 6;
    const FLOATS: u64 = 7;
    const INTS: u64 = 8;
    const STRINGS: u64 = 9;
    const TYPE: u64 = 20;
}

impl DecodeMessage for AttributeProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::NAME => {
                    msg.name = Some(field.read_string()?);
                }
                Self::F => {
                    msg.f = Some(field.get_float()?);
                }
                Self::S => {
                    msg.s = Some(field.read_string()?);
                }
                Self::I => {
                    msg.i = Some(field.get_int64()?);
                }
                Self::G => {
                    msg.g = Some(GraphProto::decode_field(&mut field)?);
                }
                Self::T => {
                    msg.t = Some(TensorProto::decode_field(&mut field)?);
                }
                Self::FLOATS => {
                    msg.floats.push(field.get_float()?);
                }
                Self::INTS => {
                    msg.ints.push(field.get_int64()?);
                }
                Self::STRINGS => {
                    msg.strings.push(field.read_string()?);
                }
                Self::TYPE => {
                    msg.r#type = Some(AttributeType(field.get_enum()?));
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for AttributeProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(name) = &self.name {
            fields.write_string(Self::NAME, name)?;
        }
        if let Some(f) = self.f {
            fields.write_float(Self::F, f)?;
        }
        if let Some(i) = self.i {
            fields.write_int64(Self::I, i)?;
        }
        if let Some(s) = &self.s {
            fields.write_string(Self::S, s)?;
        }
        if let Some(t) = &self.t {
            fields.write_message(Self::T, t)?;
        }
        if let Some(g) = &self.g {
            fields.write_message(Self::G, g)?;
        }
        fields.write_repeated_float(Self::FLOATS, &self.floats)?;
        fields.write_repeated_int64(Self::INTS, &self.ints)?;
        fields.write_repeated_string(Self::STRINGS, &self.strings)?;
        if let Some(ty) = self.r#type {
            fields.write_enum(Self::TYPE, ty.0)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct NodeProto {
    pub domain: Option<String>,
    pub name: Option<String>,
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub op_type: Option<String>,
    pub attribute: Vec<AttributeProto>,
}

impl NodeProto {
    const INPUT: u64 = 1;
    const OUTPUT: u64 = 2;
    const NAME: u64 = 3;
    const OP_TYPE: u64 = 4;
    const ATTRIBUTE: u64 = 5;
    const DOMAIN: u64 = 7;
}

impl DecodeMessage for NodeProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::INPUT => {
                    msg.input.push(field.read_string()?);
                }
                Self::OUTPUT => {
                    msg.output.push(field.read_string()?);
                }
                Self::NAME => {
                    msg.name = Some(field.read_string()?);
                }
                Self::OP_TYPE => {
                    msg.op_type = Some(field.read_string()?);
                }
                Self::ATTRIBUTE => {
                    msg.attribute
                        .push(AttributeProto::decode_field(&mut field)?);
                }
                Self::DOMAIN => {
                    msg.domain = Some(field.read_string()?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for NodeProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        fields.write_repeated_string(Self::INPUT, &self.input)?;
        fields.write_repeated_string(Self::OUTPUT, &self.output)?;
        if let Some(name) = &self.name {
            fields.write_string(Self::NAME, name)?;
        }
        if let Some(op_type) = &self.op_type {
            fields.write_string(Self::OP_TYPE, op_type)?;
        }
        fields.write_repeated_message(Self::ATTRIBUTE, &self.attribute)?;
        if let Some(domain) = &self.domain {
            fields.write_string(Self::DOMAIN, domain)?;
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct TensorProto {
    pub dims: Vec<i64>,
    pub data_type: Option<DataType>,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i32>,
    pub int64_data: Vec<i64>,
    pub double_data: Vec<f64>,

    /// Field containing tensor data as bytes in packed little-endian order.
    ///
    /// This is the field most often used to store data for large tensors. It
    /// uses a cell so that the buffer can be extracted from the message for use
    /// as backing storage of a tensor, without additional copying.
    pub raw_data: Option<RefCell<Vec<u8>>>,

    pub name: Option<String>,
    pub external_data: Vec<StringStringEntryProto>,
    pub data_location: Option<DataLocation>,
}

impl TensorProto {
    const DIMS: u64 = 1;
    const DATA_TYPE: u64 = 2;
    const FLOAT_DATA: u64 = 4;
    const INT32_DATA: u64 = 5;
    const INT64_DATA: u64 = 7;
    const NAME: u64 = 8;
    const RAW_DATA: u64 = 9;
    const DOUBLE_DATA: u64 = 10;
    const EXTERNAL_DATA: u64 = 13;
    const DATA_LOCATION: u64 = 14;
}

impl std::fmt::Debug for TensorProto {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("TensorProto")
            .field("dims", &self.dims)
            .field("data_type", &self.data_type)
            .field("name", &self.name)
            .field("data_location", &self.data_location)
            .finish()
    }
}

impl DecodeMessage for TensorProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = TensorProto::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::DIMS => {
                    msg.dims.push(field.get_int64()?);
                }
                Self::DATA_TYPE => {
                    msg.data_type = Some(DataType(field.get_enum()?));
                }
                Self::FLOAT_DATA => {
                    for float in field.read_repeated_float()? {
                        msg.float_data.push(float?);
                    }
                }
                Self::INT32_DATA => {
                    for int32 in field.read_repeated_int32()? {
                        msg.int32_data.push(int32?);
                    }
                }
                Self::INT64_DATA => {
                    for int64 in field.read_repeated_int64()? {
                        msg.int64_data.push(int64?);
                    }
                }
                Self::DOUBLE_DATA => {
                    for double in field.read_repeated_double()? {
                        msg.double_data.push(double?);
                    }
                }
                Self::NAME => {
                    msg.name = Some(field.read_string()?);
                }
                Self::RAW_DATA => {
                    msg.raw_data = Some(RefCell::new(field.read_bytes()?));
                }
                Self::EXTERNAL_DATA => {
                    msg.external_data
                        .push(StringStringEntryProto::decode_field(&mut field)?);
                }
                Self::DATA_LOCATION => {
                    msg.data_location = Some(DataLocation(field.get_enum()?));
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for TensorProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        fields.write_repeated_int64(Self::DIMS, &self.dims)?;
        if let Some(data_type) = self.data_type {
            fields.write_enum(Self::DATA_TYPE, data_type.0)?;
        }
        fields.write_packed_float(Self::FLOAT_DATA, &self.float_data)?;
        fields.write_packed_int32(Self::INT32_DATA, &self.int32_data)?;
        fields.write_packed_int64(Self::INT64_DATA, &self.int64_data)?;
        if let Some(name) = &self.name {
            fields.write_string(Self::NAME, name)?;
        }
        if let Some(raw_data) = &self.raw_data {
            fields.write_bytes(Self::RAW_DATA, raw_data.borrow().as_slice())?;
        }
        fields.write_packed_double(Self::DOUBLE_DATA, &self.double_data)?;
        fields.write_repeated_message(Self::EXTERNAL_DATA, &self.external_data)?;
        if let Some(data_location) = self.data_location {
            fields.write_enum(Self::DATA_LOCATION, data_location.0)?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct DataLocation(pub i32);

impl DataLocation {
    pub const DEFAULT: Self = Self(0);
    pub const EXTERNAL: Self = Self(1);
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct DataType(pub i32);

impl DataType {
    pub const FLOAT: Self = Self(1);
    pub const UINT8: Self = Self(2);
    pub const INT8: Self = Self(3);
    pub const UINT16: Self = Self(4);
    pub const INT16: Self = Self(5);
    pub const INT32: Self = Self(6);
    pub const INT64: Self = Self(7);
    pub const STRING: Self = Self(8);
    pub const BOOL: Self = Self(9);
    pub const FLOAT16: Self = Self(10);
    pub const DOUBLE: Self = Self(11);
    pub const UINT32: Self = Self(12);
    pub const UINT64: Self = Self(13);
    pub const COMPLEX64: Self = Self(14);
    pub const COMPLEX128: Self = Self(15);
    pub const BFLOAT16: Self = Self(16);
    pub const FLOAT8E4M3FN: Self = Self(17);
    pub const FLOAT8E4M3FNUZ: Self = Self(18);
    pub const FLOAT8E5M2: Self = Self(19);
    pub const FLOAT8E5M2FNUZ: Self = Self(20);
    pub const UINT4: Self = Self(21);
    pub const INT4: Self = Self(22);
    pub const FLOAT4E2M1: Self = Self(23);
    pub const FLOAT8E8M0: Self = Self(24);

    pub fn name(&self) -> Option<&str> {
        match *self {
            Self::FLOAT => Some("FLOAT"),
            Self::UINT8 => Some("UINT8"),
            Self::INT8 => Some("INT8"),
            Self::UINT16 => Some("UINT16"),
            Self::INT16 => Some("INT16"),
            Self::INT32 => Some("INT32"),
            Self::INT64 => Some("INT64"),
            Self::STRING => Some("STRING"),
            Self::BOOL => Some("BOOL"),
            Self::FLOAT16 => Some("FLOAT16"),
            Self::DOUBLE => Some("DOUBLE"),
            Self::UINT32 => Some("UINT32"),
            Self::UINT64 => Some("UINT64"),
            Self::COMPLEX64 => Some("COMPLEX64"),
            Self::COMPLEX128 => Some("COMPLEX128"),
            Self::BFLOAT16 => Some("BFLOAT16"),
            Self::FLOAT8E4M3FN => Some("FLOAT8E4M3FN"),
            Self::FLOAT8E4M3FNUZ => Some("FLOAT8E4M3FNUZ"),
            Self::FLOAT8E5M2 => Some("FLOAT8E5M2"),
            Self::FLOAT8E5M2FNUZ => Some("FLOAT8E5M2FNUZ"),
            Self::UINT4 => Some("UINT4"),
            Self::INT4 => Some("INT4"),
            Self::FLOAT4E2M1 => Some("FLOAT4E2M1"),
            Self::FLOAT8E8M0 => Some("FLOAT8E8M0"),
            _ => None,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(f, "{name}"),
            None => write!(f, "{}", self.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Dimension {
    pub dim_value: Option<i64>,
    pub dim_param: Option<String>,
}

impl Dimension {
    const DIM_VALUE: u64 = 1;
    const DIM_PARAM: u64 = 2;
}

impl DecodeMessage for Dimension {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::DIM_VALUE => {
                    msg.dim_value = Some(field.get_int64()?);
                }
                Self::DIM_PARAM => {
                    msg.dim_param = Some(field.read_string()?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for Dimension {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(dim_value) = self.dim_value {
            fields.write_int64(Self::DIM_VALUE, dim_value)?;
        }
        if let Some(dim_param) = &self.dim_param {
            fields.write_string(Self::DIM_PARAM, dim_param)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct StringStringEntryProto {
    pub key: Option<String>,
    pub value: Option<String>,
}

impl StringStringEntryProto {
    const KEY: u64 = 1;
    const VALUE: u64 = 2;
}

impl DecodeMessage for StringStringEntryProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::KEY => {
                    msg.key = Some(field.read_string()?);
                }
                Self::VALUE => {
                    msg.value = Some(field.read_string()?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for StringStringEntryProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(key) = &self.key {
            fields.write_string(Self::KEY, key)?;
        }
        if let Some(value) = &self.value {
            fields.write_string(Self::VALUE, value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct OperatorSetIdProto {
    pub domain: Option<String>,
    pub version: Option<i64>,
}

impl OperatorSetIdProto {
    const DOMAIN: u64 = 1;
    const VERSION: u64 = 2;
}

impl DecodeMessage for OperatorSetIdProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::DOMAIN => {
                    msg.domain = Some(field.read_string()?);
                }
                Self::VERSION => {
                    msg.version = Some(field.get_int64()?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for OperatorSetIdProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(domain) = &self.domain {
            fields.write_string(Self::DOMAIN, domain)?;
        }
        if let Some(version) = self.version {
            fields.write_int64(Self::VERSION, version)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct TensorShapeProto {
    pub dim: Vec<Dimension>,
}

impl TensorShapeProto {
    const DIM: u64 = 1;
}

impl DecodeMessage for TensorShapeProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::DIM => {
                    msg.dim.push(Dimension::decode_field(&mut field)?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for TensorShapeProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        fields.write_repeated_message(Self::DIM, &self.dim)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct TypeProtoTensor {
    pub elem_type: Option<DataType>,
    pub shape: Option<TensorShapeProto>,
}

impl TypeProtoTensor {
    const ELEM_TYPE: u64 = 1; // DataType
    const SHAPE: u64 = 2; // TensorShapeProto
}

impl DecodeMessage for TypeProtoTensor {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::ELEM_TYPE => {
                    msg.elem_type = Some(DataType(field.get_enum()?));
                }
                Self::SHAPE => {
                    msg.shape = Some(TensorShapeProto::decode_field(&mut field)?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for TypeProtoTensor {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(elem_type) = self.elem_type {
            fields.write_enum(Self::ELEM_TYPE, elem_type.0)?;
        }
        if let Some(shape) = &self.shape {
            fields.write_message(Self::SHAPE, shape)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct TypeProtoSequence {
    pub elem_type: Option<TypeProto>,
}

impl TypeProtoSequence {
    const ELEM_TYPE: u64 = 1;
}

impl DecodeMessage for TypeProtoSequence {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::ELEM_TYPE => {
                    msg.elem_type = Some(TypeProto::decode_field(&mut field)?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for TypeProtoSequence {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(elem_type) = &self.elem_type {
            fields.write_message(Self::ELEM_TYPE, elem_type)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct TypeProto {
    pub tensor_type: Option<TypeProtoTensor>,
    pub sequence: Option<Box<TypeProtoSequence>>,
}

impl TypeProto {
    const TENSOR_TYPE: u64 = 1;
    const SEQUENCE: u64 = 4;
}

impl DecodeMessage for TypeProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::TENSOR_TYPE => {
                    msg.tensor_type = Some(TypeProtoTensor::decode_field(&mut field)?);
                }
                Self::SEQUENCE => {
                    msg.sequence = Some(Box::new(TypeProtoSequence::decode_field(&mut field)?));
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for TypeProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(tensor_type) = &self.tensor_type {
            fields.write_message(Self::TENSOR_TYPE, tensor_type)?;
        }
        if let Some(sequence) = &self.sequence {
            fields.write_message(Self::SEQUENCE, sequence.as_ref())?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct ValueInfoProto {
    pub name: Option<String>,
    pub r#type: Option<TypeProto>,
}

impl ValueInfoProto {
    const NAME: u64 = 1;
    const TYPE: u64 = 2;
}

impl DecodeMessage for ValueInfoProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::NAME => {
                    msg.name = Some(field.read_string()?);
                }
                Self::TYPE => {
                    msg.r#type = Some(TypeProto::decode_field(&mut field)?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for ValueInfoProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(name) = &self.name {
            fields.write_string(Self::NAME, name)?;
        }
        if let Some(ty) = &self.r#type {
            fields.write_message(Self::TYPE, ty)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct GraphProto {
    pub node: Vec<NodeProto>,
    pub initializer: Vec<TensorProto>,
    pub input: Vec<ValueInfoProto>,
    pub output: Vec<ValueInfoProto>,
    pub value_info: Vec<ValueInfoProto>,
}

impl GraphProto {
    const NODE: u64 = 1;
    const INITIALIZER: u64 = 5;
    const INPUT: u64 = 11;
    const OUTPUT: u64 = 12;
    const VALUE_INFO: u64 = 13;
}

impl DecodeMessage for GraphProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::NODE => {
                    msg.node.push(NodeProto::decode_field(&mut field)?);
                }
                Self::INITIALIZER => {
                    msg.initializer.push(TensorProto::decode_field(&mut field)?);
                }
                Self::INPUT => {
                    msg.input.push(ValueInfoProto::decode_field(&mut field)?);
                }
                Self::OUTPUT => {
                    msg.output.push(ValueInfoProto::decode_field(&mut field)?);
                }
                Self::VALUE_INFO => {
                    msg.value_info
                        .push(ValueInfoProto::decode_field(&mut field)?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for GraphProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        fields.write_repeated_message(Self::NODE, &self.node)?;
        fields.write_repeated_message(Self::INITIALIZER, &self.initializer)?;
        fields.write_repeated_message(Self::INPUT, &self.input)?;
        fields.write_repeated_message(Self::OUTPUT, &self.output)?;
        fields.write_repeated_message(Self::VALUE_INFO, &self.value_info)?;
        Ok(())
    }
}

#[derive(Default)]
pub struct ModelProto {
    pub ir_version: Option<i64>,
    pub graph: Option<GraphProto>,
    pub opset_import: Vec<OperatorSetIdProto>,
    pub metadata_props: Vec<StringStringEntryProto>,
    pub producer_name: Option<String>,
    pub producer_version: Option<String>,
}

impl ModelProto {
    const IR_VERSION: u64 = 1;
    const PRODUCER_NAME: u64 = 2;
    const PRODUCER_VERSION: u64 = 3;
    const GRAPH: u64 = 7;
    const OPSET_IMPORT: u64 = 8;
    const METADATA_PROPS: u64 = 14;

    // The non-generic `parse_*` and `write_*` methods allow the parsing and
    // serialization code to be compiled as part of the rten-onnx crate.

    /// Deserialize a `ModelProto` from a file.
    pub fn parse_file(file: File) -> Result<Self, ProtobufError> {
        let reader = ValueReader::from_file(file);
        ModelProto::decode(reader)
    }

    /// Deserialize a `ModelProto` from a buffer.
    pub fn parse_buf(buf: &[u8]) -> Result<Self, ProtobufError> {
        let reader = ValueReader::from_buf(buf);
        ModelProto::decode(reader)
    }

    /// Serialize this model to a file.
    pub fn write_file(&self, file: File) -> Result<(), ProtobufError> {
        self.encode(ValueWriter::from_file(file))
    }

    /// Serialize this model to a buffer.
    pub fn write_buf(&self) -> Result<Vec<u8>, ProtobufError> {
        // Size the buffer up front to avoid repeatedly re-allocating and
        // copying the tensor data as it grows.
        let len = self.encoded_len()?;
        let mut buf = Vec::with_capacity(len as usize);
        self.encode(ValueWriter::new(&mut buf))?;
        Ok(buf)
    }
}

impl DecodeMessage for ModelProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                Self::IR_VERSION => {
                    msg.ir_version = Some(field.get_int64()?);
                }
                Self::GRAPH => {
                    msg.graph = Some(GraphProto::decode_field(&mut field)?);
                }
                Self::OPSET_IMPORT => {
                    msg.opset_import
                        .push(OperatorSetIdProto::decode_field(&mut field)?);
                }
                Self::PRODUCER_NAME => {
                    msg.producer_name = Some(field.read_string()?);
                }
                Self::PRODUCER_VERSION => {
                    msg.producer_version = Some(field.read_string()?);
                }
                Self::METADATA_PROPS => {
                    msg.metadata_props
                        .push(StringStringEntryProto::decode_field(&mut field)?);
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

impl EncodeMessage for ModelProto {
    fn encode_fields<W: WriteValue>(
        &self,
        fields: &mut MessageWriter<W>,
    ) -> Result<(), ProtobufError> {
        if let Some(ir_version) = self.ir_version {
            fields.write_int64(Self::IR_VERSION, ir_version)?;
        }
        if let Some(producer_name) = &self.producer_name {
            fields.write_string(Self::PRODUCER_NAME, producer_name)?;
        }
        if let Some(producer_version) = &self.producer_version {
            fields.write_string(Self::PRODUCER_VERSION, producer_version)?;
        }
        if let Some(graph) = &self.graph {
            fields.write_message(Self::GRAPH, graph)?;
        }
        fields.write_repeated_message(Self::OPSET_IMPORT, &self.opset_import)?;
        fields.write_repeated_message(Self::METADATA_PROPS, &self.metadata_props)?;
        Ok(())
    }
}

/// Simplified version of [`ModelProto`] used for file type detection.
#[derive(Debug, Default)]
struct SlimModelProto {
    pub ir_version: Option<i64>,
    pub graph: bool,
}

impl DecodeMessage for SlimModelProto {
    type Types = OwnedValues;

    fn decode_fields<R: ReadValue<Types = Self::Types>>(
        mut fields: Fields<R>,
    ) -> Result<Self, ProtobufError> {
        let mut msg = Self::default();
        while let Some(mut field) = fields.next()? {
            match field.number() {
                ModelProto::IR_VERSION => {
                    msg.ir_version = Some(field.get_int64()?);
                }
                ModelProto::GRAPH => {
                    msg.graph = true;
                    field.skip()?;
                }
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
    }
}

/// Test whether a file or buffer contains an ONNX model.
///
/// ONNX models do not contain any magic bytes that would make detection simple.
/// Instead this function attempts to parse the data as a simplified version of
/// the `ModelProto` message type, testing for the presence of a few key fields
/// but skipping over the main graph.
///
/// ```
/// use rten_onnx::protobuf::ValueReader;
/// use rten_onnx::onnx::is_onnx_model;
///
/// let value_reader = ValueReader::from_buf(b"NOT AN ONNX MODEL");
/// assert!(!is_onnx_model(value_reader));
/// ```
pub fn is_onnx_model(reader: impl ReadValue<Types = OwnedValues>) -> bool {
    let Ok(model) = SlimModelProto::decode(reader) else {
        return false;
    };
    // The `ir_version` field is required, and a model without a graph is not
    // useful.
    model.ir_version.is_some() && model.graph
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs::File;
    use std::path::PathBuf;

    use super::{
        AttributeProto, AttributeType, DataLocation, DataType, Dimension, GraphProto, ModelProto,
        NodeProto, OperatorSetIdProto, StringStringEntryProto, TensorProto, TensorShapeProto,
        TypeProto, TypeProtoSequence, TypeProtoTensor, ValueInfoProto, is_onnx_model,
    };
    use crate::protobuf::{DecodeMessage, ValueReader};

    fn test_file_path(path: &str) -> PathBuf {
        let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        abs_path.push("test-data");
        abs_path.push(path);
        abs_path
    }

    // Test decoding an empty buffer. This should succeed and return a
    // default ModelProto.
    #[test]
    fn test_decode_empty_model() {
        let value_reader = ValueReader::from_buf(Vec::new());
        let model = ModelProto::decode(value_reader).unwrap();
        assert!(model.graph.is_none());
    }

    #[test]
    fn test_decode_mnist() {
        let model_path = test_file_path("mnist.onnx");
        let file = File::open(model_path).unwrap();
        let value_reader = ValueReader::from_file(file);
        let model = ModelProto::decode(value_reader).unwrap();

        let default_opset = model
            .opset_import
            .iter()
            .find(|os| os.domain.as_deref().unwrap_or_default().is_empty());
        assert_eq!(default_opset.and_then(|os| os.version), Some(18));

        let graph = model.graph.unwrap();
        assert_eq!(graph.node.len(), 13);
        assert_eq!(graph.initializer.len(), 8);

        let ops: Vec<_> = graph
            .node
            .iter()
            .map(|node| node.op_type.as_deref().unwrap_or_default())
            .filter(|op_type| *op_type != "Constant")
            .collect();
        assert_eq!(
            ops,
            &[
                "Conv",
                "Relu",
                "MaxPool",
                "Conv",
                "Relu",
                "MaxPool",
                "Conv",
                "Relu",
                "ReduceMean",
                "Reshape",
                "Gemm"
            ]
        );

        assert_eq!(graph.input.len(), 1);
        assert_eq!(graph.input[0].name.as_deref(), Some("input"));
        assert_eq!(graph.output.len(), 1);
        assert_eq!(graph.output[0].name.as_deref(), Some("logits"));
    }

    /// Create a model which uses every message type and field that this module
    /// supports.
    fn create_test_model() -> ModelProto {
        let raw_data: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let raw_init = TensorProto {
            name: Some("weights".to_string()),
            dims: vec![2, 3],
            data_type: Some(DataType::FLOAT),
            raw_data: Some(RefCell::new(raw_data)),
            ..Default::default()
        };

        // Tensor which uses the typed data fields instead of `raw_data`, and
        // references external data.
        let typed_init = TensorProto {
            name: Some("typed".to_string()),
            dims: vec![2],
            data_type: Some(DataType::FLOAT),
            float_data: vec![1.0, 2.0],
            int32_data: vec![3, -4],
            int64_data: vec![5, -6],
            double_data: vec![7.0, 8.0],
            external_data: vec![StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights.bin".to_string()),
            }],
            data_location: Some(DataLocation::EXTERNAL),
            ..Default::default()
        };

        let subgraph = GraphProto {
            node: vec![NodeProto {
                op_type: Some("Identity".to_string()),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        };

        let attrs = vec![
            AttributeProto {
                name: Some("float_attr".to_string()),
                f: Some(0.5),
                r#type: Some(AttributeType::FLOAT),
                ..Default::default()
            },
            AttributeProto {
                name: Some("int_attr".to_string()),
                i: Some(-3),
                r#type: Some(AttributeType::INT),
                ..Default::default()
            },
            AttributeProto {
                name: Some("string_attr".to_string()),
                s: Some("str_value".to_string()),
                r#type: Some(AttributeType::STRING),
                ..Default::default()
            },
            AttributeProto {
                name: Some("floats_attr".to_string()),
                floats: vec![1.0, 2.0],
                r#type: Some(AttributeType::FLOATS),
                ..Default::default()
            },
            AttributeProto {
                name: Some("ints_attr".to_string()),
                ints: vec![3, -4],
                r#type: Some(AttributeType::INTS),
                ..Default::default()
            },
            AttributeProto {
                name: Some("strings_attr".to_string()),
                strings: vec!["a".to_string(), "b".to_string()],
                ..Default::default()
            },
            AttributeProto {
                name: Some("tensor_attr".to_string()),
                t: Some(raw_init.clone()),
                ..Default::default()
            },
            AttributeProto {
                name: Some("graph_attr".to_string()),
                g: Some(subgraph),
                r#type: Some(AttributeType::GRAPH),
                ..Default::default()
            },
        ];

        let node = NodeProto {
            name: Some("test_node".to_string()),
            op_type: Some("TestOp".to_string()),
            domain: Some("test.domain".to_string()),
            input: vec!["input".to_string(), "weights".to_string()],
            output: vec!["output".to_string()],
            attribute: attrs,
        };

        let input = ValueInfoProto {
            name: Some("input".to_string()),
            r#type: Some(TypeProto {
                tensor_type: Some(TypeProtoTensor {
                    elem_type: Some(DataType::FLOAT),
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            Dimension {
                                dim_param: Some("batch".to_string()),
                                ..Default::default()
                            },
                            Dimension {
                                dim_value: Some(3),
                                ..Default::default()
                            },
                        ],
                    }),
                }),
                ..Default::default()
            }),
        };

        // Output with a sequence type, to cover `TypeProto.sequence`.
        let output = ValueInfoProto {
            name: Some("output".to_string()),
            r#type: Some(TypeProto {
                sequence: Some(Box::new(TypeProtoSequence {
                    elem_type: Some(TypeProto {
                        tensor_type: Some(TypeProtoTensor {
                            elem_type: Some(DataType::INT64),
                            shape: None,
                        }),
                        ..Default::default()
                    }),
                })),
                ..Default::default()
            }),
        };

        let graph = GraphProto {
            node: vec![node],
            initializer: vec![raw_init, typed_init],
            input: vec![input],
            output: vec![output],
            value_info: vec![ValueInfoProto {
                name: Some("intermediate".to_string()),
                r#type: None,
            }],
        };

        ModelProto {
            ir_version: Some(9),
            producer_name: Some("rten-onnx".to_string()),
            producer_version: Some("0.1.0".to_string()),
            graph: Some(graph),
            opset_import: vec![
                OperatorSetIdProto {
                    domain: Some(String::new()),
                    version: Some(18),
                },
                OperatorSetIdProto {
                    domain: Some("test.domain".to_string()),
                    version: Some(1),
                },
            ],
            metadata_props: vec![StringStringEntryProto {
                key: Some("key".to_string()),
                value: Some("value".to_string()),
            }],
        }
    }

    #[test]
    fn test_write_model() {
        let model = create_test_model();
        let buf = model.write_buf().unwrap();
        let decoded = ModelProto::parse_buf(&buf).unwrap();

        assert_eq!(decoded.ir_version, Some(9));
        assert_eq!(decoded.producer_name.as_deref(), Some("rten-onnx"));
        assert_eq!(decoded.producer_version.as_deref(), Some("0.1.0"));

        let opsets: Vec<_> = decoded
            .opset_import
            .iter()
            .map(|os| (os.domain.as_deref().unwrap(), os.version.unwrap()))
            .collect();
        assert_eq!(opsets, [("", 18), ("test.domain", 1)]);

        let metadata: Vec<_> = decoded
            .metadata_props
            .iter()
            .map(|prop| (prop.key.as_deref().unwrap(), prop.value.as_deref().unwrap()))
            .collect();
        assert_eq!(metadata, [("key", "value")]);

        let graph = decoded.graph.unwrap();

        // Check the node and its attributes.
        assert_eq!(graph.node.len(), 1);
        let node = &graph.node[0];
        assert_eq!(node.name.as_deref(), Some("test_node"));
        assert_eq!(node.op_type.as_deref(), Some("TestOp"));
        assert_eq!(node.domain.as_deref(), Some("test.domain"));
        assert_eq!(node.input, ["input", "weights"]);
        assert_eq!(node.output, ["output"]);

        let attrs = &node.attribute;
        let attr_names: Vec<_> = attrs
            .iter()
            .map(|attr| attr.name.as_deref().unwrap())
            .collect();
        assert_eq!(
            attr_names,
            [
                "float_attr",
                "int_attr",
                "string_attr",
                "floats_attr",
                "ints_attr",
                "strings_attr",
                "tensor_attr",
                "graph_attr"
            ]
        );
        assert_eq!(attrs[0].f, Some(0.5));
        assert_eq!(attrs[0].r#type, Some(AttributeType::FLOAT));
        assert_eq!(attrs[1].i, Some(-3));
        assert_eq!(attrs[2].s.as_deref(), Some("str_value"));
        assert_eq!(attrs[3].floats, [1.0, 2.0]);
        assert_eq!(attrs[4].ints, [3, -4]);
        assert_eq!(attrs[5].strings, ["a", "b"]);
        assert_eq!(attrs[6].t.as_ref().unwrap().dims, [2, 3]);

        let subgraph = attrs[7].g.as_ref().unwrap();
        assert_eq!(subgraph.node.len(), 1);
        assert_eq!(subgraph.node[0].op_type.as_deref(), Some("Identity"));
        assert_eq!(subgraph.node[0].input, ["x"]);
        assert_eq!(subgraph.node[0].output, ["y"]);

        // Check the initializer which uses `raw_data`.
        assert_eq!(graph.initializer.len(), 2);
        let raw_init = &graph.initializer[0];
        assert_eq!(raw_init.name.as_deref(), Some("weights"));
        assert_eq!(raw_init.dims, [2, 3]);
        assert_eq!(raw_init.data_type, Some(DataType::FLOAT));
        let expected_raw: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        assert_eq!(*raw_init.raw_data.as_ref().unwrap().borrow(), expected_raw);

        // Check the initializer which uses the typed data fields.
        let typed_init = &graph.initializer[1];
        assert_eq!(typed_init.float_data, [1.0, 2.0]);
        assert_eq!(typed_init.int32_data, [3, -4]);
        assert_eq!(typed_init.int64_data, [5, -6]);
        assert_eq!(typed_init.double_data, [7.0, 8.0]);
        assert_eq!(typed_init.data_location, Some(DataLocation::EXTERNAL));
        assert_eq!(typed_init.external_data.len(), 1);
        assert_eq!(typed_init.external_data[0].key.as_deref(), Some("location"));
        assert_eq!(
            typed_init.external_data[0].value.as_deref(),
            Some("weights.bin")
        );

        // Check value types and shapes.
        let input_type = graph.input[0].r#type.as_ref().unwrap();
        let tensor_type = input_type.tensor_type.as_ref().unwrap();
        assert_eq!(graph.input[0].name.as_deref(), Some("input"));
        assert_eq!(tensor_type.elem_type, Some(DataType::FLOAT));
        let dims = &tensor_type.shape.as_ref().unwrap().dim;
        assert_eq!(dims[0].dim_param.as_deref(), Some("batch"));
        assert_eq!(dims[1].dim_value, Some(3));

        let output_type = graph.output[0].r#type.as_ref().unwrap();
        let seq_elem_type = output_type
            .sequence
            .as_ref()
            .unwrap()
            .elem_type
            .as_ref()
            .unwrap();
        assert_eq!(graph.output[0].name.as_deref(), Some("output"));
        assert_eq!(
            seq_elem_type.tensor_type.as_ref().unwrap().elem_type,
            Some(DataType::INT64)
        );

        assert_eq!(graph.value_info.len(), 1);
        assert_eq!(graph.value_info[0].name.as_deref(), Some("intermediate"));
    }

    // Test that writing a model and reading it back preserves the fields
    // that this module supports.
    #[test]
    fn test_write_mnist() {
        let model_path = test_file_path("mnist.onnx");
        let file = File::open(model_path).unwrap();
        let model = ModelProto::parse_file(file).unwrap();

        let buf = model.write_buf().unwrap();
        let decoded = ModelProto::parse_buf(&buf).unwrap();

        assert_eq!(decoded.ir_version, model.ir_version);
        assert_eq!(decoded.producer_name, model.producer_name);
        assert_eq!(decoded.producer_version, model.producer_version);
        assert_eq!(decoded.opset_import.len(), model.opset_import.len());

        let graph = model.graph.as_ref().unwrap();
        let decoded_graph = decoded.graph.as_ref().unwrap();

        let ops = |graph: &GraphProto| -> Vec<String> {
            graph
                .node
                .iter()
                .map(|node| {
                    format!(
                        "{}({}) -> ({})",
                        node.op_type.as_deref().unwrap_or_default(),
                        node.input.join(", "),
                        node.output.join(", ")
                    )
                })
                .collect()
        };
        assert_eq!(ops(decoded_graph), ops(graph));

        let weights = |graph: &GraphProto| -> Vec<(String, Vec<i64>, Vec<u8>)> {
            graph
                .initializer
                .iter()
                .map(|init| {
                    (
                        init.name.clone().unwrap_or_default(),
                        init.dims.clone(),
                        init.raw_data
                            .as_ref()
                            .map(|data| data.borrow().clone())
                            .unwrap_or_default(),
                    )
                })
                .collect()
        };
        assert_eq!(weights(decoded_graph), weights(graph));

        assert_eq!(decoded.write_buf().unwrap(), buf);
    }

    #[test]
    fn test_is_onnx_model() {
        let model_path = test_file_path("mnist.onnx");
        let file = File::open(model_path).unwrap();
        let value_reader = ValueReader::from_file(file);
        assert!(is_onnx_model(value_reader));

        let value_reader = ValueReader::from_buf(vec![]);
        assert!(!is_onnx_model(value_reader));
    }
}
