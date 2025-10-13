//! ONNX model Protocol Buffers types.
//!
//! The types in this module correspond to Protocol Buffers messages defined
//! in [onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3).
//! See the `.proto` file for detailed information on each type and field.
//!
//! These types are not complete. They only contain messages and fields which
//! are used by RTen or its associated tools.

use std::cell::RefCell;

use crate::protobuf::{DecodeMessage, Fields, OwnedValues, ProtobufError, ReadValue};

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

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
pub struct NodeProto {
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
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
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
    pub const INT32: Self = Self(6);
    pub const INT64: Self = Self(7);
    pub const UINT8: Self = Self(2);
    pub const INT8: Self = Self(3);
    pub const BOOL: Self = Self(9);
    pub const DOUBLE: Self = Self(11);
}

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
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

#[derive(Default)]
pub struct ModelProto {
    pub ir_version: Option<i64>,
    pub graph: Option<GraphProto>,
}

impl ModelProto {
    const IR_VERSION: u64 = 1;
    const GRAPH: u64 = 7;
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
                _ => {
                    field.skip()?;
                }
            }
        }
        Ok(msg)
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
    use std::fs::File;
    use std::path::PathBuf;

    use super::{ModelProto, is_onnx_model};
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
