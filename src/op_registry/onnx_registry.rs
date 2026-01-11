// Code in this module should be careful about whether narrowing conversions
// should error, wrap or saturate depending on the context.
#![deny(clippy::as_conversions)]

use std::cell::Cell;
use std::fmt;
use std::sync::Arc;

use rten_base::bit_set::BitSet;
use rten_base::num::LeBytes;
use rten_onnx::onnx;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::ReadOpError;
use crate::graph::Graph;
use crate::operator::Operator;
use crate::ops;
use crate::ops::{
    AccuracyLevel, BoxOrder, CoordTransformMode, DepthToSpaceMode, Direction, NearestMode, PadMode,
    Padding, ResizeMode, ScatterReduction,
};
use crate::value::{DataType, Scalar};

/// Deserialize operators from .onnx model files.
#[derive(Default)]
pub struct OnnxOpRegistry {
    /// Map from operator type (the `NodeProto.op_type` protobuf field) to
    /// deserialization function.
    ops: FxHashMap<OpId<'static>, &'static ReadOpFunction>,
}

impl OnnxOpRegistry {
    pub fn new() -> Self {
        OnnxOpRegistry {
            ops: FxHashMap::default(),
        }
    }

    /// Register the default/built-in implementation of an operator.
    pub fn register_op<Op: ReadOp + 'static>(&mut self) {
        self.register_op_with_factory(Op::id(), &Op::read_boxed);
    }

    /// Register an operator with a custom factory to deserialize it from a
    /// model file.
    fn register_op_with_factory(&mut self, id: OpId<'static>, factory: &'static ReadOpFunction) {
        self.ops.insert(id, factory);
    }

    /// Deserialize an operator from a model file using the operators in the
    /// registry.
    pub(crate) fn read_op(&self, op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> ReadOpResult {
        let op_type = op.op_type.as_deref().unwrap_or_default();
        let id = if let Some(domain) = op.domain.as_deref()
            && !domain.is_empty()
        {
            OpId::with_domain(domain, op_type)
        } else {
            OpId::new(op_type)
        };
        self.ops
            .get(&id)
            .ok_or_else(|| ReadOpError::OperatorUnavailable {
                name: Some(id.to_string()),
            })
            .and_then(|read_fn| read_fn(op, ctx))
    }

    pub fn with_all_ops() -> Self {
        let mut reg = OnnxOpRegistry::new();

        // As of 2025-10 there are 128 ops defined. Use a power of 2 with some
        // future headroom.
        reg.ops.reserve(256);

        macro_rules! register_op {
            ($op:ident) => {
                reg.register_op::<ops::$op>()
            };

            ($op:ident, feature=$feature:literal) => {
                #[cfg(feature = $feature)]
                reg.register_op::<ops::$op>();
                #[cfg(not(feature = $feature))]
                {
                    fn stub(_op: &onnx::NodeProto, _ctx: &dyn OpLoadContext) -> ReadOpResult {
                        Err(ReadOpError::FeatureNotEnabled {
                            name: stringify!($op).to_string(),
                            feature: $feature.to_string(),
                        })
                    }
                    let id = OpId::new(stringify!($op));
                    reg.register_op_with_factory(id, &stub);
                }
            };
        }

        // ai.onnx ops.
        register_op!(Abs);
        register_op!(Acos);
        register_op!(Add);
        register_op!(And);
        register_op!(ArgMax);
        register_op!(ArgMin);
        register_op!(Asin);
        register_op!(Atan);
        register_op!(AveragePool);
        register_op!(BatchNormalization);
        register_op!(Cast);
        register_op!(CastLike);
        register_op!(Ceil);
        register_op!(Clip);
        register_op!(Concat);
        register_op!(ConcatFromSequence);
        register_op!(Conv);
        register_op!(ConvInteger);
        register_op!(ConstantOfShape);
        register_op!(ConvTranspose);
        register_op!(Cos);
        register_op!(CumSum);
        register_op!(DequantizeLinear);
        register_op!(DepthToSpace);
        register_op!(Div);
        register_op!(Dropout, feature = "random");
        register_op!(DynamicQuantizeLinear);
        register_op!(Einsum);
        register_op!(Elu);
        register_op!(Equal);
        register_op!(Erf);
        register_op!(Exp);
        register_op!(Expand);
        register_op!(EyeLike);
        register_op!(Flatten);
        register_op!(Floor);
        register_op!(Gather);
        register_op!(GatherElements);
        register_op!(GatherND);
        register_op!(Gelu);
        register_op!(Gemm);
        register_op!(GlobalAveragePool);
        register_op!(GlobalMaxPool);
        register_op!(Greater);
        register_op!(GreaterOrEqual);
        register_op!(GridSample);
        register_op!(GRU);
        register_op!(HardSigmoid);
        register_op!(HardSwish);
        register_op!(Identity);
        register_op!(If);
        register_op!(InstanceNormalization);
        register_op!(IsInf);
        register_op!(IsNaN);
        register_op!(LayerNormalization);
        register_op!(LeakyRelu);
        register_op!(Less);
        register_op!(LessOrEqual);
        register_op!(Log);
        register_op!(LogSoftmax);
        register_op!(Loop);
        register_op!(LSTM);
        register_op!(MatMul);
        register_op!(MatMulInteger);
        register_op!(Max);
        register_op!(MaxPool);
        register_op!(Mean);
        register_op!(Min);
        register_op!(Mod);
        register_op!(Mul);
        register_op!(Neg);
        register_op!(NonMaxSuppression);
        register_op!(NonZero);
        register_op!(Not);
        register_op!(OneHot);
        register_op!(Or);
        register_op!(Pad);
        register_op!(Pow);
        register_op!(PRelu);
        register_op!(QuantizeLinear);
        register_op!(RMSNormalization);
        register_op!(RandomNormal, feature = "random");
        register_op!(RandomNormalLike, feature = "random");
        register_op!(RandomUniform, feature = "random");
        register_op!(RandomUniformLike, feature = "random");
        register_op!(Range);
        register_op!(Reciprocal);
        register_op!(ReduceL2);
        register_op!(ReduceMax);
        register_op!(ReduceMean);
        register_op!(ReduceMin);
        register_op!(ReduceProd);
        register_op!(ReduceSum);
        register_op!(ReduceSumSquare);
        register_op!(Relu);
        register_op!(Reshape);
        register_op!(Resize);
        register_op!(Round);
        register_op!(ScatterElements);
        register_op!(ScatterND);
        register_op!(SequenceAt);
        register_op!(SequenceConstruct);
        register_op!(SequenceEmpty);
        register_op!(SequenceErase);
        register_op!(SequenceInsert);
        register_op!(SequenceLength);
        register_op!(Shape);
        register_op!(Sigmoid);
        register_op!(Sign);
        register_op!(Sin);
        register_op!(Size);
        register_op!(Slice);
        register_op!(Softmax);
        register_op!(Softplus);
        register_op!(Split);
        register_op!(SplitToSequence);
        register_op!(Sqrt);
        register_op!(Squeeze);
        register_op!(STFT, feature = "fft");
        register_op!(Sub);
        register_op!(Sum);
        register_op!(Tan);
        register_op!(Tanh);
        register_op!(Tile);
        register_op!(TopK);
        register_op!(Transpose);
        register_op!(Trilu);
        register_op!(Unsqueeze);
        register_op!(Where);
        register_op!(Xor);

        // com.microsoft ops.
        register_op!(MatMulNBits);

        reg
    }
}

/// Identifier for an ONNX operator.
///
/// See https://onnx.ai/onnx/intro/concepts.html#list-of-available-operators-and-domains
/// and the `NodeProto` message in the ONNX Protocol Buffers schema.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpId<'a> {
    /// Reverse DNS domain.
    ///
    /// The default domain used by most standard operators is "ai.onnx".
    pub domain: &'a str,
    /// Name of the operator (eg. "MatMul").
    pub op_type: &'a str,
}

const DEFAULT_DOMAIN: &str = "ai.onnx";

impl<'a> OpId<'a> {
    /// Create an operator ID using the default domain.
    fn new(op_type: &'a str) -> Self {
        Self {
            domain: DEFAULT_DOMAIN,
            op_type,
        }
    }

    /// Create an operator ID using a custom domain.
    fn with_domain(domain: &'a str, op_type: &'a str) -> Self {
        Self { domain, op_type }
    }
}

impl fmt::Display for OpId<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.domain == DEFAULT_DOMAIN {
            write!(f, "{}", self.op_type)
        } else {
            write!(f, "{}/{}", self.domain, self.op_type)
        }
    }
}

/// Context object passed to [`ReadOp::read`] implementations.
pub trait OpLoadContext {
    /// Deserialize a graph definition.
    fn load_graph(&self, graph: &onnx::GraphProto) -> Result<Graph, ReadOpError>;
}

/// Value for an operator input created from an attribute (`onnx::AttributeProto`).
#[derive(Debug, PartialEq)]
pub enum ConstInput {
    Ints(Vec<i64>),
    Float(f32),
}

/// Result of deserializing an ONNX operator and converting it to RTen's
/// corresponding internal type.
pub struct ParsedOp<Op: Operator + Send + Sync> {
    op: Op,

    /// Tuples of (input_index, value) for operator inputs generated from
    /// attributes.
    ///
    /// This is used to handle cases where ONNX attributes have been upgraded
    /// to inputs in newer ONNX releases.
    const_inputs: Vec<(u32, ConstInput)>,

    /// Indices of unused attributes in the [`onnx::NodeProto::attribute`] field.
    unused_attrs: BitSet,
}

impl<Op: Operator + Send + Sync> From<Op> for ParsedOp<Op> {
    fn from(op: Op) -> Self {
        Self::new(op)
    }
}

impl<Op: Operator + Send + Sync> ParsedOp<Op> {
    fn new(op: Op) -> Self {
        Self {
            op,
            const_inputs: Vec::new(),
            unused_attrs: BitSet::default(),
        }
    }

    fn with_inputs(mut self, inputs: Vec<(u32, ConstInput)>) -> Self {
        self.const_inputs = inputs;
        self
    }

    fn with_unused_attrs(mut self, attrs: BitSet) -> Self {
        self.unused_attrs = attrs;
        self
    }
}

/// Type-erased version of [`ParsedOp`].
pub struct DynParsedOp {
    pub op: Arc<dyn Operator + Send + Sync>,
    pub const_inputs: Vec<(u32, ConstInput)>,
    pub unused_attrs: BitSet,
}

impl<Op: Operator + Send + Sync> From<ParsedOp<Op>> for DynParsedOp {
    fn from(val: ParsedOp<Op>) -> DynParsedOp {
        let ParsedOp {
            op,
            const_inputs,
            unused_attrs,
        } = val;
        DynParsedOp {
            op: Arc::new(op),
            const_inputs,
            unused_attrs,
        }
    }
}

type ReadOpResult = Result<DynParsedOp, ReadOpError>;

type ReadOpFunction = dyn Fn(&onnx::NodeProto, &dyn OpLoadContext) -> ReadOpResult;

pub trait ReadOp: Operator + Sized + Send + Sync {
    /// Return the operator domain and op type.
    fn id() -> OpId<'static>;

    /// Deserialize an operator from a `NodeProto` into an RTen operator.
    fn read(op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> Result<ParsedOp<Self>, ReadOpError>;

    /// Deserialize an operator into a boxed `dyn Operator`.
    ///
    /// The node's type must correspond to the result of `op_type`.
    fn read_boxed(op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> ReadOpResult {
        let op = Self::read(op, ctx)?;
        Ok(op.into())
    }
}

/// Wrapper around the attributes of an ONNX operator.
///
/// This provides methods to find attributes by name and convert them to a
/// target type. It also records which attributes have been read, to enable
/// detecting unsupported attributes.
struct Attrs<'a> {
    attrs: &'a [onnx::AttributeProto],
    unused_attrs: Cell<BitSet>,
}

impl<'a> Attrs<'a> {
    fn new(attrs: &'a [onnx::AttributeProto]) -> Self {
        // Assume there will be at most 32 attributes. If there are more, we'll
        // just ignore them.
        let n_attrs: u32 = attrs.len().min(BitSet::BITS).try_into().unwrap();
        let unused_attrs = Cell::new(BitSet::ones(n_attrs));

        Self {
            attrs,
            unused_attrs,
        }
    }

    /// Get an optional attribute.
    fn get(&self, name: &'static str) -> Option<Attr<'a>> {
        let (pos, val) = self
            .attrs
            .iter()
            .enumerate()
            .find(|(_pos, att)| att.name.as_deref() == Some(name))?;

        let mut unused_attrs = self.unused_attrs.take();
        if pos < BitSet::BITS {
            unused_attrs.delete(pos.try_into().unwrap());
            self.unused_attrs.set(unused_attrs);
        }

        Some(Attr::new(name, val))
    }

    /// Get an attribute and cast it to a given type.
    fn get_as<T>(&self, name: &'static str) -> Option<T>
    where
        Attr<'a>: Into<T>,
    {
        self.get(name).map(|v| v.into())
    }

    /// Get an attribute and cast it to an integer type.
    fn get_as_int<T: TryFrom<i64, Error = std::num::TryFromIntError>>(
        &self,
        name: &'static str,
    ) -> Result<Option<T>, ReadOpError> {
        self.get(name).map(|v| v.cast_int()).transpose()
    }

    /// Get a required attribute.
    fn require(&self, name: &'static str) -> Result<Attr<'a>, ReadOpError> {
        self.get(name)
            .ok_or_else(|| ReadOpError::attr_error(name, "required attribute missing"))
    }

    /// Return the indices of unused attributes
    fn unused_attrs(&self) -> BitSet {
        self.unused_attrs.get()
    }

    /// Check that an attribute is either unset or has the value `expected`.
    fn check_eq<T>(&self, name: &'static str, expected: T) -> Result<(), ReadOpError>
    where
        T: From<Attr<'a>> + PartialEq,
    {
        self.check(name, |val: T| val == expected)
    }

    /// Check that an attribute is either unset or matches `predicate`.
    fn check<T>(&self, name: &'static str, predicate: impl Fn(T) -> bool) -> Result<(), ReadOpError>
    where
        T: From<Attr<'a>>,
    {
        let Some(attr) = self.get(name) else {
            return Ok(());
        };
        let val = T::from(attr);
        if predicate(val) {
            Ok(())
        } else {
            Err(ReadOpError::attr_error(name, "unsupported value"))
        }
    }

    /// Check the type of an attribute and mark it as used, but ignore the value.
    ///
    /// This is useful for attributes which are redundant or only applicable
    /// at training time.
    fn check_unused<T>(&self, name: &'static str) -> Result<(), ReadOpError>
    where
        T: From<Attr<'a>>,
    {
        self.check(name, |_val: T| true)
    }
}

/// Wrapper around an ONNX attribute value.
///
/// This provides methods to extract the value of a given type.
#[derive(Copy, Clone)]
struct Attr<'a> {
    name: &'static str,
    attr: &'a onnx::AttributeProto,
}

impl<'a> Attr<'a> {
    fn new(name: &'static str, attr: &'a onnx::AttributeProto) -> Self {
        Self { name, attr }
    }

    fn as_bool(&self) -> bool {
        self.as_i64() != 0
    }

    fn as_f32(&self) -> f32 {
        self.attr.f.unwrap_or_default()
    }

    fn as_floats(&self) -> &'a [f32] {
        &self.attr.floats
    }

    fn as_i64(&self) -> i64 {
        self.attr.i.unwrap_or_default()
    }

    fn cast_int<T: TryFrom<i64, Error = std::num::TryFromIntError>>(
        &self,
    ) -> Result<T, ReadOpError> {
        self.as_i64()
            .try_into()
            .map_err(|_| ReadOpError::attr_error(self.name, "value is out of range"))
    }

    fn as_ints(&self) -> &'a [i64] {
        &self.attr.ints
    }

    fn as_str(&self) -> &'a str {
        self.attr.s.as_deref().unwrap_or_default()
    }

    fn as_strings(&self) -> &'a [String] {
        &self.attr.strings
    }

    /// Get the RTen data type which corresponds to the ONNX data type.
    ///
    /// This can fail if the ONNX data type is unsupported in RTen.
    fn as_dtype(&self) -> Result<DataType, ReadOpError> {
        let onnx_dtype = onnx::DataType(self.cast_int()?);

        // The conversions here should match those used when converting
        // initializers and value types in the ONNX model loader.
        match onnx_dtype {
            onnx::DataType::FLOAT | onnx::DataType::FLOAT16 | onnx::DataType::DOUBLE => {
                Ok(DataType::Float)
            }
            onnx::DataType::INT32 | onnx::DataType::INT64 | onnx::DataType::BOOL => {
                Ok(DataType::Int32)
            }
            onnx::DataType::INT8 => Ok(DataType::Int8),
            onnx::DataType::UINT8 => Ok(DataType::UInt8),
            _ => Err(ReadOpError::attr_error(
                self.name,
                format!("unsupported data type {onnx_dtype}"),
            )),
        }
    }

    fn as_graph(&self) -> Result<&onnx::GraphProto, ReadOpError> {
        self.attr
            .g
            .as_ref()
            .ok_or_else(|| ReadOpError::attr_error(self.name, "attribute is not a graph"))
    }

    /// Get the value of a string enum and convert it to enum type `T`.
    fn as_string_enum<T>(&self, map: impl Fn(&str) -> Option<T>) -> Result<T, ReadOpError> {
        let str_val = self.as_str();
        map(str_val).ok_or_else(|| ReadOpError::attr_error(self.name, "unsupported value"))
    }

    /// Get the value of an ints attribute, converted to type T.
    fn cast_ints<T: TryFrom<i64, Error = std::num::TryFromIntError>>(
        &self,
    ) -> Result<Vec<T>, ReadOpError> {
        self.as_ints()
            .iter()
            .map(|i| (*i).try_into())
            .collect::<Result<Vec<T>, _>>()
            .map_err(|_| ReadOpError::attr_error(self.name, "value is out of range"))
    }
}

macro_rules! impl_from_attr {
    ($into:ty, $as_fn:ident) => {
        impl<'a> From<Attr<'a>> for $into {
            fn from(val: Attr<'a>) -> Self {
                val.$as_fn()
            }
        }
    };
}

impl_from_attr!(f32, as_f32);
impl_from_attr!(&'a [f32], as_floats);
impl_from_attr!(i64, as_i64);
impl_from_attr!(&'a [i64], as_ints);
impl_from_attr!(bool, as_bool);
impl_from_attr!(&'a str, as_str);
impl_from_attr!(&'a [String], as_strings);

macro_rules! impl_read_op {
    ($op:ident) => {
        impl ReadOp for ops::$op {
            fn id() -> OpId<'static> {
                OpId::new(stringify!($op))
            }

            fn read(
                op: &onnx::NodeProto,
                _ctx: &dyn OpLoadContext,
            ) -> Result<ParsedOp<Self>, ReadOpError> {
                let attrs = Attrs::new(&op.attribute);
                Ok(ParsedOp::new(ops::$op {}).with_unused_attrs(attrs.unused_attrs()))
            }
        }
    };

    ($op:ident, $read:expr) => {
        impl ReadOp for ops::$op {
            fn id() -> OpId<'static> {
                OpId::new(stringify!($op))
            }

            fn read(
                op: &onnx::NodeProto,
                _ctx: &dyn OpLoadContext,
            ) -> Result<ParsedOp<Self>, ReadOpError> {
                let attrs = Attrs::new(&op.attribute);
                $read(&attrs).map(|op| ParsedOp::from(op).with_unused_attrs(attrs.unused_attrs()))
            }
        }
    };

    ($domain:literal, $op:ident, $read:expr) => {
        impl ReadOp for ops::$op {
            fn id() -> OpId<'static> {
                OpId::with_domain($domain, stringify!($op))
            }

            fn read(
                op: &onnx::NodeProto,
                _ctx: &dyn OpLoadContext,
            ) -> Result<ParsedOp<Self>, ReadOpError> {
                let attrs = Attrs::new(&op.attribute);
                $read(&attrs).map(|op| ParsedOp::from(op).with_unused_attrs(attrs.unused_attrs()))
            }
        }
    };
}

impl_read_op!(Abs);
impl_read_op!(Acos);
impl_read_op!(Add);
impl_read_op!(And);

struct ArgReduceAttrs {
    axis: isize,
    keep_dims: bool,
}

fn get_common_arg_reduce_attrs(attrs: &Attrs) -> Result<ArgReduceAttrs, ReadOpError> {
    attrs.check_eq("select_last_index", 0)?;

    let axis = attrs.get_as_int("axis")?.unwrap_or(0);
    let keep_dims = attrs.get_as("keepdims").unwrap_or(true);
    Ok(ArgReduceAttrs { axis, keep_dims })
}

impl_read_op!(ArgMax, |attrs: &Attrs| {
    let ArgReduceAttrs { axis, keep_dims } = get_common_arg_reduce_attrs(attrs)?;
    Ok(ops::ArgMax { axis, keep_dims })
});

impl_read_op!(ArgMin, |attrs: &Attrs| {
    let ArgReduceAttrs { axis, keep_dims } = get_common_arg_reduce_attrs(attrs)?;
    Ok(ops::ArgMin { axis, keep_dims })
});

impl_read_op!(Asin);
impl_read_op!(Atan);

impl_read_op!(AveragePool, |attrs: &Attrs| {
    let PoolAttrs {
        ceil_mode,
        kernel_size,
        padding,
        strides,
    } = get_common_pool_attrs(attrs)?;
    let count_include_pad = attrs.get_as("count_include_pad").unwrap_or(false);

    Ok(ops::AveragePool {
        ceil_mode,
        count_include_pad,
        kernel_size,
        padding,
        strides,
    })
});

impl_read_op!(BatchNormalization, |attrs: &Attrs| {
    attrs.check_eq("training_mode", 0)?;

    // Ignore attributes which are valid only if training_mode=1, which is
    // unsupported.
    attrs.check_unused::<f32>("momentum")?;

    let epsilon = attrs.get("epsilon").map(|v| v.as_f32()).unwrap_or(1e-05);
    Ok(ops::BatchNormalization { epsilon })
});

impl_read_op!(Cast, |attrs: &Attrs| {
    // The "saturate" attribute only applies to FP8, which is unsupported.
    // Conversions to other types do not saturate, even if this attribute is 1.
    attrs.check_eq("saturate", 1)?;

    let to = attrs.require("to")?.as_dtype()?;
    Ok(ops::Cast { to })
});

impl_read_op!(CastLike);
impl_read_op!(Ceil);

impl_read_op!(Clip, |attrs: &Attrs| {
    let mut const_inputs = Vec::new();

    if let Some(min) = attrs.get("min") {
        const_inputs.push((1, ConstInput::Float(min.as_f32())));
    }
    if let Some(max) = attrs.get("max") {
        const_inputs.push((2, ConstInput::Float(max.as_f32())));
    }

    Ok(ParsedOp::new(ops::Clip {}).with_inputs(const_inputs))
});

impl_read_op!(Concat, |attrs: &Attrs| {
    let axis = attrs.require("axis")?.cast_int()?;
    Ok(ops::Concat { axis })
});

impl_read_op!(ConcatFromSequence, |attrs: &Attrs| {
    let axis = attrs.require("axis")?.cast_int()?;
    let new_axis = attrs.get("new_axis").map(|v| v.as_bool()).unwrap_or(false);
    Ok(ops::ConcatFromSequence { axis, new_axis })
});

/// Read padding attributes from a convolution or pooling operator.
fn get_padding(attrs: &Attrs, n_spatial_dims: usize) -> Result<Padding, ReadOpError> {
    let auto_pad = match attrs.get_as("auto_pad") {
        // "VALID" means no padding. The "pads" attribute should be unset and
        // it will default to zeros.
        Some("NOTSET" | "VALID") | None => false,
        Some("SAME_UPPER" | "SAME_LOWER") => true,
        Some(_) => {
            return Err(ReadOpError::attr_error("auto_pad", "unsupported value"));
        }
    };
    let pads = attrs
        .get("pads")
        .map(|v| v.cast_ints())
        .transpose()?
        .unwrap_or_else(|| vec![0; n_spatial_dims * 2])
        .into();
    if auto_pad {
        Ok(Padding::Same)
    } else {
        Ok(Padding::Fixed(pads))
    }
}

struct ConvAttrs {
    dilations: Vec<usize>,
    padding: Padding,
    groups: usize,
    strides: Vec<usize>,
}

fn get_common_conv_attrs(attrs: &Attrs) -> Result<ConvAttrs, ReadOpError> {
    // nb. Spec says that spatial dims should be inferred from input if
    // `kernel_shape` attribute is not set. We don't have access to the input
    // here, so this would have to be handled by making various fields optional
    // in convolution operators.
    let n_spatial_dims = attrs
        .get("kernel_shape")
        .map(|v| v.as_ints().len())
        .unwrap_or(0);

    let dilations = attrs
        .get("dilations")
        .map(|v| v.cast_ints())
        .transpose()?
        .unwrap_or_else(|| vec![1; n_spatial_dims]);
    let padding = get_padding(attrs, n_spatial_dims)?;
    let groups = attrs.get_as_int("group")?.unwrap_or(1);

    let strides = attrs
        .get("strides")
        .map(|v| v.cast_ints())
        .transpose()?
        .unwrap_or_else(|| vec![1; n_spatial_dims]);

    Ok(ConvAttrs {
        dilations,
        padding,
        groups,
        strides,
    })
}

impl_read_op!(Conv, |attrs: &Attrs| {
    let ConvAttrs {
        dilations,
        padding,
        groups,
        strides,
    } = get_common_conv_attrs(attrs)?;

    Ok(ops::Conv {
        dilations,
        groups,
        padding,
        strides,
    })
});

impl_read_op!(ConvInteger, |attrs: &Attrs| {
    let ConvAttrs {
        dilations,
        padding,
        groups,
        strides,
    } = get_common_conv_attrs(attrs)?;

    Ok(ops::ConvInteger {
        dilations,
        groups,
        padding,
        strides,
    })
});

/// Helper for extracting scalar values from a `TensorProto`.
///
/// The data may be stored either as little-endian bytes in the `raw_data` field
/// or in one of the repeated numeric fields (`float_data`, `int32_data` etc.)
/// The numeric field may have a wider type than the tensor's data type (eg.
/// int8 values are stored in the int32_data field).
fn extract_scalar<T: Copy + LeBytes, U: Copy + TryInto<T>>(
    raw_data: Option<&[u8]>,
    typed_data: &[U],
) -> Result<T, ReadOpError> {
    if let Some(data) = raw_data
        && let Ok(bytes) = data.try_into()
    {
        Ok(T::from_le_bytes(bytes))
    } else if typed_data.len() == 1
        && let Ok(value) = typed_data[0].try_into()
    {
        Ok(value)
    } else {
        Err(ReadOpError::attr_error("value", "invalid scalar value"))
    }
}

impl_read_op!(ConstantOfShape, |attrs: &Attrs| {
    let value = attrs
        .get("value")
        .map(|attr| {
            let Some(tensor) = attr.attr.t.as_ref() else {
                return Err(ReadOpError::attr_error("value", "missing tensor value"));
            };

            let raw_data = tensor.raw_data.as_ref().map(|data| data.take());

            match tensor.data_type {
                Some(onnx::DataType::FLOAT) => {
                    let value = extract_scalar(raw_data.as_deref(), &tensor.float_data)?;
                    Ok(Scalar::Float(value))
                }
                Some(onnx::DataType::INT64) => {
                    let value: i64 = extract_scalar(raw_data.as_deref(), &tensor.int64_data)?;
                    #[allow(clippy::as_conversions)]
                    Ok(Scalar::Int(value as i32))
                }
                Some(onnx::DataType::INT32) => {
                    let value: i32 = extract_scalar(raw_data.as_deref(), &tensor.int32_data)?;
                    Ok(Scalar::Int(value))
                }
                Some(onnx::DataType::BOOL) => {
                    let value: u8 = extract_scalar(raw_data.as_deref(), &tensor.int32_data)?;
                    let value = if value != 0 { 1 } else { 0 };
                    Ok(Scalar::Int(value))
                }
                _ => Err(ReadOpError::attr_error(
                    "value",
                    "unsupported data type for ConstantOfShape",
                )),
            }
        })
        .transpose()?
        .unwrap_or(Scalar::Float(0.));

    Ok(ops::ConstantOfShape { value })
});

impl_read_op!(ConvTranspose, |attrs: &Attrs| {
    let ConvAttrs {
        dilations,
        padding,
        groups,
        strides,
    } = get_common_conv_attrs(attrs)?;

    if !dilations.iter().all(|d| *d == 1) {
        return Err(ReadOpError::attr_error("dilations", "unsupported value"));
    }

    let output_padding = attrs
        .get("output_padding")
        .map(|v| v.cast_ints())
        .transpose()?;

    Ok(ops::ConvTranspose {
        padding,
        strides,
        groups,
        output_padding,
    })
});

impl_read_op!(Cos);
impl_read_op!(CumSum, |attrs: &Attrs| {
    attrs.check_eq("exclusive", 0)?;
    attrs.check_eq("reverse", 0)?;
    Ok(ops::CumSum {})
});

impl_read_op!(DequantizeLinear, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(1);
    Ok(ops::DequantizeLinear { axis })
});

impl_read_op!(DepthToSpace, |attrs: &Attrs| {
    let mode = attrs
        .get("mode")
        .map(|v| {
            v.as_string_enum(|mode| match mode {
                "DCR" => Some(DepthToSpaceMode::DepthColumnRow),
                "CRD" => Some(DepthToSpaceMode::ColumnRowDepth),
                _ => None,
            })
        })
        .transpose()?
        .unwrap_or(DepthToSpaceMode::DepthColumnRow);
    let block_size = attrs.require("blocksize")?.cast_int()?;
    Ok(ops::DepthToSpace { mode, block_size })
});

impl_read_op!(Div);

#[cfg(feature = "random")]
impl_read_op!(Dropout, |attrs: &Attrs| {
    let seed = attrs.get_as_int("seed")?;
    Ok(ops::Dropout { seed })
});

impl_read_op!(DynamicQuantizeLinear);

impl_read_op!(Einsum, |attrs: &Attrs| {
    let equation = attrs.require("equation")?.as_str().to_string();
    Ok(ops::Einsum { equation })
});

impl_read_op!(Elu, |attrs: &Attrs| {
    let alpha = attrs.get_as("alpha").unwrap_or(1.0);
    Ok(ops::Elu { alpha })
});

impl_read_op!(Equal);
impl_read_op!(Erf);
impl_read_op!(Exp);
impl_read_op!(Expand);

impl_read_op!(EyeLike, |attrs: &Attrs| {
    let dtype = attrs.get("dtype").map(|v| v.as_dtype()).transpose()?;
    let k = attrs.get_as_int("k")?.unwrap_or(0);
    Ok(ops::EyeLike { dtype, k })
});

impl_read_op!(Flatten, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(1);
    Ok(ops::Flatten { axis })
});
impl_read_op!(Floor);

impl_read_op!(Gather, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(0);
    Ok(ops::Gather { axis })
});

impl_read_op!(GatherElements, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(0);
    Ok(ops::GatherElements { axis })
});

impl_read_op!(GatherND, |attrs: &Attrs| {
    let batch_dims = attrs.get_as_int("batch_dims")?.unwrap_or(0);
    Ok(ops::GatherND { batch_dims })
});

impl_read_op!(Gelu, |attrs: &Attrs| {
    let approximate = attrs
        .get("approximate")
        .map(|v| {
            v.as_string_enum(|approx| match approx {
                "tanh" => Some(true),
                "none" => Some(false),
                _ => None,
            })
        })
        .transpose()?
        .unwrap_or(false);
    Ok(ops::Gelu { approximate })
});

impl_read_op!(Gemm, |attrs: &Attrs| {
    let alpha = attrs.get_as("alpha").unwrap_or(1.0);
    let beta = attrs.get_as("beta").unwrap_or(1.0);
    let transpose_a = attrs.get_as("transA").unwrap_or(false);
    let transpose_b = attrs.get_as("transB").unwrap_or(false);

    Ok(ops::Gemm {
        alpha,
        beta,
        transpose_a,
        transpose_b,
    })
});

impl_read_op!(GlobalAveragePool);
impl_read_op!(GlobalMaxPool);
impl_read_op!(Greater);
impl_read_op!(GreaterOrEqual);

impl_read_op!(GridSample, |attrs: &Attrs| {
    let align_corners = attrs.get_as("align_corners").unwrap_or(false);
    attrs.check_eq("mode", "bilinear")?;
    attrs.check_eq("padding_mode", "zeros")?;

    Ok(ops::GridSample { align_corners })
});

impl_read_op!(GRU, |attrs: &Attrs| {
    let RnnAttrs {
        direction,
        hidden_size,
    } = get_common_rnn_attrs(attrs)?;

    let linear_before_reset = attrs.get_as("linear_before_reset").unwrap_or(false);
    Ok(ops::GRU {
        direction,
        hidden_size,
        linear_before_reset,
    })
});

impl_read_op!(HardSigmoid, |attrs: &Attrs| {
    let alpha = attrs.get_as("alpha").unwrap_or(0.2);
    let beta = attrs.get_as("beta").unwrap_or(0.5);
    Ok(ops::HardSigmoid { alpha, beta })
});

impl_read_op!(HardSwish);
impl_read_op!(Identity);

impl ReadOp for ops::If {
    fn id() -> OpId<'static> {
        OpId::new("If")
    }

    fn read(op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> Result<ParsedOp<Self>, ReadOpError> {
        let attrs = Attrs::new(&op.attribute);
        let then_branch = ctx.load_graph(attrs.require("then_branch")?.as_graph()?)?;
        let else_branch = ctx.load_graph(attrs.require("else_branch")?.as_graph()?)?;
        Ok(ops::If {
            then_branch,
            else_branch,
        }
        .into())
    }
}

impl_read_op!(InstanceNormalization, |attrs: &Attrs| {
    let epsilon = attrs.get_as("epsilon");
    Ok(ops::InstanceNormalization { epsilon })
});

impl_read_op!(IsInf, |attrs: &Attrs| {
    attrs.check_eq("detect_positive", 1)?;
    attrs.check_eq("detect_negative", 1)?;

    Ok(ops::IsInf {})
});
impl_read_op!(IsNaN);

impl_read_op!(LayerNormalization, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(-1);
    let epsilon = attrs.get_as("epsilon");
    attrs.check_eq("stash_type", 1)?;

    Ok(ops::LayerNormalization { axis, epsilon })
});

impl_read_op!(LeakyRelu, |attrs: &Attrs| {
    let alpha = attrs.get_as("alpha").unwrap_or(0.01);
    Ok(ops::LeakyRelu { alpha })
});

impl_read_op!(Less);
impl_read_op!(LessOrEqual);
impl_read_op!(Log);

impl_read_op!(LogSoftmax, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(-1);
    Ok(ops::LogSoftmax { axis })
});

impl ReadOp for ops::Loop {
    fn id() -> OpId<'static> {
        OpId::new("Loop")
    }

    fn read(op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> Result<ParsedOp<Self>, ReadOpError> {
        let attrs = Attrs::new(&op.attribute);
        let body = ctx.load_graph(attrs.require("body")?.as_graph()?)?;
        Ok(ops::Loop { body }.into())
    }
}

struct RnnAttrs {
    hidden_size: usize,
    direction: Direction,
}

fn get_common_rnn_attrs(attrs: &Attrs) -> Result<RnnAttrs, ReadOpError> {
    // ONNX spec does not state that hidden_size is required, but doesn't
    // provide a default. ONNX Runtime requires it to be present and non-zero.
    let hidden_size = attrs.require("hidden_size")?.cast_int()?;
    let direction = attrs
        .get("direction")
        .map(|v| {
            v.as_string_enum(|dir| match dir {
                "forward" => Some(Direction::Forward),
                "reverse" => Some(Direction::Reverse),
                "bidirectional" => Some(Direction::Bidirectional),
                _ => None,
            })
        })
        .transpose()?
        .unwrap_or(Direction::Forward);

    Ok(RnnAttrs {
        hidden_size,
        direction,
    })
}

impl_read_op!(LSTM, |attrs: &Attrs| {
    let RnnAttrs {
        direction,
        hidden_size,
    } = get_common_rnn_attrs(attrs)?;

    attrs.check("activation_alpha", |val: &[f32]| val.is_empty())?;
    attrs.check("activation_beta", |val: &[f32]| val.is_empty())?;
    attrs.check("activations", |val: &[String]| val.is_empty())?;
    attrs.check_eq("clip", 0.)?;
    attrs.check_eq("input_forget", 0)?;
    attrs.check_eq("layout", 0)?;

    Ok(ops::LSTM {
        direction,
        hidden_size,
    })
});

impl_read_op!(MatMul);
impl_read_op!(MatMulInteger);

impl_read_op!("com.microsoft", MatMulNBits, |attrs: &Attrs| {
    // Spec allows any value between 2 and 8.
    attrs.check_eq("bits", 4)?;

    // These are inferred from the inputs.
    attrs.check_unused::<i64>("block_size")?;
    attrs.check_unused::<i64>("K")?;
    attrs.check_unused::<i64>("N")?;

    // accuracy_level specifies the minimum compute accuracy. An implementation
    // may use higher. Current levels: 0 (unset), f32 (1), f16 (2), bf16 (3), i8
    // (4).
    let level = attrs.get_as("accuracy_level").unwrap_or(0);
    let accuracy_level = if level <= 3 {
        AccuracyLevel::Float
    } else {
        AccuracyLevel::Int8
    };

    let block_size = attrs.require("block_size")?.cast_int()?;

    Ok(ops::MatMulNBits {
        accuracy_level,
        bits: 4,
        block_size,
    })
});

impl_read_op!(Max);

struct PoolAttrs {
    ceil_mode: bool,
    kernel_size: SmallVec<[usize; 2]>,
    padding: Padding,
    strides: SmallVec<[usize; 2]>,
}

fn get_common_pool_attrs(attrs: &Attrs) -> Result<PoolAttrs, ReadOpError> {
    attrs.check_eq("storage_order", 0)?;
    attrs.check("dilations", |dilations: &[i64]| {
        dilations.iter().all(|d| *d == 1)
    })?;

    let ceil_mode = attrs.get_as("ceil_mode").unwrap_or(false);
    let kernel_size: SmallVec<[_; 2]> = attrs
        .get("kernel_shape")
        .map(|v| v.cast_ints())
        .transpose()?
        .unwrap_or_default()
        .into();
    let padding = get_padding(attrs, kernel_size.len())?;
    let strides = attrs
        .get("strides")
        .map(|v| v.cast_ints())
        .transpose()?
        .unwrap_or_default()
        .into();
    Ok(PoolAttrs {
        ceil_mode,
        kernel_size,
        padding,
        strides,
    })
}

impl_read_op!(MaxPool, |attrs: &Attrs| {
    let PoolAttrs {
        ceil_mode,
        kernel_size,
        padding,
        strides,
    } = get_common_pool_attrs(attrs)?;
    Ok(ops::MaxPool {
        ceil_mode,
        kernel_size,
        padding,
        strides,
    })
});

impl_read_op!(Mean);
impl_read_op!(Min);

impl_read_op!(Mod, |attrs: &Attrs| {
    let fmod = attrs.get_as("fmod").unwrap_or(false);
    Ok(ops::Mod { fmod })
});

impl_read_op!(Mul);
impl_read_op!(Neg);

impl_read_op!(NonMaxSuppression, |attrs: &Attrs| {
    let center_point_box = attrs.get_as("center_point_box").unwrap_or(false);
    let box_order = if center_point_box {
        BoxOrder::CenterWidthHeight
    } else {
        BoxOrder::TopLeftBottomRight
    };
    Ok(ops::NonMaxSuppression { box_order })
});

impl_read_op!(NonZero);
impl_read_op!(Not);

impl_read_op!(OneHot, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(-1);
    Ok(ops::OneHot { axis })
});
impl_read_op!(Or);

impl_read_op!(Pad, |attrs: &Attrs| {
    let mode = attrs
        .get("mode")
        .map(|v| {
            v.as_string_enum(|val| match val {
                "constant" => Some(PadMode::Constant),
                "reflect" => Some(PadMode::Reflect),
                "edge" => Some(PadMode::Edge),
                "wrap" => Some(PadMode::Wrap),
                _ => None,
            })
        })
        .transpose()?
        .unwrap_or(PadMode::Constant);
    Ok(ops::Pad { mode })
});

impl_read_op!(Pow);
impl_read_op!(PRelu);

impl_read_op!(QuantizeLinear, |attrs: &Attrs| {
    let output_dtype = attrs
        .get("output_dtype")
        .map(|v| v.as_dtype())
        .transpose()?;
    let axis = attrs.get_as_int("axis")?.unwrap_or(-1);
    Ok(ops::QuantizeLinear { axis, output_dtype })
});

impl_read_op!(RMSNormalization, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(-1);
    let epsilon = attrs.get_as("epsilon");
    attrs.check_eq("stash_type", 1)?;

    Ok(ops::RMSNormalization { axis, epsilon })
});

#[cfg(feature = "random")]
impl_read_op!(RandomNormal, |attrs: &Attrs| {
    attrs.check_eq("dtype", i64::from(onnx::DataType::FLOAT.0))?;

    let shape = attrs.require("shape")?.cast_ints()?;
    let mean = attrs.get_as("mean").unwrap_or(0.);
    let scale = attrs.get_as("scale").unwrap_or(1.);
    let seed = attrs.get_as("seed");
    Ok(ops::RandomNormal {
        shape,
        mean,
        scale,
        seed,
    })
});

#[cfg(feature = "random")]
impl_read_op!(RandomNormalLike, |attrs: &Attrs| {
    attrs.check_eq("dtype", i64::from(onnx::DataType::FLOAT.0))?;

    let mean = attrs.get_as("mean").unwrap_or(0.);
    let scale = attrs.get_as("scale").unwrap_or(1.);
    let seed = attrs.get_as("seed");
    Ok(ops::RandomNormalLike { mean, scale, seed })
});

#[cfg(feature = "random")]
impl_read_op!(RandomUniform, |attrs: &Attrs| {
    attrs.check_eq("dtype", i64::from(onnx::DataType::FLOAT.0))?;

    let shape = attrs.require("shape")?.cast_ints()?;
    let low = attrs.get_as("low").unwrap_or(0.);
    let high = attrs.get_as("high").unwrap_or(1.);
    let seed = attrs.get_as("seed");
    Ok(ops::RandomUniform {
        shape,
        low,
        high,
        seed,
    })
});

#[cfg(feature = "random")]
impl_read_op!(RandomUniformLike, |attrs: &Attrs| {
    attrs.check_eq("dtype", i64::from(onnx::DataType::FLOAT.0))?;

    let low = attrs.get_as("low").unwrap_or(0.);
    let high = attrs.get_as("high").unwrap_or(1.);
    let seed = attrs.get_as("seed");
    Ok(ops::RandomUniformLike { low, high, seed })
});

impl_read_op!(Range);
impl_read_op!(Reciprocal);

macro_rules! impl_read_op_for_reduce_op {
    ($op:ident) => {
        impl_read_op!($op, |attrs: &Attrs| {
            let axes = attrs.get("axes").map(|v| v.cast_ints()).transpose()?;
            let keep_dims = attrs.get_as("keepdims").unwrap_or(true);
            let noop_with_empty_axes = attrs.get_as("noop_with_empty_axes").unwrap_or(false);
            Ok(ops::$op {
                axes,
                keep_dims,
                noop_with_empty_axes,
            })
        });
    };
}

impl_read_op_for_reduce_op!(ReduceL2);
impl_read_op_for_reduce_op!(ReduceMax);
impl_read_op_for_reduce_op!(ReduceMean);
impl_read_op_for_reduce_op!(ReduceMin);
impl_read_op_for_reduce_op!(ReduceProd);
impl_read_op_for_reduce_op!(ReduceSum);
impl_read_op_for_reduce_op!(ReduceSumSquare);
impl_read_op!(Relu);

impl_read_op!(Reshape, |attrs: &Attrs| {
    let allow_zero = attrs.get_as("allowzero").unwrap_or(false);
    Ok(ops::Reshape { allow_zero })
});

impl_read_op!(Resize, |attrs: &Attrs| {
    attrs.check_eq("antialias", 0)?;

    // rten-convert treats differences from the default as a warning rather than
    // an error, but there is no mechanism to do that here.
    attrs.check_eq("cubic_coeff_a", -0.75)?;

    attrs.check_eq("exclude_outside", 0)?;
    attrs.check_eq("extrapolation_value", 0.)?;
    attrs.check_eq("keep_aspect_ratio_policy", "stretch")?;

    let mode = attrs
        .get("mode")
        .map(|v| {
            v.as_string_enum(|val| match val {
                "nearest" => Some(ResizeMode::Nearest),
                "linear" => Some(ResizeMode::Linear),
                // Cubic resize mode is not currently implemented, fall back
                // to linear. This may degrade accuracy.
                "cubic" => Some(ResizeMode::Linear),
                _ => None,
            })
        })
        .transpose()?
        .unwrap_or(ResizeMode::Nearest);
    let nearest_mode = attrs
        .get("nearest_mode")
        .map(|v| {
            v.as_string_enum(|val| match val {
                "floor" => Some(NearestMode::Floor),
                "ceil" => Some(NearestMode::Ceil),
                "round_prefer_floor" => Some(NearestMode::RoundPreferFloor),
                "round_prefer_ceil" => Some(NearestMode::RoundPreferCeil),
                _ => None,
            })
        })
        .transpose()?
        .unwrap_or(NearestMode::RoundPreferFloor);
    let coord_mode = attrs
        .get("coordinate_transformation_mode")
        .map(|v| {
            v.as_string_enum(|val| match val {
                "asymmetric" => Some(CoordTransformMode::Asymmetric),
                "half_pixel" => Some(CoordTransformMode::HalfPixel),
                "align_corners" => Some(CoordTransformMode::AlignCorners),
                "pytorch_half_pixel" => Some(CoordTransformMode::PytorchHalfPixel),
                _ => None,
            })
        })
        .transpose()?
        .unwrap_or(CoordTransformMode::HalfPixel);

    Ok(ops::Resize {
        mode,
        nearest_mode,
        coord_mode,
    })
});

impl_read_op!(Round);

fn convert_scatter_reduction(
    attr: &'static str,
    val: Option<&str>,
) -> Result<Option<ScatterReduction>, ReadOpError> {
    match val {
        None | Some("none") => Ok(None),
        Some("add") => Ok(Some(ScatterReduction::Add)),
        Some("mul") => Ok(Some(ScatterReduction::Mul)),
        Some("min") => Ok(Some(ScatterReduction::Min)),
        Some("max") => Ok(Some(ScatterReduction::Max)),
        _ => Err(ReadOpError::attr_error(attr, "unknown value")),
    }
}

impl_read_op!(ScatterElements, |attrs: &Attrs| {
    let reduction = attrs.get_as("reduction");
    let reduction = convert_scatter_reduction("reduction", reduction)?;
    let axis = attrs.get_as_int("axis")?.unwrap_or(0);
    Ok(ops::ScatterElements { axis, reduction })
});

impl_read_op!(ScatterND, |attrs: &Attrs| {
    let reduction = attrs.get_as("reduction");
    let reduction = convert_scatter_reduction("reduction", reduction)?;
    Ok(ops::ScatterND { reduction })
});

impl_read_op!(SequenceAt);
impl_read_op!(SequenceConstruct);

impl_read_op!(SequenceEmpty, |attrs: &Attrs| {
    let dtype = attrs.get("dtype").map(|v| v.as_dtype()).transpose()?;
    Ok(ops::SequenceEmpty { dtype })
});

impl_read_op!(SequenceErase);
impl_read_op!(SequenceInsert);
impl_read_op!(SequenceLength);

impl_read_op!(Shape, |attrs: &Attrs| {
    let start = attrs.get_as_int("start")?;
    let end = attrs.get_as_int("end")?;
    Ok(ops::Shape { start, end })
});

impl_read_op!(Sigmoid);
impl_read_op!(Sign);
impl_read_op!(Sin);
impl_read_op!(Size);
impl_read_op!(Slice);

impl_read_op!(Softmax, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(-1);
    Ok(ops::Softmax {
        axis,
        flush_nans_to_zero: false,
    })
});

impl_read_op!(Softplus);

impl_read_op!(Split, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(0);
    let num_outputs = attrs.get_as_int("num_outputs")?;

    let mut const_inputs = Vec::new();
    if let Some(splits) = attrs.get("split") {
        const_inputs.push((1, ConstInput::Ints(splits.as_ints().into())));
    }

    Ok(ParsedOp::new(ops::Split { axis, num_outputs }).with_inputs(const_inputs))
});

impl_read_op!(SplitToSequence, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?.unwrap_or(0);
    let keep_dims = attrs.get_as("keepdims").unwrap_or(true);
    Ok(ops::SplitToSequence { axis, keep_dims })
});

impl_read_op!(Sqrt);

impl_read_op!(Squeeze, |attrs: &Attrs| {
    let mut const_inputs = Vec::new();
    if let Some(axes) = attrs.get("axes") {
        const_inputs.push((1, ConstInput::Ints(axes.as_ints().to_vec())));
    }
    Ok(ParsedOp::new(ops::Squeeze {}).with_inputs(const_inputs))
});

#[cfg(feature = "fft")]
impl_read_op!(STFT, |attrs: &Attrs| {
    let onesided = attrs.get_as("onesided").unwrap_or(true);
    Ok(ops::STFT { onesided })
});

impl_read_op!(Sub);
impl_read_op!(Sum);
impl_read_op!(Tan);
impl_read_op!(Tanh);
impl_read_op!(Tile);

impl_read_op!(TopK, |attrs: &Attrs| {
    let axis = attrs.get_as_int("axis")?;
    let largest = attrs.get_as("largest").unwrap_or(true);
    let sorted = attrs.get_as("sorted").unwrap_or(true);
    Ok(ops::TopK {
        axis,
        largest,
        sorted,
    })
});

impl_read_op!(Transpose, |attrs: &Attrs| {
    let perm = attrs.get("perm").map(|v| v.cast_ints()).transpose()?;
    Ok(ops::Transpose { perm })
});

impl_read_op!(Trilu, |attrs: &Attrs| {
    let upper = attrs.get_as("upper").unwrap_or(true);
    Ok(ops::Trilu { upper })
});

impl_read_op!(Unsqueeze, |attrs: &Attrs| {
    let mut const_inputs = Vec::new();
    if let Some(axes) = attrs.get("axes") {
        const_inputs.push((1, ConstInput::Ints(axes.as_ints().to_vec())));
    }
    Ok(ParsedOp::new(ops::Unsqueeze {}).with_inputs(const_inputs))
});

impl_read_op!(Where);
impl_read_op!(Xor);

#[cfg(test)]
mod tests {
    use rten_onnx::onnx;
    use rten_testing::TestCases;

    use super::{ConstInput, OnnxOpRegistry, OpLoadContext, ReadOpError};
    use crate::graph::Graph;
    use crate::model::onnx_builder::{NodeProtoExt, TensorData, create_node, create_tensor};
    use crate::ops::{ArgMax, ConstantOfShape, Conv, Padding};
    use crate::value::Scalar;

    struct FakeOpLoadContext;

    impl OpLoadContext for FakeOpLoadContext {
        fn load_graph(&self, _graph: &onnx::GraphProto) -> Result<Graph, ReadOpError> {
            Ok(Graph::new())
        }
    }

    #[test]
    fn test_read_op() {
        let reg = OnnxOpRegistry::with_all_ops();

        // Supported op with no domain.
        let node = create_node("MatMul");
        let op = reg.read_op(&node, &FakeOpLoadContext).unwrap().op;
        assert_eq!(op.name(), "MatMul");

        // Supported op with empty domain.
        let node = create_node("MatMul").with_domain("");
        let op = reg.read_op(&node, &FakeOpLoadContext).unwrap().op;
        assert_eq!(op.name(), "MatMul");
    }

    #[test]
    fn test_read_op_with_attrs() {
        let reg = OnnxOpRegistry::with_all_ops();
        let node = create_node("ArgMax")
            .with_attr("axis", 1)
            .with_attr("keepdims", true);

        let op = reg.read_op(&node, &FakeOpLoadContext).unwrap().op;

        let argmax_op = op.downcast_ref::<ArgMax>().unwrap();
        assert_eq!(argmax_op.axis, 1);
        assert_eq!(argmax_op.keep_dims, true);
    }

    #[test]
    fn test_unused_attrs() {
        let reg = OnnxOpRegistry::with_all_ops();
        let node = create_node("ArgMax")
            .with_attr("axis", 1)
            .with_attr("unused_a", false)
            .with_attr("keepdims", true)
            .with_attr("unused_b", false);

        let op = reg.read_op(&node, &FakeOpLoadContext).unwrap();
        assert_eq!(op.unused_attrs.len(), 2);
        let unused_attrs: Vec<_> = op
            .unused_attrs
            .iter()
            .map(|i| node.attribute[i].name.as_deref().unwrap_or_default())
            .collect();
        assert_eq!(unused_attrs, &["unused_a", "unused_b"]);
    }

    #[test]
    fn test_read_unsupported_op() {
        // Default domain, unknown op type.
        let reg = OnnxOpRegistry::with_all_ops();
        let node = create_node("UnsupportedOp");
        let op = reg.read_op(&node, &FakeOpLoadContext);
        assert!(
            matches!(op, Err(ReadOpError::OperatorUnavailable { name }) if name == Some("UnsupportedOp".to_string()))
        );

        // Known op type, but custom domain.
        let node = create_node("MatMul").with_domain("com.foobar");
        let op = reg.read_op(&node, &FakeOpLoadContext);
        assert!(
            matches!(op, Err(ReadOpError::OperatorUnavailable { name }) if name == Some("com.foobar/MatMul".to_string()))
        );
    }

    #[test]
    fn test_conv_op_defaults() {
        let reg = OnnxOpRegistry::with_all_ops();
        let node = create_node("Conv").with_attr("kernel_shape", vec![3, 3]);

        let op = reg.read_op(&node, &FakeOpLoadContext).unwrap().op;
        let conv_op = op.downcast_ref::<Conv>().unwrap();

        assert_eq!(conv_op.padding, Padding::Fixed([0, 0, 0, 0].into()));
        assert_eq!(conv_op.strides, vec![1, 1]);
        assert_eq!(conv_op.dilations, vec![1, 1]);
    }

    #[test]
    fn test_conv_op_padding() {
        #[derive(Debug)]
        struct Case {
            kernel_shape: Vec<i64>,
            auto_pad: Option<String>,
            pads: Option<Vec<i64>>,
            expected: Padding,
        }

        let cases = [
            Case {
                kernel_shape: [3, 3].into(),
                auto_pad: Some("VALID".into()),
                pads: None,
                expected: Padding::Fixed([0, 0, 0, 0].into()),
            },
            Case {
                kernel_shape: [3, 3].into(),
                auto_pad: Some("SAME_UPPER".into()),
                pads: None,
                expected: Padding::Same,
            },
            Case {
                kernel_shape: [3, 3].into(),
                auto_pad: Some("NOTSET".into()),
                pads: Some([1, 2, 3, 4].into()),
                expected: Padding::Fixed([1, 2, 3, 4].into()),
            },
        ];

        cases.test_each(|case| {
            let reg = OnnxOpRegistry::with_all_ops();
            let mut node = create_node("Conv").with_attr("kernel_shape", case.kernel_shape.clone());
            if let Some(auto_pad) = &case.auto_pad {
                node = node.with_attr("auto_pad", auto_pad.clone());
            }
            if let Some(pads) = &case.pads {
                node = node.with_attr("pads", pads.clone());
            }

            let op = reg.read_op(&node, &FakeOpLoadContext).unwrap().op;
            let conv_op = op.downcast_ref::<Conv>().unwrap();

            assert_eq!(conv_op.padding, case.expected);
        });
    }

    #[test]
    fn test_constant_of_shape_dtypes() {
        #[derive(Debug)]
        struct Case {
            dtype: onnx::DataType,
            data: TensorData,
            expected: ConstantOfShape,
        }

        let cases = [
            // Conversions that don't alter the dtype.
            Case {
                dtype: onnx::DataType::FLOAT,
                data: TensorData::Raw(1.0f32.to_le_bytes().into()),
                expected: ConstantOfShape {
                    value: Scalar::Float(1.0),
                },
            },
            Case {
                dtype: onnx::DataType::INT32,
                data: TensorData::Raw(2i32.to_le_bytes().into()),
                expected: ConstantOfShape {
                    value: Scalar::Int(2),
                },
            },
            // Test conversions that alter the dtype.
            Case {
                dtype: onnx::DataType::BOOL,
                data: TensorData::Int([1].into()),
                expected: ConstantOfShape {
                    value: Scalar::Int(1),
                },
            },
            Case {
                dtype: onnx::DataType::BOOL,
                data: TensorData::Raw([0].into()),
                expected: ConstantOfShape {
                    value: Scalar::Int(0),
                },
            },
            Case {
                dtype: onnx::DataType::INT64,
                data: TensorData::Raw(42i64.to_le_bytes().into()),
                expected: ConstantOfShape {
                    value: Scalar::Int(42),
                },
            },
        ];

        cases.test_each(|case| {
            let reg = OnnxOpRegistry::with_all_ops();
            let tensor = create_tensor("test", &[], case.dtype, case.data.clone());
            let node = create_node("ConstantOfShape").with_attr("value", tensor);
            let op = reg.read_op(&node, &FakeOpLoadContext).unwrap();
            let cos_op = op.op.downcast_ref::<ConstantOfShape>().unwrap();
            assert_eq!(cos_op, &case.expected);
        });
    }

    #[test]
    fn test_promote_attributes() {
        #[derive(Debug)]
        struct Case {
            op: onnx::NodeProto,
            expected_inputs: Vec<(u32, ConstInput)>,
        }

        let cases = [
            Case {
                op: create_node("Clip")
                    .with_attr("min", -0.5)
                    .with_attr("max", 0.5),
                expected_inputs: [(1, ConstInput::Float(-0.5)), (2, ConstInput::Float(0.5))].into(),
            },
            Case {
                op: create_node("Squeeze").with_attr("axes", vec![-1]),
                expected_inputs: [(1, ConstInput::Ints([-1].into()))].into(),
            },
            Case {
                op: create_node("Split").with_attr("split", vec![10]),
                expected_inputs: [(1, ConstInput::Ints([10].into()))].into(),
            },
            Case {
                op: create_node("Unsqueeze").with_attr("axes", vec![-1]),
                expected_inputs: [(1, ConstInput::Ints([-1].into()))].into(),
            },
        ];

        cases.test_each_value(|case| {
            let reg = OnnxOpRegistry::with_all_ops();
            let op = reg.read_op(&case.op, &FakeOpLoadContext).unwrap();
            assert_eq!(op.const_inputs, case.expected_inputs);
        });
    }
}
