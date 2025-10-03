use std::cell::RefCell;
use std::sync::Arc;

use rten_base::num::LeBytes;
use rten_onnx::onnx;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::ReadOpError;
use crate::graph::Graph;
use crate::ops;
use crate::ops::{
    BoxOrder, CoordTransformMode, DepthToSpaceMode, Direction, NearestMode, Operator, PadMode,
    Padding, ResizeMode, ScatterReduction,
};
use crate::value::{DataType, Scalar};

/// Deserialize operators from .onnx model files.
#[derive(Default)]
pub struct OnnxOpRegistry {
    /// Map from operator type (the `NodeProto.op_type` protobuf field) to
    /// deserialization function.
    ops: FxHashMap<&'static str, Box<ReadOpFunction>>,
}

impl OnnxOpRegistry {
    pub fn new() -> Self {
        OnnxOpRegistry {
            ops: FxHashMap::default(),
        }
    }

    /// Register the default/built-in implementation of an operator.
    pub fn register_op<Op: ReadOp + 'static>(&mut self) {
        self.register_op_with_factory(
            Op::op_type(),
            Box::new(|op: &onnx::NodeProto, ctx: &dyn OpLoadContext| Op::read_boxed(op, ctx)),
        );
    }

    /// Register a stub implementation of an operator.
    ///
    /// This registers stubs for an operator that is not available because
    /// necessary crate features were not enabled. The purpose of the stub is
    /// to generate a more helpful error message.
    #[allow(unused)]
    pub(crate) fn register_stub_op(&mut self, op_type: &'static str, feature: &'static str) {
        self.register_op_with_factory(
            op_type,
            Box::new(move |_op, _ctx| {
                Err(ReadOpError::FeatureNotEnabled {
                    name: op_type.to_string(),
                    feature: feature.to_string(),
                })
            }),
        );
    }

    /// Register an operator with a custom factory to deserialize it from a
    /// model file.
    fn register_op_with_factory(&mut self, op_type: &'static str, factory: Box<ReadOpFunction>) {
        self.ops.insert(op_type, factory);
    }

    /// Deserialize an operator from a model file using the operators in the
    /// registry.
    #[allow(unused)] // ONNX model loading is not implemented yet.
    pub(crate) fn read_op(&self, op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> ReadOpResult {
        let op_type = op.op_type.as_deref().unwrap_or_default();
        self.ops
            .get(op_type)
            .ok_or_else(|| ReadOpError::OperatorUnavailable {
                name: op.op_type.as_ref().map(|s| s.to_string()),
            })
            .and_then(|read_fn| read_fn(op, ctx))
    }

    pub fn with_all_ops() -> Self {
        let mut reg = OnnxOpRegistry::new();

        macro_rules! register_op {
            ($op:ident) => {
                reg.register_op::<ops::$op>()
            };

            ($op:ident, feature=$feature:literal) => {
                #[cfg(feature = $feature)]
                reg.register_op::<ops::$op>();
                #[cfg(not(feature = $feature))]
                reg.register_stub_op(stringify!($op), $feature);
            };
        }

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

        reg
    }
}

/// Context object passed to [`ReadOp::read`] implementations.
pub trait OpLoadContext {
    /// Deserialize a graph definition.
    fn load_graph(&self, graph: &onnx::GraphProto) -> Result<Graph, ReadOpError>;
}

type ReadOpResult = Result<Arc<dyn Operator + Send + Sync>, ReadOpError>;

type ReadOpFunction = dyn Fn(&onnx::NodeProto, &dyn OpLoadContext) -> ReadOpResult;

pub trait ReadOp: Operator + Sized + Send + Sync {
    /// Return the operator name from the `NodeProto.op_type` field.
    fn op_type() -> &'static str;

    /// Deserialize an operator from a `NodeProto` into an RTen operator.
    fn read(op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError>;

    /// Deserialize an operator into a boxed `dyn Operator`.
    ///
    /// The node's type must correspond to the result of `op_type`.
    fn read_boxed(op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> ReadOpResult {
        let op = Self::read(op, ctx)?;
        Ok(Arc::new(op))
    }
}

/// Wrapper around the attributes of an ONNX operator.
///
/// This provides methods to find attributes by name and convert them to a
/// target type. It also records which attributes have been read, to enable
/// detecting unsupported attributes.
struct Attrs<'a> {
    attrs: &'a [onnx::AttributeProto],
    used_attrs: RefCell<SmallVec<[&'static str; 6]>>,
}

impl<'a> Attrs<'a> {
    fn new(attrs: &'a [onnx::AttributeProto]) -> Self {
        Self {
            attrs,
            used_attrs: RefCell::new(Default::default()),
        }
    }

    /// Get an optional attribute.
    fn get(&self, name: &'static str) -> Option<Attr<'a>> {
        self.used_attrs.borrow_mut().push(name);
        let val = self
            .attrs
            .iter()
            .find(|att| att.name.as_deref() == Some(name))?;
        Some(Attr::new(name, val))
    }

    /// Get an attribute and cast it to a given type.
    fn get_as<T>(&self, name: &'static str) -> Option<T>
    where
        Attr<'a>: Into<T>,
    {
        self.get(name).map(|v| v.into())
    }

    /// Get a required attribute.
    fn require(&self, name: &'static str) -> Result<Attr<'a>, ReadOpError> {
        self.get(name)
            .ok_or_else(|| ReadOpError::attr_error(name, "required attribute missing"))
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

    fn as_i64(&self) -> i64 {
        self.attr.i.unwrap_or_default()
    }

    fn as_ints(&self) -> &'a [i64] {
        &self.attr.ints
    }

    fn as_str(&self) -> &'a str {
        self.attr.s.as_deref().unwrap_or_default()
    }

    /// Get the RTen data type which corresponds to the ONNX data type.
    ///
    /// This can fail if the ONNX data type is unsupported in RTen.
    fn as_dtype(&self) -> Result<DataType, ReadOpError> {
        let onnx_dtype = onnx::DataType(self.as_i64() as i32);
        match onnx_dtype {
            onnx::DataType::FLOAT => Ok(DataType::Float),
            onnx::DataType::INT32 | onnx::DataType::INT64 | onnx::DataType::BOOL => {
                Ok(DataType::Int32)
            }
            _ => Err(ReadOpError::attr_error(self.name, "unsupported data type")),
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

    /// Get the value of an ints attribute.
    ///
    /// TODO: Report error if values are out of range.
    fn as_usize_ints(&self) -> Vec<usize> {
        self.as_ints().iter().map(|i| *i as usize).collect()
    }

    /// Get the value of an ints attribute.
    ///
    /// TODO: Report error if values are out of range.
    fn as_i32_ints(&self) -> Vec<i32> {
        self.as_ints().iter().map(|i| *i as i32).collect()
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
impl_from_attr!(i64, as_i64);
impl_from_attr!(&'a [i64], as_ints);
impl_from_attr!(bool, as_bool);
impl_from_attr!(&'a str, as_str);

macro_rules! impl_read_op {
    ($op:ident) => {
        impl ReadOp for ops::$op {
            fn op_type() -> &'static str {
                stringify!($op)
            }

            fn read(_op: &onnx::NodeProto, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
                // TODO - Check for unsupported attributes.
                Ok(ops::$op {})
            }
        }
    };

    ($op:ident, $read:expr) => {
        impl ReadOp for ops::$op {
            fn op_type() -> &'static str {
                stringify!($op)
            }

            fn read(op: &onnx::NodeProto, _ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
                // TODO - Check for unsupported attributes.
                let attrs = Attrs::new(&op.attribute);
                $read(&attrs)
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

fn get_common_arg_reduce_attrs(attrs: &Attrs) -> ArgReduceAttrs {
    let axis = attrs
        .get_as::<i64>("axis")
        .map(|val| val as isize)
        .unwrap_or(0);
    let keep_dims = attrs.get_as("keepdims").unwrap_or(true);
    ArgReduceAttrs { axis, keep_dims }
}

impl_read_op!(ArgMax, |attrs: &Attrs| {
    let ArgReduceAttrs { axis, keep_dims } = get_common_arg_reduce_attrs(attrs);
    Ok(ops::ArgMax { axis, keep_dims })
});

impl_read_op!(ArgMin, |attrs: &Attrs| {
    let ArgReduceAttrs { axis, keep_dims } = get_common_arg_reduce_attrs(attrs);
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
    } = get_common_pool_attrs(attrs);
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
    let epsilon = attrs.get("epsilon").map(|v| v.as_f32()).unwrap_or(1e-05);
    Ok(ops::BatchNormalization { epsilon })
});

impl_read_op!(Cast, |attrs: &Attrs| {
    let to = attrs.require("to")?.as_dtype()?;
    Ok(ops::Cast { to })
});

impl_read_op!(CastLike);
impl_read_op!(Ceil);
impl_read_op!(Clip);

impl_read_op!(Concat, |attrs: &Attrs| {
    let axis = attrs.require("axis")?.as_i64() as isize;
    Ok(ops::Concat { axis })
});

impl_read_op!(ConcatFromSequence, |attrs: &Attrs| {
    let axis = attrs.require("axis")?.as_i64() as i32;
    let new_axis = attrs.get("new_axis").map(|v| v.as_bool()).unwrap_or(false);
    Ok(ops::ConcatFromSequence { axis, new_axis })
});

struct ConvAttrs {
    dilations: Vec<usize>,
    padding: Padding,
    groups: usize,
    strides: Vec<usize>,
}

fn get_common_conv_attrs(attrs: &Attrs) -> ConvAttrs {
    attrs.get("kernel_shape"); // Ignored

    let dilations = attrs
        .get("dilations")
        .map(|v| v.as_usize_ints())
        .unwrap_or_default();

    // TODO - Padding should default to 0 along each spatial axis.
    let pads = attrs
        .get("pads")
        .map(|v| v.as_usize_ints())
        .unwrap_or_default();
    let padding = Padding::Fixed(pads.into());
    let groups = attrs.get("group").map(|v| v.as_i64()).unwrap_or(1);
    let groups = groups as usize; // TODO - Handle invalid values

    // TODO - This should default to [1] x spatial axis size.
    let strides = attrs
        .get("strides")
        .map(|v| v.as_usize_ints())
        .unwrap_or_default();

    ConvAttrs {
        dilations,
        padding,
        groups,
        strides,
    }
}

impl_read_op!(Conv, |attrs: &Attrs| {
    let ConvAttrs {
        dilations,
        padding,
        groups,
        strides,
    } = get_common_conv_attrs(attrs);

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
    } = get_common_conv_attrs(attrs);

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
                    Ok(Scalar::Int(value as i32))
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
        dilations: _,
        padding,
        groups,
        strides,
    } = get_common_conv_attrs(attrs);

    let output_padding = attrs.get("output_padding").map(|v| v.as_usize_ints());

    Ok(ops::ConvTranspose {
        padding,
        strides,
        groups,
        output_padding,
    })
});

impl_read_op!(Cos);
impl_read_op!(CumSum);

impl_read_op!(DequantizeLinear, |attrs: &Attrs| {
    let axis = attrs
        .get("axis")
        .map(|val| val.as_i64() as isize)
        .unwrap_or(1);
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
    let block_size = attrs.require("blocksize")?.as_i64() as u32;
    Ok(ops::DepthToSpace { mode, block_size })
});

impl_read_op!(Div);

#[cfg(feature = "random")]
impl_read_op!(Dropout, |attrs: &Attrs| {
    let seed = attrs.get_as("seed").map(|val: i64| val as i32);
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
    let k = attrs.get_as("k").map(|val: i64| val as i32).unwrap_or(0);
    Ok(ops::EyeLike { dtype, k })
});

impl_read_op!(Flatten, |attrs: &Attrs| {
    let axis = attrs.get_as("axis").unwrap_or(1) as isize;
    Ok(ops::Flatten { axis })
});
impl_read_op!(Floor);

impl_read_op!(Gather, |attrs: &Attrs| {
    let axis = attrs.get_as("axis").unwrap_or(0) as isize;
    Ok(ops::Gather { axis })
});

impl_read_op!(GatherElements, |attrs: &Attrs| {
    let axis = attrs.get_as("axis").unwrap_or(0) as isize;
    Ok(ops::GatherElements { axis })
});

impl_read_op!(GatherND, |attrs: &Attrs| {
    let batch_dims = attrs.get_as("batch_dims").unwrap_or(0) as usize;
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
impl_read_op!(Greater);
impl_read_op!(GreaterOrEqual);

impl_read_op!(GridSample, |attrs: &Attrs| {
    let align_corners = attrs.get_as("align_corners").unwrap_or(false);
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
    fn op_type() -> &'static str {
        "If"
    }

    fn read(op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
        let attrs = Attrs::new(&op.attribute);
        let then_branch = ctx.load_graph(attrs.require("then_branch")?.as_graph()?)?;
        let else_branch = ctx.load_graph(attrs.require("else_branch")?.as_graph()?)?;
        Ok(ops::If {
            then_branch,
            else_branch,
        })
    }
}

impl_read_op!(InstanceNormalization, |attrs: &Attrs| {
    let epsilon = attrs.get_as("epsilon");
    Ok(ops::InstanceNormalization { epsilon })
});

impl_read_op!(IsInf);
impl_read_op!(IsNaN);

impl_read_op!(LayerNormalization, |attrs: &Attrs| {
    let axis = attrs
        .get_as("axis")
        .map(|val: i64| val as isize)
        .unwrap_or(-1);
    let epsilon = attrs.get_as("epsilon");
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
    let axis = attrs
        .get_as("axis")
        .map(|val: i64| val as isize)
        .unwrap_or(-1);
    Ok(ops::LogSoftmax { axis })
});

impl ReadOp for ops::Loop {
    fn op_type() -> &'static str {
        "Loop"
    }

    fn read(op: &onnx::NodeProto, ctx: &dyn OpLoadContext) -> Result<Self, ReadOpError> {
        let attrs = Attrs::new(&op.attribute);
        let body = ctx.load_graph(attrs.require("body")?.as_graph()?)?;
        Ok(ops::Loop { body })
    }
}

struct RnnAttrs {
    hidden_size: usize,
    direction: Direction,
}

fn get_common_rnn_attrs(attrs: &Attrs) -> Result<RnnAttrs, ReadOpError> {
    // ONNX spec does not state that hidden_size is required, but doesn't
    // provide a default. ONNX Runtime requires it to be present and non-zero.
    let hidden_size = attrs.require("hidden_size")?.as_i64() as usize;
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

    Ok(ops::LSTM {
        direction,
        hidden_size,
    })
});

impl_read_op!(MatMul);
impl_read_op!(MatMulInteger);
impl_read_op!(Max);

struct PoolAttrs {
    ceil_mode: bool,
    kernel_size: SmallVec<[usize; 2]>,
    padding: Padding,
    strides: SmallVec<[usize; 2]>,
}

fn get_common_pool_attrs(attrs: &Attrs) -> PoolAttrs {
    let ceil_mode = attrs.get_as("ceil_mode").unwrap_or(false);
    let kernel_size = attrs
        .get("kernel_shape")
        .map(|v| v.as_usize_ints())
        .unwrap_or_default()
        .into();
    let pads = attrs
        .get("pads")
        .map(|v| v.as_usize_ints())
        .unwrap_or_default()
        .into();
    let padding = Padding::Fixed(pads);
    let strides = attrs
        .get("strides")
        .map(|v| v.as_usize_ints())
        .unwrap_or_default()
        .into();
    PoolAttrs {
        ceil_mode,
        kernel_size,
        padding,
        strides,
    }
}

impl_read_op!(MaxPool, |attrs: &Attrs| {
    let PoolAttrs {
        ceil_mode,
        kernel_size,
        padding,
        strides,
    } = get_common_pool_attrs(attrs);
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
    let axis = attrs
        .get_as("axis")
        .map(|val: i64| val as isize)
        .unwrap_or(-1);
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
    let axis = attrs
        .get_as("axis")
        .map(|val: i64| val as isize)
        .unwrap_or(-1);
    Ok(ops::QuantizeLinear { axis, output_dtype })
});

#[cfg(feature = "random")]
impl_read_op!(RandomNormal, |attrs: &Attrs| {
    let shape = attrs
        .require("shape")?
        .as_ints()
        .iter()
        .map(|val| *val as usize)
        .collect();
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
    let mean = attrs.get_as("mean").unwrap_or(0.);
    let scale = attrs.get_as("scale").unwrap_or(1.);
    let seed = attrs.get_as("seed");
    Ok(ops::RandomNormalLike { mean, scale, seed })
});

#[cfg(feature = "random")]
impl_read_op!(RandomUniform, |attrs: &Attrs| {
    let shape = attrs
        .require("shape")?
        .as_ints()
        .iter()
        .map(|val| *val as usize)
        .collect();
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
            let axes = attrs.get("axes").map(|v| v.as_i32_ints());
            let keep_dims = attrs.get_as("keepdims").unwrap_or(true);
            Ok(ops::$op { axes, keep_dims })
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
    let mode = attrs
        .get("mode")
        .map(|v| {
            v.as_string_enum(|val| match val {
                "nearest" => Some(ResizeMode::Nearest),
                "linear" => Some(ResizeMode::Linear),
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
    let axis = attrs
        .get_as("axis")
        .map(|val: i64| val as isize)
        .unwrap_or(0);
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
    let start = attrs.get_as("start").map(|i: i64| i as i32);
    let end = attrs.get_as("end").map(|i: i64| i as i32);
    Ok(ops::Shape { start, end })
});

impl_read_op!(Sigmoid);
impl_read_op!(Sign);
impl_read_op!(Sin);
impl_read_op!(Size);
impl_read_op!(Slice);

impl_read_op!(Softmax, |attrs: &Attrs| {
    let axis = attrs
        .get_as("axis")
        .map(|val: i64| val as isize)
        .unwrap_or(-1);
    Ok(ops::Softmax { axis })
});

impl_read_op!(Softplus);

impl_read_op!(Split, |attrs: &Attrs| {
    let axis = attrs
        .get_as("axis")
        .map(|val: i64| val as isize)
        .unwrap_or(0);
    let num_outputs = attrs.get_as("num_outputs").map(|val: i64| val as u32);
    Ok(ops::Split { axis, num_outputs })
});

impl_read_op!(SplitToSequence, |attrs: &Attrs| {
    let axis = attrs.get_as("axis").map(|val: i64| val as i32).unwrap_or(0);
    let keep_dims = attrs.get_as("keepdims").unwrap_or(true);
    Ok(ops::SplitToSequence { axis, keep_dims })
});

impl_read_op!(Sqrt);
impl_read_op!(Squeeze);

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
    let axis = attrs.get_as("axis").map(|val: i64| val as isize);
    let largest = attrs.get_as("largest").unwrap_or(true);
    let sorted = attrs.get_as("sorted").unwrap_or(true);
    Ok(ops::TopK {
        axis,
        largest,
        sorted,
    })
});

impl_read_op!(Transpose, |attrs: &Attrs| {
    let perm = attrs.get("perm").map(|v| v.as_usize_ints());
    Ok(ops::Transpose { perm })
});

impl_read_op!(Trilu, |attrs: &Attrs| {
    let upper = attrs.get_as("upper").unwrap_or(true);
    Ok(ops::Trilu { upper })
});

impl_read_op!(Unsqueeze);
impl_read_op!(Where);
impl_read_op!(Xor);

#[cfg(test)]
mod tests {
    use rten_onnx::onnx;

    use super::{OnnxOpRegistry, OpLoadContext, ReadOpError};
    use crate::graph::Graph;
    use crate::ops::ArgMax;

    struct FakeOpLoadContext;

    impl OpLoadContext for FakeOpLoadContext {
        fn load_graph(&self, _graph: &onnx::GraphProto) -> Result<Graph, ReadOpError> {
            Ok(Graph::new())
        }
    }

    #[derive(Clone)]
    enum AttrValue {
        Bool(bool),
        Int(i64),
    }

    fn create_attr(name: &str, value: AttrValue) -> onnx::AttributeProto {
        let mut attr = onnx::AttributeProto::default();
        attr.name = Some(name.to_string());
        match value {
            AttrValue::Bool(val) => attr.i = Some(val as i64),
            AttrValue::Int(val) => attr.i = Some(val),
        }
        attr
    }

    fn create_node(op_type: &str, attrs: &[(&str, AttrValue)]) -> onnx::NodeProto {
        let mut node = onnx::NodeProto::default();
        node.op_type = Some(op_type.to_string());
        for (name, val) in attrs {
            node.attribute.push(create_attr(name, val.clone()));
        }
        node
    }

    #[test]
    fn test_read_op() {
        let reg = OnnxOpRegistry::with_all_ops();
        let node = create_node("MatMul", &[]);

        let op = reg.read_op(&node, &FakeOpLoadContext).unwrap();

        assert_eq!(op.name(), "MatMul")
    }

    #[test]
    fn test_read_op_with_attrs() {
        let reg = OnnxOpRegistry::with_all_ops();
        let node = create_node(
            "ArgMax",
            &[
                ("axis", AttrValue::Int(1)),
                ("keepdims", AttrValue::Bool(true)),
            ],
        );

        let op = reg.read_op(&node, &FakeOpLoadContext).unwrap();

        let argmax_op = op.downcast_ref::<ArgMax>().unwrap();
        assert_eq!(argmax_op.axis, 1);
        assert_eq!(argmax_op.keep_dims, true);
    }
}
