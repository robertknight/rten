//! Shape inference types and traits.
//!
//! This module also provides implementations of shape and type inference that
//! are commonly used by many operators.

use std::borrow::Cow;
use std::collections::HashMap;

use smallvec::SmallVec;

use crate::graph::{Dimension, Graph, Node, NodeId, RunError, TypedConstant};
use crate::ops::resolve_axes;
use crate::value::DataType;

#[derive(Clone, Debug, PartialEq)]
pub enum InferShapesError {
    /// Too many or too few inputs were provided for this operator.
    IncorrectInputCount,

    /// The input shapes are incompatible.
    ///
    /// Operator execution will fail if given inputs with these shapes.
    IncompatibleShapes,

    /// An input's rank does not match that expected by the operator.
    IncorrectRank,

    /// Operator has a missing optional input, which is not currently supported
    /// by shape inference.
    MissingOptionalInput,
}

/// Shape inference value where the dimension count and values are all known.
#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Scalar(i32),
    Vector(Vec<i32>),
}

impl Constant {
    fn ndim(&self) -> usize {
        match self {
            Self::Scalar(_) => 0,
            Self::Vector(_) => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SymElem {
    Value(i32),
    Var(Box<str>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SymTensor {
    Scalar(SymElem),
    Vector(Vec<SymElem>),
}

impl SymTensor {
    fn ndim(&self) -> usize {
        match self {
            Self::Scalar(_) => 0,
            Self::Vector(_) => 1,
        }
    }
}

/// A symbolic value.
///
/// This is a tensor where dimensions can have symbolic sizes and values can
/// be either unknown or symbolic. These values are used as inputs and outputs
/// for shape inference.
///
/// A symbolic value can have:
///
///  1. A concrete shape and values, such as for graph constants.
///  2. A concrete shape and symbolic values. For example a `Shape` operator
///     with an input of shape ("batch", "seq", 64) will output a vector with
///     values `["batch", "seq", 64]` where the quoted values are symbols.
///  3. A symbolic shape and unknown values, such as for model inputs.
///  4. An unknown shape and value
///
/// For cases (1) and (2), supported values are currently limited to scalars and
/// vectors of integers. Other values are represented as case (3). This is
/// because working with concrete and symbolic values is mainly needed for
/// evaluating subgraphs that manipulate tensor shapes, where the values are
/// integers.
#[derive(Clone, Debug, PartialEq)]
pub enum SymValue {
    /// Tensor with known shape and value.
    Constant(Constant),
    /// Tensor with known shape and symbolic values.
    Value(SymTensor),
    /// Tensor with known shape but unknown values.
    Shape(Vec<Dimension>),
    /// Tensor with unknown shape.
    Unknown,
}

impl SymValue {
    pub fn from_shape(shape: impl AsRef<[Dimension]>) -> Self {
        SymValue::Shape(shape.as_ref().to_vec())
    }

    pub fn from_fixed_shape(shape: &[usize]) -> Self {
        SymValue::Shape(shape.iter().copied().map(Dimension::Fixed).collect())
    }

    /// Return the number of dimensions, if known.
    pub fn ndim(&self) -> Option<usize> {
        match self {
            Self::Constant(val) => Some(val.ndim()),
            Self::Value(val) => Some(val.ndim()),
            Self::Shape(val) => Some(val.len()),
            Self::Unknown => None,
        }
    }

    /// Return the index'th dimension, if the dimensions are known and the index
    /// is valid.
    pub fn dim(&self, index: usize) -> Option<Dimension> {
        match self {
            Self::Constant(val) => match val {
                Constant::Scalar(_) => None,
                Constant::Vector(val) => {
                    if index == 0 {
                        Some(Dimension::Fixed(val.len()))
                    } else {
                        None
                    }
                }
            },
            Self::Value(sym) => match sym {
                SymTensor::Scalar(_) => None,
                SymTensor::Vector(val) => {
                    if index == 0 {
                        Some(Dimension::Fixed(val.len()))
                    } else {
                        None
                    }
                }
            },
            Self::Shape(val) => Some(val[index].clone()),
            Self::Unknown => None,
        }
    }

    /// Return an iterator over the dimensions or `None` if unknown.
    pub fn dims(&self) -> Option<impl ExactSizeIterator<Item = Dimension>> {
        let ndim = self.ndim()?;
        let dims = (0..ndim).map(|d| self.dim(d).unwrap());
        Some(dims)
    }
}

/// Generates names for symbolic dimensions.
pub struct SymbolGen {
    prefix: Cow<'static, str>,
    next_symbol_id: u32,
}

impl Default for SymbolGen {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolGen {
    pub fn new() -> Self {
        Self::with_prefix("unknown".into())
    }

    pub fn with_prefix(prefix: Cow<'static, str>) -> Self {
        Self {
            prefix,
            next_symbol_id: 0,
        }
    }

    /// Generate a new symbolic dimension.
    pub fn gen_symbol(&mut self) -> Dimension {
        self.next_symbol_id += 1;
        let name = format!("{}_{}", self.prefix, self.next_symbol_id);
        Dimension::Symbolic(name)
    }
}

/// Infer the shapes of an operator's outputs given its inputs.
pub trait InferShapes {
    /// Infer the shapes and optionally values of an operator's outputs given
    /// its inputs.
    ///
    /// The operator may need to generate new symbolic dimensions to represent
    /// dimensions that are unknown or combinations of inputs. These should be
    /// generated using `sym_gen`.
    fn infer_shapes(
        &self,
        inputs: &[SymValue],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError>;
}

/// Shared shape inference implementation for unary operators.
pub const UNARY_OP: UnaryOpInfer = UnaryOpInfer;

/// Shape inference implementation for unary operators.
///
/// These operators take a single input and return an output with the same
/// shape.
pub struct UnaryOpInfer;

impl InferShapes for UnaryOpInfer {
    fn infer_shapes(
        &self,
        inputs: &[SymValue],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        if inputs.len() != 1 {
            return Err(InferShapesError::IncorrectInputCount);
        }
        Ok(inputs.to_vec())
    }
}

/// Shared shape inference implementation for binary operators which broadcast
/// their inputs together.
pub const BINARY_OP: BinaryOpInfer = BinaryOpInfer;

/// Shape inference implementation for binary operators.
///
/// These operators take two inputs and return an output whose shape is the
/// result of broadcasting the two input shapes together.
pub struct BinaryOpInfer;

impl InferShapes for BinaryOpInfer {
    fn infer_shapes(
        &self,
        inputs: &[SymValue],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        let [a, b] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let (Some(a_dims), Some(b_dims)) = (a.dims(), b.dims()) else {
            return Ok([SymValue::Unknown].into());
        };

        let a_pad = b_dims.len().saturating_sub(a_dims.len());
        let b_pad = a_dims.len().saturating_sub(b_dims.len());
        let mut out_shape: Vec<Dimension> = Vec::with_capacity(a_pad + a_dims.len());

        let a_iter = std::iter::repeat(Dimension::Fixed(1))
            .take(a_pad)
            .chain(a_dims);
        let b_iter = std::iter::repeat(Dimension::Fixed(1))
            .take(b_pad)
            .chain(b_dims);

        for (a, b) in a_iter.zip(b_iter) {
            let dim: Dimension = match (a, b) {
                (a, b) if a == b => a.clone(),

                // If either size is 1, it will be broadcast against the other
                // size.
                (Dimension::Fixed(1), b) => b.clone(),
                (a, Dimension::Fixed(1)) => a.clone(),

                // If both sizes are fixed and different, we know execution
                // will fail.
                (Dimension::Fixed(_), Dimension::Fixed(_)) => {
                    return Err(InferShapesError::IncompatibleShapes);
                }

                // If one dim is a fixed value other than 1 and the other
                // dim is symbolic, execution can only succeed if the symbolic
                // dim has the same size as the fixed dim.
                (Dimension::Symbolic(_a), Dimension::Fixed(b)) => Dimension::Fixed(b),
                (Dimension::Fixed(a), Dimension::Symbolic(_b)) => Dimension::Fixed(a),

                // If both dimensions are symbolic, the result can be either of
                // the dimensions.
                //
                // 1. If both sizes are equal, the result can be seen as either
                //    the first or second dim.
                // 2. If only one of the sizes is 1, the result will be the other
                //    dim.
                // 3. If the sizes are different, the op will fail.
                (Dimension::Symbolic(_), Dimension::Symbolic(_)) => sym_gen.gen_symbol(),
            };
            out_shape.push(dim);
        }

        Ok([SymValue::Shape(out_shape)].into())
    }
}

/// Shape inference for ONNX `Reduce*` operators.
#[derive(Clone, Debug, PartialEq)]
pub struct ReductionOpInfer<'a> {
    /// Axes over which the reduction is applied.
    ///
    /// Reduction ops take the axes as an attribute in ONNX opset <= 13 and an
    /// input in opset 18+.
    pub axes: Option<&'a [i32]>,

    /// True if the reduced dimension is retained as a 1-sized dimension in the
    /// output.
    pub keep_dims: bool,

    pub noop_with_empty_axes: bool,
}

impl InferShapes for ReductionOpInfer<'_> {
    fn infer_shapes(
        &self,
        inputs: &[SymValue],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        match inputs.len() {
            1 | 2 => {}
            _ => {
                return Err(InferShapesError::IncorrectInputCount);
            }
        }

        let data = &inputs[0];

        let Some(data_dims) = data.dims() else {
            return Ok([SymValue::Unknown].into());
        };

        let ndim = data_dims.len();
        let mut axes: SmallVec<[usize; 4]> =
            if let Some(SymValue::Constant(Constant::Vector(axes))) = inputs.get(1) {
                resolve_axes(ndim, axes.iter()).map_err(|_| InferShapesError::IncorrectRank)?
            } else if let Some(axes) = self.axes {
                resolve_axes(ndim, axes.iter()).map_err(|_| InferShapesError::IncorrectRank)?
            } else {
                (0..ndim).collect()
            };
        axes.sort();
        axes.dedup();

        let out_ndim = if self.keep_dims {
            ndim
        } else {
            ndim - axes.len()
        };
        let mut out_shape = Vec::with_capacity(out_ndim);

        for (i, dim) in data_dims.enumerate() {
            if !axes.contains(&i) {
                out_shape.push(dim.clone());
                continue;
            } else if self.keep_dims {
                out_shape.push(Dimension::Fixed(1));
            }
        }

        Ok([SymValue::Shape(out_shape)].into())
    }
}

/// Infer the shapes of an operator's outputs given its inputs.
pub trait InferTypes {
    fn infer_types(
        &self,
        inputs: &[Option<DataType>],
    ) -> Result<Vec<Option<DataType>>, InferShapesError>;
}

/// A type inference implementation for operators which return a single output
/// whose type is the same as one of the inputs, usually the first.
pub struct SameAsInput {
    pub index: usize,
}

impl InferTypes for SameAsInput {
    fn infer_types(
        &self,
        inputs: &[Option<DataType>],
    ) -> Result<Vec<Option<DataType>>, InferShapesError> {
        let Some(input) = inputs.get(self.index) else {
            return Err(InferShapesError::IncorrectInputCount);
        };
        Ok(vec![*input])
    }
}

/// Type inference for operators which return one output, with the same type as
/// the first input.
pub const SAME_AS_FIRST_INPUT: SameAsInput = SameAsInput { index: 0 };

/// Errors that prevent shape inference from finishing.
///
/// Shape inference can still complete if errors only happen for certain nodes.
/// In that case errors for individual operators are returned in the
/// [`InferShapesResult`] value.
pub enum InferError {
    #[allow(unused)] // Currently ignored downstream
    PlanError(RunError),
}

/// Results of shape and type inference.
pub struct InferResult {
    /// Map of value node ID to inferred shape.
    pub shapes: HashMap<NodeId, Vec<Dimension>>,

    /// Map of value node ID to inferred type.
    pub types: HashMap<NodeId, DataType>,
}

/// Infer the shapes and types of operator outputs in a graph.
///
/// Returns a map of value node ID to shape.
pub fn infer_graph(graph: &Graph) -> Result<InferResult, InferError> {
    let ops = graph
        .execution_plan(graph.input_ids(), graph.output_ids(), Default::default())
        .map_err(InferError::PlanError)?;

    let mut symbolic_values: HashMap<NodeId, SymValue> = HashMap::new();
    let mut types: HashMap<NodeId, DataType> = HashMap::new();
    let mut symbol_gen = SymbolGen::new();

    'op_loop: for op_id in ops {
        let Some(Node::Operator(op)) = graph.get_node(op_id) else {
            // TODO - Return an error if the plan includes non-op nodes.
            continue;
        };

        // Perform type inference
        if let Some(type_infer) = op.operator().as_infer_types() {
            let input_types: Vec<Option<DataType>> = op
                .input_ids()
                .iter()
                .map(|&id| {
                    let id = id?;

                    if let Some(dtype) = types.get(&id) {
                        Some(*dtype)
                    } else {
                        graph.get_node(id)?.dtype()
                    }
                })
                .collect();

            let output_types = type_infer.infer_types(&input_types);

            if let Ok(output_types) = output_types {
                for (id, dtype) in op.output_ids().iter().zip(output_types) {
                    if let Some(id) = id
                        && let Some(dtype) = dtype
                    {
                        types.insert(*id, dtype);
                    }
                }
            }
        }

        // Perform shape inference
        if let Some(infer) = op.operator().as_infer_shapes() {
            let mut inputs = Vec::new();
            for input_id in op.input_ids() {
                let Some(input_id) = input_id else {
                    // TODO - Track errors for diagnostics.
                    continue 'op_loop;
                };

                match graph.get_node(*input_id) {
                    Some(Node::Constant(constant)) => {
                        if let Some(scalar) = constant.as_scalar() {
                            inputs.push(SymValue::Constant(Constant::Scalar(scalar)));
                        } else if let Some(vec) = constant.as_vector() {
                            inputs.push(SymValue::Constant(Constant::Vector(vec.to_vec())));
                        } else {
                            inputs.push(SymValue::from_fixed_shape(constant.shape()));
                        }
                    }
                    Some(Node::Value(val)) => {
                        if let Some(dims) = symbolic_values.get(input_id) {
                            inputs.push(dims.clone());
                        } else if let Some(shape) = val.shape() {
                            inputs.push(SymValue::from_shape(shape));
                        } else {
                            inputs.push(SymValue::Unknown);
                        }
                    }
                    Some(Node::Operator(_)) | None => unreachable!("invalid input ID"),
                }
            }

            let out_shapes = infer.infer_shapes(&inputs, &mut symbol_gen);

            match out_shapes {
                Ok(out_shapes) => {
                    for (out_id, out_shape) in op.output_ids().iter().zip(out_shapes) {
                        let Some(out_id) = out_id else {
                            continue;
                        };
                        symbolic_values.insert(*out_id, out_shape);
                    }
                }
                Err(_) => {
                    // TODO - Track error for diagnostics.
                }
            }
        }
    }

    // Extract shapes from symbolic values.
    let mut shapes = HashMap::with_capacity(symbolic_values.len());
    for (value_id, sym_value) in symbolic_values {
        if let Some(dims) = sym_value.dims() {
            shapes.insert(value_id, dims.collect());
        }
    }

    Ok(InferResult { shapes, types })
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{
        BINARY_OP, Constant, InferShapes, InferShapesError, ReductionOpInfer, SymValue, SymbolGen,
        UNARY_OP,
    };
    use crate::Dimension;

    macro_rules! dims {
        ($($x:expr),* $(,)?) => {
            vec![$(crate::Dimension::from($x)),*]
        };
    }
    pub(crate) use dims;

    pub(crate) fn inputs(dims: impl IntoIterator<Item = Vec<Dimension>>) -> Vec<SymValue> {
        dims.into_iter().map(SymValue::Shape).collect()
    }

    #[test]
    fn test_unary_op_infer() {
        let input = dims!("batch", 16, "seq", 24);
        let mut sym_gen = SymbolGen::new();
        let shape = UNARY_OP
            .infer_shapes(&inputs([input.clone()]), &mut sym_gen)
            .unwrap();
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0], SymValue::Shape(input.clone()));

        let err = UNARY_OP
            .infer_shapes(&inputs([input.clone(), input]), &mut sym_gen)
            .err()
            .unwrap();
        assert_eq!(err, InferShapesError::IncorrectInputCount);
    }

    #[test]
    fn test_binary_op_infer() {
        #[derive(Debug)]
        struct Case {
            lhs: Vec<Dimension>,
            rhs: Vec<Dimension>,
            expected: Vec<Dimension>,
        }

        let cases = [
            Case {
                lhs: dims!("batch"),
                rhs: dims!("batch"),
                expected: dims!("batch"),
            },
            Case {
                lhs: dims!(2, 3),
                rhs: dims!(2, 3),
                expected: dims!(2, 3),
            },
            Case {
                lhs: dims!(1, 5),
                rhs: dims!(4, 1),
                expected: dims!(4, 5),
            },
            Case {
                lhs: dims!(1, 1),
                rhs: dims!(1, 1),
                expected: dims!(1, 1),
            },
            Case {
                lhs: dims!(1, "bar"),
                rhs: dims!("foo", 1),
                expected: dims!("foo", "bar"),
            },
            Case {
                lhs: dims!("foo"),
                rhs: dims!("bar"),
                expected: dims!("unknown_1"),
            },
        ];

        cases.test_each(|case| {
            let mut sym_gen = SymbolGen::new();
            let shape = BINARY_OP
                .infer_shapes(&inputs([case.lhs.clone(), case.rhs.clone()]), &mut sym_gen)
                .unwrap();
            assert_eq!(shape.len(), 1);
            assert_eq!(shape[0], SymValue::Shape(case.expected.clone()));
        });
    }

    #[test]
    fn test_binary_op_infer_invalid() {
        #[derive(Clone, Debug)]
        struct Case {
            inputs: Vec<Vec<Dimension>>,
            expected: InferShapesError,
        }

        let cases = [
            Case {
                inputs: [dims!(5)].into(),
                expected: InferShapesError::IncorrectInputCount,
            },
            Case {
                inputs: [dims!(5), dims!(3)].into(),
                expected: InferShapesError::IncompatibleShapes,
            },
        ];

        cases.test_each_clone(|case| {
            let mut sym_gen = SymbolGen::new();
            let inputs: Vec<_> = case.inputs.into_iter().map(SymValue::Shape).collect();
            let err = BINARY_OP.infer_shapes(&inputs, &mut sym_gen).err().unwrap();
            assert_eq!(err, case.expected);
        });
    }

    #[test]
    fn test_reduction_op_infer() {
        #[derive(Clone, Debug)]
        struct Case<'a> {
            inputs: Vec<SymValue>,
            op: ReductionOpInfer<'a>,
            expected: Vec<Dimension>,
        }

        let axes = vec![1i32];

        let default_op = ReductionOpInfer {
            axes: None,
            keep_dims: false,
            noop_with_empty_axes: false,
        };

        let cases = [
            // Reduce single axis
            Case {
                inputs: [
                    SymValue::Shape(dims!("batch", 4, 5)),
                    SymValue::Constant(Constant::Vector(axes.clone())),
                ]
                .into(),
                op: default_op.clone(),
                expected: dims!("batch", 5),
            },
            // Reduce single axis specified as an attribute
            Case {
                inputs: [SymValue::Shape(dims!("batch", 4, 5))].into(),
                op: ReductionOpInfer {
                    axes: Some(&axes),
                    ..default_op
                },
                expected: dims!("batch", 5),
            },
            // Reduce single axis with `keep_dims=true`
            Case {
                inputs: [
                    SymValue::Shape(dims!("batch", 4, 5)),
                    SymValue::Constant(Constant::Vector(axes.clone())),
                ]
                .into(),
                op: ReductionOpInfer {
                    keep_dims: true,
                    ..default_op
                },
                expected: dims!("batch", 1, 5),
            },
            // Reduce all axes
            Case {
                inputs: [SymValue::Shape(dims!(3, 4, 5))].into(),
                op: default_op.clone(),
                expected: dims!(),
            },
        ];

        cases.test_each(|case| {
            let mut sym_gen = SymbolGen::new();
            let shapes = case.op.infer_shapes(&case.inputs, &mut sym_gen).unwrap();
            assert_eq!(shapes.len(), 1);
            assert_eq!(shapes[0], SymValue::Shape(case.expected.clone()));
        });
    }
}
