//! Traits for shape inference and common implementations.

use std::error::Error;
use std::fmt;

use smallvec::SmallVec;

pub use crate::{
    sym_expr::SymExpr,
    sym_gen::SymbolGen,
    sym_tensor::{Constant, SymTensor},
};

/// Errors when performing shape inference.
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

    /// An operator input or attribute has an invalid value.
    InvalidValue,

    /// The number of outputs could not be determined.
    UnknownOutputCount,
}

impl fmt::Display for InferShapesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncorrectInputCount => write!(f, "operator has incorrect input count"),
            Self::IncompatibleShapes => write!(f, "input shapes are incompatible"),
            Self::IncorrectRank => write!(f, "incorrect input rank"),
            Self::InvalidValue => write!(f, "input or attribute has invalid value"),
            Self::UnknownOutputCount => write!(f, "unknown output count"),
        }
    }
}

impl Error for InferShapesError {}

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
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError>;
}

/// Shape inference for unary operators.
///
/// These operators take at least one input and return a single output with
/// the same shape as the first input. Unary operators may take additional
/// inputs (eg. min/max parameters for the Clip operator) that don't affect
/// the output shape.
pub struct UnaryOp;

impl InferShapes for UnaryOp {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let Some(data) = inputs.first() else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let shape = if let Some(shape) = data.shape() {
            SymTensor::from_shape(shape.collect())
        } else {
            SymTensor::unknown("unknown input shape")
        };

        Ok([shape].into())
    }
}

/// Shape inference for binary operators.
///
/// These operators take two inputs and return an output whose shape is the
/// result of broadcasting the two input shapes together following ONNX's
/// [broadcasting rules](https://onnx.ai/onnx/repo-docs/Broadcasting.html).
pub struct BinaryOp;

impl InferShapes for BinaryOp {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [a, b] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let (Some(a_dims), Some(b_dims)) = (a.shape(), b.shape()) else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let a_pad = b_dims.len().saturating_sub(a_dims.len());
        let b_pad = a_dims.len().saturating_sub(b_dims.len());
        let mut out_shape: Vec<SymExpr> = Vec::with_capacity(a_pad + a_dims.len());

        let a_iter = std::iter::repeat_n(SymExpr::Value(1), a_pad).chain(a_dims);
        let b_iter = std::iter::repeat_n(SymExpr::Value(1), b_pad).chain(b_dims);

        for (a, b) in a_iter.zip(b_iter) {
            let dim: SymExpr = match (a, b) {
                (a, b) if a == b => a.clone(),

                // If either size is 1, it will be broadcast against the other
                // size.
                (SymExpr::Value(1), b) => b.clone(),
                (a, SymExpr::Value(1)) => a.clone(),

                // If both sizes are fixed and different, we know execution
                // will fail.
                (SymExpr::Value(_), SymExpr::Value(_)) => {
                    return Err(InferShapesError::IncompatibleShapes);
                }

                // If one dim is a fixed value other than 1 and the other
                // dim is symbolic, execution can only succeed if the symbolic
                // dim has the same size as the fixed dim.
                (SymExpr::Var(_a), SymExpr::Value(b)) => SymExpr::Value(b),
                (SymExpr::Value(a), SymExpr::Var(_b)) => SymExpr::Value(a),

                // In cases where both values are unknown, the result can be
                // either of the dimensions.
                //
                // 1. If both sizes are equal, the result can be seen as either
                //    the first or second dim.
                // 2. If only one of the sizes is 1, the result will be the other
                //    dim.
                // 3. If the sizes are different, the op will fail.
                //
                // Where the op succeeds, the result is the maximum of the LHS
                // and RHS sizes.
                (a, b) => a.broadcast(&b),
            };
            out_shape.push(dim);
        }

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Shape inference for variadic operators.
///
/// This is a generalization of [`BinaryOp`] to operators which take a variable
/// number of inputs whose shapes are broadcast against each other, using the
/// same rules.
pub struct VariadicOp;

impl InferShapes for VariadicOp {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        if inputs.is_empty() {
            return Err(InferShapesError::IncorrectInputCount);
        }

        let first_shape = inputs[0]
            .shape()
            .map(|shape| SymTensor::from_shape(shape.collect()))
            .unwrap_or_else(|| SymTensor::unknown("unknown input shape"));

        let out_shape: Result<SymTensor, InferShapesError> =
            inputs
                .iter()
                .skip(1)
                .try_fold(first_shape, |out_shape, in_shape| {
                    let mut shapes =
                        BinaryOp.infer_shapes(&[out_shape, in_shape.clone()], sym_gen)?;
                    Ok(shapes.remove(0))
                });

        Ok([out_shape?].into())
    }
}

/// Shape inference for reduction operators.
#[derive(Clone, Debug, PartialEq)]
pub struct ReductionOp<'a> {
    /// Axes over which the reduction is applied.
    ///
    /// Reduction ops take the axes as an attribute in ONNX opset <= 13 and an
    /// input in opset 18+.
    pub axes: Option<&'a [i32]>,

    /// True if the reduced dimension is retained as a 1-sized dimension in the
    /// output.
    pub keep_dims: bool,
}

impl InferShapes for ReductionOp<'_> {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        match inputs.len() {
            1 | 2 => {}
            _ => {
                return Err(InferShapesError::IncorrectInputCount);
            }
        }

        let data = &inputs[0];

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let ndim = data_dims.len();
        let mut axes: SmallVec<[usize; 4]> =
            if let Some(Constant::Vector(axes)) = inputs.get(1).and_then(|x| x.to_constant()) {
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
                out_shape.push(SymExpr::Value(1));
            }
        }

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Resolve an index given as a value in `[-len, len-1]` to a positive index in
/// `[0, len)`, or return None if the index is out of bounds.
fn resolve_index(len: usize, index: i32) -> Option<usize> {
    let len = len.min(i32::MAX as usize) as i32;
    if index < -len || index >= len {
        return None;
    }

    if index >= 0 {
        Some(index as usize)
    } else {
        Some((len + index) as usize)
    }
}

/// Resolve an axis given as a value in `[-ndim, ndim-1]` to the zero-based
/// dimension of a tensor with `ndim` dimensions.
///
/// Negative axis values count backwards from the last dimension.
pub(crate) fn resolve_axis(ndim: usize, axis: i32) -> Result<usize, InferShapesError> {
    resolve_index(ndim, axis).ok_or(InferShapesError::IncorrectRank)
}

/// Resolve a sequence of axes values in `[-ndim, ndim-1]` to zero-based dimension
/// indexes in a tensor with `ndim` dimensions.
///
/// Negative axis values count backwards from the last dimension.
fn resolve_axes<'a, I: ExactSizeIterator<Item = &'a i32>>(
    ndim: usize,
    axes: I,
) -> Result<SmallVec<[usize; 4]>, InferShapesError> {
    let mut resolved_axes = SmallVec::with_capacity(axes.len());
    for axis in axes {
        let resolved = resolve_axis(ndim, *axis)?;
        resolved_axes.push(resolved);
    }
    Ok(resolved_axes)
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{
        BinaryOp, InferShapes, InferShapesError, ReductionOp, SymExpr, SymTensor, SymbolGen,
        UnaryOp, VariadicOp,
    };
    use crate::sym_tensor::{sym_elems, sym_shape};

    #[test]
    fn test_unary_op_infer() {
        let input = sym_shape!("batch", 16, "seq", 24);
        let mut sym_gen = SymbolGen::new();
        let shape = UnaryOp
            .infer_shapes(&[input.clone()], &mut sym_gen)
            .unwrap();
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0], input);

        let err = UnaryOp.infer_shapes(&[], &mut sym_gen).err().unwrap();
        assert_eq!(err, InferShapesError::IncorrectInputCount);
    }

    #[test]
    fn test_binary_op() {
        #[derive(Debug)]
        struct Case {
            lhs: SymTensor,
            rhs: SymTensor,
            expected: SymTensor,
        }

        let cases = [
            Case {
                lhs: sym_shape!("batch"),
                rhs: sym_shape!("batch"),
                expected: sym_shape!("batch"),
            },
            Case {
                lhs: sym_shape!(2, 3),
                rhs: sym_shape!(2, 3),
                expected: sym_shape!(2, 3),
            },
            Case {
                lhs: sym_shape!(1, 5),
                rhs: sym_shape!(4, 1),
                expected: sym_shape!(4, 5),
            },
            Case {
                lhs: sym_shape!(1, 1),
                rhs: sym_shape!(1, 1),
                expected: sym_shape!(1, 1),
            },
            Case {
                lhs: sym_shape!(1, "bar"),
                rhs: sym_shape!("foo", 1),
                expected: sym_shape!("foo", "bar"),
            },
            Case {
                lhs: sym_shape!("foo"),
                rhs: sym_shape!("bar"),
                expected: sym_shape!(SymExpr::from("foo").broadcast(&SymExpr::from("bar"))),
            },
        ];

        cases.test_each(|case| {
            let mut sym_gen = SymbolGen::new();
            let shape = BinaryOp
                .infer_shapes(&[case.lhs.clone(), case.rhs.clone()], &mut sym_gen)
                .unwrap();
            assert_eq!(shape.len(), 1);
            assert_eq!(shape[0], case.expected.clone());
        });
    }

    #[test]
    fn test_binary_op_invalid() {
        #[derive(Clone, Debug)]
        struct Case {
            inputs: Vec<Vec<SymExpr>>,
            expected: InferShapesError,
        }

        let cases = [
            Case {
                inputs: [sym_elems!(5)].into(),
                expected: InferShapesError::IncorrectInputCount,
            },
            Case {
                inputs: [sym_elems!(5), sym_elems!(3)].into(),
                expected: InferShapesError::IncompatibleShapes,
            },
        ];

        cases.test_each_clone(|case| {
            let mut sym_gen = SymbolGen::new();
            let inputs: Vec<_> = case.inputs.into_iter().map(SymTensor::from_shape).collect();
            let err = BinaryOp.infer_shapes(&inputs, &mut sym_gen).err().unwrap();
            assert_eq!(err, case.expected);
        });
    }

    #[test]
    fn test_variadic_op() {
        let mut sym_gen = SymbolGen::new();
        let a = sym_shape!("batch", 4, 1, 1);
        let b = sym_shape!("batch", 1, 8, 1);
        let c = sym_shape!("batch", 1, 8, 16);

        // Single input
        let result = VariadicOp.infer_shapes(&[a.clone()], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", 4, 1, 1));

        // N inputs
        let result = VariadicOp
            .infer_shapes(&[a.clone(), b, c], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 4, 8, 16));
    }

    #[test]
    fn test_reduction_op() {
        #[derive(Clone, Debug)]
        struct Case<'a> {
            inputs: Vec<SymTensor>,
            op: ReductionOp<'a>,
            expected: Vec<SymExpr>,
        }

        let axes = vec![SymExpr::Value(1i32)];

        let default_op = ReductionOp {
            axes: None,
            keep_dims: false,
        };

        let cases = [
            // Reduce single axis
            Case {
                inputs: [
                    SymTensor::from_shape(sym_elems!("batch", 4, 5)),
                    SymTensor::from_vec(axes.clone()),
                ]
                .into(),
                op: default_op.clone(),
                expected: sym_elems!("batch", 5),
            },
            // Reduce single axis specified as an attribute
            Case {
                inputs: [SymTensor::from_shape(sym_elems!("batch", 4, 5))].into(),
                op: ReductionOp {
                    axes: Some(&[1i32]),
                    ..default_op
                },
                expected: sym_elems!("batch", 5),
            },
            // Reduce single axis with `keep_dims=true`
            Case {
                inputs: [
                    SymTensor::from_shape(sym_elems!("batch", 4, 5)),
                    SymTensor::from_vec(axes.clone()),
                ]
                .into(),
                op: ReductionOp {
                    keep_dims: true,
                    ..default_op
                },
                expected: sym_elems!("batch", 1, 5),
            },
            // Reduce all axes
            Case {
                inputs: [SymTensor::from_shape(sym_elems!(3, 4, 5))].into(),
                op: default_op.clone(),
                expected: sym_elems!(),
            },
        ];

        cases.test_each(|case| {
            let mut sym_gen = SymbolGen::new();
            let shapes = case.op.infer_shapes(&case.inputs, &mut sym_gen).unwrap();
            assert_eq!(shapes.len(), 1);
            assert_eq!(shapes[0], SymTensor::from_shape(case.expected.clone()));
        });
    }
}
