//! Shape inference for various ONNX operators.
//!
//! See the [ONNX operator reference](https://onnx.ai/onnx/operators/index.html)
//! for operator details.

use crate::infer_shapes::{BinaryOp, InferShapes, InferShapesError, resolve_axis};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::{Constant, SymTensor};

mod binary;
mod concat;
mod conv_pool;
mod layout;
mod matmul;
mod pad;
mod resize;
mod slice;
mod split;
mod unary;

pub use binary::{Add, Div, Equal, Mul, Sub};
pub use concat::{Concat, Tile};
pub use conv_pool::{Conv, ConvTranspose, GlobalPool, Padding, Pool};
pub use layout::{Expand, Flatten, Reshape, Shape, Size, Squeeze, Transpose, Unsqueeze};
pub use matmul::{Gemm, MatMul, MatMulNBits};
pub use pad::Pad;
pub use resize::Resize;
pub use slice::Slice;
pub use split::Split;
pub use unary::Neg;

/// ConstantOfShape operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html>.
pub struct ConstantOfShape {
    /// The integer value. This should be set to `None` if the operator has
    /// a value attribute of a different type.
    pub value: Option<i32>,
}

impl InferShapes for ConstantOfShape {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [shape] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let out_shape = if let Some(values) = shape.values() {
            if let Some(val) = self.value
                && values.len() <= 1
            {
                if let Some(vec_len) = values.first() {
                    match vec_len {
                        &SymExpr::Value(vec_len) => {
                            if let Ok(vec_len) = vec_len.try_into() {
                                SymTensor::from_vec(vec![SymExpr::Value(val); vec_len])
                            } else {
                                return Err(InferShapesError::InvalidValue);
                            }
                        }
                        SymExpr::Var(_)
                        | SymExpr::Neg(_)
                        | SymExpr::Add(..)
                        | SymExpr::Sub(..)
                        | SymExpr::Mul(..)
                        | SymExpr::Div(..)
                        | SymExpr::DivCeil(..)
                        | SymExpr::Max(..)
                        | SymExpr::Min(..)
                        | SymExpr::Broadcast(..) => SymTensor::from_shape(vec![vec_len.clone()]),
                    }
                } else {
                    SymTensor::from_scalar(SymExpr::Value(val))
                }
            } else {
                SymTensor::from_shape(values.to_vec())
            }
        } else if let Some(dims) = shape.shape() {
            let out_shape = (0..dims.len()).map(|_| sym_gen.gen_positive()).collect();
            SymTensor::from_shape(out_shape)
        } else {
            SymTensor::unknown("unknown shape")
        };

        Ok(vec![out_shape])
    }
}

/// DynamicQuantizeLinear operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html>.
pub struct DynamicQuantizeLinear;

impl InferShapes for DynamicQuantizeLinear {
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

        let scale_shape = SymTensor::from_shape(vec![]);
        let zero_point_shape = SymTensor::from_shape(vec![]);
        Ok([shape, scale_shape, zero_point_shape].into())
    }
}

/// Gather operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Gather.html>.
pub struct Gather {
    pub axis: i32,
}

impl InferShapes for Gather {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, indices] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(mut data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown data shape")].into());
        };

        let data_ndim = data_dims.len();
        let axis = resolve_axis(data_ndim, self.axis)?;

        fn get<T: Clone>(vec: &[T], index: i32) -> Result<T, InferShapesError> {
            let index: usize = index
                .try_into()
                .map_err(|_| InferShapesError::IncorrectRank)?;
            vec.get(index)
                .cloned()
                .ok_or(InferShapesError::IncorrectRank)
        }

        // If the input is a symbolic value and indices are concrete the output
        // is a symbolic value. For example `Gather<axis=0>(Shape(X), 0)` returns
        // a symbolic scalar that is the size of the first dimension of X.
        //
        // Otherwise we do standard shape inference and return a symbolic shape.
        let value = if let Some(sym_vec) = data.values()
            && let Some(indices) = indices.to_constant()
        {
            match indices {
                Constant::Vector(idxs) => {
                    let values = idxs
                        .iter()
                        .map(|idx| get(sym_vec, *idx))
                        .collect::<Result<Vec<_>, _>>()?;
                    SymTensor::from_vec(values)
                }
                Constant::Scalar(idx) => SymTensor::from_scalar(get(sym_vec, idx)?),
            }
        } else if let Some(index_dims) = indices.shape() {
            let mut out_shape = Vec::with_capacity(data_dims.len() + index_dims.len() - 1);
            out_shape.extend(data_dims.by_ref().take(axis));
            out_shape.extend(index_dims);
            out_shape.extend(data_dims.skip(1));
            SymTensor::from_shape(out_shape)
        } else {
            SymTensor::unknown("unknown indices shape")
        };

        Ok([value].into())
    }
}

/// GatherElements operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__GatherElements.html>.
pub struct GatherElements;

impl InferShapes for GatherElements {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [_data, indices] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let shape = if let Some(dims) = indices.shape() {
            SymTensor::from_shape(dims.collect())
        } else {
            SymTensor::unknown("unknown indices shape")
        };

        Ok([shape].into())
    }
}

/// NonZero operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__NonZero.html>.
pub struct NonZero;

impl InferShapes for NonZero {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        // Output is a 2D tensor of shape `(input.ndim(), num_nonzero)`.
        let first_dim = data
            .ndim()
            .map(|n| SymExpr::Value(n as i32))
            .unwrap_or_else(|| sym_gen.gen_positive());
        let out_shape = vec![first_dim, sym_gen.gen_positive()];

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Operator which produces a tensor of a fixed shape.
pub struct FixedShape<'a> {
    pub shape: &'a [usize],
}

impl InferShapes for FixedShape<'_> {
    fn infer_shapes(
        &self,
        _inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        Ok([SymTensor::from_fixed_shape(self.shape)].into())
    }
}

/// Range operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Range.html>.
pub struct Range;

impl InferShapes for Range {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [start, limit, delta] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let start = start.values().map(|v| v[0].clone());
        let limit = limit.values().map(|v| v[0].clone());
        let delta = delta.values().map(|v| v[0].clone());

        let out_value = match (start, limit, delta) {
            (
                Some(SymExpr::Value(start)),
                Some(SymExpr::Value(limit)),
                Some(SymExpr::Value(delta)),
            ) => {
                let mut values = Vec::new();
                let mut val = start;
                while val < limit {
                    values.push(SymExpr::Value(val));
                    val += delta;
                }
                SymTensor::from_vec(values)
            }
            // Range(0, limit, 1) has shape [limit]
            (Some(SymExpr::Value(0)), Some(limit), Some(SymExpr::Value(1))) => {
                SymTensor::from_shape(vec![limit])
            }
            // Range(start, start + limit, 1) has shape [limit]
            (Some(start), Some(SymExpr::Add(limit_lhs, limit_rhs)), Some(SymExpr::Value(1)))
                if start == *limit_lhs =>
            {
                SymTensor::from_shape(vec![(*limit_rhs).clone()])
            }
            // Range(start, limit, 1) has shape [limit - start]
            (Some(start), Some(limit), Some(SymExpr::Value(1))) => {
                SymTensor::from_shape(vec![limit - start])
            }
            _ => SymTensor::from_shape(vec![sym_gen.gen_positive()]),
        };

        Ok(vec![out_value])
    }
}

/// TopK operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__TopK.html>.
pub struct TopK {
    pub axis: Option<i32>,
}

impl InferShapes for TopK {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, k] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(data_dims) = data.shape() else {
            return Ok([
                SymTensor::unknown("unknown input shape"),
                SymTensor::unknown("unknown input shape"),
            ]
            .into());
        };

        let ndim = data_dims.len();
        let axis = resolve_axis(ndim, self.axis.unwrap_or(-1))
            .map_err(|_| InferShapesError::IncorrectRank)?;

        // `k` is a 1D tensor with one element.
        let k_val = k
            .as_vector()
            .and_then(|v| v.first().cloned())
            .unwrap_or_else(|| sym_gen.gen_positive());

        let mut out_shape: Vec<SymExpr> = data_dims.collect();
        out_shape[axis] = k_val;

        let shape = SymTensor::from_shape(out_shape);
        Ok([shape.clone(), shape].into())
    }
}

/// Where operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Where.html>.
pub struct Where;

impl InferShapes for Where {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [cond, x, y] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        if let Some(cond_vals) = cond.values()
            && let Some(x_vals) = x.values()
            && let Some(y_vals) = y.values()
        {
            let len = cond_vals.len().max(x_vals.len()).max(y_vals.len());

            let cs = cond_vals.iter().cycle().take(len);
            let xs = x_vals.iter().cycle().take(len);
            let ys = y_vals.iter().cycle().take(len);

            let vals: Option<Vec<SymExpr>> = cs
                .zip(xs.zip(ys))
                .map(|(cond, (x, y))| {
                    let cond_bool = match cond {
                        SymExpr::Value(v) => Some(*v == 1),
                        SymExpr::Var(_)
                        | SymExpr::Neg(_)
                        | SymExpr::Add(..)
                        | SymExpr::Sub(..)
                        | SymExpr::Mul(..)
                        | SymExpr::Div(..)
                        | SymExpr::DivCeil(..)
                        | SymExpr::Max(..)
                        | SymExpr::Min(..)
                        | SymExpr::Broadcast(..) => None,
                    }?;
                    if cond_bool {
                        Some(x.clone())
                    } else {
                        Some(y.clone())
                    }
                })
                .collect();
            if let Some(vals) = vals {
                return Ok([SymTensor::from_vec(vals)].into());
            }
        }

        // Broadcast the first two inputs together, then broadcast the result
        // against the last input.
        let cond_x = BinaryOp
            .infer_shapes(&[cond.clone(), x.clone()], sym_gen)?
            .remove(0);
        BinaryOp.infer_shapes(&[cond_x, y.clone()], sym_gen)
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::{
        ConstantOfShape, DynamicQuantizeLinear, FixedShape, Gather, GatherElements, NonZero, Range,
        TopK, Where,
    };

    #[test]
    fn test_constant_of_shape() {
        let mut sym_gen = SymbolGen::new();

        // Scalar shape, int value.
        let shape = sym_vec!();
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes(&[shape], &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar(1.into()));

        // Vector shape, int value.
        let shape = sym_vec!(3);
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes(&[shape], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!(1, 1, 1));

        // Vector shape, non-int value.
        let shape = sym_vec!(3);
        let op = ConstantOfShape { value: None };
        let result = op.infer_shapes(&[shape], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(3));

        // 2D+ shape
        let shape = sym_vec!(2, 2);
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes(&[shape], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(2, 2));
    }

    #[test]
    fn test_dynamic_quantize_linear() {
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!(32, 32);
        let result = DynamicQuantizeLinear
            .infer_shapes(&[data], &mut sym_gen)
            .unwrap();
        assert_eq!(result, &[sym_shape!(32, 32), sym_shape!(), sym_shape!(),]);
    }

    #[test]
    fn test_gather() {
        let mut sym_gen = SymbolGen::new();

        // Gather scalar from symbolic vec.
        let shape = sym_vec!("batch", 16, "seq");
        let indices = SymTensor::from_scalar(2.into());
        let op = Gather { axis: 0 };
        let result = op.infer_shapes(&[shape, indices], &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar("seq".into()));

        // Gather vector from symbolic vec.
        let shape = sym_vec!("batch", 16, "seq");
        let indices = sym_vec!(0, 2);
        let op = Gather { axis: 0 };
        let result = op.infer_shapes(&[shape, indices], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!("batch", "seq"));

        // Gather with 2D data and symbolic vec indices
        let data = sym_shape!("vocab", "embed");
        let indices = sym_vec!(1, 2, 3);
        let op = Gather { axis: 0 };
        let result = op.infer_shapes(&[data, indices], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(3, "embed"));

        // Gather with 2D data and symbolic shape indices
        let data = sym_shape!("vocab", "embed");
        let indices = sym_shape!("n_tokens");
        let op = Gather { axis: 0 };
        let result = op.infer_shapes(&[data, indices], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("n_tokens", "embed"));
    }

    #[test]
    fn test_gather_elements() {
        let mut sym_gen = SymbolGen::new();

        // Output shape = indices shape.
        let data = sym_shape!(4, 3, 2);
        let indices = sym_shape!(2, 3, 2);
        let result = GatherElements
            .infer_shapes(&[data, indices], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(2, 3, 2));

        // Unknown indices shape.
        let data = sym_shape!(4, 3, 2);
        let indices = SymTensor::unknown("unknown");
        let result = GatherElements
            .infer_shapes(&[data, indices], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);
    }

    #[test]
    fn test_fixed_shape() {
        let mut sym_gen = SymbolGen::new();
        let op = FixedShape { shape: &[2, 3, 4] };
        let result = op.infer_shapes(&[], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(2, 3, 4));

        // Zero-dim shape produces a scalar tensor.
        let op = FixedShape { shape: &[] };
        let result = op.infer_shapes(&[], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!());
    }

    #[test]
    fn test_non_zero() {
        let mut sym_gen = SymbolGen::new();

        // Known input shape, output is 2D with first dim = ndim.
        let data = sym_shape!("batch", 16, 32);
        let result = NonZero.infer_shapes(&[data], &mut sym_gen).unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], SymExpr::Value(3));
        assert!(matches!(shape[1], SymExpr::Var(_)));

        // Unknown input shape, output is still 2D.
        let data = SymTensor::unknown("unknown");
        let result = NonZero.infer_shapes(&[data], &mut sym_gen).unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_range() {
        let mut sym_gen = SymbolGen::new();

        // Range with fixed values
        let start = sym_vec!(0);
        let limit = sym_vec!(5);
        let delta = sym_vec!(1);
        let result = Range
            .infer_shapes(&[start, limit, delta], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_vec!(0, 1, 2, 3, 4));

        // Range from 0..limit
        let start = sym_vec!(0);
        let limit = sym_vec!("limit");
        let delta = sym_vec!(1);
        let result = Range
            .infer_shapes(&[start, limit, delta], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("limit"));

        // Range from start..(start + limit)
        let start = sym_vec!("start");
        let limit = sym_vec!(SymExpr::from("start") + SymExpr::from("limit"));
        let delta = sym_vec!(1);
        let result = Range
            .infer_shapes(&[start, limit, delta], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("limit"));

        // Range from start..limit
        let start = sym_vec!("start");
        let limit = sym_vec!("limit");
        let delta = sym_vec!(1);
        let result = Range
            .infer_shapes(&[start, limit, delta], &mut sym_gen)
            .unwrap();
        assert_eq!(
            result[0],
            sym_shape!(SymExpr::from("limit") - SymExpr::from("start"))
        );

        // Range of unknown size
        let start = sym_vec!("start");
        let limit = sym_vec!("end");
        let delta = sym_vec!("delta");
        let result = Range
            .infer_shapes(&[start, limit, delta], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1"));
    }

    #[test]
    fn test_top_k() {
        let mut sym_gen = SymbolGen::new();

        // Default axis (-1) with known K.
        let data = sym_shape!("batch", 16, 32);
        let k = sym_vec!(5);
        let op = TopK { axis: None };
        let result = op.infer_shapes(&[data, k], &mut sym_gen).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], sym_shape!("batch", 16, 5));
        assert_eq!(result[1], sym_shape!("batch", 16, 5));

        // Explicit axis.
        let data = sym_shape!("batch", 16, 32);
        let k = sym_vec!(5);
        let op = TopK { axis: Some(0) };
        let result = op.infer_shapes(&[data, k], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(5, 16, 32));

        // Symbolic K value.
        let data = sym_shape!("batch", 32);
        let k = sym_vec!(SymExpr::from("k"));
        let op = TopK { axis: Some(-1) };
        let result = op.infer_shapes(&[data, k], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", "k"));
    }

    #[test]
    fn test_where() {
        let mut sym_gen = SymbolGen::new();

        // Where op with symbolic vectors.
        let cond = sym_vec!(0, 1, 0, 1);
        let x = sym_vec!(1, 2, 3, 4);
        let y = sym_vec!("foo", "bar", "baz", "meep");
        let result = Where.infer_shapes(&[cond, x, y], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!("foo", 2, "baz", 4));

        // Where op with shapes.
        //
        // This broadcasts the three inputs together.
        let cond = sym_shape!(1, 16, 1);
        let x = sym_shape!(8, 16, 1);
        let y = sym_shape!(1, 16, 24);
        let result = Where.infer_shapes(&[cond, x, y], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(8, 16, 24));
    }
}
