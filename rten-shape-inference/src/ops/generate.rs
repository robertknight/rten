use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError, resolve_axis};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

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
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let shape = inputs.require(0)?;

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
        } else if let Some(mut dims) = shape.shape()
            && dims.len() == 1
            && let Some(SymExpr::Value(out_ndim)) = dims.next()
            && let Ok(out_ndim) = usize::try_from(out_ndim)
        {
            // The rank is known, but not the shape.
            let out_shape = (0..out_ndim).map(|_| sym_gen.gen_positive()).collect();
            SymTensor::from_shape(out_shape)
        } else {
            SymTensor::unknown("unknown shape")
        };

        Ok(vec![out_shape])
    }
}

/// OneHot operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__OneHot.html>.
pub struct OneHot {
    pub axis: i32,
}

impl InferShapes for OneHot {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let indices = inputs.require(0)?;
        let depth = inputs.require(1)?;
        // `values` is required but unused for shape inference.
        inputs.require(2)?;

        let Some(indices_dims) = indices.shape() else {
            return Ok([SymTensor::unknown("unknown indices shape")].into());
        };

        let in_ndim = indices_dims.len();
        // `axis` may be in [-(ndim+1), ndim] since OneHot inserts a new
        // dimension.
        let axis =
            resolve_axis(in_ndim + 1, self.axis).map_err(|_| InferShapesError::IncorrectRank)?;

        // The depth is a scalar or vector containing one element.
        let depth_value = match depth.values() {
            Some([depth_value]) => depth_value.clone(),
            _ => sym_gen.gen_positive(),
        };

        let mut out_shape: Vec<SymExpr> = indices_dims.collect();
        out_shape.insert(axis, depth_value);

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Range operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Range.html>.
pub struct Range;

/// Maximum number of elements of a `Range` output for which the individual
/// values are computed, rather than just the length.
const MAX_RANGE_VALUES: i64 = 1024;

impl InferShapes for Range {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let start = inputs.require(0)?;
        let limit = inputs.require(1)?;
        let delta = inputs.require(2)?;

        let start = start.values().map(|v| v[0].clone());
        let limit = limit.values().map(|v| v[0].clone());
        let delta = delta.values().map(|v| v[0].clone());

        let out_value = match (start, limit, delta) {
            (
                Some(SymExpr::Value(start)),
                Some(SymExpr::Value(limit)),
                Some(SymExpr::Value(delta)),
            ) => {
                if delta == 0 {
                    return Err(InferShapesError::InvalidValue);
                }

                // Compute the element count as `max(ceil((limit - start) / delta), 0)`,
                // as specified by the ONNX spec. `i64` arithmetic avoids overflow
                // when negating `delta` or subtracting `i32` values.
                let (start, limit, delta) = (start as i64, limit as i64, delta as i64);
                let (span, step) = if delta > 0 {
                    (limit - start, delta)
                } else {
                    (start - limit, -delta)
                };
                let len = ((span + step - 1) / step).max(0);

                if len <= MAX_RANGE_VALUES {
                    // Values are between `start` and `limit`, so they fit in `i32`.
                    let values = (0..len)
                        .map(|i| SymExpr::Value((start + i * delta) as i32))
                        .collect();
                    SymTensor::from_vec(values)
                } else if let Ok(len) = i32::try_from(len) {
                    SymTensor::from_shape(vec![SymExpr::Value(len)])
                } else {
                    SymTensor::from_shape(vec![sym_gen.gen_positive()])
                }
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

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::{ConstantOfShape, OneHot, Range};

    #[test]
    fn test_constant_of_shape() {
        let mut sym_gen = SymbolGen::new();

        // Scalar shape, int value.
        let shape = sym_vec!();
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar(1.into()));

        // Vector shape, int value.
        let shape = sym_vec!(3);
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!(1, 1, 1));

        // Vector shape, non-int value.
        let shape = sym_vec!(3);
        let op = ConstantOfShape { value: None };
        let result = op.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(3));

        // 2D+ shape
        let shape = sym_vec!(2, 2);
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(2, 2));

        // Shape with unknown values but a known length.
        let mut sym_gen = SymbolGen::new();
        let shape = sym_shape!(3);
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", "unknown_2", "unknown_3"));

        // Shape with unknown values and a symbolic length.
        let shape = sym_shape!("n");
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0].ndim(), None);

        // Unknown shape input.
        let shape = SymTensor::unknown("unknown");
        let op = ConstantOfShape { value: Some(1) };
        let result = op.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0].ndim(), None);
    }

    #[test]
    fn test_one_hot() {
        let mut sym_gen = SymbolGen::new();

        // Insert depth axis at the end with a fixed depth.
        let indices = sym_shape!("batch", 8);
        let depth = SymTensor::from_scalar(10.into());
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: -1 };
        let result = op
            .infer_shapes([indices, depth, values].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 8, 10));

        // Insert depth axis at the start.
        let indices = sym_shape!("batch", 8);
        let depth = SymTensor::from_scalar(10.into());
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: 0 };
        let result = op
            .infer_shapes([indices, depth, values].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(10, "batch", 8));

        // Symbolic depth value.
        let indices = sym_shape!(4);
        let depth = SymTensor::from_scalar(SymExpr::from("d"));
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: -1 };
        let result = op
            .infer_shapes([indices, depth, values].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(4, "d"));

        // Depth as a rank-1 tensor containing exactly one element.
        let indices = sym_shape!("batch", 8);
        let depth = sym_vec!(10);
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: -1 };
        let result = op
            .infer_shapes([indices, depth, values].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 8, 10));

        // Unknown indices shape.
        let indices = SymTensor::unknown("unknown");
        let depth = SymTensor::from_scalar(10.into());
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: -1 };
        let result = op
            .infer_shapes([indices, depth, values].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);
    }

    #[test]
    fn test_range() {
        struct Case {
            start: SymTensor,
            limit: SymTensor,
            delta: SymTensor,
            expected: Result<SymTensor, InferShapesError>,
        }

        let cases = [
            // Range with fixed values
            Case {
                start: sym_vec!(0),
                limit: sym_vec!(5),
                delta: sym_vec!(1),
                expected: Ok(sym_vec!(0, 1, 2, 3, 4)),
            },
            // Fixed values with a delta that doesn't evenly divide the range
            Case {
                start: sym_vec!(0),
                limit: sym_vec!(5),
                delta: sym_vec!(2),
                expected: Ok(sym_vec!(0, 2, 4)),
            },
            // Fixed values with a negative delta
            Case {
                start: sym_vec!(15),
                limit: sym_vec!(-1),
                delta: sym_vec!(-1),
                expected: Ok(SymTensor::from_vec(
                    (0..=15).rev().map(SymExpr::Value).collect(),
                )),
            },
            Case {
                start: sym_vec!(10),
                limit: sym_vec!(0),
                delta: sym_vec!(-3),
                expected: Ok(sym_vec!(10, 7, 4, 1)),
            },
            // Empty ranges
            Case {
                start: sym_vec!(5),
                limit: sym_vec!(5),
                delta: sym_vec!(1),
                expected: Ok(sym_vec!()),
            },
            Case {
                start: sym_vec!(0),
                limit: sym_vec!(5),
                delta: sym_vec!(-1),
                expected: Ok(sym_vec!()),
            },
            // A zero delta is invalid, as it would produce an infinite range.
            Case {
                start: sym_vec!(0),
                limit: sym_vec!(5),
                delta: sym_vec!(0),
                expected: Err(InferShapesError::InvalidValue),
            },
            // Ranges longer than `MAX_RANGE_VALUES` have only their length inferred.
            Case {
                start: sym_vec!(0),
                limit: sym_vec!(5000),
                delta: sym_vec!(1),
                expected: Ok(sym_shape!(5000)),
            },
            Case {
                start: sym_vec!(5000),
                limit: sym_vec!(0),
                delta: sym_vec!(-1),
                expected: Ok(sym_shape!(5000)),
            },
            // A range whose length exceeds `i32::MAX`.
            Case {
                start: sym_vec!(i32::MIN),
                limit: sym_vec!(i32::MAX),
                delta: sym_vec!(1),
                expected: Ok(sym_shape!("unknown_1")),
            },
            // Range from 0..limit
            Case {
                start: sym_vec!(0),
                limit: sym_vec!("limit"),
                delta: sym_vec!(1),
                expected: Ok(sym_shape!("limit")),
            },
            // Range from start..(start + limit)
            Case {
                start: sym_vec!("start"),
                limit: sym_vec!(SymExpr::from("start") + SymExpr::from("limit")),
                delta: sym_vec!(1),
                expected: Ok(sym_shape!("limit")),
            },
            // Range from start..limit
            Case {
                start: sym_vec!("start"),
                limit: sym_vec!("limit"),
                delta: sym_vec!(1),
                expected: Ok(sym_shape!(SymExpr::from("limit") - SymExpr::from("start"))),
            },
            // Range of unknown size
            Case {
                start: sym_vec!("start"),
                limit: sym_vec!("end"),
                delta: sym_vec!("delta"),
                expected: Ok(sym_shape!("unknown_1")),
            },
        ];

        for Case {
            start,
            limit,
            delta,
            expected,
        } in cases
        {
            let mut sym_gen = SymbolGen::new();
            let result = Range
                .infer_shapes([start, limit, delta].into(), &mut sym_gen)
                .map(|mut outputs| outputs.remove(0));
            assert_eq!(result, expected);
        }
    }
}
