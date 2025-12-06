//! Shape inference for various ONNX operators.
//!
//! See the [ONNX operator reference](https://onnx.ai/onnx/operators/index.html)
//! for operator details.

use crate::infer_shapes::{InferShapes, InferShapesError, resolve_axis};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::{Constant, SymElem, SymTensor};

/// Concat operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Concat.html>.
pub struct Concat {
    pub axis: i32,
}

impl InferShapes for Concat {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [first, rest @ ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(first_dims) = first.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let axis = resolve_axis(first_dims.len(), self.axis)
            .map_err(|_| InferShapesError::IncorrectRank)?;

        // If input is a constant or symbolic vector, return a constant or
        // symbolic vector by concatenating each input.
        if axis == 0 && inputs.iter().all(|inp| inp.values().is_some()) {
            let value = {
                let mut values = Vec::new();
                for inp in inputs {
                    values.extend(inp.values().expect("should have values").to_vec());
                }
                SymTensor::from_vec(values)
            };
            return Ok([value].into());
        }

        let mut out_shape: Vec<_> = first_dims.collect();

        for input in rest {
            if let Some(dim) = input.shape().and_then(|mut dims| dims.nth(axis)) {
                out_shape[axis] += dim;
            } else {
                out_shape[axis] += sym_gen.gen_positive();
            }
        }

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

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
                        &SymElem::Value(vec_len) => {
                            if let Ok(vec_len) = vec_len.try_into() {
                                SymTensor::from_vec(vec![SymElem::Value(val); vec_len])
                            } else {
                                return Err(InferShapesError::InvalidValue);
                            }
                        }
                        SymElem::Var(_) | SymElem::Add(_) | SymElem::Mul(_) | SymElem::Max(_) => {
                            SymTensor::from_shape(vec![vec_len.clone()])
                        }
                    }
                } else {
                    SymTensor::from_scalar(SymElem::Value(val))
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
                Some(SymElem::Value(start)),
                Some(SymElem::Value(limit)),
                Some(SymElem::Value(delta)),
            ) => {
                let mut values = Vec::new();
                let mut val = start;
                while val < limit {
                    values.push(SymElem::Value(val));
                    val += delta;
                }
                SymTensor::from_vec(values)
            }
            // Range(0, limit, 1) has shape [limit]
            (Some(SymElem::Value(0)), Some(limit), Some(SymElem::Value(1))) => {
                SymTensor::from_shape(vec![limit])
            }
            // Range(start, start + limit, 1) has shape [limit]
            (Some(start), Some(SymElem::Add((limit_lhs, limit_rhs))), Some(SymElem::Value(1)))
                if start == *limit_lhs =>
            {
                SymTensor::from_shape(vec![(*limit_rhs).clone()])
            }
            _ => SymTensor::from_shape(vec![sym_gen.gen_positive()]),
        };

        Ok(vec![out_value])
    }
}

/// Unsqueeze operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Unsqueeze.html>.
pub struct Unsqueeze;

impl InferShapes for Unsqueeze {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, axes] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        // If data is a constant or symbolic scalar and axes is 0, the output
        // will be a length-1 vector. We can't handle higher-rank symbolic
        // values yet.
        //
        // Otherwise if the input shape is known and `axes` is a constant, the
        // output is a symbolic shape.
        //
        // Otherwise the output is unknown.

        let axes_vec = axes.to_constant().map(|c| c.into_vec());

        let value = if let Some(var) = data.as_scalar()
            && axes_vec.as_deref().map(|v| v == [0]).unwrap_or(false)
        {
            SymTensor::from_vec([var.clone()].into())
        } else if let Some(dims) = data.shape()
            && let Some(mut axes) = axes_vec
        {
            let mut dims: Vec<_> = dims.collect();
            axes.sort();

            for (i, axis) in axes.into_iter().enumerate() {
                let axis = resolve_axis(dims.len() + 1, axis)
                    .map_err(|_| InferShapesError::IncorrectRank)?;
                dims.insert(axis + i, SymElem::Value(1));
            }

            SymTensor::from_shape(dims)
        } else {
            SymTensor::unknown("unknown data shape or axes value")
        };

        Ok([value].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymElem, SymTensor, sym_elems, sym_shape, sym_vec};

    use super::{Concat, ConstantOfShape, Gather, Range, Unsqueeze};

    fn extract_shape(mut result: Vec<SymTensor>) -> Vec<SymElem> {
        result.remove(0).shape().unwrap().collect()
    }

    #[test]
    fn test_concat() {
        // Concatenation of fixed dims.
        let a = sym_shape!("batch", 16, 64);
        let b = sym_shape!("batch", 16, 64);

        let mut sym_gen = SymbolGen::new();
        let op = Concat { axis: 1 };
        let result = op.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        let shape = extract_shape(result);
        assert_eq!(
            shape,
            sym_elems!("batch", SymElem::from(16) + SymElem::from(16), 64)
        );

        // Concatenation of symbolic dims.
        let a = sym_shape!("batch", "foo", 64);
        let b = sym_shape!("batch", "bar", 64);

        let op = Concat { axis: 1 };
        let result = op.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        let shape = extract_shape(result);
        assert_eq!(
            shape,
            sym_elems!("batch", SymElem::from("foo") + SymElem::from("bar"), 64)
        );

        // Concatenation of symbolic vectors.
        let bc_dims = sym_vec!("batch", "chans");
        let hw_dims = sym_vec!("height", "width");
        let op = Concat { axis: 0 };
        let mut result = op.infer_shapes(&[bc_dims, hw_dims], &mut sym_gen).unwrap();
        assert_eq!(
            result.remove(0).as_vector().unwrap(),
            sym_elems!("batch", "chans", "height", "width")
        );
    }

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
        let limit = sym_vec!(SymElem::from("start") + SymElem::from("limit"));
        let delta = sym_vec!(1);
        let result = Range
            .infer_shapes(&[start, limit, delta], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("limit"));

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
    fn test_unsqueeze() {
        let mut sym_gen = SymbolGen::new();

        // Unsqueeze into an ND-tensor.
        let shape = sym_shape!("batch", 16, 64);
        let axes = sym_vec!(1);
        let expected = sym_shape!("batch", 1, 16, 64);

        let result = Unsqueeze
            .infer_shapes(&[shape, axes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);

        // Unsqueeze a symbolic scalar into a symbolic vec.
        let scalar = SymTensor::from_scalar(1.into());
        let axes = sym_vec!(0);
        let expected = sym_vec!(1);
        let result = Unsqueeze
            .infer_shapes(&[scalar, axes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);
    }
}
