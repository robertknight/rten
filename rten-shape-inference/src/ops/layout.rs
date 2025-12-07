use crate::infer_shapes::{BinaryOp, InferShapes, InferShapesError, resolve_axis};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::{SymElem, SymTensor};

/// Expand operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Expand.html>.
pub struct Expand;

impl InferShapes for Expand {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, shape] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        if let Some(sizes) = shape.values() {
            let rhs_shape: Vec<SymElem> = sizes.to_vec();
            BinaryOp.infer_shapes(&[data.clone(), SymTensor::from_shape(rhs_shape)], sym_gen)
        } else if let Some(data_dims) = data.shape()
            && let Some(shape_len) = shape.size(0).and_then(|d| match d {
                SymElem::Value(size) => Some(size),
                SymElem::Var(_) | SymElem::Add(_) | SymElem::Mul(_) | SymElem::Max(_) => None,
            })
        {
            // If we know the length of the shape but not the values, then we
            // can infer the output rank. Also any dims of size > 1 in the input
            // must have the same size in the output. Symbolic dimensions in the
            // input and dimensions of size 1 may broadcast.
            let data_dims: Vec<_> = data_dims.collect();
            let pad_dims = shape_len.saturating_sub(data_dims.len() as i32);
            let expanded_dims = data_dims.len().max(shape_len as usize) as i32;
            let out_dims = (0..expanded_dims)
                .map(|i| {
                    if i < pad_dims {
                        sym_gen.gen_positive()
                    } else {
                        match data_dims[(i - pad_dims) as usize] {
                            SymElem::Value(size) if size > 1 => SymElem::Value(size),
                            _ => sym_gen.gen_positive(),
                        }
                    }
                })
                .collect();
            Ok([SymTensor::from_shape(out_dims)].into())
        } else {
            Ok([SymTensor::unknown("unsupported shape")].into())
        }
    }
}

/// Flatten operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Flatten.html>.
pub struct Flatten {
    pub axis: i32,
}

impl InferShapes for Flatten {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [input] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };
        let Some(mut dims) = input.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        // nb. Partition dims into outer and inner. Note the `axis` attribute
        // is an exclusive count rather than an inclusive index.
        let ndim = dims.len();
        let n_outer_dims = if self.axis == ndim as i32 {
            ndim
        } else if let Ok(nd) = resolve_axis(ndim, self.axis) {
            nd
        } else {
            return Err(InferShapesError::IncorrectRank);
        };

        let outer_dims: Vec<_> = dims.by_ref().take(n_outer_dims).collect();
        let inner_dims: Vec<_> = dims.collect();

        let dim_product = |dims: &[SymElem]| -> SymElem {
            if let [dim] = dims {
                return dim.clone();
            }
            dims.iter()
                .fold(SymElem::Value(1), |prod, dim| prod * dim.clone())
                .simplify()
        };

        let out_shape = vec![dim_product(&outer_dims), dim_product(&inner_dims)];
        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Transpose operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Transpose.html>.
pub struct Transpose<'a> {
    pub perm: Option<&'a [usize]>,
}

impl InferShapes for Transpose<'_> {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [input] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        if let Some(dims) = input.shape() {
            let permuted_dims = if let Some(perm) = self.perm {
                let dims: Vec<_> = dims.collect();
                let mut permuted = Vec::with_capacity(perm.len());
                for &idx in perm {
                    if idx >= dims.len() {
                        return Err(InferShapesError::IncorrectRank);
                    }
                    permuted.push(dims[idx].clone());
                }
                permuted
            } else {
                let mut dims: Vec<_> = dims.collect();
                dims.reverse();
                dims
            };
            Ok([SymTensor::from_shape(permuted_dims)].into())
        } else if let Some(perm) = &self.perm {
            // If the input shape is unknown, but we have a permutation then
            // we can assume the output rank will match the permutation.
            let dims = (0..perm.len()).map(|_| sym_gen.gen_positive()).collect();
            Ok([SymTensor::from_shape(dims)].into())
        } else {
            Ok([SymTensor::unknown("unknown input shape")].into())
        }
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
    use crate::sym_tensor::{SymElem, SymTensor, sym_shape, sym_vec};

    use super::{Expand, Flatten, Transpose, Unsqueeze};

    #[test]
    fn test_expand() {
        let mut sym_gen = SymbolGen::new();

        // Broadcast with known shape.
        let data = sym_shape!("batch", 1, 16);
        let shape = sym_vec!("batch", 8, 16);
        let result = Expand.infer_shapes(&[data, shape], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", 8, 16));

        // Broadcast with known shape that is shorter than the input data.
        let data = sym_shape!("batch", 1, 16);
        let shape = sym_vec!(8, 16);
        let result = Expand.infer_shapes(&[data, shape], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", 8, 16));

        // Broadcast with known shape that is longer than the input data.
        let data = sym_shape!("batch", 1, 16);
        let shape = sym_vec!(4, "batch", 8, 16);
        let result = Expand.infer_shapes(&[data, shape], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(4, "batch", 8, 16));

        // Broadcast with shape that has a known length but unknown values.
        let data = sym_shape!("batch", 1, 16);
        let shape = sym_shape!(3);
        let result = Expand.infer_shapes(&[data, shape], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", "unknown_2", 16));
    }

    #[test]
    fn test_flatten() {
        let mut sym_gen = SymbolGen::new();

        // Combine last two dims.
        let data = sym_shape!("batch", "rows", "cols");
        let op = Flatten { axis: 1 };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!("batch", SymElem::from("rows") * SymElem::from("cols"))
        );

        // Combine first two dims.
        let data = sym_shape!("batch", "rows", "cols");
        let op = Flatten { axis: 2 };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(SymElem::from("batch") * SymElem::from("rows"), "cols")
        );

        // Combine all dims
        let data = sym_shape!("batch", "rows", "cols");
        let op = Flatten { axis: 3 };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(
                SymElem::from("batch") * SymElem::from("rows") * SymElem::from("cols"),
                1
            )
        );

        // Empty shape
        let data = sym_shape!();
        let op = Flatten { axis: 0 };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(1, 1));
    }

    #[test]
    fn test_transpose() {
        let mut sym_gen = SymbolGen::new();

        // Transpose with explicit permutation.
        let data = sym_shape!("batch", "rows", "cols");
        let op = Transpose {
            perm: Some(&[0, 2, 1]),
        };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", "cols", "rows"));

        // Transpose with implicit permutation.
        let data = sym_shape!("rows", "cols");
        let op = Transpose { perm: None };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("cols", "rows"));

        // Transpose with explicit permutation but unknown input shape.
        let data = SymTensor::unknown("unknown input shape");
        let op = Transpose {
            perm: Some(&[0, 2, 1]),
        };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", "unknown_2", "unknown_3"));

        // Transpose with implicit permutation and unknown input shape.
        let data = SymTensor::unknown("unknown input shape");
        let op = Transpose { perm: None };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::unknown("unknown input shape"));
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
