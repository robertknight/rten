use rten_tensor::Layout;

use crate::infer_shapes::{
    BinaryOp, InferShapes, InferShapesContext, InferShapesError, resolve_axis,
};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Expand operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Expand.html>.
pub struct Expand;

impl InferShapes for Expand {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let shape = inputs.require(1)?;

        if let Some(sizes) = shape.values() {
            let rhs_shape: Vec<SymExpr> = sizes.to_vec();
            BinaryOp.infer_shapes(
                [data.clone(), SymTensor::from_shape(rhs_shape)].into(),
                sym_gen,
            )
        } else if let Some(data_dims) = data.shape()
            && let Some(shape_len) = shape.size(0).and_then(|d| match d {
                SymExpr::Value(size) => Some(size),
                SymExpr::Neg(_)
                | SymExpr::Add(..)
                | SymExpr::Mul(..)
                | SymExpr::Div(..)
                | SymExpr::DivCeil(..)
                | SymExpr::Max(..)
                | SymExpr::Min(..)
                | SymExpr::Sub(..)
                | SymExpr::Var(_)
                | SymExpr::Broadcast(..) => None,
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
                            SymExpr::Value(size) if size > 1 => SymExpr::Value(size),
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
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let input = inputs.require(0)?;
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

        let dim_product = |dims: &[SymExpr]| -> SymExpr {
            if let [dim] = dims {
                return dim.clone();
            }
            dims.iter()
                .fold(SymExpr::Value(1), |prod, dim| prod * dim.clone())
                .simplify()
        };

        let out_shape = vec![dim_product(&outer_dims), dim_product(&inner_dims)];
        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Reshape operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Reshape.html>.
pub struct Reshape {
    pub allow_zero: bool,
}

impl InferShapes for Reshape {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let shape = inputs.require(1)?;

        // Minimum fixed value in the `shape` argument which doesn't require
        // special handling.
        let min_fixed = if self.allow_zero { 0 } else { 1 };

        let out_value = if let Some(dim_sizes) = shape.values() {
            let all_fixed = dim_sizes.iter().all(|v| {
                if let SymExpr::Value(v) = v
                    && *v >= min_fixed
                {
                    true
                } else {
                    false
                }
            });

            // If the input is a scalar or vector and the shape has the same
            // rank, this reshape must be a no-op. In that case we can
            // preserve the values in a symbolic value.
            if data.values().is_some()
                && dim_sizes.len() <= 1
                && data.ndim() == Some(dim_sizes.len())
            {
                data.clone()
            } else if all_fixed {
                // If all output sizes are fixed, we can generate the output shape
                // whether we know the input shape or not.
                SymTensor::from_shape(dim_sizes.to_vec())
            } else if let Some(data_dims) = data.shape() {
                let remainder_index = dim_sizes
                    .iter()
                    .position(|size| size == &SymExpr::Value(-1));

                let mut remainder = if remainder_index.is_some() {
                    Some(
                        data_dims
                            .reduce(|prod, d| prod * d)
                            .unwrap_or(SymExpr::Value(1))
                            // Combine constants into a single term where possible.
                            // eg. X * 3 * 4 => X * 12.
                            //
                            // This is important if the new shape contains a
                            // fixed term that is a product of terms in the
                            // original.
                            .simplify(),
                    )
                } else {
                    None
                };

                let mut out_shape = Vec::new();
                for (i, size) in dim_sizes.iter().enumerate() {
                    // Zero values in the shape have special handling and mean
                    // that the dimension should be copied from the input.
                    let size = if size == &SymExpr::Value(0) && !self.allow_zero {
                        if let Some(dim) = data.size(i) {
                            dim
                        } else {
                            return Err(InferShapesError::InvalidValue);
                        }
                    } else {
                        size.clone()
                    };

                    if size == SymExpr::Value(-1) {
                        // Add placeholder that we'll replace later.
                        out_shape.push(SymExpr::Value(0));
                    } else {
                        remainder = remainder.map(|r| r / size.clone());
                        out_shape.push(size);
                    }
                }

                if let Some(rem_index) = remainder_index
                    && let Some(remainder) = remainder
                {
                    out_shape[rem_index] = remainder.simplify();
                }

                SymTensor::from_shape(out_shape)
            } else {
                let out_shape = dim_sizes
                    .iter()
                    .map(|value| match value {
                        SymExpr::Value(v) if *v >= min_fixed => SymExpr::Value(*v),
                        // We don't know the input shape, so we can't determine
                        // the size of dimensions which are symbolic or require
                        // special handling.
                        _ => sym_gen.gen_positive(),
                    })
                    .collect();
                SymTensor::from_shape(out_shape)
            }
        } else if let Some(mut shape_dims) = shape.shape() {
            // If the shape is a vector with fixed length we can determine the
            // output rank, but not the sizes of any individual dimensions.
            if let Some(SymExpr::Value(size)) = shape_dims.next()
                && shape.ndim() == Some(1)
            {
                let dims = (0..size).map(|_| sym_gen.gen_positive()).collect();
                SymTensor::from_shape(dims)
            } else {
                SymTensor::unknown("unknown shape length")
            }
        } else {
            SymTensor::unknown("unknown shape")
        };

        Ok([out_value].into())
    }
}

#[derive(Debug, Default)]
pub struct Shape {
    pub start: Option<i32>,
    pub end: Option<i32>,
}

impl Shape {
    /// Convert `start` and `end` to positive values in `[0, ndim]`, clamping
    /// if out of range.
    pub fn resolve_start_end(&self, ndim: usize) -> std::ops::Range<usize> {
        // The spec says to clamp to `[0, r-1]` but this is incorrect as the end
        // bound is exclusive and so needs to be `r` to include the entire range.
        // See https://github.com/onnx/onnx/issues/6862.
        let ndim = ndim.try_into().unwrap();
        let start = self
            .start
            .map(|start| {
                let start = if start < 0 { start + ndim } else { start };
                start.clamp(0, ndim) as usize
            })
            .unwrap_or(0);

        let end = self
            .end
            .map(|end| {
                let end = if end < 0 { end + ndim } else { end };
                end.clamp(0, ndim) as usize
            })
            .unwrap_or(ndim as usize)
            // Spec doesn't say how to handle the case where `start > end`,
            // we clamp `end` to prevent this.
            .max(start);

        start..end
    }
}

impl InferShapes for Shape {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let input = inputs.require(0)?;

        let shape = input
            .shape()
            .map(|dims| {
                let std::ops::Range { start, end } = self.resolve_start_end(dims.len());
                let dims: Vec<_> = dims.skip(start).take(end - start).collect();
                SymTensor::from_vec(dims)
            })
            .unwrap_or(SymTensor::unknown("unknown input shape"));

        Ok([shape].into())
    }
}

/// Size operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Size.html>.
pub struct Size;

impl InferShapes for Size {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let input = inputs.require(0)?;

        // The value of the output is the product of the input's dimensions,
        // when they are known.
        let value = if let Some(dims) = input.shape() {
            let prod = dims.fold(SymExpr::Value(1), |prod, d| prod * d).simplify();
            SymTensor::from_scalar(prod)
        } else {
            SymTensor::from_shape(vec![])
        };

        Ok([value].into())
    }
}

/// DepthToSpace operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__DepthToSpace.html>.
pub struct DepthToSpace {
    pub block_size: u32,
}

impl InferShapes for DepthToSpace {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;

        let Some(dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let dims: Vec<_> = dims.collect();
        let [n, c, h, w]: &[SymExpr; 4] = dims
            .as_slice()
            .try_into()
            .map_err(|_| InferShapesError::IncorrectRank)?;

        // Compute `block_size ^ 2` channel divisor, reject if it can't fit in
        // a `SymExpr` value.
        let block_sq = self
            .block_size
            .checked_mul(self.block_size)
            .and_then(|sq| i32::try_from(sq).ok())
            .ok_or(InferShapesError::InvalidValue)?;
        let block_size = SymExpr::Value(self.block_size as i32);
        let block_sq = SymExpr::Value(block_sq);

        let out_shape = vec![
            n.clone(),
            c.clone() / block_sq,
            h.clone() * block_size.clone(),
            w.clone() * block_size,
        ];

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Squeeze operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Squeeze.html>.
pub struct Squeeze;

impl InferShapes for Squeeze {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;

        let Some(shape) = data.shape() else {
            return Ok([SymTensor::unknown("Unknown input shape")].into());
        };

        let axes = inputs.get(1);

        // Check if the `axes` input is constant and if so, resolve negative
        // values.
        let const_axes = axes.as_ref().and_then(|ax| match ax.to_constant() {
            Some(axes) if axes.ndim() == 1 => Some(axes.into_data()),
            // Perhaps higher-rank or scalar `axes` should error?
            _ => None,
        });

        let const_axes = const_axes
            .map(|axes| {
                axes.into_iter()
                    .map(|axis| resolve_axis(shape.len(), axis))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;

        // Symbolic vector to scalar
        if let Some(values) = data.as_vector()
            && values.len() == 1
            && matches!(const_axes.as_deref(), Some([0]) | None)
        {
            return Ok([SymTensor::from_scalar(values[0].clone())].into());
        }

        let out_shape = if let Some(const_axes) = const_axes {
            // If axes are known, remove corresponding axes from input shape.
            let out_shape = shape
                .enumerate()
                .filter(|(i, _dim)| !const_axes.contains(i))
                .map(|(_i, dim)| dim)
                .collect();
            SymTensor::from_shape(out_shape)
        } else if axes.is_none() && shape.clone().all(|size| matches!(size, SymExpr::Value(_))) {
            // If axes are not specified, but we know the exact size of all
            // dimensions, then remove all the size-1 dims.
            let out_shape = shape
                .filter(|dim| !matches!(dim, SymExpr::Value(1)))
                .collect();
            SymTensor::from_shape(out_shape)
        } else {
            SymTensor::unknown("Unknown axes")
        };

        Ok([out_shape].into())
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
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let input = inputs.require(0)?;

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
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let axes = inputs.require(1)?;

        // If data is a constant or symbolic scalar and axes is 0, the output
        // will be a length-1 vector. We can't handle higher-rank symbolic
        // values yet.
        //
        // Otherwise if the input shape is known and `axes` is a constant, the
        // output is a symbolic shape.
        //
        // Otherwise the output is unknown.

        let axes_vec = axes.to_constant().map(|c| c.into_data());

        let value = if let Some(var) = data.as_scalar()
            && axes_vec.as_deref().map(|v| v == [0]).unwrap_or(false)
        {
            SymTensor::from_vec([var.clone()].into())
        } else if let Some(dims) = data.shape()
            && let Some(axes) = axes_vec
        {
            let mut dims: Vec<_> = dims.collect();

            let out_rank = dims.len() + axes.len();
            let mut resolved_axes: Vec<_> = axes
                .into_iter()
                .map(|axis| {
                    resolve_axis(out_rank, axis).map_err(|_| InferShapesError::IncorrectRank)
                })
                .collect::<Result<_, _>>()?;
            resolved_axes.sort();

            for axis in resolved_axes {
                dims.insert(axis, SymExpr::Value(1));
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
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::{
        DepthToSpace, Expand, Flatten, Reshape, Shape, Size, Squeeze, Transpose, Unsqueeze,
    };

    #[test]
    fn test_expand() {
        let mut sym_gen = SymbolGen::new();

        // Broadcast with known shape.
        let data = sym_shape!("batch", 1, 16);
        let shape = sym_vec!("batch", 8, 16);
        let result = Expand
            .infer_shapes([data, shape].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 8, 16));

        // Broadcast with known shape that is shorter than the input data.
        let data = sym_shape!("batch", 1, 16);
        let shape = sym_vec!(8, 16);
        let result = Expand
            .infer_shapes([data, shape].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 8, 16));

        // Broadcast with known shape that is longer than the input data.
        let data = sym_shape!("batch", 1, 16);
        let shape = sym_vec!(4, "batch", 8, 16);
        let result = Expand
            .infer_shapes([data, shape].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(4, "batch", 8, 16));

        // Broadcast with shape that has a known length but unknown values.
        let data = sym_shape!("batch", 1, 16);
        let shape = sym_shape!(3);
        let result = Expand
            .infer_shapes([data, shape].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", "unknown_2", 16));
    }

    #[test]
    fn test_flatten() {
        let mut sym_gen = SymbolGen::new();

        // Combine last two dims.
        let data = sym_shape!("batch", "rows", "cols");
        let op = Flatten { axis: 1 };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!("batch", SymExpr::from("rows") * SymExpr::from("cols"))
        );

        // Combine first two dims.
        let data = sym_shape!("batch", "rows", "cols");
        let op = Flatten { axis: 2 };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(SymExpr::from("batch") * SymExpr::from("rows"), "cols")
        );

        // Combine all dims
        let data = sym_shape!("batch", "rows", "cols");
        let op = Flatten { axis: 3 };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(
                SymExpr::from("batch") * SymExpr::from("cols") * SymExpr::from("rows"),
                1
            )
        );

        // Empty shape
        let data = sym_shape!();
        let op = Flatten { axis: 0 };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(1, 1));
    }

    #[test]
    fn test_reshape() {
        let mut sym_gen = SymbolGen::new();
        let op = Reshape { allow_zero: false };
        let allow_zero_op = Reshape { allow_zero: true };

        // Simple reshape of fixed dims.
        let data = sym_shape!("batch", 8, 8);
        let shape = sym_vec!("batch", 64);
        let result = op.infer_shapes([data, shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", 64));

        // Reshape of fixed dims with -1 in shape.
        let data = sym_shape!("batch", 8, 8);
        let shape = sym_vec!("batch", -1);
        let result = op.infer_shapes([data, shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", 64));

        // Reshape where shape contains a zero and the corresponding input dim
        // is fixed.
        let data = sym_shape!(32, 8, 8);
        let shape = sym_vec!(0, -1);
        let result = op.infer_shapes([data, shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(32, 64));

        // Reshape where shape contains a zero and the corresponding input dim
        // is symbolic.
        let data = sym_shape!("batch", 8, 8);
        let shape = sym_vec!(0, -1);
        let result = op.infer_shapes([data, shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", 64));

        // Reshape where shape contains a zero and `allow_zero` is true.
        let data = sym_shape!("batch", 8, 0);
        let shape = sym_vec!("batch", 0, 8);
        let result = allow_zero_op
            .infer_shapes([data, shape].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 0, 8));

        // Reshape of symbolic scalar to scalar. This is a no-op and the symbolic
        // value should be preserved.
        let data = SymTensor::from_scalar(5.into());
        let shape = sym_vec!();
        let result = op
            .infer_shapes([data.clone(), shape].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], data);

        // Reshape of symbolic vector to vector. This is a no-op and the symbolic
        // values should be preserved.
        let data = sym_vec!("batch", 3, 8);
        let shape = sym_vec!(3);
        let result = op
            .infer_shapes([data.clone(), shape].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], data);

        // Unknown input shape, but known output shape.
        let data = SymTensor::unknown("unknown");
        let shape = sym_vec!(2, 4, 8);
        let result = allow_zero_op
            .infer_shapes([data, shape.clone()].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(2, 4, 8));

        // Unknown input shape, symbolic output shape.
        //
        // When `allow_zero=false` we have to assume that the symbolic values
        // could be zero, in which case we'd need to copy the dimensions from
        // the input. Since we don't know the input dimensions, we have to
        // represent the dim sizes as new symbols in the output.
        let data = SymTensor::unknown("unknown");
        let shape = sym_vec!("batch", "seq");
        let result = op
            .infer_shapes([data, shape.clone()].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", "unknown_2"));

        // Output shape with known length but unknown values.
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!("batch", "seq");
        let shape = sym_shape!(3);
        let result = allow_zero_op
            .infer_shapes([data, shape.clone()].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", "unknown_2", "unknown_3"));

        // Test case taken from ModernBERT model.
        let data = sym_shape!("batch", "seq", 12, 64);
        let shape = sym_vec!("batch", -1, 768);
        let result = allow_zero_op
            .infer_shapes([data, shape.clone()].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", "seq", 768));

        // Case where remainder has to be represented as a division expression.
        let data = sym_shape!("batch", "seq");
        let shape = sym_vec!("batch", 2, -1);
        let result = allow_zero_op
            .infer_shapes([data, shape.clone()].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(
            result[0],
            sym_shape!("batch", 2, SymExpr::from("seq") / SymExpr::from(2))
        );
    }

    #[test]
    fn test_shape() {
        // Shape with no start or end attribute.
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!("batch", "seq", 64);
        let op = Shape {
            start: None,
            end: None,
        };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!("batch", "seq", 64));

        // Shape with start and end attribute
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!("batch", "seq", 64);
        let op = Shape {
            start: Some(1),
            end: Some(2),
        };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!("seq"));
    }

    #[test]
    fn test_size() {
        let mut sym_gen = SymbolGen::new();

        // Fully fixed shape — result is a known scalar.
        let data = sym_shape!(2, 3, 4);
        let result = Size.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar(SymExpr::Value(24)));

        // Symbolic dims — result is a scalar product expression.
        let data = sym_shape!("batch", 16, "seq");
        let result = Size.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            SymTensor::from_scalar(
                SymExpr::from("batch") * SymExpr::from(16) * SymExpr::from("seq")
            )
        );

        // Unknown input shape — result is a scalar with unknown value.
        let data = SymTensor::unknown("unknown");
        let result = Size.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0].ndim(), Some(0));
    }

    #[test]
    fn test_depth_to_space() {
        let mut sym_gen = SymbolGen::new();

        // 4D input with block_size=2.
        let data = sym_shape!(1, 8, 4, 6);
        let op = DepthToSpace { block_size: 2 };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(1, 2, 8, 12));

        // Symbolic input.
        let data = sym_shape!("batch", "chans", "h", "w");
        let op = DepthToSpace { block_size: 2 };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!(
                "batch",
                SymExpr::from("chans") / SymExpr::from(4),
                SymExpr::from("h") * SymExpr::from(2),
                SymExpr::from("w") * SymExpr::from(2),
            )
        );

        // Wrong rank.
        let data = sym_shape!(1, 8, 4);
        let op = DepthToSpace { block_size: 2 };
        let err = op.infer_shapes([data].into(), &mut sym_gen).unwrap_err();
        assert_eq!(err, InferShapesError::IncorrectRank);

        // Block size whose square exceeds `i32::MAX` (but fits in `u32`).
        let data = sym_shape!("batch", "chans", "h", "w");
        let op = DepthToSpace { block_size: 50_000 };
        let err = op.infer_shapes([data].into(), &mut sym_gen).unwrap_err();
        assert_eq!(err, InferShapesError::InvalidValue);
    }

    #[test]
    fn test_squeeze() {
        // Shape
        let mut sym_gen = SymbolGen::new();
        let shape = sym_shape!("foo", 1, 64);
        let axes = sym_vec!(1);
        let result = Squeeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("foo", 64));

        // Symbolic vec to scalar
        let mut sym_gen = SymbolGen::new();
        let shape = sym_vec!("foo");
        let axes = sym_vec!(0);
        let result = Squeeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], SymTensor::from_scalar("foo".into()));

        // Symbolic vec to scalar, negative axis
        let shape = sym_vec!("bar");
        let axes = sym_vec!(-1);
        let result = Squeeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], SymTensor::from_scalar("bar".into()));

        // Symbolic vec to scalar, no axes
        let shape = sym_vec!("bar");
        let result = Squeeze.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar("bar".into()));

        // Non-const axes
        let mut sym_gen = SymbolGen::new();
        let shape = sym_shape!("foo");
        let axes = sym_vec!("what");
        let result = Squeeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], SymTensor::unknown("Unknown axes"));

        // Unknown input shape
        let shape = SymTensor::unknown("?");
        let axes = sym_vec!(0);
        let result = Squeeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], SymTensor::unknown("Unknown input shape"));

        // Negative axis
        let shape = sym_shape!("foo", 32, 1);
        let axes = sym_vec!(-1);
        let result = Squeeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("foo", 32));

        // Fixed shape, no axes
        let shape = sym_shape!(32, 1, 12);
        let result = Squeeze.infer_shapes([shape].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(32, 12));
    }

    #[test]
    fn test_transpose() {
        let mut sym_gen = SymbolGen::new();

        // Transpose with explicit permutation.
        let data = sym_shape!("batch", "rows", "cols");
        let op = Transpose {
            perm: Some(&[0, 2, 1]),
        };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", "cols", "rows"));

        // Transpose with implicit permutation.
        let data = sym_shape!("rows", "cols");
        let op = Transpose { perm: None };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("cols", "rows"));

        // Transpose with explicit permutation but unknown input shape.
        let data = SymTensor::unknown("unknown input shape");
        let op = Transpose {
            perm: Some(&[0, 2, 1]),
        };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", "unknown_2", "unknown_3"));

        // Transpose with implicit permutation and unknown input shape.
        let data = SymTensor::unknown("unknown input shape");
        let op = Transpose { perm: None };
        let result = op.infer_shapes([data].into(), &mut sym_gen).unwrap();
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
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);

        // Unsqueeze a symbolic scalar into a symbolic vec.
        let scalar = SymTensor::from_scalar(1.into());
        let axes = sym_vec!(0);
        let expected = sym_vec!(1);
        let result = Unsqueeze
            .infer_shapes([scalar, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);

        // Unsqueeze with multiple axes.
        let shape = sym_shape!("batch", 64);
        let axes = sym_vec!(0, 1);
        let expected = sym_shape!(1, 1, "batch", 64);
        let result = Unsqueeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);

        // Unsqueeze with multiple non-adjacent axes, where a later axis refers
        // to a position near the end of the output.
        let shape = sym_shape!("batch", 64);
        let axes = sym_vec!(1, 3);
        let expected = sym_shape!("batch", 1, 64, 1);
        let result = Unsqueeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);

        // Unsqueeze with unsorted axes.
        let shape = sym_shape!("batch", 64);
        let axes = sym_vec!(3, 1);
        let expected = sym_shape!("batch", 1, 64, 1);
        let result = Unsqueeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);

        // Unsqueeze with negative axes, resolved against the output rank.
        let shape = sym_shape!("batch", 64);
        let axes = sym_vec!(-1, -4);
        let expected = sym_shape!(1, "batch", 64, 1);
        let result = Unsqueeze
            .infer_shapes([shape, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);
    }
}
