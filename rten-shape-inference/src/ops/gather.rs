use crate::infer_shapes::{
    InferShapes, InferShapesContext, InferShapesError, resolve_axis, resolve_index,
};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::{Constant, SymTensor};

/// Gather operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Gather.html>.
pub struct Gather {
    pub axis: i32,
}

impl InferShapes for Gather {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let indices = inputs.require(1)?;

        let Some(mut data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown data shape")].into());
        };

        let data_ndim = data_dims.len();
        let axis = resolve_axis(data_ndim, self.axis)?;

        fn get<T: Clone>(vec: &[T], index: i32) -> Result<T, InferShapesError> {
            let index = resolve_index(vec.len(), index).ok_or(InferShapesError::InvalidValue)?;
            Ok(vec[index].clone())
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
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let indices = inputs.require(1)?;

        let shape = if let Some(dims) = indices.shape() {
            SymTensor::from_shape(dims.collect())
        } else {
            SymTensor::unknown("unknown indices shape")
        };

        Ok([shape].into())
    }
}

/// GatherND operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__GatherND.html>.
pub struct GatherND {
    pub batch_dims: usize,
}

impl InferShapes for GatherND {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let indices = inputs.require(1)?;

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown data shape")].into());
        };
        let Some(indices_dims) = indices.shape() else {
            return Ok([SymTensor::unknown("unknown indices shape")].into());
        };

        let indices_shape: Vec<SymExpr> = indices_dims.collect();

        // The last dim of indices is the size of the index tuple. We need this
        // to be a concrete value to determine which input dimensions are
        // gathered vs. preserved.
        let idx_tuple_size = match indices_shape.last() {
            Some(&SymExpr::Value(v)) => {
                usize::try_from(v).map_err(|_| InferShapesError::InvalidValue)?
            }
            Some(_) => {
                return Ok([SymTensor::unknown("unknown index tuple size")].into());
            }
            None => {
                return Err(InferShapesError::IncorrectRank);
            }
        };

        let suffix_start = self.batch_dims + idx_tuple_size;
        if suffix_start > data_dims.len() {
            return Err(InferShapesError::IncorrectRank);
        }
        let idx_len = indices_shape.len() - 1;

        // Output shape = indices.shape[:-1] + data.shape[batch_dims + idx_tuple_size:]
        let out_shape: Vec<SymExpr> = indices_shape
            .into_iter()
            .take(idx_len)
            .chain(data_dims.skip(suffix_start))
            .collect();

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::{Gather, GatherElements, GatherND};

    #[test]
    fn test_gather() {
        let infer = |data, indices, axis| {
            let mut sym_gen = SymbolGen::new();
            let op = Gather { axis };
            op.infer_shapes([data, indices].into(), &mut sym_gen)
        };

        // Gather scalar from symbolic vec.
        let shape = sym_vec!("batch", 16, "seq");
        let indices = SymTensor::from_scalar(2.into());
        let result = infer(shape, indices, 0).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar("seq".into()));

        // Gather vector from symbolic vec.
        let shape = sym_vec!("batch", 16, "seq");
        let indices = sym_vec!(0, 2, -2);
        let result = infer(shape, indices, 0).unwrap();
        assert_eq!(result[0], sym_vec!("batch", "seq", 16));

        // Gather with 2D data and symbolic vec indices
        let data = sym_shape!("vocab", "embed");
        let indices = sym_vec!(1, 2, 3);
        let result = infer(data, indices, 0).unwrap();
        assert_eq!(result[0], sym_shape!(3, "embed"));

        // Gather with 2D data and symbolic shape indices
        let data = sym_shape!("vocab", "embed");
        let indices = sym_shape!("n_tokens");
        let result = infer(data, indices, 0).unwrap();
        assert_eq!(result[0], sym_shape!("n_tokens", "embed"));

        // Gather with invalid index from symbolic vec.
        let shape = sym_vec!("batch", 16, "seq");
        let indices = SymTensor::from_scalar(3.into());
        let result = infer(shape, indices, 0).err().unwrap();
        assert_eq!(result, InferShapesError::InvalidValue);
    }

    #[test]
    fn test_gather_elements() {
        let mut sym_gen = SymbolGen::new();

        // Output shape = indices shape.
        let data = sym_shape!(4, 3, 2);
        let indices = sym_shape!(2, 3, 2);
        let result = GatherElements
            .infer_shapes([data, indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(2, 3, 2));

        // Unknown indices shape.
        let data = sym_shape!(4, 3, 2);
        let indices = SymTensor::unknown("unknown");
        let result = GatherElements
            .infer_shapes([data, indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);
    }

    #[test]
    fn test_gather_nd() {
        let mut sym_gen = SymbolGen::new();

        // No batch dims, index tuple selects entire dimensions.
        let data = sym_shape!(4, 3, 2);
        let indices = sym_shape!(2, 1);
        let op = GatherND { batch_dims: 0 };
        let result = op
            .infer_shapes([data, indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(2, 3, 2));

        // Index tuple covers all input dims.
        let data = sym_shape!(4, 3, 2);
        let indices = sym_shape!(2, 3);
        let op = GatherND { batch_dims: 0 };
        let result = op
            .infer_shapes([data, indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(2));

        // With batch_dims.
        let data = sym_shape!(2, 3, 4);
        let indices = sym_shape!(2, 1);
        let op = GatherND { batch_dims: 1 };
        let result = op
            .infer_shapes([data, indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(2, 4));

        // Symbolic dims preserved.
        let data = sym_shape!("batch", "seq", 64);
        let indices = sym_shape!("batch", "k", 1);
        let op = GatherND { batch_dims: 1 };
        let result = op
            .infer_shapes([data, indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", "k", 64));

        // Unknown data shape.
        let data = SymTensor::unknown("unknown");
        let indices = sym_shape!(2, 1);
        let op = GatherND { batch_dims: 0 };
        let result = op
            .infer_shapes([data, indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);

        // Symbolic index tuple size — output rank can't be determined.
        let data = sym_shape!(4, 3, 2);
        let indices = sym_shape!(2, "k");
        let op = GatherND { batch_dims: 0 };
        let result = op
            .infer_shapes([data, indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);

        // Negative index tuple size — invalid value.
        let data = sym_shape!(4, 3, 2);
        let indices = sym_shape!(2, -1);
        let op = GatherND { batch_dims: 0 };
        let result = op.infer_shapes([data, indices].into(), &mut sym_gen);
        assert_eq!(result, Err(InferShapesError::InvalidValue));
    }
}
