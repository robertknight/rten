use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError, resolve_axis};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Concat operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Concat.html>.
pub struct Concat {
    pub axis: i32,
}

impl InferShapes for Concat {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let first = inputs.require(0)?;

        let Some(first_dims) = first.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let axis = resolve_axis(first_dims.len(), self.axis)
            .map_err(|_| InferShapesError::IncorrectRank)?;

        // If input is a constant or symbolic vector, return a constant or
        // symbolic vector by concatenating each input.
        if axis == 0
            && inputs
                .iter()
                .all(|inp| inp.is_some_and(|t| t.values().is_some()))
        {
            let value = {
                let mut values = Vec::new();
                for inp in inputs.iter().flatten() {
                    values.extend(inp.values().expect("should have values").to_vec());
                }
                SymTensor::from_vec(values)
            };
            return Ok([value].into());
        }

        let mut out_shape: Vec<_> = first_dims.collect();

        for i in 1..inputs.len() {
            let input = inputs.require(i)?;
            if let Some(dim) = input.shape().and_then(|mut dims| dims.nth(axis)) {
                out_shape[axis] += dim;
            } else {
                out_shape[axis] += sym_gen.gen_positive();
            }
        }

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Tile operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Tile.html>.
pub struct Tile;

impl InferShapes for Tile {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let repeats = inputs.require(1)?;

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let ndim = data_dims.len();
        let data_dims: Vec<_> = data_dims.collect();

        let out_shape = if let Some(repeats) = repeats.as_vector() {
            if repeats.len() != ndim {
                return Err(InferShapesError::InvalidValue);
            }
            data_dims
                .into_iter()
                .zip(repeats.iter())
                .map(|(dim, r)| dim * r.clone())
                .collect()
        } else {
            // `repeats` is not a known vector. Output rank is preserved but
            // sizes are unknown.
            sym_gen.gen_shape(ndim)
        };

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_elems, sym_shape, sym_vec};

    use super::{Concat, Tile};

    fn extract_shape(mut result: Vec<SymTensor>) -> Vec<SymExpr> {
        result.remove(0).shape().unwrap().collect()
    }

    #[test]
    fn test_concat() {
        // Concatenation of fixed dims.
        let a = sym_shape!("batch", 16, 64);
        let b = sym_shape!("batch", 16, 64);

        let mut sym_gen = SymbolGen::new();
        let op = Concat { axis: 1 };
        let result = op.infer_shapes([a, b].into(), &mut sym_gen).unwrap();
        let shape = extract_shape(result);
        assert_eq!(
            shape,
            sym_elems!("batch", SymExpr::from(16) + SymExpr::from(16), 64)
        );

        // Concatenation of symbolic dims.
        let a = sym_shape!("batch", "foo", 64);
        let b = sym_shape!("batch", "bar", 64);

        let op = Concat { axis: 1 };
        let result = op.infer_shapes([a, b].into(), &mut sym_gen).unwrap();
        let shape = extract_shape(result);
        assert_eq!(
            shape,
            sym_elems!("batch", SymExpr::from("foo") + SymExpr::from("bar"), 64)
        );

        // Concatenation of symbolic vectors.
        let bc_dims = sym_vec!("batch", "chans");
        let hw_dims = sym_vec!("height", "width");
        let op = Concat { axis: 0 };
        let mut result = op
            .infer_shapes([bc_dims, hw_dims].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(
            result.remove(0).as_vector().unwrap(),
            sym_elems!("batch", "chans", "height", "width")
        );
    }

    #[test]
    fn test_tile() {
        let mut sym_gen = SymbolGen::new();

        // Fixed shape and known repeats.
        let data = sym_shape!(2, 3, 4);
        let repeats = sym_vec!(2, 1, 3);
        let result = Tile
            .infer_shapes([data, repeats].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(4, 3, 12));

        // Symbolic dim multiplied by a known repeat.
        let data = sym_shape!("batch", 16);
        let repeats = sym_vec!(2, 1);
        let result = Tile
            .infer_shapes([data, repeats].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!(SymExpr::from("batch") * SymExpr::from(2), 16)
        );

        // Unknown repeats — output rank is known but dim sizes are fresh
        // symbols.
        let data = sym_shape!(2, 3);
        let repeats = SymTensor::unknown("unknown");
        let result = Tile
            .infer_shapes([data, repeats].into(), &mut sym_gen)
            .unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
        for dim in &shape {
            assert!(matches!(dim, SymExpr::Var(_)));
        }

        // Repeats length doesn't match input rank.
        let data = sym_shape!(2, 3);
        let repeats = sym_vec!(2, 2, 2);
        let err = Tile
            .infer_shapes([data, repeats].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::InvalidValue);
    }
}
