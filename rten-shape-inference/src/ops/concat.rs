use crate::infer_shapes::{InferShapes, InferShapesError};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Tile operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Tile.html>.
pub struct Tile;

impl InferShapes for Tile {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, repeats] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

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
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::Tile;

    #[test]
    fn test_tile() {
        let mut sym_gen = SymbolGen::new();

        // Fixed shape and known repeats.
        let data = sym_shape!(2, 3, 4);
        let repeats = sym_vec!(2, 1, 3);
        let result = Tile.infer_shapes(&[data, repeats], &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(4, 3, 12));

        // Symbolic dim multiplied by a known repeat.
        let data = sym_shape!("batch", 16);
        let repeats = sym_vec!(2, 1);
        let result = Tile.infer_shapes(&[data, repeats], &mut sym_gen).unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!(SymExpr::from("batch") * SymExpr::from(2), 16)
        );

        // Unknown repeats — output rank is known but dim sizes are fresh
        // symbols.
        let data = sym_shape!(2, 3);
        let repeats = SymTensor::unknown("unknown");
        let result = Tile.infer_shapes(&[data, repeats], &mut sym_gen).unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
        for dim in &shape {
            assert!(matches!(dim, SymExpr::Var(_)));
        }

        // Repeats length doesn't match input rank.
        let data = sym_shape!(2, 3);
        let repeats = sym_vec!(2, 2, 2);
        let err = Tile
            .infer_shapes(&[data, repeats], &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::InvalidValue);
    }
}
