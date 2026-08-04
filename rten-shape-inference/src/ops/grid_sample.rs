use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// GridSample operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__GridSample.html>.
pub struct GridSample;

impl InferShapes for GridSample {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let grid = inputs.require(1)?;

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };
        let Some(grid_dims) = grid.shape() else {
            return Ok([SymTensor::unknown("unknown grid shape")].into());
        };

        // data is (N, C, D1, D2) and grid is (N, D1_out, D2_out, ..., r) where
        // D1..Dn are the spatial dims.
        let data_shape: Vec<_> = data_dims.collect();
        let grid_shape: Vec<_> = grid_dims.collect();
        if data_shape.len() < 3 || data_shape.len() != grid_shape.len() {
            return Err(InferShapesError::IncorrectRank);
        }

        // Output is (N, C, D1_out, D2_out, ...).
        let spatial_dims = data_shape.len() - 2;
        let out_shape = data_shape
            .into_iter()
            .take(2) // (N, C)
            .chain(grid_shape.into_iter().skip(1).take(spatial_dims))
            .collect();

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::GridSample;

    #[test]
    fn test_grid_sample() {
        let mut sym_gen = SymbolGen::new();

        // 2D sampling: data is (N, C, H, W), grid is (N, H_out, W_out, 2).
        let data = sym_shape!("batch", 3, 224, 224);
        let grid = sym_shape!("batch", 32, 64, 2);
        let result = GridSample
            .infer_shapes([data, grid].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 3, 32, 64));

        // 1D sampling: data is (N, C, W), grid is (N, W_out, 1).
        let data = sym_shape!("batch", 3, 224);
        let grid = sym_shape!("batch", 32, 1);
        let result = GridSample
            .infer_shapes([data, grid].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 3, 32));

        // Unknown input shape.
        let data = SymTensor::unknown("unknown");
        let grid = sym_shape!(1, 32, 64, 2);
        let result = GridSample
            .infer_shapes([data, grid].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);

        // Wrong rank.
        let data = sym_shape!(1, 3, 224);
        let grid = sym_shape!(1, 32, 64, 2);
        let err = GridSample
            .infer_shapes([data, grid].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::IncorrectRank);
    }
}
