use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// SkipLayerNormalization / SkipSimplifiedLayerNormalization operators.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipLayerNormalization>.
pub struct SkipLayerNormalization;

impl InferShapes for SkipLayerNormalization {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;

        let shape = if let Some(dims) = data.shape() {
            SymTensor::from_shape(dims.collect())
        } else {
            SymTensor::unknown("unknown input shape")
        };

        // Outputs are `output`, `mean`, `inv_std_var` and `input_skip_bias_sum`.
        //
        // `output` and `input_skip_bias_sum` have the same shape as the input.
        // `mean` and `inv_std_var` are not computed by the rten implementation
        // and are returned as empty placeholders.
        let placeholder = SymTensor::from_fixed_shape(&[0]);
        Ok([shape.clone(), placeholder.clone(), placeholder, shape].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::SkipLayerNormalization;

    #[test]
    fn test_skip_layer_normalization() {
        let mut sym_gen = SymbolGen::new();

        // `output` and `input_skip_bias_sum` match the input shape, while
        // `mean` and `inv_std_var` are empty placeholders.
        let data = sym_shape!("batch", "seq", 32);
        let result = SkipLayerNormalization
            .infer_shapes([data].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(
            result,
            &[
                sym_shape!("batch", "seq", 32),
                sym_shape!(0),
                sym_shape!(0),
                sym_shape!("batch", "seq", 32),
            ]
        );

        // Unknown input shape.
        let data = SymTensor::unknown("unknown");
        let result = SkipLayerNormalization
            .infer_shapes([data].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].ndim(), None);
        assert_eq!(result[1], sym_shape!(0));
        assert_eq!(result[2], sym_shape!(0));
        assert_eq!(result[3].ndim(), None);

        // Missing input.
        let err = SkipLayerNormalization
            .infer_shapes(InferShapesContext::new(&[]), &mut sym_gen)
            .err();
        assert_eq!(err, Some(InferShapesError::IncorrectInputCount));
    }
}
