use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Multinomial operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Multinomial.html>.
pub struct Multinomial {
    /// Number of times to sample for each row of the input.
    pub sample_size: usize,
}

impl InferShapes for Multinomial {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let input = inputs.require(0)?;

        if input.ndim().is_some_and(|ndim| ndim != 2) {
            return Err(InferShapesError::IncorrectRank);
        }

        // Input is `(batch_size, class_size)`, output is
        // `(batch_size, sample_size)`.
        let batch_size = input.size(0).unwrap_or_else(|| sym_gen.gen_positive());
        let sample_size = SymExpr::Value(self.sample_size as i32);
        let out_shape = vec![batch_size, sample_size];
        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Dropout operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Dropout.html>.
pub struct Dropout;

impl InferShapes for Dropout {
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

        // Output 0 is the dropped-out data; output 1 is the boolean mask. Both
        // have the same shape as the input.
        Ok([shape.clone(), shape].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::{Dropout, Multinomial};

    #[test]
    fn test_multinomial() {
        let mut sym_gen = SymbolGen::new();

        // Known batch size.
        let data = sym_shape!("batch", 32);
        let result = Multinomial { sample_size: 4 }
            .infer_shapes([data].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result, &[sym_shape!("batch", 4)]);

        // Unknown input shape still yields a known sample size.
        let data = SymTensor::unknown("unknown");
        let result = Multinomial { sample_size: 4 }
            .infer_shapes([data].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), Some(2));
        assert_eq!(result[0].size(1), Some(4.into()));

        // Input with a known rank other than 2 is an error.
        let data = sym_shape!("batch", 32, 8);
        let err = Multinomial { sample_size: 4 }
            .infer_shapes([data].into(), &mut sym_gen)
            .err();
        assert_eq!(err, Some(InferShapesError::IncorrectRank));
    }

    #[test]
    fn test_dropout() {
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!("batch", 16, 32);
        let result = Dropout.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], sym_shape!("batch", 16, 32));
        assert_eq!(result[1], sym_shape!("batch", 16, 32));

        // Unknown input shape.
        let data = SymTensor::unknown("unknown");
        let result = Dropout.infer_shapes([data].into(), &mut sym_gen).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].ndim(), None);
        assert_eq!(result[1].ndim(), None);
    }
}
