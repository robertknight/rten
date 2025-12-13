use crate::infer_shapes::{InferShapes, InferShapesError, UnaryOp};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Neg operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Neg.html>.
pub struct Neg;

impl InferShapes for Neg {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };
        if let Some(item) = data.as_scalar() {
            return Ok([SymTensor::from_scalar(-item.clone())].into());
        } else if let Some(vec) = data.as_vector() {
            let neg_vec = vec.iter().map(|item| -item.clone()).collect();
            return Ok([SymTensor::from_vec(neg_vec)].into());
        }
        UnaryOp.infer_shapes(inputs, sym_gen)
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymElem, SymTensor, sym_shape, sym_vec};

    use super::Neg;

    #[test]
    fn test_neg() {
        let mut sym_gen = SymbolGen::new();
        let x = sym_shape!("batch", 4, 4);
        let result = Neg.infer_shapes(&[x], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", 4, 4));

        // Symbolic scalar
        let x = SymTensor::from_scalar("batch".into());
        let result = Neg.infer_shapes(&[x], &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar(-SymElem::from("batch")));

        // Symbolic vec
        let x = sym_vec!("batch", 64);
        let result = Neg.infer_shapes(&[x], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            SymTensor::from_vec(vec![-SymElem::from("batch"), -SymElem::from(64),])
        );
    }
}
