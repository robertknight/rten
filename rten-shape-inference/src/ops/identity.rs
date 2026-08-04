use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Identity operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Identity.html>.
///
/// Unlike [`UnaryOp`](crate::UnaryOp), this copies the input's values (when
/// known) to the output, not just its shape.
pub struct Identity;

impl InferShapes for Identity {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        Ok([data.clone()].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_vec};

    use super::Identity;

    #[test]
    fn test_identity() {
        let mut sym_gen = SymbolGen::new();

        let input = sym_vec!("batch", 16, "seq", 24);
        let result = Identity
            .infer_shapes([input.clone()].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result, &[input]);

        let err = Identity
            .infer_shapes(InferShapesContext::new(&[]), &mut sym_gen)
            .err()
            .unwrap();
        assert_eq!(err, InferShapesError::IncorrectInputCount);
    }
}
