use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// NonMaxSuppression operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html>.
pub struct NonMaxSuppression;

impl InferShapes for NonMaxSuppression {
    fn infer_shapes(
        &self,
        _inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        // Output is `(num_selected, 3)`. `num_selected` is data-dependent.
        let out_shape = vec![sym_gen.gen_positive(), SymExpr::Value(3)];
        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::NonMaxSuppression;

    #[test]
    fn test_non_max_suppression() {
        let mut sym_gen = SymbolGen::new();

        // Output is `(num_selected, 3)` with a symbolic first dim.
        let boxes = sym_shape!(1, 100, 4);
        let scores = sym_shape!(1, 80, 100);
        let result = NonMaxSuppression
            .infer_shapes([boxes, scores].into(), &mut sym_gen)
            .unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
        assert!(matches!(shape[0], SymExpr::Var(_)));
        assert_eq!(shape[1], SymExpr::Value(3));
    }
}
