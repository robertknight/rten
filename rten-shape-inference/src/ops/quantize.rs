use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// DynamicQuantizeLinear operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html>.
pub struct DynamicQuantizeLinear;

impl InferShapes for DynamicQuantizeLinear {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;

        let shape = if let Some(shape) = data.shape() {
            SymTensor::from_shape(shape.collect())
        } else {
            SymTensor::unknown("unknown input shape")
        };

        let scale_shape = SymTensor::from_shape(vec![]);
        let zero_point_shape = SymTensor::from_shape(vec![]);
        Ok([shape, scale_shape, zero_point_shape].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::DynamicQuantizeLinear;

    #[test]
    fn test_dynamic_quantize_linear() {
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!(32, 32);
        let result = DynamicQuantizeLinear
            .infer_shapes([data].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result, &[sym_shape!(32, 32), sym_shape!(), sym_shape!(),]);
    }
}
