use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError, resolve_axis};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// TopK operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__TopK.html>.
pub struct TopK {
    pub axis: Option<i32>,
}

impl InferShapes for TopK {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let k = inputs.require(1)?;

        let Some(data_dims) = data.shape() else {
            return Ok([
                SymTensor::unknown("unknown input shape"),
                SymTensor::unknown("unknown input shape"),
            ]
            .into());
        };

        let ndim = data_dims.len();
        let axis = resolve_axis(ndim, self.axis.unwrap_or(-1))
            .map_err(|_| InferShapesError::IncorrectRank)?;

        // `k` is a 1D tensor with one element.
        let k_val = k
            .as_vector()
            .and_then(|v| v.first().cloned())
            .unwrap_or_else(|| sym_gen.gen_positive());

        let mut out_shape: Vec<SymExpr> = data_dims.collect();
        out_shape[axis] = k_val;

        let shape = SymTensor::from_shape(out_shape);
        Ok([shape.clone(), shape].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::TopK;

    #[test]
    fn test_top_k() {
        let mut sym_gen = SymbolGen::new();

        // Default axis (-1) with known K.
        let data = sym_shape!("batch", 16, 32);
        let k = sym_vec!(5);
        let op = TopK { axis: None };
        let result = op.infer_shapes([data, k].into(), &mut sym_gen).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], sym_shape!("batch", 16, 5));
        assert_eq!(result[1], sym_shape!("batch", 16, 5));

        // Explicit axis.
        let data = sym_shape!("batch", 16, 32);
        let k = sym_vec!(5);
        let op = TopK { axis: Some(0) };
        let result = op.infer_shapes([data, k].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(5, 16, 32));

        // Symbolic K value.
        let data = sym_shape!("batch", 32);
        let k = sym_vec!(SymExpr::from("k"));
        let op = TopK { axis: Some(-1) };
        let result = op.infer_shapes([data, k].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", "k"));
    }
}
