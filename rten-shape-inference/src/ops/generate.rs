use crate::infer_shapes::{InferShapes, InferShapesError, resolve_axis};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// OneHot operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__OneHot.html>.
pub struct OneHot {
    pub axis: i32,
}

impl InferShapes for OneHot {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [indices, depth, _values] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(indices_dims) = indices.shape() else {
            return Ok([SymTensor::unknown("unknown indices shape")].into());
        };

        let in_ndim = indices_dims.len();
        // `axis` may be in [-(ndim+1), ndim] since OneHot inserts a new
        // dimension.
        let axis =
            resolve_axis(in_ndim + 1, self.axis).map_err(|_| InferShapesError::IncorrectRank)?;

        // The depth is a scalar or vector containing one element.
        let depth_value = match depth.values() {
            Some([depth_value]) => depth_value.clone(),
            _ => sym_gen.gen_positive(),
        };

        let mut out_shape: Vec<SymExpr> = indices_dims.collect();
        out_shape.insert(axis, depth_value);

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::OneHot;

    #[test]
    fn test_one_hot() {
        let mut sym_gen = SymbolGen::new();

        // Insert depth axis at the end with a fixed depth.
        let indices = sym_shape!("batch", 8);
        let depth = SymTensor::from_scalar(10.into());
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: -1 };
        let result = op
            .infer_shapes(&[indices, depth, values], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 8, 10));

        // Insert depth axis at the start.
        let indices = sym_shape!("batch", 8);
        let depth = SymTensor::from_scalar(10.into());
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: 0 };
        let result = op
            .infer_shapes(&[indices, depth, values], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(10, "batch", 8));

        // Symbolic depth value.
        let indices = sym_shape!(4);
        let depth = SymTensor::from_scalar(SymExpr::from("d"));
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: -1 };
        let result = op
            .infer_shapes(&[indices, depth, values], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(4, "d"));

        // Depth as a rank-1 tensor containing exactly one element.
        let indices = sym_shape!("batch", 8);
        let depth = sym_vec!(10);
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: -1 };
        let result = op
            .infer_shapes(&[indices, depth, values], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 8, 10));

        // Unknown indices shape.
        let indices = SymTensor::unknown("unknown");
        let depth = SymTensor::from_scalar(10.into());
        let values = sym_vec!(0, 1);
        let op = OneHot { axis: -1 };
        let result = op
            .infer_shapes(&[indices, depth, values], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);
    }
}
