use crate::infer_shapes::{BinaryOp, InferShapes, InferShapesError};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// MatMul operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__MatMul.html>.
pub struct MatMul;

impl InferShapes for MatMul {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [lhs, rhs, ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };
        let Some(mut lhs_dims) = lhs.shape() else {
            return Ok([SymTensor::unknown("unknown lhs shape")].into());
        };
        let Some(mut rhs_dims) = rhs.shape() else {
            return Ok([SymTensor::unknown("unknown rhs shape")].into());
        };

        let lhs_ndim = lhs_dims.len();
        let rhs_ndim = rhs_dims.len();

        if lhs_ndim < 2 || rhs_ndim < 2 {
            // TODO - Handle the case where the LHS or RHS is a vector.
            return Ok([SymTensor::unknown("rank < 2")].into());
        };

        // Output shape is (broadcast(lhs_batch_dims, rhs_batch_dims), M, N)
        let lhs_batch_dims = SymTensor::from_shape(lhs_dims.by_ref().take(lhs_ndim - 2).collect());
        let rhs_batch_dims = SymTensor::from_shape(rhs_dims.by_ref().take(rhs_ndim - 2).collect());

        let [batch_dims] = BinaryOp
            .infer_shapes(&[lhs_batch_dims, rhs_batch_dims], sym_gen)?
            .try_into()
            .expect("should have one output");

        let Some(batch_dims) = batch_dims.shape() else {
            return Ok(vec![SymTensor::unknown("unknown batch dims")]);
        };

        let m = lhs_dims.next().unwrap();
        let n = rhs_dims.nth(1).unwrap();
        let mut batch_dims: Vec<_> = batch_dims.collect();
        batch_dims.push(m);
        batch_dims.push(n);

        Ok(vec![SymTensor::from_shape(batch_dims)])
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymElem, SymTensor, sym_shape};

    use super::MatMul;

    #[test]
    fn test_matmul() {
        let mut sym_gen = SymbolGen::new();

        // MatMul with no batch dims
        let lhs = sym_shape!("m", "k");
        let rhs = sym_shape!("k", "n");
        let result = MatMul.infer_shapes(&[lhs, rhs], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("m", "n"));

        // MatMul with batch dim
        let lhs = sym_shape!("batch", "m", "k");
        let rhs = sym_shape!("batch", "k", "n");
        let result = MatMul.infer_shapes(&[lhs, rhs], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", "m", "n"));

        // MatMul with batch dims that are broadcast
        let lhs = sym_shape!(1, "batch_b", "m", "k");
        let rhs = sym_shape!("batch_a", 1, "k", "n");
        let result = MatMul.infer_shapes(&[lhs, rhs], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch_a", "batch_b", "m", "n"));

        // Case where LHS is a vector.
        let lhs = sym_shape!("k");
        let rhs = sym_shape!("k", "n");
        let result = MatMul.infer_shapes(&[lhs, rhs], &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::unknown("rank < 2"));
    }
}
