use crate::infer_shapes::{BinaryOp, InferShapes, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Perform a binary operation on the symbolic _values_ of two tensors or return
/// None if a comparison is not possible.
fn symbolic_binary_op(
    lhs: &SymTensor,
    rhs: &SymTensor,
    mut op: impl FnMut(&SymExpr, &SymExpr) -> Option<SymExpr>,
) -> Option<SymTensor> {
    if let Some(x) = lhs.as_scalar()
        && let Some(y) = rhs.as_scalar()
    {
        let result = op(x, y)?;
        Some(SymTensor::from_scalar(result))
    } else if let Some(lhs_values) = lhs.values()
        && let Some(rhs_values) = rhs.values()
    {
        let bin_op = |(x, y)| op(x, y);
        let elems: Option<Vec<SymExpr>> = match (lhs_values.len(), rhs_values.len()) {
            (1, _) => lhs_values
                .iter()
                .cycle()
                .zip(rhs_values)
                .map(bin_op)
                .collect(),
            (_, 1) => lhs_values
                .iter()
                .zip(rhs_values.iter().cycle())
                .map(bin_op)
                .collect(),
            _ => lhs_values.iter().zip(rhs_values).map(bin_op).collect(),
        };
        Some(SymTensor::from_vec(elems?))
    } else {
        None
    }
}

/// Evaluate a binary operation on symbolic tensors.
///
/// This will attempt to evaluate the operation on values in the tensor,
/// otherwise it will fall back to inferring just the shape.
fn binary_op_infer_shapes(
    inputs: &[SymTensor],
    sym_gen: &mut SymbolGen,
    op: impl FnMut(&SymExpr, &SymExpr) -> Option<SymExpr>,
) -> Result<Vec<SymTensor>, InferShapesError> {
    let [lhs, rhs] = inputs else {
        return Err(InferShapesError::IncorrectInputCount);
    };

    if let Some(result) = symbolic_binary_op(lhs, rhs, op) {
        return Ok([result].into());
    }

    BinaryOp.infer_shapes(inputs, sym_gen)
}

/// Add operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Add.html>.
pub struct Add;

impl InferShapes for Add {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let add = |x: &SymExpr, y: &SymExpr| {
            Some(match (x, y) {
                (SymExpr::Value(x), SymExpr::Value(y)) => SymExpr::Value(x + y),
                _ => x.clone() + y.clone(),
            })
        };
        binary_op_infer_shapes(inputs, sym_gen, add)
    }
}

/// Sub operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Sub.html>.
pub struct Sub;

impl InferShapes for Sub {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let sub = |x: &SymExpr, y: &SymExpr| {
            Some(match (x, y) {
                (SymExpr::Value(x), SymExpr::Value(y)) => SymExpr::Value(x - y),
                _ => x.clone() - y.clone(),
            })
        };
        binary_op_infer_shapes(inputs, sym_gen, sub)
    }
}

/// Div operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Div.html>.
pub struct Div;

impl InferShapes for Div {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let div = |x: &SymExpr, y: &SymExpr| {
            Some(match (x, y) {
                (SymExpr::Value(x), SymExpr::Value(y)) if *y != 0 => SymExpr::Value(x / y),
                _ => x.clone() / y.clone(),
            })
        };
        binary_op_infer_shapes(inputs, sym_gen, div)
    }
}

/// Equal operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Equal.html>.
pub struct Equal;

impl InferShapes for Equal {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let eq = |x: &SymExpr, y: &SymExpr| {
            let (x_min, x_max) = x.range();
            let (y_min, y_max) = y.range();

            if x == y {
                // Same symbol or value.
                Some(SymExpr::Value(1))
            } else if x_max < y_min || y_max < x_min {
                // Value ranges do not overlap, so the symbols must be
                // non-equal.
                Some(SymExpr::Value(0))
            } else {
                // Possible ranges overlap, so we don't know if the symbols
                // are equal.
                None
            }
        };
        binary_op_infer_shapes(inputs, sym_gen, eq)
    }
}

/// Mul operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Mul.html>.
pub struct Mul;

impl InferShapes for Mul {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let mul = |x: &SymExpr, y: &SymExpr| {
            Some(match (x, y) {
                (SymExpr::Value(x), SymExpr::Value(y)) => SymExpr::Value(x * y),
                _ => x.clone() * y.clone(),
            })
        };
        binary_op_infer_shapes(inputs, sym_gen, mul)
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::{Add, Div, Equal, Mul, Sub};

    #[test]
    fn test_add() {
        let mut sym_gen = SymbolGen::new();

        // Symbolic scalar
        let a = SymTensor::from_scalar(6.into());
        let b = SymTensor::from_scalar(5.into());
        let result = Add.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar(11.into()));

        // Symbolic vector
        let a = sym_vec!(5, "foo");
        let b = sym_vec!(6, "bar");
        let result = Add.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_vec!(11, SymExpr::from("foo") + SymExpr::from("bar"))
        );

        // Other shape
        let a = sym_shape!(5, "foo");
        let b = sym_shape!(1, "foo");
        let result = Add.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(5, "foo"));
    }

    #[test]
    fn test_sub() {
        let mut sym_gen = SymbolGen::new();

        // Symbolic scalar
        let a = SymTensor::from_scalar(6.into());
        let b = SymTensor::from_scalar(5.into());
        let result = Sub.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar(1.into()));

        // Symbolic vector
        let a = sym_vec!(5, "foo");
        let b = sym_vec!(6, "bar");
        let result = Sub.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_vec!(-1, SymExpr::from("foo") - SymExpr::from("bar"))
        );

        // Other shape
        let a = sym_shape!(5, "foo");
        let b = sym_shape!(1, "foo");
        let result = Sub.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(5, "foo"));
    }

    #[test]
    fn test_div() {
        let mut sym_gen = SymbolGen::new();

        // Symbolic vector with fixed values.
        let a = sym_vec!(16);
        let b = sym_vec!(2);
        let result = Div.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!(8));

        // Symbolic vector with symbolic values.
        let a = sym_vec!(16);
        let b = sym_vec!("foo");
        let result = Div.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_vec!(SymExpr::from(16) / SymExpr::from("foo"))
        );

        // Other shape
        let a = sym_shape!(5, "foo");
        let b = sym_shape!(1, "foo");
        let result = Div.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(5, "foo"));
    }

    #[test]
    fn test_equal() {
        let mut sym_gen = SymbolGen::new();

        // Comparison of fixed values.
        let a = sym_vec!(4, 8, 12);
        let b = sym_vec!(4, 2, 12);
        let result = Equal.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!(1, 0, 1));

        // Comparison of negative values with symbols that are known to have
        // a value >= 0.
        let a = sym_vec!("foo", "bar");
        let b = sym_vec!(-1, -1);
        let result = Equal.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_vec!(0, 0));

        // Comparison of positive values with symbols.
        //
        // In this case since the ranges overlap, we don't know the result and
        // fall back to regular shape inference.
        let a = sym_vec!("foo", "bar");
        let b = sym_vec!(2, 3);
        let result = Equal.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(2));
    }

    #[test]
    fn test_mul() {
        let mut sym_gen = SymbolGen::new();

        // Symbolic scalar
        let a = SymTensor::from_scalar(6.into());
        let b = SymTensor::from_scalar(5.into());
        let result = Mul.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], SymTensor::from_scalar(30.into()));

        // Symbolic vector
        let a = sym_vec!(5, "foo");
        let b = sym_vec!(6, "bar");
        let result = Mul.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_vec!(30, SymExpr::from("foo") * SymExpr::from("bar"))
        );

        // Other shape
        let a = sym_shape!(5, "foo");
        let b = sym_shape!(1, "foo");
        let result = Mul.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!(5, "foo"));
    }
}
