use std::fmt::Debug;
use std::iter::zip;

use crate::ops::{from_data, get_input_as_float, Input, IntoOpResult, OpError, Operator, Output};
use crate::tensor::Tensor;

/// Given the shapes of two inputs to a binary operation, choose the one that
/// will be used as the output shape. The other tensor will be broadcasted
/// to match.
pub fn choose_broadcast_shape<'a>(a: &'a [usize], b: &'a [usize]) -> &'a [usize] {
    if a.len() != b.len() {
        if a.len() < b.len() {
            b
        } else {
            a
        }
    } else if a < b {
        b
    } else {
        a
    }
}

/// Compute the result of applying the binary operation `op` to corresponding
/// elements of `a` and `b`. The shapes of `a` and `b` are broadcast to a
/// matching shape if necessary.
fn binary_op<T: Copy + Debug, F: Fn(T, T) -> T>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    op: F,
) -> Result<Tensor<T>, OpError> {
    let out_shape = choose_broadcast_shape(a.shape(), b.shape());
    if !a.can_broadcast(out_shape) || !b.can_broadcast(out_shape) {
        return Err(OpError::IncompatibleInputShapes(
            "Cannot broadcast inputs to compatible shape",
        ));
    }
    let a_elts = a.broadcast_elements(out_shape);
    let b_elts = b.broadcast_elements(out_shape);
    let out_data = zip(a_elts, b_elts).map(|(a, b)| op(a, b)).collect();
    Ok(from_data(out_shape.into(), out_data))
}

/// Return true if an elementwise binary operation can be performed in-place
/// on `a` given `b` as the other argument.
fn can_run_binary_op_in_place<T: Copy>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    b.can_broadcast(a.shape())
}

/// Perform an elementwise binary operation in-place.
///
/// This requires that `b` can be broadcast to the shape of `a`.
fn binary_op_in_place<T: Copy + Debug, F: Fn(&mut T, T)>(a: &mut Tensor<T>, b: &Tensor<T>, op: F) {
    // Fast paths for contiguous LHS
    if a.is_contiguous() {
        if let Some(scalar) = b.item() {
            // When RHS is a scalar, we don't need to iterate over it at all.
            for a_elt in a.data_mut().iter_mut() {
                op(a_elt, scalar);
            }
        } else if a.shape() == b.shape() && b.is_contiguous() {
            // When RHS is contiguous and same shape as LHS we can use a simple iterator.
            for (a_elt, b_elt) in zip(a.data_mut().iter_mut(), b.data().iter()) {
                op(a_elt, *b_elt);
            }
        } else {
            // Otherwise a more complex RHS iterator is required.
            let b_elts = b.broadcast_elements(a.shape());
            for (a_elt, b_elt) in zip(a.data_mut().iter_mut(), b_elts) {
                op(a_elt, b_elt);
            }
        }
        return;
    }

    let b_elts = b.broadcast_elements(a.shape());
    let a_offsets = a.offsets();
    let a_data = a.data_mut();
    for (a_offset, b_elt) in zip(a_offsets, b_elts) {
        op(&mut a_data[a_offset], b_elt);
    }
}

/// Perform a commutative elementwise binary operation.
///
/// This is an optimized alternative to `binary_op` for the case where the
/// operands can be swapped without affecting the result. In this case we
/// copy the larger of the two operands and then perform the operation in-place
/// on it. This benefits from various optimizations in `binary_op_in_place`.
fn binary_commutative_op<T: Copy + Debug, F: Fn(&mut T, T)>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    op: F,
) -> Result<Tensor<T>, OpError> {
    let mut out;
    let other;
    if b.can_broadcast(a.shape()) {
        out = a.clone();
        other = b;
    } else if a.can_broadcast(b.shape()) {
        out = b.clone();
        other = a;
    } else {
        return Err(OpError::IncompatibleInputShapes(
            "Cannot broadcast inputs to compatible shape",
        ));
    }
    binary_op_in_place(&mut out, other, op);
    Ok(out)
}

/// Perform elementwise addition of two tensors.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    binary_commutative_op(a, b, |x, y| *x += y)
}

/// Perform in-place elementwise addition of two tensors.
pub fn add_in_place(a: &mut Tensor, b: &Tensor) {
    binary_op_in_place(a, b, |x, y| *x += y);
}

#[derive(Debug)]
pub struct Add {}

impl Operator for Add {
    fn name(&self) -> &str {
        "Add"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let a = get_input_as_float(inputs, 0)?;
        let b = get_input_as_float(inputs, 1)?;
        add(a, b).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Result<Output, OpError> {
        let mut a = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        let b = get_input_as_float(other, 0)?;

        if can_run_binary_op_in_place(&a, b) {
            add_in_place(&mut a, b);
            Ok(a.into())
        } else {
            add(&a, &b).map(|t| t.into())
        }
    }
}

/// Perform elementwise division of two tensors.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    binary_op(a, b, |x, y| x / y)
}

/// Perform in-place elementwise division of two tensors.
pub fn div_in_place(a: &mut Tensor, b: &Tensor) {
    binary_op_in_place(a, b, |x, y| *x /= y);
}

#[derive(Debug)]
pub struct Div {}

impl Operator for Div {
    fn name(&self) -> &str {
        "Div"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let a = get_input_as_float(inputs, 0)?;
        let b = get_input_as_float(inputs, 1)?;
        div(a, b).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Result<Output, OpError> {
        let mut a = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        let b = get_input_as_float(other, 0)?;

        if can_run_binary_op_in_place(&a, b) {
            div_in_place(&mut a, b);
            Ok(a.into())
        } else {
            div(&a, &b).map(|t| t.into())
        }
    }
}

/// Multiply two tensors elementwise.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    binary_commutative_op(a, b, |x, y| *x *= y)
}

/// Perform in-place elementwise multiplication of two tensors.
pub fn mul_in_place(a: &mut Tensor, b: &Tensor) {
    binary_op_in_place(a, b, |a_elt, b_elt| *a_elt *= b_elt);
}

#[derive(Debug)]
pub struct Mul {}

impl Operator for Mul {
    fn name(&self) -> &str {
        "Mul"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let a = get_input_as_float(inputs, 0)?;
        let b = get_input_as_float(inputs, 1)?;
        mul(a, b).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Result<Output, OpError> {
        let mut a = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        let b = get_input_as_float(other, 0)?;

        if can_run_binary_op_in_place(&a, b) {
            mul_in_place(&mut a, b);
            Ok(a.into())
        } else {
            mul(&a, &b).map(|t| t.into())
        }
    }
}

/// Perform elementwise subtraction of two tensors.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    binary_op(a, b, |x, y| x - y)
}

/// Perform in-place elementwise subtraction of two tensors.
pub fn sub_in_place(a: &mut Tensor, b: &Tensor) {
    binary_op_in_place(a, b, |x, y| *x -= y);
}

#[derive(Debug)]
pub struct Sub {}

impl Operator for Sub {
    fn name(&self) -> &str {
        "Sub"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let a = get_input_as_float(inputs, 0)?;
        let b = get_input_as_float(inputs, 1)?;
        sub(a, b).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Result<Output, OpError> {
        let mut a = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        let b = get_input_as_float(other, 0)?;

        if can_run_binary_op_in_place(&a, b) {
            sub_in_place(&mut a, b);
            Ok(a.into())
        } else {
            sub(&a, &b).map(|t| t.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{
        add, add_in_place, div, div_in_place, mul, mul_in_place, sub, sub_in_place, Add, OpError,
        Operator, Output,
    };
    use crate::tensor::{from_data, from_scalar, from_vec};
    use crate::test_util::expect_equal;

    #[test]
    fn test_add() -> Result<(), String> {
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![11., 22., 33., 44.]);
        let result = add(&a, &b).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_add_broadcasted() -> Result<(), String> {
        // Simple case where comparing ordering of tensor shapes tells us
        // target shape.
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![1], vec![10.]);
        let expected = from_data(vec![2, 2], vec![11., 12., 13., 14.]);
        let result = add(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // Try alternative ordering for inputs.
        let result = add(&b, &a).unwrap();
        expect_equal(&result, &expected)?;

        // Case where the length of tensor shapes needs to be compared before
        // the ordering, since ([5] > [1,5]).
        let a = from_data(vec![5], vec![1., 2., 3., 4., 5.]);
        let b = from_data(vec![1, 5], vec![1., 2., 3., 4., 5.]);
        let expected = from_data(vec![1, 5], vec![2., 4., 6., 8., 10.]);

        let result = add(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // Case where one of the inputs is a scalar.
        let a = from_scalar(3.0);
        let b = from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let result = add(&a, &b).unwrap();
        let expected = from_data(vec![2, 2], vec![4.0, 5.0, 6.0, 7.0]);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_add_in_place() -> Result<(), String> {
        // In-place addition with inputs that have the same shape.
        let mut a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let a_copy = a.clone();
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![11., 22., 33., 44.]);
        add_in_place(&mut a, &b);
        expect_equal(&a, &expected)?;

        // Run `Add` operator in place with inputs that support in-place addition.
        let op = Add {};
        let result = op
            .run_in_place(Output::FloatTensor(a_copy), &[(&b).into()])
            .unwrap();
        expect_equal(result.as_float_ref().unwrap(), &expected)?;

        // Run `Add` operator in-place with inputs that don't support in-place
        // addition. The operator should fall back to creating a new output tensor.
        let scalar = from_scalar(1.0);
        let expected = from_data(vec![2, 2], vec![11., 21., 31., 41.]);
        let result = op
            .run_in_place(Output::FloatTensor(scalar), &[(&b).into()])
            .unwrap();
        expect_equal(result.as_float_ref().unwrap(), &expected)?;

        // In-place addition where the second input must be broadcast to the
        // shape of the first.
        let mut a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_vec(vec![1., 2.]);
        let expected = from_data(vec![2, 2], vec![2., 4., 4., 6.]);

        add_in_place(&mut a, &b);
        expect_equal(&a, &expected)?;

        // In-place addition where the second input must be broadcast to the
        // shape of the first, and the first has a non-contiguous layout.
        let mut a = from_data(vec![2, 3], vec![1., 2., 0., 3., 4., 0.]);
        a.clip_dim(1, 0, 2);
        assert!(!a.is_contiguous());
        let b = from_vec(vec![1., 2.]);
        let expected = from_data(vec![2, 2], vec![2., 4., 4., 6.]);

        add_in_place(&mut a, &b);
        expect_equal(&a, &expected)
    }

    #[test]
    fn test_add_invalid_broadcast() {
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);

        let op = Add {};
        let result = op.run(&[(&a).into(), (&b).into()]);

        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Cannot broadcast inputs to compatible shape"
            ))
        );
    }

    #[test]
    fn test_div() -> Result<(), String> {
        let a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let expected = from_data(vec![2, 2], vec![10., 10., 10., 10.]);
        let result = div(&a, &b).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_div_in_place() -> Result<(), String> {
        let mut a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let expected = from_data(vec![2, 2], vec![10., 10., 10., 10.]);
        div_in_place(&mut a, &b);
        expect_equal(&a, &expected)
    }

    #[test]
    fn test_mul() -> Result<(), String> {
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![10., 40., 90., 160.]);
        let result = mul(&a, &b).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_mul_in_place() -> Result<(), String> {
        let mut a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![10., 40., 90., 160.]);
        mul_in_place(&mut a, &b);
        expect_equal(&a, &expected)
    }

    #[test]
    fn test_sub() -> Result<(), String> {
        let a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let expected = from_data(vec![2, 2], vec![9., 18., 27., 36.]);
        let result = sub(&a, &b).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sub_in_place() -> Result<(), String> {
        let mut a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let expected = from_data(vec![2, 2], vec![9., 18., 27., 36.]);
        sub_in_place(&mut a, &b);
        expect_equal(&a, &expected)
    }
}
