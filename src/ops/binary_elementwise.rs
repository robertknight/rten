use std::fmt::Debug;
use std::iter::zip;

use crate::ops::{from_data, get_input_as_float, Input, OpError, Operator, Output};
use crate::tensor::Tensor;

/// Given the shapes of two inputs to a binary operation, choose the one that
/// will be used as the output shape. The other tensor will be broadcasted
/// to match.
fn choose_broadcast_shape<'a>(a: &'a [usize], b: &'a [usize]) -> &'a [usize] {
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
    a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()
}

/// Perform an elementwise binary operation in-place.
///
/// This currently only supports the case where both inputs have exactly the
/// same shape, so no broadcasting is required, and the inputs are contigious.
fn binary_op_in_place<T: Copy + Debug, F: Fn(&mut T, T)>(a: &mut Tensor<T>, b: &Tensor<T>, op: F) {
    assert!(a.is_contiguous());
    assert!(b.is_contiguous());
    for (a_elt, b_elt) in zip(a.data_mut().iter_mut(), b.data().iter()) {
        op(a_elt, *b_elt);
    }
}

/// Perform elementwise addition of two tensors.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    binary_op(a, b, |x, y| x + y)
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

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let a = get_input_as_float(inputs, 0)?;
        let b = get_input_as_float(inputs, 1)?;
        add(a, b).map(|t| t.into())
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Result<Output, OpError> {
        let mut a = input.as_float().ok_or(OpError::UnsupportedInputType)?;
        let b = get_input_as_float(other, 0)?;

        if can_run_binary_op_in_place(&a, b) {
            add_in_place(&mut a, b);
            Ok(a.into())
        } else {
            self.run(&[(&a).into(), b.into()])
        }
    }
}

/// Multiply two tensors elementwise.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    binary_op(a, b, |x, y| x * y)
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

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let a = get_input_as_float(inputs, 0)?;
        let b = get_input_as_float(inputs, 1)?;
        mul(a, b).map(|t| t.into())
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Result<Output, OpError> {
        let mut a = input.as_float().ok_or(OpError::UnsupportedInputType)?;
        let b = get_input_as_float(other, 0)?;

        if can_run_binary_op_in_place(&a, b) {
            mul_in_place(&mut a, b);
            Ok(a.into())
        } else {
            self.run(&[(&a).into(), b.into()])
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{add, add_in_place, mul, mul_in_place, Add, Operator, Output};
    use crate::tensor::{from_data, from_scalar};
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
        // Invoke `add_in_place` directly.
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
        expect_equal(result.as_float_ref().unwrap(), &expected)
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
}
