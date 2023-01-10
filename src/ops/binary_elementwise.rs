use std::fmt::Debug;
use std::iter::{repeat, zip};

use crate::number::{Identities, IsInt};
use crate::ops::{from_data, Input, InputList, IntoOpResult, OpError, Operator, Output};
use crate::tensor::Tensor;

/// Given the shapes of two inputs to a binary operation, return the shape
/// that will result from broadcasting them following NumPy rules or `None`
/// if the shapes are not compatible.
///
/// Broadcasting works by left-padding the input shapes with 1s so they are
/// the same length, then matching dimensions starting from the right. For
/// each dimension, the values are compatible if they are the same or one of
/// them is 1. The larger of the two values is the size of that dimension in
/// the output shape.
///
/// See https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let a_pad = b.len().saturating_sub(a.len());
    let b_pad = a.len().saturating_sub(b.len());

    let a_iter = a.iter().copied().rev().chain(repeat(1).take(a_pad));
    let b_iter = b.iter().copied().rev().chain(repeat(1).take(b_pad));

    let mut result = Vec::with_capacity(a.len().max(b.len()));
    for (a, b) in zip(a_iter, b_iter) {
        if a == b {
            result.push(a);
        } else if a == 1 {
            result.push(b);
        } else if b == 1 {
            result.push(a);
        } else {
            println!("cannot match {} and {}", a, b);
            return None;
        }
    }
    result.reverse();

    Some(result)
}

/// Compute the result of applying the binary operation `op` to corresponding
/// elements of `a` and `b`. The shapes of `a` and `b` are broadcast to a
/// matching shape if necessary.
fn binary_op<T: Copy + Debug, R: Copy, F: Fn(T, T) -> R>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    op: F,
) -> Result<Tensor<R>, OpError> {
    if let Some(scalar) = b.item() {
        return Ok(a.map(|x| op(x, scalar)));
    }

    let out_shape = broadcast_shapes(a.shape(), b.shape())
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;

    let a_elts = a.broadcast_elements(&out_shape);
    let b_elts = b.broadcast_elements(&out_shape);
    let out_data = zip(a_elts, b_elts).map(|(a, b)| op(a, b)).collect();
    Ok(from_data(out_shape.into(), out_data))
}

/// Return true if an elementwise binary operation can be performed in-place
/// on `a` given `b` as the other argument.
fn can_run_binary_op_in_place<T: Copy>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    b.can_broadcast_to(a.shape())
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
        } else if &a.shape()[a.ndim() - b.ndim()..] == b.shape() && b.is_contiguous() {
            // Variation of the above for when broadcasting just involves cycling
            // the RHS.
            for (a_elt, b_elt) in zip(a.data_mut().iter_mut(), b.data().iter().cycle()) {
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
///
/// When broadcasting is involved, the output may be larger than either of the
/// inputs (eg. if the inputs are `[1, 5]` and `[5, 1]` respectively). In that
/// case this falls back to a non-place op.
fn binary_commutative_op<T: Copy + Debug, F: Fn(&mut T, T), F2: Fn(T, T) -> T>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    op_mut: F,
    op: F2,
) -> Result<Tensor<T>, OpError> {
    let mut out;
    let other;
    if b.can_broadcast_to(a.shape()) {
        out = a.clone();
        other = b;
    } else if a.can_broadcast_to(b.shape()) {
        out = b.clone();
        other = a;
    } else {
        return binary_op(a, b, op);
    }
    binary_op_in_place(&mut out, other, op_mut);
    Ok(out)
}

/// Extract two input operands from `$inputs` and invoke the appropriate
/// instantiation of `$op_func` depending on the tensor type.
macro_rules! run_typed_op {
    ($inputs:expr, $op_func:ident) => {{
        let a = $inputs.require(0)?;
        match a {
            Input::FloatTensor(a) => {
                let b = $inputs.require_as::<f32>(1)?;
                $op_func(a, b).into_op_result()
            }
            Input::IntTensor(a) => {
                let b = $inputs.require_as::<i32>(1)?;
                $op_func(a, b).into_op_result()
            }
        }
    }};
}

/// Extract two input operands from `$input` and `$other` and invoke the
/// appropriate instantiations of `$in_place_op_func` or `$op_func` depending
/// on the tensor type.
macro_rules! run_typed_op_in_place {
    ($input:expr, $other: expr, $in_place_op_func:ident, $op_func:ident) => {{
        match $input {
            Output::FloatTensor(mut a) => {
                let b = $other.require_as::<f32>(0)?;
                if can_run_binary_op_in_place(&a, b) {
                    $in_place_op_func(&mut a, b);
                    Ok(a.into())
                } else {
                    $op_func(&a, b).map(|t| t.into())
                }
            }
            Output::IntTensor(mut a) => {
                let b = $other.require_as::<i32>(0)?;
                if can_run_binary_op_in_place(&a, b) {
                    $in_place_op_func(&mut a, b);
                    Ok(a.into())
                } else {
                    $op_func(&a, b).map(|t| t.into())
                }
            }
        }
    }};
}

/// Perform elementwise addition of two tensors.
pub fn add<T: Copy + Debug + std::ops::Add<Output = T> + std::ops::AddAssign>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, OpError> {
    binary_commutative_op(a, b, |x, y| *x += y, |x, y| x + y)
}

/// Perform in-place elementwise addition of two tensors.
pub fn add_in_place<T: Copy + Debug + std::ops::AddAssign>(a: &mut Tensor<T>, b: &Tensor<T>) {
    binary_op_in_place(a, b, |x, y| *x += y);
}

#[derive(Debug)]
pub struct Add {}

impl Operator for Add {
    fn name(&self) -> &str {
        "Add"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        run_typed_op!(inputs, add)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        run_typed_op_in_place!(input, other, add_in_place, add)
    }
}

/// Perform elementwise division of two tensors.
pub fn div<
    T: Copy
        + Debug
        + std::ops::Mul<Output = T>
        + std::ops::MulAssign
        + std::ops::Div<Output = T>
        + IsInt
        + Identities,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, OpError> {
    match (T::is_int(), b.item()) {
        // Optimize division as multiplication-by-reciprocal.
        //
        // This loses some precision, so we might want to revisit this in future.
        (false, Some(scalar)) => mul(a, &Tensor::from_scalar(T::one() / scalar)),
        _ => binary_op(a, b, |x, y| x / y),
    }
}

/// Perform in-place elementwise division of two tensors.
pub fn div_in_place<
    T: Copy
        + Debug
        + std::ops::Mul<Output = T>
        + std::ops::MulAssign
        + std::ops::Div<Output = T>
        + std::ops::DivAssign
        + IsInt
        + Identities,
>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
) {
    match (T::is_int(), b.item()) {
        (false, Some(scalar)) => mul_in_place(a, &Tensor::from_scalar(T::one() / scalar)),
        _ => binary_op_in_place(a, b, |x, y| *x /= y),
    }
}

#[derive(Debug)]
pub struct Div {}

impl Operator for Div {
    fn name(&self) -> &str {
        "Div"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        run_typed_op!(inputs, div)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        run_typed_op_in_place!(input, other, div_in_place, div)
    }
}

pub fn equal<T: Copy + Debug + PartialEq>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<i32>, OpError> {
    binary_op(a, b, |x, y| i32::from(x == y))
}

#[derive(Debug)]
pub struct Equal {}

impl Operator for Equal {
    fn name(&self) -> &str {
        "Equal"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        run_typed_op!(inputs, equal)
    }
}

pub fn less<T: Copy + Debug + PartialOrd>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<i32>, OpError> {
    binary_op(a, b, |x, y| i32::from(x < y))
}

#[derive(Debug)]
pub struct Less {}

impl Operator for Less {
    fn name(&self) -> &str {
        "Less"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        run_typed_op!(inputs, less)
    }
}

/// Multiply two tensors elementwise.
pub fn mul<T: Copy + Debug + std::ops::Mul<Output = T> + std::ops::MulAssign>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, OpError> {
    binary_commutative_op(a, b, |x, y| *x *= y, |x, y| x * y)
}

/// Perform in-place elementwise multiplication of two tensors.
pub fn mul_in_place<T: Copy + Debug + std::ops::MulAssign>(a: &mut Tensor<T>, b: &Tensor<T>) {
    binary_op_in_place(a, b, |a_elt, b_elt| *a_elt *= b_elt);
}

#[derive(Debug)]
pub struct Mul {}

impl Operator for Mul {
    fn name(&self) -> &str {
        "Mul"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        run_typed_op!(inputs, mul)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        run_typed_op_in_place!(input, other, mul_in_place, mul)
    }
}

/// Raise elements of `a` to powers of corresponding elements in `b`.
pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    if b.item() == Some(2.0) {
        Ok(a.map(|x| x * x))
    } else {
        binary_op(a, b, |x, y| x.powf(y))
    }
}

/// Perform in-place raise of elements of `a` to power of corresponding elements in `b`.
pub fn pow_in_place(a: &mut Tensor, b: &Tensor) {
    if b.item() == Some(2.0) {
        a.apply(|x| x * x);
    } else {
        binary_op_in_place(a, b, |a_elt, b_elt| *a_elt = a_elt.powf(b_elt));
    }
}

#[derive(Debug)]
pub struct Pow {}

impl Operator for Pow {
    fn name(&self) -> &str {
        "Pow"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        pow(a, b).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        let mut a = input.into_float().ok_or(OpError::IncorrectInputType)?;
        let b = other.require_as(0)?;

        if can_run_binary_op_in_place(&a, b) {
            pow_in_place(&mut a, b);
            Ok(a.into())
        } else {
            pow(&a, b).map(|t| t.into())
        }
    }
}

/// Perform elementwise subtraction of two tensors.
pub fn sub<T: Copy + Debug + std::ops::Sub<Output = T>>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, OpError> {
    binary_op(a, b, |x, y| x - y)
}

/// Perform in-place elementwise subtraction of two tensors.
pub fn sub_in_place<T: Copy + Debug + std::ops::SubAssign>(a: &mut Tensor<T>, b: &Tensor<T>) {
    binary_op_in_place(a, b, |x, y| *x -= y);
}

#[derive(Debug)]
pub struct Sub {}

impl Operator for Sub {
    fn name(&self) -> &str {
        "Sub"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        run_typed_op!(inputs, sub)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        run_typed_op_in_place!(input, other, sub_in_place, sub)
    }
}

pub fn where_op<T: Copy>(
    cond: &Tensor<i32>,
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> Result<Tensor<T>, OpError> {
    let broadcast_xy_shape = broadcast_shapes(x.shape(), y.shape())
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;
    let result_shape = broadcast_shapes(cond.shape(), &broadcast_xy_shape)
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;

    let result_elts = zip(
        cond.broadcast_elements(&result_shape),
        zip(
            x.broadcast_elements(&result_shape),
            y.broadcast_elements(&result_shape),
        ),
    )
    .map(|(cond, (x, y))| if cond != 0 { x } else { y })
    .collect();
    Ok(from_data(result_shape.into(), result_elts))
}

#[derive(Debug)]
pub struct Where {}

impl Operator for Where {
    fn name(&self) -> &str {
        "Where"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let condition = inputs.require_as::<i32>(0)?;
        let x = inputs.require(1)?;
        let y = inputs.require(2)?;
        match x {
            Input::FloatTensor(x) => {
                let y = y.try_into()?;
                where_op(condition, x, y).into_op_result()
            }
            Input::IntTensor(x) => {
                let y = y.try_into()?;
                where_op(condition, x, y).into_op_result()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{
        add, add_in_place, div, div_in_place, equal, less, mul, mul_in_place, pow, pow_in_place,
        sub, sub_in_place, where_op, Add, InputList, OpError, Operator, Output,
    };
    use crate::tensor::{from_data, from_scalar, from_vec, Tensor};
    use crate::test_util::expect_equal;

    #[test]
    fn test_add() -> Result<(), String> {
        // Float tensor
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![11., 22., 33., 44.]);
        let result = add(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = from_data(vec![2, 2], vec![1, 2, 3, 4]);
        let b = from_data(vec![2, 2], vec![10, 20, 30, 40]);
        let expected = from_data(vec![2, 2], vec![11, 22, 33, 44]);
        let result = add(&a, &b).unwrap();
        assert_eq!(result, expected);

        Ok(())
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
        expect_equal(&result, &expected)?;

        // Case where broadcast shape uses dimensions from both inputs.
        let a = from_data(vec![2, 1], vec![1, 2]);
        let b = from_data(vec![1, 2], vec![3, 4]);
        let result = add(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.elements_vec(), &[4, 5, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_add_broadcast_first_input() {
        let a: Tensor<i32> = Tensor::zeros(&[1, 1, 10]);
        let b = Tensor::zeros(&[1, 5, 10]);
        let result = add(&a, &b).unwrap();
        assert_eq!(result.shape(), &[1, 5, 10]);
    }

    #[test]
    fn test_add_in_place() -> Result<(), String> {
        // In-place addition with float inputs that have the same shape.
        let mut a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let a_copy = a.clone();
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![11., 22., 33., 44.]);
        add_in_place(&mut a, &b);
        expect_equal(&a, &expected)?;

        // In-place addition with int inputs that have the same shape.
        let mut a_ints = from_data(vec![2, 2], vec![1, 2, 3, 4]);
        let b_ints = from_data(vec![2, 2], vec![10, 20, 30, 40]);
        let expected_ints = from_data(vec![2, 2], vec![11, 22, 33, 44]);
        add_in_place(&mut a_ints, &b_ints);
        assert_eq!(&a_ints, &expected_ints);

        // Run `Add` operator in place with inputs that support in-place addition.
        let op = Add {};
        let result = op
            .run_in_place(Output::FloatTensor(a_copy), InputList::from(&[(&b).into()]))
            .unwrap();
        expect_equal(result.as_float_ref().unwrap(), &expected)?;

        // Run `Add` operator in-place with inputs that don't support in-place
        // addition. The operator should fall back to creating a new output tensor.
        let scalar = from_scalar(1.0);
        let expected = from_data(vec![2, 2], vec![11., 21., 31., 41.]);
        let result = op
            .run_in_place(Output::FloatTensor(scalar), InputList::from(&[(&b).into()]))
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
        let result = op.run(InputList::from(&[(&a).into(), (&b).into()]));

        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );
    }

    #[test]
    fn test_div() -> Result<(), String> {
        // Non-scalar a and b
        let a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let expected = from_data(vec![2, 2], vec![10., 10., 10., 10.]);
        let result = div(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // Scalar b
        let a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_scalar(10.);
        let expected = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let result = div(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // Non-scalar a and b ints
        let a = from_vec(vec![1, 2, 3, 4]);
        let b = from_vec(vec![2, 2, 2, 2]);
        let expected = from_vec(vec![0, 1, 1, 2]);
        let result = div(&a, &b).unwrap();
        assert_eq!(&result, &expected);

        // Scalar b int
        let a = from_vec(vec![1, 2, 3, 4]);
        let b = from_scalar(2);
        let expected = from_vec(vec![0, 1, 1, 2]);
        let result = div(&a, &b).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_div_in_place() -> Result<(), String> {
        // Non-scalar a and b
        let mut a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let expected = from_data(vec![2, 2], vec![10., 10., 10., 10.]);
        div_in_place(&mut a, &b);
        expect_equal(&a, &expected)?;

        // Scalar b
        let mut a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_scalar(10.);
        let expected = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        div_in_place(&mut a, &b);
        expect_equal(&a, &expected)?;

        // Non-scalar a and b ints
        let mut a = from_vec(vec![1, 2, 3, 4]);
        let b = from_vec(vec![2, 2, 2, 2]);
        let expected = from_vec(vec![0, 1, 1, 2]);
        div_in_place(&mut a, &b);
        assert_eq!(&a, &expected);

        // Scalar b int
        let mut a = from_vec(vec![1, 2, 3, 4]);
        let b = from_scalar(2);
        let expected = from_vec(vec![0, 1, 1, 2]);
        div_in_place(&mut a, &b);
        assert_eq!(&a, &expected);

        Ok(())
    }

    #[test]
    fn test_equal() {
        // Int tensor
        let a = from_vec(vec![1, 2]);
        let b = from_vec(vec![1, 3]);
        let expected = from_vec(vec![1, 0]);
        let result = equal(&a, &b).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = from_vec(vec![1., 2.]);
        let b = from_vec(vec![1., 3.]);
        let expected = from_vec(vec![1, 0]);
        let result = equal(&a, &b).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_less() {
        // Int tensor
        let a = from_vec(vec![1, 2]);
        let b = from_vec(vec![1, 3]);
        let expected = from_vec(vec![0, 1]);
        let result = less(&a, &b).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = from_vec(vec![1., 2.]);
        let b = from_vec(vec![1., 3.]);
        let expected = from_vec(vec![0, 1]);
        let result = less(&a, &b).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_mul() -> Result<(), String> {
        // Float tensor
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![10., 40., 90., 160.]);
        let result = mul(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = from_data(vec![2, 2], vec![1, 2, 3, 4]);
        let b = from_data(vec![2, 2], vec![10, 20, 30, 40]);
        let expected = from_data(vec![2, 2], vec![10, 40, 90, 160]);
        let result = mul(&a, &b).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_mul_in_place() -> Result<(), String> {
        // Float tensor
        let mut a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![10., 40., 90., 160.]);
        mul_in_place(&mut a, &b);
        expect_equal(&a, &expected)?;

        // Int tensor
        let mut a = from_data(vec![2, 2], vec![1, 2, 3, 4]);
        let b = from_data(vec![2, 2], vec![10, 20, 30, 40]);
        let expected = from_data(vec![2, 2], vec![10, 40, 90, 160]);
        mul_in_place(&mut a, &b);
        assert_eq!(&a, &expected);

        Ok(())
    }

    #[test]
    fn test_pow() -> Result<(), String> {
        struct Case {
            a: Tensor<f32>,
            b: Tensor<f32>,
            expected: Tensor<f32>,
        }

        let cases = [
            // Square input
            Case {
                a: from_vec(vec![2., 3., 4.]),
                b: from_scalar(2.),
                expected: from_vec(vec![4., 9., 16.]),
            },
            // Raise all inputs to scalar
            Case {
                a: from_vec(vec![2., 3., 4.]),
                b: from_scalar(3.),
                expected: from_vec(vec![8., 27., 64.]),
            },
            // Raise each input to different powers
            Case {
                a: from_vec(vec![2., 3., 4.]),
                b: from_vec(vec![1., 2., 3.]),
                expected: from_vec(vec![2., 9., 64.]),
            },
        ];

        for case in cases {
            // Copying variant
            let result = pow(&case.a, &case.b).unwrap();
            expect_equal(&result, &case.expected)?;

            // In-place variant
            let mut a = case.a.clone();
            pow_in_place(&mut a, &case.b);
            expect_equal(&a, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_sub() -> Result<(), String> {
        // Float tensor
        let a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let expected = from_data(vec![2, 2], vec![9., 18., 27., 36.]);
        let result = sub(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = from_data(vec![2, 2], vec![10, 20, 30, 40]);
        let b = from_data(vec![2, 2], vec![1, 2, 3, 4]);
        let expected = from_data(vec![2, 2], vec![9, 18, 27, 36]);
        let result = sub(&a, &b).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_sub_in_place() -> Result<(), String> {
        // Float tensor
        let mut a = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let b = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let expected = from_data(vec![2, 2], vec![9., 18., 27., 36.]);
        sub_in_place(&mut a, &b);
        expect_equal(&a, &expected)?;

        // Int tensor
        let mut a = from_data(vec![2, 2], vec![10, 20, 30, 40]);
        let b = from_data(vec![2, 2], vec![1, 2, 3, 4]);
        let expected = from_data(vec![2, 2], vec![9, 18, 27, 36]);
        sub_in_place(&mut a, &b);
        assert_eq!(&a, &expected);

        Ok(())
    }

    #[test]
    fn test_where() {
        // Float tensor with exact matching shapes
        let cond = from_data(vec![2, 2], vec![1, 0, 0, 1]);
        let x = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let y = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let result = where_op(&cond, &x, &y).unwrap();
        let expected = from_data(vec![2, 2], vec![1., 20., 30., 4.]);
        assert_eq!(&result, &expected);

        // Float tensor broadcasting `x` and `y`
        let cond = from_vec(vec![1, 1, 0, 0]);
        let x = from_scalar(1.);
        let y = from_scalar(2.);
        let result = where_op(&cond, &x, &y).unwrap();
        let expected = from_vec(vec![1., 1., 2., 2.]);
        assert_eq!(&result, &expected);

        // Float tensor broadcasting `cond`
        let cond = from_scalar(1);
        let x = from_vec(vec![1., 2.]);
        let y = from_vec(vec![3., 4.]);
        let result = where_op(&cond, &x, &y).unwrap();
        let expected = from_vec(vec![1., 2.]);
        assert_eq!(&result, &expected);

        // Int tensor broadcasting `x` and `y`
        let cond = from_vec(vec![1, 1, 0, 0]);
        let x = from_scalar(3);
        let y = from_scalar(4);
        let result = where_op(&cond, &x, &y).unwrap();
        let expected = from_vec(vec![3, 3, 4, 4]);
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_where_invalid_inputs() {
        let cond = from_vec(vec![1, 1]);
        let x = from_vec(vec![1, 2, 3]);
        let y = from_vec(vec![2, 2]);

        // Failure to broadcast `x` to match `cond`
        let result = where_op(&cond, &x, &y);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );

        // Failure to broadcast `y` to match `cond`
        let result = where_op(&cond, &y, &x);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );
    }
}
