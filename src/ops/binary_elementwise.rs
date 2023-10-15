use std::fmt::Debug;
use std::iter::{repeat, zip};

use wasnn_tensor::{Layout, Tensor, TensorView, View};

use crate::number::{Identities, IsInt};
use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output};

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
    a: TensorView<T>,
    b: TensorView<T>,
    op: F,
) -> Result<Tensor<R>, OpError> {
    if let Some(scalar) = b.item() {
        return Ok(a.map(|x| op(*x, *scalar)));
    }

    let out_shape = broadcast_shapes(a.shape(), b.shape())
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;

    let a_elts = a.broadcast_iter(&out_shape);
    let b_elts = b.broadcast_iter(&out_shape);
    let out_data: Vec<_> = zip(a_elts, b_elts).map(|(a, b)| op(*a, *b)).collect();
    Ok(Tensor::from_data(&out_shape, out_data))
}

/// Return true if an elementwise binary operation can be performed in-place
/// on `a` given `b` as the other argument.
fn can_run_binary_op_in_place<T: Copy>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    b.can_broadcast_to(a.shape())
}

/// Perform an elementwise binary operation in-place.
///
/// This requires that `b` can be broadcast to the shape of `a`.
fn binary_op_in_place<T: Copy + Debug, F: Fn(T, T) -> T>(
    a: &mut Tensor<T>,
    b: TensorView<T>,
    op: F,
) {
    // Fast paths for contiguous LHS
    if a.is_contiguous() {
        if let Some(scalar) = b.item() {
            // When RHS is a scalar, we don't need to iterate over it at all.
            for a_elt in a.data_mut().iter_mut() {
                *a_elt = op(*a_elt, *scalar);
            }
        } else if a.shape() == b.shape() && b.is_contiguous() {
            // When RHS is contiguous and same shape as LHS we can use a simple iterator.
            for (a_elt, b_elt) in zip(a.data_mut().iter_mut(), b.data().iter()) {
                *a_elt = op(*a_elt, *b_elt);
            }
        } else if &a.shape()[a.ndim() - b.ndim()..] == b.shape() && b.is_contiguous() {
            // Variation of the above for when broadcasting just involves cycling
            // the RHS.
            for (a_elt, b_elt) in zip(a.data_mut().iter_mut(), b.data().iter().cycle()) {
                *a_elt = op(*a_elt, *b_elt);
            }
        } else {
            // Otherwise a more complex RHS iterator is required.
            let b_elts = b.broadcast_iter(a.shape());
            for (a_elt, b_elt) in zip(a.data_mut().iter_mut(), b_elts) {
                *a_elt = op(*a_elt, *b_elt);
            }
        }
        return;
    }

    let b_elts = b.broadcast_iter(a.shape());
    for (a_elt, b_elt) in zip(a.iter_mut(), b_elts) {
        *a_elt = op(*a_elt, *b_elt);
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
fn binary_commutative_op<T: Copy + Debug, F: Fn(T, T) -> T>(
    a: TensorView<T>,
    b: TensorView<T>,
    op: F,
) -> Result<Tensor<T>, OpError> {
    let mut out;
    let other;
    if b.can_broadcast_to(a.shape()) {
        out = a.to_tensor();
        other = b;
    } else if a.can_broadcast_to(b.shape()) {
        out = b.to_tensor();
        other = a;
    } else {
        return binary_op(a, b, op);
    }
    binary_op_in_place(&mut out, other, op);
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
                $op_func(a.view(), b.view()).into_op_result()
            }
            Input::IntTensor(a) => {
                let b = $inputs.require_as::<i32>(1)?;
                $op_func(a.view(), b.view()).into_op_result()
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
                    $in_place_op_func(&mut a, b.view());
                    Ok(a.into())
                } else {
                    $op_func(a.view(), b.view()).map(|t| t.into())
                }
            }
            Output::IntTensor(mut a) => {
                let b = $other.require_as::<i32>(0)?;
                if can_run_binary_op_in_place(&a, b) {
                    $in_place_op_func(&mut a, b.view());
                    Ok(a.into())
                } else {
                    $op_func(a.view(), b.view()).map(|t| t.into())
                }
            }
        }
    }};
}

/// Perform elementwise addition of two tensors.
pub fn add<T: Copy + Debug + std::ops::Add<Output = T>>(
    a: TensorView<T>,
    b: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    binary_commutative_op(a, b, |x, y| x + y)
}

/// Perform in-place elementwise addition of two tensors.
pub fn add_in_place<T: Copy + Debug + std::ops::Add<Output = T>>(
    a: &mut Tensor<T>,
    b: TensorView<T>,
) {
    binary_op_in_place(a, b, |x, y| x + y);
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
    T: Copy + Debug + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + IsInt + Identities,
>(
    a: TensorView<T>,
    b: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    match (T::is_int(), b.item()) {
        // Optimize division as multiplication-by-reciprocal.
        //
        // This loses some precision, so we might want to revisit this in future.
        (false, Some(scalar)) => mul(a, Tensor::from_scalar(T::one() / *scalar).view()),
        _ => binary_op(a, b, |x, y| x / y),
    }
}

/// Perform in-place elementwise division of two tensors.
pub fn div_in_place<
    T: Copy + Debug + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + IsInt + Identities,
>(
    a: &mut Tensor<T>,
    b: TensorView<T>,
) {
    match (T::is_int(), b.item()) {
        (false, Some(scalar)) => mul_in_place(a, Tensor::from_scalar(T::one() / *scalar).view()),
        _ => binary_op_in_place(a, b, |x, y| x / y),
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

enum BooleanOp {
    Equal,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

fn boolean_op<T: Copy + Debug + PartialEq + PartialOrd>(
    a: TensorView<T>,
    b: TensorView<T>,
    op: BooleanOp,
) -> Result<Tensor<i32>, OpError> {
    binary_op(a, b, |x, y| {
        i32::from(match op {
            BooleanOp::Equal => x == y,
            BooleanOp::Less => x < y,
            BooleanOp::LessOrEqual => x <= y,
            BooleanOp::Greater => x > y,
            BooleanOp::GreaterOrEqual => x >= y,
        })
    })
}

/// Define a boolean comparison operator which supports all numeric tensor
/// types.
macro_rules! boolean_cmp_op {
    ($name:ident, $func:ident) => {
        pub fn $func<T: Copy + Debug + PartialEq + PartialOrd>(
            a: TensorView<T>,
            b: TensorView<T>,
        ) -> Result<Tensor<i32>, OpError> {
            boolean_op(a, b, BooleanOp::$name)
        }

        #[derive(Debug)]
        pub struct $name {}

        impl Operator for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
                run_typed_op!(inputs, $func)
            }
        }
    };
}

boolean_cmp_op!(Equal, equal);
boolean_cmp_op!(Greater, greater);
boolean_cmp_op!(GreaterOrEqual, greater_or_equal);
boolean_cmp_op!(Less, less);
boolean_cmp_op!(LessOrEqual, less_or_equal);

/// Calculate the remainder of `x / y` using floored division. See
/// [DivMode] for an explanation.
fn rem_floor<
    T: Copy + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Rem<Output = T>,
>(
    x: T,
    y: T,
) -> T {
    // See https://en.wikipedia.org/wiki/Modulo#Implementing_other_modulo_definitions_using_truncation
    let zero = T::default();
    let mut rem = x % y;
    if rem > zero && y < zero || rem < zero && y > zero {
        rem = rem + y;
    }
    rem
}

/// Division method to use. When both operators to a division or modulus
/// operator are positive, the different methods produce the same results.
///
/// When one or both of the operands is negative however, the different methods
/// produce different results.
///
/// See <https://en.wikipedia.org/wiki/Modulo#Variants_of_the_definition>.
pub enum DivMode {
    /// Use flooring division, like Python's `%` operator and `numpy.mod`.
    FloorDiv,

    /// Use truncated division, like C and Rust's `%` operator and `numpy.fmod`.
    TruncDiv,
}

/// Return the elementwise remainder of dividing `a / b`.
pub fn mod_op<
    T: Copy + Debug + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Rem<Output = T>,
>(
    a: TensorView<T>,
    b: TensorView<T>,
    mode: DivMode,
) -> Result<Tensor<T>, OpError> {
    binary_op(
        a,
        b,
        match mode {
            DivMode::FloorDiv => |x, y| rem_floor(x, y),
            DivMode::TruncDiv => |x, y| x % y,
        },
    )
}

#[derive(Debug)]
pub struct Mod {
    /// If true, use truncated division (see [DivMode::TruncDiv], otherwise
    /// use flooring division (see [DivMode::FloorDiv]).
    pub fmod: bool,
}

impl Operator for Mod {
    fn name(&self) -> &str {
        "Mod"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let a = inputs.require(0)?;
        let mode = if self.fmod {
            DivMode::TruncDiv
        } else {
            DivMode::FloorDiv
        };

        match a {
            Input::FloatTensor(a) => {
                let b = inputs.require_as::<f32>(1)?;
                mod_op(a.view(), b.view(), mode).into_op_result()
            }
            Input::IntTensor(a) => {
                let b = inputs.require_as::<i32>(1)?;
                mod_op(a.view(), b.view(), mode).into_op_result()
            }
        }
    }
}

/// Multiply two tensors elementwise.
pub fn mul<T: Copy + Debug + std::ops::Mul<Output = T>>(
    a: TensorView<T>,
    b: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    binary_commutative_op(a, b, |x, y| x * y)
}

/// Perform in-place elementwise multiplication of two tensors.
pub fn mul_in_place<T: Copy + Debug + std::ops::Mul<Output = T>>(
    a: &mut Tensor<T>,
    b: TensorView<T>,
) {
    binary_op_in_place(a, b, |a_elt, b_elt| a_elt * b_elt);
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
pub fn pow(a: TensorView, b: TensorView) -> Result<Tensor, OpError> {
    if b.item() == Some(&2.0) {
        Ok(a.map(|x| x * x))
    } else {
        binary_op(a, b, |x, y| x.powf(y))
    }
}

/// Perform in-place raise of elements of `a` to power of corresponding elements in `b`.
pub fn pow_in_place(a: &mut Tensor, b: TensorView) {
    if b.item() == Some(&2.0) {
        a.apply(|x| x * x);
    } else {
        binary_op_in_place(a, b, |a_elt, b_elt| a_elt.powf(b_elt));
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
        pow(a.view(), b.view()).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        let mut a = input.into_float().ok_or(OpError::IncorrectInputType)?;
        let b = other.require_as(0)?;

        if can_run_binary_op_in_place(&a, b) {
            pow_in_place(&mut a, b.view());
            Ok(a.into())
        } else {
            pow(a.view(), b.view()).map(|t| t.into())
        }
    }
}

/// Perform elementwise subtraction of two tensors.
pub fn sub<T: Copy + Debug + std::ops::Sub<Output = T>>(
    a: TensorView<T>,
    b: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    binary_op(a, b, |x, y| x - y)
}

/// Perform in-place elementwise subtraction of two tensors.
pub fn sub_in_place<T: Copy + Debug + std::ops::Sub<Output = T>>(
    a: &mut Tensor<T>,
    b: TensorView<T>,
) {
    binary_op_in_place(a, b, |x, y| x - y);
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
    cond: TensorView<i32>,
    x: TensorView<T>,
    y: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    let broadcast_xy_shape = broadcast_shapes(x.shape(), y.shape())
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;
    let result_shape = broadcast_shapes(cond.shape(), &broadcast_xy_shape)
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;

    let result_elts: Vec<_> = zip(
        cond.broadcast_iter(&result_shape),
        zip(
            x.broadcast_iter(&result_shape),
            y.broadcast_iter(&result_shape),
        ),
    )
    .map(|(&cond, (&x, &y))| if cond != 0 { x } else { y })
    .collect();
    Ok(Tensor::from_data(&result_shape, result_elts))
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
                let y: &Tensor = y.try_into()?;
                where_op(condition.view(), x.view(), y.view()).into_op_result()
            }
            Input::IntTensor(x) => {
                let y: &Tensor<i32> = y.try_into()?;
                where_op(condition.view(), x.view(), y.view()).into_op_result()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{tensor, Layout, Tensor, View};

    use crate::ops::{
        add, add_in_place, div, div_in_place, equal, greater, greater_or_equal, less,
        less_or_equal, mod_op, mul, mul_in_place, pow, pow_in_place, sub, sub_in_place, where_op,
        Add, DivMode, OpError, Operator, Output,
    };

    #[test]
    fn test_add() -> Result<(), String> {
        // Float tensor
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[2, 2], vec![11., 22., 33., 44.]);
        let result = add(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let expected = Tensor::from_data(&[2, 2], vec![11, 22, 33, 44]);
        let result = add(a.view(), b.view()).unwrap();
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_add_broadcasted() -> Result<(), String> {
        // Simple case where comparing ordering of tensor shapes tells us
        // target shape.
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[1], vec![10.]);
        let expected = Tensor::from_data(&[2, 2], vec![11., 12., 13., 14.]);
        let result = add(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Try alternative ordering for inputs.
        let result = add(b.view(), a.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Case where the length of tensor shapes needs to be compared before
        // the ordering, since ([5] > [1,5]).
        let a = Tensor::from_data(&[5], vec![1., 2., 3., 4., 5.]);
        let b = Tensor::from_data(&[1, 5], vec![1., 2., 3., 4., 5.]);
        let expected = Tensor::from_data(&[1, 5], vec![2., 4., 6., 8., 10.]);

        let result = add(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Case where one of the inputs is a scalar.
        let a = Tensor::from_scalar(3.0);
        let b = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let result = add(a.view(), b.view()).unwrap();
        let expected = Tensor::from_data(&[2, 2], vec![4.0, 5.0, 6.0, 7.0]);
        expect_equal(&result, &expected)?;

        // Case where broadcast shape uses dimensions from both inputs.
        let a = Tensor::from_data(&[2, 1], vec![1, 2]);
        let b = Tensor::from_data(&[1, 2], vec![3, 4]);
        let result = add(a.view(), b.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), &[4, 5, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_add_broadcast_first_input() {
        let a: Tensor<i32> = Tensor::zeros(&[1, 1, 10]);
        let b = Tensor::zeros(&[1, 5, 10]);
        let result = add(a.view(), b.view()).unwrap();
        assert_eq!(result.shape(), &[1, 5, 10]);
    }

    #[test]
    fn test_add_in_place() -> Result<(), String> {
        // In-place addition with float inputs that have the same shape.
        let mut a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let a_copy = a.clone();
        let b = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[2, 2], vec![11., 22., 33., 44.]);
        add_in_place(&mut a, b.view());
        expect_equal(&a, &expected)?;

        // In-place addition with int inputs that have the same shape.
        let mut a_ints = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let b_ints = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let expected_ints = Tensor::from_data(&[2, 2], vec![11, 22, 33, 44]);
        add_in_place(&mut a_ints, b_ints.view());
        assert_eq!(&a_ints, &expected_ints);

        // Run `Add` operator in place with inputs that support in-place addition.
        let op = Add {};
        let result = op
            .run_in_place(Output::FloatTensor(a_copy), (&b).into())
            .unwrap();
        expect_equal(result.as_float_ref().unwrap(), &expected)?;

        // Run `Add` operator in-place with inputs that don't support in-place
        // addition. The operator should fall back to creating a new output tensor.
        let scalar = Tensor::from_scalar(1.0);
        let expected = Tensor::from_data(&[2, 2], vec![11., 21., 31., 41.]);
        let result = op
            .run_in_place(Output::FloatTensor(scalar), (&b).into())
            .unwrap();
        expect_equal(result.as_float_ref().unwrap(), &expected)?;

        // In-place addition where the second input must be broadcast to the
        // shape of the first.
        let mut a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = tensor!([1., 2.]);
        let expected = Tensor::from_data(&[2, 2], vec![2., 4., 4., 6.]);

        add_in_place(&mut a, b.view());
        expect_equal(&a, &expected)?;

        // In-place addition where the second input must be broadcast to the
        // shape of the first, and the first has a non-contiguous layout.
        let mut a = Tensor::from_data(&[2, 3], vec![1., 2., 0., 3., 4., 0.]);
        a.clip_dim(1, 0..2);
        assert!(!a.is_contiguous());
        let b = tensor!([1., 2.]);
        let expected = Tensor::from_data(&[2, 2], vec![2., 4., 4., 6.]);

        add_in_place(&mut a, b.view());
        expect_equal(&a, &expected)
    }

    #[test]
    fn test_add_invalid_broadcast() {
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);

        let op = Add {};
        let result = op.run((&a, &b).into());

        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );
    }

    #[test]
    fn test_div() -> Result<(), String> {
        // Non-scalar a and b
        let a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let expected = Tensor::from_data(&[2, 2], vec![10., 10., 10., 10.]);
        let result = div(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Scalar b
        let a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_scalar(10.);
        let expected = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let result = div(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Non-scalar a and b ints
        let a = tensor!([1, 2, 3, 4]);
        let b = tensor!([2, 2, 2, 2]);
        let expected = tensor!([0, 1, 1, 2]);
        let result = div(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Scalar b int
        let a = tensor!([1, 2, 3, 4]);
        let b = Tensor::from_scalar(2);
        let expected = tensor!([0, 1, 1, 2]);
        let result = div(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_div_in_place() -> Result<(), String> {
        // Non-scalar a and b
        let mut a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let expected = Tensor::from_data(&[2, 2], vec![10., 10., 10., 10.]);
        div_in_place(&mut a, b.view());
        expect_equal(&a, &expected)?;

        // Scalar b
        let mut a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_scalar(10.);
        let expected = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        div_in_place(&mut a, b.view());
        expect_equal(&a, &expected)?;

        // Non-scalar a and b ints
        let mut a = tensor!([1, 2, 3, 4]);
        let b = tensor!([2, 2, 2, 2]);
        let expected = tensor!([0, 1, 1, 2]);
        div_in_place(&mut a, b.view());
        assert_eq!(&a, &expected);

        // Scalar b int
        let mut a = tensor!([1, 2, 3, 4]);
        let b = Tensor::from_scalar(2);
        let expected = tensor!([0, 1, 1, 2]);
        div_in_place(&mut a, b.view());
        assert_eq!(&a, &expected);

        Ok(())
    }

    #[test]
    fn test_equal() {
        // Int tensor
        let a = tensor!([1, 2]);
        let b = tensor!([1, 3]);
        let expected = tensor!([1, 0]);
        let result = equal(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = tensor!([1., 2.]);
        let b = tensor!([1., 3.]);
        let expected = tensor!([1, 0]);
        let result = equal(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_greater() {
        // Int tensor
        let a = tensor!([1, 2, 5]);
        let b = tensor!([1, 3, 4]);
        let expected = tensor!([0, 0, 1]);
        let result = greater(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = tensor!([1., 2., 5.]);
        let b = tensor!([1., 3., 4.]);
        let expected = tensor!([0, 0, 1]);
        let result = greater(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_greater_or_equal() {
        // Int tensor
        let a = tensor!([1, 2, 5]);
        let b = tensor!([1, 3, 4]);
        let expected = tensor!([1, 0, 1]);
        let result = greater_or_equal(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = tensor!([1., 2., 5.]);
        let b = tensor!([1., 3., 4.]);
        let expected = tensor!([1, 0, 1]);
        let result = greater_or_equal(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_less() {
        // Int tensor
        let a = tensor!([1, 2]);
        let b = tensor!([1, 3]);
        let expected = tensor!([0, 1]);
        let result = less(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = tensor!([1., 2.]);
        let b = tensor!([1., 3.]);
        let expected = tensor!([0, 1]);
        let result = less(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_less_or_equal() {
        // Int tensor
        let a = tensor!([1, 2, 5]);
        let b = tensor!([1, 3, 4]);
        let expected = tensor!([1, 1, 0]);
        let result = less_or_equal(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = tensor!([1., 2., 5.]);
        let b = tensor!([1., 3., 4.]);
        let expected = tensor!([1, 1, 0]);
        let result = less_or_equal(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_mod_op() {
        // Int tensor, floor division (like Python's `%`, `numpy.mod`).
        let a = tensor!([10, -10, 10, -10]);
        let b = tensor!([3, 3, -3, -3]);
        let expected = tensor!([1, 2, -2, -1]);
        let result = mod_op(a.view(), b.view(), DivMode::FloorDiv).unwrap();
        assert_eq!(&result, &expected);

        // Int tensor, truncated division (like Rust's `%`, `numpy.fmod`).
        let expected = tensor!([1, -1, 1, -1]);
        let result = mod_op(a.view(), b.view(), DivMode::TruncDiv).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor, floor division.
        let af = tensor!([3.5, -3.5, 3.5, -3.5]);
        let bf = tensor!([2.5, 2.5, -2.5, -2.5]);
        let expected = tensor!([1., 1.5, -1.5, -1.]);
        let result = mod_op(af.view(), bf.view(), DivMode::FloorDiv).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor, truncated division.
        let expected = tensor!([1., -1., 1., -1.]);
        let result = mod_op(af.view(), bf.view(), DivMode::TruncDiv).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_mul() -> Result<(), String> {
        // Float tensor
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[2, 2], vec![10., 40., 90., 160.]);
        let result = mul(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let expected = Tensor::from_data(&[2, 2], vec![10, 40, 90, 160]);
        let result = mul(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_mul_in_place() -> Result<(), String> {
        // Float tensor
        let mut a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[2, 2], vec![10., 40., 90., 160.]);
        mul_in_place(&mut a, b.view());
        expect_equal(&a, &expected)?;

        // Int tensor
        let mut a = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let expected = Tensor::from_data(&[2, 2], vec![10, 40, 90, 160]);
        mul_in_place(&mut a, b.view());
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
                a: tensor!([2., 3., 4.]),
                b: Tensor::from_scalar(2.),
                expected: tensor!([4., 9., 16.]),
            },
            // Raise all inputs to scalar
            Case {
                a: tensor!([2., 3., 4.]),
                b: Tensor::from_scalar(3.),
                expected: tensor!([8., 27., 64.]),
            },
            // Raise each input to different powers
            Case {
                a: tensor!([2., 3., 4.]),
                b: tensor!([1., 2., 3.]),
                expected: tensor!([2., 9., 64.]),
            },
        ];

        for case in cases {
            // Copying variant
            let result = pow(case.a.view(), case.b.view()).unwrap();
            expect_equal(&result, &case.expected)?;

            // In-place variant
            let mut a = case.a.clone();
            pow_in_place(&mut a, case.b.view());
            expect_equal(&a, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_sub() -> Result<(), String> {
        // Float tensor
        let a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let expected = Tensor::from_data(&[2, 2], vec![9., 18., 27., 36.]);
        let result = sub(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let b = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let expected = Tensor::from_data(&[2, 2], vec![9, 18, 27, 36]);
        let result = sub(a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_sub_in_place() -> Result<(), String> {
        // Float tensor
        let mut a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let expected = Tensor::from_data(&[2, 2], vec![9., 18., 27., 36.]);
        sub_in_place(&mut a, b.view());
        expect_equal(&a, &expected)?;

        // Int tensor
        let mut a = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let b = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let expected = Tensor::from_data(&[2, 2], vec![9, 18, 27, 36]);
        sub_in_place(&mut a, b.view());
        assert_eq!(&a, &expected);

        Ok(())
    }

    #[test]
    fn test_where() {
        // Float tensor with exact matching shapes
        let cond = Tensor::from_data(&[2, 2], vec![1, 0, 0, 1]);
        let x = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let y = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let result = where_op(cond.view(), x.view(), y.view()).unwrap();
        let expected = Tensor::from_data(&[2, 2], vec![1., 20., 30., 4.]);
        assert_eq!(&result, &expected);

        // Float tensor broadcasting `x` and `y`
        let cond = tensor!([1, 1, 0, 0]);
        let x = Tensor::from_scalar(1.);
        let y = Tensor::from_scalar(2.);
        let result = where_op(cond.view(), x.view(), y.view()).unwrap();
        let expected = tensor!([1., 1., 2., 2.]);
        assert_eq!(&result, &expected);

        // Float tensor broadcasting `cond`
        let cond = Tensor::from_scalar(1);
        let x = tensor!([1., 2.]);
        let y = tensor!([3., 4.]);
        let result = where_op(cond.view(), x.view(), y.view()).unwrap();
        let expected = tensor!([1., 2.]);
        assert_eq!(&result, &expected);

        // Int tensor broadcasting `x` and `y`
        let cond = tensor!([1, 1, 0, 0]);
        let x = Tensor::from_scalar(3);
        let y = Tensor::from_scalar(4);
        let result = where_op(cond.view(), x.view(), y.view()).unwrap();
        let expected = tensor!([3, 3, 4, 4]);
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_where_invalid_inputs() {
        let cond = tensor!([1, 1]);
        let x = tensor!([1, 2, 3]);
        let y = tensor!([2, 2]);

        // Failure to broadcast `x` to match `cond`
        let result = where_op(cond.view(), x.view(), y.view());
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );

        // Failure to broadcast `y` to match `cond`
        let result = where_op(cond.view(), y.view(), x.view());
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );
    }
}
