extern crate libm;

use std::fmt::Debug;

use wasnn_tensor::{Tensor, TensorView, TensorViewMut, View};

use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output};

/// Trait for operators which take a single float tensor and apply a function
/// to each element.
pub trait UnaryFloatOp {
    fn name(&self) -> &str;

    /// Apply the operator to a single element.
    fn map_element(&self, val: f32) -> f32;

    /// Apply the operator to all elements in `input`.
    fn map(&self, input: TensorView) -> Tensor {
        input.map(|val| self.map_element(*val))
    }

    /// Apply the operator to all elements in `input`.
    fn apply(&self, input: &mut Tensor) {
        input.apply(|val| self.map_element(*val))
    }
}

impl<Op: UnaryFloatOp + Debug> Operator for Op {
    fn name(&self) -> &str {
        self.name()
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        self.map(input.view()).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: InputList) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::IncorrectInputType)?;
        self.apply(&mut output);
        Ok(output.into())
    }
}

/// Define a unary operator, with no arguments, which supports all numeric
/// tensor types.
///
/// The operator is defined by a name and generic functions which apply this
/// operator to 1) an immutable view and 2) a mutable tensor.
macro_rules! unary_numeric_op {
    ($name:ident, $view_impl:ident, $mut_impl:ident) => {
        #[derive(Debug)]
        pub struct $name {}

        impl Operator for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
                let input = inputs.require(0)?;
                match input {
                    Input::FloatTensor(input) => $view_impl(input.view()).into_op_result(),
                    Input::IntTensor(input) => $view_impl(input.view()).into_op_result(),
                }
            }

            fn can_run_in_place(&self) -> bool {
                true
            }

            fn run_in_place(&self, input: Output, _: InputList) -> Result<Output, OpError> {
                match input {
                    Output::FloatTensor(mut input) => {
                        $mut_impl(&mut input);
                        Ok(input.into())
                    }
                    Output::IntTensor(mut input) => {
                        $mut_impl(&mut input);
                        Ok(input.into())
                    }
                }
            }
        }
    };
}

/// Define a unary operator, with no arguments, which supports all float tensor
/// types.
///
/// The operator is defined by the names for the operator struct and associated
/// functions, and a closure which evaluates the operator for a single function.
macro_rules! unary_float_op {
    ($name:ident, $func_name:ident, $in_place_func_name:ident, $expr:expr) => {
        pub fn $func_name(input: TensorView) -> Tensor {
            $name {}.map(input)
        }

        pub fn $in_place_func_name(input: &mut Tensor) {
            $name {}.apply(input)
        }

        #[derive(Debug)]
        pub struct $name {}

        impl UnaryFloatOp for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn map_element(&self, val: f32) -> f32 {
                $expr(val)
            }
        }
    };
}

pub trait AbsValue {
    fn abs(&self) -> Self;
}

impl AbsValue for f32 {
    fn abs(&self) -> f32 {
        (*self).abs()
    }
}

impl AbsValue for i32 {
    fn abs(&self) -> i32 {
        (*self).abs()
    }
}

pub fn abs<T: AbsValue>(input: TensorView<T>) -> Tensor<T> {
    input.map(|x| x.abs())
}

pub fn abs_in_place<T: AbsValue>(input: &mut Tensor<T>) {
    input.apply(|x| x.abs())
}

unary_numeric_op!(Abs, abs, abs_in_place);
unary_float_op!(Acos, acos, acos_in_place, |val: f32| val.acos());
unary_float_op!(Asin, asin, asin_in_place, |val: f32| val.asin());
unary_float_op!(Atan, atan, atan_in_place, |val: f32| val.atan());

unary_float_op!(Ceil, ceil, ceil_in_place, |val: f32| val.ceil());

/// Numeric value with a finite minimum and maximum and operations to clamp
/// values.
pub trait Clamp: Copy + PartialOrd {
    /// Return the minimum possible finite value for this type.
    fn min_val() -> Self;

    /// Return the maximum possible finite value for this type.
    fn max_val() -> Self;

    /// Return the minimum of `self` and `val`.
    fn min(&self, val: Self) -> Self {
        if *self < val {
            *self
        } else {
            val
        }
    }

    /// Return the maximum of `self` and `val`.
    fn max(&self, val: Self) -> Self {
        if *self > val {
            *self
        } else {
            val
        }
    }

    /// Return self constrained to the range `[min, max]`.
    fn clamp(&self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl Clamp for i32 {
    fn min_val() -> Self {
        i32::MIN
    }

    fn max_val() -> Self {
        i32::MAX
    }
}

impl Clamp for f32 {
    fn min_val() -> Self {
        f32::MIN
    }

    fn max_val() -> Self {
        f32::MAX
    }
}

pub fn clip<T: Copy + Clamp>(input: TensorView<T>, min: Option<T>, max: Option<T>) -> Tensor<T> {
    let min = min.unwrap_or(T::min_val());
    let max = max.unwrap_or(T::max_val());
    input.map(|x| x.clamp(min, max))
}

pub fn clip_in_place<T: Copy + Clamp>(input: &mut Tensor<T>, min: Option<T>, max: Option<T>) {
    let min = min.unwrap_or(T::min_val());
    let max = max.unwrap_or(T::max_val());
    input.apply(|x| x.clamp(min, max))
}

// TODO - Move `Clip` operator into another module since it is no longer a
// unary op (it used to take `min` and `max` as attributes).

#[derive(Debug)]
pub struct Clip {}

impl Operator for Clip {
    fn name(&self) -> &str {
        "Clip"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        match input {
            Input::FloatTensor(input) => {
                let min = inputs.get_as_scalar(1)?;
                let max = inputs.get_as_scalar(2)?;
                clip(input.view(), min, max).into_op_result()
            }
            Input::IntTensor(input) => {
                let min = inputs.get_as_scalar(1)?;
                let max = inputs.get_as_scalar(2)?;
                clip(input.view(), min, max).into_op_result()
            }
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        match input {
            Output::FloatTensor(mut input) => {
                let min = other.get_as_scalar(0)?;
                let max = other.get_as_scalar(1)?;
                clip_in_place(&mut input, min, max);
                Ok(input.into())
            }
            Output::IntTensor(mut input) => {
                let min = other.get_as_scalar(0)?;
                let max = other.get_as_scalar(1)?;
                clip_in_place(&mut input, min, max);
                Ok(input.into())
            }
        }
    }
}

unary_float_op!(Cos, cos, cos_in_place, |val: f32| val.cos());
unary_float_op!(Erf, erf, erf_in_place, libm::erff);
unary_float_op!(Exp, exp, exp_in_place, |val: f32| val.exp());
unary_float_op!(Floor, floor, floor_in_place, |val: f32| val.floor());

pub fn leaky_relu(input: TensorView, alpha: f32) -> Tensor {
    LeakyRelu { alpha }.map(input)
}

pub fn leaky_relu_in_place(input: &mut Tensor, alpha: f32) {
    LeakyRelu { alpha }.apply(input)
}

#[derive(Debug)]
pub struct LeakyRelu {
    pub alpha: f32,
}

impl UnaryFloatOp for LeakyRelu {
    fn name(&self) -> &str {
        "LeakyRelu"
    }

    fn map_element(&self, val: f32) -> f32 {
        if val < 0.0 {
            self.alpha * val
        } else {
            val
        }
    }
}

unary_float_op!(Log, log, log_in_place, |val: f32| val.ln());

pub fn neg<T: Copy + std::ops::Neg<Output = T>>(input: TensorView<T>) -> Tensor<T> {
    input.map(|x| x.neg())
}

pub fn neg_in_place<T: Copy + std::ops::Neg<Output = T>>(input: &mut Tensor<T>) {
    input.apply(|x| x.neg())
}

unary_numeric_op!(Neg, neg, neg_in_place);

pub fn not<T: Default + PartialEq>(input: TensorView<T>) -> Tensor<i32> {
    input.map(|x| if *x == T::default() { 1 } else { 0 })
}

pub fn not_in_place(mut input: TensorViewMut<i32>) {
    input.apply(|&x| if x == 0 { 1 } else { 0 });
}

#[derive(Debug)]
pub struct Not {}

impl Operator for Not {
    fn name(&self) -> &str {
        "Not"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as::<i32>(0)?;
        not(input.view()).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: InputList) -> Result<Output, OpError> {
        let mut output = input.into_int().ok_or(OpError::IncorrectInputType)?;
        not_in_place(output.view_mut());
        Ok(output.into())
    }
}

unary_float_op!(Reciprocal, reciprocal, reciprocal_in_place, |val: f32| 1.
    / val);
unary_float_op!(Relu, relu, relu_in_place, |val: f32| val.max(0.));

/// Round float values to the nearest integer. Values with a fractional part
/// of 0.5 are rounded to the nearest even number, like `round` in Python and
/// unlike `f32::round` in Rust.
#[derive(Debug)]
pub struct Round {}
impl UnaryFloatOp for Round {
    fn name(&self) -> &str {
        "Round"
    }

    fn map_element(&self, val: f32) -> f32 {
        // Replace this with `f32::round_ties_even` when that is stabilized.
        libm::rintf(val)
    }
}

pub fn round(x: TensorView) -> Tensor {
    Round {}.map(x)
}

pub fn round_in_place(x: &mut Tensor) {
    Round {}.apply(x)
}

unary_float_op!(Sigmoid, sigmoid, sigmoid_in_place, |val: f32| 1.
    / (1. + (-val).exp()));
unary_float_op!(Sin, sin, sin_in_place, |val: f32| val.sin());
unary_float_op!(Sqrt, sqrt, sqrt_in_place, |val: f32| val.sqrt());
unary_float_op!(Tan, tan, tan_in_place, |val: f32| val.tan());
unary_float_op!(Tanh, tanh, tanh_in_place, |val: f32| val.tanh());

#[cfg(test)]
mod tests {
    use wasnn_tensor::test_util::{eq_with_nans, expect_equal};
    use wasnn_tensor::{tensor, Tensor, View};

    use crate::ops::{
        abs, acos, acos_in_place, asin, asin_in_place, atan, atan_in_place, ceil, clip,
        clip_in_place, cos, cos_in_place, erf, erf_in_place, exp, exp_in_place, floor, leaky_relu,
        leaky_relu_in_place, log, log_in_place, neg, neg_in_place, not, not_in_place, reciprocal,
        relu, relu_in_place, round, round_in_place, sigmoid, sigmoid_in_place, sin, sin_in_place,
        sqrt, sqrt_in_place, tan, tan_in_place, tanh, tanh_in_place,
    };

    /// Define a test for a simple unary operator which applies the function
    /// `$gen_expected` to each input element.
    macro_rules! test_unary_op {
        ($test_name:ident, $op:ident, $in_place_op:ident, $gen_expected:expr) => {
            #[test]
            fn $test_name() -> Result<(), String> {
                // Test inputs here chosen to be in the domain of inverse trig
                // operators (ie. (-1, 1)).
                let input = tensor!([0., 0.1, -0.1, 0.9, -0.9]);
                let expected = input.map($gen_expected);
                let result = $op(input.view());
                expect_equal(&result, &expected)?;

                let mut input = input.clone();
                $in_place_op(&mut input);
                expect_equal(&input, &expected)?;

                Ok(())
            }
        };
    }

    #[test]
    fn test_abs() {
        // Float tensor
        let x: Tensor<f32> = tensor!([1., -1., 0.]);
        let result = abs(x.view());
        assert_eq!(result, tensor!([1., 1., 0.]));

        // Int tensor
        let x: Tensor<i32> = tensor!([1, -1, 0]);
        let result = abs(x.view());
        assert_eq!(result, tensor!([1, 1, 0]));
    }

    test_unary_op!(test_acos, acos, acos_in_place, |x: &f32| x.acos());
    test_unary_op!(test_asin, asin, asin_in_place, |x: &f32| x.asin());
    test_unary_op!(test_atan, atan, atan_in_place, |x: &f32| x.atan());

    #[test]
    fn test_ceil() {
        let input = tensor!([
            1.,
            1.2,
            1.5,
            1.8,
            0.,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY
        ]);
        let expected = tensor!([
            1.,
            2.,
            2.,
            2.,
            0.,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY
        ]);
        let result = ceil(input.view());
        assert!(eq_with_nans(result.view(), expected.view()));
    }

    #[test]
    fn test_clip() -> Result<(), String> {
        struct Case {
            input: Tensor,
            min: Option<f32>,
            max: Option<f32>,
            expected: Tensor,
        }

        let cases = [
            Case {
                input: tensor!((2, 2); [-5., -2., 3., 20.]),
                min: Some(1.),
                max: Some(5.),
                expected: tensor!((2, 2); [1., 1., 3., 5.]),
            },
            Case {
                input: tensor!((2, 2); [-5., -2., 3., 20.]),
                min: Some(1.),
                max: None,
                expected: tensor!((2, 2); [1., 1., 3., 20.]),
            },
            Case {
                input: tensor!((2, 2); [-5., -2., 3., 20.]),
                min: None,
                max: Some(5.),
                expected: tensor!((2, 2); [-5., -2., 3., 5.]),
            },
        ];

        for case in cases {
            let result = clip(case.input.view(), case.min, case.max);
            expect_equal(&result, &case.expected)?;

            let mut input = case.input.clone();
            clip_in_place(&mut input, case.min, case.max);
            expect_equal(&input, &case.expected)?;
        }

        Ok(())
    }

    // TODO: Eliminate the duplication for tests that apply the operator
    // in-place vs returning a new tensor.

    test_unary_op!(test_cos, cos, cos_in_place, |x: &f32| x.cos());

    #[test]
    fn test_erf() -> Result<(), String> {
        let input = tensor!([-2.0, -0.5, 0.5, 2.0]);
        let expected = tensor!([
            -0.9953222650189527,
            -0.5204998778130465,
            0.5204998778130465,
            0.9953222650189527,
        ]);
        let result = erf(input.view());
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_erf_in_place() -> Result<(), String> {
        let mut input = tensor!([-2.0, -0.5, 0.5, 2.0]);
        let expected = tensor!([
            -0.9953222650189527,
            -0.5204998778130465,
            0.5204998778130465,
            0.9953222650189527,
        ]);
        erf_in_place(&mut input);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_exp() -> Result<(), String> {
        let input = tensor!([-2.0, -0.5, 0.5, 2.0]);
        let expected = tensor!([
            0.1353352832366127,
            0.6065306597126334,
            1.6487212707001282,
            7.38905609893065
        ]);
        let result = exp(input.view());
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_exp_in_place() -> Result<(), String> {
        let mut input = tensor!([-2.0, -0.5, 0.5, 2.0]);
        let expected = tensor!([
            0.1353352832366127,
            0.6065306597126334,
            1.6487212707001282,
            7.38905609893065
        ]);
        exp_in_place(&mut input);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_floor() {
        let input = tensor!([
            1.,
            1.2,
            1.5,
            1.8,
            0.,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY
        ]);
        let expected = tensor!([
            1.,
            1.,
            1.,
            1.,
            0.,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY
        ]);
        let result = floor(input.view());
        assert!(eq_with_nans(result.view(), expected.view()));
    }

    #[test]
    fn test_leaky_relu() -> Result<(), String> {
        let input = Tensor::from_data(&[2, 2], vec![-5., -2., 3., 20.]);
        let alpha = 0.1;
        let expected = Tensor::from_data(&[2, 2], vec![-5. * alpha, -2. * alpha, 3., 20.]);
        let result = leaky_relu(input.view(), alpha);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_leaky_relu_in_place() -> Result<(), String> {
        let mut input = Tensor::from_data(&[2, 2], vec![-5., -2., 3., 20.]);
        let alpha = 0.1;
        let expected = Tensor::from_data(&[2, 2], vec![-5. * alpha, -2. * alpha, 3., 20.]);
        leaky_relu_in_place(&mut input, alpha);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_log() -> Result<(), String> {
        let input = tensor!([0.1, 0.5, 1., 10.]);
        let expected = tensor!([
            -2.3025850929940455,
            -0.6931471805599453,
            0.,
            2.302585092994046
        ]);
        let result = log(input.view());
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_log_in_place() -> Result<(), String> {
        let mut input = tensor!([0.1, 0.5, 1., 10.]);
        let expected = tensor!([
            -2.3025850929940455,
            -0.6931471805599453,
            0.,
            2.302585092994046
        ]);
        log_in_place(&mut input);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_neg() {
        let input = tensor!([0, 1, -1, 2]);
        let expected = tensor!([0, -1, 1, -2]);
        let result = neg(input.view());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_in_place() {
        let mut input = tensor!([0, 1, -1, 2]);
        let expected = tensor!([0, -1, 1, -2]);
        neg_in_place(&mut input);
        assert_eq!(input, expected);
    }

    #[test]
    fn test_not() {
        let input = tensor!([0, 1, 1, 0]);
        let expected = tensor!([1, 0, 0, 1]);
        let result = not(input.view());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_not_in_place() {
        let mut input = tensor!([0, 1, 1, 0]);
        let expected = tensor!([1, 0, 0, 1]);
        not_in_place(input.view_mut());
        assert_eq!(input, expected);
    }

    #[test]
    fn test_reciprocal() {
        let input = tensor!([1., 2., 0.5, 0.]);
        let expected = input.map(|x| 1. / x);
        let result = reciprocal(input.view());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_relu() -> Result<(), String> {
        let input = Tensor::from_data(&[2, 2, 1], vec![-0.5, 0.5, 3.0, -5.5]);
        let expected = Tensor::from_data(&[2, 2, 1], vec![0.0, 0.5, 3.0, 0.0]);

        let result = relu(input.view());
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        relu_in_place(&mut result);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_round() -> Result<(), String> {
        // Example from ONNX spec.
        let input = tensor!([0.9, 2.5, 2.3, 1.5, -4.5]);
        let expected = tensor!([1., 2., 2., 2., -4.]);
        let result = round(input.view());
        expect_equal(&result, &expected)?;

        let mut input = input.clone();
        round_in_place(&mut input);
        expect_equal(&input, &expected)?;

        // Per spec, integral, zero, NaN and infinities are unchanged.
        let input = tensor!([1., 0., -0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        let result = round(input.view());
        assert!(eq_with_nans(input.view(), result.view()));

        Ok(())
    }

    #[test]
    fn test_sigmoid() -> Result<(), String> {
        let input = Tensor::from_data(
            &[9],
            vec![-500.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 500.0],
        );
        let expected = Tensor::from_data(
            &[9],
            vec![
                0.0000, 0.0474, 0.2689, 0.3775, 0.5000, 0.6225, 0.7311, 0.9526, 1.0000,
            ],
        );

        let result = sigmoid(input.view());
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        sigmoid_in_place(&mut result);
        expect_equal(&result, &expected)
    }

    test_unary_op!(test_sin, sin, sin_in_place, |x: &f32| x.sin());

    #[test]
    fn test_sqrt() -> Result<(), String> {
        let input = tensor!([4., 9., 16.]);
        let expected = tensor!([2., 3., 4.]);
        let result = sqrt(input.view());
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sqrt_in_place() -> Result<(), String> {
        let mut input = tensor!([4., 9., 16.]);
        let expected = tensor!([2., 3., 4.]);
        sqrt_in_place(&mut input);
        expect_equal(&input, &expected)
    }

    test_unary_op!(test_tan, tan, tan_in_place, |x: &f32| x.tan());
    test_unary_op!(test_tanh, tanh, tanh_in_place, |x: &f32| x.tanh());
}
