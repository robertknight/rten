extern crate libm;

use std::fmt::Debug;

use wasnn_tensor::{Tensor, TensorCommon, TensorView, TensorViewMut};

use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output};

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

pub fn clip(input: TensorView, min: Option<f32>, max: Option<f32>) -> Tensor {
    let min = min.unwrap_or(f32::MIN);
    let max = max.unwrap_or(f32::MAX);
    input.map(|x| x.clamp(min, max))
}

pub fn clip_in_place(input: &mut Tensor, min: Option<f32>, max: Option<f32>) {
    let min = min.unwrap_or(f32::MIN);
    let max = max.unwrap_or(f32::MAX);
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
        let input = inputs.require_as(0)?;
        let min = inputs.get_as_scalar(1)?;
        let max = inputs.get_as_scalar(2)?;
        clip(input.view(), min, max).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        let mut input = input.into_float().ok_or(OpError::IncorrectInputType)?;
        let min = other.get_as_scalar(0)?;
        let max = other.get_as_scalar(1)?;
        clip_in_place(&mut input, min, max);
        Ok(input.into())
    }
}

pub fn cos(input: TensorView) -> Tensor {
    Cos {}.map(input)
}

pub fn cos_in_place(input: &mut Tensor) {
    Cos {}.apply(input)
}

#[derive(Debug)]
pub struct Cos {}

impl UnaryFloatOp for Cos {
    fn name(&self) -> &str {
        "Cos"
    }

    fn map_element(&self, val: f32) -> f32 {
        val.cos()
    }
}

pub fn erf(input: TensorView) -> Tensor {
    Erf {}.map(input)
}

pub fn erf_in_place(input: &mut Tensor) {
    Erf {}.apply(input)
}

#[derive(Debug)]
pub struct Erf {}

impl UnaryFloatOp for Erf {
    fn name(&self) -> &str {
        "Erf"
    }

    fn map_element(&self, val: f32) -> f32 {
        libm::erff(val)
    }
}

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

pub fn log(input: TensorView) -> Tensor {
    Log {}.map(input)
}

pub fn log_in_place(input: &mut Tensor) {
    Log {}.apply(input)
}

#[derive(Debug)]
pub struct Log {}

impl UnaryFloatOp for Log {
    fn name(&self) -> &str {
        "Log"
    }

    fn map_element(&self, val: f32) -> f32 {
        val.ln()
    }
}

pub fn relu_in_place(x: &mut Tensor) {
    Relu {}.apply(x)
}

pub fn relu(x: TensorView) -> Tensor {
    Relu {}.map(x)
}

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

#[derive(Debug)]
pub struct Relu {}
impl UnaryFloatOp for Relu {
    fn name(&self) -> &str {
        "Relu"
    }

    fn map_element(&self, val: f32) -> f32 {
        val.max(0.)
    }
}

pub fn sigmoid(x: TensorView) -> Tensor {
    Sigmoid {}.map(x)
}

pub fn sigmoid_in_place(x: &mut Tensor) {
    Sigmoid {}.apply(x)
}

#[derive(Debug)]
pub struct Sigmoid {}
impl UnaryFloatOp for Sigmoid {
    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn map_element(&self, val: f32) -> f32 {
        1. / (1. + (-val).exp())
    }
}

pub fn sin(input: TensorView) -> Tensor {
    Sin {}.map(input)
}

pub fn sin_in_place(input: &mut Tensor) {
    Sin {}.apply(input)
}

#[derive(Debug)]
pub struct Sin {}

impl UnaryFloatOp for Sin {
    fn name(&self) -> &str {
        "Sin"
    }

    fn map_element(&self, val: f32) -> f32 {
        val.sin()
    }
}

pub fn sqrt(input: TensorView) -> Tensor {
    Sqrt {}.map(input)
}

pub fn sqrt_in_place(input: &mut Tensor) {
    Sqrt {}.apply(input)
}

#[derive(Debug)]
pub struct Sqrt {}

impl UnaryFloatOp for Sqrt {
    fn name(&self) -> &str {
        "Sqrt"
    }

    fn map_element(&self, val: f32) -> f32 {
        val.sqrt()
    }
}

pub fn tanh(input: TensorView) -> Tensor {
    Tanh {}.map(input)
}

pub fn tanh_in_place(input: &mut Tensor) {
    Tanh {}.apply(input)
}

#[derive(Debug)]
pub struct Tanh {}

impl UnaryFloatOp for Tanh {
    fn name(&self) -> &str {
        "Tanh"
    }

    fn map_element(&self, val: f32) -> f32 {
        val.tanh()
    }
}

#[cfg(test)]
mod tests {
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{tensor, Tensor, TensorCommon};

    use crate::ops::{
        clip, clip_in_place, cos, cos_in_place, erf, erf_in_place, leaky_relu, leaky_relu_in_place,
        log, log_in_place, not, not_in_place, relu, relu_in_place, sigmoid, sigmoid_in_place, sin,
        sin_in_place, sqrt, sqrt_in_place, tanh, tanh_in_place,
    };

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

    #[test]
    fn test_cos() -> Result<(), String> {
        let input = tensor!([0.1, 3.14, -5.]);
        let expected = input.map(|x: &f32| x.cos());
        let result = cos(input.view());
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_cos_in_place() -> Result<(), String> {
        let mut input = tensor!([0.1, 3.14, -5.]);
        let expected = input.map(|x: &f32| x.cos());
        cos_in_place(&mut input);
        expect_equal(&input, &expected)
    }

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

    #[test]
    fn test_sin() -> Result<(), String> {
        let input = tensor!([0.1, 3.14, -5.]);
        let expected = input.map(|x: &f32| x.sin());
        let result = sin(input.view());
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sin_in_place() -> Result<(), String> {
        let mut input = tensor!([0.1, 3.14, -5.]);
        let expected = input.map(|x: &f32| x.sin());
        sin_in_place(&mut input);
        expect_equal(&input, &expected)
    }

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

    #[test]
    fn test_tanh() -> Result<(), String> {
        let input = tensor!([0.1, 3.14, -5.]);
        let expected = input.map(|x: &f32| x.tanh());
        let result = tanh(input.view());
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_tanh_in_place() -> Result<(), String> {
        let mut input = tensor!([0.1, 3.14, -5.]);
        let expected = input.map(|x: &f32| x.tanh());
        tanh_in_place(&mut input);
        expect_equal(&input, &expected)
    }
}
