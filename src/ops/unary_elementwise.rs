extern crate libm;

use std::fmt::Debug;

use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output};
use crate::tensor::Tensor;

/// Trait for operators which take a single float tensor and apply a function
/// to each element.
pub trait UnaryFloatOp {
    fn name(&self) -> &str;

    /// Apply the operator to a single element.
    fn map_element(&self, val: f32) -> f32;

    /// Apply the operator to all elements in `input`.
    fn map(&self, input: &Tensor) -> Tensor {
        input.map(|val| self.map_element(val))
    }

    /// Apply the operator to all elements in `input`.
    fn apply(&self, input: &mut Tensor) {
        input.apply(|val| self.map_element(val))
    }
}

impl<Op: UnaryFloatOp + Debug> Operator for Op {
    fn name(&self) -> &str {
        self.name()
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        self.map(input).into_op_result()
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

pub fn clip(input: &Tensor, min: Option<f32>, max: Option<f32>) -> Tensor {
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
        clip(input, min, max).into_op_result()
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

pub fn cos(input: &Tensor) -> Tensor {
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

pub fn erf(input: &Tensor) -> Tensor {
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

pub fn leaky_relu(input: &Tensor, alpha: f32) -> Tensor {
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

pub fn relu_in_place(x: &mut Tensor) {
    Relu {}.apply(x)
}

pub fn relu(x: &Tensor) -> Tensor {
    Relu {}.map(x)
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

pub fn sigmoid(x: &Tensor) -> Tensor {
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

pub fn sin(input: &Tensor) -> Tensor {
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

pub fn sqrt(input: &Tensor) -> Tensor {
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

pub fn tanh(input: &Tensor) -> Tensor {
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
    use crate::ops::{
        clip, clip_in_place, cos, cos_in_place, erf, erf_in_place, leaky_relu, leaky_relu_in_place,
        relu, relu_in_place, sigmoid, sigmoid_in_place, sin, sin_in_place, sqrt, sqrt_in_place,
        tanh, tanh_in_place,
    };
    use crate::tensor;
    use crate::tensor::{from_data, from_vec, Tensor};
    use crate::test_util::expect_equal;

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
            let result = clip(&case.input, case.min, case.max);
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
        let input = from_vec(vec![0.1, 3.14, -5.]);
        let expected = input.map(|x: f32| x.cos());
        let result = cos(&input);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_cos_in_place() -> Result<(), String> {
        let mut input = from_vec(vec![0.1, 3.14, -5.]);
        let expected = input.map(|x: f32| x.cos());
        cos_in_place(&mut input);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_erf() -> Result<(), String> {
        let input = from_vec(vec![-2.0, -0.5, 0.5, 2.0]);
        let expected = from_vec(vec![
            -0.9953222650189527,
            -0.5204998778130465,
            0.5204998778130465,
            0.9953222650189527,
        ]);
        let result = erf(&input);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_erf_in_place() -> Result<(), String> {
        let mut input = from_vec(vec![-2.0, -0.5, 0.5, 2.0]);
        let expected = from_vec(vec![
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
        let input = from_data(&[2, 2], vec![-5., -2., 3., 20.]);
        let alpha = 0.1;
        let expected = from_data(&[2, 2], vec![-5. * alpha, -2. * alpha, 3., 20.]);
        let result = leaky_relu(&input, alpha);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_leaky_relu_in_place() -> Result<(), String> {
        let mut input = from_data(&[2, 2], vec![-5., -2., 3., 20.]);
        let alpha = 0.1;
        let expected = from_data(&[2, 2], vec![-5. * alpha, -2. * alpha, 3., 20.]);
        leaky_relu_in_place(&mut input, alpha);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_relu() -> Result<(), String> {
        let input = from_data(&[2, 2, 1], vec![-0.5, 0.5, 3.0, -5.5]);
        let expected = from_data(&[2, 2, 1], vec![0.0, 0.5, 3.0, 0.0]);

        let result = relu(&input);
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        relu_in_place(&mut result);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sigmoid() -> Result<(), String> {
        let input = from_data(
            &[9],
            vec![-500.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 500.0],
        );
        let expected = from_data(
            &[9],
            vec![
                0.0000, 0.0474, 0.2689, 0.3775, 0.5000, 0.6225, 0.7311, 0.9526, 1.0000,
            ],
        );

        let result = sigmoid(&input);
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        sigmoid_in_place(&mut result);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sin() -> Result<(), String> {
        let input = from_vec(vec![0.1, 3.14, -5.]);
        let expected = input.map(|x: f32| x.sin());
        let result = sin(&input);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sin_in_place() -> Result<(), String> {
        let mut input = from_vec(vec![0.1, 3.14, -5.]);
        let expected = input.map(|x: f32| x.sin());
        sin_in_place(&mut input);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_sqrt() -> Result<(), String> {
        let input = from_vec(vec![4., 9., 16.]);
        let expected = from_vec(vec![2., 3., 4.]);
        let result = sqrt(&input);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sqrt_in_place() -> Result<(), String> {
        let mut input = from_vec(vec![4., 9., 16.]);
        let expected = from_vec(vec![2., 3., 4.]);
        sqrt_in_place(&mut input);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_tanh() -> Result<(), String> {
        let input = from_vec(vec![0.1, 3.14, -5.]);
        let expected = input.map(|x: f32| x.tanh());
        let result = tanh(&input);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_tanh_in_place() -> Result<(), String> {
        let mut input = from_vec(vec![0.1, 3.14, -5.]);
        let expected = input.map(|x: f32| x.tanh());
        tanh_in_place(&mut input);
        expect_equal(&input, &expected)
    }
}
