use crate::ops::{Input, Operator, Output};
use crate::tensor::Tensor;

pub fn clip(input: &Tensor, min: f32, max: f32) -> Tensor {
    input.map(|x| x.max(min).min(max))
}

pub fn clip_in_place(input: &mut Tensor, min: f32, max: f32) {
    for val in input.data_mut().iter_mut() {
        *val = val.max(min).min(max)
    }
}

#[derive(Debug)]
pub struct Clip {
    pub min: f32,
    pub max: f32,
}

impl Operator for Clip {
    fn name(&self) -> &str {
        "Clip"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        clip(input, self.min, self.max).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        clip_in_place(&mut output, self.min, self.max);
        output.into()
    }
}

pub fn leaky_relu(input: &Tensor, alpha: f32) -> Tensor {
    input.map(|el| if el < 0.0 { alpha * el } else { el })
}

pub fn leaky_relu_in_place(input: &mut Tensor, alpha: f32) {
    for val in input.data_mut().iter_mut() {
        *val = if (*val) < 0.0 { alpha * (*val) } else { *val }
    }
}

#[derive(Debug)]
pub struct LeakyRelu {
    pub alpha: f32,
}

impl Operator for LeakyRelu {
    fn name(&self) -> &str {
        "LeakyRelu"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        leaky_relu(input, self.alpha).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        leaky_relu_in_place(&mut output, self.alpha);
        output.into()
    }
}

pub fn relu_in_place(x: &mut Tensor) {
    for val in x.data_mut().iter_mut() {
        *val = val.max(0f32);
    }
}

pub fn relu(x: &Tensor) -> Tensor {
    x.map(|e| e.max(0f32))
}

#[derive(Debug)]
pub struct Relu {}
impl Operator for Relu {
    fn name(&self) -> &str {
        "Relu"
    }

    /// Run `relu` operator with `[input]` inputs.
    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        relu(input).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        relu_in_place(&mut output);
        output.into()
    }
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    x.map(|e| 1. / (1. + (-e).exp()))
}

pub fn sigmoid_in_place(x: &mut Tensor) {
    for val in x.data_mut().iter_mut() {
        *val = 1. / (1. + (-*val).exp());
    }
}

#[derive(Debug)]
pub struct Sigmoid {}
impl Operator for Sigmoid {
    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        sigmoid(input).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        sigmoid_in_place(&mut output);
        output.into()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{
        clip, clip_in_place, leaky_relu, leaky_relu_in_place, relu, relu_in_place, sigmoid,
        sigmoid_in_place,
    };
    use crate::tensor::from_data;
    use crate::test_util::expect_equal;

    #[test]
    fn test_clip() -> Result<(), String> {
        let input = from_data(vec![2, 2], vec![-5., -2., 3., 20.]);
        let expected = from_data(vec![2, 2], vec![1., 1., 3., 5.]);
        let result = clip(&input, 1.0, 5.0);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_clip_in_place() -> Result<(), String> {
        let mut input = from_data(vec![2, 2], vec![-5., -2., 3., 20.]);
        let expected = from_data(vec![2, 2], vec![1., 1., 3., 5.]);
        clip_in_place(&mut input, 1.0, 5.0);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_leaky_relu() -> Result<(), String> {
        let input = from_data(vec![2, 2], vec![-5., -2., 3., 20.]);
        let alpha = 0.1;
        let expected = from_data(vec![2, 2], vec![-5. * alpha, -2. * alpha, 3., 20.]);
        let result = leaky_relu(&input, alpha);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_leaky_relu_in_place() -> Result<(), String> {
        let mut input = from_data(vec![2, 2], vec![-5., -2., 3., 20.]);
        let alpha = 0.1;
        let expected = from_data(vec![2, 2], vec![-5. * alpha, -2. * alpha, 3., 20.]);
        leaky_relu_in_place(&mut input, alpha);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_relu() -> Result<(), String> {
        let input = from_data(vec![2, 2, 1], vec![-0.5, 0.5, 3.0, -5.5]);
        let expected = from_data(vec![2, 2, 1], vec![0.0, 0.5, 3.0, 0.0]);

        let result = relu(&input);
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        relu_in_place(&mut result);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sigmoid() -> Result<(), String> {
        let input = from_data(
            vec![9],
            vec![-500.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 500.0],
        );
        let expected = from_data(
            vec![9],
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
}
