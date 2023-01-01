extern crate libm;

use std::fmt::Debug;

use crate::ops::{get_input, Input, IntoOpResult, OpError, Operator, Output};
use crate::tensor::Tensor;

/// Trait for operators which take a single float tensor and apply a function
/// to each element.
trait UnaryFloatOp {
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

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = get_input(inputs, 0)?;
        self.map(input).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: &[Input]) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        self.apply(&mut output);
        Ok(output.into())
    }
}

pub fn clip(input: &Tensor, min: f32, max: f32) -> Tensor {
    Clip { min, max }.map(input)
}

pub fn clip_in_place(input: &mut Tensor, min: f32, max: f32) {
    Clip { min, max }.apply(input)
}

#[derive(Debug)]
pub struct Clip {
    pub min: f32,
    pub max: f32,
}

impl UnaryFloatOp for Clip {
    fn name(&self) -> &str {
        "Clip"
    }

    fn map_element(&self, val: f32) -> f32 {
        val.clamp(self.min, self.max)
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

pub fn softmax(input: &Tensor, axis: usize) -> Tensor {
    let mut output = input.clone();
    softmax_in_place(&mut output, axis);
    output
}

pub fn softmax_in_place(output: &mut Tensor, axis: usize) {
    output.make_contiguous();

    let outer_stride = if axis == 0 {
        output.len()
    } else {
        output.stride(axis - 1)
    };

    let mut offset = 0;
    while offset < output.len() {
        let els = &mut output.data_mut()[offset..offset + outer_stride];

        // Numerically stable softmax. See
        // https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html.
        let max_val = els
            .iter()
            .copied()
            .fold(f32::MIN, |max_val, x| max_val.max(x));
        let mut exp_sum = 0.0;
        for el in els.iter_mut() {
            *el = (*el - max_val).exp();
            exp_sum += *el;
        }

        for el in els.iter_mut() {
            *el /= exp_sum
        }

        offset += outer_stride;
    }
}

#[derive(Debug)]
pub struct Softmax {
    pub axis: usize,
}

impl Operator for Softmax {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = get_input(inputs, 0)?;
        softmax(input, self.axis).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        softmax_in_place(&mut output, self.axis);
        Ok(output.into())
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

#[cfg(test)]
mod tests {
    use crate::ops::{
        clip, clip_in_place, cos, cos_in_place, erf, erf_in_place, leaky_relu, leaky_relu_in_place,
        relu, relu_in_place, sigmoid, sigmoid_in_place, sin, sin_in_place, softmax, sqrt,
        sqrt_in_place,
    };
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, from_vec, rand};
    use crate::test_util::expect_equal;

    // TODO: Eliminate the duplication for tests that apply the operator
    // in-place vs returning a new tensor.

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
    fn test_softmax() -> Result<(), String> {
        // Softmax on a 1D input
        let mut input = from_vec(vec![0.1634, 0.8647, 0.6401, 0.8265, 0.0560]);
        let mut expected = from_vec(vec![0.1339, 0.2701, 0.2157, 0.2599, 0.1203]);
        let result = softmax(&input, 0);
        expect_equal(&result, &expected)?;

        // Softmax on final dimension of 2D input
        input.reshape(&[1, 5]);
        expected.reshape(&[1, 5]);
        let result = softmax(&input, 1);
        expect_equal(&result, &expected)?;

        // Softmax on first dimension of 2D input
        input.reshape(&[5, 1]);
        expected.reshape(&[5, 1]);
        let result = softmax(&input, 0);
        expect_equal(&result, &expected)?;

        // Softmax on second dimension of 2D input with multiple entries in
        // first dim
        let matrix_input = from_data(
            vec![2, 5],
            vec![
                0.1634, 0.8647, 0.6401, 0.8265, 0.0560, // First row
                0.1634, 0.8647, 0.6401, 0.8265, 0.0560, // Second row
            ],
        );
        let matrix_expected = from_data(
            vec![2, 5],
            vec![
                0.1339, 0.2701, 0.2157, 0.2599, 0.1203, // First row
                0.1339, 0.2701, 0.2157, 0.2599, 0.1203, // Second row
            ],
        );
        let result = softmax(&matrix_input, 1);
        expect_equal(&result, &matrix_expected)
    }

    // Test softmax with non-contiguous input.
    #[test]
    fn test_softmax_transposed() -> Result<(), String> {
        let mut input = from_data(
            vec![4, 4],
            vec![
                0.6427, 0.7435, 0.9762, 0.0611, 0.1249, 0.9742, 0.5826, 0.4704, 0.1420, 0.8376,
                0.6692, 0.7090, 0.2448, 0.9083, 0.2881, 0.4971,
            ],
        );
        let expected = from_data(
            vec![4, 4],
            vec![
                0.3480, 0.2073, 0.2109, 0.2337, 0.2204, 0.2776, 0.2421, 0.2599, 0.3433, 0.2316,
                0.2525, 0.1725, 0.1677, 0.2525, 0.3205, 0.2593,
            ],
        );

        input.permute(&[1, 0]);
        let result = softmax(&input, 1);

        expect_equal(&result, &expected)
    }

    // Test softmax with some additional input sizes and axis dimensions.
    // These tests don't check the individual output values in detail, but they
    // do check the shape and that the values sum to 1.
    #[test]
    fn test_softmax_sizes() {
        let mut rng = XorShiftRNG::new(1234);
        let input = rand(&[1, 1, 3, 3], &mut rng);
        let result = softmax(&input, 1);
        assert_eq!(result.shape(), input.shape());
        assert!((result.elements().sum::<f32>() - 1.0).abs() < 0.001);
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
}
