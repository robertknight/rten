use std::ops::Range;

use crate::ops::{get_input_as_float, Input, OpError, Operator, Output};
use crate::tensor::{IndexIterator, Tensor};

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

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        Ok(clip(input, self.min, self.max).into())
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: &[Input]) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        clip_in_place(&mut output, self.min, self.max);
        Ok(output.into())
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

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        Ok(leaky_relu(input, self.alpha).into())
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        leaky_relu_in_place(&mut output, self.alpha);
        Ok(output.into())
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
    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        Ok(relu(input).into())
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        relu_in_place(&mut output);
        Ok(output.into())
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

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        Ok(sigmoid(input).into())
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        sigmoid_in_place(&mut output);
        Ok(output.into())
    }
}

pub fn softmax(input: &Tensor, axis: usize) -> Tensor {
    let mut output = input.clone();
    softmax_in_place(&mut output, axis);
    output
}

pub fn softmax_in_place(output: &mut Tensor, axis: usize) {
    let outer_range: Vec<Range<usize>> = output
        .shape()
        .iter()
        .enumerate()
        .map(|(dim, &size)| if dim >= axis { 0..1 } else { 0..size })
        .collect();

    let mut outer_iter = IndexIterator::from_ranges(&outer_range);
    while let Some(outer_index) = outer_iter.next() {
        let inner_range: Vec<_> = output
            .shape()
            .iter()
            .enumerate()
            .map(|(dim, &size)| {
                if dim < axis {
                    (outer_index[dim], outer_index[dim] + 1, 1)
                } else {
                    (0, size, 1)
                }
            })
            .collect();

        // Numerically stable softmax. See
        // https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html.
        let mut max_val: f32 = 0.0;
        for el in output.slice_elements(&inner_range) {
            max_val = max_val.max(el);
        }

        let mut exp_sum = 0.0;
        for offset in output.slice_offsets(&inner_range) {
            let el = &mut output.data_mut()[offset];
            *el = (*el - max_val).exp();
            exp_sum += *el;
        }

        for offset in output.slice_offsets(&inner_range) {
            let el = &mut output.data_mut()[offset];
            *el /= exp_sum
        }
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

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        Ok(softmax(input, self.axis).into())
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

#[cfg(test)]
mod tests {
    use crate::ops::{
        clip, clip_in_place, leaky_relu, leaky_relu_in_place, relu, relu_in_place, sigmoid,
        sigmoid_in_place, softmax,
    };
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, from_vec, random_tensor};
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
        let input = random_tensor(&[1, 1, 3, 3], &mut rng);
        let result = softmax(&input, 1);
        assert_eq!(result.shape(), input.shape());
        assert!((result.elements().sum::<f32>() - 1.0).abs() < 0.001);
    }
}
