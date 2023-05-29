use crate::check_dims;
use crate::ops::{resolve_axis, InputList, IntoOpResult, OpError, Operator, Output};
use crate::tensor::{Tensor, TensorLayout, TensorView};

/// Perform in-place batch normalization on the NCHW tensor `out`.
///
/// See https://github.com/onnx/onnx/blob/main/docs/Operators.md#batchnormalization
pub fn batch_norm_in_place(
    input: &mut Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<(), OpError> {
    let [batch, chans, in_h, in_w] = check_dims!(input, 4, "NCHW");
    for n in 0..batch {
        for c in 0..chans {
            let chan_mean = mean[[c]];
            let chan_var = var[[c]];
            let chan_scale = scale[[c]];
            let chan_bias = bias[[c]];

            let mut out_view = input.nd_slice_mut([n, c]);
            let mut out_view = out_view.unchecked_mut();

            // The batch norm formula, from the ONNX spec, is:
            //
            // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + bias
            //
            // It has been rewritten here to simplify the inner loop below.
            let scaled_std_dev_reciprocal = chan_scale / (chan_var + epsilon).sqrt();

            for y in 0..in_h {
                for x in 0..in_w {
                    let el = &mut out_view[[y, x]];
                    *el = (*el - chan_mean) * scaled_std_dev_reciprocal + chan_bias;
                }
            }
        }
    }

    Ok(())
}

/// Perform batch normalization on the NCHW tensor `input`.
///
/// See https://github.com/onnx/onnx/blob/main/docs/Operators.md#batchnormalization
pub fn batch_norm(
    input: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    let mut output = input.clone();
    batch_norm_in_place(&mut output, scale, bias, mean, var, epsilon)?;
    Ok(output)
}

#[derive(Debug)]
pub struct BatchNormalization {
    pub epsilon: f32,
}

impl Operator for BatchNormalization {
    fn name(&self) -> &str {
        "BatchNormalization"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        let scale = inputs.require_as(1)?;
        let bias = inputs.require_as(2)?;
        let mean = inputs.require_as(3)?;
        let var = inputs.require_as(4)?;

        batch_norm(input, scale, bias, mean, var, self.epsilon).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::IncorrectInputType)?;
        let scale = other.require_as(0)?;
        let bias = other.require_as(1)?;
        let mean = other.require_as(2)?;
        let var = other.require_as(3)?;

        batch_norm_in_place(&mut output, scale, bias, mean, var, self.epsilon)?;

        Ok(output.into())
    }
}

pub fn log_softmax(input: TensorView, axis: isize) -> Result<Tensor, OpError> {
    let mut output = input.to_tensor();
    log_softmax_in_place(&mut output, axis)?;
    Ok(output)
}

pub fn log_softmax_in_place(output: &mut Tensor, axis: isize) -> Result<(), OpError> {
    let resolved_axis = resolve_axis(output.ndim(), axis)?;

    output.make_contiguous();

    let outer_stride = if resolved_axis == 0 {
        output.len()
    } else {
        output.stride(resolved_axis - 1)
    };

    // This operator computes:
    //
    //   log(exp(xi) / sum(exp(x)))
    //
    // Improve numerical stability by first subtracting max value, as we do
    // for the softmax op:
    //
    //   log(exp(xi - xmax) / sum(exp(x - xmax)))
    //
    // Then using log identities to simplify:
    //
    //   = log(exp(xi - xmax)) - log(sum(exp(x - xmax)))
    //   = xi - xmax - log(sum(exp(x - xmax)))

    for els in output.data_mut().chunks_mut(outer_stride) {
        let max_val = els
            .iter()
            .copied()
            .fold(f32::MIN, |max_val, x| max_val.max(x));
        let log_exp_sum = els
            .iter()
            .fold(0., |exp_sum, x| exp_sum + (x - max_val).exp())
            .ln();
        for el in els.iter_mut() {
            *el = (*el - max_val) - log_exp_sum
        }
    }

    Ok(())
}

#[derive(Debug)]
pub struct LogSoftmax {
    pub axis: isize,
}

impl Operator for LogSoftmax {
    fn name(&self) -> &str {
        "LogSoftmax"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        log_softmax(input.view(), self.axis).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: InputList) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::IncorrectInputType)?;
        log_softmax_in_place(&mut output, self.axis)?;
        Ok(output.into())
    }
}

pub fn softmax(input: TensorView, axis: isize) -> Result<Tensor, OpError> {
    let mut output = input.to_tensor();
    softmax_in_place(&mut output, axis)?;
    Ok(output)
}

pub fn softmax_in_place(output: &mut Tensor, axis: isize) -> Result<(), OpError> {
    let resolved_axis = resolve_axis(output.ndim(), axis)?;

    output.make_contiguous();

    let outer_stride = if resolved_axis == 0 {
        output.len()
    } else {
        output.stride(resolved_axis - 1)
    };

    for els in output.data_mut().chunks_mut(outer_stride) {
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
    }

    Ok(())
}

#[derive(Debug)]
pub struct Softmax {
    pub axis: isize,
}

impl Operator for Softmax {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        softmax(input.view(), self.axis).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: InputList) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::IncorrectInputType)?;
        softmax_in_place(&mut output, self.axis)?;
        Ok(output.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{batch_norm, batch_norm_in_place, log_softmax, softmax};
    use crate::rng::XorShiftRng;
    use crate::tensor;
    use crate::tensor::{from_data, rand, TensorLayout};
    use crate::test_util::expect_equal;

    #[test]
    fn test_batch_norm() -> Result<(), String> {
        let input = from_data(&[1, 2, 1, 1], vec![1.0, 2.0]);
        let scale = from_data(&[2], vec![3.0, 3.0]);
        let bias = from_data(&[2], vec![0.1, 0.2]);
        let mean = from_data(&[2], vec![0.5, -0.5]);
        let var = from_data(&[2], vec![1.0, 2.0]);

        let epsilon = 1e-5 as f32;

        let y1 = (input[[0, 0, 0, 0]] - mean[[0]]) / (var[[0]] + epsilon).sqrt() * scale[[0]]
            + bias[[0]];
        let y2 = (input[[0, 1, 0, 0]] - mean[[1]]) / (var[[1]] + epsilon).sqrt() * scale[[1]]
            + bias[[1]];
        let expected = from_data(&[1, 2, 1, 1], vec![y1, y2]);
        let result = batch_norm(&input, &scale, &bias, &mean, &var, epsilon).unwrap();

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_batch_norm_in_place() -> Result<(), String> {
        let mut input = from_data(&[1, 2, 1, 1], vec![1.0, 2.0]);
        let scale = from_data(&[2], vec![3.0, 3.0]);
        let bias = from_data(&[2], vec![0.1, 0.2]);
        let mean = from_data(&[2], vec![0.5, -0.5]);
        let var = from_data(&[2], vec![1.0, 2.0]);

        let epsilon = 1e-5 as f32;

        let y1 = (input[[0, 0, 0, 0]] - mean[[0]]) / (var[[0]] + epsilon).sqrt() * scale[[0]]
            + bias[[0]];
        let y2 = (input[[0, 1, 0, 0]] - mean[[1]]) / (var[[1]] + epsilon).sqrt() * scale[[1]]
            + bias[[1]];
        let expected = from_data(&[1, 2, 1, 1], vec![y1, y2]);

        batch_norm_in_place(&mut input, &scale, &bias, &mean, &var, epsilon).unwrap();

        expect_equal(&input, &expected)
    }

    #[test]
    fn test_log_softmax() -> Result<(), String> {
        // 1D input
        let mut input = tensor!([0.1634, 0.8647, 0.6401, 0.8265, 0.0560]);
        let mut expected = tensor!([-2.0104, -1.3091, -1.5337, -1.3473, -2.1178]);
        let result = log_softmax(input.view(), 0).unwrap();
        expect_equal(&result, &expected)?;

        // Second dimension of 2D input
        input.reshape(&[1, 5]);
        expected.reshape(&[1, 5]);
        let result = log_softmax(input.view(), 1).unwrap();
        expect_equal(&result, &expected)?;

        // First dimension of 2D input
        input.reshape(&[5, 1]);
        expected.reshape(&[5, 1]);
        let result = log_softmax(input.view(), 0).unwrap();
        expect_equal(&result, &expected)?;

        // Second dimension of 2D input with multiple entries in first dim
        let matrix_input = from_data(
            &[2, 5],
            vec![
                0.1634, 0.8647, 0.6401, 0.8265, 0.0560, // First row
                0.1634, 0.8647, 0.6401, 0.8265, 0.0560, // Second row
            ],
        );
        let matrix_expected = from_data(
            &[2, 5],
            vec![
                -2.0104, -1.3091, -1.5337, -1.3473, -2.1178, // First row
                -2.0104, -1.3091, -1.5337, -1.3473, -2.1178, // Second row
            ],
        );
        let result = log_softmax(matrix_input.view(), 1).unwrap();
        expect_equal(&result, &matrix_expected)
    }

    #[test]
    fn test_softmax() -> Result<(), String> {
        // Softmax on a 1D input
        let mut input = tensor!([0.1634, 0.8647, 0.6401, 0.8265, 0.0560]);
        let mut expected = tensor!([0.1339, 0.2701, 0.2157, 0.2599, 0.1203]);
        let result = softmax(input.view(), 0).unwrap();
        expect_equal(&result, &expected)?;

        // Softmax on final dimension of 2D input
        input.reshape(&[1, 5]);
        expected.reshape(&[1, 5]);
        let result = softmax(input.view(), 1).unwrap();
        expect_equal(&result, &expected)?;

        // Softmax on first dimension of 2D input
        input.reshape(&[5, 1]);
        expected.reshape(&[5, 1]);
        let result = softmax(input.view(), 0).unwrap();
        expect_equal(&result, &expected)?;

        // Softmax on second dimension of 2D input with multiple entries in
        // first dim
        let matrix_input = from_data(
            &[2, 5],
            vec![
                0.1634, 0.8647, 0.6401, 0.8265, 0.0560, // First row
                0.1634, 0.8647, 0.6401, 0.8265, 0.0560, // Second row
            ],
        );
        let matrix_expected = from_data(
            &[2, 5],
            vec![
                0.1339, 0.2701, 0.2157, 0.2599, 0.1203, // First row
                0.1339, 0.2701, 0.2157, 0.2599, 0.1203, // Second row
            ],
        );
        let result = softmax(matrix_input.view(), 1).unwrap();
        expect_equal(&result, &matrix_expected)
    }

    // Test softmax with non-contiguous input.
    #[test]
    fn test_softmax_transposed() -> Result<(), String> {
        let mut input = from_data(
            &[4, 4],
            vec![
                0.6427, 0.7435, 0.9762, 0.0611, 0.1249, 0.9742, 0.5826, 0.4704, 0.1420, 0.8376,
                0.6692, 0.7090, 0.2448, 0.9083, 0.2881, 0.4971,
            ],
        );
        let expected = from_data(
            &[4, 4],
            vec![
                0.3480, 0.2073, 0.2109, 0.2337, 0.2204, 0.2776, 0.2421, 0.2599, 0.3433, 0.2316,
                0.2525, 0.1725, 0.1677, 0.2525, 0.3205, 0.2593,
            ],
        );

        input.permute(&[1, 0]);
        let result = softmax(input.view(), 1).unwrap();

        expect_equal(&result, &expected)
    }

    // Test softmax with some additional input sizes and axis dimensions.
    // These tests don't check the individual output values in detail, but they
    // do check the shape and that the values sum to 1.
    #[test]
    fn test_softmax_sizes() {
        let mut rng = XorShiftRng::new(1234);
        let input = rand(&[1, 1, 3, 3], &mut rng);
        let result = softmax(input.view(), 1).unwrap();
        assert_eq!(result.shape(), input.shape());
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 0.001);
    }
}
