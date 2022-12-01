use crate::ops::{get_input_as_float, Input, IntoOpResult, OpError, Operator, Output};
use crate::tensor::Tensor;

/// Perform in-place batch normalization on the NCHW tensor `out`.
///
/// See https://github.com/onnx/onnx/blob/main/docs/Operators.md#batchnormalization
pub fn batch_norm_in_place(
    out: &mut Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) {
    let [batch, chans, in_h, in_w] = out.dims();
    for n in 0..batch {
        for c in 0..chans {
            let chan_mean = mean[[c]];
            let chan_var = var[[c]];
            let chan_scale = scale[[c]];
            let chan_bias = bias[[c]];

            let mut out_view = out.unchecked_view_mut([n, c, 0, 0]);

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
) -> Tensor {
    let mut output = input.clone();
    batch_norm_in_place(&mut output, scale, bias, mean, var, epsilon);
    output
}

#[derive(Debug)]
pub struct BatchNormalization {
    pub epsilon: f32,
}

impl Operator for BatchNormalization {
    fn name(&self) -> &str {
        "BatchNormalization"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        let scale = get_input_as_float(inputs, 1)?;
        let bias = get_input_as_float(inputs, 2)?;
        let mean = get_input_as_float(inputs, 3)?;
        let var = get_input_as_float(inputs, 4)?;

        batch_norm(input, scale, bias, mean, var, self.epsilon).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Result<Output, OpError> {
        let mut output = input.into_float().ok_or(OpError::UnsupportedInputType)?;
        let scale = get_input_as_float(other, 0)?;
        let bias = get_input_as_float(other, 1)?;
        let mean = get_input_as_float(other, 2)?;
        let var = get_input_as_float(other, 3)?;

        batch_norm_in_place(&mut output, scale, bias, mean, var, self.epsilon);

        Ok(output.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{batch_norm, batch_norm_in_place};
    use crate::tensor::from_data;
    use crate::test_util::expect_equal;

    #[test]
    fn test_batch_norm() -> Result<(), String> {
        let input = from_data(vec![1, 2, 1, 1], vec![1.0, 2.0]);
        let scale = from_data(vec![2], vec![3.0, 3.0]);
        let bias = from_data(vec![2], vec![0.1, 0.2]);
        let mean = from_data(vec![2], vec![0.5, -0.5]);
        let var = from_data(vec![2], vec![1.0, 2.0]);

        let epsilon = 1e-5 as f32;

        let y1 = (input[[0, 0, 0, 0]] - mean[[0]]) / (var[[0]] + epsilon).sqrt() * scale[[0]]
            + bias[[0]];
        let y2 = (input[[0, 1, 0, 0]] - mean[[1]]) / (var[[1]] + epsilon).sqrt() * scale[[1]]
            + bias[[1]];
        let expected = from_data(vec![1, 2, 1, 1], vec![y1, y2]);
        let result = batch_norm(&input, &scale, &bias, &mean, &var, epsilon);

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_batch_norm_in_place() -> Result<(), String> {
        let mut input = from_data(vec![1, 2, 1, 1], vec![1.0, 2.0]);
        let scale = from_data(vec![2], vec![3.0, 3.0]);
        let bias = from_data(vec![2], vec![0.1, 0.2]);
        let mean = from_data(vec![2], vec![0.5, -0.5]);
        let var = from_data(vec![2], vec![1.0, 2.0]);

        let epsilon = 1e-5 as f32;

        let y1 = (input[[0, 0, 0, 0]] - mean[[0]]) / (var[[0]] + epsilon).sqrt() * scale[[0]]
            + bias[[0]];
        let y2 = (input[[0, 1, 0, 0]] - mean[[1]]) / (var[[1]] + epsilon).sqrt() * scale[[1]]
            + bias[[1]];
        let expected = from_data(vec![1, 2, 1, 1], vec![y1, y2]);

        batch_norm_in_place(&mut input, &scale, &bias, &mean, &var, epsilon);

        expect_equal(&input, &expected)
    }
}
