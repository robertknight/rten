use crate::linalg::div_ceil;
use crate::ops::{get_input_as_float, Input, OpError, Operator, Output, Padding};
use crate::tensor::{zero_tensor, Tensor};

/// Calculate the output size and padding for a convolution or pooling operation.
///
/// Depending on the padding mode, the output size is be calculated from the
/// input size and padding, or the padding size is calculated from the input
/// size.
///
/// See https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool for
/// formulae. These includes extensions to support dilations in future.
///
/// Returns an `(out_h, out_w, [pad_top, pad_left, pad_bottom, pad_right])`
/// tuple.
pub fn calc_output_size_and_padding(
    in_size: (usize, usize),
    kernel_size: (usize, usize),
    stride: usize,
    padding: Padding,
) -> (usize, usize, [usize; 4]) {
    let (in_h, in_w) = in_size;
    let (k_h, k_w) = kernel_size;

    assert!(in_h >= k_h);
    assert!(in_w >= k_w);

    let (out_h, out_w, padding) = match padding {
        Padding::Same => {
            let out_h = div_ceil(in_h, stride);
            let out_w = div_ceil(in_w, stride);

            let pad_total_h = (out_h - 1) * stride + k_h.saturating_sub(in_h);
            let pad_total_w = (out_w - 1) * stride + k_w.saturating_sub(in_w);

            let pad_top = pad_total_h / 2;
            let pad_left = pad_total_w / 2;

            // If the total padding is not even, we assign the remaining unit to
            // the ends of the axis. This matches the ONNX "SAME_UPPER"
            // value for `auto_pad`.
            let pad_bottom = div_ceil(pad_total_h, 2);
            let pad_right = div_ceil(pad_total_w, 2);

            (out_h, out_w, [pad_top, pad_left, pad_bottom, pad_right])
        }
        Padding::Fixed([pad_top, pad_left, pad_bottom, pad_right]) => {
            let out_h = (in_h + pad_top + pad_bottom - k_h) / stride + 1;
            let out_w = (in_w + pad_left + pad_right - k_w) / stride + 1;
            (out_h, out_w, [pad_top, pad_left, pad_bottom, pad_right])
        }
    };
    (out_h, out_w, padding)
}

pub fn average_pool_2d(
    input: &Tensor,
    kernel_size: usize,
    stride: usize,
    padding: Padding,
) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let (out_h, out_w, fixed_padding) =
        calc_output_size_and_padding((in_h, in_w), (kernel_size, kernel_size), stride, padding);
    let [pad_top, pad_left, _pad_bottom, _pad_right] = fixed_padding;

    let mut output = zero_tensor::<f32>(&[batch, in_c, out_h, out_w]);

    for n in 0..batch {
        for chan in 0..in_c {
            let mut out_view = output.unchecked_view_mut([n, chan, 0, 0]);
            let in_view = input.unchecked_view([n, chan, 0, 0]);

            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut accumulator = 0.0;
                    let mut non_padding_elements = 0.0;

                    for k_y in 0..kernel_size {
                        for k_x in 0..kernel_size {
                            let in_y = out_y * stride + k_y;
                            let in_x = out_x * stride + k_x;
                            if in_y >= pad_top
                                && in_y < in_h + pad_top
                                && in_x >= pad_left
                                && in_x < in_w + pad_left
                            {
                                let val = in_view[[in_y - pad_top, in_x - pad_left]];
                                accumulator += val;
                                non_padding_elements += 1.0;
                            }
                        }
                    }

                    out_view[[out_y, out_x]] = accumulator / non_padding_elements;
                }
            }
        }
    }

    output
}

#[derive(Debug)]
pub struct AveragePool2d {
    pub kernel_size: usize,
    pub padding: Padding,
    pub stride: usize,
}

impl Operator for AveragePool2d {
    fn name(&self) -> &str {
        "AveragePool2d"
    }

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        Ok(average_pool_2d(input, self.kernel_size, self.stride, self.padding).into())
    }
}

pub fn global_average_pool(input: &Tensor) -> Tensor {
    let [batch, chans, in_h, in_w] = input.dims();
    let mut output = zero_tensor(&[batch, chans, 1, 1]);

    let hw_float = (in_h * in_w) as f32;

    for n in 0..batch {
        for c in 0..chans {
            let in_view = input.unchecked_view([n, c, 0, 0]);
            let mut sum = 0.0;
            for y in 0..in_h {
                for x in 0..in_w {
                    sum += in_view[[y, x]];
                }
            }
            output[[n, c, 0, 0]] = sum / hw_float;
        }
    }

    output
}

#[derive(Debug)]
pub struct GlobalAveragePool {}

impl Operator for GlobalAveragePool {
    fn name(&self) -> &str {
        "GlobalAveragePool"
    }

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        Ok(global_average_pool(input).into())
    }
}

pub fn max_pool_2d(input: &Tensor, kernel_size: usize, stride: usize, padding: Padding) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let (out_h, out_w, fixed_padding) =
        calc_output_size_and_padding((in_h, in_w), (kernel_size, kernel_size), stride, padding);
    let [pad_top, pad_left, _pad_bottom, _pad_right] = fixed_padding;

    let mut output = zero_tensor::<f32>(&[batch, in_c, out_h, out_w]);

    for n in 0..batch {
        for chan in 0..in_c {
            let mut out_view = output.unchecked_view_mut([n, chan, 0, 0]);
            let in_view = input.unchecked_view([n, chan, 0, 0]);

            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut accumulator = f32::NEG_INFINITY;
                    for k_y in 0..kernel_size {
                        for k_x in 0..kernel_size {
                            let in_y = out_y * stride + k_y;
                            let in_x = out_x * stride + k_x;
                            if in_y >= pad_top
                                && in_y < in_h + pad_top
                                && in_x >= pad_left
                                && in_x < in_w + pad_left
                            {
                                let val = in_view[[in_y - pad_top, in_x - pad_left]];
                                accumulator = accumulator.max(val);
                            }
                        }
                    }
                    out_view[[out_y, out_x]] = accumulator;
                }
            }
        }
    }

    output
}

#[derive(Debug)]
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub padding: Padding,
    pub stride: usize,
}

impl Operator for MaxPool2d {
    fn name(&self) -> &str {
        "MaxPool2d"
    }

    fn run(&self, inputs: &[Input]) -> Result<Output, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        Ok(max_pool_2d(input, self.kernel_size, self.stride, self.padding).into())
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{average_pool_2d, global_average_pool, max_pool_2d, Padding};
    use crate::tensor::{from_2d_slice, from_data, zero_tensor};
    use crate::test_util::expect_equal;

    #[test]
    fn test_average_pool_2d() -> Result<(), String> {
        let height = 4;
        let width = 4;
        let mut input = zero_tensor(&[1, 1, height, width]);

        for y in 0..height {
            for x in 0..width {
                input[[0, 0, y, x]] = (y as f32) * 10.0 + (x as f32);
            }
        }

        let sum_a: f32 = input
            .slice_elements(&[(0, 1, 1), (0, 1, 1), (0, 2, 1), (0, 2, 1)])
            .sum();
        let sum_b: f32 = input
            .slice_elements(&[(0, 1, 1), (0, 1, 1), (0, 2, 1), (2, 4, 1)])
            .sum();
        let sum_c: f32 = input
            .slice_elements(&[(0, 1, 1), (0, 1, 1), (2, 4, 1), (0, 2, 1)])
            .sum();
        let sum_d: f32 = input
            .slice_elements(&[(0, 1, 1), (0, 1, 1), (2, 4, 1), (2, 4, 1)])
            .sum();

        let expected = from_data(
            vec![1, 1, 2, 2],
            vec![sum_a / 4.0, sum_b / 4.0, sum_c / 4.0, sum_d / 4.0],
        );

        let result = average_pool_2d(&input, 2, 2 /* stride */, Padding::Fixed([0, 0, 0, 0]));
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_average_pool_2d_padding() -> Result<(), String> {
        let mut input = from_2d_slice(&[
            &[0.0809, 0.5529, 0.1534, 0.7507],
            &[0.4698, 0.7771, 0.9896, 0.4873],
            &[0.9750, 0.5160, 0.6419, 0.3670],
            &[0.4101, 0.3762, 0.9689, 0.4389],
        ]);
        let [rows, cols] = input.dims();
        input.reshape(&[1, 1, rows, cols]);

        // Computed with `torch.nn.functional.avg_pool2d` in PyTorch with
        // `padding=1` and `count_include_pad=False`.
        let mut expected = from_2d_slice(&[
            &[0.0809, 0.3531, 0.7507],
            &[0.7224, 0.7312, 0.4271],
            &[0.4101, 0.6725, 0.4389],
        ]);
        let [rows, cols] = expected.dims();
        expected.reshape(&[1, 1, rows, cols]);

        let result = average_pool_2d(&input, 2, 2 /* stride */, Padding::Fixed([1, 1, 1, 1]));
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_global_average_pool() -> Result<(), String> {
        let input = from_data(vec![1, 2, 2, 2], vec![1., 2., 3., 4., 10., 20., 30., 40.]);
        let expected = from_data(vec![1, 2, 1, 1], vec![2.5, 25.]);
        let result = global_average_pool(&input);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_max_pool_2d() -> Result<(), String> {
        let height = 4;
        let width = 8;
        let mut input = zero_tensor(&[1, 1, height, width]);

        input[[0, 0, 0, 0]] = 1.0;
        input[[0, 0, 0, 1]] = 2.0;
        input[[0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 1]] = 4.0;

        input[[0, 0, 0, 2]] = 0.1;
        input[[0, 0, 0, 3]] = 0.2;
        input[[0, 0, 1, 2]] = 0.3;
        input[[0, 0, 1, 3]] = 0.4;

        let expected = from_data(
            vec![1, 1, 2, 4],
            vec![4.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let result = max_pool_2d(&input, 2, 2 /* stride */, Padding::Fixed([0, 0, 0, 0]));
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_max_pool_2d_stride() -> Result<(), String> {
        let mut input = zero_tensor(&[1, 1, 9, 9]);

        for y in 0..9 {
            for x in 0..9 {
                if x % 3 == 2 && y % 3 == 2 {
                    // Set every third element along each axis to a large
                    // value. These should be skipped over due to the stride
                    // below.
                    input[[0, 0, y, x]] = 1000.0;
                } else {
                    // Result should be the pooled values of these entries.
                    input[[0, 0, y, x]] = ((y / 3) * 10 + x / 3) as f32;
                }
            }
        }

        let result = max_pool_2d(&input, 2, 3 /* stride */, Padding::Fixed([0, 0, 0, 0]));
        let expected = from_data(
            vec![1, 1, 3, 3],
            vec![0., 1., 2., 10., 11., 12., 20., 21., 22.],
        );

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_max_pool_2d_padding() {
        let input = zero_tensor(&[1, 1, 9, 9]);

        let result = max_pool_2d(&input, 2, 2, Padding::Fixed([0, 0, 0, 0]));
        assert_eq!(result.shape(), &[1, 1, 4, 4]);

        let result = max_pool_2d(&input, 2, 2, Padding::Fixed([1, 1, 1, 1]));
        assert_eq!(result.shape(), &[1, 1, 5, 5]);

        let result = max_pool_2d(&input, 2, 2, Padding::Fixed([2, 2, 2, 2]));
        assert_eq!(result.shape(), &[1, 1, 6, 6]);

        let result = max_pool_2d(&input, 2, 2, Padding::Same);
        assert_eq!(result.shape(), &[1, 1, 5, 5]);

        let result = max_pool_2d(&input, 2, 3, Padding::Same);
        assert_eq!(result.shape(), &[1, 1, 3, 3]);
    }
}
