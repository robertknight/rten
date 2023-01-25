use crate::check_dims;
use crate::linalg::div_ceil;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output, Padding};
use crate::tensor::{Tensor, TensorLayout};

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
    strides: (usize, usize),
    padding: Padding,
) -> (usize, usize, [usize; 4]) {
    let (in_h, in_w) = in_size;
    let (k_h, k_w) = kernel_size;
    let (stride_h, stride_w) = strides;

    assert!(in_h >= k_h);
    assert!(in_w >= k_w);

    let (out_h, out_w, padding) = match padding {
        Padding::Same => {
            let out_h = div_ceil(in_h, stride_h);
            let out_w = div_ceil(in_w, stride_w);

            let pad_total_h = (out_h - 1) * stride_h + k_h.saturating_sub(in_h);
            let pad_total_w = (out_w - 1) * stride_w + k_w.saturating_sub(in_w);

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
            let out_h = (in_h + pad_top + pad_bottom - k_h) / stride_h + 1;
            let out_w = (in_w + pad_left + pad_right - k_w) / stride_w + 1;
            (out_h, out_w, [pad_top, pad_left, pad_bottom, pad_right])
        }
    };
    (out_h, out_w, padding)
}

pub fn average_pool(
    input: &Tensor,
    kernel_size: [usize; 2],
    strides: [usize; 2],
    padding: Padding,
) -> Result<Tensor, OpError> {
    check_dims!(input, 4);

    let [batch, in_c, in_h, in_w] = input.dims();
    let (out_h, out_w, fixed_padding) = calc_output_size_and_padding(
        (in_h, in_w),
        (kernel_size[0], kernel_size[1]),
        (strides[0], strides[1]),
        padding,
    );
    let [pad_top, pad_left, _pad_bottom, _pad_right] = fixed_padding;
    let [kernel_h, kernel_w] = kernel_size;
    let [stride_h, stride_w] = strides;

    let mut output = Tensor::zeros(&[batch, in_c, out_h, out_w]);

    for n in 0..batch {
        for chan in 0..in_c {
            let mut out_view = output.unchecked_view_mut([n, chan, 0, 0]);
            let in_view = input.unchecked_view([n, chan, 0, 0]);

            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut accumulator = 0.0;
                    let mut non_padding_elements = 0.0;

                    for k_y in 0..kernel_h {
                        for k_x in 0..kernel_w {
                            let in_y = out_y * stride_h + k_y;
                            let in_x = out_x * stride_w + k_x;
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

    Ok(output)
}

#[derive(Debug)]
pub struct AveragePool {
    pub kernel_size: [usize; 2],
    pub padding: Padding,
    pub strides: [usize; 2],
}

impl Operator for AveragePool {
    fn name(&self) -> &str {
        "AveragePool"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        average_pool(input, self.kernel_size, self.strides, self.padding).into_op_result()
    }
}

pub fn global_average_pool(input: &Tensor) -> Result<Tensor, OpError> {
    check_dims!(input, 4);

    let [batch, chans, in_h, in_w] = input.dims();
    let mut output = Tensor::zeros(&[batch, chans, 1, 1]);

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

    Ok(output)
}

#[derive(Debug)]
pub struct GlobalAveragePool {}

impl Operator for GlobalAveragePool {
    fn name(&self) -> &str {
        "GlobalAveragePool"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        global_average_pool(input).into_op_result()
    }
}

pub fn max_pool(
    input: &Tensor,
    kernel_size: [usize; 2],
    strides: [usize; 2],
    padding: Padding,
) -> Result<Tensor, OpError> {
    check_dims!(input, 4);

    let [batch, in_c, in_h, in_w] = input.dims();
    let (out_h, out_w, fixed_padding) = calc_output_size_and_padding(
        (in_h, in_w),
        (kernel_size[0], kernel_size[1]),
        (strides[0], strides[1]),
        padding,
    );
    let [pad_top, pad_left, _pad_bottom, _pad_right] = fixed_padding;
    let [kernel_h, kernel_w] = kernel_size;
    let [stride_h, stride_w] = strides;

    let mut output = Tensor::zeros(&[batch, in_c, out_h, out_w]);

    for n in 0..batch {
        for chan in 0..in_c {
            let mut out_view = output.unchecked_view_mut([n, chan, 0, 0]);
            let in_view = input.unchecked_view([n, chan, 0, 0]);

            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut accumulator = f32::NEG_INFINITY;
                    for k_y in 0..kernel_h {
                        for k_x in 0..kernel_w {
                            let in_y = out_y * stride_h + k_y;
                            let in_x = out_x * stride_w + k_x;
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

    Ok(output)
}

#[derive(Debug)]
pub struct MaxPool {
    pub kernel_size: [usize; 2],
    pub padding: Padding,
    pub strides: [usize; 2],
}

impl Operator for MaxPool {
    fn name(&self) -> &str {
        "MaxPool"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        max_pool(input, self.kernel_size, self.strides, self.padding).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{average_pool, global_average_pool, max_pool, Padding};
    use crate::tensor::{from_2d_slice, from_data, zeros, Tensor, TensorLayout};
    use crate::test_util::expect_equal;

    #[test]
    fn test_average_pool() -> Result<(), String> {
        let input = from_data(
            &[1, 1, 4, 4],
            vec![
                0.1, 0.2, 0.3, 0.4, // Y=0
                0.5, 0.6, 0.7, 0.8, // Y=1
                0.1, 0.2, 0.3, 0.4, // Y=2
                0.6, 0.7, 0.8, 0.9, // Y=3
            ],
        );

        struct Case {
            kernel_size: [usize; 2],
            strides: [usize; 2],
            expected: Tensor,
        }

        let cases = [
            // Most common case of uniform stride and kernel size
            Case {
                kernel_size: [2, 2],
                strides: [2, 2],
                expected: from_data(&[1, 1, 2, 2], vec![0.35, 0.55, 0.4, 0.6]),
            },
            // Large uniform kernel size and stride
            Case {
                kernel_size: [4, 4],
                strides: [4, 4],
                expected: from_data(&[1, 1, 1, 1], vec![0.475]),
            },
            // Kernel height > kernel width
            Case {
                kernel_size: [2, 4],
                strides: [2, 4],
                expected: from_data(&[1, 1, 2, 1], vec![0.45, 0.5]),
            },
            // W stride > H stride
            Case {
                kernel_size: [2, 2],
                strides: [1, 2],
                expected: from_data(
                    &[1, 1, 3, 2],
                    vec![
                        0.35, 0.55, // Y=0
                        0.35, 0.55, // Y=1
                        0.4, 0.6, // Y=2
                    ],
                ),
            },
            // H stride > W stride
            Case {
                kernel_size: [2, 2],
                strides: [2, 1],
                expected: from_data(
                    &[1, 1, 2, 3],
                    vec![
                        0.35, 0.45, // Y=0
                        0.55, 0.4, // Y=1
                        0.5, 0.6, // Y=2
                    ],
                ),
            },
        ];

        for case in cases {
            let result = average_pool(
                &input,
                case.kernel_size,
                case.strides,
                Padding::Fixed([0, 0, 0, 0]),
            )
            .unwrap();
            expect_equal(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_average_pool_padding() -> Result<(), String> {
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

        let result = average_pool(
            &input,
            [2, 2],
            [2, 2], /* stride */
            Padding::Fixed([1, 1, 1, 1]),
        )
        .unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_global_average_pool() -> Result<(), String> {
        let input = from_data(&[1, 2, 2, 2], vec![1., 2., 3., 4., 10., 20., 30., 40.]);
        let expected = from_data(&[1, 2, 1, 1], vec![2.5, 25.]);
        let result = global_average_pool(&input).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_max_pool() -> Result<(), String> {
        let input = from_data(
            &[1, 1, 4, 4],
            vec![
                0.1, 0.2, 0.3, 0.4, // Y=0
                0.5, 0.6, 0.7, 0.8, // Y=1
                0.1, 0.2, 0.3, 0.4, // Y=2
                0.6, 0.7, 0.8, 0.9, // Y=3
            ],
        );

        struct Case {
            kernel_size: [usize; 2],
            strides: [usize; 2],
            expected: Tensor,
        }

        let cases = [
            // Most common case of uniform stride and kernel size
            Case {
                kernel_size: [2, 2],
                strides: [2, 2],
                expected: from_data(&[1, 1, 2, 2], vec![0.6, 0.8, 0.7, 0.9]),
            },
            // Large uniform kernel size and stride
            Case {
                kernel_size: [4, 4],
                strides: [4, 4],
                expected: from_data(&[1, 1, 1, 1], vec![0.9]),
            },
            // Kernel height > kernel width
            Case {
                kernel_size: [2, 4],
                strides: [2, 4],
                expected: from_data(&[1, 1, 2, 1], vec![0.8, 0.9]),
            },
            // W stride > H stride
            Case {
                kernel_size: [2, 2],
                strides: [1, 2],
                expected: from_data(
                    &[1, 1, 3, 2],
                    vec![
                        0.6, 0.8, // Y=0
                        0.6, 0.8, // Y=1
                        0.7, 0.9, // Y=2
                    ],
                ),
            },
            // H stride > W stride
            Case {
                kernel_size: [2, 2],
                strides: [2, 1],
                expected: from_data(
                    &[1, 1, 2, 3],
                    vec![
                        0.6, 0.7, 0.8, // Y=0
                        0.7, 0.8, 0.9, // Y=1
                    ],
                ),
            },
        ];

        for case in cases {
            let result = max_pool(
                &input,
                case.kernel_size,
                case.strides,
                Padding::Fixed([0, 0, 0, 0]),
            )
            .unwrap();
            expect_equal(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_max_pool_padding() {
        let input = zeros(&[1, 1, 9, 9]);

        let result = max_pool(&input, [2, 2], [2, 2], Padding::Fixed([0, 0, 0, 0])).unwrap();
        assert_eq!(result.shape(), &[1, 1, 4, 4]);

        let result = max_pool(&input, [2, 2], [2, 2], Padding::Fixed([1, 1, 1, 1])).unwrap();
        assert_eq!(result.shape(), &[1, 1, 5, 5]);

        let result = max_pool(&input, [2, 2], [2, 2], Padding::Fixed([2, 2, 2, 2])).unwrap();
        assert_eq!(result.shape(), &[1, 1, 6, 6]);

        let result = max_pool(&input, [2, 2], [2, 2], Padding::Same).unwrap();
        assert_eq!(result.shape(), &[1, 1, 5, 5]);

        let result = max_pool(&input, [2, 2], [3, 3], Padding::Same).unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 3]);
    }
}
