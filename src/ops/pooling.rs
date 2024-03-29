use std::iter::zip;

use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut, Tensor, TensorView};

use crate::check_dims;
use crate::gemm::div_ceil;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output, Padding};

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
///
/// Returns an error if the padded input size is too small for the kernel
/// size.
pub fn calc_output_size_and_padding(
    in_size: (usize, usize),
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: Padding,
    dilations: Option<(usize, usize)>,
) -> Result<(usize, usize, [usize; 4]), OpError> {
    let (in_h, in_w) = in_size;
    let (k_h, k_w) = kernel_size;
    let (stride_h, stride_w) = strides;
    let (dilation_y, dilation_x) = dilations.unwrap_or((1, 1));

    if dilation_y == 0 || dilation_x == 0 {
        return Err(OpError::InvalidValue("Dilations must be > 0"));
    }

    if stride_h == 0 || stride_w == 0 {
        return Err(OpError::InvalidValue("Strides must be > 0"));
    }

    let (out_h, out_w, padding) = match padding {
        Padding::Same => {
            let out_h = div_ceil(in_h, stride_h);
            let out_w = div_ceil(in_w, stride_w);

            let pad_total_h =
                ((out_h - 1) * stride_h + (k_h - 1) * dilation_y + 1).saturating_sub(in_h);
            let pad_total_w =
                ((out_w - 1) * stride_w + (k_w - 1) * dilation_x + 1).saturating_sub(in_w);

            let pad_top = pad_total_h / 2;
            let pad_left = pad_total_w / 2;

            // If the total padding is not even, we assign the remaining unit to
            // the ends of the axis. This matches the ONNX "SAME_UPPER"
            // value for `auto_pad`.
            let pad_bottom = div_ceil(pad_total_h, 2);
            let pad_right = div_ceil(pad_total_w, 2);

            (out_h, out_w, [pad_top, pad_left, pad_bottom, pad_right])
        }
        Padding::Fixed(pads) => {
            let [pad_top, pad_left, pad_bottom, pad_right]: [usize; 4] = pads
                .as_slice()
                .try_into()
                .map_err(|_| OpError::InvalidValue("Expected 4 padding values"))?;
            let padded_in_h = in_h + pad_top + pad_bottom;
            let padded_in_w = in_w + pad_left + pad_right;

            let dilated_k_h = k_h + (k_h - 1) * (dilation_y - 1);
            let dilated_k_w = k_w + (k_w - 1) * (dilation_x - 1);

            if padded_in_h < dilated_k_h || padded_in_w < dilated_k_w {
                return Err(OpError::InvalidValue("Input too small for kernel size"));
            }

            let out_h = (padded_in_h - dilation_y * (k_h - 1) - 1) / stride_h + 1;
            let out_w = (padded_in_w - dilation_x * (k_w - 1) - 1) / stride_w + 1;
            (out_h, out_w, [pad_top, pad_left, pad_bottom, pad_right])
        }
    };
    Ok((out_h, out_w, padding))
}

pub fn average_pool(
    input: TensorView,
    kernel_size: [usize; 2],
    strides: [usize; 2],
    padding: Padding,
    count_include_pad: bool,
) -> Result<Tensor, OpError> {
    let [batch, in_c, in_h, in_w] = check_dims!(input, 4, "NCHW");
    let (out_h, out_w, fixed_padding) = calc_output_size_and_padding(
        (in_h, in_w),
        (kernel_size[0], kernel_size[1]),
        (strides[0], strides[1]),
        padding,
        None,
    )?;
    let [pad_top, pad_left, _pad_bottom, _pad_right] = fixed_padding;
    let [kernel_h, kernel_w] = kernel_size;
    let [stride_h, stride_w] = strides;

    let mut output = NdTensor::zeros([batch, in_c, out_h, out_w]);
    let input = input.nd_view::<4>();

    for n in 0..batch {
        for chan in 0..in_c {
            let mut out_view = output.slice_mut([n, chan]);
            let mut out_view = out_view.weakly_checked_view_mut();
            let in_view = input.slice([n, chan]).weakly_checked_view();

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

                    let counted_elems = if count_include_pad {
                        (kernel_h * kernel_w) as f32
                    } else {
                        non_padding_elements
                    };

                    out_view[[out_y, out_x]] = accumulator / counted_elems;
                }
            }
        }
    }

    Ok(output.into_dyn())
}

#[derive(Debug)]
pub struct AveragePool {
    pub kernel_size: [usize; 2],
    pub padding: Padding,
    pub count_include_pad: bool,
    pub strides: [usize; 2],
}

impl Operator for AveragePool {
    fn name(&self) -> &str {
        "AveragePool"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        average_pool(
            input,
            self.kernel_size,
            self.strides,
            self.padding.clone(),
            self.count_include_pad,
        )
        .into_op_result()
    }
}

pub fn global_average_pool(input: TensorView) -> Result<Tensor, OpError> {
    let [batch, chans, in_h, in_w] = check_dims!(input, 4, "NCHW");

    let mut output = Tensor::zeros(&[batch, chans, 1, 1]);

    for n in 0..batch {
        const N: usize = 4;

        for (chan_group, mut out_group) in zip(
            input.slice::<3, _>(n).axis_chunks(0, N),
            output
                .slice_mut::<1, _>((n, .., 0, 0))
                .axis_chunks_mut(0, N),
        ) {
            if chan_group.size(0) == N {
                // Compute average over batch of N channels in parallel.
                let chan_group = chan_group.nd_view();

                let mut sums = [0.; N];
                for y in 0..chan_group.size(1) {
                    for x in 0..chan_group.size(2) {
                        let vals: [f32; N] = chan_group.get_array([0, y, x], 0);
                        for i in 0..N {
                            sums[i] += vals[i];
                        }
                    }
                }

                for i in 0..N {
                    out_group[[i]] = sums[i] / (in_h * in_w) as f32;
                }
            } else {
                // Compute average over remaining channels.
                for i in 0..chan_group.size(0) {
                    let sum: f32 = chan_group.slice::<2, _>([i]).iter().sum();
                    out_group[[i]] = sum / (in_h * in_w) as f32;
                }
            }
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
    input: TensorView,
    kernel_size: [usize; 2],
    strides: [usize; 2],
    padding: Padding,
) -> Result<Tensor, OpError> {
    let [batch, in_c, in_h, in_w] = check_dims!(input, 4, "NCHW");
    let (out_h, out_w, fixed_padding) = calc_output_size_and_padding(
        (in_h, in_w),
        (kernel_size[0], kernel_size[1]),
        (strides[0], strides[1]),
        padding,
        None, /* dilations */
    )?;
    let [pad_top, pad_left, _pad_bottom, _pad_right] = fixed_padding;
    let mut output = Tensor::zeros(&[batch, in_c, out_h, out_w]);

    // Apply max-pooling to the channel indexes specified by `chans`.
    // Assuming `N` is chosen appropriately the inner loop should get unrolled /
    // autovectorized.
    fn max_pool_chans<const N: usize>(
        mut out: NdTensorViewMut<f32, 3>,
        in_view: NdTensorView<f32, 3>,
        chans: [usize; N],
        [kernel_h, kernel_w]: [usize; 2],
        [stride_h, stride_w]: [usize; 2],
        [pad_top, pad_left]: [usize; 2],
    ) {
        let [out_chans, out_h, out_w] = out.shape();
        let [in_chans, in_h, in_w] = in_view.shape();
        assert!(chans.into_iter().all(|c| c < out_chans && c < in_chans));

        for out_y in 0..out_h {
            for out_x in 0..out_w {
                let mut accumulator = [f32::NEG_INFINITY; N];
                for k_y in 0..kernel_h {
                    for k_x in 0..kernel_w {
                        let in_y = out_y * stride_h + k_y;
                        let in_x = out_x * stride_w + k_x;
                        if in_y >= pad_top
                            && in_y < in_h + pad_top
                            && in_x >= pad_left
                            && in_x < in_w + pad_left
                        {
                            for (i, chan) in chans.into_iter().enumerate() {
                                // Safety:
                                //  - We checked all `chans` are in-bounds
                                //  - `in_y` and `in_x` are >= pad_top and pad_left
                                let val = unsafe {
                                    *in_view.get_unchecked([chan, in_y - pad_top, in_x - pad_left])
                                };
                                accumulator[i] = accumulator[i].max(val);
                            }
                        }
                    }
                }
                for (i, chan) in chans.into_iter().enumerate() {
                    // Safety:
                    //  - We checked all `chans` are in-bounds
                    //  - `out_y` and `out_x` are in 0..out_h, 0..out_w
                    unsafe {
                        *out.get_unchecked_mut([chan, out_y, out_x]) = accumulator[i];
                    }
                }
            }
        }
    }

    zip(output.axis_iter_mut(0), input.axis_iter(0))
        .par_bridge()
        .for_each(|(mut out_item, in_item)| {
            let mut out_item = out_item.nd_view_mut();
            let in_item = in_item.nd_view();

            // Loop over channel groups.
            const N: usize = 4;
            for chan in (0..in_c).step_by(N) {
                if in_c - chan < N {
                    break;
                }
                max_pool_chans(
                    out_item.view_mut(),
                    in_item,
                    [chan, chan + 1, chan + 2, chan + 3],
                    kernel_size,
                    strides,
                    [pad_top, pad_left],
                );
            }

            // Loop over remaining channels.
            for chan in (in_c - in_c % N)..in_c {
                max_pool_chans(
                    out_item.view_mut(),
                    in_item,
                    [chan],
                    kernel_size,
                    strides,
                    [pad_top, pad_left],
                );
            }
        });

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
        max_pool(input, self.kernel_size, self.strides, self.padding.clone()).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::Tensor;

    use super::calc_output_size_and_padding;
    use crate::ops::tests::expect_eq_1e4;
    use crate::ops::{average_pool, global_average_pool, max_pool, OpError, Padding};

    #[test]
    fn test_average_pool() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from_data(
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
                expected: Tensor::from_data(&[1, 1, 2, 2], vec![0.35, 0.55, 0.4, 0.6]),
            },
            // Large uniform kernel size and stride
            Case {
                kernel_size: [4, 4],
                strides: [4, 4],
                expected: Tensor::from_data(&[1, 1, 1, 1], vec![0.475]),
            },
            // Kernel height > kernel width
            Case {
                kernel_size: [2, 4],
                strides: [2, 4],
                expected: Tensor::from_data(&[1, 1, 2, 1], vec![0.45, 0.5]),
            },
            // W stride > H stride
            Case {
                kernel_size: [2, 2],
                strides: [1, 2],
                expected: Tensor::from_data(
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
                expected: Tensor::from_data(
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
                input.view(),
                case.kernel_size,
                case.strides,
                [0, 0, 0, 0].into(),
                false, /* count_include_pad */
            )
            .unwrap();
            expect_equal(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_average_pool_padding() -> Result<(), Box<dyn Error>> {
        let mut input = Tensor::from([
            [0.0809, 0.5529, 0.1534, 0.7507],
            [0.4698, 0.7771, 0.9896, 0.4873],
            [0.9750, 0.5160, 0.6419, 0.3670],
            [0.4101, 0.3762, 0.9689, 0.4389],
        ]);
        let [rows, cols]: [usize; 2] = input.shape().try_into().unwrap();
        input.reshape(&[1, 1, rows, cols]);

        // Computed with `torch.nn.functional.avg_pool2d` in PyTorch with
        // `padding=1` and `count_include_pad=False`.
        let mut expected = Tensor::from([
            [0.0809, 0.3531, 0.7507],
            [0.7224, 0.7312, 0.4271],
            [0.4101, 0.6725, 0.4389],
        ]);
        let [rows, cols]: [usize; 2] = expected.shape().try_into().unwrap();
        expected.reshape(&[1, 1, rows, cols]);

        let result = average_pool(
            input.view(),
            [2, 2],
            [2, 2], /* stride */
            [1, 1, 1, 1].into(),
            false, /* count_include_pad */
        )
        .unwrap();
        expect_eq_1e4(&result, &expected)?;

        // As above, but with `count_include_pad=True`.
        let expected_include_pad = Tensor::from([
            [0.0202, 0.1766, 0.1877],
            [0.3612, 0.7312, 0.2136],
            [0.1025, 0.3363, 0.1097],
        ])
        .into_shape([1, 1, 3, 3])
        .into_dyn();
        let result = average_pool(
            input.view(),
            [2, 2],
            [2, 2], /* stride */
            [1, 1, 1, 1].into(),
            true, /* count_include_pad */
        )
        .unwrap();
        expect_eq_1e4(&result, &expected_include_pad)?;

        Ok(())
    }

    #[test]
    fn test_global_average_pool() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from_data(&[1, 2, 2, 2], vec![1., 2., 3., 4., 10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[1, 2, 1, 1], vec![2.5, 25.]);
        let result = global_average_pool(input.view()).unwrap();
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_max_pool() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from_data(
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
                expected: Tensor::from_data(&[1, 1, 2, 2], vec![0.6, 0.8, 0.7, 0.9]),
            },
            // Large uniform kernel size and stride
            Case {
                kernel_size: [4, 4],
                strides: [4, 4],
                expected: Tensor::from_data(&[1, 1, 1, 1], vec![0.9]),
            },
            // Kernel height > kernel width
            Case {
                kernel_size: [2, 4],
                strides: [2, 4],
                expected: Tensor::from_data(&[1, 1, 2, 1], vec![0.8, 0.9]),
            },
            // W stride > H stride
            Case {
                kernel_size: [2, 2],
                strides: [1, 2],
                expected: Tensor::from_data(
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
                expected: Tensor::from_data(
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
                input.view(),
                case.kernel_size,
                case.strides,
                [0, 0, 0, 0].into(),
            )
            .unwrap();
            expect_equal(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_max_pool_padding() {
        let input = Tensor::zeros(&[1, 1, 9, 9]);

        let result = max_pool(input.view(), [2, 2], [2, 2], [0, 0, 0, 0].into()).unwrap();
        assert_eq!(result.shape(), &[1, 1, 4, 4]);

        let result = max_pool(input.view(), [2, 2], [2, 2], [1, 1, 1, 1].into()).unwrap();
        assert_eq!(result.shape(), &[1, 1, 5, 5]);

        let result = max_pool(input.view(), [2, 2], [2, 2], [2, 2, 2, 2].into()).unwrap();
        assert_eq!(result.shape(), &[1, 1, 6, 6]);

        let result = max_pool(input.view(), [2, 2], [2, 2], Padding::Same).unwrap();
        assert_eq!(result.shape(), &[1, 1, 5, 5]);

        let result = max_pool(input.view(), [2, 2], [3, 3], Padding::Same).unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_calc_output_size_and_padding() {
        struct Case {
            in_size: (usize, usize),
            kernel_size: (usize, usize),
            dilations: (usize, usize),
            strides: (usize, usize),
            padding: Padding,
            expected: Result<(usize, usize, [usize; 4]), OpError>,
        }

        let zero_padding: Padding = [0, 0, 0, 0].into();

        let cases = [
            // Simple case with no padding
            Case {
                in_size: (5, 5),
                kernel_size: (3, 3),
                dilations: (1, 1),
                strides: (1, 1),
                padding: zero_padding.clone(),
                expected: Ok((3, 3, [0, 0, 0, 0])),
            },
            // Fixed padding
            Case {
                in_size: (5, 5),
                kernel_size: (3, 3),
                dilations: (1, 1),
                strides: (1, 1),
                padding: [1, 1, 1, 1].into(),
                expected: Ok((5, 5, [1, 1, 1, 1])),
            },
            // Strides > 1
            Case {
                in_size: (5, 5),
                kernel_size: (3, 3),
                dilations: (1, 1),
                strides: (2, 2),
                padding: zero_padding.clone(),
                expected: Ok((2, 2, [0, 0, 0, 0])),
            },
            // Dilations > 1
            Case {
                in_size: (5, 5),
                kernel_size: (3, 3),
                dilations: (2, 2),
                strides: (1, 1),
                padding: zero_padding.clone(),
                expected: Ok((1, 1, [0, 0, 0, 0])),
            },
            // `Same` padding, uneven
            Case {
                in_size: (1, 20),
                kernel_size: (1, 3),
                dilations: (1, 1),
                strides: (1, 1),
                padding: Padding::Same,
                expected: Ok((1, 20, [0, 1, 0, 1])),
            },
            // Strides > kernel size. This would cause underflow if the
            // clamping the padding to be >= 0.
            Case {
                in_size: (9, 9),
                dilations: (1, 1),
                strides: (3, 3),
                kernel_size: (2, 2),
                padding: Padding::Same,
                expected: Ok((3, 3, [0, 0, 0, 0])),
            },
            // Zero stride
            Case {
                in_size: (5, 5),
                dilations: (1, 1),
                strides: (0, 0),
                kernel_size: (3, 3),
                padding: Padding::Same,
                expected: Err(OpError::InvalidValue("Strides must be > 0")),
            },
            // Zero dilation
            Case {
                in_size: (5, 5),
                dilations: (0, 0),
                strides: (1, 1),
                kernel_size: (3, 3),
                padding: Padding::Same,
                expected: Err(OpError::InvalidValue("Dilations must be > 0")),
            },
            // Incorrect padding length
            Case {
                in_size: (5, 5),
                kernel_size: (3, 3),
                dilations: (1, 1),
                strides: (1, 1),
                padding: [0, 0].into(),
                expected: Err(OpError::InvalidValue("Expected 4 padding values")),
            },
            // Dilated kernel size > input size
            Case {
                in_size: (4, 4),
                kernel_size: (3, 3),
                dilations: (2, 2),
                strides: (1, 1),
                padding: zero_padding.clone(),
                expected: Err(OpError::InvalidValue("Input too small for kernel size")),
            },
        ];

        for Case {
            in_size,
            kernel_size,
            dilations,
            strides,
            padding,
            expected,
        } in cases
        {
            assert_eq!(
                calc_output_size_and_padding(
                    in_size,
                    kernel_size,
                    strides,
                    padding,
                    Some(dilations),
                ),
                expected
            );
        }
    }
}
