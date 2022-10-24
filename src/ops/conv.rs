use crate::linalg::{add_scaled_vector, div_ceil, gemm, gemm_slice};
use crate::ops::{Input, Operator};
use crate::tensor::{zero_tensor, Tensor};

/// Calculate the spatial size of a convolution output given the spatial
/// dimensions of the input, kernel and padding. All size tuples are (height,
/// width).
pub fn conv_output_size(
    in_size: (usize, usize),
    kernel_size: (usize, usize),
    padding: (usize, usize),
    stride: usize,
) -> (usize, usize) {
    let (in_h, in_w) = in_size;
    let (k_h, k_w) = kernel_size;
    let (pad_h, pad_w) = padding;

    let out_h = div_ceil(in_h + pad_h * 2 - k_h + 1, stride);
    let out_w = div_ceil(in_w + pad_w * 2 - k_w + 1, stride);

    (out_h, out_w)
}

// Calculate the min and max output X coordinates that are valid when updating
// a row of convolution output using a loop:
//
// ```
// for out_x in min_out_x..max_out_x {
//   out_row[out_x] += in_row[out_x * stride + k_x - pad_w] * kernel_element
// }
// ```
//
// Where `k_x` is the X coordinate of `kernel_element` and `in_row` is the
// un-padded input row.
fn min_max_out_x_coords(
    k_x: usize,
    in_w: usize,
    pad_w: usize,
    stride: usize,
    out_w: usize,
) -> (usize, usize) {
    let min_out_x = pad_w.saturating_sub(k_x);
    let max_out_x = div_ceil((in_w + pad_w).saturating_sub(k_x), stride).min(out_w);
    (min_out_x, max_out_x)
}

/// Unroll patches from an image into a matrix.
///
/// The input has shape NCHW. The result has shape (GHW)xP where G is the subset
/// of image channels from `start_chan` to `end_chan` and P is the number of
/// patches that the padded input divides into.
fn im2col(
    output: &mut Tensor,
    input: &Tensor,
    image_index: usize,
    patch_h: usize,
    patch_w: usize,
    start_chan: usize,
    end_chan: usize,
    pad_h: usize,
    pad_w: usize,
    stride: usize,
) {
    let [_, out_w] = output.dims();
    let [_, _, in_h, in_w] = input.dims();

    let (y_patches, x_patches) =
        conv_output_size((in_h, in_w), (patch_h, patch_w), (pad_h, pad_w), stride);
    let n_chans = end_chan - start_chan;

    for c in 0..n_chans {
        // The loop ordering here is chosen to maximize the number of
        // consecutive steps that we read/write the same rows of the inputs and
        // outputs. This is more efficient assuming the tensors are stored in
        // row-major order.
        for py in 0..y_patches {
            let out_col_left = py * x_patches;

            // Calculate range of kernel rows that will lead to valid input
            // row coordinates. For other rows zero padding is used, meaning
            // the output will be zero.
            let min_ky = pad_h.saturating_sub(py * stride);
            let max_ky = (in_h + pad_h).saturating_sub(py * stride).min(patch_h);

            for k_y in min_ky..max_ky {
                let img_y = py * stride + k_y;
                let out_row_top = c * patch_h * patch_w + k_y * patch_w;
                let in_row =
                    input.last_dim_slice([image_index, start_chan + c, img_y - pad_h, 0], in_w);

                for k_x in 0..patch_w {
                    let out_row = out_row_top + k_x;
                    let (min_px, max_px) =
                        min_max_out_x_coords(k_x, in_w, pad_w, stride, x_patches);
                    let out_row_data =
                        &mut output.last_dim_slice_mut([out_row, 0], out_w)[out_col_left..];

                    for px in min_px..max_px {
                        out_row_data[px] = in_row[px * stride + k_x - pad_w]
                    }
                }
            }
        }
    }
}

/// Unroll a subset of channels from a convolution kernel into a matrix of shape
/// O x (CHW) where O is the number of output channels and C is the number of
/// input channels.
fn unroll_kernel(output: &mut Tensor, kernel: &Tensor, out_chan_start: usize, out_chan_end: usize) {
    let [_, k_in_c, k_h, k_w] = kernel.dims();
    for out_c in out_chan_start..out_chan_end {
        for in_c in 0..k_in_c {
            for y in 0..k_h {
                for x in 0..k_w {
                    let out_row = out_c - out_chan_start;
                    let out_col = in_c * (k_h * k_w) + y * k_w + x;
                    output[[out_row, out_col]] = kernel[[out_c, in_c, y, x]];
                }
            }
        }
    }
}

/// Convert a matrix of image patches back into an image.
///
/// The output tensor has shape NCHW. The input image has shape GP where G is a
/// subset of channels C given by `start_chan..start_chan+G` and P is the number
/// of patches.
fn col2im(
    output: &mut Tensor,
    input: &Tensor,
    image_index: usize,
    start_chan: usize,
    y_patches: usize,
    x_patches: usize,
) {
    let [group_chans, n_patches] = input.dims();

    for c in 0..group_chans {
        let mut out_view = output.unchecked_view_mut([image_index, start_chan + c, 0, 0]);
        let in_row = input.last_dim_slice([c, 0], n_patches);

        for y in 0..y_patches {
            for x in 0..x_patches {
                let patch = y * x_patches + x;
                out_view[[y, x]] = in_row[patch];
            }
        }
    }
}

/// Add per-channel biases to an NCHW tensor. Bias is a C-length vector.
fn add_channel_bias(output: &mut Tensor, bias: &Tensor) {
    let [batch, chans, height, width] = output.dims();

    for n in 0..batch {
        for c in 0..chans {
            let mut out_view = output.unchecked_view_mut([n, c, 0, 0]);
            let chan_bias = bias[[c]];

            for y in 0..height {
                for x in 0..width {
                    out_view[[y, x]] += chan_bias;
                }
            }
        }
    }
}

/// Specialization of conv_2d for pointwise convolutions over one image. This
/// can be reduced to tensor reshaping and matrix multiplication.
fn conv_2d_pointwise(input: &Tensor, kernel: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let [_, _, in_h, in_w] = input.dims();
    let [out_c, in_c, _, _] = kernel.dims();

    let mut output = zero_tensor(&[out_c, in_h * in_w]);
    let out_row_stride = output.stride(0);

    // Use the low-level gemm_slice API to simplicitly reshape the input and
    // kernel to `in_c x in_h*in_w` and `out_c x in_c` matrices respectively.
    //
    // If this package supported creating reshaped views of existing tensors,
    // we could use that instead.
    gemm_slice(
        output.data_mut(),
        out_row_stride,
        kernel.data(),
        out_c, /* a rows */
        in_c,  /* a columns */
        in_c,  /* a row stride */
        input.data(),
        in_c,        /* b rows */
        in_h * in_w, /* b columns */
        in_h * in_w, /* b row stride */
    );

    output.reshape(&[1, out_c, in_h, in_w]);

    if let Some(bias) = bias {
        add_channel_bias(&mut output, bias);
    }

    output
}

/// Specialization of conv_2d for depthwise convolutions.
///
/// Depthwise convolutions operate over a single input/output channel at
/// a time and hence the transformation of convolution to matrix multiplication
/// doesn't pay off. An optimized direct method works better.
fn conv_2d_depthwise(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    padding: (usize, usize),
    stride: usize,
) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let [out_c, _, k_h, k_w] = kernel.dims();
    let (pad_h, pad_w) = padding;
    let (out_h, out_w) = conv_output_size((in_h, in_w), (k_h, k_w), (pad_h, pad_w), stride);

    let mut output = zero_tensor::<f32>(&[batch, out_c, out_h, out_w]);

    for n in 0..batch {
        for c in 0..in_c {
            let kernel_view = kernel.unchecked_view([c, 0, 0, 0]);

            // The loops here are ordered so that the inner-most loop is as
            // efficient as possible and runs for as long as possible over a
            // contiguous slice of memory.
            for out_y in 0..out_h {
                let out_row = output.last_dim_slice_mut([n, c, out_y, 0], out_w);

                for k_y in 0..k_h {
                    let in_y = out_y * stride + k_y;
                    if in_y < pad_h || in_y >= in_h + pad_h {
                        continue;
                    }
                    let in_row = input.last_dim_slice([n, c, in_y - pad_h, 0], in_w);

                    for k_x in 0..k_w {
                        let kernel_val = kernel_view[[k_y, k_x]];
                        let (min_out_x, max_out_x) =
                            min_max_out_x_coords(k_x, in_w, pad_w, stride, out_w);

                        if min_out_x == max_out_x {
                            continue;
                        }

                        let out_row_slice = &mut out_row[min_out_x..max_out_x];
                        let in_row_slice = &in_row[min_out_x * stride + k_x - pad_w
                            ..(max_out_x - 1) * stride + k_x - pad_w + 1];

                        add_scaled_vector(
                            out_row_slice,
                            in_row_slice,
                            1,      /* dest_stride */
                            stride, /* src_stride */
                            kernel_val,
                        );
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        add_channel_bias(&mut output, bias);
    }

    output
}

/// Perform a 2D convolution of `input` with `kernel`.
///
/// `input` has dimensions NCHW while `kernel` has OGHW where `G` is `C / groups`.
///
/// - `padding` specifies the amount of horizontal and vertical padding respectively
///   that is added to each side.
/// - `groups` controls which input and output channels are convolved. It must
///   be a positive integer that divides the input and output channel count.
///   A value of 1 convolves every input channel with every output channel.
///   A value of 2 convolves each half of the input channels with the corresponding
///   half of the output channels.
///   A value equal to the input channel count convolves each input channel
///   separately with `output_channels / groups` outputs. This is known as
///   depthwise convolution.
pub fn conv_2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    padding: (usize, usize),
    groups: usize,
    stride: usize,
) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let [out_c, k_in_c, k_h, k_w] = kernel.dims();
    let (pad_h, pad_w) = padding;

    if batch == 1 && k_h == 1 && k_w == 1 && pad_h == 0 && pad_w == 0 && groups == 1 {
        return conv_2d_pointwise(input, kernel, bias);
    }

    let out_channels_per_group = out_c / groups;
    let in_channels_per_group = in_c / groups;

    if in_channels_per_group != k_in_c {
        panic!(
            "Input channels (per group) {} does not match kernel input channels {}",
            in_channels_per_group, k_in_c
        );
    }

    if groups == 0 || in_c % groups != 0 || out_c % groups != 0 {
        panic!(
            "Input channels {} and output channels {} must be divisible by group count {}",
            in_c, out_c, groups
        );
    }

    if in_c == out_c && groups == in_c {
        return conv_2d_depthwise(input, kernel, bias, padding, stride);
    }

    let (out_h, out_w) = conv_output_size((in_h, in_w), (k_h, k_w), (pad_h, pad_w), stride);

    let n_patches = out_h * out_w;
    let mut im2col_mat = zero_tensor(&[in_channels_per_group * k_h * k_w, n_patches]);
    let mut output = zero_tensor(&[batch, out_c, out_h, out_w]);
    let mut kernel_mat = zero_tensor(&[out_channels_per_group, in_channels_per_group * k_h * k_w]);
    let mut output_mat = zero_tensor(&[out_channels_per_group, n_patches]);

    for n in 0..batch {
        for group in 0..groups {
            let in_chan_start = group * in_channels_per_group;
            let in_chan_end = in_chan_start + in_channels_per_group;
            let out_chan_start = group * out_channels_per_group;
            let out_chan_end = out_chan_start + out_channels_per_group;

            // Perform convolution for group. This uses an indirect method,
            // where image patches and the kernel are first packed into
            // matrices. The matrices are multiplied, and the results unpacked
            // into the output tensor.
            im2col(
                &mut im2col_mat,
                input,
                n,
                k_h,
                k_w,
                in_chan_start,
                in_chan_end,
                pad_h,
                pad_w,
                stride,
            );
            unroll_kernel(&mut kernel_mat, kernel, out_chan_start, out_chan_end);
            gemm(&mut output_mat, &kernel_mat, &im2col_mat);
            col2im(&mut output, &output_mat, n, out_chan_start, out_h, out_w);

            // `gemm` currently accumulates into the output buffer, so we need
            // to clear it between iterations.
            output_mat.data_mut().fill(0.0);
        }
    }

    if let Some(bias) = bias {
        add_channel_bias(&mut output, bias);
    }

    output
}

#[derive(Copy, Clone, Debug)]
pub enum Padding {
    /// Apply enough padding such that the output and input have the same size.
    Same,

    /// Apply an even amount of padding to the start and end of the height and
    /// width dimensions respectively.
    Fixed((usize, usize)),
}

/// Calculate the specific amount of padding required for an operation that
/// will receive an NCHW input tensor and apply a kernel of a given size.
///
/// The kernel size must be >= the input size.
fn calc_fixed_padding(
    pad: Padding,
    input_shape: &[usize],
    kernel_size: (usize, usize),
) -> (usize, usize) {
    match pad {
        Padding::Fixed(pads) => pads,
        Padding::Same => {
            let [_, _, in_h, in_w]: [usize; 4] = input_shape.try_into().unwrap();
            let (k_h, k_w) = kernel_size;

            let unpadded_h = (in_h - k_h) + 1;
            let unpadded_w = (in_w - k_w) + 1;

            let pad_h = (in_h - unpadded_h) / 2;
            let pad_w = (in_w - unpadded_w) / 2;

            (pad_h, pad_w)
        }
    }
}

#[derive(Debug)]
pub struct Conv2d {
    pub padding: Padding,
    pub groups: usize,
    pub stride: usize,
}

impl Operator for Conv2d {
    fn name(&self) -> &str {
        "Conv2d"
    }

    /// Run `conv_2d` operator with `[input, weight, bias?]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        let weight = inputs[1].as_float().unwrap();
        let bias = inputs.get(2).map(|t| t.as_float().unwrap());

        let [_, _, k_h, k_w] = weight.dims();

        conv_2d(
            input,
            weight,
            bias,
            calc_fixed_padding(self.padding, input.shape(), (k_h, k_w)),
            self.groups,
            self.stride,
        )
    }
}

/// Perform a transposed 2D convolution of a tensor by a kernel.
///
/// `input` has dimensions NCHW and kernel has dimensions COHW where `O` is
/// the number of output channels.
pub fn conv_transpose_2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let [k_in_c, out_c, k_h, k_w] = kernel.dims();

    if in_c != k_in_c {
        panic!(
            "Input channels {} does not match kernel input channels {}",
            in_c, k_in_c
        )
    }

    let out_h = (in_h - 1) * stride + k_h;
    let out_w = (in_w - 1) * stride + k_w;

    let mut output = zero_tensor::<f32>(&[batch, out_c, out_h, out_w]);

    for n in 0..batch {
        for out_chan in 0..out_c {
            for in_chan in 0..in_c {
                let kernel_view = kernel.unchecked_view([in_chan, out_chan, 0, 0]);

                for in_y in 0..in_h {
                    let in_row = input.last_dim_slice([n, in_chan, in_y, 0], in_w);

                    for k_y in 0..k_h {
                        let out_y = in_y * stride + k_y;
                        let out_row = output.last_dim_slice_mut([n, out_chan, out_y, 0], out_w);

                        for k_x in 0..k_w {
                            add_scaled_vector(
                                &mut out_row[k_x..out_w - k_w + k_x + 1],
                                in_row,
                                stride,
                                1, // src_stride
                                kernel_view[[k_y, k_x]],
                            );
                        }
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        add_channel_bias(&mut output, bias);
    }

    output
}

#[derive(Debug)]
pub struct ConvTranspose2d {
    pub stride: usize,
}

impl Operator for ConvTranspose2d {
    fn name(&self) -> &str {
        "ConvTranspose2d"
    }

    /// Run `conv_2d` operator with `[input, weight]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        let weight = inputs[1].as_float().unwrap();
        let bias = inputs.get(2).map(|t| t.as_float().unwrap());
        conv_transpose_2d(input, weight, bias, self.stride)
    }
}

#[cfg(test)]
mod tests {
    use super::conv_output_size;
    use crate::ops::{conv_2d, conv_transpose_2d, Conv2d, Operator, Padding};
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, random_tensor, zero_tensor, Tensor};
    use crate::test_util::expect_equal;

    /// Un-optimized reference implementation of convolution.
    fn reference_conv(
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        padding: (usize, usize),
        groups: usize,
        stride: usize,
    ) -> Tensor {
        let [batch, in_chans, in_h, in_w] = input.dims();
        let [out_chans, k_in_chans, k_h, k_w] = kernel.dims();
        let (pad_h, pad_w) = padding;

        let in_channels_per_group = in_chans / groups;
        let out_channels_per_group = out_chans / groups;
        assert_eq!(in_channels_per_group, k_in_chans);

        let (out_h, out_w) = conv_output_size((in_h, in_w), (k_h, k_w), (pad_h, pad_w), stride);
        let mut output = zero_tensor(&[batch, out_chans, out_h, out_w]);

        for n in 0..batch {
            for group in 0..groups {
                let in_chan_start = group * in_channels_per_group;
                let in_chan_end = in_chan_start + in_channels_per_group;
                let out_chan_start = group * out_channels_per_group;
                let out_chan_end = out_chan_start + out_channels_per_group;

                for out_chan in out_chan_start..out_chan_end {
                    let chan_bias = if let Some(bias) = bias {
                        bias[[out_chan]]
                    } else {
                        0.0
                    };
                    for out_y in 0..out_h {
                        for out_x in 0..out_w {
                            let mut accum = 0.0;
                            for in_chan in in_chan_start..in_chan_end {
                                for k_y in 0..k_h {
                                    for k_x in 0..k_w {
                                        let in_y = out_y * stride + k_y;
                                        let in_x = out_x * stride + k_x;

                                        if in_y >= pad_h
                                            && in_y < in_h + pad_h
                                            && in_x >= pad_w
                                            && in_x < in_w + pad_w
                                        {
                                            accum += input
                                                [[n, in_chan, in_y - pad_h, in_x - pad_w]]
                                                * kernel
                                                    [[out_chan, in_chan - in_chan_start, k_y, k_x]];
                                        }
                                    }
                                }
                            }
                            output[[n, out_chan, out_y, out_x]] = accum + chan_bias;
                        }
                    }
                }
            }
        }

        output
    }

    /// Basic tests for conv_2d. These compare the results against values
    /// computed from PyTorch as well as the reference implementation.
    #[test]
    fn test_conv_2d() -> Result<(), String> {
        let kernel = from_data(
            vec![1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );

        let input = from_data(
            vec![1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let expected_with_same_padding = from_data(
            vec![1, 1, 3, 3],
            vec![
                1.5202, 1.5592, 0.9939, 1.7475, 2.6358, 1.3428, 1.0165, 1.1806, 0.8685,
            ],
        );

        let result = conv_2d(
            &input,
            &kernel,
            None,
            (1, 1),
            1, /* groups */
            1, /* stride */
        );
        let reference_result = reference_conv(
            &input,
            &kernel,
            None,
            (1, 1),
            1, /* groups */
            1, /* stride */
        );
        expect_equal(&result, &expected_with_same_padding)?;
        expect_equal(&result, &reference_result)?;

        let expected_with_no_padding = from_data(vec![1, 1, 1, 1], vec![2.6358]);

        let result = conv_2d(
            &input,
            &kernel,
            None,
            (0, 0),
            1, /* groups */
            1, /* stride */
        );
        let reference_result = reference_conv(
            &input,
            &kernel,
            None,
            (0, 0),
            1, /* groups */
            1, /* stride */
        );
        expect_equal(&result, &expected_with_no_padding)?;
        expect_equal(&result, &reference_result)?;

        let expected_with_bias = from_data(vec![1, 1, 1, 1], vec![3.6358]);
        let bias = from_data(vec![1], vec![1.0]);
        let result = conv_2d(
            &input,
            &kernel,
            Some(&bias),
            (0, 0),
            1, /* groups */
            1, /* stride */
        );
        let reference_result = reference_conv(
            &input,
            &kernel,
            Some(&bias),
            (0, 0),
            1, /* groups */
            1, /* stride */
        );
        expect_equal(&result, &expected_with_bias)?;
        expect_equal(&result, &reference_result)
    }

    #[test]
    fn test_conv_2d_same_padding() -> Result<(), String> {
        let kernel = &from_data(
            vec![1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );

        let input = &from_data(
            vec![1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let op = Conv2d {
            padding: Padding::Same,
            groups: 1,
            stride: 1,
        };
        let result = op.run(&[input.into(), kernel.into()]);
        let reference_result = reference_conv(
            input,
            kernel,
            None,
            (1, 1),
            1, /* groups */
            1, /* stride */
        );

        expect_equal(&result, &reference_result)
    }

    // Specific tests for convolutions with a 1x1 kernel.
    #[test]
    fn test_conv_2d_pointwise() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let kernel = random_tensor(&[10, 5, 1, 1], &mut rng);
        let input = random_tensor(&[1, 5, 20, 20], &mut rng);

        let result = conv_2d(
            &input,
            &kernel,
            None,
            (0, 0),
            1, /* groups */
            1, /* stride */
        );
        let reference_result = reference_conv(
            &input,
            &kernel,
            None,
            (0, 0),
            1, /* groups */
            1, /* stride */
        );

        assert_eq!(result.shape(), [1, 10, 20, 20]);
        expect_equal(&result, &reference_result)
    }

    // Specific tests for convolutions that operate over one output channel and
    // one input channel at a time.
    #[test]
    fn test_conv_2d_depthwise() -> Result<(), String> {
        let input = from_data(
            vec![1, 3, 2, 2],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 1.5202, 1.5592,
                0.9939, 1.7475,
            ],
        );
        let kernel = from_data(
            vec![3, 1, 2, 2],
            vec![
                -0.0862, -0.4111, 0.0813, 0.4993, -0.4641, 0.1715, -0.0532, -0.2429, -0.4325,
                0.4273, 0.4180, 0.4338,
            ],
        );
        let expected = from_data(vec![1, 3, 1, 1], vec![0.09020272, -0.09061745, 1.1822754]);
        let reference_result = reference_conv(
            &input,
            &kernel,
            None,
            (0, 0),
            3, /* groups */
            1, /* stride */
        );

        let result = conv_2d(
            &input,
            &kernel,
            None,
            (0, 0),
            3, /* groups */
            1, /* stride */
        );

        expect_equal(&result, &expected)?;
        expect_equal(&result, &reference_result)
    }

    // Tests for convolutions that are neither pointwise nor depthwise. In
    // other words, the kernel has a spatial size > 1x1 and a channel depth > 1.
    #[test]
    fn test_conv_2d_not_depthwise_or_pointwise() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let kernel = random_tensor(&[4, 3, 3, 3], &mut rng);
        let input = random_tensor(&[2, 3, 20, 20], &mut rng);

        let result = conv_2d(
            &input,
            &kernel,
            None,
            (1, 1),
            1, /* groups */
            1, /* stride */
        );
        let reference_result = reference_conv(
            &input,
            &kernel,
            None,
            (1, 1),
            1, /* groups */
            1, /* stride */
        );

        expect_equal(&result, &reference_result)
    }

    #[test]
    fn test_conv_2d_strided() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let kernel = random_tensor(&[4, 3, 3, 3], &mut rng);

        for stride in [2, 3] {
            for pad in [0, 1] {
                for input_size in [3, 4, 5, 10, 20] {
                    let input = random_tensor(&[2, 3, input_size, input_size], &mut rng);
                    let result = conv_2d(
                        &input,
                        &kernel,
                        None,
                        (pad, pad),
                        1,      /* groups */
                        stride, /* stride */
                    );
                    let reference_result = reference_conv(
                        &input,
                        &kernel,
                        None,
                        (pad, pad),
                        1,      /* groups */
                        stride, /* stride */
                    );
                    expect_equal(&result, &reference_result)?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_conv_2d_strided_depthwise() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let kernel = random_tensor(&[3, 1, 3, 3], &mut rng);

        for stride in [2, 3] {
            for pad in [0, 1] {
                for input_size in [3, 4, 5, 10, 20] {
                    let input = random_tensor(&[1, 3, input_size, input_size], &mut rng);
                    let result = conv_2d(
                        &input,
                        &kernel,
                        None,
                        (pad, pad),
                        3,      /* groups */
                        stride, /* stride */
                    );
                    let reference_result = reference_conv(
                        &input,
                        &kernel,
                        None,
                        (pad, pad),
                        3,      /* groups */
                        stride, /* stride */
                    );
                    expect_equal(&result, &reference_result)?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_conv_transpose_2d() -> Result<(), String> {
        let input = from_data(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = from_data(vec![1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);
        let expected = from_data(
            vec![1, 1, 4, 4],
            vec![
                0.1000, 0.2000, 0.2000, 0.4000, 0.3000, 0.4000, 0.6000, 0.8000, 0.3000, 0.6000,
                0.4000, 0.8000, 0.9000, 1.2000, 1.2000, 1.6000,
            ],
        );

        let result = conv_transpose_2d(&input, &kernel, None, 2);
        expect_equal(&result, &expected)?;

        let mut expected_with_bias = from_data(expected.shape().into(), expected.data().into());
        for i in 0..expected_with_bias.len() {
            expected_with_bias.data_mut()[i] += 1.234;
        }
        let bias = from_data(vec![1], vec![1.234]);
        let result = conv_transpose_2d(&input, &kernel, Some(&bias), 2);
        expect_equal(&result, &expected_with_bias)
    }
}
