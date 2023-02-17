use std::borrow::Cow;

use crate::check_dims;
use crate::linalg::{add_scaled_vector, div_ceil, gemm};
use crate::ops::pooling::calc_output_size_and_padding;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output, Padding};
use crate::tensor::{AsMatrix, Tensor, TensorLayout, TensorView, TensorViewMut};

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
    pad_left: usize,
    stride: usize,
    out_w: usize,
) -> (usize, usize) {
    let min_out_x = pad_left.saturating_sub(k_x);
    let max_out_x = div_ceil((in_w + pad_left).saturating_sub(k_x), stride).min(out_w);
    (min_out_x, max_out_x)
}

/// Unroll patches from an image into a matrix.
///
/// `input` has shape [C,H,W]. `output` has shape [C * Kh * Kw, Oh * Ow] where
/// Kh/Kw are the patch sizes and Oh/Ow are the number of patches in the Y and
/// X directions.
fn im2col(
    output: &mut Tensor,
    input: &TensorView,
    patch_h: usize,
    patch_w: usize,
    padding: [usize; 4],
    strides: [usize; 2],
    out_hw: [usize; 2],
) {
    let [_, out_w] = output.dims();
    let [in_chans, in_h, in_w] = input.dims();
    let [pad_top, pad_left, _pad_bottom, _pad_right] = padding;
    let [stride_h, stride_w] = strides;
    let [y_patches, x_patches] = out_hw;

    for c in 0..in_chans {
        // The loop ordering here is chosen to maximize the number of
        // consecutive steps that we read/write the same rows of the inputs and
        // outputs. This is more efficient assuming the tensors are stored in
        // row-major order.
        for py in 0..y_patches {
            let out_col_left = py * x_patches;

            // Calculate range of kernel rows that will lead to valid input
            // row coordinates. For other rows zero padding is used, meaning
            // the output will be zero.
            let min_ky = pad_top.saturating_sub(py * stride_h);
            let max_ky = (in_h + pad_top).saturating_sub(py * stride_h).min(patch_h);

            for k_y in min_ky..max_ky {
                let img_y = py * stride_h + k_y;
                let out_row_top = c * patch_h * patch_w + k_y * patch_w;

                let in_row = input.slice(&[c.into(), (img_y - pad_top).into()]);

                for k_x in 0..patch_w {
                    let out_row = out_row_top + k_x;
                    let (min_px, max_px) =
                        min_max_out_x_coords(k_x, in_w, pad_left, stride_w, x_patches);
                    let out_row_data =
                        &mut output.last_dim_slice_mut([out_row, 0], out_w)[out_col_left..];

                    for px in min_px..max_px {
                        out_row_data[px] = in_row[[px * stride_w + k_x - pad_left]]
                    }
                }
            }
        }
    }
}

/// Initialize a tensor, typically with NCHW or CHW dimensions, with a vector
/// of per-channel biases.
///
/// This is slightly more efficient than creating a zero-filled tensor and then
/// adding the appropriate bias to each element.
fn init_tensor_with_channel_bias(shape: &[usize], chan_dim: usize, bias: &Tensor) -> Tensor {
    let mut out_data = Vec::with_capacity(shape.iter().product());

    let chan_elts: usize = shape[chan_dim + 1..].iter().product();
    let all_chan_elts: usize = chan_elts * shape[chan_dim];
    let repeats = shape[0..chan_dim].iter().product();

    for n in 0..repeats {
        for c in 0..shape[chan_dim] {
            out_data.resize(n * all_chan_elts + (c + 1) * chan_elts, bias[[c]]);
        }
    }

    Tensor::from_data(shape, out_data)
}

/// Specialization of conv_2d for pointwise convolutions over one image. This
/// can be reduced to tensor reshaping and matrix multiplication.
fn conv_2d_pointwise(input: &Tensor, kernel: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let [batch, _, in_h, in_w] = input.dims();
    let [out_c, in_c, _, _] = kernel.dims();

    let mut output = if let Some(bias) = bias {
        init_tensor_with_channel_bias(&[batch, out_c, in_h * in_w], 1, bias)
    } else {
        Tensor::zeros(&[batch, out_c, in_h * in_w])
    };

    // Get input and kernel as contiguous tensors so we can create reshaped
    // views.
    let input = input.as_contiguous();
    let kernel = kernel.as_contiguous();
    let kernel_mat = kernel.view().reshaped(&[out_c, in_c]).as_matrix();

    for n in 0..batch {
        let mut out_item = output.slice_mut(&[n.into()]);
        let out_row_stride = out_item.stride(0);

        let in_mat = input
            .slice(&[n.into()])
            .reshaped(&[in_c, in_h * in_w])
            .as_matrix();

        gemm(
            out_item.data_mut(),
            out_row_stride,
            kernel_mat,
            in_mat,
            1.,                                   // alpha
            if bias.is_some() { 1. } else { 0. }, // beta
        );
    }

    output.reshape(&[batch, out_c, in_h, in_w]);

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
    padding: [usize; 4],
    strides: [usize; 2],
    out_hw: [usize; 2],
) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let [out_c, _, k_h, k_w] = kernel.dims();
    let [pad_top, pad_left, _pad_bottom, _pad_right] = padding;
    let [stride_h, stride_w] = strides;
    let [out_h, out_w] = out_hw;

    let mut output = if let Some(bias) = bias {
        init_tensor_with_channel_bias(&[batch, out_c, out_h, out_w], 1, bias)
    } else {
        Tensor::zeros(&[batch, out_c, out_h, out_w])
    };

    // Use of `last_dim_slice` requires contiguous last dimension.
    let input: Cow<_> = if input.stride(input.ndim() - 1) == 1 {
        Cow::Borrowed(input)
    } else {
        input.as_contiguous()
    };

    for n in 0..batch {
        for c in 0..in_c {
            let kernel_view = kernel.nd_slice([c, 0]).unchecked();

            // The loops here are ordered so that the inner-most loop is as
            // efficient as possible and runs for as long as possible over a
            // contiguous slice of memory.
            for out_y in 0..out_h {
                let out_row = output.last_dim_slice_mut([n, c, out_y, 0], out_w);

                for k_y in 0..k_h {
                    let in_y = out_y * stride_h + k_y;
                    if in_y < pad_top || in_y >= in_h + pad_top {
                        continue;
                    }

                    let in_row = input.last_dim_slice([n, c, in_y - pad_top, 0], in_w);

                    for k_x in 0..k_w {
                        let kernel_val = kernel_view[[k_y, k_x]];
                        let (min_out_x, max_out_x) =
                            min_max_out_x_coords(k_x, in_w, pad_left, stride_w, out_w);

                        if min_out_x == max_out_x {
                            continue;
                        }

                        let out_row_slice = &mut out_row[min_out_x..max_out_x];
                        let in_row_slice = &in_row[min_out_x * stride_w + k_x - pad_left
                            ..(max_out_x - 1) * stride_w + k_x - pad_left + 1];

                        add_scaled_vector(
                            out_row_slice,
                            in_row_slice,
                            1,        /* dest_stride */
                            stride_w, /* src_stride */
                            kernel_val,
                        );
                    }
                }
            }
        }
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
pub fn conv(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    padding: Padding,
    groups: usize,
    strides: [usize; 2],
) -> Result<Tensor, OpError> {
    let [batch, in_c, in_h, in_w] = check_dims!(input, 4, "NCHW");
    let [out_c, k_in_c, k_h, k_w] = check_dims!(kernel, 4, "OCHW");
    check_dims!(bias?, 1);

    let [stride_h, stride_w] = strides;
    let (out_h, out_w, fixed_padding) =
        calc_output_size_and_padding((in_h, in_w), (k_h, k_w), (stride_h, stride_w), padding)?;

    let [pad_top, pad_left, pad_bottom, pad_right] = fixed_padding;

    let has_padding = pad_top > 0 || pad_left > 0 || pad_bottom > 0 || pad_right > 0;

    if k_h == 1 && k_w == 1 && !has_padding && groups == 1 {
        return Ok(conv_2d_pointwise(input, kernel, bias));
    }

    let out_channels_per_group = out_c / groups;
    let in_channels_per_group = in_c / groups;

    if in_channels_per_group != k_in_c {
        return Err(OpError::IncompatibleInputShapes(
            "Input channels (per group) does not match kernel input channels",
        ));
    }

    if groups == 0 || in_c % groups != 0 || out_c % groups != 0 {
        return Err(OpError::IncompatibleInputShapes(
            "Input channels and output channels must be divisible by group count",
        ));
    }

    if in_c == out_c && groups == in_c {
        return Ok(conv_2d_depthwise(
            input,
            kernel,
            bias,
            fixed_padding,
            strides,
            [out_h, out_w],
        ));
    }

    let n_patches = out_h * out_w;
    let mut output = if let Some(bias) = bias {
        init_tensor_with_channel_bias(&[batch, out_c, n_patches], 1, bias)
    } else {
        Tensor::zeros(&[batch, out_c, n_patches])
    };
    let mut im2col_mat = Tensor::zeros(&[in_channels_per_group * k_h * k_w, n_patches]);

    for n in 0..batch {
        for group in 0..groups {
            let in_chan_start = group * in_channels_per_group;
            let in_chan_end = in_chan_start + in_channels_per_group;
            let out_chan_start = group * out_channels_per_group;

            let in_group = input.slice(&[n.into(), (in_chan_start..in_chan_end).into()]);

            // Perform convolution for group. This uses an indirect method,
            // where image patches and the kernel are first packed into
            // matrices. The matrices are then multiplied with the results
            // written into the output tensor.
            im2col(
                &mut im2col_mat,
                &in_group,
                k_h,
                k_w,
                fixed_padding,
                strides,
                [out_h, out_w],
            );

            let kernel_mat = kernel
                .slice(&[(out_chan_start..out_chan_start + out_channels_per_group).into()])
                .reshaped(&[out_channels_per_group, in_channels_per_group * k_h * k_w])
                .as_matrix();

            let mut out_item = output.slice_mut(&[
                n.into(),
                (out_chan_start..out_chan_start + out_channels_per_group).into(),
            ]);
            let mut out_mat = out_item.reshaped(&[out_channels_per_group, out_h * out_w]);
            let out_row_stride = out_mat.stride(0);

            gemm(
                out_mat.data_mut(),
                out_row_stride,
                kernel_mat,
                im2col_mat.view().as_matrix(),
                1.,                                   // alpha
                if bias.is_some() { 1. } else { 0. }, // beta
            );
        }
    }

    output.reshape(&[batch, out_c, out_h, out_w]);

    Ok(output)
}

#[derive(Debug)]
pub struct Conv {
    pub padding: Padding,
    pub groups: usize,
    pub strides: [usize; 2],
}

impl Operator for Conv {
    fn name(&self) -> &str {
        "Conv"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        let weight = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;
        conv(input, weight, bias, self.padding, self.groups, self.strides).into_op_result()
    }
}

/// Unpack columns of a matrix into an image. This is the inverse of the
/// `im2col` operation.
///
/// `output` has shape [O,H,W] where O is the number of output channels and H/W
/// are the output height/width.
///
/// `columns` is a view of a matrix (Hi x Wi, O x Kh x Kw) reshaped to
/// [Hi,Wi,O,Kh,Kw], where Hi and Wi are the image size, and Kh/Kw are the patch
/// sizes. This matrix is passed as a view to avoid needing to pass the
/// sub-dimensions separately.
///
/// The unpacked columns are added to the existing output values to preserve
/// any bias stored in the output.
fn col2im(output: &mut TensorViewMut, columns: &TensorView, strides: [usize; 2]) {
    let [in_h, in_w, _out_chans, k_h, k_w] = columns.dims();
    let [out_chans, _, _] = output.dims();
    let [stride_h, stride_w] = strides;

    let col_view = columns.nd_view().unchecked();
    let mut out_view = output.nd_view_mut();
    let mut out_view = out_view.unchecked_mut();

    for y in 0..in_h {
        for x in 0..in_w {
            for out_c in 0..out_chans {
                for k_y in 0..k_h {
                    let out_y = y * stride_h + k_y;
                    for k_x in 0..k_w {
                        let out_x = x * stride_w + k_x;
                        out_view[[out_c, out_y, out_x]] += col_view[[y, x, out_c, k_y, k_x]];
                    }
                }
            }
        }
    }
}

/// Perform a transposed 2D convolution of a tensor by a kernel.
///
/// `input` has dimensions NCHW and `kernel` has dimensions COHW where `O` is
/// the number of output channels.
pub fn conv_transpose(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
) -> Result<Tensor, OpError> {
    let [batch, in_c, in_h, in_w] = check_dims!(input, 4, "NCHW");
    let [k_in_c, out_c, k_h, k_w] = check_dims!(kernel, 4, "OCHW");
    check_dims!(bias?, 1);

    if in_c != k_in_c {
        return Err(OpError::IncompatibleInputShapes(
            "Input channels does not match kernel input channels",
        ));
    }

    let [stride_h, stride_w] = strides;
    let out_h = (in_h - 1) * stride_h + k_h;
    let out_w = (in_w - 1) * stride_w + k_w;

    let mut output = if let Some(bias) = bias {
        init_tensor_with_channel_bias(&[batch, out_c, out_h, out_w], 1, bias)
    } else {
        Tensor::zeros(&[batch, out_c, out_h, out_w])
    };

    // Ensure input and kernel are contiguous to support reshaping.
    let input = input.as_contiguous();
    let kernel = kernel.as_contiguous();

    let mut col2im_mat = Tensor::zeros(&[in_h * in_w, out_c * k_h * k_w]);
    let kernel_mat = kernel.view().reshaped(&[k_in_c, out_c * k_h * k_w]);

    // The implementation here is the inverse of the im2col-based convolution.
    for n in 0..batch {
        let input_mat = input
            .slice(&[n.into()])
            .reshaped(&[in_c, in_h * in_w])
            .transposed();

        let col2im_row_stride = col2im_mat.stride(0);
        gemm(
            col2im_mat.data_mut(),
            col2im_row_stride,
            input_mat.as_matrix(),
            kernel_mat.as_matrix(),
            1., /* alpha */
            1., /* beta */
        );

        col2im(
            &mut output.slice_mut(&[n.into()]),
            &col2im_mat.view().reshaped(&[in_h, in_w, out_c, k_h, k_w]),
            strides,
        );
    }

    Ok(output)
}

#[derive(Debug)]
pub struct ConvTranspose {
    pub strides: [usize; 2],
}

impl Operator for ConvTranspose {
    fn name(&self) -> &str {
        "ConvTranspose"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        let weight = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;
        conv_transpose(input, weight, bias, self.strides).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::pooling::calc_output_size_and_padding;
    use crate::ops::{conv, conv_transpose, Conv, InputList, OpError, Operator, Padding};
    use crate::rng::XorShiftRng;
    use crate::tensor::{from_data, rand, zeros, Tensor, TensorLayout};
    use crate::test_util::expect_equal;

    /// Un-optimized reference implementation of convolution.
    fn reference_conv(
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        padding: [usize; 4],
        groups: usize,
        strides: [usize; 2],
    ) -> Tensor {
        let [batch, in_chans, in_h, in_w] = input.dims();
        let [out_chans, k_in_chans, k_h, k_w] = kernel.dims();
        let [stride_h, stride_w] = strides;
        let (out_h, out_w, _) = calc_output_size_and_padding(
            (in_h, in_w),
            (k_h, k_w),
            (stride_h, stride_w),
            Padding::Fixed(padding),
        )
        .expect("Input too small");
        let [pad_top, pad_left, _pad_bottom, _pad_right] = padding;

        let in_channels_per_group = in_chans / groups;
        let out_channels_per_group = out_chans / groups;
        assert_eq!(in_channels_per_group, k_in_chans);

        let mut output = zeros(&[batch, out_chans, out_h, out_w]);

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
                                        let in_y = out_y * stride_h + k_y;
                                        let in_x = out_x * stride_w + k_x;

                                        if in_y >= pad_top
                                            && in_y < in_h + pad_top
                                            && in_x >= pad_left
                                            && in_x < in_w + pad_left
                                        {
                                            accum += input
                                                [[n, in_chan, in_y - pad_top, in_x - pad_left]]
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

    /// Basic tests for conv. These compare the results against values
    /// computed from PyTorch as well as the reference implementation.
    #[test]
    fn test_conv() -> Result<(), String> {
        let kernel = from_data(
            &[1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );

        let input = from_data(
            &[1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let expected_with_same_padding = from_data(
            &[1, 1, 3, 3],
            vec![
                1.5202, 1.5592, 0.9939, 1.7475, 2.6358, 1.3428, 1.0165, 1.1806, 0.8685,
            ],
        );

        let result = conv(
            &input,
            &kernel,
            None,
            Padding::Fixed([1, 1, 1, 1]),
            1,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input,
            &kernel,
            None,
            [1, 1, 1, 1],
            1,      /* groups */
            [1, 1], /* stride */
        );
        expect_equal(&result, &expected_with_same_padding)?;
        expect_equal(&result, &reference_result)?;

        let expected_with_no_padding = from_data(&[1, 1, 1, 1], vec![2.6358]);

        let result = conv(
            &input,
            &kernel,
            None,
            Padding::Fixed([0, 0, 0, 0]),
            1,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input,
            &kernel,
            None,
            [0, 0, 0, 0],
            1,      /* groups */
            [1, 1], /* stride */
        );
        expect_equal(&result, &expected_with_no_padding)?;
        expect_equal(&result, &reference_result)?;

        let expected_with_bias = from_data(&[1, 1, 1, 1], vec![3.6358]);
        let bias = from_data(&[1], vec![1.0]);
        let result = conv(
            &input,
            &kernel,
            Some(&bias),
            Padding::Fixed([0, 0, 0, 0]),
            1,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input,
            &kernel,
            Some(&bias),
            [0, 0, 0, 0],
            1,      /* groups */
            [1, 1], /* stride */
        );
        expect_equal(&result, &expected_with_bias)?;
        expect_equal(&result, &reference_result)
    }

    #[test]
    fn test_conv_same_padding() -> Result<(), String> {
        let kernel = &from_data(
            &[1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );

        let input = &from_data(
            &[1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let op = Conv {
            padding: Padding::Same,
            groups: 1,
            strides: [1, 1],
        };
        let result = op
            .run(InputList::from(&[input.into(), kernel.into()]))
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        let reference_result = reference_conv(
            input,
            kernel,
            None,
            [1, 1, 1, 1],
            1,      /* groups */
            [1, 1], /* stride */
        );

        expect_equal(&result, &reference_result)
    }

    #[test]
    fn test_conv_uneven_padding() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = rand(&[10, 5, 3, 3], &mut rng);
        let input = rand(&[1, 5, 10, 10], &mut rng);
        let bias = rand(&[10], &mut rng);

        let result = conv(
            &input,
            &kernel,
            Some(&bias),
            Padding::Fixed([0, 0, 1, 1]),
            1,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input,
            &kernel,
            Some(&bias),
            [0, 0, 1, 1],
            1,      /* groups */
            [1, 1], /* stride */
        );

        expect_equal(&result, &reference_result)
    }

    #[test]
    fn test_conv_depthwise_uneven_padding() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = rand(&[10, 1, 3, 3], &mut rng);
        let input = rand(&[1, 10, 10, 10], &mut rng);
        let bias = rand(&[10], &mut rng);

        let result = conv(
            &input,
            &kernel,
            Some(&bias),
            Padding::Fixed([0, 0, 1, 1]),
            10,     /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input,
            &kernel,
            Some(&bias),
            [0, 0, 1, 1],
            10,     /* groups */
            [1, 1], /* stride */
        );

        expect_equal(&result, &reference_result)
    }

    // Specific tests for convolutions with a 1x1 kernel.
    #[test]
    fn test_conv_pointwise() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = rand(&[10, 5, 1, 1], &mut rng);
        let input = rand(&[1, 5, 20, 20], &mut rng);
        let bias = rand(&[10], &mut rng);

        // Contiguous inputs
        let result = conv(
            &input,
            &kernel,
            Some(&bias),
            Padding::Fixed([0, 0, 0, 0]),
            1,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input,
            &kernel,
            Some(&bias),
            [0, 0, 0, 0],
            1,      /* groups */
            [1, 1], /* stride */
        );

        assert_eq!(result.shape(), [1, 10, 20, 20]);
        expect_equal(&result, &reference_result)?;

        // Non-contiguous inputs
        let mut input_transposed = input.clone();
        input_transposed.permute(&[0, 1, 3, 2]);
        assert!(!input_transposed.is_contiguous());

        let result = conv(
            &input_transposed,
            &kernel,
            Some(&bias),
            Padding::Fixed([0, 0, 0, 0]),
            1,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input_transposed,
            &kernel,
            Some(&bias),
            [0, 0, 0, 0],
            1,      /* groups */
            [1, 1], /* stride */
        );
        assert_eq!(result.shape(), [1, 10, 20, 20]);
        expect_equal(&result, &reference_result)?;

        // Batch size > 1
        let input = rand(&[2, 5, 20, 20], &mut rng);
        let result = conv(
            &input,
            &kernel,
            Some(&bias),
            Padding::Fixed([0, 0, 0, 0]),
            1,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input,
            &kernel,
            Some(&bias),
            [0, 0, 0, 0],
            1,      /* groups */
            [1, 1], /* stride */
        );
        assert_eq!(result.shape(), [2, 10, 20, 20]);
        expect_equal(&result, &reference_result)?;

        Ok(())
    }

    // Specific tests for convolutions that operate over one output channel and
    // one input channel at a time.
    #[test]
    fn test_conv_depthwise() -> Result<(), String> {
        let input = from_data(
            &[1, 3, 2, 2],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 1.5202, 1.5592,
                0.9939, 1.7475,
            ],
        );
        let kernel = from_data(
            &[3, 1, 2, 2],
            vec![
                -0.0862, -0.4111, 0.0813, 0.4993, -0.4641, 0.1715, -0.0532, -0.2429, -0.4325,
                0.4273, 0.4180, 0.4338,
            ],
        );
        let bias = from_data(&[3], vec![0.1, 0.2, 0.3]);
        let expected = from_data(
            &[1, 3, 1, 1],
            vec![
                0.09020272 + bias[[0]],
                -0.09061745 + bias[[1]],
                1.1822754 + bias[[2]],
            ],
        );
        let reference_result = reference_conv(
            &input,
            &kernel,
            Some(&bias),
            [0, 0, 0, 0],
            3,      /* groups */
            [1, 1], /* stride */
        );

        let result = conv(
            &input,
            &kernel,
            Some(&bias),
            Padding::Fixed([0, 0, 0, 0]),
            3,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();

        expect_equal(&result, &expected)?;
        expect_equal(&result, &reference_result)
    }

    // Tests for convolutions that are neither pointwise nor depthwise. In
    // other words, the kernel has a spatial size > 1x1 and a channel depth > 1.
    #[test]
    fn test_conv_not_depthwise_or_pointwise() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = rand(&[4, 2, 3, 3], &mut rng);
        let input = rand(&[2, 4, 20, 20], &mut rng);
        let bias = rand(&[4], &mut rng);

        let result = conv(
            &input,
            &kernel,
            Some(&bias),
            Padding::Fixed([1, 1, 1, 1]),
            2,      /* groups */
            [1, 1], /* stride */
        )
        .unwrap();
        let reference_result = reference_conv(
            &input,
            &kernel,
            Some(&bias),
            [1, 1, 1, 1],
            2,      /* groups */
            [1, 1], /* stride */
        );

        expect_equal(&result, &reference_result)
    }

    #[test]
    fn test_conv_strided() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = rand(&[4, 3, 3, 3], &mut rng);

        for strides in [[2, 2], [3, 3], [1, 3]] {
            for pad in [0, 1] {
                for input_size in [3, 4, 5, 10, 20] {
                    let input = rand(&[2, 3, input_size, input_size], &mut rng);
                    let result = conv(
                        &input,
                        &kernel,
                        None,
                        Padding::Fixed([pad, pad, pad, pad]),
                        1, /* groups */
                        strides,
                    )
                    .unwrap();
                    let reference_result = reference_conv(
                        &input,
                        &kernel,
                        None,
                        [pad, pad, pad, pad],
                        1, /* groups */
                        strides,
                    );
                    expect_equal(&result, &reference_result)?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_conv_strided_depthwise() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = rand(&[3, 1, 3, 3], &mut rng);

        for strides in [[2, 2], [3, 3], [1, 3]] {
            for pad in [0, 1] {
                for input_size in [3, 4, 5, 10, 20] {
                    let input = rand(&[1, 3, input_size, input_size], &mut rng);
                    let result = conv(
                        &input,
                        &kernel,
                        None,
                        Padding::Fixed([pad, pad, pad, pad]),
                        3, /* groups */
                        strides,
                    )
                    .unwrap();
                    let reference_result = reference_conv(
                        &input,
                        &kernel,
                        None,
                        [pad, pad, pad, pad],
                        3, /* groups */
                        strides,
                    );
                    expect_equal(&result, &reference_result)?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_conv_input_too_small() {
        let mut rng = XorShiftRng::new(1234);
        let input = rand(&[1, 1, 2, 2], &mut rng);
        let kernel = rand(&[1, 1, 3, 3], &mut rng);

        let result = conv(
            &input,
            &kernel,
            None,
            Padding::Fixed([0; 4]),
            1,      /* groups */
            [1, 1], /* stride */
        );

        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Input too small for kernel size"))
        );
    }

    #[test]
    fn test_conv_zero_stride() {
        let mut rng = XorShiftRng::new(1234);
        let input = rand(&[1, 1, 2, 2], &mut rng);
        let kernel = rand(&[1, 1, 2, 2], &mut rng);

        let result = conv(
            &input,
            &kernel,
            None,
            Padding::Fixed([0; 4]),
            1,      /* groups */
            [0, 0], /* stride */
        );

        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Stride must be > 0"))
        );
    }

    #[test]
    fn test_conv_transpose() -> Result<(), String> {
        let input = from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = from_data(&[1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);
        let expected = from_data(
            &[1, 1, 4, 4],
            vec![
                0.1000, 0.2000, 0.2000, 0.4000, 0.3000, 0.4000, 0.6000, 0.8000, 0.3000, 0.6000,
                0.4000, 0.8000, 0.9000, 1.2000, 1.2000, 1.6000,
            ],
        );

        let result = conv_transpose(&input, &kernel, None, [2, 2]).unwrap();
        expect_equal(&result, &expected)?;

        let mut expected_with_bias = from_data(expected.shape().into(), expected.data().into());
        for i in 0..expected_with_bias.len() {
            expected_with_bias.data_mut()[i] += 1.234;
        }
        let bias = from_data(&[1], vec![1.234]);
        let result = conv_transpose(&input, &kernel, Some(&bias), [2, 2]).unwrap();
        expect_equal(&result, &expected_with_bias)
    }
}
