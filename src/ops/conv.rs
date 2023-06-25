use std::borrow::Cow;
use std::iter::zip;
use std::ops::Range;

use rayon::prelude::*;

use wasnn_tensor::{NdTensorLayout, NdTensorView, Tensor, TensorLayout, TensorView, TensorViewMut};

use crate::check_dims;
use crate::linalg::{
    add_scaled_vector, div_ceil, gemm, round_up, GemmExecutor, GemmInputA, GemmInputB,
    VirtualMatrix,
};
use crate::ops::pooling::calc_output_size_and_padding;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output, Padding};

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

/// Unrolls patches of an image as columns of a virtual matrix.
///
/// The input image has shape [C,H,W] and is transformed into a matrix with
/// shape [C * Kh * kW, Oh * Ow] where Kh/Kw are convolution kernel sizes and
/// Oh/Ow are the number of patches in the Y and X directions.
///
/// The transform is virtual because the matrix is not actually materialized
/// in memory. Instead blocks of it are produced on-demand during a matrix
/// multiplication operation.
struct VirtualIm2Col<'a> {
    image: NdTensorView<'a, f32, 3>,
    kernel: [usize; 2],
    padding: [usize; 4],
    strides: [usize; 2],
    y_patches: usize,
    x_patches: usize,
}

impl<'a> VirtualIm2Col<'a> {
    fn new(
        image: NdTensorView<'a, f32, 3>,
        kernel: [usize; 2],
        padding: [usize; 4],
        strides: [usize; 2],
    ) -> VirtualIm2Col {
        let [_, h, w] = image.shape();
        let [k_h, k_w] = kernel;
        let [stride_h, stride_w] = strides;
        let (y_patches, x_patches, _) = calc_output_size_and_padding(
            (h, w),
            (k_h, k_w),
            (stride_h, stride_w),
            Padding::Fixed(padding),
        )
        .unwrap();

        VirtualIm2Col {
            image,
            kernel,
            padding,
            strides,
            y_patches,
            x_patches,
        }
    }
}

impl<'a> VirtualMatrix for VirtualIm2Col<'a> {
    fn rows(&self) -> usize {
        let [chans, _h, _w] = self.image.shape();
        let [k_h, k_w] = self.kernel;
        chans * k_h * k_w
    }

    fn cols(&self) -> usize {
        self.y_patches * self.x_patches
    }

    fn pack_b(&self, out: &mut [f32], panel_width: usize, rows: Range<usize>, cols: Range<usize>) {
        let [k_h, k_w] = self.kernel;
        let [stride_h, stride_w] = self.strides;
        let [pad_top, pad_left, _pad_bottom, _pad_right] = self.padding;

        // Build lookup table of column index in the virtual im2col matrix to
        // patch coordinate in the input image.
        let patch_coords: Vec<[i32; 2]> = (cols.start..round_up(cols.end, panel_width))
            .map(|col| {
                let patch_y = col as i32 / self.x_patches as i32;
                let patch_x = col as i32 % self.x_patches as i32;
                let img_x = (patch_x * stride_w as i32) - pad_left as i32;
                let img_y = (patch_y * stride_h as i32) - pad_top as i32;
                [img_y, img_x]
            })
            .collect();

        // Build lookup table of row index in the virtual im2col matrix to input
        // channel and kernel coordinates.
        let kernel_coords: Vec<[i32; 3]> = rows
            .map(|row| {
                let in_chan = row as i32 / (k_h * k_w) as i32;
                let kernel_element = row as i32 % (k_h * k_w) as i32;
                let k_y = kernel_element / k_w as i32;
                let k_x = kernel_element % k_w as i32;
                [in_chan, k_y, k_x]
            })
            .collect();

        // Loop over the output by column panel, then row, then element.
        let mut out_rows = out.chunks_exact_mut(panel_width);
        for panel_patch_coords in patch_coords.chunks_exact(panel_width) {
            for [in_chan, k_y, k_x] in kernel_coords.iter().copied() {
                let out_row = out_rows.next().unwrap();
                for ([img_y, img_x], out_el) in zip(panel_patch_coords.iter(), out_row.iter_mut()) {
                    let in_y = img_y + k_y;
                    let in_x = img_x + k_x;

                    // `in_y` or `in_x` may be negative here, in which case it will
                    // wrap around and `image.get` will still return None.
                    *out_el = self
                        .image
                        .get([in_chan as usize, in_y as usize, in_x as usize])
                        .copied()
                        .unwrap_or(0.);
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
    let mut output = Tensor::zeros(&[batch, out_c, in_h * in_w]);

    // Get input and kernel as contiguous tensors so we can create reshaped
    // views.
    let input = input.as_contiguous();
    let kernel = kernel.as_contiguous();
    let kernel_mat = kernel.view().reshaped(&[out_c, in_c]).to_nd_view();

    // Bias must be contiguous for use with `gemm_bias`.
    let bias = bias.map(|b| b.as_contiguous());

    let gemm = GemmExecutor::new();

    for n in 0..batch {
        let mut out_item = output.slice_mut([n]);
        let out_row_stride = out_item.stride(0);

        let in_mat = input.slice([n]).reshaped(&[in_c, in_h * in_w]).to_nd_view();

        gemm.gemm_bias(
            out_item.data_mut(),
            out_row_stride,
            GemmInputA::Unpacked(kernel_mat),
            GemmInputB::Unpacked(in_mat),
            1., // alpha
            0., // beta
            bias.as_ref().map(|b| b.data()),
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

    // Use of input rows below assumes contiguous last dimension.
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
                let mut out_row = output.nd_slice_mut::<3, 1>([n, c, out_y]);
                let out_row = out_row.data_mut();

                for k_y in 0..k_h {
                    let in_y = out_y * stride_h + k_y;
                    if in_y < pad_top || in_y >= in_h + pad_top {
                        continue;
                    }

                    let in_row = input.nd_slice::<3, 1>([n, c, in_y - pad_top]).to_data();

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
    let mut output = Tensor::zeros(&[batch, out_c, n_patches]);
    let gemm = GemmExecutor::new();

    // Bias must be contiguous for use with `gemm_bias`.
    let bias = bias.map(|b| b.as_contiguous());

    for group in 0..groups {
        let in_chan_start = group * in_channels_per_group;
        let in_chan_end = in_chan_start + in_channels_per_group;
        let out_chan_start = group * out_channels_per_group;
        let out_chans = out_chan_start..out_chan_start + out_channels_per_group;

        let kernel_mat = kernel
            .slice([out_chans.clone()])
            .reshaped(&[out_channels_per_group, in_channels_per_group * k_h * k_w])
            .to_nd_view();
        let prepacked_kernel = gemm.prepack_a(kernel_mat);

        let in_group = input.slice((.., in_chan_start..in_chan_end));
        let mut out_group = output.slice_mut((.., out_chans.clone()));

        zip(out_group.axis_iter_mut(0), in_group.axis_iter(0))
            .par_bridge()
            .for_each(|(mut out_item, in_item)| {
                let mut out_mat = out_item.reshaped(&[out_channels_per_group, out_h * out_w]);
                let out_row_stride = out_mat.stride(0);

                let im2col =
                    VirtualIm2Col::new(in_item.nd_view(), [k_h, k_w], fixed_padding, strides);

                gemm.gemm_bias(
                    out_mat.data_mut(),
                    out_row_stride,
                    GemmInputA::Packed(&prepacked_kernel),
                    GemmInputB::Virtual(&im2col),
                    1., // alpha
                    0., // beta
                    bias.as_ref().map(|b| &b.data()[out_chans.clone()]),
                );
            });
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
        let input_mat = input.slice([n]).reshaped(&[in_c, in_h * in_w]).transposed();

        let col2im_row_stride = col2im_mat.stride(0);
        gemm(
            col2im_mat.data_mut(),
            col2im_row_stride,
            input_mat.nd_view(),
            kernel_mat.nd_view(),
            1., /* alpha */
            1., /* beta */
        );

        col2im(
            &mut output.slice_mut([n]),
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
    use wasnn_tensor::rng::XorShiftRng;
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{Tensor, TensorLayout};

    use crate::ops::pooling::calc_output_size_and_padding;
    use crate::ops::{conv, conv_transpose, Conv, InputList, OpError, Operator, Padding};

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

        let mut output = Tensor::zeros(&[batch, out_chans, out_h, out_w]);

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
        let kernel = Tensor::from_data(
            &[1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );

        let input = Tensor::from_data(
            &[1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let expected_with_same_padding = Tensor::from_data(
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

        let expected_with_no_padding = Tensor::from_data(&[1, 1, 1, 1], vec![2.6358]);

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

        let expected_with_bias = Tensor::from_data(&[1, 1, 1, 1], vec![3.6358]);
        let bias = Tensor::from_data(&[1], vec![1.0]);
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
        let kernel = &Tensor::from_data(
            &[1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );

        let input = &Tensor::from_data(
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
        let kernel = Tensor::rand(&[10, 5, 3, 3], &mut rng);
        let input = Tensor::rand(&[1, 5, 10, 10], &mut rng);
        let bias = Tensor::rand(&[10], &mut rng);

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
        let kernel = Tensor::rand(&[10, 1, 3, 3], &mut rng);
        let input = Tensor::rand(&[1, 10, 10, 10], &mut rng);
        let bias = Tensor::rand(&[10], &mut rng);

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
        let kernel = Tensor::rand(&[10, 5, 1, 1], &mut rng);
        let input = Tensor::rand(&[1, 5, 20, 20], &mut rng);
        let bias = Tensor::rand(&[10], &mut rng);

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
        let input = Tensor::rand(&[2, 5, 20, 20], &mut rng);
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
        let input = Tensor::from_data(
            &[1, 3, 2, 2],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 1.5202, 1.5592,
                0.9939, 1.7475,
            ],
        );
        let kernel = Tensor::from_data(
            &[3, 1, 2, 2],
            vec![
                -0.0862, -0.4111, 0.0813, 0.4993, -0.4641, 0.1715, -0.0532, -0.2429, -0.4325,
                0.4273, 0.4180, 0.4338,
            ],
        );
        let bias = Tensor::from_data(&[3], vec![0.1, 0.2, 0.3]);
        let expected = Tensor::from_data(
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
        let kernel = Tensor::rand(&[4, 2, 3, 3], &mut rng);
        let input = Tensor::rand(&[2, 4, 20, 20], &mut rng);
        let bias = Tensor::rand(&[4], &mut rng);

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
        let kernel = Tensor::rand(&[4, 3, 3, 3], &mut rng);

        for strides in [[2, 2], [3, 3], [1, 3]] {
            for pad in [0, 1] {
                for input_size in [3, 4, 5, 10, 20] {
                    let input = Tensor::rand(&[2, 3, input_size, input_size], &mut rng);
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
        let kernel = Tensor::rand(&[3, 1, 3, 3], &mut rng);

        for strides in [[2, 2], [3, 3], [1, 3]] {
            for pad in [0, 1] {
                for input_size in [3, 4, 5, 10, 20] {
                    let input = Tensor::rand(&[1, 3, input_size, input_size], &mut rng);
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
        let input = Tensor::rand(&[1, 1, 2, 2], &mut rng);
        let kernel = Tensor::rand(&[1, 1, 3, 3], &mut rng);

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
        let input = Tensor::rand(&[1, 1, 2, 2], &mut rng);
        let kernel = Tensor::rand(&[1, 1, 2, 2], &mut rng);

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
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = Tensor::from_data(&[1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);
        let expected = Tensor::from_data(
            &[1, 1, 4, 4],
            vec![
                0.1000, 0.2000, 0.2000, 0.4000, 0.3000, 0.4000, 0.6000, 0.8000, 0.3000, 0.6000,
                0.4000, 0.8000, 0.9000, 1.2000, 1.2000, 1.6000,
            ],
        );

        let result = conv_transpose(&input, &kernel, None, [2, 2]).unwrap();
        expect_equal(&result, &expected)?;

        let mut expected_with_bias =
            Tensor::from_data(expected.shape().into(), expected.data().to_vec());
        for i in 0..expected_with_bias.len() {
            expected_with_bias.data_mut()[i] += 1.234;
        }
        let bias = Tensor::from_data(&[1], vec![1.234]);
        let result = conv_transpose(&input, &kernel, Some(&bias), [2, 2]).unwrap();
        expect_equal(&result, &expected_with_bias)
    }
}
