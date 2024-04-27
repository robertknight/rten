use std::iter::zip;
use std::mem::MaybeUninit;

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, NdTensorViewMut, Tensor, TensorView};
use smallvec::SmallVec;

use crate::check_dims;
use crate::gemm::{add_scaled_vector, div_ceil, GemmExecutor, GemmInputA, GemmInputB};
use crate::iter_util::range_chunks;
use crate::ops::pooling::calc_output_size_and_padding;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output, Padding};
use crate::tensor_pool::{AutoReturn, TensorPool};

mod im2col;
use im2col::VirtualIm2Col;

// Calculate the min and max output X coordinates that are valid when updating
// a row of convolution output using a loop:
//
// ```
// for out_x in min_out_x..max_out_x {
//   out_row[out_x] += in_row[out_x * stride + k_x * dilation - pad_w] * kernel_element
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
    dilation: usize,
    out_w: usize,
) -> (usize, usize) {
    let min_out_x = pad_left.saturating_sub(k_x * dilation);
    let max_out_x = div_ceil((in_w + pad_left).saturating_sub(k_x * dilation), stride).min(out_w);
    (min_out_x, max_out_x)
}

/// Initialize a tensor, typically with NCHW or CHW dimensions, with a vector
/// of per-channel biases.
///
/// This is slightly more efficient than creating a zero-filled tensor and then
/// adding the appropriate bias to each element.
fn init_tensor_with_channel_bias(
    pool: &TensorPool,
    shape: &[usize],
    chan_dim: usize,
    bias: &NdTensorView<f32, 1>,
) -> Tensor {
    let mut out_data = pool.alloc_vec(shape.iter().product());

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
fn conv_2d_pointwise(
    pool: &TensorPool,
    input: &NdTensorView<f32, 4>,
    kernel: &NdTensorView<f32, 4>,
    bias: Option<NdTensorView<f32, 1>>,
) -> Tensor {
    let [batch, _, in_h, in_w]: [usize; 4] = input.shape();
    let [out_c, in_c, _, _]: [usize; 4] = kernel.shape();
    let mut output = pool.alloc([batch, out_c, in_h * in_w].as_slice());

    // Get input and kernel as contiguous tensors so we can create reshaped
    // views.
    let input = input.to_contiguous();
    let kernel = kernel.to_contiguous();
    let kernel_mat = kernel.reshaped([out_c, in_c]);

    // Bias must be contiguous for use with `gemm_bias`.
    let bias = bias.as_ref().map(|b| b.to_contiguous());

    let gemm = GemmExecutor::new();
    let mut n_init = 0;

    for n in 0..batch {
        let mut out_item = output.slice_mut::<2, _>([n]);
        let out_row_stride = out_item.stride(0);

        let in_mat = input.slice::<3, _>([n]).reshaped([in_c, in_h * in_w]);

        gemm.gemm_uninit_bias(
            out_item.data_mut().unwrap(),
            out_row_stride,
            GemmInputA::Unpacked(kernel_mat),
            GemmInputB::Unpacked(in_mat),
            1., // alpha
            bias.as_ref().map(|b| b.data().unwrap()),
        );
        n_init += out_item.len();
    }

    output.reshape(&[batch, out_c, in_h, in_w]);

    // Safety: We used `gemm_uninit_bias` to initialize all elements.
    assert!(n_init == output.len());
    unsafe { output.assume_init() }
}

/// Compute depthwise convolution for the block of channels from `input`
/// specified by `chan_range` into `out_chans`.
///
/// `col_range_for_kernel_x` is a precomputed map of kernel X coordinate to
/// `(in_range, out_range)` of column ranges that are valid for the input and
/// output.
///
/// When this function returns, all elements of `out_chans` will have been
/// initialized.
fn conv_2d_depthwise_block(
    mut output: NdTensorViewMut<MaybeUninit<f32>, 3>, // C, H, W
    chan_range: Range<usize>,
    input: NdTensorView<f32, 3>,  // C, H, W
    kernel: NdTensorView<f32, 4>, // C, _, Kh, Kw
    bias: Option<NdTensorView<f32, 1>>,
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
    col_range_for_kernel_x: &[(Range<usize>, Range<usize>)],
) {
    let [_, out_h, _out_w] = output.shape();
    let [_, _, k_h, _k_w] = kernel.shape();
    let [_, in_h, _in_w] = input.shape();
    let [stride_h, stride_w] = strides;
    let [pad_top, _pad_left, _pad_bottom, _pad_right] = padding;
    let [dilation_y, _dilation_x] = dilations;

    for c in chan_range.clone() {
        let kernel_view = kernel.slice([c, 0]).weakly_checked_view();

        // For efficiency, use manual slicing in the inner loops to extract
        // input/output rows.
        let mut out_chan = output.slice_mut::<2, _>([c - chan_range.start]);
        let out_row_stride = out_chan.stride(0);
        let out_chan_data = out_chan.data_mut().unwrap();

        let in_chan = input.slice::<2, _>([c]);
        let in_row_stride = in_chan.stride(0);
        let in_chan_data = in_chan.data().unwrap();

        let init_value = if let Some(bias) = bias { bias[[c]] } else { 0. };

        // The loops here are ordered so that the inner-most loop is as
        // efficient as possible and runs for as long as possible over a
        // contiguous slice of memory.
        for out_y in 0..out_h {
            let out_row = &mut out_chan_data[out_y * out_row_stride..][..out_row_stride];

            // Initialize output row.
            for x in out_row.iter_mut() {
                x.write(init_value);
            }
            let out_row: &mut [f32] = unsafe { std::mem::transmute(out_row) };

            for k_y in 0..k_h {
                let in_y = out_y * stride_h + k_y * dilation_y;
                if in_y < pad_top || in_y >= in_h + pad_top {
                    continue;
                }

                let in_row_y = in_y - pad_top;
                let in_row = &in_chan_data[in_row_y * in_row_stride..][..in_row_stride];

                for (k_x, (in_range, out_range)) in col_range_for_kernel_x.iter().enumerate() {
                    add_scaled_vector(
                        &mut out_row[out_range.clone()],
                        &in_row[in_range.clone()],
                        1,        /* dest_stride */
                        stride_w, /* src_stride */
                        kernel_view[[k_y, k_x]],
                    );
                }
            }
        }
    }
}

/// Specialization of conv_2d for depthwise convolutions.
///
/// Depthwise convolutions operate over a single input/output channel at
/// a time and hence the transformation of convolution to matrix multiplication
/// doesn't pay off. An optimized direct method works better.
fn conv_2d_depthwise(
    pool: &TensorPool,
    input: &NdTensorView<f32, 4>,
    kernel: &NdTensorView<f32, 4>,
    bias: Option<NdTensorView<f32, 1>>,
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
    out_hw: [usize; 2],
) -> Tensor {
    let [batch, _in_c, _in_h, in_w]: [usize; 4] = input.shape();
    let [out_c, _, _k_h, k_w]: [usize; 4] = kernel.shape();
    let [_pad_top, pad_left, _pad_bottom, _pad_right] = padding;
    let [_stride_h, stride_w] = strides;
    let [_dilation_y, dilation_x] = dilations;
    let [out_h, out_w] = out_hw;

    let mut output = pool.alloc([batch, out_c, out_h, out_w]);

    // Use of input rows below assumes contiguous last dimension.
    let input = input.to_contiguous();

    // Map of kernel X position to `(in_range, out_range)` of column ranges that
    // are used in the inner loop.
    let col_range_for_kernel_x: SmallVec<[_; 7]> = (0..k_w)
        .map(|k_x| {
            let (min_out_x, max_out_x) =
                min_max_out_x_coords(k_x, in_w, pad_left, stride_w, dilation_x, out_w);
            let out_range = min_out_x..max_out_x;

            let min_in_x = min_out_x * stride_w + k_x * dilation_x - pad_left;
            let max_in_x = if out_range.is_empty() {
                // `max_out_x` could be zero, so `max_out_x - 1` would underflow.
                // If the output range is empty, the input range must be too.
                min_in_x
            } else {
                (max_out_x - 1) * stride_w + k_x * dilation_x - pad_left + 1
            };

            (min_in_x..max_in_x, min_out_x..max_out_x)
        })
        .collect();

    // Minimum number of elements in a channel chunk.
    let target_chunk_size = 32 * 1024;
    let channel_chunk_size = (target_chunk_size / (out_h * out_w)).clamp(1, out_c);

    let n_init = AtomicUsize::new(0);
    for n in 0..batch {
        let mut out_chans = output.slice_mut::<3, _>(n);
        let input = input.slice::<3, _>(n);

        out_chans
            .axis_chunks_mut(0, channel_chunk_size)
            .zip(range_chunks(0..out_c, channel_chunk_size))
            .par_bridge()
            .for_each(|(mut out_chans, chan_range)| {
                conv_2d_depthwise_block(
                    out_chans.nd_view_mut(),
                    chan_range,
                    input,
                    kernel.view(),
                    bias,
                    padding,
                    strides,
                    dilations,
                    &col_range_for_kernel_x,
                );

                n_init.fetch_add(out_chans.len(), Ordering::SeqCst);
            });
    }

    // Safety: We initialized all output rows
    assert!(n_init.load(Ordering::SeqCst) == output.len());
    unsafe { output.into_dyn().assume_init() }
}

/// Perform a convolution of `input` with `kernel`.
///
/// For a 2D convolution `input` has dimensions NCHW while `kernel` has OGHW
/// where `G` is `C / groups`. 1D convolutions are similar except the "H"
/// dimension is omitted.
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
    pool: &TensorPool,
    input: TensorView,
    kernel: TensorView,
    bias: Option<TensorView>,
    padding: Padding,
    groups: usize,
    strides: &[usize],
    dilations: &[usize],
) -> Result<Tensor, OpError> {
    // Handle 1D convolution by expanding to 2D and then removing the extra
    // dimension from the result.
    if input.ndim() == 3 {
        let [n, c, w] = check_dims!(input, 3, "NCW");
        let [out_c, k_in_c, k_w] = check_dims!(kernel, 3, "OCW");

        let mut input_2d = input.clone();
        input_2d.reshape(&[n, c, 1, w]);

        let mut kernel_2d = kernel.clone();
        kernel_2d.reshape(&[out_c, k_in_c, 1, k_w]);

        let padding_2d: Padding = match padding {
            Padding::Same => Padding::Same,
            Padding::Fixed(pads) => match pads.as_slice() {
                &[pad_start, pad_end] => [0, pad_start, 0, pad_end].into(),
                _ => {
                    return Err(OpError::InvalidValue("expected 2 pad values"));
                }
            },
        };

        let strides_2d = match strides {
            &[stride] => [1, stride],
            _ => {
                return Err(OpError::InvalidValue("expected 1 stride value"));
            }
        };

        let dilations_2d = match dilations {
            &[dilation] => [1, dilation],
            _ => {
                return Err(OpError::InvalidValue("expected 1 dilation value"));
            }
        };

        let result_2d = conv(
            pool,
            input_2d,
            kernel_2d,
            bias,
            padding_2d,
            groups,
            &strides_2d,
            &dilations_2d,
        );

        return result_2d.map(|mut t| {
            let [n, c, _h, w]: [usize; 4] = t.shape().try_into().expect("expected 4D output");
            t.reshape(&[n, c, w]);
            t
        });
    }

    let [batch, in_c, in_h, in_w] = check_dims!(input, 4, "NCHW");
    let [out_c, k_in_c, k_h, k_w] = check_dims!(kernel, 4, "OCHW");
    check_dims!(bias?, 1);

    let input = input.view();
    let kernel = kernel.view();

    let [stride_y, stride_x]: [usize; 2] = strides
        .try_into()
        .map_err(|_| OpError::InvalidValue("expected 2 stride values"))?;
    let [dilation_y, dilation_x]: [usize; 2] = dilations
        .try_into()
        .map_err(|_| OpError::InvalidValue("expected 2 dilation values"))?;

    let (out_h, out_w, fixed_padding) = calc_output_size_and_padding(
        (in_h, in_w),
        (k_h, k_w),
        (stride_y, stride_x),
        padding,
        Some((dilation_y, dilation_x)),
    )?;

    let [pad_top, pad_left, pad_bottom, pad_right] = fixed_padding;

    let has_padding = pad_top > 0 || pad_left > 0 || pad_bottom > 0 || pad_right > 0;

    if k_h == 1
        && k_w == 1
        && !has_padding
        && groups == 1
        && stride_y == 1
        && stride_x == 1
        && dilation_y == 1
        && dilation_x == 1
    {
        return Ok(conv_2d_pointwise(
            pool,
            &input.nd_view(),
            &kernel.nd_view(),
            bias.as_ref().map(|b| b.nd_view()),
        ));
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
            pool,
            &input.nd_view(),
            &kernel.nd_view(),
            bias.map(|b| b.nd_view()),
            fixed_padding,
            [stride_y, stride_x],
            [dilation_y, dilation_x],
            [out_h, out_w],
        ));
    }

    let n_patches = out_h * out_w;
    let mut output = pool.alloc([batch, out_c, n_patches].as_slice());
    let gemm = GemmExecutor::new();

    // Bias must be contiguous for use with `gemm_bias`.
    let bias = bias.map(|b| b.to_contiguous());
    let bias = bias.as_ref().map(|b| b.view());

    let n_init = AtomicUsize::new(0);

    for group in 0..groups {
        let in_chan_start = group * in_channels_per_group;
        let in_chan_end = in_chan_start + in_channels_per_group;
        let out_chan_start = group * out_channels_per_group;
        let out_chans = out_chan_start..out_chan_start + out_channels_per_group;

        let in_group = input.slice_dyn((.., in_chan_start..in_chan_end));
        let mut out_group = output.slice_mut_dyn((.., out_chans.clone()));

        let kernel_mat = kernel
            .slice::<4, _>([out_chans.clone()])
            .reshaped([out_channels_per_group, in_channels_per_group * k_h * k_w]);

        // Prepack kernel if we'll be able to reuse packed weights.
        let prepacked_kernel = if in_group.size(0) > 1 {
            Some(gemm.prepack_a(kernel_mat))
        } else {
            None
        };

        zip(out_group.axis_iter_mut(0), in_group.axis_iter(0))
            .par_bridge()
            .for_each(|(mut out_item, in_item)| {
                let mut out_mat = out_item.reshaped_mut([out_channels_per_group, out_h * out_w]);
                let out_row_stride = out_mat.stride(0);

                let im2col = VirtualIm2Col::new(
                    gemm.kernel_type(),
                    in_item.nd_view(),
                    [k_h, k_w],
                    fixed_padding,
                    [stride_y, stride_x],
                    [dilation_y, dilation_x],
                    gemm.b_panel_width(),
                );

                gemm.gemm_uninit_bias(
                    out_mat.data_mut().unwrap(),
                    out_row_stride,
                    prepacked_kernel
                        .as_ref()
                        .map(GemmInputA::Packed)
                        .unwrap_or(GemmInputA::Unpacked(kernel_mat)),
                    GemmInputB::Virtual(&im2col),
                    1., // alpha
                    bias.as_ref().map(|b| &b.data().unwrap()[out_chans.clone()]),
                );
                n_init.fetch_add(out_mat.len(), Ordering::SeqCst);
            });
    }

    output.reshape(&[batch, out_c, out_h, out_w]);

    // Safety: We used `gemm_uninit_bias` to initialize all elements.
    assert!(n_init.load(Ordering::SeqCst) == output.len());
    let output = unsafe { output.assume_init() };

    Ok(output)
}

#[derive(Debug)]
pub struct Conv {
    pub groups: usize,
    pub dilations: Vec<usize>,
    pub padding: Padding,
    pub strides: Vec<usize>,
}

impl Operator for Conv {
    fn name(&self) -> &str {
        "Conv"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        let weight = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;
        conv(
            pool,
            input,
            weight,
            bias,
            self.padding.clone(),
            self.groups,
            &self.strides,
            &self.dilations,
        )
        .into_op_result()
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
fn col2im(
    output: &mut NdTensorViewMut<f32, 3>,
    columns: &NdTensorView<f32, 5>,
    strides: [usize; 2],
) {
    let [stride_h, stride_w] = strides;

    // If we assume `columns` is likely already contiguous, we can avoid offset
    // calculations and just iterate over the underlying data.
    let columns = columns.to_contiguous();
    let columns_shape = columns.shape();
    let mut col_data_iter = columns.data().unwrap().iter();

    let mut out_view = output.weakly_checked_view_mut();

    // Loop order must match dim order of `columns`.
    for y in 0..columns_shape[0] {
        for x in 0..columns_shape[1] {
            for out_c in 0..columns_shape[2] {
                for k_y in 0..columns_shape[3] {
                    let out_y = y * stride_h + k_y;
                    for k_x in 0..columns_shape[4] {
                        let out_x = x * stride_w + k_x;
                        out_view[[out_c, out_y, out_x]] += col_data_iter.next().unwrap();
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
    pool: &TensorPool,
    input: TensorView,
    kernel: TensorView,
    bias: Option<TensorView>,
    strides: [usize; 2],
) -> Result<Tensor, OpError> {
    let [batch, in_c, in_h, in_w] = check_dims!(input, 4, "NCHW");
    let [k_in_c, out_c, k_h, k_w] = check_dims!(kernel, 4, "OCHW");
    check_dims!(bias?, 1);

    let bias = bias.map(|b| b.nd_view());

    if in_c != k_in_c {
        return Err(OpError::IncompatibleInputShapes(
            "Input channels does not match kernel input channels",
        ));
    }

    let [stride_h, stride_w] = strides;
    let out_h = (in_h - 1) * stride_h + k_h;
    let out_w = (in_w - 1) * stride_w + k_w;

    let mut output = if let Some(bias) = bias {
        init_tensor_with_channel_bias(pool, &[batch, out_c, out_h, out_w], 1, &bias)
    } else {
        pool.alloc_zeroed([batch, out_c, out_h, out_w].as_slice())
    };

    // Ensure input and kernel are contiguous to support reshaping.
    let input = input.to_contiguous();
    let kernel = kernel.to_contiguous();

    let mut col2im_mat = pool
        .alloc([in_h * in_w, out_c * k_h * k_w])
        .auto_return(pool);
    let kernel_mat = kernel.reshaped([k_in_c, out_c * k_h * k_w]);
    let gemm = GemmExecutor::new();

    // The implementation here is the inverse of the im2col-based convolution.
    for n in 0..batch {
        let input_mat = input
            .slice::<3, _>([n])
            .reshaped([in_c, in_h * in_w])
            .transposed();

        let col2im_row_stride = col2im_mat.stride(0);
        gemm.gemm_uninit(
            col2im_mat.data_mut().unwrap(),
            col2im_row_stride,
            GemmInputA::Unpacked(input_mat),
            GemmInputB::Unpacked(kernel_mat),
            1., /* alpha */
        );

        // Safety: `gemm_uninit` initialized col2im_mat.
        let col2im_mat = unsafe { col2im_mat.view().assume_init() };

        col2im(
            &mut output.nd_view_mut::<4>().slice_mut([n]),
            &col2im_mat.reshaped([in_h, in_w, out_c, k_h, k_w]),
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        let weight = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;
        conv_transpose(pool, input, weight, bias, self.strides).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{expect_equal, ExpectEqualError};
    use rten_tensor::{Tensor, TensorView};

    use crate::ops::pooling::calc_output_size_and_padding;
    use crate::ops::tests::expect_eq_1e4;
    use crate::ops::tests::new_pool;
    use crate::ops::{conv, conv_transpose, Conv, OpError, Operator, Padding};
    use crate::tensor_pool::AutoReturn;

    /// Un-optimized reference implementation of convolution.
    ///
    /// This has the same interface as [conv].
    fn reference_conv(
        input: TensorView,
        kernel: TensorView,
        bias: Option<TensorView>,
        padding: Padding,
        groups: usize,
        strides: &[usize],
        dilations: &[usize],
    ) -> Tensor {
        let [batch, in_chans, in_h, in_w]: [usize; 4] =
            input.shape().try_into().expect("expected NCHW input");
        let [out_chans, k_in_chans, k_h, k_w]: [usize; 4] =
            kernel.shape().try_into().expect("expected OCHW input");
        let [stride_y, stride_x] = strides.try_into().expect("expected 2 stride values");
        let [dilation_y, dilation_x] = dilations.try_into().expect("expected 2 stride values");
        let (out_h, out_w, fixed_pads) = calc_output_size_and_padding(
            (in_h, in_w),
            (k_h, k_w),
            (stride_y, stride_x),
            padding.into(),
            Some((dilation_y, dilation_x)),
        )
        .expect("Input too small");
        let [pad_top, pad_left, _pad_bottom, _pad_right] = fixed_pads;

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
                    let chan_bias = if let Some(ref bias) = bias {
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
                                        let in_y = out_y * stride_y + k_y * dilation_y;
                                        let in_x = out_x * stride_x + k_x * dilation_x;

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

    /// Perform a convolution using the optimized and reference implementations
    /// and check that the results are approximately equal.
    fn check_conv(
        input: TensorView,
        kernel: TensorView,
        bias: Option<TensorView>,
        pads: Padding,
        groups: usize,
        strides: &[usize],
        dilations: &[usize],
    ) -> Result<Tensor, ExpectEqualError> {
        let pool = new_pool();
        let result = conv(
            &pool,
            input.view(),
            kernel.view(),
            bias.clone(),
            pads.clone(),
            groups,
            &strides,
            &dilations,
        )
        .expect("conv operation failed");
        let reference_result =
            reference_conv(input, kernel, bias, pads, groups, strides, dilations);
        expect_equal(&result, &reference_result)?;
        Ok(result)
    }

    /// Basic tests for conv. These compare the results against values
    /// computed from PyTorch as well as the reference implementation.
    #[test]
    fn test_conv() -> Result<(), Box<dyn Error>> {
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

        let result = check_conv(
            input.view(),
            kernel.view(),
            None,
            [1, 1, 1, 1].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;
        expect_eq_1e4(&result, &expected_with_same_padding)?;

        let expected_with_no_padding = Tensor::from_data(&[1, 1, 1, 1], vec![2.6358]);

        let result = check_conv(
            input.view(),
            kernel.view(),
            None,
            [0, 0, 0, 0].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;
        expect_eq_1e4(&result, &expected_with_no_padding)?;

        let expected_with_bias = Tensor::from_data(&[1, 1, 1, 1], vec![3.6358]);
        let bias = Tensor::from_data(&[1], vec![1.0]);
        let result = check_conv(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [0, 0, 0, 0].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;
        expect_eq_1e4(&result, &expected_with_bias)?;

        Ok(())
    }

    #[test]
    fn test_conv_same_padding() -> Result<(), Box<dyn Error>> {
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

        let pool = new_pool();
        let op = Conv {
            padding: Padding::Same,
            groups: 1,
            strides: vec![1, 1],
            dilations: vec![1, 1],
        };
        let result = op
            .run(&pool, (&input, &kernel).into())
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        let reference_result = reference_conv(
            input.view(),
            kernel.view(),
            None,
            [1, 1, 1, 1].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        );

        expect_equal(&result, &reference_result)?;

        Ok(())
    }

    #[test]
    fn test_conv_uneven_padding() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = Tensor::rand(&[10, 5, 3, 3], &mut rng);
        let input = Tensor::rand(&[1, 5, 10, 10], &mut rng);
        let bias = Tensor::rand(&[10], &mut rng);

        check_conv(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [0, 0, 1, 1].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;

        Ok(())
    }

    #[test]
    fn test_conv_depthwise_uneven_padding() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = Tensor::rand(&[10, 1, 3, 3], &mut rng);
        let input = Tensor::rand(&[1, 10, 10, 10], &mut rng);
        let bias = Tensor::rand(&[10], &mut rng);

        check_conv(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [0, 0, 1, 1].into(),
            10,      /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;

        Ok(())
    }

    // Specific tests for convolutions with a 1x1 kernel.
    #[test]
    fn test_conv_pointwise() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = Tensor::rand(&[10, 5, 1, 1], &mut rng);
        let input = Tensor::rand(&[1, 5, 20, 20], &mut rng);
        let bias = Tensor::rand(&[10], &mut rng);

        // Contiguous inputs
        let result = check_conv(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [0, 0, 0, 0].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;
        assert_eq!(result.shape(), [1, 10, 20, 20]);

        // Non-contiguous inputs
        let mut input_transposed = input.clone();
        input_transposed.permute(&[0, 1, 3, 2]);
        assert!(!input_transposed.is_contiguous());

        let result = check_conv(
            input_transposed.view(),
            kernel.view(),
            Some(bias.view()),
            [0, 0, 0, 0].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;
        assert_eq!(result.shape(), [1, 10, 20, 20]);

        // Batch size > 1
        let input = Tensor::rand(&[2, 5, 20, 20], &mut rng);
        let result = check_conv(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [0, 0, 0, 0].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;
        assert_eq!(result.shape(), [2, 10, 20, 20]);

        // Stride > 1
        let input = Tensor::rand(&[1, 5, 20, 20], &mut rng);
        let result = check_conv(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [0, 0, 0, 0].into(),
            1,       /* groups */
            &[2, 2], /* stride */
            &[1, 1], /* dilations */
        )?;
        assert_eq!(result.shape(), [1, 10, 10, 10]);

        Ok(())
    }

    // Specific tests for convolutions that operate over one output channel and
    // one input channel at a time.
    #[test]
    fn test_conv_depthwise() -> Result<(), Box<dyn Error>> {
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

        let result = check_conv(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [0, 0, 0, 0].into(),
            3,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;

        expect_equal(&result, &expected)?;

        Ok(())
    }

    // Tests for convolutions that are neither pointwise nor depthwise. In
    // other words, the kernel has a spatial size > 1x1 and a channel depth > 1.
    #[test]
    fn test_conv_not_depthwise_or_pointwise() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = Tensor::rand(&[4, 2, 3, 3], &mut rng);
        let input = Tensor::rand(&[2, 4, 20, 20], &mut rng);
        let bias = Tensor::rand(&[4], &mut rng);

        check_conv(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [1, 1, 1, 1].into(),
            2,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
        )?;

        Ok(())
    }

    #[test]
    fn test_conv_strided() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = Tensor::rand(&[4, 3, 3, 3], &mut rng);

        for strides in [[2, 2], [3, 3], [1, 3]] {
            for pad in [0, 1] {
                for input_size in [3, 4, 5, 10, 20] {
                    let input = Tensor::rand(&[2, 3, input_size, input_size], &mut rng);
                    check_conv(
                        input.view(),
                        kernel.view(),
                        None,
                        [pad, pad, pad, pad].into(),
                        1, /* groups */
                        &strides,
                        &[1, 1], /* dilations */
                    )?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_conv_strided_depthwise() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = Tensor::rand(&[3, 1, 3, 3], &mut rng);

        for strides in [[2, 2], [3, 3], [1, 3]] {
            for pad in [0, 1] {
                for input_size in [3, 4, 5, 10, 20] {
                    let input = Tensor::rand(&[1, 3, input_size, input_size], &mut rng);
                    check_conv(
                        input.view(),
                        kernel.view(),
                        None,
                        [pad, pad, pad, pad].into(),
                        3, /* groups */
                        &strides,
                        &[1, 1], /* dilations */
                    )?;
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

        let pool = new_pool();
        let result = conv(
            &pool,
            input.view(),
            kernel.view(),
            None,
            [0; 4].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
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

        let pool = new_pool();
        let result = conv(
            &pool,
            input.view(),
            kernel.view(),
            None,
            [0; 4].into(),
            1,       /* groups */
            &[0, 0], /* stride */
            &[1, 1], /* dilations */
        );

        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Strides must be > 0"))
        );
    }

    #[test]
    fn test_conv_dilated() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = Tensor::rand(&[4, 3, 3, 3], &mut rng);

        for dilations in [[2, 2], [3, 3], [1, 3]] {
            for pad in [0, 1] {
                for input_size in [7, 10, 20] {
                    let input = Tensor::rand(&[2, 3, input_size, input_size], &mut rng);
                    check_conv(
                        input.view(),
                        kernel.view(),
                        None,
                        [pad, pad, pad, pad].into(),
                        1, /* groups */
                        &[1, 1],
                        &dilations,
                    )?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_conv_dilated_depthwise() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let chans = 3;
        let kernel = Tensor::rand(&[chans, 1, 3, 3], &mut rng);

        for dilations in [[2, 2], [3, 3], [1, 3]] {
            for pad in [0, 1] {
                for input_size in [7, 10, 20] {
                    let input = Tensor::rand(&[2, chans, input_size, input_size], &mut rng);
                    check_conv(
                        input.view(),
                        kernel.view(),
                        None,
                        [pad, pad, pad, pad].into(),
                        chans, /* groups */
                        &[1, 1],
                        &dilations,
                    )?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_conv_1d() {
        let mut rng = XorShiftRng::new(1234);
        let [n, in_c, out_c, in_w, k_w] = [1, 5, 10, 20, 3];
        let input = Tensor::rand(&[n, in_c, in_w], &mut rng);
        let kernel = Tensor::rand(&[out_c, in_c, k_w], &mut rng);

        let pool = new_pool();
        let result = conv(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::Same,
            1,    /* groups */
            &[1], /* stride */
            &[1], /* dilation */
        )
        .unwrap();

        assert_eq!(result.shape(), &[n, out_c, in_w]);
    }

    #[test]
    fn test_conv_transpose() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = Tensor::from_data(&[1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);
        let expected = Tensor::from_data(
            &[1, 1, 4, 4],
            vec![
                0.1000, 0.2000, 0.2000, 0.4000, 0.3000, 0.4000, 0.6000, 0.8000, 0.3000, 0.6000,
                0.4000, 0.8000, 0.9000, 1.2000, 1.2000, 1.6000,
            ],
        );

        let result = conv_transpose(&pool, input.view(), kernel.view(), None, [2, 2]).unwrap();
        expect_equal(&result, &expected)?;

        let mut expected_with_bias = Tensor::from_data(expected.shape().into(), expected.to_vec());
        for eb in expected_with_bias.iter_mut() {
            *eb += 1.234;
        }
        let bias = Tensor::from_data(&[1], vec![1.234]);
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [2, 2],
        )
        .unwrap();
        expect_equal(&result, &expected_with_bias)?;

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_depthwise_conv() {
        let mut rng = XorShiftRng::new(1234);

        // Input and kernel sizes copied from the last layers of MobileNetV2.
        // This has a small spatial shape, so it measures overhead around the
        // inner loop. A larger spatial shape would be more affected by the
        // efficiency of the innermost loops.
        let input = Tensor::rand(&[1, 576, 14, 14], &mut rng);
        let kernel = Tensor::rand(&[576, 1, 3, 3], &mut rng);

        let n_groups = input.size(1);
        let padding = Padding::Fixed([1, 1, 1, 1].into());
        let bias = None;
        let dilations = [1, 1];

        let iters = 100;
        let pool = new_pool();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            for stride in [1, 1, 2] {
                conv(
                    &pool,
                    input.view(),
                    kernel.view(),
                    bias.clone(),
                    padding.clone(),
                    n_groups,
                    &[stride, stride],
                    &dilations,
                )
                .unwrap()
                .auto_return(&pool);
            }
        }
        let elapsed = start.elapsed().as_secs_f32() * 1000.0;

        println!("depthwise_conv {elapsed:.3}ms",);
    }

    #[test]
    #[ignore]
    fn bench_col2im() {
        use rten_bench::run_bench;
        use rten_tensor::NdTensor;

        use super::col2im;

        let out_chans = 32;
        let in_height = 64;
        let in_width = 64;
        let kernel_height = 3;
        let kernel_width = 3;
        let [stride_y, stride_x] = [2, 2];
        let out_height = (in_height - 1) * stride_y + (kernel_height - 1) + 1;
        let out_width = (in_width - 1) * stride_x + (kernel_width - 1) + 1;

        let mut rng = XorShiftRng::new(1234);
        let mut output = NdTensor::zeros([out_chans, out_height, out_width]);
        let columns = NdTensor::rand(
            [in_height, in_width, out_chans, kernel_height, kernel_width],
            &mut rng,
        );

        run_bench(100, Some("col2im"), || {
            col2im(
                &mut output.view_mut(),
                &columns.view(),
                [stride_y, stride_x],
            );
        });
    }
}
