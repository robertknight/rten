use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use rten_base::unroll::unroll_loop;
use rten_tensor::prelude::*;
use rten_tensor::{AssumeInit, NdTensor, NdTensorView, NdTensorViewMut};
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};

/// Calculate the output coordinate range for which all input / output
/// coordinates are valid when updating a row of output using:
///
/// ```text
/// for out_x in min_out_x..max_out_x {
///   out_row[out_x] += in_row[out_x * stride + k_x * dilation - pad_left] * kernel_row[k_x]
/// }
/// ```
///
/// Where `in_row` is the un-padded input row of width `in_w` and `out_row` is
/// an output row of width `out_w`.
///
/// In other words, we want to find the minimum and maximum values of `out_x`
/// such that:
///
/// - out_x * stride + k_x * dilation - pad_left >= 0
/// - out_x * stride + k_x * dilation - pad_left <= in_w
/// - out_x >= 0
/// - out_x <= out_w
fn min_max_out_x_coords(
    k_x: usize,
    in_w: usize,
    pad_left: usize,
    stride: usize,
    dilation: usize,
    out_w: usize,
) -> (usize, usize) {
    let min_out_x = pad_left
        .saturating_sub(k_x * dilation)
        .div_ceil(stride)
        .min(out_w);
    let max_out_x = (in_w + pad_left)
        .saturating_sub(k_x * dilation)
        .div_ceil(stride)
        .min(out_w);
    (min_out_x, max_out_x)
}

/// Convolution parameters passed to the kernel which are constant across
/// all calls within a single depthwise conv operation.
struct ConvParams<'a> {
    kernel_h: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_y: usize,
    pad_top: usize,
    in_h: usize,
    in_row_len: usize,
    in_row_stride: usize,

    /// Precomputed map of kernel X coordinate to input and output row
    /// range.
    col_range_for_kernel_x: &'a [(Range<usize>, Range<usize>)],
}

/// Trait for architecture and data-type specific kernels for depthwise
/// convolution.
trait DepthwiseConvKernel<X, W, Y> {
    /// Compute one output row of the depthwise convolution.
    ///
    /// `image_chan` and `kernel_chan` contain the input data and weights for
    /// the channel that corresponds to the output row.
    fn compute_row(
        &self,
        params: &ConvParams,
        out_row: &mut [MaybeUninit<Y>],
        out_y: usize,
        out_init: Y,
        image_chan: &[X],
        kernel_chan: NdTensorView<W, 2>,
        input_zero: X,
        kernel_zero: W,
    );
}

struct GenericDepthwiseConvKernel {}

impl DepthwiseConvKernel<f32, f32, f32> for GenericDepthwiseConvKernel {
    fn compute_row(
        &self,
        params: &ConvParams,
        out_row: &mut [MaybeUninit<f32>],
        out_y: usize,
        out_init: f32,
        image_chan: &[f32],
        kernel_chan: NdTensorView<f32, 2>,
        _input_zero: f32,
        _kernel_zero: f32,
    ) {
        let kernel_view = kernel_chan.weakly_checked_view();

        // Initialize output row.
        for x in out_row.iter_mut() {
            x.write(out_init);
        }
        let out_row: &mut [f32] = unsafe { out_row.assume_init() };

        for k_y in 0..params.kernel_h {
            let in_y = out_y * params.stride_h + k_y * params.dilation_y;
            if in_y < params.pad_top || in_y >= params.in_h + params.pad_top {
                continue;
            }

            let in_row_y = in_y - params.pad_top;
            let in_row = &image_chan[in_row_y * params.in_row_stride..][..params.in_row_len];

            for (k_x, (in_range, out_range)) in params.col_range_for_kernel_x.iter().enumerate() {
                let dest = &mut out_row[out_range.clone()];
                let src = &in_row[in_range.clone()];
                let scale = kernel_view[[k_y, k_x]];

                let src_els = src.len().div_ceil(params.stride_w);
                debug_assert!(src_els == dest.len());

                unroll_loop!(0..src_els, i, 4, {
                    unsafe {
                        *dest.get_unchecked_mut(i) +=
                            *src.get_unchecked(i * params.stride_w) * scale;
                    }
                });
            }
        }
    }
}

impl DepthwiseConvKernel<i8, u8, i32> for GenericDepthwiseConvKernel {
    fn compute_row(
        &self,
        params: &ConvParams,
        out_row: &mut [MaybeUninit<i32>],
        out_y: usize,
        out_init: i32,
        image_chan: &[i8],
        kernel_chan: NdTensorView<u8, 2>,
        input_zero: i8,
        kernel_zero: u8,
    ) {
        let kernel_view = kernel_chan.weakly_checked_view();
        let input_zero = input_zero as i32;
        let kernel_zero = kernel_zero as i32;

        // Initialize output row.
        for x in out_row.iter_mut() {
            x.write(out_init);
        }
        let out_row: &mut [i32] = unsafe { out_row.assume_init() };

        for k_y in 0..params.kernel_h {
            let in_y = out_y * params.stride_h + k_y * params.dilation_y;
            if in_y < params.pad_top || in_y >= params.in_h + params.pad_top {
                continue;
            }

            let in_row_y = in_y - params.pad_top;
            let in_row = &image_chan[in_row_y * params.in_row_stride..][..params.in_row_len];

            for (k_x, (in_range, out_range)) in params.col_range_for_kernel_x.iter().enumerate() {
                let dest = &mut out_row[out_range.clone()];
                let src = &in_row[in_range.clone()];
                let scale = kernel_view[[k_y, k_x]] as i32 - kernel_zero;

                let src_els = src.len().div_ceil(params.stride_w);
                debug_assert!(src_els == dest.len());

                unroll_loop!(0..src_els, i, 4, {
                    unsafe {
                        *dest.get_unchecked_mut(i) +=
                            (*src.get_unchecked(i * params.stride_w) as i32 - input_zero) * scale;
                    }
                });
            }
        }
    }
}

/// Compute depthwise convolution for the block of channels from `input`
/// specified by `chan_range` into `output`.
///
/// Convolutions where the group count is equal to the input and output
/// channel count are referred to as _depthwise_. These use a different
/// implementation than general convolutions for efficiency.
///
/// `X`, `W` and `Y` are the input, weight and output data types respectively.
pub struct DepthwiseConvExecutor<X: Copy, W, Y: Copy + Default> {
    kernel: Box<dyn DepthwiseConvKernel<X, W, Y> + Sync>,
}

impl<X: Copy + Default + Sync, W: Copy + Default + Sync, Y: Copy + Default>
    DepthwiseConvExecutor<X, W, Y>
{
    /// Compute depthwise convolution for the block of channels from `input`
    /// specified by `chan_range` into `output`.
    ///
    /// `col_range_for_kernel_x` is a precomputed map of kernel X coordinate to
    /// `(in_range, out_range)` of column ranges that are valid for the input and
    /// output.
    ///
    /// When this function returns, all elements of `output` will have been
    /// initialized.
    fn depthwise_conv_2d_block(
        &self,
        mut output: NdTensorViewMut<MaybeUninit<Y>, 3>, // C, H, W
        chan_range: Range<usize>,
        input: NdTensorView<X, 3>,  // C, H, W
        kernel: NdTensorView<W, 4>, // C, _, Kh, Kw
        bias: Option<NdTensorView<Y, 1>>,
        padding: [usize; 4],
        strides: [usize; 2],
        dilations: [usize; 2],
        col_range_for_kernel_x: &[(Range<usize>, Range<usize>)],
        input_zero: Option<X>,
        kernel_zero: Option<&[W]>,
    ) {
        debug_assert_eq!(input.stride(2), 1, "last dim of input is not contiguous");
        debug_assert_eq!(output.stride(2), 1, "last dim of output is not contiguous");

        let [_, out_h, _out_w] = output.shape();
        let [_, _, k_h, _k_w] = kernel.shape();
        let [_, in_h, _in_w] = input.shape();
        let [stride_h, stride_w] = strides;
        let [pad_top, _pad_left, _pad_bottom, _pad_right] = padding;
        let [dilation_y, _dilation_x] = dilations;

        let conv_params = ConvParams {
            stride_h,
            stride_w,
            kernel_h: k_h,
            pad_top,
            dilation_y,
            in_h,
            in_row_stride: input.stride(1),
            in_row_len: input.size(2),
            col_range_for_kernel_x,
        };

        let input_zero = input_zero.unwrap_or_default();
        for c in chan_range.clone() {
            let kernel_view = kernel.slice([c, 0]);

            let mut out_chan = output.slice_mut([c - chan_range.start]);
            let out_row_stride = out_chan.stride(0);
            let out_row_len = out_chan.size(1);
            let out_chan_data = out_chan.data_mut().unwrap();

            let in_chan = input.slice([c]);
            let in_chan_data = in_chan.data().unwrap();

            let init_value = if let Some(bias) = bias {
                bias[c]
            } else {
                Y::default()
            };

            let kernel_zero = kernel_zero.map(|kz| kz[c]).unwrap_or_default();

            for out_y in 0..out_h {
                // Here and in the `compute_row` implementation we use manual
                // slicing of the input data rather than tensor `slice` methods
                // for efficiency.
                let out_row = &mut out_chan_data[out_y * out_row_stride..][..out_row_len];
                self.kernel.compute_row(
                    &conv_params,
                    out_row,
                    out_y,
                    init_value,
                    in_chan_data,
                    kernel_view,
                    input_zero,
                    kernel_zero,
                );
            }
        }
    }

    /// Compute a depthwise convolution of a 2D image.
    pub fn depthwise_conv_2d(
        &self,
        pool: &BufferPool,
        input: &NdTensorView<X, 4>,
        kernel: &NdTensorView<W, 4>,
        bias: Option<NdTensorView<Y, 1>>,
        padding: [usize; 4],
        strides: [usize; 2],
        dilations: [usize; 2],
        out_hw: [usize; 2],
        input_zero: Option<X>,
        kernel_zero: Option<&[W]>,
    ) -> NdTensor<Y, 4> {
        let [batch, _in_c, _in_h, in_w]: [usize; 4] = input.shape();
        let [out_c, _, _k_h, k_w]: [usize; 4] = kernel.shape();
        let [_pad_top, pad_left, _pad_bottom, _pad_right] = padding;
        let [_stride_h, stride_w] = strides;
        let [_dilation_y, dilation_x] = dilations;
        let [out_h, out_w] = out_hw;

        let mut output = NdTensor::uninit_in(pool, [batch, out_c, out_h, out_w]);

        // `conv_2d_depthwise_block` assumes contiguous last dimension.
        let input = input.to_contiguous_in(pool).auto_return(pool);

        // Map of kernel X position to `(in_range, out_range)` of column ranges that
        // are used in the inner loop.
        let col_range_for_kernel_x: SmallVec<[_; 7]> = (0..k_w)
            .map(|k_x| {
                let (min_out_x, max_out_x) =
                    min_max_out_x_coords(k_x, in_w, pad_left, stride_w, dilation_x, out_w);

                // If the output range is empty, the input range must be too. Exit
                // early in case `max_out_x` is zero.
                //
                // This can happen if all the input coordinates which this kernel
                // column would be multiplied with are part of the padding region.
                if min_out_x == max_out_x {
                    return (0..0, 0..0);
                }

                let min_in_x = min_out_x * stride_w + k_x * dilation_x - pad_left;
                let max_in_x = (max_out_x - 1) * stride_w + k_x * dilation_x - pad_left + 1;

                (min_in_x..max_in_x, min_out_x..max_out_x)
            })
            .collect();

        let n_init = AtomicUsize::new(0);
        for n in 0..batch {
            let mut out_chans = output.slice_mut(n);
            let input = input.slice(n);

            out_chans
                .axis_iter_mut(0)
                .into_par_iter()
                .enumerate()
                .for_each(|(out_chan, mut out_chans)| {
                    let mut out_chans = out_chans
                        .reshaped_mut([1, out_chans.size(0), out_chans.size(1)])
                        .unwrap();
                    self.depthwise_conv_2d_block(
                        out_chans.view_mut(),
                        out_chan..out_chan + 1,
                        input,
                        kernel.view(),
                        bias,
                        padding,
                        strides,
                        dilations,
                        &col_range_for_kernel_x,
                        input_zero,
                        kernel_zero,
                    );

                    n_init.fetch_add(out_chans.len(), Ordering::SeqCst);
                });
        }

        // Safety: We initialized all output rows
        assert!(n_init.load(Ordering::SeqCst) == output.len());
        unsafe { output.assume_init() }
    }
}

impl Default for DepthwiseConvExecutor<f32, f32, f32> {
    fn default() -> Self {
        DepthwiseConvExecutor {
            kernel: Box::new(GenericDepthwiseConvKernel {}),
        }
    }
}

impl Default for DepthwiseConvExecutor<i8, u8, i32> {
    fn default() -> Self {
        DepthwiseConvExecutor {
            kernel: Box::new(GenericDepthwiseConvKernel {}),
        }
    }
}

// nb. Tests for depthwise conv are implemented in the main `conv.rs` module.
