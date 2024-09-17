use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut, Tensor};
use smallvec::SmallVec;

use crate::iter_util::{range_chunks, unroll_loop};
use crate::tensor_pool::{AutoReturn, TensorPool};

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

/// Compute depthwise convolution for the block of channels from `input`
/// specified by `chan_range` into `output`.
///
/// `col_range_for_kernel_x` is a precomputed map of kernel X coordinate to
/// `(in_range, out_range)` of column ranges that are valid for the input and
/// output.
///
/// When this function returns, all elements of `output` will have been
/// initialized.
fn conv_2d_depthwise_block<X, W, Y>(
    mut output: NdTensorViewMut<MaybeUninit<Y>, 3>, // C, H, W
    chan_range: Range<usize>,
    input: NdTensorView<X, 3>,  // C, H, W
    kernel: NdTensorView<W, 4>, // C, _, Kh, Kw
    bias: Option<NdTensorView<Y, 1>>,
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
    col_range_for_kernel_x: &[(Range<usize>, Range<usize>)],
) where
    X: Copy + std::ops::Mul<W, Output = Y>,
    W: Copy,
    Y: Copy + Default + std::ops::AddAssign<Y>,
{
    debug_assert_eq!(input.stride(2), 1, "last dim of input is not contiguous");
    debug_assert_eq!(output.stride(2), 1, "last dim of output is not contiguous");

    let [_, out_h, _out_w] = output.shape();
    let [_, _, k_h, _k_w] = kernel.shape();
    let [_, in_h, _in_w] = input.shape();
    let [stride_h, stride_w] = strides;
    let [pad_top, _pad_left, _pad_bottom, _pad_right] = padding;
    let [dilation_y, _dilation_x] = dilations;

    for c in chan_range.clone() {
        let kernel_view = kernel.slice_with([c, 0]).weakly_checked_view();

        // For efficiency, use manual slicing in the inner loops to extract
        // input/output rows.
        let mut out_chan = output.slice_with_mut([c - chan_range.start]);
        let out_row_stride = out_chan.stride(0);
        let out_row_len = out_chan.size(1);
        let out_chan_data = out_chan.data_mut().unwrap();

        let in_chan = input.slice_with([c]);
        let in_row_stride = in_chan.stride(0);
        let in_row_len = in_chan.size(1);
        let in_chan_data = in_chan.data().unwrap();

        let init_value = if let Some(bias) = bias {
            bias[[c]]
        } else {
            Y::default()
        };

        // The loops here are ordered so that the inner-most loop is as
        // efficient as possible and runs for as long as possible over a
        // contiguous slice of memory.
        for out_y in 0..out_h {
            let out_row = &mut out_chan_data[out_y * out_row_stride..][..out_row_len];

            // Initialize output row.
            for x in out_row.iter_mut() {
                x.write(init_value);
            }
            let out_row: &mut [Y] = unsafe { std::mem::transmute(out_row) };

            for k_y in 0..k_h {
                let in_y = out_y * stride_h + k_y * dilation_y;
                if in_y < pad_top || in_y >= in_h + pad_top {
                    continue;
                }

                let in_row_y = in_y - pad_top;
                let in_row = &in_chan_data[in_row_y * in_row_stride..][..in_row_len];

                for (k_x, (in_range, out_range)) in col_range_for_kernel_x.iter().enumerate() {
                    let dest = &mut out_row[out_range.clone()];
                    let src = &in_row[in_range.clone()];
                    let scale = kernel_view[[k_y, k_x]];

                    let src_els = src.len().div_ceil(stride_w);
                    debug_assert!(src_els == dest.len());

                    unroll_loop!(0..src_els, i, 4, {
                        unsafe {
                            *dest.get_unchecked_mut(i) += *src.get_unchecked(i * stride_w) * scale;
                        }
                    });
                }
            }
        }
    }
}

/// Specialization of 2D convolution for depthwise convolutions.
///
/// Depthwise convolutions operate over a single input/output channel at
/// a time and hence the transformation of convolution to matrix multiplication
/// doesn't pay off. An optimized direct method works better.
pub fn conv_2d_depthwise<X, W, Y>(
    pool: &TensorPool,
    input: &NdTensorView<X, 4>,
    kernel: &NdTensorView<W, 4>,
    bias: Option<NdTensorView<Y, 1>>,
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
    out_hw: [usize; 2],
) -> Tensor<Y>
where
    X: Copy + std::ops::Mul<W, Output = Y>,
    W: Copy,
    Y: Copy + Default + std::ops::AddAssign<Y>,
{
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

    // Minimum number of elements in a channel chunk.
    let target_chunk_size = 32 * 1024;
    let channel_chunk_size = (target_chunk_size / (out_h * out_w)).clamp(1, out_c);

    let n_init = AtomicUsize::new(0);
    for n in 0..batch {
        let mut out_chans = output.slice_with_mut(n);
        let input = input.slice_with(n);

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

// nb. Tests for depthwise conv are implemented in the main `conv.rs` module.
