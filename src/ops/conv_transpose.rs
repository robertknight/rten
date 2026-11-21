use std::mem::MaybeUninit;

use rayon::prelude::*;
use rten_base::iter::range_chunks;
use rten_base::num::div_ceil;
use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmUninitOptions, PackedAMatrix};
use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut, Tensor, TensorView};
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext, PrepackedInput, check_eq, static_dims,
};
use crate::ops::Padding;
use crate::ops::conv::{conv_1x1_dynamic_quant, pack_weights};
use crate::value::ValueView;

/// Compute the range of input positions along a spatial axis that result in
/// valid output positions for a col2im operation.
///
/// The input and output positions are related by:
///
///    out_x = in_x * stride + kernel_pos
///
/// Output positions are valid where:
///
///    out_x >= pad_start && out_x < output_size + pad_start
///
/// Input positions are also constrained to [0, input_size).
fn col2im_input_range(
    input_size: usize,
    output_size: usize,
    pad_start: usize,
    kernel_pos: usize,
    stride: usize,
) -> std::ops::Range<usize> {
    // Compute with signed values to avoid underflow in subtraction.
    let input_size = input_size as isize;
    let output_size = output_size as isize;
    let pad_start = pad_start as isize;
    let kernel_pos = kernel_pos as isize;
    let stride = stride as isize;

    // out_x >= pad_start
    // => in_x * stride + kernel_pos >= pad_start
    // => in_x * stride >= pad_start - kernel_pos
    // => in_x >= ⌈(pad_start - kernel_pos) / stride⌉
    let x_start = div_ceil(pad_start - kernel_pos, stride).max(0);

    // out_x < output_size + pad_start
    // => in_x * stride + kernel_pos < out_size + pad_start
    // => in_x * stride < out_size + pad_start - kernel_pos
    // => in_x < ⌈(out_size + pad_start - kernel_pos) / stride⌉
    let x_end = div_ceil(output_size + pad_start - kernel_pos, stride).min(input_size);

    if x_start > x_end {
        return 0..0;
    }

    x_start as usize..x_end as usize
}

/// Unpack columns of a matrix into an image. This is the inverse of the
/// `im2col` operation.
///
/// `output` has shape [O,H,W] where O is the number of output channels and H/W
/// are the output height/width.
///
/// `columns` is a view of a matrix (O x Kh x Kw, Hi * Wi) reshaped to
/// [O,Kh,Kw,Hi,Wi], where Hi and Wi are the image size, and Kh/Kw are the patch
/// sizes. This matrix is passed as a 5D view to avoid needing to pass the
/// sub-dimensions separately.
///
/// `bias` is a vector of per-channel biases.
///
/// Each channel of the output image is initialized with the corresponding bias
/// or zero, and then the unpacked columns for that channel are accumulated into
/// it.
fn col2im(
    output: &mut NdTensorViewMut<MaybeUninit<f32>, 3>,
    columns: &NdTensorView<f32, 5>,
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
    bias: Option<NdTensorView<f32, 1>>,
) {
    let [col_chans, _, _, _, _] = columns.shape();
    let [out_chans, _, _] = output.shape();
    assert!(col_chans == out_chans);

    output
        .axis_iter_mut(0)
        .into_par_iter()
        .enumerate()
        .for_each(|(out_c, mut out_img)| {
            // Initialize each output channel just before we accumulate into it.
            out_img.fill(MaybeUninit::new(bias.map(|b| b[[out_c]]).unwrap_or(0.)));

            // Safety: We just initialized all elements of `out_img`.
            let mut out_img = unsafe { out_img.assume_init() };

            col2im_channel(
                &mut out_img,
                &columns.slice(out_c),
                [0, 0],
                padding,
                strides,
                dilations,
            );
        });
}

/// Accumulate columns of a matrix into an image, for a region of the input
/// image starting at `in_offset`.
///
/// This is like [`col2im`], except that `columns` has shape
/// `[O,Kh,Kw,Hc,Wc]`, where `Hc` and `Wc` are the size of the region, and the
/// columns are added to the existing values of `output`.
fn col2im_accumulate(
    output: &mut NdTensorViewMut<f32, 3>,
    columns: &NdTensorView<f32, 5>,
    in_offset: [usize; 2],
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
) {
    let [col_chans, _, _, _, _] = columns.shape();
    let [out_chans, _, _] = output.shape();
    assert!(col_chans == out_chans);

    output
        .axis_iter_mut(0)
        .into_par_iter()
        .enumerate()
        .for_each(|(out_c, mut out_img)| {
            col2im_channel(
                &mut out_img,
                &columns.slice(out_c),
                in_offset,
                padding,
                strides,
                dilations,
            );
        });
}

/// Accumulate the columns for one output channel into `out_img`.
///
/// `columns` has shape `[Kh,Kw,Hc,Wc]` and contains the columns for a region
/// of the input image starting at `in_offset`.
fn col2im_channel(
    out_img: &mut NdTensorViewMut<f32, 2>,
    columns: &NdTensorView<f32, 4>,
    in_offset: [usize; 2],
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
) {
    let [stride_h, stride_w] = strides;
    let [dilation_h, dilation_w] = dilations;
    let [pad_top, pad_left, _pad_bottom, _pad_right] = padding;
    let [kernel_h, kernel_w, img_h, img_w] = columns.shape();
    let [out_h, out_w] = out_img.shape();
    let [y_offset, x_offset] = in_offset;

    for k_y in 0..kernel_h {
        for k_x in 0..kernel_w {
            let in_img = columns.slice([k_y, k_x]);

            // Ranges of input positions in the region which map to valid
            // output positions.
            let y_range =
                col2im_input_range(y_offset + img_h, out_h, pad_top, k_y * dilation_h, stride_h);
            let y_range = y_range.start.max(y_offset)..y_range.end.max(y_offset);
            let x_range = col2im_input_range(
                x_offset + img_w,
                out_w,
                pad_left,
                k_x * dilation_w,
                stride_w,
            );
            let x_range = x_range.start.max(x_offset)..x_range.end.max(x_offset);

            for y in y_range {
                let out_y = y * stride_h + k_y * dilation_h;
                debug_assert!(out_y >= pad_top);
                debug_assert!(out_y < out_h + pad_top);

                for x in x_range.clone() {
                    let out_x = x * stride_w + k_x * dilation_w;
                    debug_assert!(out_x >= pad_left);
                    debug_assert!(out_x < out_w + pad_left);

                    // Safety: We computed x, y, out_x and out_y such that they are
                    // in-bounds for out_img and in_img.
                    unsafe {
                        *out_img.get_unchecked_mut([out_y - pad_top, out_x - pad_left]) +=
                            in_img.get_unchecked([y - y_offset, x - x_offset]);
                    }
                }
            }
        }
    }
}

/// Perform a depthwise transposed convolution, where each group has one input
/// and one output channel.
///
/// `kernel` has shape `[C, 1, kH, kW]`. Each output channel is computed
/// directly by accumulating the input, scaled by each kernel element, into
/// the output. This avoids the overhead of a separate GEMM and col2im for
/// each channel.
#[allow(clippy::too_many_arguments)]
fn conv_transpose_depthwise(
    pool: &BufferPool,
    input: NdTensorView<f32, 4>,
    kernel: NdTensorView<f32, 4>,
    bias: Option<NdTensorView<f32, 1>>,
    out_size: [usize; 2],
    padding: [usize; 2],
    strides: [usize; 2],
    dilations: [usize; 2],
) -> NdTensor<f32, 4> {
    let [batch, chans, in_h, in_w] = input.shape();
    let [_, _, k_h, k_w] = kernel.shape();
    let [out_h, out_w] = out_size;
    let [pad_top, pad_left] = padding;
    let [stride_h, stride_w] = strides;
    let [dilation_h, dilation_w] = dilations;

    let input = input.to_contiguous_in(pool).auto_return(pool);
    let mut output = NdTensor::uninit_in(pool, [batch, chans, out_h, out_w]);

    output
        .data_mut()
        .unwrap()
        .par_chunks_mut(out_h * out_w)
        .zip(input.data().par_chunks(in_h * in_w))
        .enumerate()
        .for_each(|(batch_chan, (out_img, in_img))| {
            let chan = batch_chan % chans;
            out_img.fill(MaybeUninit::new(bias.map(|b| b[chan]).unwrap_or(0.)));

            // Safety: We just initialized all elements of `out_img`.
            let out_img = unsafe { out_img.assume_init_mut() };

            for k_y in 0..k_h {
                let y_range = col2im_input_range(in_h, out_h, pad_top, k_y * dilation_h, stride_h);
                for k_x in 0..k_w {
                    let x_range =
                        col2im_input_range(in_w, out_w, pad_left, k_x * dilation_w, stride_w);
                    let weight = kernel[[chan, 0, k_y, k_x]];

                    for y in y_range.clone() {
                        let out_y = y * stride_h + k_y * dilation_h - pad_top;
                        let in_row = &in_img[y * in_w..(y + 1) * in_w];
                        let out_row = &mut out_img[out_y * out_w..(out_y + 1) * out_w];
                        for x in x_range.clone() {
                            let out_x = x * stride_w + k_x * dilation_w - pad_left;
                            out_row[out_x] += in_row[x] * weight;
                        }
                    }
                }
            }
        });

    // Safety: Every output channel was initialized.
    unsafe { output.assume_init() }
}

/// Calculate ConvTranspose output spatial shape and padding.
///
/// See formulae in https://onnx.ai/onnx/operators/onnx__ConvTranspose.html.
///
/// Returns a tuple of (out_shape, padding).
fn conv_transpose_output_size_and_padding(
    input_shape: [usize; 2],
    kernel_shape: [usize; 2],
    padding: Padding,
    strides: [usize; 2],
    dilations: [usize; 2],
    output_padding: [usize; 2],
) -> Result<([usize; 2], [usize; 4]), OpError> {
    let [in_h, in_w] = input_shape;
    let [stride_h, stride_w] = strides;
    let [dilation_h, dilation_w] = dilations;
    let [out_pad_h, out_pad_w] = output_padding;

    if stride_h == 0 || stride_w == 0 {
        return Err(OpError::invalid_value("Strides must be > 0"));
    }

    if dilation_h == 0 || dilation_w == 0 {
        return Err(OpError::invalid_value("Dilations must be > 0"));
    }

    if kernel_shape.contains(&0) {
        return Err(OpError::invalid_value("Kernel size must be > 0"));
    }

    if in_h == 0 || in_w == 0 {
        return Err(OpError::invalid_value("Input width and height must be > 0"));
    }

    // Effective kernel size after dilation.
    let effective_kernel_size = |k: usize, dilation: usize| (k - 1) * dilation + 1;
    let k_h = effective_kernel_size(kernel_shape[0], dilation_h);
    let k_w = effective_kernel_size(kernel_shape[1], dilation_w);

    match padding {
        Padding::Same => {
            // Per spec, pad the input so that:
            // output_shape[i] = input_shape[i] * strides[i] for each axis i.
            let out_h = in_h * stride_h;
            let out_w = in_w * stride_w;

            let pad_h = ((in_h - 1) * stride_h + k_h + out_pad_h).checked_sub(out_h);
            let pad_w = ((in_w - 1) * stride_w + k_w + out_pad_w).checked_sub(out_w);

            let (Some(pad_h), Some(pad_w)) = (pad_h, pad_w) else {
                // We can't achieve an output size of (out_h, out_w) even with
                // no padding.
                return Err(OpError::invalid_value("Input is too small"));
            };

            // If the total padding is not even, we assign the remaining unit to
            // the ends of the axis. This matches the ONNX "SAME_UPPER"
            // value for `auto_pad`.
            let pad_top = pad_h / 2;
            let pad_bottom = pad_h.div_ceil(2);
            let pad_left = pad_w / 2;
            let pad_right = pad_w.div_ceil(2);

            Ok(([out_h, out_w], [pad_top, pad_left, pad_bottom, pad_right]))
        }
        Padding::Fixed(pads) => match pads.as_slice() {
            &[pad_top, pad_left, pad_bottom, pad_right] => {
                let out_h =
                    ((in_h - 1) * stride_h + out_pad_h + k_h).checked_sub(pad_top + pad_bottom);
                let out_w =
                    ((in_w - 1) * stride_w + out_pad_w + k_w).checked_sub(pad_left + pad_right);

                let (Some(out_h), Some(out_w)) = (out_h, out_w) else {
                    return Err(OpError::invalid_value("Input is too small"));
                };

                Ok(([out_h, out_w], [pad_top, pad_left, pad_bottom, pad_right]))
            }
            _ => Err(OpError::invalid_value("Wrong number of pad values")),
        },
    }
}

/// Perform a transposed 2D convolution of a tensor by a kernel.
///
/// `input` has dimensions NCHW and `kernel` has dimensions COHW where `O` is
/// the number of output channels.
pub fn conv_transpose(
    pool: &BufferPool,
    input: TensorView,
    kernel: TensorView,
    bias: Option<NdTensorView<f32, 1>>,
    padding: Padding,
    groups: usize,
    strides: &[usize],
    dilations: &[usize],
    output_padding: Option<&[usize]>,
) -> Result<Tensor, OpError> {
    conv_transpose_impl(
        pool,
        input,
        kernel,
        None,
        bias,
        padding,
        groups,
        strides,
        dilations,
        output_padding,
    )
}

/// Pack the kernel of a transposed convolution with one group.
///
/// `kernel` has shape `[C_in, C_out, ...]`. The result can be passed to
/// [`conv_transpose_impl`] in place of the kernel matrix it computes.
fn pack_kernel(kernel: TensorView) -> Option<PackedAMatrix<f32>> {
    let in_c = *kernel.shape().first()?;
    let kernel_mat = kernel.reshaped([in_c, kernel.len() / in_c.max(1)]);
    let gemm = GemmExecutor::new();
    Some(gemm.prepack_a(kernel_mat.transposed()))
}

/// Perform a transposed convolution.
///
/// If `packed_kernel` is provided, it is used instead of `kernel` for the
/// matrix multiplication when there is one group. It must be the result of
/// [`pack_kernel`] for `kernel`.
#[allow(clippy::too_many_arguments)]
fn conv_transpose_impl(
    pool: &BufferPool,
    input: TensorView,
    kernel: TensorView,
    packed_kernel: Option<&PackedAMatrix<f32>>,
    bias: Option<NdTensorView<f32, 1>>,
    padding: Padding,
    groups: usize,
    strides: &[usize],
    dilations: &[usize],
    output_padding: Option<&[usize]>,
) -> Result<Tensor, OpError> {
    // Handle 1D transposed convolution by expanding to 2D and then removing
    // the extra dimension from the result.
    if let &[n, c, w] = input.shape() {
        let [out_c, k_in_c, k_w] = static_dims!(kernel, 3, "OCW")?.shape();

        let input_2d = input
            .reshaped_in(pool, [n, c, 1, w].as_slice())
            .auto_return(pool);
        let kernel_2d = kernel
            .reshaped_in(pool, [out_c, k_in_c, 1, k_w].as_slice())
            .auto_return(pool);

        let padding_2d = padding.expand_1d_to_2d()?;

        let strides_2d = match strides {
            &[stride] => [1, stride],
            _ => {
                return Err(OpError::invalid_value("expected 1 stride value"));
            }
        };

        let dilations_2d = match dilations {
            &[dilation] => [1, dilation],
            _ => {
                return Err(OpError::invalid_value("expected 1 dilation value"));
            }
        };

        let output_padding_2d = match output_padding {
            Some(&[pad]) => [0, pad],
            None => [0, 0],
            _ => {
                return Err(OpError::invalid_value("expected 1 output_padding value"));
            }
        };

        let result_2d = conv_transpose_impl(
            pool,
            input_2d.view(),
            kernel_2d.view(),
            packed_kernel,
            bias,
            padding_2d,
            groups,
            &strides_2d,
            &dilations_2d,
            Some(&output_padding_2d),
        );

        return result_2d.map(|mut t| {
            let [n, c, _h, w]: [usize; 4] = t.shape().try_into().expect("expected 4D output");
            t.reshape(&[n, c, w]);
            t
        });
    }

    if groups == 0 {
        return Err(OpError::invalid_value("Group count must be > 0"));
    }

    let input = static_dims!(input, 4, "NCHW")?;
    let [batch, in_c, in_h, in_w] = input.shape();
    let kernel = static_dims!(kernel, 4, "COHW")?;
    let [k_in_c, out_chans_per_group, k_h, k_w] = kernel.shape();
    let out_channels = out_chans_per_group * groups;

    if let Some(bias) = bias {
        check_eq!(bias.size(0), out_channels)?;
    }

    let bias = bias.map(|b| b.nd_view());

    if in_c != k_in_c {
        return Err(OpError::incompatible_input_shapes(
            "Input channels does not match kernel input channels",
        ));
    }

    if k_in_c % groups != 0 {
        return Err(OpError::invalid_value(
            "Input channel count not divisible by groups",
        ));
    }

    let &[stride_h, stride_w] = strides else {
        return Err(OpError::invalid_value("expected 2 stride values"));
    };
    let &[dilation_h, dilation_w] = dilations else {
        return Err(OpError::invalid_value("expected 2 dilation values"));
    };
    let [out_pad_h, out_pad_w] = match output_padding {
        Some(&[h, w]) => [h, w],
        None => [0, 0],
        _ => {
            return Err(OpError::invalid_value("expected 2 output_padding values"));
        }
    };

    let (out_shape, fixed_padding) = conv_transpose_output_size_and_padding(
        [in_h, in_w],
        [k_h, k_w],
        padding,
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        [out_pad_h, out_pad_w],
    )?;
    let [out_h, out_w] = out_shape;
    let [pad_top, pad_left, pad_bottom, pad_right] = fixed_padding;
    let out_c = out_chans_per_group * groups;

    // Use a direct implementation for depthwise transposed convolutions.
    if k_in_c == groups && out_chans_per_group == 1 {
        let output = conv_transpose_depthwise(
            pool,
            input,
            kernel,
            bias,
            [out_h, out_w],
            [pad_top, pad_left],
            [stride_h, stride_w],
            [dilation_h, dilation_w],
        );
        return Ok(output.into_dyn());
    }

    let mut output = NdTensor::uninit_in(pool, [batch, out_c, out_h, out_w]);

    let mut col2im_mat =
        NdTensor::uninit_in(pool, [out_chans_per_group * k_h * k_w, in_h * in_w]).auto_return(pool);
    let kernel_mat = kernel
        .reshaped_in(pool, [k_in_c, out_chans_per_group * k_h * k_w])
        .auto_return(pool);
    let kernel_mat = kernel_mat.transposed();
    let gemm = GemmExecutor::new();

    let in_chans_per_group = k_in_c / groups;

    // The implementation here is the inverse of the im2col-based convolution.
    let mut n_init = 0;

    for (in_chans, out_chans) in
        range_chunks(0..in_c, in_chans_per_group).zip(range_chunks(0..out_c, out_chans_per_group))
    {
        let in_group = input.slice((.., in_chans.clone()));
        let mut out_group = output.slice_mut((.., out_chans.clone()));
        let kernel_mat = kernel_mat.slice((.., in_chans.clone()));
        let bias = bias.map(|b| b.slice(out_chans.clone()));

        for n in 0..batch {
            let input_mat = in_group
                .slice([n])
                .reshaped_in(pool, [in_chans.len(), in_h * in_w])
                .auto_return(pool);

            let col2im_shape = col2im_mat.shape();
            let col2im_init = gemm
                .gemm_uninit(
                    col2im_mat.data_mut().unwrap(),
                    match packed_kernel {
                        Some(packed) if groups == 1 => GemmInputA::Packed(packed),
                        _ => GemmInputA::Unpacked(kernel_mat),
                    },
                    GemmInputB::Unpacked(input_mat.view()),
                    GemmUninitOptions::default(),
                )
                .unwrap();

            let col2im_mat = NdTensorView::from_data(
                col2im_shape,
                // False positive. The conversion from `&mut [f32]` -> `&[f32]` here
                // is necessary.
                #[allow(clippy::useless_asref)]
                col2im_init.as_ref(),
            );
            let mut out_img = out_group.slice_mut(n);
            col2im(
                &mut out_img,
                &col2im_mat
                    .reshaped([out_chans.len(), k_h, k_w, in_h, in_w])
                    .view(),
                [pad_top, pad_left, pad_right, pad_bottom],
                [stride_h, stride_w],
                [dilation_h, dilation_w],
                bias,
            );
            n_init += out_img.len();
        }
    }

    assert!(n_init == output.len());
    let output = unsafe { output.assume_init() };
    Ok(output.into_dyn())
}

#[derive(Debug)]
pub struct ConvTranspose {
    pub groups: usize,
    pub dilations: Vec<usize>,
    pub padding: Padding,
    pub strides: Vec<usize>,
    pub output_padding: Option<Vec<usize>>,
}

impl Operator for ConvTranspose {
    fn name(&self) -> &str {
        "ConvTranspose"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let weight = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;
        let packed_weight = match inputs.get_prepacked(1) {
            Some(PrepackedInput::FloatAMatrix(packed)) => Some(packed),
            _ => None,
        };

        conv_transpose_impl(
            ctx.pool(),
            input,
            weight,
            packed_weight,
            bias,
            self.padding.clone(),
            self.groups,
            &self.strides,
            &self.dilations,
            self.output_padding.as_deref(),
        )
        .into_op_result()
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        if self.groups == 1 {
            [1].into()
        } else {
            SmallVec::new()
        }
    }

    fn prepack(
        &self,
        index: usize,
        input: ValueView,
        _zero_point: Option<ValueView>,
    ) -> Option<PrepackedInput> {
        if index != 1 || self.groups != 1 {
            return None;
        }
        let kernel: TensorView<f32> = input.try_into().ok()?;
        pack_kernel(kernel).map(PrepackedInput::FloatAMatrix)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    ConvTranspose,
    op,
    shape_ops::ConvTranspose {
        groups: op.groups,
        padding: op.padding.as_shape_inference_padding(),
        strides: &op.strides,
        dilations: &op.dilations,
        output_padding: op.output_padding.as_deref(),
    }
);

/// Transposed convolution with int8 weights and dynamically quantized
/// activations.
///
/// Inputs are `X, W, W_scale, B`, where `W` is an int8 weight tensor with
/// shape `[C, O, ...kernel]`, `W_scale` is a scale with shape `[O * prod(kernel)]`
/// for each output channel and kernel position and `B` is an optional bias of
/// shape `[O]`. Only `groups = 1` is supported.
///
/// Activations are quantized with a separate scale and zero point for each
/// input position. This is not an ONNX operator. It is created from
/// `ConvTranspose` operators with `DequantizeLinear` weights when dynamic
/// quantization of convolutions is enabled.
#[derive(Clone, Debug)]
pub struct ConvTransposeDynamicQuant {
    pub dilations: Vec<usize>,
    pub padding: Padding,
    pub strides: Vec<usize>,
    pub output_padding: Option<Vec<usize>>,
}

impl ConvTransposeDynamicQuant {
    /// Return contiguous weights as a `[O * prod(kernel), C]` matrix.
    fn weight_matrix<'a>(weight: &TensorView<'a, i8>) -> Option<NdTensorView<'a, i8, 2>> {
        let in_c = *weight.shape().first()?;
        let mat = NdTensorView::from_data([in_c, weight.len() / in_c.max(1)], weight.data()?);
        Some(mat.transposed())
    }
}

impl Operator for ConvTransposeDynamicQuant {
    fn name(&self) -> &str {
        "ConvTransposeDynamicQuant"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let pool = ctx.pool();
        let inputs = ctx.inputs();
        let input: TensorView<f32> = inputs.require_as(0)?;
        let weight: TensorView<i8> = inputs.require_as(1)?;
        let weight_scale: NdTensorView<f32, 1> = inputs.require_as(2)?;
        let bias: Option<NdTensorView<f32, 1>> = inputs.get_as(3)?;
        let packed_weight = match inputs.get_prepacked(1) {
            Some(PrepackedInput::UInt8AMatrix(packed)) => Some(packed),
            _ => None,
        };

        // Expand 1D transposed convolutions to 2D.
        let is_1d = input.ndim() == 3;
        let (input, weight, padding, strides, dilations, output_padding) = if is_1d {
            let mut input = input;
            input.insert_axis(2);
            let mut weight = weight;
            weight.insert_axis(2);
            let (&[stride], &[dilation]) = (self.strides.as_slice(), self.dilations.as_slice())
            else {
                return Err(OpError::invalid_value(
                    "expected 1 stride and dilation value",
                ));
            };
            let out_pad = match self.output_padding.as_deref() {
                Some(&[pad]) => pad,
                None => 0,
                _ => return Err(OpError::invalid_value("expected 1 output_padding value")),
            };
            let padding = self.padding.expand_1d_to_2d()?;
            (
                input,
                weight,
                padding,
                [1, stride],
                [1, dilation],
                [0, out_pad],
            )
        } else {
            let (&[stride_h, stride_w], &[dilation_h, dilation_w]) =
                (self.strides.as_slice(), self.dilations.as_slice())
            else {
                return Err(OpError::invalid_value(
                    "expected 2 stride and dilation values",
                ));
            };
            let output_padding = match self.output_padding.as_deref() {
                Some(&[h, w]) => [h, w],
                None => [0, 0],
                _ => return Err(OpError::invalid_value("expected 2 output_padding values")),
            };
            (
                input,
                weight,
                self.padding.clone(),
                [stride_h, stride_w],
                [dilation_h, dilation_w],
                output_padding,
            )
        };

        let input = static_dims!(input, 4, "NCHW")?;
        let [batch, in_c, in_h, in_w] = input.shape();
        let [k_in_c, out_c, k_h, k_w] = static_dims!(weight, 4, "COHW")?.shape();
        check_eq!(in_c, k_in_c, "input channels do not match weights")?;
        check_eq!(
            weight_scale.size(0),
            out_c * k_h * k_w,
            "weight scale length"
        )?;
        if let Some(bias) = &bias {
            check_eq!(bias.size(0), out_c, "bias length")?;
        }

        let (out_shape, fixed_padding) = conv_transpose_output_size_and_padding(
            [in_h, in_w],
            [k_h, k_w],
            padding,
            strides,
            dilations,
            output_padding,
        )?;
        let [out_h, out_w] = out_shape;
        let [pad_top, pad_left, pad_bottom, pad_right] = fixed_padding;

        let weight = weight.to_contiguous_in(pool).auto_return(pool);
        let weight_mat = Self::weight_matrix(&weight.view()).unwrap();
        let weight_scale = weight_scale.to_contiguous_in(pool).auto_return(pool);

        // Initialize each output channel with its bias.
        let mut output = NdTensor::uninit_in(pool, [batch, out_c, out_h, out_w]);
        for mut image in output.axis_iter_mut(0) {
            image
                .axis_iter_mut(0)
                .into_par_iter()
                .enumerate()
                .for_each(|(c, mut chan)| {
                    chan.fill(MaybeUninit::new(bias.map(|b| b[c]).unwrap_or(0.)));
                });
        }
        // Safety: Every channel of each image was initialized.
        let mut output = unsafe { output.assume_init() };

        // Compute the columns for regions of the input with an int8 matrix
        // multiplication, and accumulate them into the output. Splitting the
        // input limits the size of the column buffer, which can be reused for
        // each region.
        // Regions also have a minimum size, so that each matrix multiplication
        // can be divided efficiently between threads.
        const COLUMN_BUF_BYTES: usize = 4 * 1024 * 1024;
        const MIN_POSITIONS: usize = 1024;
        let positions =
            (COLUMN_BUF_BYTES / (weight_mat.size(0) * size_of::<f32>()).max(1)).max(MIN_POSITIONS);
        let (region_h, region_w) = if positions >= in_w {
            (positions / in_w, in_w)
        } else {
            (1, positions)
        };
        for n in 0..batch {
            for y in range_chunks(0..in_h, region_h) {
                for x in range_chunks(0..in_w, region_w) {
                    let region = input.slice((n..n + 1, .., y.clone(), x.clone()));
                    let columns = conv_1x1_dynamic_quant(
                        pool,
                        region,
                        weight_mat,
                        weight_scale.data(),
                        packed_weight,
                    )?
                    .auto_return(pool);
                    col2im_accumulate(
                        &mut output.slice_mut(n),
                        &columns
                            .slice(0)
                            .reshaped([out_c, k_h, k_w, y.len(), x.len()])
                            .view(),
                        [y.start, x.start],
                        [pad_top, pad_left, pad_right, pad_bottom],
                        strides,
                        dilations,
                    );
                }
            }
        }

        let mut output = output.into_dyn();
        if is_1d {
            output.reshape(&[batch, out_c, out_w]);
        }
        output.into_op_result()
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        [1].into()
    }

    fn prepack(
        &self,
        index: usize,
        input: ValueView,
        _zero_point: Option<ValueView>,
    ) -> Option<PrepackedInput> {
        if index != 1 {
            return None;
        }
        let weight: TensorView<i8> = input.try_into().ok()?;
        let weight_mat = Self::weight_matrix(&weight)?;
        let packed = pack_weights(&BufferPool::new(), weight_mat).ok()?;
        Some(PrepackedInput::UInt8AMatrix(packed))
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    ConvTransposeDynamicQuant,
    op,
    shape_ops::ConvTranspose {
        groups: 1,
        padding: op.padding.as_shape_inference_padding(),
        strides: &op.strides,
        dilations: &op.dilations,
        output_padding: op.output_padding.as_deref(),
    }
);

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{ExpectEqualError, expect_equal};
    use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};
    use rten_testing::TestCases;

    use rten_gemm::GemmExecutor;

    use super::{
        ConvTranspose, ConvTransposeDynamicQuant, conv_transpose,
        conv_transpose_output_size_and_padding,
    };
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpRunContext, Operator, OperatorExt, OutputMask};
    use crate::ops::{OpError, Padding};

    fn reference_conv_transpose(
        input: TensorView,
        kernel: TensorView,
        bias: Option<NdTensorView<f32, 1>>,
        padding: Padding,
        groups: usize,
        [stride_h, stride_w]: [usize; 2],
        [dilation_h, dilation_w]: [usize; 2],
        [out_pad_h, out_pad_w]: [usize; 2],
    ) -> Result<Tensor, OpError> {
        let input = input.nd_view::<4>();
        let kernel = kernel.nd_view::<4>();

        let [batch, _in_c, in_h, in_w] = input.shape();
        let [k_in_c, out_chans_per_group, k_h, k_w] = kernel.shape();
        let ([out_h, out_w], fixed_padding) = conv_transpose_output_size_and_padding(
            [in_h, in_w],
            [k_h, k_w],
            padding,
            [stride_h, stride_w],
            [dilation_h, dilation_w],
            [out_pad_h, out_pad_w],
        )?;
        let out_c = out_chans_per_group * groups;
        let in_chans_per_group = k_in_c / groups;
        let mut output = NdTensor::zeros([batch, out_c, out_h, out_w]);

        let [pad_top, pad_left, _pad_bottom, _pad_right] = fixed_padding;

        for group in 0..groups {
            let in_chan_start = group * in_chans_per_group;
            let in_chan_end = in_chan_start + in_chans_per_group;
            let out_chan_start = group * out_chans_per_group;
            let out_chan_end = out_chan_start + out_chans_per_group;

            for n in 0..batch {
                for out_c in out_chan_start..out_chan_end {
                    for y in 0..out_h {
                        for x in 0..out_w {
                            let mut accum = 0.;

                            for in_chan in in_chan_start..in_chan_end {
                                for k_y in 0..k_h {
                                    for k_x in 0..k_w {
                                        // Position of the kernel element in
                                        // the output, offset by padding.
                                        let out_k_y = k_y * dilation_h;
                                        let out_k_x = k_x * dilation_w;
                                        if y + pad_top >= out_k_y
                                            && x + pad_left >= out_k_x
                                            && (y + pad_top - out_k_y) % stride_h == 0
                                            && (x + pad_left - out_k_x) % stride_w == 0
                                        {
                                            let in_y = (y + pad_top - out_k_y) / stride_h;
                                            let in_x = (x + pad_left - out_k_x) / stride_w;
                                            accum += input
                                                .get([n, in_chan, in_y, in_x])
                                                .copied()
                                                .unwrap_or(0.)
                                                * kernel
                                                    [[in_chan, out_c - out_chan_start, k_y, k_x]];
                                        }
                                    }
                                }
                            }

                            output[[n, out_c, y, x]] =
                                accum + bias.as_ref().map(|b| b[[out_c]]).unwrap_or(0.);
                        }
                    }
                }
            }
        }

        Ok(output.into())
    }

    /// Perform a transposed convolution using the optimized and reference
    /// implementations and check that the results are approximately equal.
    fn check_conv_transpose(
        input: TensorView<f32>,
        kernel: TensorView<f32>,
        bias: Option<NdTensorView<f32, 1>>,
        pads: Padding,
        groups: usize,
        strides: [usize; 2],
        dilations: [usize; 2],
        output_padding: [usize; 2],
    ) -> Result<Tensor<f32>, ExpectEqualError> {
        let pool = BufferPool::new();
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            bias,
            pads.clone(),
            groups,
            &strides,
            &dilations,
            Some(output_padding.as_slice()),
        )
        .expect("conv operation failed");
        let reference_result = reference_conv_transpose(
            input,
            kernel,
            bias,
            pads,
            groups,
            strides,
            dilations,
            output_padding,
        )
        .unwrap();
        expect_equal(&result, &reference_result)?;
        Ok(result)
    }

    #[test]
    fn test_conv_transpose() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = Tensor::from_data(&[1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);

        // Expected values computed with `torch.nn.functional.conv_transpose2d`.
        let expected = Tensor::from_data(
            &[1, 1, 4, 4],
            vec![
                0.1000, 0.2000, 0.2000, 0.4000, 0.3000, 0.4000, 0.6000, 0.8000, 0.3000, 0.6000,
                0.4000, 0.8000, 0.9000, 1.2000, 1.2000, 1.6000,
            ],
        );

        let groups = 1;
        let strides = [2, 2];
        let dilations = [1, 1];
        let output_padding = Some([0, 0].as_slice());

        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::zero::<2>(),
            groups,
            &strides,
            &dilations,
            output_padding,
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        let mut expected_with_bias = Tensor::from_data(expected.shape(), expected.to_vec());
        for eb in expected_with_bias.iter_mut() {
            *eb += 1.234;
        }
        let bias = NdTensor::from([1.234]);
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            Some(bias.view()),
            Padding::zero::<2>(),
            groups,
            &strides,
            &dilations,
            output_padding,
        )
        .unwrap();
        expect_equal(&result, &expected_with_bias)?;

        Ok(())
    }

    #[test]
    fn test_conv_transpose_padding() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = Tensor::from_data(&[1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);

        // Expected values computed with `torch.nn.functional.conv_transpose2d`.
        let expected = Tensor::from_data(&[1, 1, 2, 2], vec![0.4, 0.6, 0.6, 0.4]);
        let strides = [2, 2];
        let dilations = [1, 1];
        let groups = 1;
        let output_padding = Some([0, 0].as_slice());

        // Fixed padding. The output shape should have rows and columns
        // subtracted on each side according to the corresponding padding.
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::Fixed([1, 1, 1, 1].into()),
            groups,
            &strides,
            &dilations,
            output_padding,
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // "Same" padding. The output shape should be `input_size * stride`
        // for each spatial axis.
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::Same,
            groups,
            &strides,
            &dilations,
            output_padding,
        )
        .unwrap();
        assert_eq!(
            result.shape(),
            &[
                input.size(0),
                input.size(1),
                input.size(2) * strides[0],
                input.size(3) * strides[1]
            ]
        );

        Ok(())
    }

    #[test]
    fn test_conv_transpose_1d() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();
        let input = Tensor::from_data(&[1, 1, 2], vec![1., 2.]);
        let kernel = Tensor::from_data(&[1, 1, 2], vec![0.1, 0.2]);

        // Expected values computed with `torch.nn.functional.conv_transpose1d`.
        let expected = Tensor::from_data(&[1, 1, 4], vec![0.1, 0.2, 0.2, 0.4]);

        let groups = 1;
        let strides = [2];
        let dilations = [1];
        let output_padding = Some([0].as_slice());

        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::zero::<1>(),
            groups,
            &strides,
            &dilations,
            output_padding,
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        let bias = NdTensor::from([0.5]);
        let expected_with_bias = expected.map(|x| x + bias[[0]]);
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            Some(bias.view()),
            Padding::zero::<1>(),
            groups,
            &strides,
            &dilations,
            output_padding,
        )
        .unwrap();
        expect_equal(&result, &expected_with_bias)?;

        Ok(())
    }

    #[test]
    fn test_conv_transpose_output_size_and_padding() {
        #[derive(Debug)]
        struct Case {
            input_shape: [usize; 2],
            kernel_shape: [usize; 2],
            padding: Padding,
            strides: [usize; 2],
            dilations: [usize; 2],
            output_padding: [usize; 2],
            expected: Result<([usize; 2], [usize; 4]), OpError>,
        }

        impl Default for Case {
            fn default() -> Case {
                Case {
                    input_shape: [1, 1],
                    kernel_shape: [1, 1],
                    padding: Padding::zero::<2>(),
                    strides: [1, 1],
                    dilations: [1, 1],
                    output_padding: [0, 0],
                    expected: Err(OpError::invalid_value("default value")),
                }
            }
        }

        let cases = [
            // Zero padding, stride of 1
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [1, 1],
                expected: Ok(([7, 7], [0, 0, 0, 0])),
                ..Default::default()
            },
            // Zero padding, stride of 3
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [3, 3],
                expected: Ok(([15, 15], [0, 0, 0, 0])),
                ..Default::default()
            },
            // Non-zero padding, stride of 1
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([1, 1, 1, 1].into()),
                strides: [1, 1],
                expected: Ok(([5, 5], [1, 1, 1, 1])),
                ..Default::default()
            },
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([2, 2, 2, 2].into()),
                strides: [1, 1],
                expected: Ok(([3, 3], [2, 2, 2, 2])),
                ..Default::default()
            },
            // Uneven padding
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([1, 2, 1, 2].into()),
                strides: [1, 1],
                expected: Ok(([5, 3], [1, 2, 1, 2])),
                ..Default::default()
            },
            // Same padding
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Same,
                strides: [1, 1],
                expected: Ok(([5, 5], [1, 1, 1, 1])),
                ..Default::default()
            },
            // Same padding with non-square kernel
            Case {
                input_shape: [5, 5],
                kernel_shape: [1, 3],
                padding: Padding::Same,
                strides: [1, 1],
                expected: Ok(([5, 5], [0, 1, 0, 1])),
                ..Default::default()
            },
            // Same padding. Case where output size is smaller than
            // `input_shape * stride` even with no padding.
            Case {
                input_shape: [5, 5],
                kernel_shape: [1, 1],
                padding: Padding::Same,
                strides: [3, 3],
                expected: Err(OpError::invalid_value("Input is too small")),
                ..Default::default()
            },
            // Padding too large
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([4, 4, 4, 4].into()),
                strides: [1, 1],
                expected: Err(OpError::invalid_value("Input is too small")),
                ..Default::default()
            },
            // Dilations, stride of 1
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                dilations: [2, 2],
                expected: Ok(([9, 9], [0, 0, 0, 0])),
                ..Default::default()
            },
            // Non-uniform dilations with padding
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([1, 1, 1, 1].into()),
                dilations: [2, 3],
                expected: Ok(([7, 9], [1, 1, 1, 1])),
                ..Default::default()
            },
            // Same padding with dilations
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Same,
                dilations: [2, 2],
                expected: Ok(([5, 5], [2, 2, 2, 2])),
                ..Default::default()
            },
            // Invalid strides
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [0, 0],
                expected: Err(OpError::invalid_value("Strides must be > 0")),
                ..Default::default()
            },
            // Invalid dilations
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                dilations: [0, 0],
                expected: Err(OpError::invalid_value("Dilations must be > 0")),
                ..Default::default()
            },
            // Empty kernel
            Case {
                input_shape: [5, 5],
                kernel_shape: [0, 0],
                expected: Err(OpError::invalid_value("Kernel size must be > 0")),
                ..Default::default()
            },
            // Empty input
            Case {
                input_shape: [0, 0],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [1, 1],
                expected: Err(OpError::invalid_value("Input width and height must be > 0")),
                ..Default::default()
            },
            // Wrong padding size for input spatial shape.
            Case {
                input_shape: [1, 1],
                kernel_shape: [3, 3],
                padding: Padding::zero::<1>(),
                strides: [1, 1],
                expected: Err(OpError::invalid_value("Wrong number of pad values")),
                ..Default::default()
            },
            // Output padding on Y axis
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                output_padding: [1, 0],
                expected: Ok(([8, 7], [0, 0, 0, 0])),
                ..Default::default()
            },
            // Output padding on X axis
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                output_padding: [0, 1],
                expected: Ok(([7, 8], [0, 0, 0, 0])),
                ..Default::default()
            },
            // Output padding with padding mode `Same`
            Case {
                input_shape: [7, 7],
                kernel_shape: [3, 3],
                padding: Padding::Same,
                output_padding: [1, 1],
                strides: [2, 2],
                expected: Ok(([14, 14], [1, 1, 1, 1])),
                ..Default::default()
            },
        ];

        cases.test_each(|case| {
            let result = conv_transpose_output_size_and_padding(
                case.input_shape,
                case.kernel_shape,
                case.padding.clone(),
                case.strides,
                case.dilations,
                case.output_padding,
            );
            assert_eq!(result, case.expected);
        })
    }

    #[derive(Debug)]
    struct ConvTransposeCase {
        input_shape: [usize; 4],
        kernel_shape: [usize; 4],
        pads: Padding,
        groups: usize,
        strides: [usize; 2],
        dilations: [usize; 2],
        output_padding: [usize; 2],
    }

    impl Default for ConvTransposeCase {
        fn default() -> Self {
            Self {
                input_shape: [1, 1, 1, 1],
                kernel_shape: [1, 1, 1, 1],
                pads: Padding::zero::<2>(),
                groups: 1,
                strides: [1, 1],
                dilations: [1, 1],
                output_padding: [0, 0],
            }
        }
    }

    // Compare reference vs optimized implementation of ConvTranspose for a
    // given set of parameters.
    fn test_conv_transpose_cases(cases: &[ConvTransposeCase]) {
        cases.test_each(|case| {
            let mut rng = XorShiftRng::new(1234);
            let input = Tensor::rand(&case.input_shape, &mut rng);
            let kernel = Tensor::rand(&case.kernel_shape, &mut rng);
            let bias = None;
            check_conv_transpose(
                input.view(),
                kernel.view(),
                bias,
                case.pads.clone(),
                case.groups,
                case.strides,
                case.dilations,
                case.output_padding,
            )
            .unwrap();
        });
    }

    #[test]
    fn test_conv_transpose_groups() {
        test_conv_transpose_cases(&[
            // Single group
            ConvTransposeCase {
                input_shape: [1, 3, 5, 5],
                kernel_shape: [3, 4, 3, 3],
                groups: 1,
                ..Default::default()
            },
            // Multiple groups
            ConvTransposeCase {
                input_shape: [1, 4, 5, 5],
                kernel_shape: [4, 2, 3, 3],
                groups: 2,
                ..Default::default()
            },
        ]);
    }

    #[test]
    fn test_conv_transpose_depthwise() {
        test_conv_transpose_cases(&[
            // Batch of images
            ConvTransposeCase {
                input_shape: [2, 4, 5, 5],
                kernel_shape: [4, 1, 3, 3],
                groups: 4,
                ..Default::default()
            },
            // Upsampling with padding, as used by learned upsampling layers
            ConvTransposeCase {
                input_shape: [1, 4, 1, 9],
                kernel_shape: [4, 1, 1, 3],
                pads: [0, 1, 0, 1].into(),
                groups: 4,
                strides: [1, 2],
                output_padding: [0, 1],
                ..Default::default()
            },
            // Strides and dilations
            ConvTransposeCase {
                input_shape: [1, 3, 4, 5],
                kernel_shape: [3, 1, 3, 2],
                groups: 3,
                strides: [2, 3],
                dilations: [2, 1],
                ..Default::default()
            },
        ]);

        // Depthwise with bias
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::rand(&[1, 4, 5, 5], &mut rng);
        let kernel = Tensor::rand(&[4, 1, 3, 3], &mut rng);
        let bias = NdTensor::rand([4], &mut rng);
        check_conv_transpose(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            [1, 1, 1, 1].into(),
            4, /* groups */
            [2, 2],
            [1, 1],
            [1, 1],
        )
        .unwrap();
    }

    #[test]
    fn test_conv_transpose_dynamic_quant() {
        #[derive(Debug)]
        struct Case {
            input_shape: Vec<usize>,
            kernel_shape: Vec<usize>,
            strides: Vec<usize>,
            pads: Vec<usize>,
        }

        let cases = [
            // 1D upsampling
            Case {
                input_shape: [1, 8, 20].into(),
                kernel_shape: [8, 4, 6].into(),
                strides: [3].into(),
                pads: [1, 2].into(),
            },
            // 1D, with the input split into several regions
            Case {
                input_shape: [1, 4, 2500].into(),
                kernel_shape: [4, 128, 16].into(),
                strides: [8].into(),
                pads: [4, 4].into(),
            },
            // 2D with batch size > 1
            Case {
                input_shape: [2, 3, 5, 6].into(),
                kernel_shape: [3, 2, 3, 3].into(),
                strides: [2, 2].into(),
                pads: [1, 1, 1, 1].into(),
            },
            // 2D, with the input split into regions of several rows
            Case {
                input_shape: [1, 2, 60, 40].into(),
                kernel_shape: [2, 128, 3, 3].into(),
                strides: [1, 1].into(),
                pads: [1, 1, 1, 1].into(),
            },
        ];

        cases.test_each(|case| {
            let mut rng = XorShiftRng::new(1234);
            let input = Tensor::<f32>::rand(&case.input_shape, &mut rng).map(|x| x - 0.3);
            let weight = Tensor::<i8>::rand(&case.kernel_shape, &mut rng).map(|&x| x.max(-127));
            let [in_c, out_c] = [case.kernel_shape[0], case.kernel_shape[1]];
            let n_cols = weight.len() / in_c;
            let weight_scale = Tensor::from_fn(&[n_cols], |idx| 0.01 + 0.001 * (idx[0] % 7) as f32);
            let bias = Tensor::from_fn(&[out_c], |idx| idx[0] as f32 * 0.1);
            let n_spatial = case.strides.len();
            let op = ConvTransposeDynamicQuant {
                dilations: vec![1; n_spatial],
                padding: Padding::Fixed(case.pads.as_slice().into()),
                strides: case.strides.clone(),
                output_padding: None,
            };
            let result: Tensor<f32> = op
                .run_simple((
                    input.view(),
                    weight.view(),
                    weight_scale.view(),
                    bias.view(),
                ))
                .unwrap();

            // Run again with prepacked weights. The unpacked weights are
            // replaced with zeros to check that the prepacked ones are used.
            let packed = op.prepack(1, weight.view().into(), None).unwrap();
            let get_prepacked = |idx| (idx == 1).then_some(&packed);
            let zero_weight = Tensor::<i8>::zeros(weight.shape());
            let input_values = [
                Some(input.view().into()),
                Some(zero_weight.view().into()),
                Some(weight_scale.view().into()),
                Some(bias.view().into()),
            ];
            let inputs = InputList::from_optional(&input_values).with_prepacked(&get_prepacked);
            let pool = BufferPool::new();
            let ctx = OpRunContext::new(&pool, &inputs, OutputMask::all_used(1));
            let prepacked_result: Tensor<f32> = op.run(&ctx).unwrap().remove(0).try_into().unwrap();
            assert_eq!(prepacked_result, result);

            // Quantize the input as the operator does, with a scale for each
            // position, and compare against a float transposed convolution
            // using the quantized input and weights.
            let levels = if GemmExecutor::<u8, i8, i32>::default().may_saturate() {
                127.
            } else {
                255.
            };
            let mut input_q = input.clone();
            let [batch, chans] = [input.size(0), input.size(1)];
            let n_pos = input.len() / (batch * chans);
            let data = input_q.data_mut().unwrap();
            for n in 0..batch {
                for pos in 0..n_pos {
                    let idx = |c: usize| (n * chans + c) * n_pos + pos;
                    let (lo, hi) = (0..chans)
                        .map(|c| data[idx(c)])
                        .fold((0f32, 0f32), |(lo, hi), x| (lo.min(x), hi.max(x)));
                    let scale = if hi > lo { (hi - lo) / levels } else { 1. };
                    let zp =
                        ((-lo / scale).round_ties_even() - (levels + 1.) / 2.).clamp(-128., 127.);
                    for c in 0..chans {
                        let q = (data[idx(c)] / scale + zp)
                            .round_ties_even()
                            .clamp(-128., 127.);
                        data[idx(c)] = (q - zp) * scale;
                    }
                }
            }
            let weight_f32 = Tensor::from_fn(weight.shape(), |idx| {
                let col = idx[1..]
                    .iter()
                    .zip(&weight.shape()[1..])
                    .fold(0, |acc, (&i, &s)| acc * s + i);
                weight[idx] as f32 * weight_scale[[col]]
            });
            let expected = conv_transpose(
                &BufferPool::new(),
                input_q.view(),
                weight_f32.view(),
                Some(bias.nd_view()),
                op.padding.clone(),
                1,
                &op.strides,
                &op.dilations,
                None,
            )
            .unwrap();
            assert_eq!(result.shape(), expected.shape());
            for (r, e) in result.iter().zip(expected.iter()) {
                assert!((r - e).abs() <= 1e-4 * e.abs().max(1.), "{r} != {e}");
            }
        });
    }

    #[test]
    fn test_conv_transpose_prepacked() {
        #[derive(Debug)]
        struct Case {
            input_shape: Vec<usize>,
            kernel_shape: Vec<usize>,
            strides: Vec<usize>,
            groups: usize,
        }

        let cases = [
            // 1D upsampling, where the kernel is a multiple of the stride
            Case {
                input_shape: [1, 16, 10].into(),
                kernel_shape: [16, 8, 6].into(),
                strides: [3].into(),
                groups: 1,
            },
            // 2D
            Case {
                input_shape: [2, 9, 5, 5].into(),
                kernel_shape: [9, 3, 3, 3].into(),
                strides: [2, 2].into(),
                groups: 1,
            },
            // Grouped. Weights are not prepacked.
            Case {
                input_shape: [1, 4, 5, 5].into(),
                kernel_shape: [4, 3, 3, 3].into(),
                strides: [1, 1].into(),
                groups: 2,
            },
        ];

        cases.test_each(|case| {
            let mut rng = XorShiftRng::new(1234);
            let input = Tensor::<f32>::rand(&case.input_shape, &mut rng);
            let kernel = Tensor::<f32>::rand(&case.kernel_shape, &mut rng);
            let out_c = case.kernel_shape[1] * case.groups;
            let bias = Tensor::<f32>::rand(&[out_c], &mut rng);
            let n_spatial = case.strides.len();
            let op = ConvTranspose {
                groups: case.groups,
                dilations: vec![1; n_spatial],
                padding: Padding::Fixed(vec![1; n_spatial * 2].into()),
                strides: case.strides.clone(),
                output_padding: None,
            };
            let result: Tensor<f32> = op
                .run_simple((input.view(), kernel.view(), bias.view()))
                .unwrap();

            let packed = op.prepack(1, kernel.view().into(), None);
            assert_eq!(packed.is_some(), case.groups == 1);
            let Some(packed) = packed else {
                return;
            };

            // Run again with prepacked weights. The unpacked weights are
            // replaced with zeros to check that the prepacked ones are used.
            let get_prepacked = |idx| (idx == 1).then_some(&packed);
            let zero_kernel = Tensor::<f32>::zeros(kernel.shape());
            let input_values = [
                Some(input.view().into()),
                Some(zero_kernel.view().into()),
                Some(bias.view().into()),
            ];
            let inputs = InputList::from_optional(&input_values).with_prepacked(&get_prepacked);
            let pool = BufferPool::new();
            let ctx = OpRunContext::new(&pool, &inputs, OutputMask::all_used(1));
            let prepacked_result: Tensor<f32> = op.run(&ctx).unwrap().remove(0).try_into().unwrap();
            assert_eq!(prepacked_result, result);
        });
    }

    #[test]
    fn test_conv_transpose_groups_bias() {
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::rand(&[1, 4, 5, 5], &mut rng);
        let kernel = Tensor::rand(&[4, 2, 3, 3], &mut rng);
        let bias = NdTensor::rand([4], &mut rng);

        check_conv_transpose(
            input.view(),
            kernel.view(),
            Some(bias.view()),
            Padding::zero::<2>(),
            2, /* groups */
            [1, 1],
            [1, 1],
            [0, 0],
        )
        .unwrap();
    }

    // Test cases where certain kernel positions have an empty range of input
    // positions to use.
    #[test]
    fn test_conv_transpose_empty_input_range() {
        test_conv_transpose_cases(&[
            // Padding at start.
            ConvTransposeCase {
                input_shape: [1, 1, 1, 1],
                kernel_shape: [1, 1, 1, 2],
                pads: Padding::Fixed([0, 1, 0, 0].into()),
                ..Default::default()
            },
            // Padding at end.
            ConvTransposeCase {
                input_shape: [1, 1, 1, 1],
                kernel_shape: [1, 1, 1, 2],
                pads: Padding::Fixed([0, 0, 0, 1].into()),
                ..Default::default()
            },
        ]);
    }

    #[test]
    fn test_conv_transpose_dilations() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = Tensor::from_data(&[1, 1, 2, 2], vec![0.1, 0.5, 0.3, 0.4]);

        // Expected values computed with `torch.nn.functional.conv_transpose2d`.
        let expected = Tensor::from_data(
            &[1, 1, 4, 4],
            vec![
                0.1, 0.2, 0.5, 1.0, 0.3, 0.4, 1.5, 2.0, 0.3, 0.6, 0.4, 0.8, 0.9, 1.2, 1.2, 1.6,
            ],
        );

        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::zero::<2>(),
            1, /* groups */
            &[1, 1],
            &[2, 2],
            Some([0, 0].as_slice()),
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // Compare against the reference implementation for various
        // combinations of dilations with other parameters.
        test_conv_transpose_cases(&[
            // Uniform dilations
            ConvTransposeCase {
                input_shape: [1, 2, 5, 5],
                kernel_shape: [2, 3, 3, 3],
                dilations: [2, 2],
                ..Default::default()
            },
            // Non-uniform dilations with strides and padding
            ConvTransposeCase {
                input_shape: [1, 2, 5, 5],
                kernel_shape: [2, 3, 3, 3],
                strides: [2, 2],
                dilations: [2, 3],
                pads: Padding::Fixed([1, 1, 1, 1].into()),
                ..Default::default()
            },
            // Dilations with groups
            ConvTransposeCase {
                input_shape: [1, 4, 5, 5],
                kernel_shape: [4, 2, 3, 3],
                groups: 2,
                dilations: [2, 2],
                ..Default::default()
            },
        ]);

        Ok(())
    }

    #[test]
    fn test_conv_transpose_output_padding() {
        test_conv_transpose_cases(&[
            // Without output padding
            ConvTransposeCase {
                input_shape: [1, 3, 5, 5],
                kernel_shape: [3, 4, 3, 3],
                output_padding: [0, 0],
                ..Default::default()
            },
            // With output padding
            ConvTransposeCase {
                input_shape: [1, 4, 5, 5],
                kernel_shape: [4, 2, 3, 3],
                output_padding: [1, 1],
                ..Default::default()
            },
        ]);
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
        let mut output = NdTensor::uninit([out_chans, out_height, out_width]);
        let columns = NdTensor::rand(
            [in_height, in_width, out_chans, kernel_height, kernel_width],
            &mut rng,
        );

        // Without padding.
        run_bench(100, Some("col2im"), || {
            col2im(
                &mut output.view_mut(),
                &columns.view(),
                [0, 0, 0, 0], // Padding
                [stride_y, stride_x],
                [1, 1], // Dilations
                None,
            );
        });

        // With padding.
        run_bench(100, Some("col2im"), || {
            col2im(
                &mut output.slice_mut((.., 2.., 2..)),
                &columns.view(),
                [1, 1, 1, 1], // Padding
                [stride_y, stride_x],
                [1, 1], // Dilations
                None,
            );
        });
    }
}
