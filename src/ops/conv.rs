use std::any::TypeId;
use std::mem::MaybeUninit;

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{CowTensor, NdTensor, NdTensorView, NdTensorViewMut, Tensor, TensorView};

use crate::gemm::{
    BiasVector, GemmExecutor, GemmInT, GemmInputA, GemmInputB, GemmOutT, QuantParams,
};
use crate::ops::matmul::zero_point_to_vec;
use crate::ops::pooling::calc_output_size_and_padding;
use crate::ops::{
    static_dims, IntoOpResult, OpError, OpRunContext, Operator, OutputList, Padding, ValueView,
};
use crate::shift_cast::ShiftCast;
use crate::tensor_pool::{AutoReturn, PoolRef, TensorPool};

mod depthwise;
mod im2col;

use depthwise::DepthwiseConvExecutor;
use im2col::build_im2col;

/// Specialization of conv_2d for pointwise convolutions over one image. This
/// can be reduced to tensor reshaping and matrix multiplication.
fn conv_2d_pointwise<X: GemmInT, W: GemmInT, Y: GemmOutT>(
    pool: &TensorPool,
    input: &NdTensorView<X, 4>,
    kernel: &NdTensorView<W, 4>,
    bias: Option<NdTensorView<Y, 1>>,
    input_quant: Option<QuantParams<X>>,
    kernel_quant: Option<QuantParams<W>>,
) -> Tensor<Y>
where
    GemmExecutor<W, X, Y>: Default,
{
    let [batch, _, in_h, in_w]: [usize; 4] = input.shape();
    let [out_c, in_c, _, _]: [usize; 4] = kernel.shape();
    let mut output = NdTensor::uninit_in(pool, [batch, out_c, in_h * in_w]);

    let kernel_mat = kernel.reshaped_in(pool, [out_c, in_c]).auto_return(pool);

    // Bias must be contiguous for use with `gemm_bias`.
    let bias = bias.as_ref().map(|b| b.to_contiguous());
    let bias_vec = bias.as_ref().map(|b| BiasVector::Column(b.data().unwrap()));

    let gemm = GemmExecutor::<W, X, Y>::default();
    let mut n_init = 0;

    for n in 0..batch {
        let mut out_item = output.slice_mut([n]);

        let in_mat = input
            .slice([n])
            .reshaped_in(pool, [in_c, in_h * in_w])
            .auto_return(pool);

        let out_item = gemm
            .gemm_uninit(
                out_item.data_mut().unwrap(),
                GemmInputA::Unpacked(kernel_mat.view()),
                GemmInputB::Unpacked(in_mat.view()),
                1., // alpha
                bias_vec,
                kernel_quant,
                input_quant,
            )
            .unwrap();
        n_init += out_item.len();
    }

    let output = output.into_shape([batch, out_c, in_h, in_w]);

    // Safety: We used `gemm_uninit_bias` to initialize all elements.
    assert!(n_init == output.len());
    unsafe { output.assume_init().into_dyn() }
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
pub fn conv<X: GemmInT, W: GemmInT, Y: GemmOutT + Default>(
    pool: &TensorPool,
    input: TensorView<X>,
    kernel: TensorView<W>,
    bias: Option<TensorView<Y>>,
    padding: Padding,
    groups: usize,
    strides: &[usize],
    dilations: &[usize],
) -> Result<Tensor<Y>, OpError>
where
    DepthwiseConvExecutor<X, W, Y>: Default,
    GemmExecutor<W, X, Y>: Default,
{
    conv_impl(
        pool, input, kernel, bias, padding, groups, strides, dilations, None, None,
    )
}

fn conv_impl<X: GemmInT, W: GemmInT, Y: GemmOutT + Default>(
    pool: &TensorPool,
    input: TensorView<X>,
    kernel: TensorView<W>,
    bias: Option<TensorView<Y>>,
    padding: Padding,
    groups: usize,
    strides: &[usize],
    dilations: &[usize],
    input_zero: Option<X>,
    kernel_zero: Option<&[W]>,
) -> Result<Tensor<Y>, OpError>
where
    DepthwiseConvExecutor<X, W, Y>: Default,
    GemmExecutor<W, X, Y>: Default,
{
    // Handle 1D convolution by expanding to 2D and then removing the extra
    // dimension from the result.
    if let &[_n, _c, _w] = input.shape() {
        let [_out_c, _k_in_c, _k_w] = static_dims!(kernel, 3, "OCW")?.shape();

        let mut input_2d = input.clone();
        input_2d.insert_axis(2);

        let mut kernel_2d = kernel.clone();
        kernel_2d.insert_axis(2);

        let padding_2d = padding.expand_1d_to_2d()?;

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

        let result_2d = conv_impl(
            pool,
            input_2d,
            kernel_2d,
            bias,
            padding_2d,
            groups,
            &strides_2d,
            &dilations_2d,
            input_zero,
            kernel_zero,
        );

        return result_2d.map(|mut t| {
            let [n, c, _h, w]: [usize; 4] = t.shape().try_into().expect("expected 4D output");
            t.reshape(&[n, c, w]);
            t
        });
    }

    let input = static_dims!(input, 4, "NCHW")?;
    let [batch, in_c, in_h, in_w] = input.shape();

    let kernel = static_dims!(kernel, 4, "OCHW")?;
    let [out_c, k_in_c, k_h, k_w] = kernel.shape();
    static_dims!(bias?, 1).transpose()?;

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
    let im2col_cols = out_h * out_w;

    let kernel_quant = kernel_zero.map(|zero_point| QuantParams { zero_point });
    let input_zero_vec = input_zero.map(|zero_point| vec![zero_point; im2col_cols]);
    let input_quant = input_zero_vec
        .as_ref()
        .map(|zero_point| QuantParams { zero_point });

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
            input_quant,
            kernel_quant,
        ));
    }

    if groups == 0 {
        return Err(OpError::InvalidValue("Group count must be > 0"));
    }

    let out_channels_per_group = out_c / groups;
    let in_channels_per_group = in_c / groups;

    if in_channels_per_group != k_in_c {
        return Err(OpError::IncompatibleInputShapes(
            "Input channels (per group) does not match kernel input channels",
        ));
    }

    if in_c % groups != 0 || out_c % groups != 0 {
        return Err(OpError::IncompatibleInputShapes(
            "Input channels and output channels must be divisible by group count",
        ));
    }

    if in_c == out_c && groups == in_c {
        let dw_conv = DepthwiseConvExecutor::default();
        let output = dw_conv.depthwise_conv_2d(
            pool,
            &input.nd_view(),
            &kernel.nd_view(),
            bias.map(|b| b.nd_view()),
            fixed_padding,
            [stride_y, stride_x],
            [dilation_y, dilation_x],
            [out_h, out_w],
            input_zero,
            kernel_zero,
        );
        return Ok(output.into_dyn());
    }

    let n_patches = out_h * out_w;
    let mut output = NdTensor::uninit_in(pool, [batch, out_c, n_patches]);
    let gemm = GemmExecutor::<W, X, Y>::default();

    // Bias must be contiguous for use with `gemm_bias`.
    let bias = bias.map(|b| b.to_contiguous());
    let bias = bias.as_ref().map(|b| b.view());

    let n_init = AtomicUsize::new(0);

    for group in 0..groups {
        let in_chan_start = group * in_channels_per_group;
        let in_chan_end = in_chan_start + in_channels_per_group;
        let out_chan_start = group * out_channels_per_group;
        let out_chans = out_chan_start..out_chan_start + out_channels_per_group;

        let in_group = input.slice((.., in_chan_start..in_chan_end));
        let mut out_group = output.slice_mut((.., out_chans.clone()));

        let kernel_mat = kernel.slice([out_chans.clone()]).reshaped_in(
            pool,
            [out_channels_per_group, in_channels_per_group * k_h * k_w],
        );

        // Prepack kernel if we'll be able to reuse packed weights.
        let prepacked_kernel = if in_group.size(0) > 1 {
            Some(gemm.prepack_a_in(pool, kernel_mat.view()).auto_return(pool))
        } else {
            None
        };
        let prepacked_kernel = prepacked_kernel.as_deref();

        out_group
            .axis_iter_mut(0)
            .into_par_iter()
            .zip(in_group.axis_iter(0))
            .for_each(|(mut out_item, in_item)| {
                let out_mat = out_item
                    .reshaped_mut([out_channels_per_group, out_h * out_w])
                    .unwrap();

                let im2col = build_im2col(
                    in_item,
                    [k_h, k_w],
                    fixed_padding,
                    [stride_y, stride_x],
                    [dilation_y, dilation_x],
                    gemm.im2col_col_count_step(),
                    gemm.im2col_row_count_step(),
                );

                let bias_vec = bias
                    .as_ref()
                    .map(|b| BiasVector::Column(&b.data().unwrap()[out_chans.clone()]));
                let out_mat = gemm
                    .gemm_uninit(
                        out_mat.into_slice_mut().unwrap(),
                        prepacked_kernel
                            .map(GemmInputA::Packed)
                            .unwrap_or(GemmInputA::Unpacked(kernel_mat.view())),
                        GemmInputB::Im2Col(&im2col),
                        1., // alpha
                        bias_vec,
                        kernel_quant,
                        input_quant,
                    )
                    .unwrap();
                n_init.fetch_add(out_mat.len(), Ordering::SeqCst);
            });
    }

    let output = output.into_shape([batch, out_c, out_h, out_w]);

    // Safety: We used `gemm_uninit_bias` to initialize all elements.
    assert!(n_init.load(Ordering::SeqCst) == output.len());
    let output = unsafe { output.assume_init() };

    Ok(output.into())
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

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let weight = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;
        conv::<f32, f32, f32>(
            ctx.pool(),
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

pub fn conv_integer<X, W>(
    pool: &TensorPool,
    input: TensorView<X>,
    kernel: TensorView<W>,
    padding: Padding,
    groups: usize,
    strides: &[usize],
    dilations: &[usize],
    input_zero: Option<TensorView<X>>,
    kernel_zero: Option<TensorView<W>>,
) -> Result<Tensor<i32>, OpError>
where
    X: Copy + Default + ShiftCast<i8>,
    W: Copy + Default + Into<i16> + ShiftCast<u8> + 'static,
    for<'a> TensorView<'a, X>: ShiftCast<CowTensor<'a, i8>>,
    for<'a> TensorView<'a, W>: ShiftCast<CowTensor<'a, u8>>,
{
    let out_chans = if kernel.ndim() >= 1 {
        kernel.size(0)
    } else {
        // Kernel has too few dimensions. Defaulting the channel count to zero
        // here is easy, but results in an error that is confusing.
        0
    };

    let input_zero = if let Some(zero_point) = input_zero {
        let Some(&zero) = zero_point.item() else {
            return Err(OpError::InvalidValue("input zero point must be a scalar"));
        };
        zero
    } else {
        X::default()
    };
    let kernel_zero = zero_point_to_vec(kernel_zero, out_chans)?
        .map(|zp| zp.to_vec())
        .unwrap_or_else(|| vec![W::default(); out_chans]);

    // Only i8 x u8 -> i32 convolution is currently supported directly, because
    // this conveniently maps to the supported input combinations for GEMM
    // ops.
    //
    // For other input types we map the int8 inputs to the opposite sign by
    // shifting the input and zero point by 128.
    //
    // If the lower-level GEMM ops gain support for more int8 signed-ness
    // combinations natively, this copy can be avoided.
    let input: PoolRef<CowTensor<i8>> = input.shift_cast_in(pool).auto_return(pool);
    let input_zero: i8 = input_zero.shift_cast();

    let (kernel, kernel_zero) = if TypeId::of::<W>() == TypeId::of::<i8>() {
        let gemm = GemmExecutor::<u8, i8, i32>::default();
        if gemm.may_saturate() {
            // If we are on a platform (x64 without VNNI) where int8 GEMM can
            // encounter i16 saturation then we need to make sure the u8 weights
            // lie within the `u7` safe range ([0, 127]).
            //
            // To avoid the saturation hazard, the model should be converted with `reduce_range`
            // enabled
            // (https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-and-why-do-i-need-to-try-u8u8).
            // Then the weights will be in `[-64, 63]` and we can shift them to
            // the safe range by adding 64.
            //
            // To handle the case where the weights are outside this range, we
            // shift by the minimum amount needed to avoid underflow when
            // converting i8 -> i16 -> u8. Saturation may then occur, but we
            // limit the amount and that's better than underflow.
            let kernel_min: i16 = kernel
                .iter()
                .copied()
                .fold(0i16, |acc, x| acc.min(x.into()));
            let shift = -kernel_min;

            let kernel: Tensor<u8> =
                kernel.map_in(pool, |w| (<W as Into<i16>>::into(*w) + shift) as u8);
            let kernel_zero: Vec<u8> = kernel_zero
                .into_iter()
                .map(|w| (w.into() + shift) as u8)
                .collect();
            (kernel.into_cow(), kernel_zero)
        } else {
            (kernel.shift_cast_in(pool), kernel_zero.shift_cast())
        }
    } else {
        // No-op cast
        (kernel.shift_cast_in(pool), kernel_zero.shift_cast())
    };
    let kernel = kernel.auto_return(pool);

    conv_impl::<i8, u8, i32>(
        pool,
        input.view(),
        kernel.view(),
        None, // bias
        padding,
        groups,
        strides,
        dilations,
        Some(input_zero),
        Some(&kernel_zero),
    )
}

#[derive(Debug)]
pub struct ConvInteger {
    pub groups: usize,
    pub dilations: Vec<usize>,
    pub padding: Padding,
    pub strides: Vec<usize>,
}

impl Operator for ConvInteger {
    fn name(&self) -> &str {
        "ConvInteger"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let weight = inputs.require(1)?;

        macro_rules! conv_integer {
            ($x:expr, $w:expr) => {{
                let input_zero = inputs.get_as(2)?;
                let weight_zero = inputs.get_as(3)?;
                conv_integer(
                    ctx.pool(),
                    $x,
                    $w,
                    self.padding.clone(),
                    self.groups,
                    &self.strides,
                    &self.dilations,
                    input_zero,
                    weight_zero,
                )
                .into_op_result()
            }};
        }

        match (input, weight) {
            (ValueView::Int8Tensor(x), ValueView::Int8Tensor(w)) => conv_integer!(x, w),
            (ValueView::Int8Tensor(x), ValueView::UInt8Tensor(w)) => conv_integer!(x, w),
            (ValueView::UInt8Tensor(x), ValueView::Int8Tensor(w)) => conv_integer!(x, w),
            (ValueView::UInt8Tensor(x), ValueView::UInt8Tensor(w)) => conv_integer!(x, w),
            _ => Err(OpError::UnsupportedType),
        }
    }
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
    bias: Option<NdTensorView<f32, 1>>,
) {
    let [stride_h, stride_w] = strides;
    let [pad_top, pad_left, _pad_bottom, _pad_right] = padding;
    let [col_chans, kernel_h, kernel_w, _img_h, _img_w] = columns.shape();
    let [out_chans, out_h, out_w] = output.shape();
    assert!(col_chans == out_chans);

    for out_c in 0..out_chans {
        // Initialize each output channel just before we accumulate into it.
        let mut out_img = output.slice_mut([out_c]);
        out_img.fill(MaybeUninit::new(bias.map(|b| b[[out_c]]).unwrap_or(0.)));

        // Safety: We just initialized all elements of `out_img`.
        let mut out_img = unsafe { out_img.assume_init() };

        for k_y in 0..kernel_h {
            for k_x in 0..kernel_w {
                let in_img = columns.slice([out_c, k_y, k_x]);
                let [img_h, img_w] = in_img.shape();

                for y in 0..img_h {
                    let out_y = y * stride_h + k_y;
                    if out_y < pad_top || out_y >= out_h + pad_top {
                        continue;
                    }

                    for x in 0..img_w {
                        let out_x = x * stride_w + k_x;
                        if out_x < pad_left || out_x >= out_w + pad_left {
                            continue;
                        }
                        unsafe {
                            *out_img.get_unchecked_mut([out_y - pad_top, out_x - pad_left]) +=
                                in_img.get_unchecked([y, x]);
                        }
                    }
                }
            }
        }
    }
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
) -> Result<([usize; 2], [usize; 4]), OpError> {
    let [in_h, in_w] = input_shape;
    let [stride_h, stride_w] = strides;
    let [k_h, k_w] = kernel_shape;

    if stride_h == 0 || stride_w == 0 {
        return Err(OpError::InvalidValue("Strides must be > 0"));
    }

    if in_h == 0 || in_w == 0 {
        return Err(OpError::InvalidValue("Input width and height must be > 0"));
    }

    match padding {
        Padding::Same => {
            // Per spec, pad the input so that:
            // output_shape[i] = input_shape[i] * strides[i] for each axis i.
            let out_h = in_h * stride_h;
            let out_w = in_w * stride_w;

            let pad_h = ((in_h - 1) * stride_h + k_h).checked_sub(out_h);
            let pad_w = ((in_w - 1) * stride_w + k_w).checked_sub(out_w);

            let (Some(pad_h), Some(pad_w)) = (pad_h, pad_w) else {
                // We can't achieve an output size of (out_h, out_w) even with
                // no padding.
                return Err(OpError::InvalidValue("Input is too small"));
            };

            // If the total padding is not even, we assign the remaining unit to
            // the ends of the axis. This matches the ONNX "SAME_UPPER"
            // value for `auto_pad`.
            let pad_top = pad_h / 2;
            let pad_bottom = pad_h.div_ceil(2);
            let pad_left = pad_w / 2;
            let pad_right = pad_w.div_ceil(2);

            Ok(([out_h, out_w], [pad_top, pad_bottom, pad_left, pad_right]))
        }
        Padding::Fixed(pads) => match pads.as_slice() {
            &[pad_top, pad_left, pad_bottom, pad_right] => {
                let out_h = ((in_h - 1) * stride_h + k_h).checked_sub(pad_top + pad_bottom);
                let out_w = ((in_w - 1) * stride_w + k_w).checked_sub(pad_left + pad_right);

                let (Some(out_h), Some(out_w)) = (out_h, out_w) else {
                    return Err(OpError::InvalidValue("Input is too small"));
                };

                Ok(([out_h, out_w], [pad_top, pad_left, pad_bottom, pad_right]))
            }
            _ => Err(OpError::InvalidValue("Wrong number of pad values")),
        },
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
    padding: Padding,
    strides: &[usize],
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
                return Err(OpError::InvalidValue("expected 1 stride value"));
            }
        };

        let result_2d = conv_transpose(
            pool,
            input_2d.view(),
            kernel_2d.view(),
            bias,
            padding_2d,
            &strides_2d,
        );

        return result_2d.map(|mut t| {
            let [n, c, _h, w]: [usize; 4] = t.shape().try_into().expect("expected 4D output");
            t.reshape(&[n, c, w]);
            t
        });
    }

    let input = static_dims!(input, 4, "NCHW")?;
    let [batch, in_c, in_h, in_w] = input.shape();
    let kernel = static_dims!(kernel, 4, "OCHW")?;
    let [k_in_c, out_c, k_h, k_w] = kernel.shape();
    static_dims!(bias?, 1).transpose()?;

    let bias = bias.map(|b| b.nd_view());

    if in_c != k_in_c {
        return Err(OpError::IncompatibleInputShapes(
            "Input channels does not match kernel input channels",
        ));
    }

    let &[stride_h, stride_w] = strides else {
        return Err(OpError::InvalidValue("expected 2 stride values"));
    };

    let (out_shape, fixed_padding) = conv_transpose_output_size_and_padding(
        [in_h, in_w],
        [k_h, k_w],
        padding,
        [stride_h, stride_w],
    )?;
    let [out_h, out_w] = out_shape;
    let [pad_top, pad_left, pad_bottom, pad_right] = fixed_padding;

    let mut output = NdTensor::uninit_in(pool, [batch, out_c, out_h, out_w]);

    let mut col2im_mat =
        NdTensor::uninit_in(pool, [out_c * k_h * k_w, in_h * in_w]).auto_return(pool);
    let kernel_mat = kernel
        .reshaped_in(pool, [k_in_c, out_c * k_h * k_w])
        .auto_return(pool);
    let kernel_mat = kernel_mat.transposed();
    let gemm = GemmExecutor::new();

    // The implementation here is the inverse of the im2col-based convolution.
    let mut n_init = 0;
    for n in 0..batch {
        let input_mat = input
            .slice([n])
            .reshaped_in(pool, [in_c, in_h * in_w])
            .auto_return(pool);

        gemm.gemm_uninit(
            col2im_mat.data_mut().unwrap(),
            GemmInputA::Unpacked(kernel_mat),
            GemmInputB::Unpacked(input_mat.view()),
            1.,   // alpha
            None, // bias
            None, // a_quant
            None, // b_quant
        )
        .unwrap();

        // Safety: `gemm_uninit` initialized col2im_mat.
        let col2im_mat = unsafe { col2im_mat.view().assume_init() };
        let mut out_img = output.slice_mut(n);

        col2im(
            &mut out_img,
            &col2im_mat.reshaped([out_c, k_h, k_w, in_h, in_w]).view(),
            [pad_top, pad_left, pad_right, pad_bottom],
            [stride_h, stride_w],
            bias,
        );
        n_init += out_img.len();
    }

    assert!(n_init == output.len());
    let output = unsafe { output.assume_init() };
    Ok(output.into_dyn())
}

#[derive(Debug)]
pub struct ConvTranspose {
    pub padding: Padding,
    pub strides: Vec<usize>,
}

impl Operator for ConvTranspose {
    fn name(&self) -> &str {
        "ConvTranspose"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let weight = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;
        conv_transpose(
            ctx.pool(),
            input,
            weight,
            bias,
            self.padding.clone(),
            &self.strides,
        )
        .into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{expect_equal, ExpectEqualError};
    use rten_tensor::{Tensor, TensorView};
    use rten_testing::TestCases;

    use crate::gemm::ReducedRangeRng;
    use crate::ops::pooling::calc_output_size_and_padding;
    use crate::ops::tests::expect_eq_1e4;
    use crate::ops::tests::new_pool;
    use crate::ops::{conv, conv_integer, conv_transpose, Conv, OpError, OperatorExt, Padding};
    use crate::tensor_pool::AutoReturn;

    use super::conv_transpose_output_size_and_padding;

    trait ReferenceConvKernel<X, W> {
        /// Update a single output element (`self`) with a given input and weight value.
        fn conv_kernel(self, x: X, w: W, x_zero: X, w_zero: W) -> Self;
    }

    impl ReferenceConvKernel<f32, f32> for f32 {
        fn conv_kernel(self, x: f32, w: f32, x_zero: f32, w_zero: f32) -> Self {
            self + (x - x_zero) * (w - w_zero)
        }
    }

    impl<X, W> ReferenceConvKernel<X, W> for i32
    where
        i32: From<X> + From<W>,
    {
        fn conv_kernel(self, x: X, w: W, x_zero: X, w_zero: W) -> Self {
            let (x, x_zero) = (i32::from(x), i32::from(x_zero));
            let (w, w_zero) = (i32::from(w), i32::from(w_zero));
            self + (x - x_zero) * (w - w_zero)
        }
    }

    /// Un-optimized reference implementation of convolution.
    ///
    /// This has the same interface as [`conv`].
    fn reference_conv<X, W, Y>(
        input: TensorView<X>,
        kernel: TensorView<W>,
        bias: Option<TensorView<Y>>,
        padding: Padding,
        groups: usize,
        strides: &[usize],
        dilations: &[usize],
        input_zero: Option<X>,
        kernel_zero: Option<&[W]>,
    ) -> Tensor<Y>
    where
        X: Copy + Default,
        W: Copy + Default,
        Y: Copy + Default + ReferenceConvKernel<X, W> + std::ops::Add<Output = Y>,
    {
        // If this is a 1D conv, insert a dummy H axis, perform a 2D convolution
        // and then remove the H axis from the result.
        if input.ndim() == 3 {
            let mut input_2d = input.clone();
            input_2d.insert_axis(2);
            let mut kernel_2d = kernel.clone();
            kernel_2d.insert_axis(2);
            let padding_2d = match padding {
                Padding::Fixed(pads) => Padding::Fixed([0, pads[0], 0, pads[1]].into()),
                Padding::Same => Padding::Same,
            };

            let mut result = reference_conv(
                input_2d,
                kernel_2d,
                bias,
                padding_2d,
                groups,
                &[1, strides[0]],
                &[1, dilations[0]],
                input_zero,
                kernel_zero,
            );

            result.remove_axis(2);

            return result;
        }

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

        let x_zero = input_zero.unwrap_or(X::default());
        let w_zero = kernel_zero
            .map(|kz| kz.to_vec())
            .unwrap_or(vec![W::default(); out_chans]);

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
                        Y::default()
                    };
                    for out_y in 0..out_h {
                        for out_x in 0..out_w {
                            let mut accum = Y::default();
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
                                            let x = input
                                                [[n, in_chan, in_y - pad_top, in_x - pad_left]];
                                            let w = kernel
                                                [[out_chan, in_chan - in_chan_start, k_y, k_x]];
                                            accum =
                                                accum.conv_kernel(x, w, x_zero, w_zero[out_chan]);
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
        input: TensorView<f32>,
        kernel: TensorView<f32>,
        bias: Option<TensorView<f32>>,
        pads: Padding,
        groups: usize,
        strides: &[usize],
        dilations: &[usize],
    ) -> Result<Tensor<f32>, ExpectEqualError> {
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
        let reference_result = reference_conv(
            input, kernel, bias, pads, groups, strides, dilations, None, None,
        );
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
        let bias = Tensor::from([1.0]);
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

        let op = Conv {
            padding: Padding::Same,
            groups: 1,
            strides: vec![1, 1],
            dilations: vec![1, 1],
        };
        let result: Tensor<f32> = op.run_simple((&input, &kernel)).unwrap();
        let reference_result = reference_conv(
            input.view(),
            kernel.view(),
            None,
            [1, 1, 1, 1].into(),
            1,       /* groups */
            &[1, 1], /* stride */
            &[1, 1], /* dilations */
            None,
            None,
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
        let bias = Tensor::from([0.1, 0.2, 0.3]);
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

    #[test]
    fn test_conv_depthwise_row_stride_row_len() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let kernel = Tensor::rand(&[1, 1, 3], &mut rng);

        // Create an input which is contiguous, but where the stride of the
        // second-to-last axis is greater than the size of the last axis.
        let mut input = Tensor::rand(&[1, 1, 20], &mut rng);
        input.clip_dim(2, 0..10);

        check_conv(
            input.view(),
            kernel.view(),
            None,
            [0, 0].into(),
            1,    /* groups */
            &[1], /* stride */
            &[1], /* dilations */
        )?;

        Ok(())
    }

    // Test various combinations of input and kernel shape and attributes.
    #[test]
    fn test_conv_shapes() {
        #[derive(Debug)]
        struct Case {
            input: Vec<usize>,
            kernel: Vec<usize>,
            padding: Padding,
            strides: Vec<usize>,
            dilations: Vec<usize>,
            output: Vec<usize>,
        }

        let cases = Vec::from([
            // Depthwise conv where the input is just large enough to fit the
            // kernel after padding.
            Case {
                input: [1, 1, 1].into(),
                kernel: [1, 1, 5].into(),
                padding: [2, 2].into(),
                strides: [1].into(),
                dilations: [1].into(),
                output: [1, 1, 1].into(),
            },
            // Catches an issue where depthwise conv did consider stride when
            // computing the output coordinate range to update for a row.
            Case {
                input: [1, 1, 1].into(),
                kernel: [1, 1, 1].into(),
                padding: [2, 0].into(),
                strides: [2].into(),
                dilations: [1].into(),
                output: [1, 1, 2].into(),
            },
        ]);

        cases.test_each(|case| {
            let mut rng = XorShiftRng::new(1234);
            let input = Tensor::rand(&case.input, &mut rng);
            let kernel = Tensor::rand(&case.kernel, &mut rng);
            let result = check_conv(
                input.view(),
                kernel.view(),
                None,
                case.padding.clone(),
                1, /* groups */
                &case.strides,
                &case.dilations,
            )
            .unwrap();
            assert_eq!(result.shape(), &case.output);
        })
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
    fn test_conv_invalid() {
        #[derive(Debug)]
        struct Case<'a> {
            input: Tensor<f32>,
            kernel: Tensor<f32>,
            strides: &'a [usize],
            groups: usize,
            dilations: &'a [usize],
            expected: OpError,
        }

        let mut rng = XorShiftRng::new(1234);
        let cases = [
            // Input too small
            Case {
                input: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                kernel: Tensor::rand(&[1, 1, 3, 3], &mut rng),
                strides: &[1, 1],
                dilations: &[1, 1],
                groups: 1,
                expected: OpError::InvalidValue("Input too small for kernel size"),
            },
            // Zero stride
            Case {
                input: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                kernel: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                strides: &[0, 0],
                dilations: &[1, 1],
                groups: 1,
                expected: OpError::InvalidValue("Strides must be > 0"),
            },
            // Unsupported stride count
            Case {
                input: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                kernel: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                strides: &[1, 1, 1],
                dilations: &[1, 1],
                groups: 1,
                expected: OpError::InvalidValue("expected 2 stride values"),
            },
            // Unsupported dilation count
            Case {
                input: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                kernel: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                strides: &[1, 1],
                dilations: &[1, 1, 1],
                groups: 1,
                expected: OpError::InvalidValue("expected 2 dilation values"),
            },
            // Zero groups
            Case {
                input: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                kernel: Tensor::rand(&[1, 1, 2, 2], &mut rng),
                strides: &[1, 1],
                dilations: &[1, 1],
                groups: 0,
                expected: OpError::InvalidValue("Group count must be > 0"),
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let result = conv(
                &pool,
                case.input.view(),
                case.kernel.view(),
                None,
                [0; 4].into(),
                case.groups,
                case.strides,
                case.dilations,
            );
            assert_eq!(result.err().as_ref(), Some(&case.expected));
        })
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

        #[derive(Debug)]
        struct Case {
            input: Tensor,
            kernel: Tensor,
        }

        let cases = [
            Case {
                input: Tensor::rand(&[n, in_c, in_w], &mut rng),
                kernel: Tensor::rand(&[out_c, in_c, k_w], &mut rng),
            },
            // Non-contiguous inputs
            Case {
                input: {
                    let mut input = Tensor::rand(&[n, in_w, in_c], &mut rng);
                    input.permute(&[0, 2, 1]);
                    input
                },
                kernel: {
                    let mut kernel = Tensor::rand(&[out_c, k_w, in_c], &mut rng);
                    kernel.permute(&[0, 2, 1]);
                    kernel
                },
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let result = conv(
                &pool,
                case.input.view(),
                case.kernel.view(),
                None,
                Padding::Same,
                1,    /* groups */
                &[1], /* stride */
                &[1], /* dilation */
            )
            .unwrap();

            assert_eq!(result.shape(), &[n, out_c, in_w]);
        })
    }

    macro_rules! impl_conv_integer_test {
        ($name:ident, $input_ty:ty, $weight_ty:ty) => {
            #[test]
            fn $name() {
                fn check_conv_int8(
                    input: TensorView<$input_ty>,
                    kernel: TensorView<$weight_ty>,
                    pads: Padding,
                    groups: usize,
                    strides: &[usize],
                    dilations: &[usize],
                    input_zero: Option<TensorView<$input_ty>>,
                    kernel_zero: Option<TensorView<$weight_ty>>,
                ) -> Result<Tensor<i32>, ExpectEqualError> {
                    let pool = new_pool();
                    let result = conv_integer(
                        &pool,
                        input.view(),
                        kernel.view(),
                        pads.clone(),
                        groups,
                        &strides,
                        &dilations,
                        input_zero.clone(),
                        kernel_zero.clone(),
                    )
                    .expect("conv operation failed");
                    let reference_result = reference_conv(
                        input,
                        kernel,
                        None,
                        pads,
                        groups,
                        strides,
                        dilations,
                        input_zero.map(|kz| kz.item().copied().unwrap()),
                        kernel_zero.map(|kz| kz.data().unwrap()),
                    );
                    expect_equal(&result, &reference_result)?;
                    Ok(result)
                }

                let mut rng = XorShiftRng::new(1234);
                let mut kernel_rng = ReducedRangeRng::new(true /* reduce_range */, 1234);

                // Minimum number of depthwise channels to exercise multi-threading.
                // This needs to be updated if channel blocking is added to the
                // depthwise conv impl.
                let min_depthwise_channels = 3;

                #[derive(Debug)]
                struct Case {
                    input: Tensor<$input_ty>,
                    kernel: Tensor<$weight_ty>,
                    input_zero: Option<$input_ty>,
                    kernel_zero: Option<Vec<$weight_ty>>,
                    groups: usize,
                }

                let cases = [
                    // General convolution.
                    Case {
                        input: Tensor::rand(&[1, 2, 5, 5], &mut rng),
                        kernel: Tensor::rand(&[1, 2, 3, 3], &mut kernel_rng),
                        input_zero: Some(12),
                        kernel_zero: Some([1].into()),
                        groups: 1,
                    },
                    // General convolution with multiple output channels.
                    Case {
                        input: Tensor::rand(&[1, 2, 5, 5], &mut rng),
                        kernel: Tensor::rand(&[3, 2, 3, 3], &mut kernel_rng),
                        input_zero: Some(12),
                        kernel_zero: Some([1, 2, 3].into()),
                        groups: 1,
                    },
                    // General convolution with no zero point.
                    Case {
                        input: Tensor::rand(&[1, 2, 5, 5], &mut rng),
                        kernel: Tensor::rand(&[1, 2, 3, 3], &mut kernel_rng),
                        input_zero: None,
                        kernel_zero: None,
                        groups: 1,
                    },
                    // Pointwise convolution.
                    Case {
                        input: Tensor::rand(&[1, 2, 5, 5], &mut rng),
                        kernel: Tensor::rand(&[1, 2, 1, 1], &mut kernel_rng),
                        input_zero: Some(12),
                        kernel_zero: Some([1].into()),
                        groups: 1,
                    },
                    // Pointwise convolution with no zero point.
                    Case {
                        input: Tensor::rand(&[1, 2, 1, 1], &mut rng),
                        kernel: Tensor::rand(&[1, 2, 1, 1], &mut kernel_rng),
                        input_zero: Some(12),
                        kernel_zero: Some([1].into()),
                        groups: 1,
                    },
                    // Depthwise convolution.
                    Case {
                        input: Tensor::rand(&[2, min_depthwise_channels, 4, 4], &mut rng),
                        kernel: Tensor::rand(&[min_depthwise_channels, 1, 3, 3], &mut kernel_rng),
                        input_zero: Some(12),
                        kernel_zero: Some(
                            (0..min_depthwise_channels)
                                .map(|x| x as $weight_ty)
                                .collect(),
                        ),
                        groups: min_depthwise_channels,
                    },
                    // Depthwise convolution with no zero point.
                    Case {
                        input: Tensor::rand(&[2, 2, 5, 5], &mut rng),
                        kernel: Tensor::rand(&[2, 1, 3, 3], &mut kernel_rng),
                        input_zero: None,
                        kernel_zero: None,
                        groups: 2,
                    },
                ];

                cases.test_each(|case| {
                    let output_chans = case.kernel.size(0);
                    check_conv_int8(
                        case.input.view(),
                        case.kernel.view(),
                        Padding::zero::<2>(),
                        case.groups,
                        &[1, 1], // strides
                        &[1, 1], // dilations
                        case.input_zero
                            .map(|zero| Tensor::from(zero))
                            .as_ref()
                            .map(|t| t.view()),
                        case.kernel_zero
                            .clone()
                            .map(|zero| Tensor::from_data(&[output_chans], zero))
                            .as_ref()
                            .map(|t| t.view()),
                    )
                    .unwrap();
                })
            }
        };
    }
    impl_conv_integer_test!(test_conv_integer_u8_u8, u8, u8);
    impl_conv_integer_test!(test_conv_integer_u8_i8, u8, i8);
    impl_conv_integer_test!(test_conv_integer_i8_u8, i8, u8);
    impl_conv_integer_test!(test_conv_integer_i8_i8, i8, i8);

    #[test]
    fn test_conv_transpose() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
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

        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::zero::<2>(),
            &[2, 2],
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        let mut expected_with_bias = Tensor::from_data(expected.shape().into(), expected.to_vec());
        for eb in expected_with_bias.iter_mut() {
            *eb += 1.234;
        }
        let bias = Tensor::from([1.234]);
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            Some(bias.view()),
            Padding::zero::<2>(),
            &[2, 2],
        )
        .unwrap();
        expect_equal(&result, &expected_with_bias)?;

        Ok(())
    }

    #[test]
    fn test_conv_transpose_padding() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = Tensor::from_data(&[1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);

        // Expected values computed with `torch.nn.functional.conv_transpose2d`.
        let expected = Tensor::from_data(&[1, 1, 2, 2], vec![0.4, 0.6, 0.6, 0.4]);
        let strides = [2, 2];

        // Fixed padding. The output shape should have rows and columns
        // subtracted on each side according to the corresponding padding.
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::Fixed([1, 1, 1, 1].into()),
            &strides,
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
            &strides,
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
        let pool = new_pool();
        let input = Tensor::from_data(&[1, 1, 2], vec![1., 2.]);
        let kernel = Tensor::from_data(&[1, 1, 2], vec![0.1, 0.2]);

        // Expected values computed with `torch.nn.functional.conv_transpose1d`.
        let expected = Tensor::from_data(&[1, 1, 4], vec![0.1, 0.2, 0.2, 0.4]);

        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::zero::<1>(),
            &[2],
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        let bias = Tensor::from([0.5]);
        let expected_with_bias = expected.map(|x| x + bias[[0]]);
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            Some(bias.view()),
            Padding::zero::<1>(),
            &[2],
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
            expected: Result<([usize; 2], [usize; 4]), OpError>,
        }

        let cases = [
            // Zero padding, stride of 1
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [1, 1],
                expected: Ok(([7, 7], [0, 0, 0, 0])),
            },
            // Zero padding, stride of 3
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [3, 3],
                expected: Ok(([15, 15], [0, 0, 0, 0])),
            },
            // Non-zero padding, stride of 1
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([1, 1, 1, 1].into()),
                strides: [1, 1],
                expected: Ok(([5, 5], [1, 1, 1, 1])),
            },
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([2, 2, 2, 2].into()),
                strides: [1, 1],
                expected: Ok(([3, 3], [2, 2, 2, 2])),
            },
            // Uneven padding
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([1, 2, 1, 2].into()),
                strides: [1, 1],
                expected: Ok(([5, 3], [1, 2, 1, 2])),
            },
            // Same padding
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Same,
                strides: [1, 1],
                expected: Ok(([5, 5], [1, 1, 1, 1])),
            },
            // Same padding. Case where output size is smaller than
            // `input_shape * stride` even with no padding.
            Case {
                input_shape: [5, 5],
                kernel_shape: [1, 1],
                padding: Padding::Same,
                strides: [3, 3],
                expected: Err(OpError::InvalidValue("Input is too small")),
            },
            // Padding too large
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([4, 4, 4, 4].into()),
                strides: [1, 1],
                expected: Err(OpError::InvalidValue("Input is too small")),
            },
            // Invalid strides
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [0, 0],
                expected: Err(OpError::InvalidValue("Strides must be > 0")),
            },
            // Empty input
            Case {
                input_shape: [0, 0],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [1, 1],
                expected: Err(OpError::InvalidValue("Input width and height must be > 0")),
            },
            // Wrong padding size for input spatial shape.
            Case {
                input_shape: [1, 1],
                kernel_shape: [3, 3],
                padding: Padding::zero::<1>(),
                strides: [1, 1],
                expected: Err(OpError::InvalidValue("Wrong number of pad values")),
            },
        ];

        cases.test_each(|case| {
            let result = conv_transpose_output_size_and_padding(
                case.input_shape,
                case.kernel_shape,
                case.padding.clone(),
                case.strides,
            );
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    #[ignore]
    fn bench_depthwise_conv() {
        let mut rng = XorShiftRng::new(1234);

        // Input and kernel sizes copied from the last layers of MobileNetV2.
        // This has a small spatial shape, so it measures overhead around the
        // inner loop. A larger spatial shape would be more affected by the
        // efficiency of the innermost loops.
        let input = Tensor::<f32>::rand(&[1, 576, 14, 14], &mut rng);
        let kernel = Tensor::<f32>::rand(&[576, 1, 3, 3], &mut rng);

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
                None,
            );
        });
    }
}
