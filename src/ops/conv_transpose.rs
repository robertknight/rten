use std::mem::MaybeUninit;

use rayon::prelude::*;
use rten_base::iter::range_chunks;
use rten_base::num::div_ceil;
use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmUninitOptions};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut, Tensor, TensorView};

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext, static_dims,
};
use crate::ops::Padding;

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
) -> std::ops::RangeInclusive<usize> {
    // Compute with signed values to avoid underflow in subtraction.
    let input_size = input_size as isize;
    let output_size = output_size as isize;
    let pad_start = pad_start as isize;
    let kernel_pos = kernel_pos as isize;
    let stride = stride as isize;

    let x_start = div_ceil(pad_start - kernel_pos, stride).clamp(0, input_size - 1);
    let x_end = ((pad_start + output_size - 1 - kernel_pos) / stride).clamp(0, input_size - 1);

    x_start as usize..=x_end as usize
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

    output
        .axis_iter_mut(0)
        .into_par_iter()
        .enumerate()
        .for_each(|(out_c, mut out_img)| {
            // Initialize each output channel just before we accumulate into it.
            out_img.fill(MaybeUninit::new(bias.map(|b| b[[out_c]]).unwrap_or(0.)));

            // Safety: We just initialized all elements of `out_img`.
            let mut out_img = unsafe { out_img.assume_init() };

            for k_y in 0..kernel_h {
                for k_x in 0..kernel_w {
                    let in_img = columns.slice([out_c, k_y, k_x]);
                    let [img_h, img_w] = in_img.shape();

                    let y_range = col2im_input_range(img_h, out_h, pad_top, k_y, stride_h);
                    let x_range = col2im_input_range(img_w, out_w, pad_left, k_x, stride_w);

                    for y in y_range {
                        let out_y = y * stride_h + k_y;
                        debug_assert!(out_y >= pad_top && out_y < out_h + pad_top);

                        for x in x_range.clone() {
                            let out_x = x * stride_w + k_x;
                            debug_assert!(out_x >= pad_left && out_x < out_w + pad_left);

                            // Safety: We computed x, y, out_x and out_y such that they are
                            // in-bounds for out_img and in_img.
                            unsafe {
                                *out_img.get_unchecked_mut([out_y - pad_top, out_x - pad_left]) +=
                                    in_img.get_unchecked([y, x]);
                            }
                        }
                    }
                }
            }
        });
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
    output_padding: [usize; 2],
) -> Result<([usize; 2], [usize; 4]), OpError> {
    let [in_h, in_w] = input_shape;
    let [stride_h, stride_w] = strides;
    let [k_h, k_w] = kernel_shape;
    let [out_pad_h, out_pad_w] = output_padding;

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

            let pad_h = ((in_h - 1) * stride_h + k_h + out_pad_h).checked_sub(out_h);
            let pad_w = ((in_w - 1) * stride_w + k_w + out_pad_w).checked_sub(out_w);

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
                let out_h =
                    ((in_h - 1) * stride_h + out_pad_h + k_h).checked_sub(pad_top + pad_bottom);
                let out_w =
                    ((in_w - 1) * stride_w + out_pad_w + k_w).checked_sub(pad_left + pad_right);

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
    pool: &BufferPool,
    input: TensorView,
    kernel: TensorView,
    bias: Option<TensorView>,
    padding: Padding,
    groups: usize,
    strides: &[usize],
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
                return Err(OpError::InvalidValue("expected 1 stride value"));
            }
        };

        let output_padding_2d = match output_padding {
            Some(&[pad]) => [0, pad],
            None => [0, 0],
            _ => {
                return Err(OpError::InvalidValue("expected 1 output_padding value"));
            }
        };

        let result_2d = conv_transpose(
            pool,
            input_2d.view(),
            kernel_2d.view(),
            bias,
            padding_2d,
            groups,
            &strides_2d,
            Some(&output_padding_2d),
        );

        return result_2d.map(|mut t| {
            let [n, c, _h, w]: [usize; 4] = t.shape().try_into().expect("expected 4D output");
            t.reshape(&[n, c, w]);
            t
        });
    }

    let input = static_dims!(input, 4, "NCHW")?;
    let [batch, in_c, in_h, in_w] = input.shape();
    let kernel = static_dims!(kernel, 4, "COHW")?;
    let [k_in_c, out_chans_per_group, k_h, k_w] = kernel.shape();
    static_dims!(bias?, 1).transpose()?;

    let bias = bias.map(|b| b.nd_view());

    if in_c != k_in_c {
        return Err(OpError::IncompatibleInputShapes(
            "Input channels does not match kernel input channels",
        ));
    }

    if groups == 0 {
        return Err(OpError::InvalidValue("Group count must be > 0"));
    }

    if k_in_c % groups != 0 {
        return Err(OpError::InvalidValue(
            "Input channel count not divisible by groups",
        ));
    }

    let &[stride_h, stride_w] = strides else {
        return Err(OpError::InvalidValue("expected 2 stride values"));
    };
    let [out_pad_h, out_pad_w] = match output_padding {
        Some(&[h, w]) => [h, w],
        None => [0, 0],
        _ => {
            return Err(OpError::InvalidValue("expected 2 output_padding values"));
        }
    };

    let (out_shape, fixed_padding) = conv_transpose_output_size_and_padding(
        [in_h, in_w],
        [k_h, k_w],
        padding,
        [stride_h, stride_w],
        [out_pad_h, out_pad_w],
    )?;
    let [out_h, out_w] = out_shape;
    let [pad_top, pad_left, pad_bottom, pad_right] = fixed_padding;
    let out_c = out_chans_per_group * groups;

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

        for n in 0..batch {
            let input_mat = in_group
                .slice([n])
                .reshaped_in(pool, [in_chans.len(), in_h * in_w])
                .auto_return(pool);

            let col2im_shape = col2im_mat.shape();
            let col2im_init = gemm
                .gemm_uninit(
                    col2im_mat.data_mut().unwrap(),
                    GemmInputA::Unpacked(kernel_mat),
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

        conv_transpose(
            ctx.pool(),
            input,
            weight,
            bias,
            self.padding.clone(),
            self.groups,
            &self.strides,
            self.output_padding.as_deref(),
        )
        .into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{ExpectEqualError, expect_equal};
    use rten_tensor::{NdTensor, Tensor, TensorView};
    use rten_testing::TestCases;

    use super::{conv_transpose, conv_transpose_output_size_and_padding};
    use crate::buffer_pool::BufferPool;
    use crate::ops::{OpError, Padding};

    fn reference_conv_transpose(
        input: TensorView,
        kernel: TensorView,
        bias: Option<TensorView>,
        padding: Padding,
        groups: usize,
        [stride_h, stride_w]: [usize; 2],
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
                                        if y + pad_top >= k_y && x + pad_left >= k_x {
                                            let in_y = (y + pad_top - k_y) / stride_h;
                                            let in_x = (x + pad_left - k_x) / stride_w;
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
        bias: Option<TensorView<f32>>,
        pads: Padding,
        groups: usize,
        strides: [usize; 2],
        output_padding: [usize; 2],
    ) -> Result<Tensor<f32>, ExpectEqualError> {
        let pool = BufferPool::new();
        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            bias.clone(),
            pads.clone(),
            groups,
            &strides,
            Some(output_padding.as_slice()),
        )
        .expect("conv operation failed");
        let reference_result =
            reference_conv_transpose(input, kernel, bias, pads, groups, strides, output_padding)
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
        let output_padding = Some([0, 0].as_slice());

        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::zero::<2>(),
            groups,
            &strides,
            output_padding,
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
            groups,
            &strides,
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
        let output_padding = Some([0].as_slice());

        let result = conv_transpose(
            &pool,
            input.view(),
            kernel.view(),
            None,
            Padding::zero::<1>(),
            groups,
            &strides,
            output_padding,
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
            groups,
            &strides,
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
                    output_padding: [0, 0],
                    expected: Err(OpError::InvalidValue("default value")),
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
            // Same padding. Case where output size is smaller than
            // `input_shape * stride` even with no padding.
            Case {
                input_shape: [5, 5],
                kernel_shape: [1, 1],
                padding: Padding::Same,
                strides: [3, 3],
                expected: Err(OpError::InvalidValue("Input is too small")),
                ..Default::default()
            },
            // Padding too large
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::Fixed([4, 4, 4, 4].into()),
                strides: [1, 1],
                expected: Err(OpError::InvalidValue("Input is too small")),
                ..Default::default()
            },
            // Invalid strides
            Case {
                input_shape: [5, 5],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [0, 0],
                expected: Err(OpError::InvalidValue("Strides must be > 0")),
                ..Default::default()
            },
            // Empty input
            Case {
                input_shape: [0, 0],
                kernel_shape: [3, 3],
                padding: Padding::zero::<2>(),
                strides: [1, 1],
                expected: Err(OpError::InvalidValue("Input width and height must be > 0")),
                ..Default::default()
            },
            // Wrong padding size for input spatial shape.
            Case {
                input_shape: [1, 1],
                kernel_shape: [3, 3],
                padding: Padding::zero::<1>(),
                strides: [1, 1],
                expected: Err(OpError::InvalidValue("Wrong number of pad values")),
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
