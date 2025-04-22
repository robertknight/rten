use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use rten_simd::SimdOp;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};
use rten_vecmath as vecmath;

use crate::ops::static_dims;
use crate::ops::{
    resolve_axis, InputList, IntoOpResult, OpError, OpRunContext, Operator, Output, OutputList,
};
use crate::slice_reductions::slice_max;
use crate::tensor_pool::TensorPool;

/// Specifies how to normalize the mean and variance.
#[derive(Copy, Clone, Debug, PartialEq)]
enum MeanNormalize {
    /// Normalize mean and variance using precomputed statistics.
    ///
    /// This is used for BatchNormalization.
    Static { mean: f32, variance: f32 },

    /// Normalize mean and variance using statistics computed from the input
    /// data.
    ///
    /// This is used for LayerNormalization, InstanceNormalization and
    /// GroupNormalization.
    Dynamic,

    /// Normalize the scale of the input using [RMSNorm] but don't center the mean.
    ///
    /// The RMS is computed dynamically from the input.
    ///
    /// [RMSNorm]: <https://pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html>
    DynamicRootMeanSquare,
}

struct NormalizeOptions<'a> {
    mean_normalize: MeanNormalize,

    /// Epsilon value used to avoid divide-by-zero in sqrt.
    epsilon: f32,

    /// Constant scale to multiply normalized value by.
    scale: f32,

    /// Constant bias to add to normalized value.
    bias: f32,

    /// Per-element scale to multiply normalized value by.
    element_scale: Option<&'a [f32]>,

    /// Per-element bias to add to normalized value.
    element_bias: Option<&'a [f32]>,
}

impl Default for NormalizeOptions<'_> {
    fn default() -> Self {
        NormalizeOptions {
            mean_normalize: MeanNormalize::Dynamic,

            // Default value for ONNX BatchNormalization, InstanceNormalization
            // and LayerNormalization operators.
            epsilon: 1e-05,

            scale: 1.0,
            bias: 0.,
            element_scale: None,
            element_bias: None,
        }
    }
}

enum NormalizeData<'src, 'dst> {
    /// Read from a source slice and write normalized data to an output slice
    /// of the same length.
    SrcDest((&'src [f32], &'dst mut [MaybeUninit<f32>])),

    /// Normalize elements of a slice in place.
    InPlace(&'dst mut [f32]),
}

impl<'dst> From<&'dst mut [f32]> for NormalizeData<'dst, 'dst> {
    fn from(val: &'dst mut [f32]) -> Self {
        NormalizeData::InPlace(val)
    }
}

impl<'src, 'dst> From<(&'src [f32], &'dst mut [MaybeUninit<f32>])> for NormalizeData<'src, 'dst> {
    fn from(val: (&'src [f32], &'dst mut [MaybeUninit<f32>])) -> Self {
        NormalizeData::SrcDest(val)
    }
}

/// Normalize the mean and variance of elements in `data` and apply a scale
/// and bias to the result.
///
/// Returns the normalized elements.
fn normalize_slice<'src, 'dst>(
    data: NormalizeData<'src, 'dst>,
    opts: NormalizeOptions<'src>,
) -> &'dst mut [f32] {
    let NormalizeOptions {
        mean_normalize,
        epsilon,
        scale,
        bias,
        element_bias,
        element_scale,
    } = opts;

    let input = match &data {
        NormalizeData::InPlace(data) => *data,
        NormalizeData::SrcDest((src, _dest)) => *src,
    };

    let (mean, variance) = match mean_normalize {
        MeanNormalize::Static { mean, variance } => (mean, variance),
        MeanNormalize::Dynamic => {
            let mean = vecmath::Sum::new(input).dispatch() / input.len() as f32;
            let variance = vecmath::SumSquareSub::new(input, mean).dispatch() / input.len() as f32;
            (mean, variance)
        }
        MeanNormalize::DynamicRootMeanSquare => {
            let root_mean_square = vecmath::SumSquare::new(input).dispatch() / input.len() as f32;
            (0., root_mean_square)
        }
    };

    // To avoid divisions in the vectorized loop, we re-arrange:
    //
    // ```
    // Y = (X - mean) / sqrt(variance + epsilon) * scale + bias
    // ```
    //
    // As:
    //
    // ```
    // scaled_std_dev_reciprocal = scale / (variance + epsilon).sqrt()
    // Y = (X - mean) * scaled_std_dev_reciprocal + bias
    // ```
    let scaled_std_dev_reciprocal = scale / (variance + epsilon).sqrt();

    let opts = vecmath::NormalizeOptions {
        pre_scale_bias: mean,
        bias,
        scale: scaled_std_dev_reciprocal,
        element_bias,
        element_scale,
    };

    let op = match data {
        NormalizeData::InPlace(data) => vecmath::Normalize::new_mut(data, opts),
        NormalizeData::SrcDest((src, dest)) => vecmath::Normalize::new(src, dest, opts),
    };
    op.dispatch()
}

/// Normalize each channel separately in an `(N, C, ...)` tensor.
fn normalize_each_channel<'a>(
    input: &mut Tensor,
    chan_opts: impl Fn(usize) -> NormalizeOptions<'a>,
) {
    let batch = input.size(0);

    // Per BatchNormalization spec: "The op also accepts single dimension input
    // of size N in which case C is assumed to be 1"
    let chans = if input.ndim() >= 2 { input.size(1) } else { 1 };

    input.make_contiguous();
    let chunk_len = input.len() / (batch * chans);
    let mut chunks = input.data_mut().unwrap().chunks_mut(chunk_len);

    for _ in 0..batch {
        for c in 0..chans {
            let chan_data = chunks.next().unwrap();
            let opts = chan_opts(c);
            normalize_slice(chan_data.into(), opts);
        }
    }
}

/// Perform in-place batch normalization on an `NC*` tensor.
///
/// See <https://github.com/onnx/onnx/blob/main/docs/Operators.md#batchnormalization>.
pub fn batch_norm_in_place(
    input: &mut Tensor,
    scale: &NdTensorView<f32, 1>,
    bias: &NdTensorView<f32, 1>,
    mean: &NdTensorView<f32, 1>,
    var: &NdTensorView<f32, 1>,
    epsilon: f32,
) -> Result<(), OpError> {
    if input.ndim() < 1 {
        return Err(OpError::InvalidValue("Input must have at least 1 dim"));
    }

    normalize_each_channel(input, |chan| NormalizeOptions {
        mean_normalize: MeanNormalize::Static {
            mean: mean[chan],
            variance: var[chan],
        },
        epsilon,
        scale: scale[chan],
        bias: bias[chan],
        ..Default::default()
    });

    Ok(())
}

/// Perform batch normalization on an `NC*` tensor.
///
/// See <https://github.com/onnx/onnx/blob/main/docs/Operators.md#batchnormalization>.
pub fn batch_norm(
    pool: &TensorPool,
    input: TensorView,
    scale: &NdTensorView<f32, 1>,
    bias: &NdTensorView<f32, 1>,
    mean: &NdTensorView<f32, 1>,
    var: &NdTensorView<f32, 1>,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    let mut output = input.to_tensor_in(pool);
    batch_norm_in_place(&mut output, scale, bias, mean, var, epsilon)?;
    Ok(output)
}

#[derive(Debug)]
pub struct BatchNormalization {
    pub epsilon: f32,
}

impl Operator for BatchNormalization {
    fn name(&self) -> &str {
        "BatchNormalization"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;

        let scale = inputs.require_as(1)?;
        let scale = static_dims!(scale, 1)?;

        let bias = inputs.require_as(2)?;
        let bias = static_dims!(bias, 1)?;

        let mean = inputs.require_as(3)?;
        let mean = static_dims!(mean, 1)?;

        let var = inputs.require_as(4)?;
        let var = static_dims!(var, 1)?;

        batch_norm(ctx.pool(), input, &scale, &bias, &mean, &var, self.epsilon).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        let mut output = input
            .into_tensor::<f32>()
            .ok_or(OpError::IncorrectInputType)?;
        let scale = other.require_as(0)?;
        let scale = static_dims!(scale, 1)?;

        let bias = other.require_as(1)?;
        let bias = static_dims!(bias, 1)?;

        let mean = other.require_as(2)?;
        let mean = static_dims!(mean, 1)?;

        let var = other.require_as(3)?;
        let var = static_dims!(var, 1)?;

        batch_norm_in_place(&mut output, &scale, &bias, &mean, &var, self.epsilon)?;

        Ok(output.into())
    }
}

pub fn instance_normalization(
    pool: &TensorPool,
    input: TensorView,
    scale: NdTensorView<f32, 1>,
    bias: NdTensorView<f32, 1>,
    epsilon: Option<f32>,
) -> Result<Tensor, OpError> {
    let mut output = input.to_tensor_in(pool);
    instance_normalization_in_place(&mut output, scale, bias, epsilon)?;
    Ok(output)
}

pub fn instance_normalization_in_place(
    input: &mut Tensor,
    scale: NdTensorView<f32, 1>,
    bias: NdTensorView<f32, 1>,
    epsilon: Option<f32>,
) -> Result<(), OpError> {
    let &[_batch, chans, ..] = input.shape() else {
        return Err(OpError::InvalidValue("expected input with >= 2 dims"));
    };

    // If epsilon is None, use default from ONNX spec.
    let epsilon = epsilon.unwrap_or(1e-5);

    if scale.size(0) != chans {
        return Err(OpError::InvalidValue(
            "scale length should match channel count",
        ));
    }

    if bias.size(0) != chans {
        return Err(OpError::InvalidValue(
            "bias length should match channel count",
        ));
    }

    normalize_each_channel(input, |chan| NormalizeOptions {
        epsilon,
        scale: scale[chan],
        bias: bias[chan],
        ..Default::default()
    });

    Ok(())
}

#[derive(Debug)]
pub struct InstanceNormalization {
    pub epsilon: Option<f32>,
}

impl Operator for InstanceNormalization {
    fn name(&self) -> &str {
        "InstanceNormalization"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;

        let scale = inputs.require_as(1)?;
        let scale = static_dims!(scale, 1)?;

        let bias = inputs.require_as(2)?;
        let bias = static_dims!(bias, 1)?;

        instance_normalization(ctx.pool(), input, scale, bias, self.epsilon).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        output: Output,
        inputs: InputList,
    ) -> Result<Output, OpError> {
        let mut output = output
            .into_tensor::<f32>()
            .ok_or(OpError::IncorrectInputType)?;

        let scale = inputs.require_as(0)?;
        let scale = static_dims!(scale, 1)?;

        let bias = inputs.require_as(1)?;
        let bias = static_dims!(bias, 1)?;

        instance_normalization_in_place(&mut output, scale, bias, self.epsilon)?;

        Ok(output.into())
    }
}

pub fn rms_normalization(
    pool: &TensorPool,
    input: TensorView,
    scale: TensorView,
    axis: isize,
    epsilon: Option<f32>,
) -> Result<Tensor, OpError> {
    layer_normalization_impl(
        pool,
        input,
        scale,
        None, // bias
        axis,
        epsilon,
        MeanNormalize::DynamicRootMeanSquare,
    )
}

pub fn layer_normalization(
    pool: &TensorPool,
    input: TensorView,
    scale: TensorView,
    bias: Option<TensorView>,
    axis: isize,
    epsilon: Option<f32>,
) -> Result<Tensor, OpError> {
    layer_normalization_impl(
        pool,
        input,
        scale,
        bias,
        axis,
        epsilon,
        MeanNormalize::Dynamic,
    )
}

fn layer_normalization_impl(
    pool: &TensorPool,
    input: TensorView,
    scale: TensorView,
    bias: Option<TensorView>,
    axis: isize,
    epsilon: Option<f32>,
    mean_normalize: MeanNormalize,
) -> Result<Tensor, OpError> {
    let epsilon = epsilon.unwrap_or(1e-5);
    let resolved_axis = resolve_axis(input.ndim(), axis)?;
    let normalized_slice_shape = &input.shape()[resolved_axis..];

    if !scale.can_broadcast_to(input.shape()) {
        return Err(OpError::IncompatibleInputShapes(
            "`scale` cannot be broadcast to input shape",
        ));
    }
    if scale.shape() != normalized_slice_shape {
        return Err(OpError::UnsupportedValue(
            "`scale` shape does not match normalized axes of input",
        ));
    }

    if let Some(bias) = bias.as_ref() {
        if !bias.can_broadcast_to(input.shape()) {
            return Err(OpError::IncompatibleInputShapes(
                "`bias` cannot be broadcast to input shape",
            ));
        }
        if bias.shape() != normalized_slice_shape {
            return Err(OpError::UnsupportedValue(
                "`bias` shape does not match normalized axes of input",
            ));
        }
    }

    let input = input.to_contiguous_in(pool);

    let mut output = pool.alloc(input.len());
    let chunk_size = input.shape()[resolved_axis..].iter().product();

    let bias = bias.map(|b| b.to_contiguous_in(pool));
    let bias_data = bias.as_ref().map(|b| b.data().unwrap());

    let scale = scale.to_contiguous_in(pool);
    let scale_data = scale.data().unwrap();

    let n_init = AtomicUsize::new(0);
    let out_uninit = &mut output.spare_capacity_mut()[..input.len()];
    input
        .data()
        .unwrap()
        .par_chunks(chunk_size)
        .zip(out_uninit.par_chunks_mut(chunk_size))
        .for_each(|(in_chunk, out_chunk)| {
            let normalized = normalize_slice(
                (in_chunk, out_chunk).into(),
                NormalizeOptions {
                    mean_normalize,
                    epsilon,
                    element_scale: Some(scale_data),
                    element_bias: bias_data,
                    ..Default::default()
                },
            );
            n_init.fetch_add(normalized.len(), Ordering::SeqCst);
        });

    assert_eq!(n_init.load(Ordering::SeqCst), input.len());
    unsafe {
        output.set_len(input.len());
    }

    Ok(Tensor::from_data(input.shape(), output))
}

#[derive(Debug)]
pub struct LayerNormalization {
    pub axis: isize,
    pub epsilon: Option<f32>,
}

impl Operator for LayerNormalization {
    fn name(&self) -> &str {
        "LayerNormalization"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let scale = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;

        layer_normalization(
            ctx.pool(),
            input.view(),
            scale,
            bias,
            self.axis,
            self.epsilon,
        )
        .into_op_result()
    }
}

/// Root Mean Square normalization.
///
/// This is a simplified version of [`LayerNormalization`] which does not center
/// the mean and uses a Root Mean Square statistic instead of variance to
/// normalize the scale.
///
/// See <https://pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html>.
#[derive(Debug)]
pub struct RmsNormalization {
    pub axis: isize,
    pub epsilon: Option<f32>,
}

impl Operator for RmsNormalization {
    fn name(&self) -> &str {
        "RmsNormalization"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let scale = inputs.require_as(1)?;

        rms_normalization(ctx.pool(), input.view(), scale, self.axis, self.epsilon).into_op_result()
    }
}

pub fn log_softmax(pool: &TensorPool, input: TensorView, axis: isize) -> Result<Tensor, OpError> {
    let mut output = input.to_tensor_in(pool);
    log_softmax_in_place(&mut output, axis)?;
    Ok(output)
}

/// Grain size for parallelizing softmax.
const SOFTMAX_GRAIN_SIZE: usize = 1024;

/// Apply an operation `op` to all 1D lanes of the tensor along a given axis.
fn softmax_lanes<F: Fn(&mut [f32]) + Send + Sync>(
    output: &mut Tensor,
    axis: isize,
    apply_op: F,
) -> Result<(), OpError> {
    let resolved_axis = resolve_axis(output.ndim(), axis)?;
    if output.size(resolved_axis) == 0 {
        return Ok(());
    }

    // Make the lanes over which the operation is applied contiguous. This
    // allows the `apply_op` function to use optimized code that works with
    // contiguous slices.
    //
    // In the common case where softmax is applied over the last dimension of
    // an already-contiguous tensor, the data is already laid out in the
    // ideal order.
    if resolved_axis != output.ndim() - 1 {
        output.move_axis(resolved_axis, output.ndim() - 1);
    }
    output.make_contiguous();

    let lane_size = if output.ndim() == 1 {
        output.len()
    } else {
        output.size(output.ndim() - 1)
    };

    let grain_size = SOFTMAX_GRAIN_SIZE.max(lane_size);
    let n_grains = output.len().div_ceil(grain_size);

    let out_data = output.data_mut().unwrap();
    if n_grains == 1 {
        // Avoid parallelism overhead for small outputs
        out_data.chunks_mut(lane_size).for_each(apply_op);
    } else {
        let n_lanes_per_grain = grain_size.div_ceil(lane_size);
        out_data
            .par_chunks_mut(n_lanes_per_grain * lane_size)
            .for_each(move |grain| {
                grain.chunks_mut(lane_size).for_each(&apply_op);
            });
    }

    if resolved_axis != output.ndim() - 1 {
        output.move_axis(output.ndim() - 1, resolved_axis);
        output.make_contiguous();
    }

    Ok(())
}

pub fn log_softmax_in_place(output: &mut Tensor, axis: isize) -> Result<(), OpError> {
    softmax_lanes(output, axis, |lane| {
        // This operator computes:
        //
        //   log(exp(xi) / sum(exp(x)))
        //
        // Improve numerical stability by first subtracting max value, as we do
        // for the softmax op:
        //
        //   log(exp(xi - xmax) / sum(exp(x - xmax)))
        //
        // Then using log identities to simplify:
        //
        //   = log(exp(xi - xmax)) - log(sum(exp(x - xmax)))
        //   = xi - xmax - log(sum(exp(x - xmax)))

        let max_val = slice_max(lane);
        let log_exp_sum = lane
            .iter()
            .fold(0., |exp_sum, x| exp_sum + (x - max_val).exp())
            .ln();
        for el in lane.iter_mut() {
            *el = (*el - max_val) - log_exp_sum
        }
    })
}

#[derive(Debug)]
pub struct LogSoftmax {
    pub axis: isize,
}

impl Operator for LogSoftmax {
    fn name(&self) -> &str {
        "LogSoftmax"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as(0)?;
        log_softmax(ctx.pool(), input.view(), self.axis).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        _other: InputList,
    ) -> Result<Output, OpError> {
        let mut output = input
            .into_tensor::<f32>()
            .ok_or(OpError::IncorrectInputType)?;
        log_softmax_in_place(&mut output, self.axis)?;
        Ok(output.into())
    }
}

pub fn softmax(pool: &TensorPool, input: TensorView, axis: isize) -> Result<Tensor, OpError> {
    let mut output = input.to_tensor_in(pool);
    softmax_in_place(&mut output, axis)?;
    Ok(output)
}

pub fn softmax_in_place(output: &mut Tensor, axis: isize) -> Result<(), OpError> {
    softmax_lanes(output, axis, |lane| {
        vecmath::Softmax::new_mut(lane).dispatch();
    })?;
    Ok(())
}

#[derive(Debug)]
pub struct Softmax {
    pub axis: isize,
}

impl Operator for Softmax {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as(0)?;
        softmax(ctx.pool(), input.view(), self.axis).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        _other: InputList,
    ) -> Result<Output, OpError> {
        let mut output = input
            .into_tensor::<f32>()
            .ok_or(OpError::IncorrectInputType)?;
        softmax_in_place(&mut output, self.axis)?;
        Ok(output.into())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, NdTensorView, Tensor};
    use rten_testing::TestCases;

    use super::SOFTMAX_GRAIN_SIZE;
    use crate::ops::tests::{expect_eq_1e4, new_pool};
    use crate::ops::OpError;
    use crate::ops::{
        batch_norm, batch_norm_in_place, instance_normalization, layer_normalization, log_softmax,
        rms_normalization, softmax,
    };

    #[test]
    fn test_batch_norm() {
        #[derive(Debug)]
        struct Case {
            input: Tensor,
        }

        let cases = [
            // 4D input (eg. NCHW image)
            Case {
                input: Tensor::from_data(&[1, 2, 1, 1], vec![1.0, 2.0]),
            },
            // 3D input (eg. NCT for audio)
            Case {
                input: Tensor::from_data(&[1, 2, 1], vec![1.0, 2.0]),
            },
            // 2D input
            Case {
                input: Tensor::from_data(&[1, 2], vec![1.0, 2.0]),
            },
            // 1D input. Channel count is implicitly 1.
            Case {
                input: Tensor::from([1.0, 2.0]),
            },
        ];

        cases.test_each(|Case { input }| {
            let pool = new_pool();
            let scale = &[3.0, 3.0];
            let bias = &[0.1, 0.2];
            let mean = &[0.5, -0.5];
            let var = &[1.0, 2.0];
            let epsilon = 1e-5 as f32;

            let expected = if input.ndim() >= 2 {
                let flattened = input.reshaped([input.len()]);
                let y1 = (flattened[0] - mean[0]) / (var[0] + epsilon).sqrt() * scale[0] + bias[0];
                let y2 = (flattened[1] - mean[1]) / (var[1] + epsilon).sqrt() * scale[1] + bias[1];
                Tensor::from_data(input.shape(), vec![y1, y2])
            } else {
                input.map(|&x| (x - mean[0]) / (var[0] + epsilon).sqrt() * scale[0] + bias[0])
            };

            let result = batch_norm(
                &pool,
                input.view(),
                &scale.into(),
                &bias.into(),
                &mean.into(),
                &var.into(),
                epsilon,
            )
            .unwrap();

            expect_equal(&result, &expected).unwrap();
        })
    }

    #[test]
    fn test_batch_norm_invalid() {
        let scale = &[3.0, 3.0];
        let bias = &[0.1, 0.2];
        let mean = &[0.5, -0.5];
        let var = &[1.0, 2.0];
        let epsilon = 1e-5 as f32;
        let input = Tensor::from(5.0);

        let pool = new_pool();
        let result = batch_norm(
            &pool,
            input.view(),
            &scale.into(),
            &bias.into(),
            &mean.into(),
            &var.into(),
            epsilon,
        );

        assert_eq!(
            result,
            Err(OpError::InvalidValue("Input must have at least 1 dim"))
        );
    }

    #[test]
    fn test_batch_norm_in_place() -> Result<(), Box<dyn Error>> {
        let mut input = Tensor::from_data(&[1, 2, 1, 1], vec![1.0, 2.0]);
        let scale = &[3.0, 3.0];
        let bias = &[0.1, 0.2];
        let mean = &[0.5, -0.5];
        let var = &[1.0, 2.0];

        let epsilon = 1e-5 as f32;

        let y1 = (input[[0, 0, 0, 0]] - mean[0]) / (var[0] + epsilon).sqrt() * scale[0] + bias[0];
        let y2 = (input[[0, 1, 0, 0]] - mean[1]) / (var[1] + epsilon).sqrt() * scale[1] + bias[1];
        let expected = Tensor::from_data(&[1, 2, 1, 1], vec![y1, y2]);

        batch_norm_in_place(
            &mut input,
            &scale.into(),
            &bias.into(),
            &mean.into(),
            &var.into(),
            epsilon,
        )
        .unwrap();

        expect_equal(&input, &expected)?;

        Ok(())
    }

    #[test]
    fn test_instance_normalization() -> Result<(), Box<dyn Error>> {
        // Sample values generated using `torch.rand`.
        let input = Tensor::from([[
            [0.9562, 0.0572],
            [0.4366, 0.5655],
            [0.2017, 0.0230],
            [0.7941, 0.1554],
            [0.3226, 0.120],
        ]]);
        let scale = Tensor::from([0.0751, 0.6952, 0.5800, 0.6791, 0.9884]);
        let bias = Tensor::from([0.9993, 0.7632, 0.7679, 0.2427, 0.0728]);

        // Expected result computed with `torch.nn.functional.instance_norm`.
        // The `scale` parameter in ONNX is called `weight` in PyTorch.
        let expected = Tensor::from([[
            [1.0744, 0.9242],
            [0.0688, 1.4576],
            [1.3476, 0.1883],
            [0.9217, -0.4364],
            [1.0608, -0.9152],
        ]]);

        let pool = new_pool();
        let result =
            instance_normalization(&pool, input.view(), scale.nd_view(), bias.nd_view(), None)
                .unwrap();

        expect_eq_1e4(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_layer_normalization() {
        #[derive(Debug)]
        struct Case {
            input: Tensor,
            scale: Tensor,
            bias: Option<Tensor>,
            axis: isize,
            expected: Result<Tensor, OpError>,
        }

        let cases = [
            // Normalize last axis
            Case {
                // Sample values generated using `torch.rand`.
                input: Tensor::from([[
                    [0.9562, 0.0572],
                    [0.4366, 0.5655],
                    [0.2017, 0.0230],
                    [0.7941, 0.1554],
                    [0.3226, 0.120],
                ]]),
                scale: Tensor::from([0.0751, 0.6952]),
                bias: Some(Tensor::from([0.9993, 0.7632])),
                axis: -1,
                expected: Ok(Tensor::from([[
                    [1.0744, 0.0680],
                    [0.9243, 1.4576],
                    [1.0744, 0.0684],
                    [1.0744, 0.0680],
                    [1.0744, 0.0683],
                ]])),
            },
            // Normalize multiple axes
            Case {
                // Sample values generated using `torch.rand`.
                input: Tensor::from([[
                    [0.9562, 0.0572],
                    [0.4366, 0.5655],
                    [0.2017, 0.0230],
                    [0.7941, 0.1554],
                    [0.3226, 0.120],
                ]]),
                scale: Tensor::full(&[5, 2], 1.1),
                bias: Some(Tensor::full(&[5, 2], 0.1)),
                axis: -2,
                expected: Ok(Tensor::from([[
                    [2.2467697, -1.0079411],
                    [0.36562642, 0.83229196],
                    [-0.48479798, -1.1317577],
                    [1.6599079, -0.65242106],
                    [-0.04709549, -0.7805821],
                ]])),
            },
            // Unsupported scale shape
            Case {
                input: Tensor::from([[1., 2., 3.], [4., 5., 6.]]),
                scale: Tensor::full(&[2, 3], 1.0),
                bias: None,
                axis: -1,
                expected: Err(OpError::UnsupportedValue(
                    "`scale` shape does not match normalized axes of input",
                )),
            },
            // Unsupported bias shape
            Case {
                input: Tensor::from([[1., 2., 3.], [4., 5., 6.]]),
                scale: Tensor::from([1., 1., 1.]),
                bias: Some(Tensor::full(&[2, 3], 1.0)),
                axis: -1,
                expected: Err(OpError::UnsupportedValue(
                    "`bias` shape does not match normalized axes of input",
                )),
            },
        ];

        cases.test_each(|case| {
            let Case {
                input,
                scale,
                bias,
                axis,
                expected,
            } = case;

            let pool = new_pool();
            let result = layer_normalization(
                &pool,
                input.view(),
                scale.view(),
                bias.as_ref().map(|b| b.view()),
                *axis,
                None, /* epsilon */
            );

            match (result, expected) {
                (Ok(result), Ok(expected)) => {
                    expect_eq_1e4(&result, &expected).unwrap();
                }
                (result, expected) => assert_eq!(result, *expected),
            }
        })
    }

    fn reference_rms(input: NdTensorView<f32, 1>, scale: NdTensorView<f32, 1>) -> NdTensor<f32, 1> {
        let sum_square = input.iter().map(|x| x * x).sum::<f32>();
        let rms = (sum_square / input.len() as f32).sqrt();
        let out: Vec<f32> = input
            .iter()
            .zip(scale.iter())
            .map(|(x, scale)| (x / rms) * scale)
            .collect();
        NdTensor::from_data(input.shape(), out)
    }

    #[test]
    fn test_rms_normalization() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[10], &mut rng);
        let scale = Tensor::rand(&[10], &mut rng);
        let epsilon = 1e-5;

        let pool = new_pool();
        let result =
            rms_normalization(&pool, input.view(), scale.view(), 0, Some(epsilon)).unwrap();

        let expected = reference_rms(input.nd_view(), scale.nd_view()).into_dyn();
        expect_eq_1e4(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_log_softmax() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // 1D input
        let mut input = Tensor::from([0.1634, 0.8647, 0.6401, 0.8265, 0.0560, 0.2345]);
        let expected = Tensor::from([-2.1447, -1.4434, -1.6680, -1.4816, -2.2521, -2.0736]);
        let result = log_softmax(&pool, input.view(), 0).unwrap();
        expect_eq_1e4(&result, &expected)?;

        // Second dimension of 2D input
        input.reshape(&[2, 3]);
        let expected = Tensor::from([[-1.5319, -0.8306, -1.0552], [-0.7011, -1.4716, -1.2931]]);
        let result = log_softmax(&pool, input.view(), 1).unwrap();
        expect_eq_1e4(&result, &expected)?;

        // First dimension of 2D input
        let expected = Tensor::from([[-1.0787, -0.3684, -0.5108], [-0.4156, -1.1771, -0.9164]]);
        let result = log_softmax(&pool, input.view(), 0).unwrap();
        expect_eq_1e4(&result, &expected)?;

        // Second dimension of 2D input with multiple entries in first dim
        let matrix_input = Tensor::from([
            [0.1634, 0.8647, 0.6401, 0.8265, 0.0560],
            [0.1634, 0.8647, 0.6401, 0.8265, 0.0560],
        ]);
        let matrix_expected = Tensor::from([
            [-2.0104, -1.3091, -1.5337, -1.3473, -2.1178],
            [-2.0104, -1.3091, -1.5337, -1.3473, -2.1178],
        ]);
        let result = log_softmax(&pool, matrix_input.view(), 1).unwrap();
        expect_eq_1e4(&result, &matrix_expected)?;

        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Softmax on a 1D input
        let mut input = Tensor::from([0.1634, 0.8647, 0.6401, 0.8265, 0.0560, 0.2304]);
        let expected = Tensor::from([0.1172, 0.2362, 0.1887, 0.2274, 0.1052, 0.1253]);
        let result = softmax(&pool, input.view(), 0).unwrap();
        expect_eq_1e4(&result, &expected)?;

        // Softmax over empty axis
        let empty_vec = Tensor::zeros(&[0]);
        let result = softmax(&pool, empty_vec.view(), 0).unwrap();
        expect_eq_1e4(&result, &empty_vec)?;

        // Softmax on final dimension of 2D input
        input.reshape(&[2, 3]);
        let expected = Tensor::from([[0.2161, 0.4358, 0.3481], [0.4966, 0.2298, 0.2736]]);
        let result = softmax(&pool, input.view(), 1).unwrap();
        expect_eq_1e4(&result, &expected)?;

        // Softmax on first dimension of 2D input
        let expected = Tensor::from([[0.3400, 0.6918, 0.6010], [0.6600, 0.3082, 0.3990]]);
        let result = softmax(&pool, input.view(), 0).unwrap();
        expect_eq_1e4(&result, &expected)?;

        // Softmax on second dimension of 2D input with multiple entries in
        // first dim
        let matrix_input = Tensor::from([
            [0.1634, 0.8647, 0.6401, 0.8265, 0.0560],
            [0.1634, 0.8647, 0.6401, 0.8265, 0.0560],
        ]);
        let matrix_expected = Tensor::from([
            [0.1339, 0.2701, 0.2157, 0.2599, 0.1203],
            [0.1339, 0.2701, 0.2157, 0.2599, 0.1203],
        ]);
        let result = softmax(&pool, matrix_input.view(), 1).unwrap();
        expect_eq_1e4(&result, &matrix_expected)?;

        Ok(())
    }

    // Test softmax with non-contiguous input.
    #[test]
    fn test_softmax_transposed() -> Result<(), Box<dyn Error>> {
        let mut input = Tensor::from_data(
            &[4, 4],
            vec![
                0.6427, 0.7435, 0.9762, 0.0611, 0.1249, 0.9742, 0.5826, 0.4704, 0.1420, 0.8376,
                0.6692, 0.7090, 0.2448, 0.9083, 0.2881, 0.4971,
            ],
        );
        let expected = Tensor::from_data(
            &[4, 4],
            vec![
                0.3480, 0.2073, 0.2109, 0.2337, 0.2204, 0.2776, 0.2421, 0.2599, 0.3433, 0.2316,
                0.2525, 0.1725, 0.1677, 0.2525, 0.3205, 0.2593,
            ],
        );

        input.permute(&[1, 0]);
        let pool = new_pool();
        let result = softmax(&pool, input.view(), 1).unwrap();

        expect_eq_1e4(&result, &expected)?;

        Ok(())
    }

    // Test softmax with some additional input sizes and axis dimensions.
    // These tests don't check the individual output values in detail, but they
    // do check the shape and that each lane sums to 1.
    #[test]
    fn test_softmax_sizes() {
        let pool = new_pool();

        let check_result = |result: Tensor<f32>| {
            for lane in result.lanes(1) {
                assert!((lane.sum::<f32>() - 1.0).abs() < 0.001);
            }
        };

        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::rand(&[1, 1, 3, 3], &mut rng);
        let result = softmax(&pool, input.view(), 1).unwrap();
        check_result(result);

        // "Large" output, where output size exceeds the parallelism grain size.
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::rand(&[4, SOFTMAX_GRAIN_SIZE / 2], &mut rng);
        let result = softmax(&pool, input.view(), 1).unwrap();
        check_result(result);
    }
}
