use rayon::prelude::*;

use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};
use rten_vecmath::vec_softmax_in_place;
use smallvec::SmallVec;

use crate::ops::reduce::reduce_inverse_rms;
use crate::ops::{add_in_place, mul_in_place, reduce_mean, static_dims, sub};
use crate::ops::{resolve_axis, InputList, IntoOpResult, OpError, Operator, Output, OutputList};
use crate::slice_reductions::{slice_max, slice_sum};
use crate::tensor_pool::{AutoReturn, TensorPool};

/// Perform in-place batch normalization on the `NC*` tensor `out`.
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
    if input.ndim() < 3 {
        return Err(OpError::InvalidValue("Input must have at least 3 dims"));
    }

    let batch = input.size(0);
    let chans = input.size(1);

    for n in 0..batch {
        for c in 0..chans {
            let chan_mean = mean[[c]];
            let chan_var = var[[c]];
            let chan_scale = scale[[c]];
            let chan_bias = bias[[c]];

            // The batch norm formula, from the ONNX spec, is:
            //
            // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + bias
            //
            // It has been rewritten here to simplify the inner loop below.
            let scaled_std_dev_reciprocal = chan_scale / (chan_var + epsilon).sqrt();

            input
                .slice_mut([n, c])
                .apply(|el| (*el - chan_mean) * scaled_std_dev_reciprocal + chan_bias);
        }
    }

    Ok(())
}

/// Perform batch normalization on the `NC*` tensor `input`.
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;

        let scale = inputs.require_as(1)?;
        let scale = static_dims!(scale, 1)?;

        let bias = inputs.require_as(2)?;
        let bias = static_dims!(bias, 1)?;

        let mean = inputs.require_as(3)?;
        let mean = static_dims!(mean, 1)?;

        let var = inputs.require_as(4)?;
        let var = static_dims!(var, 1)?;

        batch_norm(pool, input, &scale, &bias, &mean, &var, self.epsilon).into_op_result()
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
    let &[batch, chans, ..] = input.shape() else {
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

    // Needed for `slice_sum` below.
    input.make_contiguous();

    for n in 0..batch {
        for c in 0..chans {
            let mut slice = input.slice_mut([n, c]);
            let chan_scale = scale[[c]];
            let chan_bias = bias[[c]];
            let chan_mean = slice_sum(slice.data().unwrap()) / slice.len() as f32;
            let chan_variance = slice
                .iter()
                .map(|x| {
                    let diff = *x - chan_mean;
                    diff * diff
                })
                .sum::<f32>()
                / slice.len() as f32;

            // The instance norm formula, from the ONNX spec, is:
            //
            // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + bias
            //
            // It has been rewritten here to optimize the inner loop.
            let scaled_std_dev_reciprocal = chan_scale / (chan_variance + epsilon).sqrt();

            slice.apply(|x| (*x - chan_mean) * scaled_std_dev_reciprocal + chan_bias)
        }
    }

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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;

        let scale = inputs.require_as(1)?;
        let scale = static_dims!(scale, 1)?;

        let bias = inputs.require_as(2)?;
        let bias = static_dims!(bias, 1)?;

        instance_normalization(pool, input, scale, bias, self.epsilon).into_op_result()
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

pub fn layer_normalization(
    pool: &TensorPool,
    input: TensorView,
    scale: TensorView,
    bias: Option<TensorView>,
    axis: isize,
    epsilon: Option<f32>,
) -> Result<Tensor, OpError> {
    if !scale.can_broadcast_to(input.shape()) {
        return Err(OpError::IncompatibleInputShapes(
            "`scale` cannot be broadcast to input shape",
        ));
    }
    if let Some(bias) = bias.as_ref() {
        if !bias.can_broadcast_to(input.shape()) {
            return Err(OpError::IncompatibleInputShapes(
                "`bias` cannot be broadcast to input shape",
            ));
        }
    }

    let epsilon = epsilon.unwrap_or(1e-5);
    let resolved_axis = resolve_axis(input.ndim(), axis)?;
    let normalized_axes: SmallVec<[i32; 5]> = (resolved_axis..input.ndim())
        .map(|axis| axis as i32)
        .collect();

    // First step: standardize input elements to have unit mean and variance.
    let mean = reduce_mean(
        pool,
        input.view(),
        Some(normalized_axes.as_slice()),
        true, /* keep_dims */
    )?
    .auto_return(pool);
    let mut normalized = sub(pool, input, mean.view())?.auto_return(pool);

    let inverse_std_dev = reduce_inverse_rms(
        pool,
        normalized.view(),
        Some(normalized_axes.as_slice()),
        true, /* keep_dims */
        epsilon,
    )?
    .auto_return(pool);
    mul_in_place(normalized.view_mut(), inverse_std_dev.view());

    // Second step: Shift and scale input.
    mul_in_place(normalized.view_mut(), scale);
    if let Some(bias) = bias {
        add_in_place(normalized.view_mut(), bias);
    }

    Ok(normalized.take())
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;
        let scale = inputs.require_as(1)?;
        let bias = inputs.get_as(2)?;

        layer_normalization(pool, input.view(), scale, bias, self.axis, self.epsilon)
            .into_op_result()
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;
        log_softmax(pool, input.view(), self.axis).into_op_result()
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
    softmax_lanes(output, axis, vec_softmax_in_place)?;
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;
        softmax(pool, input.view(), self.axis).into_op_result()
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
    use rten_tensor::Tensor;

    use super::SOFTMAX_GRAIN_SIZE;
    use crate::ops::tests::{expect_eq_1e4, new_pool};
    use crate::ops::OpError;
    use crate::ops::{
        batch_norm, batch_norm_in_place, instance_normalization, layer_normalization, log_softmax,
        softmax,
    };

    #[test]
    fn test_batch_norm() -> Result<(), Box<dyn Error>> {
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
        ];

        let pool = new_pool();
        for Case { input } in cases {
            let scale = &[3.0, 3.0];
            let bias = &[0.1, 0.2];
            let mean = &[0.5, -0.5];
            let var = &[1.0, 2.0];

            let epsilon = 1e-5 as f32;

            let flattened = input.reshaped([input.len()]);

            let y1 = (flattened[[0]] - mean[0]) / (var[0] + epsilon).sqrt() * scale[0] + bias[0];
            let y2 = (flattened[[1]] - mean[1]) / (var[1] + epsilon).sqrt() * scale[1] + bias[1];
            let expected = Tensor::from_data(input.shape(), vec![y1, y2]);
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

            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_batch_norm_invalid() {
        let scale = &[3.0, 3.0];
        let bias = &[0.1, 0.2];
        let mean = &[0.5, -0.5];
        let var = &[1.0, 2.0];
        let epsilon = 1e-5 as f32;
        let input = Tensor::zeros(&[2]);

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
            Err(OpError::InvalidValue("Input must have at least 3 dims"))
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
    fn test_layer_normalization() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Sample values generated using `torch.rand`.
        let input = Tensor::from([[
            [0.9562, 0.0572],
            [0.4366, 0.5655],
            [0.2017, 0.0230],
            [0.7941, 0.1554],
            [0.3226, 0.120],
        ]]);
        let scale = Tensor::from([0.0751, 0.6952]);
        let bias = Tensor::from([0.9993, 0.7632]);

        let result = layer_normalization(
            &pool,
            input.view(),
            scale.view(),
            Some(bias.view()),
            -1,   /* axis */
            None, /* epsilon */
        )
        .unwrap();

        let expected = Tensor::from([[
            [1.0744, 0.0680],
            [0.9243, 1.4576],
            [1.0744, 0.0684],
            [1.0744, 0.0680],
            [1.0744, 0.0683],
        ]]);
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
