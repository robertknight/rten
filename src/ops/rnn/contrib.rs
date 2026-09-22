//! ONNX Runtime contrib recurrent operators.

use crate::shift_cast::ShiftCast;
use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{CowTensor, NdTensor, NdTensorView, Tensor, TensorView};

use crate::buffer_pool::{AutoReturn, BufferPool, PoolRef};
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext, check_eq, static_dims,
};
use crate::ops::binary_elementwise::add_in_place;
use crate::ops::map_value_view;
use crate::ops::matmul::{OutputScale, cast_scale, matmul_integer};
use crate::ops::quantize::{DynamicQuantizeOutput, dynamic_quantize_linear};
use crate::value::{DataType, ValueType, ValueView};

use super::{Direction, LSTM_GATES, lstm_step, sequence_for_dir};

/// A `W` or `R` input of [`DynamicQuantizeLSTM`].
#[derive(Clone)]
struct QuantizedWeights<'a> {
    /// Weights of shape `[directions, in_size, 4 * hidden_size]` where `in_size`
    /// is the input size for `W` and the hidden size for `R`.
    weights: NdTensorView<'a, i8, 3>,
    /// Scale with shape `[directions]` for per-tensor quantization or
    /// `[directions, 4 * hidden_size]` for per-channel quantization along the
    /// `in_size` axis.
    scale: TensorView<'a, f32>,
    /// Zero point with same shape as `scale`.
    zero_point: Option<TensorView<'a, i8>>,
}

/// One direction of [`QuantizedWeights`].
struct DirectionWeights<'a> {
    /// Shape `[in_size, 4 * hidden_size]`.
    weights: NdTensorView<'a, i8, 2>,
    /// Scale of size `[4 * hidden_size]`.
    scale: NdTensorView<'a, f32, 1>,
    /// Zero point of shape `[4 * hidden_size]` or a scalar.
    zero_point: Option<TensorView<'a, i8>>,
}

impl QuantizedWeights<'_> {
    /// Check that the quantization parameters match the weights.
    ///
    /// `name` identifies the weights in error messages.
    fn check_shapes(&self, name: &str) -> Result<(), OpError> {
        let [num_directions, _in_size, gates] = self.weights.shape();

        match self.scale.ndim() {
            1 => check_eq!(
                self.scale.shape(),
                [num_directions],
                format!("{name} scale must have shape [directions]")
            )?,
            2 => check_eq!(
                self.scale.shape(),
                [num_directions, gates],
                format!("{name} scale must have shape [directions, 4 * hidden_size]")
            )?,
            _ => {
                return Err(OpError::unsupported_value(format!(
                    "{name} scale must have 1 or 2 dims"
                )));
            }
        }

        if let Some(zero_point) = self.zero_point.as_ref() {
            check_eq!(
                zero_point.shape(),
                self.scale.shape(),
                format!("{name} scale and zero point must have the same shape")
            )?;
        }

        Ok(())
    }

    /// Extract one direction, broadcasting a per-tensor scale to the gate axis
    /// so the result is always a vector.
    fn direction(&self, dir: usize) -> DirectionWeights<'_> {
        let weights = self.weights.slice(dir);
        let gates = weights.size(1);

        let scale = self.scale.slice(dir);
        let scale = if scale.ndim() == 0 {
            scale.broadcast([gates])
        } else {
            scale.nd_view::<1>()
        };

        DirectionWeights {
            weights,
            scale,
            zero_point: self.zero_point.as_ref().map(|zp| zp.slice(dir)),
        }
    }
}

/// Convert a `W` or `R` input and its zero point to i8.
fn shift_cast_weights<'a, T>(
    pool: &'a BufferPool,
    weights: TensorView<'a, T>,
    zero_point: Option<TensorView<'a, T>>,
) -> (
    PoolRef<'a, CowTensor<'a, i8>>,
    Option<PoolRef<'a, CowTensor<'a, i8>>>,
)
where
    TensorView<'a, T>: ShiftCast<CowTensor<'a, i8>>,
{
    (
        weights.shift_cast_in(pool).auto_return(pool),
        zero_point.map(|zp| zp.shift_cast_in(pool).auto_return(pool)),
    )
}

/// Multiply a float matrix by a quantized one.
///
/// `a` has shape `[m, k]` and `b` has shape `[k, n]`. `b_scale` has length `n`
/// and `b_zero_point`, if given, is a scalar or a vector of length `n`.
fn dynamic_quantize_matmul(
    pool: &BufferPool,
    a: NdTensorView<f32, 2>,
    b: NdTensorView<i8, 2>,
    b_scale: NdTensorView<f32, 1>,
    b_zero_point: Option<TensorView<i8>>,
) -> Result<NdTensor<f32, 2>, OpError> {
    let DynamicQuantizeOutput {
        quantized: a_quant,
        scale: a_scale,
        zero_point: a_zero_point,
    } = dynamic_quantize_linear::<u8>(pool, a.as_dyn())?;

    let product = matmul_integer(
        pool,
        a_quant.view(),
        b.as_dyn(),
        Some(a_zero_point.view()),
        b_zero_point,
        None,
    )?;

    let a_scale = a_scale.item().copied().unwrap_or(1.);
    let scale: Vec<f32> = b_scale.iter().map(|&b_scale| a_scale * b_scale).collect();
    let scale = OutputScale::Vector(NdTensorView::from_data([scale.len()], scale.as_slice()));
    let output = cast_scale(pool, product, scale)?;

    Ok(output.try_into().unwrap())
}

/// Compute the output for a single LSTM layer with quantized weights. The
/// input is dynamically quantized, multiplied with the weights using int8
/// matrix multiplication and the output is dequantized.
///
/// `input` has shape `[sequence_length, batch, input_size]`.
///
/// `bias` has shape `[directions, 8 * hidden_size]`. The last dimension is a
/// concatenation of input biases for the input, output, forget and cell gates
/// followed by hidden biases for the same gates.
///
/// `initial_hidden` and `initial_cell` have shape `[directions, batch, hidden_size]`.
#[allow(clippy::too_many_arguments)]
fn dynamic_quantize_lstm(
    pool: &BufferPool,
    direction: Direction,
    input: NdTensorView<f32, 3>,
    weights: QuantizedWeights,
    recurrent_weights: QuantizedWeights,
    bias: Option<NdTensorView<f32, 2>>,
    initial_hidden: Option<NdTensorView<f32, 3>>,
    initial_cell: Option<NdTensorView<f32, 3>>,
) -> Result<Vec<Tensor>, OpError> {
    let [seq_len, batch, input_size] = input.shape();
    let num_directions = direction.num_directions();

    let hidden_x4 = weights.weights.size(2);
    if !hidden_x4.is_multiple_of(4) {
        return Err(OpError::invalid_value(
            "weights dim 2 must be 4 * hidden_size",
        ));
    }
    let hidden_size = hidden_x4 / 4;
    check_eq!(
        weights.weights.shape(),
        [num_directions, input_size, hidden_size * 4],
        "weights.shape() != [num_directions, input_size, hidden_size * 4]"
    )?;
    check_eq!(
        recurrent_weights.weights.shape(),
        [num_directions, hidden_size, hidden_size * 4],
        "recurrent_weights.shape() != [num_directions, hidden_size, hidden_size * 4]"
    )?;
    weights.check_shapes("weights")?;
    recurrent_weights.check_shapes("recurrent_weights")?;

    if let Some(bias) = bias.as_ref() {
        check_eq!(bias.shape(), [num_directions, hidden_size * 8])?;
    }

    if let Some(initial_hidden) = initial_hidden.as_ref() {
        check_eq!(initial_hidden.shape(), [num_directions, batch, hidden_size])?;
    }

    if let Some(initial_cell) = initial_cell.as_ref() {
        check_eq!(initial_cell.shape(), [num_directions, batch, hidden_size])?;
    }

    let input_mat = input
        .reshaped_in(pool, [seq_len * batch, input_size])
        .auto_return(pool);

    let n_gates = LSTM_GATES;

    let mut cell = initial_cell
        .map(|t| t.to_tensor_in(pool))
        .unwrap_or_else(|| NdTensor::zeros_in(pool, [num_directions, batch, hidden_size]));
    let mut hidden = initial_hidden
        .map(|t| t.to_tensor_in(pool))
        .unwrap_or_else(|| NdTensor::zeros_in(pool, [num_directions, batch, hidden_size]));

    let mut hidden_seq = NdTensor::uninit_in(pool, [seq_len, num_directions, batch, hidden_size]);

    for dir in 0..num_directions {
        let input_weights = weights.direction(dir);
        let hidden_weights = recurrent_weights.direction(dir);

        let fused_bias = bias.as_ref().map(|b| {
            let mut fused = b.slice((dir, ..(n_gates * hidden_size))).to_tensor_in(pool);
            add_in_place(
                fused.as_dyn_mut(),
                b.slice((dir, (n_gates * hidden_size)..)).as_dyn(),
            );
            fused.auto_return(pool)
        });

        // Compute the input projection for the whole sequence in a single large
        // matmul.
        let mut input_gates = dynamic_quantize_matmul(
            pool,
            input_mat.view(),
            input_weights.weights.view(),
            input_weights.scale.view(),
            input_weights.zero_point.clone(),
        )?
        .into_shape([seq_len, batch, n_gates * hidden_size])
        .auto_return(pool);

        for seq in sequence_for_dir(direction, dir, seq_len) {
            let hidden_item = hidden.slice([dir]);
            let hidden_gates = dynamic_quantize_matmul(
                pool,
                hidden_item,
                hidden_weights.weights.view(),
                hidden_weights.scale.view(),
                hidden_weights.zero_point.clone(),
            )?
            .auto_return(pool);

            let mut gates = input_gates.slice_mut(seq);
            add_in_place(gates.as_dyn_mut(), hidden_gates.as_dyn());
            if let Some(fused_bias) = fused_bias.as_ref() {
                add_in_place(gates.as_dyn_mut(), fused_bias.as_dyn());
            }

            lstm_step(
                hidden_size,
                gates.view_mut(),
                hidden.slice_mut([dir]),
                cell.slice_mut([dir]),
                hidden_seq.slice_mut([seq, dir]),
            );
        }
    }

    // Safety: The loop above wrote to every element of `hidden_seq`.
    let hidden_seq = unsafe { hidden_seq.assume_init() };

    Ok([hidden_seq.into_dyn(), hidden.into_dyn(), cell.into_dyn()].into())
}

/// Long Short-Term Memory operator with quantized weights.
///
/// Differs from [`LSTM`](super::LSTM) in that `W` and `R` are quantized and
/// stored with the gate axis last rather than first. The input and hidden
/// states are dynamically quantized before multiplication with the weights.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DynamicQuantizeLSTM>.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct DynamicQuantizeLSTM {
    pub direction: Direction,

    #[allow(unused)]
    pub hidden_size: usize, // Currently inferred from operator inputs.
}

impl Operator for DynamicQuantizeLSTM {
    fn name(&self) -> &str {
        "DynamicQuantizeLSTM"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(12)
    }

    fn max_outputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let weights = inputs.require(1)?;
        let recurrent_weights = inputs.require(2)?;
        let bias = inputs.get_as(3)?;
        let _seq_len = inputs.get_as::<TensorView<i32>>(4)?;
        let initial_hidden = inputs.get_as(5)?;
        let initial_cell = inputs.get_as(6)?;

        if let Some(peepholes) = inputs.get_as::<TensorView>(7)?
            && !peepholes.is_empty()
        {
            return Err(OpError::unsupported_value(
                "peephole weights are not supported",
            ));
        }

        let weight_scale = inputs.require_as(8)?;
        let weight_zero_point = inputs.get(9);
        let rec_weight_scale = inputs.require_as(10)?;
        let rec_weight_zero_point = inputs.get(11);

        if weights.dtype() != recurrent_weights.dtype() {
            return Err(OpError::invalid_value("W and R must have the same type"));
        }

        let pool = ctx.pool();

        let (w, w_zero_point) = map_value_view!(weights, w, [UInt8Tensor, Int8Tensor], {
            shift_cast_weights(
                pool,
                w,
                weight_zero_point.map(|zp| zp.try_into()).transpose()?,
            )
        });
        let (r, r_zero_point) = map_value_view!(recurrent_weights, r, [UInt8Tensor, Int8Tensor], {
            shift_cast_weights(
                pool,
                r,
                rec_weight_zero_point.map(|zp| zp.try_into()).transpose()?,
            )
        });

        dynamic_quantize_lstm(
            pool,
            self.direction,
            input,
            QuantizedWeights {
                weights: static_dims!(w, 3, "dir, input, hidden x 4")?,
                scale: weight_scale,
                zero_point: w_zero_point.as_ref().map(|zp| zp.view()),
            },
            QuantizedWeights {
                weights: static_dims!(r, 3, "dir, hidden, hidden x 4")?,
                scale: rec_weight_scale,
                zero_point: r_zero_point.as_ref().map(|zp| zp.view()),
            },
            bias,
            initial_hidden,
            initial_cell,
        )
        .into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some(OutputTypeList::from_slice(&[
            OutputType::Fixed(ValueType::Tensor(DataType::Float)),
            OutputType::Fixed(ValueType::Tensor(DataType::Float)),
            OutputType::Fixed(ValueType::Tensor(DataType::Float)),
        ]))
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    DynamicQuantizeLSTM,
    op,
    shape_ops::LSTM {
        direction: op.direction.into(),
    }
);

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal_with_tolerance;
    use rten_tensor::{NdTensor, NdTensorView, Tensor};
    use rten_testing::TestCases;

    use super::{DynamicQuantizeLSTM, QuantizedWeights, dynamic_quantize_lstm};
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpError, OpRunContext, Operator, OutputMask};
    use crate::ops::{Direction, lstm};
    use crate::shift_cast::ShiftCast;
    use crate::value::{Value, ValueView};

    /// Minimum and maximum quantized weight values.
    ///
    /// Weights are restricted to the i7 range to avoid saturation on x86
    /// without VNNI. See notes in rten-gemm source.
    const WEIGHT_MIN: i8 = -64;
    const WEIGHT_MAX: i8 = 63;

    /// Result of quantizing a float tensor with [`quantize_per_channel`].
    struct Quantized {
        /// Quantized values, with the same shape as the input.
        values: NdTensor<i8, 2>,
        /// Scale for each element of the last axis.
        scale: NdTensor<f32, 1>,
        /// Zero point for each element of the last axis.
        zero_point: NdTensor<i8, 1>,
        /// Floats represented by `values`.
        dequantized: NdTensor<f32, 2>,
    }

    /// Quantize `x` to the range `[WEIGHT_MIN, WEIGHT_MAX]`.
    fn quantize(x: f32, scale: f32, zero_point: i8) -> i8 {
        ((x / scale).round_ties_even() + zero_point as f32)
            .clamp(WEIGHT_MIN as f32, WEIGHT_MAX as f32) as i8
    }

    /// Dequantize a value produced by [`quantize`].
    fn dequantize(q: i8, scale: f32, zero_point: i8) -> f32 {
        (q as i32 - zero_point as i32) as f32 * scale
    }

    /// Quantize `x` to i8 per element of the last axis.
    fn quantize_per_channel(x: NdTensorView<f32, 2>) -> Quantized {
        let channels = x.size(x.ndim() - 1);
        let mut scales = NdTensor::zeros([channels]);
        let mut zero_points = NdTensor::<i8, 1>::zeros([channels]);

        let mut min = vec![0f32; channels];
        let mut max = vec![0f32; channels];
        for (idx, &val) in x.indices().zip(x.iter()) {
            let c = idx[idx.len() - 1];
            min[c] = min[c].min(val);
            max[c] = max[c].max(val);
        }
        for c in 0..channels {
            let scale = (max[c] - min[c]) / (WEIGHT_MAX - WEIGHT_MIN) as f32;
            scales[[c]] = scale;
            zero_points[[c]] = (WEIGHT_MIN as f32 - min[c] / scale).round() as i8;
        }

        let mut quantized = NdTensor::<i8, 2>::zeros(x.shape());
        let mut dequantized = NdTensor::zeros(x.shape());
        for (idx, &val) in x.indices().zip(x.iter()) {
            let c = idx[idx.len() - 1];
            let (scale, zero_point) = (scales[[c]], zero_points[[c]]);
            let q = quantize(val, scale, zero_point);
            quantized[idx] = q;
            dequantized[idx] = dequantize(q, scale, zero_point);
        }

        Quantized {
            values: quantized,
            scale: scales,
            zero_point: zero_points,
            dequantized,
        }
    }

    #[derive(Debug)]
    struct Case {
        per_channel: bool,
        with_bias: bool,
        with_initial_state: bool,
        /// Shift-cast weights and zero points to u8 before running the op.
        u8_weights: bool,
    }

    #[test]
    fn test_dynamic_quantize_lstm() {
        let cases = [
            Case {
                per_channel: true,
                with_bias: true,
                with_initial_state: false,
                u8_weights: false,
            },
            Case {
                per_channel: false,
                with_bias: true,
                with_initial_state: false,
                u8_weights: false,
            },
            Case {
                per_channel: true,
                with_bias: false,
                with_initial_state: false,
                u8_weights: false,
            },
            Case {
                per_channel: true,
                with_bias: true,
                with_initial_state: true,
                u8_weights: false,
            },
            Case {
                per_channel: true,
                with_bias: true,
                with_initial_state: true,
                u8_weights: true,
            },
        ];

        cases.test_each(|case| {
            let &Case {
                per_channel,
                with_bias,
                with_initial_state,
                u8_weights,
            } = case;

            let mut rng = XorShiftRng::new(1234);

            let seq_len = 3;
            let batch = 2;
            let input_size = 4;
            let hidden_size = 5;
            let input = NdTensor::rand([seq_len, batch, input_size], &mut rng);
            let weights = NdTensor::rand([input_size, 4 * hidden_size], &mut rng);
            let rec_weights = NdTensor::rand([hidden_size, 4 * hidden_size], &mut rng);

            let Quantized {
                values: q_weights,
                scale: w_scale,
                zero_point: w_zero_point,
                dequantized: deq_weights,
            } = quantize_per_channel(weights.view());

            let Quantized {
                values: q_rec_weights,
                scale: r_scale,
                zero_point: r_zero_point,
                dequantized: deq_rec_weights,
            } = quantize_per_channel(rec_weights.view());

            let (w_scale, w_zero_point, r_scale, r_zero_point) = if per_channel {
                (
                    w_scale.with_new_axis(0).into_dyn(),
                    w_zero_point.into_shape([1, 4 * hidden_size]).into_dyn(),
                    r_scale.with_new_axis(0).into_dyn(),
                    r_zero_point.into_shape([1, 4 * hidden_size]).into_dyn(),
                )
            } else {
                // Per-tensor quantization has one scale per direction.
                (
                    Tensor::from([w_scale[[0]]]),
                    Tensor::from([w_zero_point[[0]]]),
                    Tensor::from([r_scale[[0]]]),
                    Tensor::from([r_zero_point[[0]]]),
                )
            };

            let (q_weights, deq_weights, q_rec_weights, deq_rec_weights) = if per_channel {
                (q_weights, deq_weights, q_rec_weights, deq_rec_weights)
            } else {
                // Per-tensor: requantize using the first channel's parameters.
                let requant = |x: &NdTensor<f32, 2>, scale: f32, zp: i8| {
                    let q = x.map(|&v| quantize(v, scale, zp));
                    let deq = q.map(|&q| dequantize(q, scale, zp));
                    (q, deq)
                };
                let (qw, dw) = requant(&weights, w_scale[[0]], w_zero_point[[0]]);
                let (qr, dr) = requant(&rec_weights, r_scale[[0]], r_zero_point[[0]]);
                (qw, dw, qr, dr)
            };

            let to_value = |t: Tensor<i8>| -> Value {
                if u8_weights {
                    t.map(|&x| ShiftCast::<u8>::shift_cast(x)).into()
                } else {
                    t.into()
                }
            };
            let q_weights = to_value(q_weights.with_new_axis(0).into_dyn());
            let q_rec_weights = to_value(q_rec_weights.with_new_axis(0).into_dyn());
            let w_zero_point = to_value(w_zero_point);
            let r_zero_point = to_value(r_zero_point);

            let bias = with_bias.then(|| NdTensor::rand([1, 8 * hidden_size], &mut rng));
            let initial_hidden =
                with_initial_state.then(|| NdTensor::rand([1, batch, hidden_size], &mut rng));
            let initial_cell =
                with_initial_state.then(|| NdTensor::rand([1, batch, hidden_size], &mut rng));

            // Compute reference result using un-quantized LSTM op.
            let pool = BufferPool::new();
            let expected = lstm(
                &pool,
                Direction::Forward,
                input.view(),
                // Reorder `[input, n_gates * hidden]` => `[dir, n_gates * hidden, input`]
                deq_weights.with_new_axis(0).permuted([0, 2, 1]),
                // Reorder `[hidden, n_gates * hidden]` => `[dir, n_gates * hidden, hidden`]
                deq_rec_weights.with_new_axis(0).permuted([0, 2, 1]),
                bias.as_ref().map(|b| b.view()),
                initial_hidden.as_ref().map(|h| h.view()),
                initial_cell.as_ref().map(|c| c.view()),
            )
            .unwrap();

            let op = DynamicQuantizeLSTM {
                direction: Direction::Forward,
                hidden_size,
            };
            let inputs: Vec<Option<ValueView>> = vec![
                Some(input.view().into()),
                Some(q_weights.as_view()),
                Some(q_rec_weights.as_view()),
                bias.as_ref().map(|b| b.view().into()),
                None, // sequence_lens
                initial_hidden.as_ref().map(|h| h.view().into()),
                initial_cell.as_ref().map(|c| c.view().into()),
                None, // P
                Some(w_scale.view().into()),
                Some(w_zero_point.as_view()),
                Some(r_scale.view().into()),
                Some(r_zero_point.as_view()),
            ];
            let inputs = InputList::from_optional(&inputs);
            let ctx = OpRunContext::new(&pool, &inputs, OutputMask::all_used(3));
            let outputs = op.run(&ctx).unwrap();

            for (actual, expected) in outputs.into_iter().zip(expected.iter()) {
                let actual: Tensor<f32> = actual.try_into().unwrap();
                expect_equal_with_tolerance(&actual, expected, 1e-2, 0.).unwrap();
            }
        })
    }

    #[test]
    fn test_dynamic_quantize_lstm_invalid() {
        let op = DynamicQuantizeLSTM {
            direction: Direction::Forward,
            hidden_size: 2,
        };
        let pool = BufferPool::new();

        let input = NdTensor::<f32, 3>::zeros([1, 1, 2]);
        let weights = NdTensor::<u8, 3>::zeros([1, 2, 8]);
        let rec_weights = NdTensor::<u8, 3>::zeros([1, 2, 8]);
        let scale = NdTensor::from([0.1]);
        let zero_point = NdTensor::from([0u8]);
        let peepholes = NdTensor::<f32, 2>::zeros([1, 6]);

        let inputs: Vec<Option<ValueView>> = vec![
            Some(input.view().into()),
            Some(weights.view().into()),
            Some(rec_weights.view().into()),
            None,
            None,
            None,
            None,
            Some(peepholes.view().into()),
            Some(scale.view().into()),
            Some(zero_point.view().into()),
            Some(scale.view().into()),
            Some(zero_point.view().into()),
        ];
        let inputs = InputList::from_optional(&inputs);
        let ctx = OpRunContext::new(&pool, &inputs, OutputMask::all_used(3));

        assert_eq!(
            op.run(&ctx).err(),
            Some(OpError::unsupported_value(
                "peephole weights are not supported"
            ))
        );
    }

    #[test]
    fn test_dynamic_quantize_lstm_invalid_input_shapes() {
        const SEQ_LEN: usize = 5;
        const BATCH: usize = 2;
        const HIDDEN: usize = 3;
        const FEATURES: usize = 4;

        /// Shapes of the inputs to [`dynamic_quantize_lstm`].
        #[derive(Debug)]
        struct Shapes {
            input: [usize; 3],
            weights: [usize; 3],
            weight_scale: Vec<usize>,
            weight_zero_point: Vec<usize>,
            recurrent_weights: [usize; 3],
            rec_weight_scale: Vec<usize>,
            rec_weight_zero_point: Vec<usize>,
            bias: [usize; 2],
            initial_hidden: [usize; 3],
            initial_cell: [usize; 3],
        }

        /// Return valid input shapes, using per-channel weight quantization.
        fn valid_shapes(dir: Direction) -> Shapes {
            let dirs = dir.num_directions();
            Shapes {
                input: [SEQ_LEN, BATCH, FEATURES],
                weights: [dirs, FEATURES, 4 * HIDDEN],
                weight_scale: vec![dirs, 4 * HIDDEN],
                weight_zero_point: vec![dirs, 4 * HIDDEN],
                recurrent_weights: [dirs, HIDDEN, 4 * HIDDEN],
                rec_weight_scale: vec![dirs, 4 * HIDDEN],
                rec_weight_zero_point: vec![dirs, 4 * HIDDEN],
                bias: [dirs, 8 * HIDDEN],
                initial_hidden: [dirs, BATCH, HIDDEN],
                initial_cell: [dirs, BATCH, HIDDEN],
            }
        }

        let weights_err = || {
            OpError::incompatible_input_shapes(
                "weights.shape() != [num_directions, input_size, hidden_size * 4]",
            )
        };

        let rec_weights_err = || {
            OpError::incompatible_input_shapes(
                "recurrent_weights.shape() != [num_directions, hidden_size, hidden_size * 4]",
            )
        };

        #[derive(Debug)]
        struct Case {
            dir: Direction,

            /// Change applied to a set of otherwise-valid input shapes.
            invalidate: fn(&mut Shapes),

            expected: OpError,
        }

        let cases = [
            // Weight dim 2 is not a multiple of the number of gates.
            Case {
                dir: Direction::Forward,
                invalidate: |s| s.weights[2] += 1,
                expected: OpError::invalid_value("weights dim 2 must be 4 * hidden_size"),
            },
            // Inputs disagree with the `direction` attribute.
            Case {
                dir: Direction::Bidirectional,
                invalidate: |s| s.weights[0] = 1,
                expected: weights_err(),
            },
            Case {
                dir: Direction::Forward,
                invalidate: |s| s.recurrent_weights[0] = 2,
                expected: rec_weights_err(),
            },
            Case {
                dir: Direction::Forward,
                invalidate: |s| s.bias[0] = 2,
                expected: OpError::incompatible_input_shapes(
                    "bias.shape() != [num_directions, hidden_size * 8]",
                ),
            },
            // Inputs disagree about the input size.
            Case {
                dir: Direction::Forward,
                invalidate: |s| s.weights[1] += 1,
                expected: weights_err(),
            },
            // Inputs disagree about the hidden size.
            Case {
                dir: Direction::Forward,
                invalidate: |s| s.recurrent_weights[1] += 1,
                expected: rec_weights_err(),
            },
            Case {
                dir: Direction::Forward,
                invalidate: |s| s.initial_hidden[2] += 1,
                expected: OpError::incompatible_input_shapes(
                    "initial_hidden.shape() != [num_directions, batch, hidden_size]",
                ),
            },
            // Inputs disagree about the batch size.
            Case {
                dir: Direction::Forward,
                invalidate: |s| s.initial_cell[1] += 1,
                expected: OpError::incompatible_input_shapes(
                    "initial_cell.shape() != [num_directions, batch, hidden_size]",
                ),
            },
            // Quantization parameters do not match the weights they belong to.
            Case {
                dir: Direction::Bidirectional,
                invalidate: |s| {
                    s.weight_scale = vec![1, 4 * HIDDEN];
                    s.weight_zero_point = vec![1, 4 * HIDDEN];
                },
                expected: OpError::incompatible_input_shapes(
                    "weights scale must have shape [directions, 4 * hidden_size]",
                ),
            },
            Case {
                dir: Direction::Forward,
                invalidate: |s| {
                    s.weight_scale = vec![1, 4 * HIDDEN + 1];
                    s.weight_zero_point = vec![1, 4 * HIDDEN + 1];
                },
                expected: OpError::incompatible_input_shapes(
                    "weights scale must have shape [directions, 4 * hidden_size]",
                ),
            },
            Case {
                dir: Direction::Forward,
                invalidate: |s| {
                    s.rec_weight_scale = vec![2];
                    s.rec_weight_zero_point = vec![2];
                },
                expected: OpError::incompatible_input_shapes(
                    "recurrent_weights scale must have shape [directions]",
                ),
            },
            Case {
                dir: Direction::Forward,
                invalidate: |s| {
                    s.weight_scale = vec![1, 4 * HIDDEN, 1];
                    s.weight_zero_point = vec![1, 4 * HIDDEN, 1];
                },
                expected: OpError::unsupported_value("weights scale must have 1 or 2 dims"),
            },
            Case {
                dir: Direction::Forward,
                invalidate: |s| s.weight_zero_point = vec![1],
                expected: OpError::incompatible_input_shapes(
                    "weights scale and zero point must have the same shape",
                ),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();

            let mut shapes = valid_shapes(case.dir);
            (case.invalidate)(&mut shapes);

            let input = NdTensor::<f32, 3>::zeros(shapes.input);
            let weights = NdTensor::<i8, 3>::zeros(shapes.weights);
            let weight_scale = Tensor::<f32>::zeros(&shapes.weight_scale);
            let weight_zero_point = Tensor::<i8>::zeros(&shapes.weight_zero_point);
            let rec_weights = NdTensor::<i8, 3>::zeros(shapes.recurrent_weights);
            let rec_weight_scale = Tensor::<f32>::zeros(&shapes.rec_weight_scale);
            let rec_weight_zero_point = Tensor::<i8>::zeros(&shapes.rec_weight_zero_point);
            let bias = NdTensor::<f32, 2>::zeros(shapes.bias);
            let initial_hidden = NdTensor::<f32, 3>::zeros(shapes.initial_hidden);
            let initial_cell = NdTensor::<f32, 3>::zeros(shapes.initial_cell);

            let result = dynamic_quantize_lstm(
                &pool,
                case.dir,
                input.view(),
                QuantizedWeights {
                    weights: weights.view(),
                    scale: weight_scale.view(),
                    zero_point: Some(weight_zero_point.view()),
                },
                QuantizedWeights {
                    weights: rec_weights.view(),
                    scale: rec_weight_scale.view(),
                    zero_point: Some(rec_weight_zero_point.view()),
                },
                Some(bias.view()),
                Some(initial_hidden.view()),
                Some(initial_cell.view()),
            );

            assert_eq!(result.err().as_ref(), Some(&case.expected));
        })
    }
}
