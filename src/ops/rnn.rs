use std::iter::Rev;
use std::mem::MaybeUninit;
use std::ops::Range;

use rayon::prelude::*;
use rten_gemm::{BiasVector, GemmExecutor, GemmInputA, GemmInputB, GemmOptions, GemmUninitOptions};
use rten_shape_inference::ops as shape_ops;
use rten_simd::SimdUnaryOp;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor};
use rten_vecmath as vecmath;
use rten_vecmath::ExtendInit;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::value::{DataType, ValueType};

/// Direction that an RNN operator will traverse the input sequence in.
#[derive(Copy, Clone, Debug)]
pub enum Direction {
    Forward,
    Reverse,
    Bidirectional,
}

impl Direction {
    /// Number of directions that an RNN operator will traverse the sequence in.
    pub fn num_directions(self) -> usize {
        match self {
            Self::Forward | Self::Reverse => 1,
            Self::Bidirectional => 2,
        }
    }
}

impl From<Direction> for shape_ops::Direction {
    fn from(direction: Direction) -> Self {
        match direction {
            Direction::Forward | Direction::Reverse => Self::Unidirectional,
            Direction::Bidirectional => Self::Bidirectional,
        }
    }
}

/// Forward or backward iterator over values in a range.
enum Sequence {
    Forward(Range<usize>),
    Backward(Rev<Range<usize>>),
}

impl Iterator for Sequence {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match self {
            Sequence::Forward(range) => range.next(),
            Sequence::Backward(rev_range) => rev_range.next(),
        }
    }
}

/// Return an iterator over sequence indices for an RNN operator.
///
/// `op_dirs` is the direction mode of the operator, `dir` is the direction
/// index (0 or 1) and `seq_len` is the input sequence length.
fn sequence_for_dir(op_dirs: Direction, dir: usize, seq_len: usize) -> Sequence {
    let reversed = matches!(
        (dir, op_dirs),
        (0, Direction::Reverse) | (1, Direction::Bidirectional)
    );
    if reversed {
        Sequence::Backward((0..seq_len).rev())
    } else {
        Sequence::Forward(0..seq_len)
    }
}

/// Like [`std::iter::zip`], but combines 3 iterators.
fn zip3<T1, T2, T3>(
    a: impl Iterator<Item = T1>,
    b: impl Iterator<Item = T2>,
    c: impl Iterator<Item = T3>,
) -> impl Iterator<Item = (T1, T2, T3)> {
    a.zip(b.zip(c)).map(|(a, (b, c))| (a, b, c))
}

/// Like [`std::iter::zip`], but combines 4 iterators.
fn zip4<T1, T2, T3, T4>(
    a: impl Iterator<Item = T1>,
    b: impl Iterator<Item = T2>,
    c: impl Iterator<Item = T3>,
    d: impl Iterator<Item = T4>,
) -> impl Iterator<Item = (T1, T2, T3, T4)> {
    zip3(a, b, c.zip(d)).map(|(a, b, (c, d))| (a, b, c, d))
}

/// Compute the input projection `input @ input_weights + input_bias` for every
/// gate and timestep in a single GEMM.
///
/// This is shared by the GRU and LSTM operators. The input projection does not
/// depend on the hidden state, so unlike the recurrent projection it can be
/// computed for the whole sequence up-front.
///
/// `input_mat` has shape `[seq_len * batch, input_size]` and `input_weights`
/// has shape `[input_size, n_gates * hidden_size]`. The result has shape
/// `[seq_len, batch, n_gates * hidden_size]`.
fn input_projection(
    pool: &BufferPool,
    gemm: &GemmExecutor,
    input_mat: NdTensorView<f32, 2>,
    input_weights: NdTensorView<f32, 2>,
    input_bias: Option<&[f32]>,
    seq_len: usize,
    batch: usize,
) -> NdTensor<f32, 3> {
    let mut output = NdTensor::uninit_in(pool, [seq_len, batch, input_weights.size(1)]);
    gemm.gemm_uninit(
        output.data_mut().unwrap(),
        GemmInputA::Unpacked(input_mat),
        GemmInputB::Unpacked(input_weights),
        GemmUninitOptions {
            bias: input_bias.map(BiasVector::Row),
            ..Default::default()
        },
    )
    .unwrap();

    // Safety: `gemm_uninit` initialized every element of `output`.
    unsafe { output.assume_init() }
}

/// Sequence length threshold for prepacking weights.
///
/// For sufficiently long input sequences, prepacking weights can speed up
/// execution by amortizing packing costs over the sequence length. For
/// short sequences the added memory usage means this won't be worthwhile.
///
/// TODO: This value was chosen because it seemed reasonable. It needs tuning.
const PREPACK_MIN_SEQ_LEN: usize = 5;

/// Compute the output for a single GRU layer.
///
/// `input` has shape [sequence_length, batch, input_size].
///
/// `weights` has shape `[directions, 3 * hidden_size, input_size]`. The middle
/// dimension is a concatenation of weights for the update, reset and hidden
/// gates.
///
/// `recurrent_weights` has shape `[directions, 3 * hidden_size, hidden_size]`.
/// The middle dimension is a concatenation of weights for the update, reset and
/// hidden gates.
///
/// `bias` has shape `[directions, 6 * hidden_size]`. The last dimension is a
/// concatenation of input biases for the update, reset and hidden gates
/// followed by hidden biases for the same gates.
///
/// `initial_hidden` has shape `[directions, batch, hidden_size]`.
pub fn gru(
    pool: &BufferPool,
    direction: Direction,
    input: NdTensorView<f32, 3>,
    weights: NdTensorView<f32, 3>,
    recurrent_weights: NdTensorView<f32, 3>,
    bias: Option<NdTensorView<f32, 2>>,
    initial_hidden: Option<NdTensorView<f32, 3>>,
    linear_before_reset: bool,
) -> Result<Vec<Tensor>, OpError> {
    // PyTorch and cuDNN only support the `linear_before_reset=true` case, as
    // it enables better efficiency. The `linear_before_reset=false` case
    // matches the paper that introduced the GRU operator.
    //
    // See note in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
    if !linear_before_reset {
        // PyTorch and cuDNN
        return Err(OpError::UnsupportedValue(
            "`linear_before_reset=0` is not supported",
        ));
    }

    let [seq_len, batch, input_size] = input.shape();
    let [_directions, hidden_x3, _input_size] = weights.shape();
    let num_directions = direction.num_directions();

    if !hidden_x3.is_multiple_of(3) {
        return Err(OpError::InvalidValue(
            "weights dim 1 must be 3 * hidden_size",
        ));
    }
    let hidden_size = hidden_x3 / 3;
    let n_gates = 3;

    // Contiguous input needed so it can be reshaped into a matrix below.
    // Contiguous bias needed so per-gate slices can be passed to GEMM.
    let input = input.to_contiguous_in(pool).auto_return(pool);
    let input_mat = input.view().reshaped([seq_len * batch, input_size]);
    let bias = bias.map(|b| b.to_contiguous());

    let mut hidden = initial_hidden
        .map(|t| t.to_tensor_in(pool))
        .unwrap_or_else(|| NdTensor::zeros_in(pool, [num_directions, batch, hidden_size]));
    let mut hidden_seq = NdTensor::uninit_in(pool, [seq_len, num_directions, batch, hidden_size]);

    let gemm = GemmExecutor::new();

    // From the ONNX spec, the intermediate values are computed as:
    //
    //   zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    //   rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    //
    //   If `linear_before_reset` is true:
    //     ht = tanh(dot(input, hidden_w) + rt * (dot(hidden, rec_hidden_w) + rec_hidden_bias) + hidden_bias)
    //   Else:
    //     ht = tanh(dot(input, hidden_w) + dot((rt * hidden), rec_hidden_w) + rec_hidden_bias + hidden_bias)
    //
    //   Ht = (1 - zt) (.) ht + zt (.) (Ht-1)
    //
    // Where:
    //
    //  - `zt`, `rt` and `ht` are the update, reset and hidden gates
    //  - `Xt`, `Ht` are the input and hidden states at time `t`
    //  - `W{z,r,h}` and `R{z,r,h}` are the input and recurrent weights
    //  - `Wb{z,r,h}` and `Rb{z,r,h}` are the input and recurrent biases
    //  - `f` and `g` are activations. f=sigmoid, g=tanh
    //
    // In the `linear_before_reset=true` case, which is all we currently
    // support, the matrix multiplications for all gates can be combined into
    // two: one for `input @ input_weights`, one for `hidden @ hidden_weights`.
    // The input projection does not depend on the hidden state, so it is
    // additionally computed for all timesteps at once.

    // Each direction is independent, so process them in parallel.
    hidden
        .axis_iter_mut(0)
        .into_par_iter()
        .zip(hidden_seq.axis_iter_mut(1))
        .enumerate()
        .for_each(|(dir, (mut hidden, mut hidden_seq))| {
            let input_bias = bias
                .as_ref()
                .map(|b| b.slice((dir, ..(n_gates * hidden_size))).data().unwrap());
            let hidden_bias = bias
                .as_ref()
                .map(|b| b.slice((dir, (n_gates * hidden_size)..)).data().unwrap());

            let mut input_proj = input_projection(
                pool,
                &gemm,
                input_mat.view(),
                weights.slice(dir).transposed(),
                input_bias,
                seq_len,
                batch,
            )
            .auto_return(pool);

            let prepack = seq_len >= PREPACK_MIN_SEQ_LEN;
            let hidden_weights = recurrent_weights.slice(dir).transposed();
            let packed_hidden_weights =
                prepack.then(|| gemm.prepack_b_in(pool, hidden_weights).auto_return(pool));
            let hidden_weights = packed_hidden_weights
                .as_ref()
                .map(|packed| GemmInputB::Packed(packed))
                .unwrap_or(GemmInputB::Unpacked(hidden_weights));

            // Scratch space for output of `hidden_state @ hidden_weights` matmul.
            let mut hidden_scratch =
                NdTensor::zeros_in(pool, [batch, n_gates * hidden_size]).auto_return(pool);

            for seq in sequence_for_dir(direction, dir, seq_len) {
                // Compute `hidden @ hidden_weights + hidden_bias` for all gates.
                gemm.gemm(
                    hidden_scratch.data_mut().unwrap(),
                    GemmInputA::Unpacked(hidden.view()),
                    hidden_weights,
                    GemmOptions {
                        bias: hidden_bias.map(BiasVector::Row),
                        ..Default::default()
                    },
                )
                .unwrap();

                let mut gates = input_proj.slice_mut([seq]);
                gru_step(
                    hidden_size,
                    gates.data_mut().unwrap(),
                    hidden_scratch.data().unwrap(),
                    hidden.data_mut().unwrap(),
                    hidden_seq.slice_mut([seq]).data_mut().unwrap(),
                );
            }
        });

    // Safety: The loop above wrote to every element of `hidden_seq`.
    let hidden_seq = unsafe { hidden_seq.assume_init() };

    Ok([hidden_seq.into_dyn(), hidden.into_dyn()].into())
}

/// Compute one GRU timestep for a batch of inputs, updating the hidden state
/// in place and writing a copy of the new hidden state to `out`.
///
/// `gates` has shape `[batch, 3 * hidden_size]` and contains the input
/// projection `Xt @ W^T + Wb` for this timestep, with the update, reset and
/// hidden gates concatenated along the last dimension. It is also used as
/// scratch space. `hidden_scratch` has the same shape and contains the
/// recurrent projection `Ht-1 @ R^T + Rb`. `hidden` and `out` have shape
/// `[batch, hidden_size]`.
fn gru_step(
    hidden_size: usize,
    gates: &mut [f32],
    hidden_scratch: &[f32],
    hidden: &mut [f32],
    out: &mut [MaybeUninit<f32>],
) {
    // `chunks_exact` panics on a zero chunk size.
    if hidden_size == 0 {
        return;
    }

    for (gates, scratch, hidden, out) in zip4(
        gates.chunks_exact_mut(3 * hidden_size),
        hidden_scratch.chunks_exact(3 * hidden_size),
        hidden.chunks_exact_mut(hidden_size),
        out.chunks_exact_mut(hidden_size),
    ) {
        // zt = sigmoid(Xt*(Wz^T) + Wbz + Ht-1*(Rz^T) + Rbz), rt likewise.
        let (update_reset, hidden_gate) = gates.split_at_mut(2 * hidden_size);
        let (scratch_update_reset, scratch_hidden) = scratch.split_at(2 * hidden_size);
        for (x, s) in update_reset.iter_mut().zip(scratch_update_reset) {
            *x += s;
        }
        vecmath::Sigmoid {}.map_mut(update_reset);
        let (update, reset) = update_reset.split_at(hidden_size);

        // ht = tanh(Xt*(Wh^T) + Wbh + rt (.) (Ht-1*(Rh^T) + Rbh))
        for (x, s, r) in zip3(hidden_gate.iter_mut(), scratch_hidden.iter(), reset.iter()) {
            *x += r * s;
        }
        vecmath::Tanh {}.map_mut(hidden_gate);

        // Ht = (1 - zt) (.) ht + zt (.) Ht-1
        for (hidden, update, hidden_gate, out) in zip4(
            hidden.iter_mut(),
            update.iter(),
            hidden_gate.iter(),
            out.iter_mut(),
        ) {
            *hidden = (1. - update) * hidden_gate + update * (*hidden);
            out.write(*hidden);
        }
    }
}

/// Gated Recurrent Unit operator.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct GRU {
    pub direction: Direction,

    #[allow(unused)] // Currently inferred from operator inputs.
    pub hidden_size: usize,

    /// When computing the output of the hidden gate, apply the linear
    /// transformation before multiplying by the output of the reset gate.
    pub linear_before_reset: bool,
}

impl Operator for GRU {
    fn name(&self) -> &str {
        "GRU"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(6)
    }

    fn max_outputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let weights = inputs.require_as(1)?;
        let recurrent_weights = inputs.require_as(2)?;
        let bias = inputs.get_as(3)?;
        let _seq_len = inputs.get_as::<NdTensorView<i32, 1>>(4)?;
        let initial_hidden = inputs.get_as(5)?;

        gru(
            ctx.pool(),
            self.direction,
            input,
            weights,
            recurrent_weights,
            bias,
            initial_hidden,
            self.linear_before_reset,
        )
        .into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some(OutputTypeList::from_slice(&[
            OutputType::Fixed(ValueType::Tensor(DataType::Float)),
            OutputType::Fixed(ValueType::Tensor(DataType::Float)),
        ]))
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    GRU,
    op,
    shape_ops::GRU {
        direction: op.direction.into(),
    }
);

/// Compute the output for a single LSTM layer.
///
/// `input` has shape [sequence_length, batch, input_size].
///
/// `weights` has shape `[directions, 4 * hidden_size, input_size]`. The middle
/// dimension is a concatenation of weights for the input, output, forget and
/// cell gates.
///
/// `recurrent_weights` has shape `[directions, 4 * hidden_size, hidden_size]`.
/// The middle dimension is a concatenation of weights for the input, output,
/// forget and cell gates.
///
/// `bias` has shape `[directions, 8 * hidden_size]`. The last dimension is
/// a concatenation of input biases for the input, output, forget and cell gates
/// followed by hidden biases for the same gates.
///
/// `initial_hidden` has shape `[directions, batch, hidden_size]`.
/// `initial_cell` has shape `[directions, batch, hidden_size]`.
pub fn lstm(
    pool: &BufferPool,
    direction: Direction,
    input: NdTensorView<f32, 3>,
    weights: NdTensorView<f32, 3>,
    recurrent_weights: NdTensorView<f32, 3>,
    bias: Option<NdTensorView<f32, 2>>,
    initial_hidden: Option<NdTensorView<f32, 3>>,
    initial_cell: Option<NdTensorView<f32, 3>>,
) -> Result<Vec<Tensor>, OpError> {
    // TODO - Add validation of the sizes of individual dimensions in the inputs.
    let [seq_len, batch, input_size] = input.shape();
    let [_directions, hidden_x4, _input_size] = weights.shape();

    let num_directions = direction.num_directions();

    if !weights.size(1).is_multiple_of(4) {
        return Err(OpError::InvalidValue(
            "weights dim 1 must be 4 * hidden_size",
        ));
    }
    let hidden_size = hidden_x4 / 4;
    let n_gates = 4;

    if let Some(bias) = bias.as_ref()
        && !bias.size(1).is_multiple_of(8)
    {
        return Err(OpError::InvalidValue("bias dim 1 must be 8 * hidden_size"));
    }

    // Contiguous input needed so it can be reshaped into a matrix below.
    // Contiguous bias needed so per-gate slices can be passed to GEMM.
    let input = input.to_contiguous_in(pool).auto_return(pool);
    let input_mat = input.view().reshaped([seq_len * batch, input_size]);
    let bias = bias.map(|t| t.to_contiguous());

    let mut cell = initial_cell
        .map(|t| t.to_tensor_in(pool))
        .unwrap_or_else(|| NdTensor::zeros_in(pool, [num_directions, batch, hidden_size]));
    let mut hidden = initial_hidden
        .map(|t| t.to_tensor_in(pool))
        .unwrap_or_else(|| NdTensor::zeros_in(pool, [num_directions, batch, hidden_size]));

    let mut hidden_seq = NdTensor::uninit_in(pool, [seq_len, num_directions, batch, hidden_size]);

    let gemm = GemmExecutor::new();

    // From the ONNX spec, the intermediate values are computed as:
    //
    // - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    // - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    // - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    // - Ct = ft (.) Ct-1 + it (.) ct
    // - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    // - Ht = ot (.) h(Ct)
    //
    // Where:
    //
    //  - `it`, `ft`, `ct` and `ot` are the input, forget, cell and output gates
    //  - `Xt`, `Ht` and `Ct` are the input, hidden state and cell state at time `t`
    //  - `W{i,o,f,c}` and `R{i,o,f,c}` are the input and recurrent gate weights
    //  - `Wb{i,o,f,c}` and `Rb{i,o,f,c}` are the input and recurrent gate biases
    //  - `P{i,o,f,c}` are peephole weights. These are not currently
    //    supported.
    //  - `f`, `g` and `h` are activations. `f`=sigmoid, `g` and `h`
    //    are tanh.
    //
    // The matrix multiplications for all gates are combined into two: one for
    // `input @ input_weights`, one for `hidden @ hidden_weights`. The input
    // projection does not depend on the hidden state, so it is additionally
    // computed for all timesteps at once.

    // Each direction is independent, so process them in parallel.
    let hidden_dirs: Vec<_> = hidden.axis_iter_mut(0).collect();
    let cell_dirs: Vec<_> = cell.axis_iter_mut(0).collect();
    let hidden_seq_dirs: Vec<_> = hidden_seq.axis_iter_mut(1).collect();

    hidden_dirs
        .into_par_iter()
        .zip(cell_dirs)
        .zip(hidden_seq_dirs)
        .enumerate()
        .for_each(|(dir, ((mut hidden, mut cell), mut hidden_seq))| {
            let input_bias = bias
                .as_ref()
                .map(|b| b.slice((dir, ..(n_gates * hidden_size))).data().unwrap());
            let hidden_bias = bias
                .as_ref()
                .map(|b| b.slice((dir, (n_gates * hidden_size)..)).data().unwrap());

            // Compute `input @ input_weights + input_bias` for all gates and
            // timesteps in one GEMM.
            let input_weights = weights.slice(dir).transposed();
            let proj_len = seq_len * batch * n_gates * hidden_size;
            let mut input_proj = pool.alloc(proj_len);
            input_proj.extend_init(|uninit| {
                gemm.gemm_uninit(
                    &mut uninit[..proj_len],
                    GemmInputA::Unpacked(input_mat.view()),
                    GemmInputB::Unpacked(input_weights),
                    GemmUninitOptions {
                        bias: input_bias.map(BiasVector::Row),
                        ..Default::default()
                    },
                )
                .unwrap()
            });
            let mut input_proj =
                NdTensor::from_data([seq_len, batch, n_gates * hidden_size], input_proj)
                    .auto_return(pool);

            let prepack = seq_len >= PREPACK_MIN_SEQ_LEN;
            let hidden_weights = recurrent_weights.slice(dir).transposed();
            let packed_hidden_weights =
                prepack.then(|| gemm.prepack_b_in(pool, hidden_weights).auto_return(pool));
            let hidden_weights = packed_hidden_weights
                .as_ref()
                .map(|packed| GemmInputB::Packed(packed))
                .unwrap_or(GemmInputB::Unpacked(hidden_weights));

            // Scratch space for output of `hidden_state @ hidden_weights` matmul.
            let mut hidden_scratch =
                NdTensor::zeros_in(pool, [batch, n_gates * hidden_size]).auto_return(pool);

            for seq in sequence_for_dir(direction, dir, seq_len) {
                // Compute `hidden @ hidden_weights + hidden_bias` for all gates.
                gemm.gemm(
                    hidden_scratch.data_mut().unwrap(),
                    GemmInputA::Unpacked(hidden.view()),
                    hidden_weights,
                    GemmOptions {
                        bias: hidden_bias.map(BiasVector::Row),
                        ..Default::default()
                    },
                )
                .unwrap();

                let mut gates = input_proj.slice_mut([seq]);
                lstm_step(
                    hidden_size,
                    gates.data_mut().unwrap(),
                    hidden_scratch.data().unwrap(),
                    hidden.data_mut().unwrap(),
                    cell.data_mut().unwrap(),
                    hidden_seq.slice_mut([seq]).data_mut().unwrap(),
                );
            }
        });

    // Safety: The loop above wrote to every element of `hidden_seq`.
    let hidden_seq = unsafe { hidden_seq.assume_init() };

    Ok([hidden_seq.into_dyn(), hidden.into_dyn(), cell.into_dyn()].into())
}

/// Compute one LSTM timestep for a batch of inputs, updating the hidden and
/// cell states in place and writing a copy of the new hidden state to `out`.
///
/// `gates` has shape `[batch, 4 * hidden_size]` and contains the input
/// projection `Xt @ W^T + Wb` for this timestep, with the input, output,
/// forget and cell gates concatenated along the last dimension. It is also
/// used as scratch space. `hidden_scratch` has the same shape and contains the
/// recurrent projection `Ht-1 @ R^T + Rb`. `hidden`, `cell` and `out` have
/// shape `[batch, hidden_size]`.
fn lstm_step(
    hidden_size: usize,
    gates: &mut [f32],
    hidden_scratch: &[f32],
    hidden: &mut [f32],
    cell: &mut [f32],
    out: &mut [MaybeUninit<f32>],
) {
    if hidden_size == 0 {
        return;
    }

    for (gates, scratch, hidden, (cell, out)) in zip4(
        gates.chunks_exact_mut(4 * hidden_size),
        hidden_scratch.chunks_exact(4 * hidden_size),
        hidden.chunks_exact_mut(hidden_size),
        cell.chunks_exact_mut(hidden_size)
            .zip(out.chunks_exact_mut(hidden_size)),
    ) {
        // it = sigmoid(Xt*(Wi^T) + Wbi + Ht-1*(Ri^T) + Rbi), ot and ft likewise.
        let (iof_gates, cell_gate) = gates.split_at_mut(3 * hidden_size);
        let (scratch_iof, scratch_cell) = scratch.split_at(3 * hidden_size);
        for (x, s) in iof_gates.iter_mut().zip(scratch_iof) {
            *x += s;
        }
        vecmath::Sigmoid {}.map_mut(iof_gates);
        let (input_gate, of_gates) = iof_gates.split_at(hidden_size);
        let (out_gate, forget_gate) = of_gates.split_at(hidden_size);

        // ct = tanh(Xt*(Wc^T) + Wbc + Ht-1*(Rc^T) + Rbc)
        for (x, s) in cell_gate.iter_mut().zip(scratch_cell) {
            *x += s;
        }
        vecmath::Tanh {}.map_mut(cell_gate);

        // Ct = ft (.) Ct-1 + it (.) ct
        for (cell, forget, input, cell_gate) in zip4(
            cell.iter_mut(),
            forget_gate.iter(),
            input_gate.iter(),
            cell_gate.iter(),
        ) {
            *cell = forget * *cell + input * cell_gate;
        }

        // Ht = ot (.) tanh(Ct). The cell gate is no longer needed, so reuse it
        // as scratch space for computing `tanh(Ct)`.
        cell_gate.copy_from_slice(cell);
        vecmath::Tanh {}.map_mut(cell_gate);
        for (hidden, out_gate, tanh_cell, out) in zip4(
            hidden.iter_mut(),
            out_gate.iter(),
            cell_gate.iter(),
            out.iter_mut(),
        ) {
            *hidden = out_gate * tanh_cell;
            out.write(*hidden);
        }
    }
}

/// Long Short-Term Memory operator.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct LSTM {
    pub direction: Direction,

    #[allow(unused)]
    pub hidden_size: usize, // Currently inferred from operator inputs.
}

impl Operator for LSTM {
    fn name(&self) -> &str {
        "LSTM"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(7)
    }

    fn max_outputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let weights = inputs.require_as(1)?;
        let recurrent_weights = inputs.require_as(2)?;
        let bias = inputs.get_as(3)?;
        let _seq_len = inputs.get_as::<NdTensorView<i32, 1>>(4)?;
        let initial_hidden = inputs.get_as(5)?;
        let initial_cell = inputs.get_as(6)?;

        lstm(
            ctx.pool(),
            self.direction,
            input,
            weights,
            recurrent_weights,
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
    LSTM,
    op,
    shape_ops::LSTM {
        direction: op.direction.into(),
    }
);

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;
    use serde_json::Value;

    use crate::buffer_pool::BufferPool;
    use crate::ops::{Direction, concat, gru, lstm, split};

    /// Read a float tensor from a JSON value.
    ///
    /// The JSON value is expected to be of the form `[shape, data]` where
    /// `shape` is an int array and `data` is a float array.
    pub fn read_tensor(val: &Value) -> Result<Tensor<f32>, &'static str> {
        let vec = match val {
            Value::Array(vec) => vec,
            _ => return Err("Expected array"),
        };

        let (shape, data) = match vec.as_slice() {
            [Value::Array(shape), Value::Array(data)] => (shape, data),
            _ => return Err("Expected [shape, data] array"),
        };

        let shape = shape
            .iter()
            .map(|v| v.as_i64().map(|v| v as usize).ok_or("Expected int array"))
            .collect::<Result<Vec<usize>, _>>()?;

        let data = data
            .iter()
            .map(|v| v.as_f64().map(|v| v as f32).ok_or("Expected float array"))
            .collect::<Result<Vec<f32>, _>>()?;

        Ok(Tensor::from_data(&shape, data))
    }

    pub fn read_json_file(path: &str) -> Value {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).unwrap()
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Op {
        Gru,
        Lstm,
    }

    // Basic test that runs bidirectional RNN operators with random inputs and
    // checks that the operator doesn't crash, produces outputs of the right
    // shape and that the last hidden / hidden seq outputs are consistent.
    #[test]
    fn test_rnn_ops_with_random_input() {
        let batch = 2;
        let seq_len = 5;
        let dir = Direction::Bidirectional;

        let hidden_size = 3;
        let features = 2;

        #[derive(Clone, Debug)]
        struct Case {
            op: Op,
            with_bias: bool,
            with_hidden_init: bool,
            with_initial_cell: bool,
        }

        let cases = [
            Case {
                op: Op::Lstm,
                with_bias: true,
                with_hidden_init: true,
                with_initial_cell: true,
            },
            Case {
                op: Op::Lstm,
                with_bias: false,
                with_hidden_init: false,
                with_initial_cell: false,
            },
            Case {
                op: Op::Gru,
                with_bias: true,
                with_hidden_init: true,
                with_initial_cell: false,
            },
            Case {
                op: Op::Gru,
                with_bias: false,
                with_hidden_init: false,
                with_initial_cell: false,
            },
        ];

        cases.test_each_clone(|case| {
            let mut rng = XorShiftRng::new(1234);
            let pool = BufferPool::new();
            let num_gates = match case.op {
                Op::Gru => 3,
                Op::Lstm => 4,
            };

            let input =
                NdTensor::<f32, 3>::rand([seq_len, batch, features], &mut rng).map(|x| x - 0.5);
            let weights = NdTensor::<f32, 3>::rand(
                [dir.num_directions(), num_gates * hidden_size, features],
                &mut rng,
            )
            .map(|x| x - 0.5);
            let recurrent_weights = NdTensor::<f32, 3>::rand(
                [dir.num_directions(), num_gates * hidden_size, hidden_size],
                &mut rng,
            )
            .map(|x| x - 0.5);
            let bias = NdTensor::rand(
                [dir.num_directions(), 2 * num_gates * hidden_size],
                &mut rng,
            );
            let initial_hidden =
                NdTensor::rand([dir.num_directions(), batch, hidden_size], &mut rng);
            let initial_cell = NdTensor::rand([dir.num_directions(), batch, hidden_size], &mut rng);

            let result = match case.op {
                Op::Lstm => lstm(
                    &pool,
                    dir,
                    input.nd_view(),
                    weights.nd_view(),
                    recurrent_weights.nd_view(),
                    case.with_bias.then_some(bias.nd_view()),
                    case.with_hidden_init.then_some(initial_hidden.nd_view()),
                    case.with_initial_cell.then_some(initial_cell.nd_view()),
                )
                .expect("lstm op failed"),
                Op::Gru => gru(
                    &pool,
                    dir,
                    input.view(),
                    weights.view(),
                    recurrent_weights.view(),
                    case.with_bias.then_some(bias.view()),
                    case.with_hidden_init.then_some(initial_hidden.view()),
                    true, /* linear_before_reset */
                )
                .expect("gru op failed"),
            };

            // Check that outputs have the right shapes.
            assert_eq!(
                result.len(),
                match case.op {
                    Op::Gru => 2,
                    Op::Lstm => 3,
                }
            );
            let hidden_seq = &result[0];
            assert_eq!(
                hidden_seq.shape(),
                &[seq_len, dir.num_directions(), batch, hidden_size]
            );

            let last_hidden = &result[1];
            assert_eq!(
                last_hidden.shape(),
                &[dir.num_directions(), batch, hidden_size]
            );

            if case.op == Op::Lstm {
                let last_cell = &result[2];
                assert_eq!(
                    last_cell.shape(),
                    &[dir.num_directions(), batch, hidden_size]
                );
            }

            // The last hidden state should match the end of the hidden sequence
            // for the forwards direction, and the start of the hidden sequence
            // for the reverse direction.
            let hidden_seq_fwd = hidden_seq.slice((
                -1, // seq
                0,  // direction
            ));
            let last_hidden_fwd = last_hidden.slice(0);
            assert_eq!(hidden_seq_fwd, last_hidden_fwd);

            let hidden_seq_rev = hidden_seq.slice((
                0, // seq
                1, // direction
            ));
            let last_hidden_rev = last_hidden.slice(1);
            assert_eq!(hidden_seq_rev, last_hidden_rev);
        })
    }

    /// Re-order a weight or bias tensor for LSTM gates from (input, forget,
    /// cell, output) as used by PyTorch to (input, output, forget, cell) as
    /// used by ONNX.
    fn reorder_ifco_to_iofc(x: &Tensor, axis: isize) -> Tensor {
        let pool = BufferPool::new();
        let size = x.size(axis as usize) / 4;
        let splits = &[size as i32; 4];

        // Split input into seperate tensor for each of the gates.
        let ifco = split(&pool, x.view(), axis, splits.as_slice().into()).expect("split failed");

        // Recombine in a new gate order.
        concat(
            &pool,
            &[
                ifco[0].view(),
                ifco[3].view(),
                ifco[1].view(),
                ifco[2].view(),
            ],
            axis,
        )
        .expect("concat failed")
    }

    /// Re-order a weight or bias tensor for GRU gates from (reset, update,
    /// hidden) as used by PyTorch to (update, reset, hidden) as used by ONNX.
    fn reorder_ruh_to_urh(x: &Tensor, axis: isize) -> Tensor {
        let pool = BufferPool::new();
        let size = x.size(axis as usize) / 3;
        let splits = &[size as i32; 3];

        // Split input into seperate tensor for each of the gates.
        let ruh = split(&pool, x.view(), axis, splits.as_slice().into()).expect("split failed");

        // Recombine in a new gate order.
        concat(&pool, &[ruh[1].view(), ruh[0].view(), ruh[2].view()], axis).expect("concat failed")
    }

    struct RNNRefTest {
        /// Input as [seq, batch, feature]
        input: Tensor,

        /// Expected output as [seq, direction, batch, hidden]
        expected: Tensor,

        /// Input-hidden weights as [direction, num_gates * hidden, feature]
        weights: Tensor,

        /// Hidden-hidden weights as [direction, num_gates * hidden, num_gates * hidden]
        hidden_weights: Tensor,

        /// Bias as [direction, 2 * num_gates * hidden]
        bias: Option<Tensor>,

        /// Initial value of the hidden state as [direction, batch, hidden]
        initial_hidden: Option<Tensor>,

        /// Initial value of the cell state as [direction, batch, hidden].
        ///
        /// Only applicable for LSTM operator.
        initial_cell: Option<Tensor>,
    }

    /// Read inputs for a PyTorch reference test for RNN ops from a JSON value.
    fn read_pytorch_ref_test(op: Op, case: &Value) -> RNNRefTest {
        let pool = BufferPool::new();
        let params = &case["params"];

        let is_bidirectional = params.get("weight_ih_l0_reverse").is_some();

        let mut input = read_tensor(&case["input"]).expect("failed to read input");
        input.insert_axis(1); // Add batch dim

        let mut expected = read_tensor(&case["output"]).expect("failed to read output");

        // Reshape from [seq, dir * hidden_size] to [seq, dir, hidden_size]
        if is_bidirectional {
            let es = expected.shape();
            expected.reshape(&[es[0], 2, es[1] / 2]);
        } else {
            expected.insert_axis(1);
        }
        expected.insert_axis(2); // Add batch dim

        let read_param = |name| match op {
            Op::Lstm => reorder_ifco_to_iofc(
                &read_tensor(&params[name]).expect("failed to read weight"),
                0,
            ),
            Op::Gru => reorder_ruh_to_urh(
                &read_tensor(&params[name]).expect("failed to read weight"),
                0,
            ),
        };

        let mut weights = read_param("weight_ih_l0");
        weights.insert_axis(0); // Add directions dim

        let mut hidden_weights = read_param("weight_hh_l0");
        hidden_weights.insert_axis(0); // Add directions dim

        let input_bias = read_param("bias_ih_l0");
        let hidden_bias = read_param("bias_hh_l0");
        let mut bias = concat(&pool, &[input_bias.view(), hidden_bias.view()], 0).unwrap();
        bias.insert_axis(0); // Add directions dim

        // If this is a bidirectional RNN, there will be `_reverse`-suffixed
        // versions of the bias and weight params. Extract these and concatenate
        // with the forwards direction values.
        if is_bidirectional {
            let mut rev_weights = read_param("weight_ih_l0_reverse");
            rev_weights.insert_axis(0); // Add directions dim
            weights = concat(&pool, &[weights.view(), rev_weights.view()], 0).unwrap();

            let mut rev_hidden_weights = read_param("weight_hh_l0_reverse");
            rev_hidden_weights.insert_axis(0); // Add directions dim
            hidden_weights = concat(
                &pool,
                &[hidden_weights.view(), rev_hidden_weights.view()],
                0,
            )
            .unwrap();

            let rev_input_bias = read_param("bias_ih_l0_reverse");
            let rev_hidden_bias = read_param("bias_hh_l0_reverse");
            let mut rev_bias =
                concat(&pool, &[rev_input_bias.view(), rev_hidden_bias.view()], 0).unwrap();
            rev_bias.insert_axis(0); // Add directions dim
            bias = concat(&pool, &[bias.view(), rev_bias.view()], 0).unwrap();
        }

        let initial_hidden = case.get("initial_hidden").map(|param| {
            let mut init = read_tensor(param).expect("failed to read initial hidden state");
            init.insert_axis(1); // Add batch dim
            init
        });

        let initial_cell = case.get("initial_cell").map(|param| {
            let mut init = read_tensor(param).expect("failed to read initial cell state");
            init.insert_axis(1); // Add batch dim
            init
        });

        RNNRefTest {
            input,
            weights,
            hidden_weights,
            bias: Some(bias),
            expected,
            initial_hidden,
            initial_cell,
        }
    }

    #[test]
    fn test_rnn_pytorch() {
        let dict = read_json_file("pytorch-ref-tests/rnn.json");

        #[derive(Debug)]
        struct Case {
            name: &'static str,
            dir: Direction,
        }

        let cases = &[
            Case {
                name: "lstm_forwards",
                dir: Direction::Forward,
            },
            Case {
                name: "lstm_initial",
                dir: Direction::Forward,
            },
            Case {
                name: "lstm_bidirectional",
                dir: Direction::Bidirectional,
            },
            Case {
                name: "gru_forwards",
                dir: Direction::Forward,
            },
            Case {
                name: "gru_initial",
                dir: Direction::Forward,
            },
            Case {
                name: "gru_bidirectional",
                dir: Direction::Bidirectional,
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let op = if case.name.starts_with("lstm") {
                Op::Lstm
            } else {
                Op::Gru
            };
            let data = read_pytorch_ref_test(op, &dict[case.name]);
            let result = match op {
                Op::Lstm => lstm(
                    &pool,
                    case.dir,
                    data.input.nd_view(),
                    data.weights.nd_view(),
                    data.hidden_weights.nd_view(),
                    data.bias.as_ref().map(|b| b.nd_view()),
                    data.initial_hidden.as_ref().map(|ih| ih.nd_view()),
                    data.initial_cell.as_ref().map(|ic| ic.nd_view()),
                )
                .expect("LSTM op failed"),
                Op::Gru => gru(
                    &pool,
                    case.dir,
                    data.input.nd_view(),
                    data.weights.nd_view(),
                    data.hidden_weights.nd_view(),
                    data.bias.as_ref().map(|b| b.nd_view()),
                    data.initial_hidden.as_ref().map(|ih| ih.nd_view()),
                    true, /* linear_before_reset */
                )
                .expect("GRU op failed"),
            };
            let output = &result[0];

            expect_equal(output, &data.expected).unwrap();
        })
    }

    // TODO - Add tests for incorrect input shapes
}
