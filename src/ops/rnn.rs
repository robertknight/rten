use std::iter::Rev;
use std::ops::Range;

use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor, TensorView};

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::operator::{IntoOpResult, OpError, OpRunContext, Operator, OutputList, static_dims};
use crate::ops::binary_elementwise::{add_in_place, mul_in_place};
use crate::ops::unary_elementwise::{sigmoid, tanh};

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

/// Sequence length threshold for prepacking weights.
///
/// For sufficiently long input sequences, prepacking weights can speed up
/// execution by amortizing packing costs over the sequence length. For
/// short sequences the added memory usage means this won't be worthwhile.
///
/// TODO: This value was chosen because it seemed reasonable. It needs tuning.
const PREPACK_MIN_SEQ_LEN: usize = 5;

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
    input: TensorView,
    weights: TensorView,
    recurrent_weights: TensorView,
    bias: Option<TensorView>,
    initial_hidden: Option<TensorView>,
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

    let input = static_dims!(input, 3, "seq, batch, input")?;
    let weights = static_dims!(weights, 3, "dir, hidden x 3, input")?;
    let recurrent_weights = static_dims!(recurrent_weights, 3)?;
    let bias = bias
        .map(|bias| static_dims!(bias, 2, "dir, hidden x 6"))
        .transpose()?;

    let [seq_len, batch, _input_size] = input.shape();
    let [_directions, hidden_x3, _input_size] = weights.shape();

    let initial_hidden = initial_hidden
        .map(|initial_hidden| static_dims!(initial_hidden, 3))
        .transpose()?;

    let num_directions = direction.num_directions();
    let hidden_size = hidden_x3 / 3;

    let mut hidden = initial_hidden
        .map(|t| t.to_tensor_in(pool))
        .unwrap_or_else(|| NdTensor::zeros_in(pool, [num_directions, batch, hidden_size]));
    let mut hidden_seq = NdTensor::zeros_in(pool, [seq_len, num_directions, batch, hidden_size]);

    // Indices of gates in the concatenated weight and bias tensors.
    const UPDATE_GATE: usize = 0;
    const RESET_GATE: usize = 1;
    const HIDDEN_GATE: usize = 2;

    let n_gates = 3;
    let mut gates = NdTensor::zeros_in(pool, [batch, n_gates * hidden_size]).auto_return(pool);
    let gate_range = |gate| (gate * hidden_size)..((gate + 1) * hidden_size);

    // Scratch space for output of `hidden_state @ hidden_weights` matmul.
    let mut hidden_scratch =
        NdTensor::zeros_in(pool, [batch, n_gates * hidden_size]).auto_return(pool);

    let gemm = GemmExecutor::new();
    for dir in 0..num_directions {
        let prepack = seq_len >= PREPACK_MIN_SEQ_LEN;

        let input_weights = weights.slice(dir).transposed();
        let packed_input_weights =
            prepack.then(|| gemm.prepack_b_in(pool, input_weights).auto_return(pool));
        let input_weights = packed_input_weights
            .as_ref()
            .map(|packed| GemmInputB::Packed(packed))
            .unwrap_or(GemmInputB::Unpacked(input_weights));

        let hidden_weights = recurrent_weights.slice(dir).transposed();
        let packed_hidden_weights =
            prepack.then(|| gemm.prepack_b_in(pool, hidden_weights).auto_return(pool));
        let hidden_weights = packed_hidden_weights
            .as_ref()
            .map(|packed| GemmInputB::Packed(packed))
            .unwrap_or(GemmInputB::Unpacked(hidden_weights));

        let input_bias = bias
            .as_ref()
            .map(|b| b.slice((dir, ..(n_gates * hidden_size))));
        let hidden_bias = bias
            .as_ref()
            .map(|b| b.slice((dir, (n_gates * hidden_size)..)));

        for seq in sequence_for_dir(direction, dir, seq_len) {
            let in_item = input.slice([seq]);
            let hidden_item = hidden.slice([dir]);

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
            // support, the matrix multiplications for all gates can be
            // combined into two: one for `input @ input_weights`, one for
            // `hidden @ hidden_weights`.

            // Compute `input @ weights + bias` for all gates.
            gemm.gemm(
                gates.data_mut().expect("expected contiguous input"),
                GemmInputA::Unpacked(in_item),
                input_weights,
                1.,   // alpha
                0.,   // beta
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap();
            if let Some(input_bias) = input_bias {
                add_in_place(gates.as_dyn_mut(), input_bias.as_dyn());
            }

            // Compute `hidden @ hidden_weights + hidden_bias` for all gates.
            gemm.gemm(
                hidden_scratch.data_mut().unwrap(),
                GemmInputA::Unpacked(hidden_item),
                hidden_weights,
                1.,   // alpha
                0.,   // beta
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap();
            if let Some(hidden_bias) = hidden_bias {
                add_in_place(hidden_scratch.as_dyn_mut(), hidden_bias.as_dyn());
            }

            // Combine inputs for reset and update gates and apply activation.
            let mut update_reset_gates = gates.slice_mut((
                ..,
                gate_range(UPDATE_GATE).start..gate_range(RESET_GATE).end,
            ));
            let hidden_scratch_reset_update_gates = hidden_scratch.slice((
                ..,
                gate_range(UPDATE_GATE).start..gate_range(RESET_GATE).end,
            ));
            add_in_place(
                update_reset_gates.as_dyn_mut(),
                hidden_scratch_reset_update_gates.as_dyn(),
            );

            // Copy gates before applying activation because `sigmoid_in_place`
            // and `tanh_in_place` are slow with non-contiguous tensors, and
            // `update_reset_gates` will be non-contiguous if the batch size is
            // > 1. See https://github.com/robertknight/rten/issues/192.
            //
            // Note `gate_range` can be still used because the update and reset
            // gates are in the same positions in the `update_reset_gates` slice
            // as `gates`.
            let update_reset_gates = sigmoid(pool, update_reset_gates.as_dyn()).auto_return(pool);
            let update_reset_gates = update_reset_gates.nd_view::<2>();
            let update_gate = update_reset_gates.slice((.., gate_range(UPDATE_GATE)));
            let reset_gate = update_reset_gates.slice((.., gate_range(RESET_GATE)));

            // Combine inputs for hidden gate and apply activation.
            let mut hidden_gate_recurrent = hidden_scratch.slice_mut((.., gate_range(HIDDEN_GATE)));
            mul_in_place(hidden_gate_recurrent.as_dyn_mut(), reset_gate.as_dyn());

            let mut hidden_gate = gates.slice_mut((.., gate_range(HIDDEN_GATE)));
            add_in_place(hidden_gate.as_dyn_mut(), hidden_gate_recurrent.as_dyn());

            // See note above about `sigmoid_in_place`.
            let hidden_gate = tanh(pool, hidden_gate.as_dyn()).auto_return(pool);

            // Compute next hidden state
            let mut hidden_item = hidden.slice_mut([dir]);

            for (hidden, update, hidden_gate) in zip3(
                hidden_item.iter_mut(),
                update_gate.iter(),
                hidden_gate.iter(),
            ) {
                *hidden = (1. - update) * hidden_gate + update * (*hidden);
            }

            hidden_seq.slice_mut([seq, dir]).copy_from(&hidden_item);
        }
    }

    Ok([hidden_seq.into_dyn(), hidden.into_dyn()].into())
}

impl Operator for GRU {
    fn name(&self) -> &str {
        "GRU"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let weights = inputs.require_as(1)?;
        let recurrent_weights = inputs.require_as(2)?;
        let bias = inputs.get_as(3)?;
        let _seq_len = inputs.get_as::<TensorView<i32>>(4)?;
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
}

/// Long Short-Term Memory operator.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct LSTM {
    pub direction: Direction,

    #[allow(unused)]
    pub hidden_size: usize, // Currently inferred from operator inputs.
}

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
    input: TensorView,
    weights: TensorView,
    recurrent_weights: TensorView,
    bias: Option<TensorView>,
    initial_hidden: Option<TensorView>,
    initial_cell: Option<TensorView>,
) -> Result<Vec<Tensor>, OpError> {
    // TODO - Add validation of the sizes of individual dimensions in the inputs.
    let input = static_dims!(input, 3, "seq, batch, input")?;
    let [seq_len, batch, _input_size] = input.shape();

    let weights = static_dims!(weights, 3, "dir, hidden x 4, input")?;
    let [_directions, hidden_x4, _input_size] = weights.shape();

    let recurrent_weights = static_dims!(recurrent_weights, 3, "dir, hidden x 4, hidden")?;

    let num_directions = direction.num_directions();
    let hidden_size = hidden_x4 / 4;

    if weights.size(1) % 4 != 0 {
        return Err(OpError::InvalidValue(
            "weights dim 1 must be 4 * hidden_size",
        ));
    }

    let bias = bias.map(|bias| static_dims!(bias, 2)).transpose()?;
    if let Some(bias) = bias.as_ref()
        && bias.size(1) % 8 != 0
    {
        return Err(OpError::InvalidValue("bias dim 1 must be 8 * hidden_size"));
    }

    let initial_hidden = initial_hidden
        .map(|initial_hidden| static_dims!(initial_hidden, 3))
        .transpose()?;
    let initial_cell = initial_cell
        .map(|initial_cell| static_dims!(initial_cell, 3))
        .transpose()?;

    // Contiguous input and bias needed to allow reshaping below.
    let input = input.to_contiguous_in(pool).auto_return(pool);
    let bias = bias.map(|t| t.to_contiguous());

    // Indices of gates in the concatenated weight and bias tensors.
    const INPUT_GATE: usize = 0;
    const OUTPUT_GATE: usize = 1;
    const FORGET_GATE: usize = 2;
    const CELL_GATE: usize = 3;

    let n_gates = 4;
    let mut gates = NdTensor::zeros_in(pool, [batch, n_gates * hidden_size]);

    let mut cell = initial_cell
        .map(|t| t.to_tensor_in(pool))
        .unwrap_or_else(|| NdTensor::zeros_in(pool, [num_directions, batch, hidden_size]));
    let mut hidden = initial_hidden
        .map(|t| t.to_tensor_in(pool))
        .unwrap_or_else(|| NdTensor::zeros_in(pool, [num_directions, batch, hidden_size]));

    let mut hidden_seq =
        NdTensor::<f32, 4>::zeros_in(pool, [seq_len, num_directions, batch, hidden_size]);

    let gemm = GemmExecutor::new();

    let gate_range = |gate| (gate * hidden_size)..((gate + 1) * hidden_size);

    for dir in 0..num_directions {
        let prepack = seq_len >= PREPACK_MIN_SEQ_LEN;

        let input_weights = weights.slice(dir).transposed();
        let packed_input_weights =
            prepack.then(|| gemm.prepack_b_in(pool, input_weights).auto_return(pool));
        let input_weights = packed_input_weights
            .as_ref()
            .map(|packed| GemmInputB::Packed(packed))
            .unwrap_or(GemmInputB::Unpacked(input_weights));

        let hidden_weights = recurrent_weights.slice(dir).transposed();
        let packed_hidden_weights =
            prepack.then(|| gemm.prepack_b_in(pool, hidden_weights).auto_return(pool));
        let hidden_weights = packed_hidden_weights
            .as_ref()
            .map(|packed| GemmInputB::Packed(packed))
            .unwrap_or(GemmInputB::Unpacked(hidden_weights));

        let input_bias = bias
            .as_ref()
            .map(|b| b.slice((dir, ..(n_gates * hidden_size))));
        let hidden_bias = bias
            .as_ref()
            .map(|b| b.slice((dir, (n_gates * hidden_size)..)));

        for seq in sequence_for_dir(direction, dir, seq_len) {
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
            let in_item = input.slice([seq]);
            let hidden_item = hidden.slice([dir]);

            // Update input, output, forget and cell gates.
            gemm.gemm(
                gates.data_mut().expect("expected contiguous input"),
                GemmInputA::Unpacked(in_item),
                input_weights,
                1.,   // alpha
                0.,   // beta
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap();
            if let Some(input_bias) = input_bias {
                add_in_place(gates.as_dyn_mut(), input_bias.as_dyn());
            }

            gemm.gemm(
                gates.data_mut().expect("expected contiguous input"),
                GemmInputA::Unpacked(hidden_item),
                hidden_weights,
                1.,   // alpha
                1.,   // beta
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap();
            if let Some(hidden_bias) = hidden_bias {
                add_in_place(gates.as_dyn_mut(), hidden_bias.as_dyn());
            }

            // Copy gates to work around `tanh_in_place` and `sigmoid_in_place`
            // being slow for non-contiguous inputs. See notes in GRU op.
            let iof_gates = gates.slice((
                ..,
                gate_range(INPUT_GATE).start..gate_range(FORGET_GATE).end,
            ));
            let iof_gates = sigmoid(pool, iof_gates.as_dyn()).auto_return(pool);
            let iof_gates = iof_gates.nd_view::<2>();

            let input_gate = iof_gates.slice((.., gate_range(INPUT_GATE)));
            let out_gate = iof_gates.slice((.., gate_range(OUTPUT_GATE)));
            let forget_gate = iof_gates.slice((.., gate_range(FORGET_GATE)));

            let cell_gate = gates.slice((.., gate_range(CELL_GATE)));
            let cell_gate = tanh(pool, cell_gate.as_dyn()).auto_return(pool);

            // Update cell and hidden state
            let mut cell_item = cell.slice_mut([dir]);

            for (cell, forget_gate, input_gate, cell_gate) in zip4(
                cell_item.iter_mut(),
                forget_gate.iter(),
                input_gate.iter(),
                cell_gate.iter(),
            ) {
                *cell = forget_gate * *cell + input_gate * cell_gate;
            }

            let mut hidden_item = hidden.slice_mut([dir]);
            for (hidden, out_gate, cell) in
                zip3(hidden_item.iter_mut(), out_gate.iter(), cell_item.iter())
            {
                *hidden = out_gate * cell.tanh()
            }

            hidden_seq.slice_mut([seq, dir]).copy_from(&hidden_item);
        }
    }

    Ok([hidden_seq.into_dyn(), hidden.into_dyn(), cell.into_dyn()].into())
}

impl Operator for LSTM {
    fn name(&self) -> &str {
        "LSTM"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let weights = inputs.require_as(1)?;
        let recurrent_weights = inputs.require_as(2)?;
        let bias = inputs.get_as(3)?;
        let _seq_len = inputs.get_as::<TensorView<i32>>(4)?;
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
}

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

    use crate::ops::tests::new_pool;
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
            let pool = new_pool();
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
                    input.as_dyn(),
                    weights.as_dyn(),
                    recurrent_weights.as_dyn(),
                    case.with_bias.then_some(bias.as_dyn()),
                    case.with_hidden_init.then_some(initial_hidden.as_dyn()),
                    case.with_initial_cell.then_some(initial_cell.as_dyn()),
                )
                .expect("lstm op failed"),
                Op::Gru => gru(
                    &pool,
                    dir,
                    input.as_dyn(),
                    weights.as_dyn(),
                    recurrent_weights.as_dyn(),
                    case.with_bias.then_some(bias.as_dyn()),
                    case.with_hidden_init.then_some(initial_hidden.as_dyn()),
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
        let pool = new_pool();
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
        let pool = new_pool();
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
        let pool = new_pool();
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
            let pool = new_pool();
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
                    data.input.view(),
                    data.weights.view(),
                    data.hidden_weights.view(),
                    data.bias.as_ref().map(|b| b.view()),
                    data.initial_hidden.as_ref().map(|ih| ih.view()),
                    data.initial_cell.as_ref().map(|ic| ic.view()),
                )
                .expect("LSTM op failed"),
                Op::Gru => gru(
                    &pool,
                    case.dir,
                    data.input.view(),
                    data.weights.view(),
                    data.hidden_weights.view(),
                    data.bias.as_ref().map(|b| b.view()),
                    data.initial_hidden.as_ref().map(|ih| ih.view()),
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
