use std::iter::{zip, Rev};
use std::ops::Range;

use rten_tensor::prelude::*;
use rten_tensor::Matrix;
use rten_tensor::{NdTensorView, Tensor, TensorView, TensorViewMut};

use crate::check_dims;
use crate::gemm::{GemmExecutor, GemmInputA, GemmInputB};
use crate::ops::{
    add_in_place, sigmoid_in_place, tanh_in_place, InputList, IntoOpResult, OpError, Operator,
    Output,
};

/// Direction that an RNN operator will traverse the input sequence in.
#[derive(Copy, Clone, Debug)]
pub enum Direction {
    Forwards,
    Reverse,
    Bidirectional,
}

impl Direction {
    /// Number of directions that an RNN operator will traverse the sequence in.
    pub fn num_directions(self) -> usize {
        match self {
            Self::Forwards | Self::Reverse => 1,
            Self::Bidirectional => 2,
        }
    }
}

/// Forwards or backwards iterator over values in a range.
enum Sequence {
    Forwards(Range<usize>),
    Backwards(Rev<Range<usize>>),
}

impl Iterator for Sequence {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match self {
            Sequence::Forwards(range) => range.next(),
            Sequence::Backwards(rev_range) => rev_range.next(),
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
        Sequence::Backwards((0..seq_len).rev())
    } else {
        Sequence::Forwards(0..seq_len)
    }
}

/// Like [std::iter::zip], but combines 3 iterators.
fn zip3<T1, T2, T3>(
    a: impl Iterator<Item = T1>,
    b: impl Iterator<Item = T2>,
    c: impl Iterator<Item = T3>,
) -> impl Iterator<Item = (T1, T2, T3)> {
    zip(a, zip(b, c)).map(|(a, (b, c))| (a, b, c))
}

/// Like [std::iter::zip], but combines 4 iterators.
fn zip4<T1, T2, T3, T4>(
    a: impl Iterator<Item = T1>,
    b: impl Iterator<Item = T2>,
    c: impl Iterator<Item = T3>,
    d: impl Iterator<Item = T4>,
) -> impl Iterator<Item = (T1, T2, T3, T4)> {
    zip3(a, b, zip(c, d)).map(|(a, b, (c, d))| (a, b, c, d))
}

/// Like [std::iter::zip], but combines 5 iterators.
fn zip5<T1, T2, T3, T4, T5>(
    a: impl Iterator<Item = T1>,
    b: impl Iterator<Item = T2>,
    c: impl Iterator<Item = T3>,
    d: impl Iterator<Item = T4>,
    e: impl Iterator<Item = T5>,
) -> impl Iterator<Item = (T1, T2, T3, T4, T5)> {
    zip4(a, b, c, zip(d, e)).map(|(a, b, c, (d, e))| (a, b, c, d, e))
}

#[derive(Copy, Clone)]
enum Activation {
    Sigmoid,
    Tanh,
}

/// Compute `output = dot(a, b)`
fn matmul(gemm: &GemmExecutor, mut output: TensorViewMut, a: Matrix, b: GemmInputB) {
    let row_stride = output.stride(output.ndim() - 2);
    gemm.gemm(
        output.data_mut().expect("expected contiguous input"),
        row_stride,
        GemmInputA::Unpacked(a),
        b,
        1., /* alpha */
        0., /* beta */
    );
}

/// Compute `output += dot(a, b)`
fn add_matmul(gemm: &GemmExecutor, mut output: TensorViewMut, a: Matrix, b: GemmInputB) {
    let row_stride = output.stride(output.ndim() - 2);
    gemm.gemm(
        output.data_mut().expect("expected contiguous input"),
        row_stride,
        GemmInputA::Unpacked(a),
        b,
        1., /* alpha */
        1., /* beta */
    );
}

/// Compute output of an RNN gate as:
///
/// `output = act(dot(input, input_weight) + dot(hidden, hidden_weight) + input_bias + hidden_bias)`.
///
/// `output` has shape (batch, hidden_size)
/// `input` has shape (batch, input_size)
/// `input_weight` has shape (input_size, hidden_size)
/// `hidden` has shape (batch, hidden_size)
/// `hidden_weight` has shape (hidden_size, hidden_size)
/// `bias` is a tuple of `(input_bias, hidden_bias)` where each bias has length `hidden_size`
fn compute_rnn_gate(
    gemm: &GemmExecutor,
    mut output: TensorViewMut,
    act: Activation,
    input: &TensorView,
    input_weight: GemmInputB,
    hidden: &TensorView,
    hidden_weight: GemmInputB,
    bias: Option<(NdTensorView<f32, 1>, NdTensorView<f32, 1>)>,
) {
    matmul(gemm, output.view_mut(), input.nd_view(), input_weight);
    add_matmul(gemm, output.view_mut(), hidden.nd_view(), hidden_weight);

    if let Some((in_bias, hidden_bias)) = bias {
        add_in_place(output.view_mut(), in_bias.as_dyn());
        add_in_place(output.view_mut(), hidden_bias.as_dyn());
    }

    match act {
        Activation::Sigmoid => sigmoid_in_place(output),
        Activation::Tanh => tanh_in_place(output),
    }
}

/// Extract a gate weight matrix from a tensor. The tensor has dims
/// `[direction, num_gates * hidden_size, x]`. The result has shape
/// `[x, hidden_size]`.
fn extract_matrix(tensor: TensorView, dir: usize, num_gates: usize, gate_index: usize) -> Matrix {
    let hidden_total = tensor.size(1);
    assert!(hidden_total % num_gates == 0);
    let hidden_size = hidden_total / num_gates;
    tensor
        .slice((
            dir,
            (gate_index * hidden_size..(gate_index + 1) * hidden_size),
        ))
        .nd_view()
}

/// Extract weights and biases for a specific RNN gate/output from a tensor that
/// contains concatenated weights/biases for different gates.
///
/// `weights` has shape `[num_directions, num_gates * hidden_size, input_size]`
/// `recurrent_weights` has shape `[num_directions, num_gates * hidden_size, hidden_size]`
/// `bias` has shape `[num_directions, 2 * num_gates * hidden_size]`.
///
/// Returns `(gate_weights, hidden_weights, bias)` where `gate_weights` has
/// shape `[input_size, hidden_size]`, `hidden_weights` has shape `[hidden_size,
/// hidden_size]` and each element of `bias` has size `hidden_size`.
///
#[allow(clippy::type_complexity)] // Ignore warning about return type
fn extract_weights_and_bias<'a>(
    weights: TensorView<'a>,
    recurrent_weights: TensorView<'a>,
    bias: Option<TensorView<'a>>,
    dir: usize,
    num_gates: usize,
    gate_index: usize,
) -> (
    Matrix<'a>,
    Matrix<'a>,
    Option<(NdTensorView<'a, f32, 1>, NdTensorView<'a, f32, 1>)>,
) {
    let hidden_size = weights.size(1) / num_gates;
    let weight = extract_matrix(weights, dir, num_gates, gate_index).transposed();
    let rec_weight = extract_matrix(recurrent_weights, dir, num_gates, gate_index).transposed();
    let bias = bias.map(|bias| {
        let nth_gate = |gate_index| (gate_index * hidden_size)..((gate_index + 1) * hidden_size);
        let input_bias = bias.slice((dir, nth_gate(gate_index))).nd_view();
        let hidden_bias = bias
            .slice((dir, nth_gate(gate_index + num_gates)))
            .nd_view();
        (input_bias, hidden_bias)
    });
    (weight, rec_weight, bias)
}

/// Gated Recurrent Unit operator.
#[derive(Debug)]
pub struct GRU {
    pub direction: Direction,
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
    direction: Direction,
    input: TensorView,
    weights: TensorView,
    recurrent_weights: TensorView,
    bias: Option<TensorView>,
    initial_hidden: Option<TensorView>,
    linear_before_reset: bool,
) -> Result<Vec<Tensor>, OpError> {
    let [seq_len, batch, input_size] = check_dims!(input, 3, "seq, batch, input");
    let [_directions, hidden_x3, _input_size] = check_dims!(weights, 3, "dir, hidden x 3, input");
    check_dims!(recurrent_weights, 3);
    check_dims!(initial_hidden?, 3);

    let input = input.view();

    let num_directions = direction.num_directions();
    let hidden_size = hidden_x3 / 3;

    let mut hidden = initial_hidden
        .map(|t| t.to_tensor())
        .unwrap_or_else(|| Tensor::zeros(&[num_directions, batch, hidden_size]));
    let mut hidden_seq = Tensor::zeros(&[seq_len, num_directions, batch, hidden_size]);
    let new_gate = || Tensor::zeros(&[batch, hidden_size]);

    let mut update_gate = new_gate();
    let mut reset_gate = new_gate();
    let mut hidden_gate = new_gate();

    // `extract_weights_and_bias` requires a contiguous tensor.
    let bias = bias.map(|t| t.to_contiguous());

    let gemm = GemmExecutor::new();

    // Extract and prepack weights for a gate.
    let extract_gru_weights_and_bias = |dir, gate_index| {
        let (gate_weights, rec_gate_weights, gate_bias) = extract_weights_and_bias(
            weights.view(),
            recurrent_weights.view(),
            bias.as_ref().map(|b| b.view()),
            dir,
            3,
            gate_index,
        );
        let gate_weights = gemm.prepack_b(gate_weights, input_size);
        let rec_gate_weights = gemm.prepack_b(rec_gate_weights, hidden_size);
        (gate_weights, rec_gate_weights, gate_bias)
    };

    // Indices of gates in the concatenated weight and bias tensors.
    const UPDATE_GATE: usize = 0;
    const RESET_GATE: usize = 1;
    const HIDDEN_GATE: usize = 2;

    // Scratch buffer for computing new hidden state.
    let mut hidden_tmp = new_gate();

    for dir in 0..num_directions {
        // Extract and prepack update gate weights.
        let (weight_update, rec_weight_update, bias_update) =
            extract_gru_weights_and_bias(dir, UPDATE_GATE);

        // Extract and prepack reset gate weights.
        let (weight_reset, rec_weight_reset, bias_reset) =
            extract_gru_weights_and_bias(dir, RESET_GATE);

        // Extract and prepack hidden gate weights.
        let (weight_hidden, rec_weight_hidden, bias_hidden) =
            extract_gru_weights_and_bias(dir, HIDDEN_GATE);

        for seq in sequence_for_dir(direction, dir, seq_len) {
            let in_item = input.slice([seq]);
            let hidden_item = hidden.slice([dir]);

            // From the ONNX spec, the intermediate values are computed as:
            //
            //   zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
            //   rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
            //
            //   If `linear_before_reset` is false:
            //     ht = tanh(dot(input, hidden_w) + reset * (dot(hidden, rec_hidden_w) + rec_hidden_bias) + hidden_bias)
            //   Else:
            //     ht = tanh(dot(input, hidden_w) + dot((reset * hidden), rec_hidden_w) + rec_hidden_bias + hidden_bias)
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

            // Compute update gate.
            compute_rnn_gate(
                &gemm,
                update_gate.view_mut(),
                Activation::Sigmoid,
                &in_item,
                GemmInputB::Packed(&weight_update),
                &hidden_item,
                GemmInputB::Packed(&rec_weight_update),
                bias_update,
            );

            // Compute reset gate.
            compute_rnn_gate(
                &gemm,
                reset_gate.view_mut(),
                Activation::Sigmoid,
                &in_item,
                GemmInputB::Packed(&weight_reset),
                &hidden_item,
                GemmInputB::Packed(&rec_weight_reset),
                bias_reset,
            );

            // Compute hidden gate.
            matmul(
                &gemm,
                hidden_gate.view_mut(),
                in_item.nd_view(),
                GemmInputB::Packed(&weight_hidden),
            );
            if linear_before_reset {
                matmul(
                    &gemm,
                    hidden_tmp.view_mut(),
                    hidden_item.nd_view(),
                    GemmInputB::Packed(&rec_weight_hidden),
                );

                // Compute `hidden_gate = tanh(hidden_gate + hidden_bias + reset * (dot(update_tmp) + rec_hidden_bias))`
                if let Some((hidden_bias, rec_hidden_bias)) = bias_hidden {
                    for (hidden_gate, update_tmp, reset, rec_hidden_bias, hidden_bias) in zip5(
                        hidden_gate.iter_mut(),
                        hidden_tmp.iter(),
                        reset_gate.iter(),
                        // Cycle to repeat for each item in batch.
                        rec_hidden_bias.iter().cycle(),
                        hidden_bias.iter().cycle(),
                    ) {
                        let update = reset * (update_tmp + rec_hidden_bias);
                        *hidden_gate = (*hidden_gate + update + hidden_bias).tanh();
                    }
                } else {
                    for (hidden_gate, update_tmp, reset) in
                        zip3(hidden_gate.iter_mut(), hidden_tmp.iter(), reset_gate.iter())
                    {
                        let update = reset * update_tmp;
                        *hidden_gate = (*hidden_gate + update).tanh();
                    }
                }
            } else {
                // TODO - Support alternate GRU variant where hidden gate is
                // computed as:
                //
                //   `hidden_gate = tanh(dot(input, hidden_w) + dot((reset * hidden), rec_hidden_w) + rec_hidden_bias + hidden_bias)
                //
                // Note that cuDNN and PyTorch use the semantics of
                // `linear_before_reset = true`.
                unimplemented!("`linear_before_reset == false` is not supported");
            }

            // Compute next hidden state
            let mut hidden_item = hidden.slice_mut([dir]);
            for (hidden, update, hidden_gate) in zip3(
                hidden_item.iter_mut(),
                update_gate.iter(),
                hidden_gate.iter(),
            ) {
                *hidden = (1. - update) * hidden_gate + update * (*hidden);
            }

            hidden_seq
                .slice_mut([seq, dir])
                .copy_from(&hidden_item.view());
        }
    }

    Ok([hidden_seq, hidden].into())
}

impl Operator for GRU {
    fn name(&self) -> &str {
        "GRU"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        let weights = inputs.require_as(1)?;
        let recurrent_weights = inputs.require_as(2)?;
        let bias = inputs.get_as(3)?;
        let _seq_len = inputs.get_as::<i32>(4)?;
        let initial_hidden = inputs.get_as(5)?;

        gru(
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
pub struct LSTM {
    pub direction: Direction,
    pub hidden_size: usize,
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
    direction: Direction,
    input: TensorView,
    weights: TensorView,
    recurrent_weights: TensorView,
    bias: Option<TensorView>,
    initial_hidden: Option<TensorView>,
    initial_cell: Option<TensorView>,
) -> Result<Vec<Tensor>, OpError> {
    // TODO - Add validation of the sizes of individual dimensions in the inputs.
    let [seq_len, batch, input_size] = check_dims!(input, 3, "seq, batch, input");
    let [_directions, hidden_x4, _input_size] = check_dims!(weights, 3, "dir, hidden x 4, input");
    check_dims!(recurrent_weights, 3);

    let num_directions = direction.num_directions();
    let hidden_size = hidden_x4 / 4;

    if weights.size(1) % 4 != 0 {
        return Err(OpError::InvalidValue(
            "weights dim 1 must be 4 * hidden_size",
        ));
    }
    if let Some(bias) = bias.as_ref() {
        check_dims!(bias, 2);
        if bias.size(1) % 8 != 0 {
            return Err(OpError::InvalidValue("bias dim 1 must be 8 * hidden_size"));
        }
    }
    check_dims!(initial_hidden?, 3);
    check_dims!(initial_cell?, 3);

    // Contiguous input and bias needed to allow reshaping below.
    let input = input.to_contiguous();
    let bias = bias.map(|t| t.to_contiguous());

    // Indices of gates in the concatenated weight and bias tensors.
    const INPUT_GATE: usize = 0;
    const OUTPUT_GATE: usize = 1;
    const FORGET_GATE: usize = 2;
    const CELL_GATE: usize = 3;

    let new_gate = || Tensor::zeros(&[batch, hidden_size]);
    let mut input_gate = new_gate();
    let mut out_gate = new_gate();
    let mut forget_gate = new_gate();
    let mut cell_gate = new_gate();

    let mut cell = initial_cell
        .map(|t| t.to_tensor())
        .unwrap_or_else(|| Tensor::zeros(&[num_directions, batch, hidden_size]));
    let mut hidden = initial_hidden
        .map(|t| t.to_tensor())
        .unwrap_or_else(|| Tensor::zeros(&[num_directions, batch, hidden_size]));

    let mut hidden_seq = Tensor::<f32>::zeros(&[seq_len, num_directions, batch, hidden_size]);

    let gemm = GemmExecutor::new();

    let extract_lstm_weights_and_bias = |dir, gate_index| {
        let (gate_weights, rec_gate_weights, gate_bias) = extract_weights_and_bias(
            weights.view(),
            recurrent_weights.view(),
            bias.as_ref().map(|b| b.view()),
            dir,
            4,
            gate_index,
        );
        let gate_weights = gemm.prepack_b(gate_weights, input_size);
        let rec_gate_weights = gemm.prepack_b(rec_gate_weights, hidden_size);
        (gate_weights, rec_gate_weights, gate_bias)
    };

    for dir in 0..num_directions {
        let (weight_input, rec_weight_input, bias_input) =
            extract_lstm_weights_and_bias(dir, INPUT_GATE);
        let (weight_out, rec_weight_out, bias_out) =
            extract_lstm_weights_and_bias(dir, OUTPUT_GATE);
        let (weight_forget, rec_weight_forget, bias_forget) =
            extract_lstm_weights_and_bias(dir, FORGET_GATE);
        let (weight_cell, rec_weight_cell, bias_cell) =
            extract_lstm_weights_and_bias(dir, CELL_GATE);

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

            // Compute outputs for input, forget, cell and output gates.
            compute_rnn_gate(
                &gemm,
                input_gate.view_mut(),
                Activation::Sigmoid,
                &in_item,
                GemmInputB::Packed(&weight_input),
                &hidden_item,
                GemmInputB::Packed(&rec_weight_input),
                bias_input,
            );

            compute_rnn_gate(
                &gemm,
                forget_gate.view_mut(),
                Activation::Sigmoid,
                &in_item,
                GemmInputB::Packed(&weight_forget),
                &hidden_item,
                GemmInputB::Packed(&rec_weight_forget),
                bias_forget,
            );

            compute_rnn_gate(
                &gemm,
                cell_gate.view_mut(),
                Activation::Tanh,
                &in_item,
                GemmInputB::Packed(&weight_cell),
                &hidden_item,
                GemmInputB::Packed(&rec_weight_cell),
                bias_cell,
            );

            compute_rnn_gate(
                &gemm,
                out_gate.view_mut(),
                Activation::Sigmoid,
                &in_item,
                GemmInputB::Packed(&weight_out),
                &hidden_item,
                GemmInputB::Packed(&rec_weight_out),
                bias_out,
            );

            // Compute new values of cell and hidden state
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

            hidden_seq
                .slice_mut([seq, dir])
                .copy_from(&hidden_item.view());
        }
    }

    Ok([hidden_seq, hidden, cell].into())
}

impl Operator for LSTM {
    fn name(&self) -> &str {
        "LSTM"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        let weights = inputs.require_as(1)?;
        let recurrent_weights = inputs.require_as(2)?;
        let bias = inputs.get_as(3)?;
        let _seq_len = inputs.get_as::<i32>(4)?;
        let initial_hidden = inputs.get_as(5)?;
        let initial_cell = inputs.get_as(6)?;

        lstm(
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
    use std::error::Error;
    use std::fs::File;
    use std::io::BufReader;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::Tensor;
    use serde_json::Value;

    use crate::ops::{concat, gru, lstm, split, Direction};

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

    #[derive(Clone, Copy, PartialEq)]
    enum Op {
        Gru,
        Lstm,
    }

    // Basic test that runs bidirectional RNN operators with random inputs and
    // checks that the operator doesn't crash, produces outputs of the right
    // shape and that the last hidden / hidden seq outputs are consistent.
    #[test]
    fn test_rnn_ops_with_random_input() {
        let mut rng = XorShiftRng::new(1234);
        let batch = 2;
        let seq_len = 5;
        let dir = Direction::Bidirectional;

        let hidden_size = 3;
        let features = 2;

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

        for case in cases {
            let num_gates = match case.op {
                Op::Gru => 3,
                Op::Lstm => 4,
            };

            let input = Tensor::rand(&[seq_len, batch, features], &mut rng).map(|x| x - 0.5);
            let weights = Tensor::rand(
                &[dir.num_directions(), num_gates * hidden_size, features],
                &mut rng,
            )
            .map(|x| x - 0.5);
            let recurrent_weights = Tensor::rand(
                &[dir.num_directions(), num_gates * hidden_size, hidden_size],
                &mut rng,
            )
            .map(|x| x - 0.5);
            let bias = Tensor::rand(
                &[dir.num_directions(), 2 * num_gates * hidden_size],
                &mut rng,
            );
            let initial_hidden =
                Tensor::rand(&[dir.num_directions(), batch, hidden_size], &mut rng);
            let initial_cell = Tensor::rand(&[dir.num_directions(), batch, hidden_size], &mut rng);

            let result = match case.op {
                Op::Lstm => lstm(
                    dir,
                    input.view(),
                    weights.view(),
                    recurrent_weights.view(),
                    case.with_bias.then_some(bias.view()),
                    case.with_hidden_init.then_some(initial_hidden.view()),
                    case.with_initial_cell.then_some(initial_cell.view()),
                )
                .expect("lstm op failed"),
                Op::Gru => gru(
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
            let hss = hidden_seq.shape();
            let hidden_seq_fwd = hidden_seq
                .slice_iter(&[
                    (..hss[0]).into(), // seq
                    (0..1).into(),     // direction
                    (..hss[2]).into(), // batch
                    (..hss[3]).into(), // hidden
                ])
                .collect::<Vec<_>>();
            let last_hidden_fwd = last_hidden
                .slice_iter(&[(0..1).into(), (..batch).into(), (..hidden_size).into()])
                .collect::<Vec<_>>();

            assert_eq!(
                hidden_seq_fwd[hidden_seq_fwd.len() - batch * hidden_size..],
                last_hidden_fwd
            );

            let hidden_seq_rev = hidden_seq
                .slice_iter(&[
                    (..hss[0]).into(), // seq
                    (1..2).into(),     // direction
                    (..hss[2]).into(), // batch
                    (..hss[3]).into(), // hidden
                ])
                .collect::<Vec<_>>();
            let last_hidden_rev = last_hidden
                .slice_iter(&[(1..2).into(), (..batch).into(), (..hidden_size).into()])
                .collect::<Vec<_>>();
            assert_eq!(hidden_seq_rev[0..batch * hidden_size], last_hidden_rev);
        }
    }

    /// Re-order a weight or bias tensor for LSTM gates from (input, forget,
    /// cell, output) as used by PyTorch to (input, output, forget, cell) as
    /// used by ONNX.
    fn reorder_ifco_to_iofc(x: &Tensor, axis: isize) -> Tensor {
        let size = x.size(axis as usize) / 4;
        let splits = &[size as i32; 4];

        // Split input into seperate tensor for each of the gates.
        let ifco = split(x.view(), axis, &splits.into()).expect("split failed");

        // Recombine in a new gate order.
        concat(
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
        let size = x.size(axis as usize) / 3;
        let splits = &[size as i32; 3];

        // Split input into seperate tensor for each of the gates.
        let ruh = split(x.view(), axis, &splits.into()).expect("split failed");

        // Recombine in a new gate order.
        concat(&[ruh[1].view(), ruh[0].view(), ruh[2].view()], axis).expect("concat failed")
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
        let params = &case["params"];

        let is_bidirectional = params.get("weight_ih_l0_reverse").is_some();

        let mut input = read_tensor(&case["input"]).expect("failed to read input");
        input.insert_dim(1); // Add batch dim

        let mut expected = read_tensor(&case["output"]).expect("failed to read output");

        // Reshape from [seq, dir * hidden_size] to [seq, dir, hidden_size]
        if is_bidirectional {
            let es = expected.shape();
            expected.reshape(&[es[0], 2, es[1] / 2]);
        } else {
            expected.insert_dim(1);
        }
        expected.insert_dim(2); // Add batch dim

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
        weights.insert_dim(0); // Add directions dim

        let mut hidden_weights = read_param("weight_hh_l0");
        hidden_weights.insert_dim(0); // Add directions dim

        let input_bias = read_param("bias_ih_l0");
        let hidden_bias = read_param("bias_hh_l0");
        let mut bias = concat(&[input_bias.view(), hidden_bias.view()], 0).unwrap();
        bias.insert_dim(0); // Add directions dim

        // If this is a bidirectional RNN, there will be `_reverse`-suffixed
        // versions of the bias and weight params. Extract these and concatenate
        // with the forwards direction values.
        if is_bidirectional {
            let mut rev_weights = read_param("weight_ih_l0_reverse");
            rev_weights.insert_dim(0); // Add directions dim
            weights = concat(&[weights.view(), rev_weights.view()], 0).unwrap();

            let mut rev_hidden_weights = read_param("weight_hh_l0_reverse");
            rev_hidden_weights.insert_dim(0); // Add directions dim
            hidden_weights =
                concat(&[hidden_weights.view(), rev_hidden_weights.view()], 0).unwrap();

            let rev_input_bias = read_param("bias_ih_l0_reverse");
            let rev_hidden_bias = read_param("bias_hh_l0_reverse");
            let mut rev_bias = concat(&[rev_input_bias.view(), rev_hidden_bias.view()], 0).unwrap();
            rev_bias.insert_dim(0); // Add directions dim
            bias = concat(&[bias.view(), rev_bias.view()], 0).unwrap();
        }

        let initial_hidden = case.get("initial_hidden").map(|param| {
            let mut init = read_tensor(param).expect("failed to read initial hidden state");
            init.insert_dim(1); // Add batch dim
            init
        });

        let initial_cell = case.get("initial_cell").map(|param| {
            let mut init = read_tensor(param).expect("failed to read initial cell state");
            init.insert_dim(1); // Add batch dim
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
    fn test_rnn_pytorch() -> Result<(), Box<dyn Error>> {
        let dict = read_json_file("pytorch-ref-tests/rnn.json");

        struct Case {
            name: &'static str,
            dir: Direction,
        }

        let cases = &[
            Case {
                name: "lstm_forwards",
                dir: Direction::Forwards,
            },
            Case {
                name: "lstm_initial",
                dir: Direction::Forwards,
            },
            Case {
                name: "lstm_bidirectional",
                dir: Direction::Bidirectional,
            },
            Case {
                name: "gru_forwards",
                dir: Direction::Forwards,
            },
            Case {
                name: "gru_initial",
                dir: Direction::Forwards,
            },
            Case {
                name: "gru_bidirectional",
                dir: Direction::Bidirectional,
            },
        ];

        for case in cases {
            let op = if case.name.starts_with("lstm") {
                Op::Lstm
            } else {
                Op::Gru
            };
            let data = read_pytorch_ref_test(op, &dict[case.name]);
            let result = match op {
                Op::Lstm => lstm(
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

            expect_equal(output, &data.expected)?;
        }

        Ok(())
    }

    // TODO - Add tests for incorrect input shapes
}
