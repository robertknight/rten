use std::iter::zip;

use crate::check_dims;
use crate::linalg::gemm;
use crate::ndtensorview::Matrix;
use crate::ops::unary_elementwise::UnaryFloatOp;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output, Sigmoid, Tanh};
use crate::tensor::{AsMatrix, Tensor, TensorLayout, TensorView};

#[derive(Copy, Clone, Debug)]
pub enum LSTMDirection {
    Forwards,
    Reverse,
    Bidirectional,
}

impl LSTMDirection {
    /// Number of directions that an LSTM operator will traverse the sequence in.
    ///
    /// The sizes of various inputs and outputs depend on this.
    pub fn num_directions(self) -> usize {
        match self {
            Self::Forwards | Self::Reverse => 1,
            Self::Bidirectional => 2,
        }
    }
}

#[derive(Debug)]
pub struct LSTM {
    pub direction: LSTMDirection,
    pub hidden_size: usize,
}

#[derive(Copy, Clone)]
enum Activation {
    Sigmoid,
    Tanh,
}

/// Compute output of an LSTM gate, according to the formula:
///
/// `output = act(dot(input, input_weight) + dot(hidden, hidden_weight) + input_bias + hidden_bias)`.
///
/// `input_weight` has shape (input_size, hidden_size)
/// `hidden_weight` has shape (hidden_size, hidden_size)
/// `bias` is a tuple of `(input_bias, hidden_bias)`.
fn update_lstm_gate(
    output: &mut [f32],
    act: Activation,
    input: &TensorView,
    input_weight: Matrix,
    hidden: &TensorView,
    hidden_weight: Matrix,
    bias: Option<(&[f32], &[f32])>,
) {
    let sigmoid_op = Sigmoid {};
    let tanh_op = Tanh {};

    gemm(
        output,
        output.len(),
        input.as_matrix(),
        input_weight,
        1., /* alpha */
        0., /* beta */
    );

    gemm(
        output,
        output.len(),
        hidden.as_matrix(),
        hidden_weight,
        1., /* alpha */
        1., /* beta */
    );

    let apply_act = |el: f32| match act {
        Activation::Sigmoid => sigmoid_op.map_element(el),
        Activation::Tanh => tanh_op.map_element(el),
    };

    if let Some((in_bias, hidden_bias)) = bias {
        let combined_bias = zip(in_bias.iter(), hidden_bias.iter()).map(|(ib, hb)| ib + hb);
        for (el, bias) in zip(output.iter_mut(), combined_bias) {
            *el = apply_act(*el + bias);
        }
    } else {
        for el in output.iter_mut() {
            *el = apply_act(*el);
        }
    }
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
    direction: LSTMDirection,
    input: &Tensor,
    weights: &Tensor,
    recurrent_weights: &Tensor,
    bias: Option<&Tensor>,
    initial_hidden: Option<&Tensor>,
    initial_cell: Option<&Tensor>,
) -> Result<Vec<Tensor>, OpError> {
    // TODO - Add validation of the sizes of individual dimensions in the inputs.
    let [seq_len, batch, input_size] = check_dims!(input, 3, "seq, batch, input");
    let [_directions, hidden_x4, _input_size] = check_dims!(weights, 3, "dir, hidden x 4, input");
    check_dims!(recurrent_weights, 3);

    let num_directions = direction.num_directions();
    let hidden_size = hidden_x4 / 4;

    if weights.shape()[1] % 4 != 0 {
        return Err(OpError::InvalidValue(
            "weights dim 1 must be 4 * hidden_size",
        ));
    }
    if let Some(bias) = bias {
        check_dims!(bias, 2);
        if bias.shape()[1] % 8 != 0 {
            return Err(OpError::InvalidValue("bias dim 1 must be 8 * hidden_size"));
        }
    }
    check_dims!(initial_hidden?, 3);
    check_dims!(initial_cell?, 3);

    // Contiguous input and bias needed to allow reshaping below.
    let input = input.as_contiguous();
    let bias = bias.map(|t| t.as_contiguous());

    // Extract an LSTM gate weight matrix from a tensor. The tensor has dims
    // [direction, 4 * hidden_size, *].
    fn extract_matrix(tensor: &Tensor, dir: usize, index: usize) -> Matrix {
        let num_gates = 4;
        let hidden_total = tensor.shape()[1];
        assert!(hidden_total % num_gates == 0);
        let hidden_size = hidden_total / num_gates;

        tensor
            .slice(&[
                dir.into(),
                (index * hidden_size..(index + 1) * hidden_size).into(),
            ])
            .as_matrix()
    }

    // Specifies which gate's weight or bias to extract.
    const INPUT_GATE: usize = 0;
    const OUTPUT_GATE: usize = 1;
    const FORGET_GATE: usize = 2;
    const CELL_GATE: usize = 3;

    let extract_weights_and_bias = |dir, gate_index| {
        let weight = extract_matrix(weights, dir, gate_index).transposed();
        let rec_weight = extract_matrix(recurrent_weights, dir, gate_index).transposed();
        let bias = bias.as_ref().map(|bias| {
            let input_bias = bias.last_dim_slice([dir, gate_index * hidden_size], hidden_size);
            let hidden_bias =
                bias.last_dim_slice([dir, (gate_index + 4) * hidden_size], hidden_size);
            (input_bias, hidden_bias)
        });
        (weight, rec_weight, bias)
    };

    let mut input_gate = Tensor::zeros(&[hidden_size]);
    let mut out_gate = Tensor::zeros(&[hidden_size]);
    let mut forget_gate = Tensor::zeros(&[hidden_size]);
    let mut cell_gate = Tensor::zeros(&[hidden_size]);

    let mut cell = initial_cell
        .cloned()
        .unwrap_or_else(|| Tensor::zeros(&[num_directions, batch, hidden_size]));
    let mut hidden = initial_hidden
        .map(|t| t.as_contiguous().into_owned()) // Needed due to `last_dim_slice` usage
        .unwrap_or_else(|| Tensor::zeros(&[num_directions, batch, hidden_size]));

    let mut hidden_seq = Tensor::<f32>::zeros(&[seq_len, num_directions, batch, hidden_size]);

    for dir in 0..num_directions {
        let (weight_input, rec_weight_input, bias_input) =
            extract_weights_and_bias(dir, INPUT_GATE);
        let (weight_out, rec_weight_out, bias_out) = extract_weights_and_bias(dir, OUTPUT_GATE);
        let (weight_forget, rec_weight_forget, bias_forget) =
            extract_weights_and_bias(dir, FORGET_GATE);
        let (weight_cell, rec_weight_cell, bias_cell) = extract_weights_and_bias(dir, CELL_GATE);

        let reversed = matches!(
            (dir, direction),
            (0, LSTMDirection::Reverse) | (1, LSTMDirection::Bidirectional)
        );

        let mut forward_seq = 0..seq_len;
        let mut rev_seq = (0..seq_len).rev();
        let seq_iter: &mut dyn Iterator<Item = _> = if reversed {
            &mut rev_seq
        } else {
            &mut forward_seq
        };

        for seq in seq_iter {
            for b in 0..batch {
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
                //  - `f`, `g` and `h` are activation functions. `f`=sigmoid, `g` and `h`
                //    are tanh.
                //
                //  - `W{i,o,f,c}` are the gate weights
                //  - `Wb{i,o,f,c}` are the gate biases
                //  - `R{i,o,f,c}` are the hidden/recurrent gate weights
                //  - `Rb{i,o,f,c}` are the hidden/recurrent gate biases
                //
                //  - `P{i,o,f,c}` are peephole weights. These are not currently
                //    supported.

                let in_item = input
                    .slice(&[seq.into(), b.into()])
                    .reshaped(&[1, input_size]);
                let hidden_item = hidden
                    .slice(&[dir.into(), b.into()])
                    .reshaped(&[1, hidden_size]);

                // Compute outputs for input, forget, cell and output gates.
                update_lstm_gate(
                    input_gate.data_mut(),
                    Activation::Sigmoid,
                    &in_item,
                    weight_input,
                    &hidden_item,
                    rec_weight_input,
                    bias_input,
                );

                update_lstm_gate(
                    forget_gate.data_mut(),
                    Activation::Sigmoid,
                    &in_item,
                    weight_forget,
                    &hidden_item,
                    rec_weight_forget,
                    bias_forget,
                );

                update_lstm_gate(
                    cell_gate.data_mut(),
                    Activation::Tanh,
                    &in_item,
                    weight_cell,
                    &hidden_item,
                    rec_weight_cell,
                    bias_cell,
                );

                update_lstm_gate(
                    out_gate.data_mut(),
                    Activation::Sigmoid,
                    &in_item,
                    weight_out,
                    &hidden_item,
                    rec_weight_out,
                    bias_out,
                );

                // Compute new values of cell and hidden state
                let mut cell_item = cell.slice_mut(&[dir.into(), b.into()]);

                for (cell, (forget_gate, (input_gate, cell_gate))) in zip(
                    cell_item.iter_mut(),
                    zip(forget_gate.iter(), zip(input_gate.iter(), cell_gate.iter())),
                ) {
                    *cell = forget_gate * *cell + input_gate * cell_gate;
                }

                let mut hidden_item = hidden.slice_mut(&[dir.into(), b.into()]);
                let tanh_op = Tanh {};
                for (hidden, (out_gate, cell)) in zip(
                    hidden_item.iter_mut(),
                    zip(out_gate.iter(), cell_item.iter()),
                ) {
                    *hidden = out_gate * tanh_op.map_element(cell)
                }

                // Copy latest value of hidden seq to output tensor
                let mut hidden_seq_item = hidden_seq.slice_mut(&[seq.into(), dir.into(), b.into()]);
                hidden_seq_item
                    .data_mut()
                    .clone_from_slice(hidden_item.data_mut());
            }
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
    use serde_json::Value;

    use crate::ops::{concat, lstm, split, LSTMDirection};
    use crate::rng::XorShiftRng;
    use crate::tensor::{rand, Tensor, TensorLayout};
    use crate::test_util::{expect_equal, read_json_file, read_tensor};

    // Basic test that runs a bidirectional LSTM with random inputs and checks
    // that the operator doesn't crash, produces outputs of the right shape
    // and that the last hidden / hidden seq outputs are consistent.
    #[test]
    fn test_lstm_with_random_input() {
        let mut rng = XorShiftRng::new(1234);
        let batch = 2;
        let seq_len = 5;
        let dir = LSTMDirection::Bidirectional;

        let hidden_size = 3;
        let features = 2;
        let input = rand(&[seq_len, batch, features], &mut rng).map(|x| x - 0.5);
        let weights =
            rand(&[dir.num_directions(), 4 * hidden_size, features], &mut rng).map(|x| x - 0.5);
        let recurrent_weights = rand(
            &[dir.num_directions(), 4 * hidden_size, hidden_size],
            &mut rng,
        )
        .map(|x| x - 0.5);
        let bias = rand(&[dir.num_directions(), 8 * hidden_size], &mut rng);
        let initial_hidden = rand(&[dir.num_directions(), batch, hidden_size], &mut rng);
        let initial_cell = rand(&[dir.num_directions(), batch, hidden_size], &mut rng);

        struct Case {
            with_bias: bool,
            with_hidden_init: bool,
            with_initial_cell: bool,
        }

        let cases = [
            Case {
                with_bias: true,
                with_hidden_init: true,
                with_initial_cell: true,
            },
            Case {
                with_bias: false,
                with_hidden_init: false,
                with_initial_cell: false,
            },
        ];

        for case in cases {
            let result = lstm(
                dir,
                &input,
                &weights,
                &recurrent_weights,
                case.with_bias.then_some(&bias),
                case.with_hidden_init.then_some(&initial_hidden),
                case.with_initial_cell.then_some(&initial_cell),
            )
            .expect("lstm op failed");

            // Check that outputs have the right shapes.
            assert_eq!(result.len(), 3);
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

            let last_cell = &result[2];
            assert_eq!(
                last_cell.shape(),
                &[dir.num_directions(), batch, hidden_size]
            );

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
    fn reorder_ifco_to_iofc(x: &Tensor, dim: usize) -> Tensor {
        let size = x.shape()[dim] / 4;
        let splits = Tensor::from_vec(vec![size as i32; 4]);

        // Split input into seperate tensor for each of the gates.
        let ifco = split(x, dim as isize, &splits).expect("split failed");

        // Recombine in a new gate order.
        concat(&[&ifco[0], &ifco[3], &ifco[1], &ifco[2]], dim).expect("concat failed")
    }

    struct LSTMRefTest {
        /// Input as [seq, batch, feature]
        input: Tensor,

        /// Expected output as [seq, direction, batch, hidden]
        expected: Tensor,

        /// Input-hidden weights as [direction, 4 * hidden, feature]
        weights: Tensor,

        /// Hidden-hidden weights as [direction, 4 * hidden, 4 * hidden]
        hidden_weights: Tensor,

        /// Bias as [direction, 8 * hidden]
        bias: Option<Tensor>,

        /// Initial value of the hidden state as [direction, batch, hidden]
        initial_hidden: Option<Tensor>,

        /// Initial value of the cell state as [direction, batch, hidden]
        initial_cell: Option<Tensor>,
    }

    /// Read inputs for a PyTorch reference test for LSTM ops from a JSON value.
    fn read_pytorch_ref_test(case: &Value) -> LSTMRefTest {
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

        let read_param = |name| {
            reorder_ifco_to_iofc(
                &read_tensor(&params[name]).expect("failed to read weight"),
                0,
            )
        };

        let mut weights = read_param("weight_ih_l0");
        weights.insert_dim(0); // Add directions dim

        let mut hidden_weights = read_param("weight_hh_l0");
        hidden_weights.insert_dim(0); // Add directions dim

        let input_bias = read_param("bias_ih_l0");
        let hidden_bias = read_param("bias_hh_l0");
        let mut bias = concat(&[&input_bias, &hidden_bias], 0).unwrap();
        bias.insert_dim(0); // Add directions dim

        // If this is a bidirectional LSTM, there will be `_reverse`-suffixed
        // versions of the bias and weight params. Extract these and concatenate
        // with the forwards direction values.
        if is_bidirectional {
            let mut rev_weights = read_param("weight_ih_l0_reverse");
            rev_weights.insert_dim(0); // Add directions dim
            weights = concat(&[&weights, &rev_weights], 0).unwrap();

            let mut rev_hidden_weights = read_param("weight_hh_l0_reverse");
            rev_hidden_weights.insert_dim(0); // Add directions dim
            hidden_weights = concat(&[&hidden_weights, &rev_hidden_weights], 0).unwrap();

            let rev_input_bias = read_param("bias_ih_l0_reverse");
            let rev_hidden_bias = read_param("bias_hh_l0_reverse");
            let mut rev_bias = concat(&[&rev_input_bias, &rev_hidden_bias], 0).unwrap();
            rev_bias.insert_dim(0); // Add directions dim
            bias = concat(&[&bias, &rev_bias], 0).unwrap();
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

        LSTMRefTest {
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
    fn test_lstm_pytorch() -> Result<(), String> {
        let dict = read_json_file("pytorch-ref-tests/lstm.json");

        struct Case {
            name: &'static str,
            dir: LSTMDirection,
        }

        let cases = &[
            Case {
                name: "lstm_forwards",
                dir: LSTMDirection::Forwards,
            },
            Case {
                name: "lstm_initial",
                dir: LSTMDirection::Forwards,
            },
            Case {
                name: "lstm_bidirectional",
                dir: LSTMDirection::Bidirectional,
            },
        ];

        for case in cases {
            let data = read_pytorch_ref_test(&dict[case.name]);
            let result = lstm(
                case.dir,
                &data.input,
                &data.weights,
                &data.hidden_weights,
                data.bias.as_ref(),
                data.initial_hidden.as_ref(),
                data.initial_cell.as_ref(),
            )
            .expect("LSTM op failed");
            let output = &result[0];

            expect_equal(&output, &data.expected)?;
        }

        Ok(())
    }

    // TODO - Add tests for incorrect input shapes
}
