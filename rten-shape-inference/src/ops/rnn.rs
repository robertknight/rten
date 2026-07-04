use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Number of directions an RNN operator traverses the input sequence in.
///
/// Shape inference only depends on whether the operator is uni- or
/// bidirectional, not on whether a unidirectional operator runs forwards or
/// in reverse.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Direction {
    Unidirectional,
    Bidirectional,
}

/// Compute shapes of (Y, Y_h) outputs of GRU and LSTM ops.
///
/// The LSTM op has an additional Y_c output with the same shape as Y_h.
///
/// Returns `Ok(None)` if the output shapes cannot be determined because an
/// input shape is unknown, or an error if a known input shape has the wrong
/// rank.
fn rnn_output_shapes(
    input: &SymTensor,
    weights: &SymTensor,
    direction: Direction,
    gates: i32,
) -> Result<Option<(SymTensor, SymTensor)>, InferShapesError> {
    let num_directions = match direction {
        Direction::Unidirectional => SymExpr::Value(1),
        Direction::Bidirectional => SymExpr::Value(2),
    };

    // Input shape is (seq_len, batch, input_size).
    let Some(input_dims) = input.shape() else {
        return Ok(None);
    };
    let input_dims: Vec<_> = input_dims.collect();
    if input_dims.len() != 3 {
        return Err(InferShapesError::IncorrectRank);
    }
    let seq_len = input_dims[0].clone();
    let batch = input_dims[1].clone();

    // Weights shape is (directions, gates * hidden, input_size).
    let Some(mut wdims) = weights.shape() else {
        return Ok(None);
    };
    let (Some(_dirs), Some(gates_x_hidden), Some(_input)) =
        (wdims.next(), wdims.next(), wdims.next())
    else {
        return Err(InferShapesError::IncorrectRank);
    };
    let hidden = gates_x_hidden / SymExpr::Value(gates);

    let y_shape = vec![
        seq_len,
        num_directions.clone(),
        batch.clone(),
        hidden.clone(),
    ];
    let y_h_shape = vec![num_directions, batch, hidden];

    Ok(Some((
        SymTensor::from_shape(y_shape),
        SymTensor::from_shape(y_h_shape),
    )))
}

/// GRU operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__GRU.html>.
pub struct GRU {
    pub direction: Direction,
}

impl InferShapes for GRU {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let input = inputs.require(0)?;
        let weights = inputs.require(1)?;
        // Recurrent weights (input 2) are required, though not used here.
        inputs.require(2)?;

        match rnn_output_shapes(input, weights, self.direction, 3)? {
            Some((y, y_h)) => Ok([y, y_h].into()),
            None => Ok([
                SymTensor::unknown("unknown rnn shape"),
                SymTensor::unknown("unknown rnn shape"),
            ]
            .into()),
        }
    }
}

/// LSTM operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__LSTM.html>.
pub struct LSTM {
    pub direction: Direction,
}

impl InferShapes for LSTM {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let input = inputs.require(0)?;
        let weights = inputs.require(1)?;
        // Recurrent weights (input 2) are required, though not used here.
        inputs.require(2)?;

        match rnn_output_shapes(input, weights, self.direction, 4)? {
            Some((y, y_h)) => {
                // Y_c has the same shape as Y_h.
                let y_c = y_h.clone();
                Ok([y, y_h, y_c].into())
            }
            None => Ok([
                SymTensor::unknown("unknown rnn shape"),
                SymTensor::unknown("unknown rnn shape"),
                SymTensor::unknown("unknown rnn shape"),
            ]
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::{Direction, GRU, LSTM};

    #[test]
    fn test_gru() {
        let mut sym_gen = SymbolGen::new();

        let hidden = 32;

        // Forward GRU. hidden_size is inferred from the weights' middle dim
        // (3 * hidden).
        let input = sym_shape!("seq", "batch", 64);
        let weights = sym_shape!(1, 3 * hidden, 64);
        let recurrent = sym_shape!(1, 3 * hidden, hidden);
        let op = GRU {
            direction: Direction::Unidirectional,
        };
        let result = op
            .infer_shapes([input, weights, recurrent].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!("seq", 1, "batch", hidden)
        );
        assert_eq!(result[1].clone().simplify(), sym_shape!(1, "batch", hidden));

        // Bidirectional GRU.
        let input = sym_shape!("seq", "batch", 64);
        let weights = sym_shape!(2, 3 * hidden, 64);
        let recurrent = sym_shape!(2, 3 * hidden, hidden);
        let op = GRU {
            direction: Direction::Bidirectional,
        };
        let result = op
            .infer_shapes([input, weights, recurrent].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!("seq", 2, "batch", hidden)
        );
        assert_eq!(result[1].clone().simplify(), sym_shape!(2, "batch", hidden));

        // Symbolic hidden size inferred from weights.
        let input = sym_shape!("seq", "batch", "input");
        let weights = sym_shape!(1, SymExpr::from("hx3"), "input");
        let recurrent = sym_shape!(1, SymExpr::from("hx3"), "hidden");
        let op = GRU {
            direction: Direction::Unidirectional,
        };
        let result = op
            .infer_shapes([input, weights, recurrent].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!("seq", 1, "batch", SymExpr::from("hx3") / SymExpr::from(3))
        );
    }

    #[test]
    fn test_lstm() {
        let mut sym_gen = SymbolGen::new();

        let hidden = 32;

        // Forward LSTM. Produces Y, Y_h, Y_c. hidden_size is inferred from the
        // weights' middle dim (4 * hidden).
        let input = sym_shape!("seq", "batch", 64);
        let weights = sym_shape!(1, 4 * hidden, 64);
        let recurrent = sym_shape!(1, 4 * hidden, hidden);
        let op = LSTM {
            direction: Direction::Unidirectional,
        };
        let result = op
            .infer_shapes([input, weights, recurrent].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!("seq", 1, "batch", hidden)
        );
        assert_eq!(result[1].clone().simplify(), sym_shape!(1, "batch", hidden));
        assert_eq!(result[2].clone().simplify(), sym_shape!(1, "batch", hidden));

        // Unknown weights — output shapes are unknown.
        let input = sym_shape!("seq", "batch", 64);
        let weights = SymTensor::unknown("unknown");
        let recurrent = SymTensor::unknown("unknown");
        let op = LSTM {
            direction: Direction::Unidirectional,
        };
        let result = op
            .infer_shapes([input, weights, recurrent].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);
        assert_eq!(result[1].ndim(), None);
        assert_eq!(result[2].ndim(), None);

        // Input with the wrong rank — this is an error.
        let input = sym_shape!("batch", 64);
        let weights = sym_shape!(1, 4 * hidden, 64);
        let recurrent = sym_shape!(1, 4 * hidden, hidden);
        let op = LSTM {
            direction: Direction::Unidirectional,
        };
        let result = op.infer_shapes([input, weights, recurrent].into(), &mut sym_gen);
        assert_eq!(result, Err(InferShapesError::IncorrectRank));
    }
}
