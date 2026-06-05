use crate::infer_shapes::{InferShapes, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// STFT operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__STFT.html>.
pub struct STFT {
    pub onesided: bool,
}

impl InferShapes for STFT {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [signal, frame_step, rest @ ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };
        let window = rest.first();
        let frame_length = rest.get(1);

        // At least one of `window` or `frame_length` must be set so `n_fft`
        // can be determined.
        if window.is_none() && frame_length.is_none() {
            return Err(InferShapesError::InvalidValue);
        }

        let (Some(batch), Some(signal_len)) = (signal.size(0), signal.size(1)) else {
            return Ok([SymTensor::unknown("unknown signal shape")].into());
        };

        // `n_fft` comes from `frame_length` if it's a known scalar, otherwise
        // from the size of the `window` input.
        //
        // If both `frame_length` and `window` are set, they must be equal. We
        // don't enforce that here, as the infrastructure for handling symbolic
        // equality doesn't exist yet.
        let n_fft = if let Some(fl) = frame_length
            && let Some(val) = fl.as_scalar()
        {
            val.clone()
        } else if let Some(window) = window
            && let Some(len) = window.size(0)
        {
            len
        } else {
            sym_gen.gen_positive()
        };

        let frame_step = frame_step
            .as_scalar()
            .cloned()
            .unwrap_or_else(|| sym_gen.gen_positive());

        // The ONNX STFT documentation does not specify an expression for
        // `n_frames`. See the `torch.stft` docs instead:
        // https://docs.pytorch.org/docs/stable/generated/torch.stft.html.
        let n_frames = (signal_len - n_fft.clone()) / frame_step + SymExpr::Value(1);

        let dft_unique_bins = if self.onesided {
            n_fft / SymExpr::Value(2) + SymExpr::Value(1)
        } else {
            n_fft
        };

        let out_shape = vec![batch, n_frames, dft_unique_bins, SymExpr::Value(2)];
        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_scalar, sym_shape, sym_vec};

    use super::STFT;

    fn infer_stft(
        onesided: bool,
        signal: SymTensor,
        frame_step: SymTensor,
        window: Option<SymTensor>,
        frame_length: Option<SymTensor>,
    ) -> SymTensor {
        let mut inputs = vec![signal, frame_step];
        // `window` (input 2) is optional but positional, so a placeholder is
        // needed when `frame_length` (input 3) is present without it.
        match (window, &frame_length) {
            (Some(window), _) => inputs.push(window),
            (None, Some(_)) => inputs.push(SymTensor::unknown("no window")),
            (None, None) => {}
        }
        if let Some(frame_length) = frame_length {
            inputs.push(frame_length);
        }

        let mut sym_gen = SymbolGen::new();
        let mut result = STFT { onesided }
            .infer_shapes(&inputs, &mut sym_gen)
            .unwrap();
        result.remove(0).simplify()
    }

    #[test]
    fn test_stft() {
        #[derive(Debug)]
        struct Case {
            onesided: bool,
            signal: SymTensor,
            frame_step: SymTensor,
            window: Option<SymTensor>,
            frame_length: Option<SymTensor>,
            expected: SymTensor,
        }

        let cases = [
            // onesided=true
            Case {
                onesided: true,
                signal: sym_shape!("batch", 8),
                frame_step: sym_scalar!(4),
                window: None,
                frame_length: Some(sym_scalar!(4)),
                expected: sym_shape!("batch", 2, 3, 2),
            },
            // onesided=false
            Case {
                onesided: false,
                signal: sym_shape!(1, 8),
                frame_step: sym_scalar!(4),
                window: None,
                frame_length: Some(sym_scalar!(4)),
                expected: sym_shape!(1, 2, 4, 2),
            },
            // `window` input set. n_fft is taken from window length.
            Case {
                onesided: true,
                signal: sym_shape!("batch", 16),
                frame_step: sym_scalar!(4),
                window: Some(sym_vec!(0, 0, 0, 0)),
                frame_length: None,
                expected: sym_shape!("batch", 4, 3, 2),
            },
            // Complex signal
            Case {
                onesided: false,
                signal: sym_shape!("batch", 8, 2),
                frame_step: sym_scalar!(4),
                window: None,
                frame_length: Some(sym_scalar!(4)),
                expected: sym_shape!("batch", 2, 4, 2),
            },
        ];

        cases.test_each(|case| {
            let out = infer_stft(
                case.onesided,
                case.signal.clone(),
                case.frame_step.clone(),
                case.window.clone(),
                case.frame_length.clone(),
            );
            assert_eq!(out, case.expected);
        });
    }

    #[test]
    fn test_stft_unknown_signal() {
        let out = infer_stft(
            true,
            SymTensor::unknown("unknown"),
            sym_scalar!(4),
            None,
            Some(sym_scalar!(4)),
        );
        assert_eq!(out.ndim(), None);
    }

    #[test]
    fn test_stft_symbolic_frame_step_and_length() {
        let out = infer_stft(
            true,
            sym_shape!("batch", "sig"),
            sym_shape!(),
            None,
            Some(sym_shape!()),
        );
        let shape: Vec<_> = out.shape().unwrap().collect();
        assert_eq!(shape.len(), 4);

        let n_fft = SymExpr::pos_var("unknown_1");
        let frame_step = SymExpr::pos_var("unknown_2");
        let signal_len = SymExpr::from("sig");

        assert_eq!(shape[0], SymExpr::from("batch"));
        assert_eq!(
            shape[1],
            ((signal_len - n_fft.clone()) / frame_step + SymExpr::Value(1)).simplify()
        );
        assert_eq!(
            shape[2],
            (n_fft / SymExpr::Value(2) + SymExpr::Value(1)).simplify()
        );
        assert_eq!(shape[3], SymExpr::Value(2));
    }

    #[test]
    fn test_stft_missing_window_and_frame_length() {
        let mut sym_gen = SymbolGen::new();
        let result = STFT { onesided: true }
            .infer_shapes(&[sym_shape!("batch", 8), sym_scalar!(4)], &mut sym_gen);
        assert_eq!(result, Err(InferShapesError::InvalidValue));
    }
}
