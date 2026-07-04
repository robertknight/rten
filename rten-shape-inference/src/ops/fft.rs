use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError, resolve_axis};
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
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let signal = inputs.require(0)?;
        let frame_step = inputs.require(1)?;
        let window = inputs.get(2);
        let frame_length = inputs.get(3);

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

/// DFT operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__DFT.html>.
pub struct DFT {
    pub inverse: bool,
    pub onesided: bool,
}

impl InferShapes for DFT {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let input = inputs.require(0)?;
        let dft_length = inputs.get(1);

        let irfft = self.inverse && self.onesided;

        // `axis` is an optional input (input 2). It must be a known constant to
        // determine the output shape; otherwise the shape is unknown. The
        // default of -2 matches the `axis` input added in opset 20.
        let axis = match inputs.get(2) {
            None => -2,
            Some(axis) => match axis.as_scalar() {
                Some(&SymExpr::Value(val)) => val,
                _ => return Ok([SymTensor::unknown("DFT axis is not constant")].into()),
            },
        };

        let Some(dims) = input.shape() else {
            return Ok([SymTensor::unknown("unknown DFT input shape")].into());
        };
        let mut shape: Vec<SymExpr> = dims.collect();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(InferShapesError::InvalidValue);
        }
        let complex_dim = ndim - 1;

        // Resolve the signal axis. The final (complex) dimension is excluded.
        let dft_axis = resolve_axis(ndim, axis).map_err(|_| InferShapesError::InvalidValue)?;
        if dft_axis == complex_dim {
            return Err(InferShapesError::InvalidValue);
        }

        let signal_len = shape[dft_axis].clone();

        // `n_fft` (the real signal length) comes from the `dft_length` input
        // when it is present, so its value must be known. If `dft_length` is
        // missing, the signal length is used. For IRFFT the input is the half
        // spectrum of length `floor(n_fft / 2) + 1`, so the default is
        // `2 * (signal_len - 1)`.
        let n_fft = if let Some(dl) = dft_length {
            match dl.as_scalar() {
                Some(val) => val.clone(),
                None => sym_gen.gen_positive(),
            }
        } else if irfft {
            (signal_len - SymExpr::Value(1)) * SymExpr::Value(2)
        } else {
            signal_len
        };

        // IRFFT returns the full real signal (component dim of size 1); a
        // forward one-sided transform returns the unique half of the spectrum.
        let (out_len, n_components) = if irfft {
            (n_fft, SymExpr::Value(1))
        } else if self.onesided {
            (
                n_fft / SymExpr::Value(2) + SymExpr::Value(1),
                SymExpr::Value(2),
            )
        } else {
            (n_fft, SymExpr::Value(2))
        };

        shape[dft_axis] = out_len;
        shape[complex_dim] = n_components;

        Ok([SymTensor::from_shape(shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_scalar, sym_shape, sym_vec};

    use super::{DFT, STFT};

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
            .infer_shapes(inputs.into(), &mut sym_gen)
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
        let result = STFT { onesided: true }.infer_shapes(
            [sym_shape!("batch", 8), sym_scalar!(4)].into(),
            &mut sym_gen,
        );
        assert_eq!(result, Err(InferShapesError::InvalidValue));
    }

    fn infer_dft(
        axis: i32,
        inverse: bool,
        onesided: bool,
        input: SymTensor,
        dft_length: Option<SymTensor>,
    ) -> SymTensor {
        let inputs = vec![
            Some(input),
            dft_length,
            Some(SymTensor::from_scalar(SymExpr::Value(axis))),
        ];
        let mut sym_gen = SymbolGen::new();
        let mut result = DFT { inverse, onesided }
            .infer_shapes(inputs.into(), &mut sym_gen)
            .unwrap();
        result.remove(0).simplify()
    }

    #[test]
    fn test_dft() {
        #[derive(Debug)]
        struct Case {
            axis: i32,
            inverse: bool,
            onesided: bool,
            input: SymTensor,
            dft_length: Option<SymTensor>,
            expected: SymTensor,
        }

        let cases = [
            // Two-sided forward transform.
            Case {
                axis: -2,
                inverse: false,
                onesided: false,
                input: sym_shape!("batch", 8, 1),
                dft_length: None,
                expected: sym_shape!("batch", 8, 2),
            },
            // One-sided forward transform.
            Case {
                axis: -2,
                inverse: false,
                onesided: true,
                input: sym_shape!("batch", 8, 1),
                dft_length: None,
                expected: sym_shape!("batch", 5, 2),
            },
            // `dft_length` overrides the signal length.
            Case {
                axis: -2,
                inverse: false,
                onesided: true,
                input: sym_shape!("batch", 8, 1),
                dft_length: Some(sym_scalar!(16)),
                expected: sym_shape!("batch", 9, 2),
            },
            // Inverse transform has the same shape as the input.
            Case {
                axis: -2,
                inverse: true,
                onesided: false,
                input: sym_shape!("batch", 8, 2),
                dft_length: None,
                expected: sym_shape!("batch", 8, 2),
            },
            // Inverse one-sided transform (IRFFT). The half spectrum of length
            // 5 reconstructs a real signal of length `2 * (5 - 1) = 8`.
            Case {
                axis: -2,
                inverse: true,
                onesided: true,
                input: sym_shape!("batch", 5, 2),
                dft_length: None,
                expected: sym_shape!("batch", 8, 1),
            },
            // IRFFT with an explicit `dft_length`.
            Case {
                axis: -2,
                inverse: true,
                onesided: true,
                input: sym_shape!("batch", 5, 2),
                dft_length: Some(sym_scalar!(7)),
                expected: sym_shape!("batch", 7, 1),
            },
        ];

        cases.test_each(|case| {
            let out = infer_dft(
                case.axis,
                case.inverse,
                case.onesided,
                case.input.clone(),
                case.dft_length.clone(),
            );
            assert_eq!(out, case.expected);
        });
    }

    #[test]
    fn test_dft_unknown_dft_length() {
        // If `dft_length` is present but its value is unknown, the size of
        // the transformed axis is unknown.
        let out = infer_dft(
            -2,
            false,
            false,
            sym_shape!("batch", 8, 1),
            Some(SymTensor::unknown("runtime-computed dft_length")),
        );
        let shape: Vec<_> = out.shape().unwrap().collect();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], SymExpr::from("batch"));
        assert!(matches!(shape[1], SymExpr::Var(_)));
        assert_eq!(shape[2], SymExpr::Value(2));
    }
}
