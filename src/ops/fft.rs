use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, TensorView};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

use crate::buffer_pool::BufferPool;
use crate::operator::{IntoOpResult, OpError, OpRunContext, Operator, OutputList};

enum FftType {
    /// FFT with real input signal.
    Real,
    /// FFT with complex input signal.
    Complex,
}

pub fn stft(
    pool: &BufferPool,
    signal: TensorView<f32>,
    frame_step: i32,
    window: Option<NdTensorView<f32, 1>>,
    frame_length: Option<i32>,
    onesided: bool,
) -> Result<NdTensor<f32, 4>, OpError> {
    let signal: NdTensorView<f32, 3> = match signal.ndim() {
        2 => {
            // Spec says signal must be 3D, but ORT accepts 2D for real inputs.
            // See https://github.com/onnx/onnx/issues/7277.
            let mut signal = signal;
            signal.insert_axis(2);
            signal.nd_view()
        }
        3 => signal.nd_view(),
        _ => {
            return Err(OpError::InvalidValue("signal must have 2 or 3 dims"));
        }
    };

    let [batch, signal_len, n_components] = signal.shape();

    let frame_length = frame_length
        .map(|fl| {
            if fl >= 1 && fl as usize <= signal_len {
                Ok(fl as usize)
            } else {
                Err(OpError::InvalidValue(
                    "frame_length must be in range [1, signal_length]",
                ))
            }
        })
        .transpose()?;

    let frame_step = if frame_step > 0 {
        frame_step as usize
    } else {
        return Err(OpError::InvalidValue("frame_step must be > 0"));
    };

    let Some(n_fft) = frame_length.or_else(|| window.map(|w| w.size(0))) else {
        return Err(OpError::InvalidValue(
            "Either frame_length or window must be set",
        ));
    };

    let fft_type = match n_components {
        1 => FftType::Real,
        2 => FftType::Complex,
        _ => {
            return Err(OpError::InvalidValue(
                "Last dimension of signal must have size 1 or 2",
            ));
        }
    };

    if matches!(fft_type, FftType::Complex) && onesided {
        return Err(OpError::InvalidValue(
            "FFT cannot be one-sided if input is complex",
        ));
    }

    // The ONNX STFT documentation does not specify an expression for
    // `n_frames`. See the `torch.stft` docs instead:
    // https://docs.pytorch.org/docs/stable/generated/torch.stft.html.
    let n_frames = (signal_len - n_fft) / frame_step + 1;
    let dft_unique_bins = if onesided { n_fft / 2 + 1 } else { n_fft };
    let mut output = NdTensor::zeros_in(pool, [batch, n_frames, dft_unique_bins, 2]);

    // Temporary buffer for FFT input and output.
    let mut tmp_buf = Vec::new();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    for (signal_batch, mut out_batch) in signal.axis_iter(0).zip(output.axis_iter_mut(0)) {
        for (frame, mut out_frame) in out_batch.axis_iter_mut(0).enumerate() {
            tmp_buf.clear();
            tmp_buf.extend((0..n_fft).map(|k| {
                let offset = frame * frame_step + k;
                let weight = window.as_ref().map(|win| win[[k]]).unwrap_or(1.);
                let re = weight * signal_batch[[offset, 0]];
                let im = match fft_type {
                    FftType::Real => 0.,
                    FftType::Complex => weight * signal_batch[[offset, 1]],
                };
                Complex32 { re, im }
            }));

            fft.process(&mut tmp_buf);

            for (bin, val) in tmp_buf.iter().take(out_frame.size(0)).enumerate() {
                out_frame[[bin, 0]] = val.re;
                out_frame[[bin, 1]] = val.im;
            }
        }
    }

    Ok(output)
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct STFT {
    pub onesided: bool,
}

impl Operator for STFT {
    fn name(&self) -> &str {
        "STFT"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let signal = ctx.inputs().require_as(0)?;
        let frame_step = ctx.inputs().require_as(1)?;

        // This is not documented, but least one of the `window` or
        // `frame_length` inputs must be set. See
        // https://github.com/onnx/onnx/issues/4464.
        let window = ctx.inputs().get_as(2)?;
        let frame_length = ctx.inputs().get_as(3)?;

        stft(
            ctx.pool(),
            signal,
            frame_step,
            window,
            frame_length,
            self.onesided,
        )
        .into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;

    use super::STFT;
    use crate::operator::{InputList, OpError, OperatorExt};
    use crate::ops::tests::{IntoNDim, expect_eq_1e4};

    #[test]
    fn test_stft() {
        // Reference results are computed using PyTorch with:
        //
        // ```
        // y = torch.stft(signal, return_complex=True, center=False)
        // y = y.permute((1, 0)) # (freq, frame) => (frame, freq)
        // torch.view_as_real(y) # (frame, freq) => (frame, freq, component)
        // ```

        let real_signal = Tensor::from([[
            0.9753, 0.6473, 0.8170, 0.6239, 0.8245, 0.8314, 0.4374, 0.8641,
        ]]);

        // Generated with `torch.rand([8], dtype=torch.complex64)`
        let complex_signal = Tensor::from([[
            [0.0161, 0.5205],
            [0.0405, 0.2422],
            [0.7515, 0.5957],
            [0.4097, 0.0244],
            [0.9038, 0.2969],
            [0.3779, 0.5801],
            [0.0117, 0.8315],
            [0.5308, 0.0854],
        ]]);

        #[derive(Debug)]
        struct Case {
            signal: Tensor, // (batch, signal) or (batch, signal, component)
            frame_step: i32,
            window: Option<NdTensor<f32, 1>>,
            frame_length: Option<i32>,
            onesided: bool,
            expected: Result<NdTensor<f32, 4>, OpError>,
        }

        impl Default for Case {
            fn default() -> Case {
                Case {
                    signal: Tensor::from(0.),
                    frame_step: 0,
                    window: None,
                    frame_length: None,
                    onesided: true,
                    expected: Err(OpError::InvalidValue("Invalid expectation")),
                }
            }
        }

        let cases = [
            // One-sided real-to-complex (2D signal)
            Case {
                signal: real_signal.clone(),
                frame_step: 4,
                frame_length: Some(4),
                expected: Ok(NdTensor::from([
                    [[3.0635, 0.0000], [0.1583, -0.0234], [0.5211, 0.0000]],
                    [[2.9574, 0.0000], [0.3871, 0.0327], [-0.4336, 0.0000]],
                ])
                .into_ndim()),
                ..Default::default()
            },
            // One-sided real-to-complex (3D signal)
            Case {
                signal: real_signal
                    .clone()
                    .into_shape([real_signal.size(0), real_signal.size(1), 1])
                    .into_dyn(),
                frame_step: 4,
                frame_length: Some(4),
                expected: Ok(NdTensor::from([
                    [[3.0635, 0.0000], [0.1583, -0.0234], [0.5211, 0.0000]],
                    [[2.9574, 0.0000], [0.3871, 0.0327], [-0.4336, 0.0000]],
                ])
                .into_ndim()),
                ..Default::default()
            },
            // One-sided real-to-complex with a window
            Case {
                signal: real_signal.clone(),
                frame_step: 4,
                frame_length: Some(4),
                // torch.hann_window(4)
                window: Some([0., 0.5, 1.0, 0.5].into()),
                expected: Ok(NdTensor::from([
                    [[1.4526, 0.0000], [-0.8170, -0.0117], [0.1814, 0.0000]],
                    [[1.2851, 0.0000], [-0.4374, 0.0164], [-0.4103, 0.0000]],
                ])
                .into_ndim()),
                ..Default::default()
            },
            // Two-sided real-to-complex
            Case {
                signal: real_signal.clone(),
                frame_step: 4,
                frame_length: Some(4),
                onesided: false,
                expected: Ok(NdTensor::from([
                    [
                        [3.0635, 0.0000],
                        [0.1583, -0.0234],
                        [0.5211, 0.0000],
                        [0.1583, 0.0234],
                    ],
                    [
                        [2.9574, 0.0000],
                        [0.3871, 0.0327],
                        [-0.4336, 0.0000],
                        [0.3871, -0.0327],
                    ],
                ])
                .into_ndim()),
                ..Default::default()
            },
            // Complex-to-complex
            Case {
                signal: complex_signal.clone(),
                frame_step: 4,
                frame_length: Some(4),
                onesided: false,
                expected: Ok(NdTensor::from([
                    [
                        [1.2178, 1.3828],
                        [-0.5176, 0.2939],
                        [0.3174, 0.8496],
                        [-0.9532, -0.4444],
                    ],
                    [
                        [1.8242, 1.7939],
                        [1.3868, -0.3818],
                        [0.0068, 0.4629],
                        [0.3975, -0.6875],
                    ],
                ])
                .into_ndim()),
                ..Default::default()
            },
            // Invalid frame_length
            Case {
                signal: real_signal.clone(),
                frame_step: 4,
                frame_length: Some(0),
                expected: Err(OpError::InvalidValue(
                    "frame_length must be in range [1, signal_length]",
                )),
                ..Default::default()
            },
            Case {
                signal: real_signal.clone(),
                frame_step: 4,
                frame_length: Some(real_signal.size(1) as i32 + 1),
                expected: Err(OpError::InvalidValue(
                    "frame_length must be in range [1, signal_length]",
                )),
                ..Default::default()
            },
            // Invalid frame_step
            Case {
                signal: real_signal.clone(),
                frame_step: 0,
                frame_length: Some(4),
                expected: Err(OpError::InvalidValue("frame_step must be > 0")),
                ..Default::default()
            },
            // Missing window and frame_length
            Case {
                signal: real_signal.clone(),
                frame_step: 4,
                expected: Err(OpError::InvalidValue(
                    "Either frame_length or window must be set",
                )),
                ..Default::default()
            },
            // One-sided output with complex input
            Case {
                signal: complex_signal.clone(),
                frame_step: 4,
                frame_length: Some(4),
                onesided: true,
                expected: Err(OpError::InvalidValue(
                    "FFT cannot be one-sided if input is complex",
                )),
                ..Default::default()
            },
        ];

        cases.test_each(|case| {
            let frame_step = NdTensor::from(case.frame_step);
            let frame_length = case.frame_length.map(NdTensor::from);

            let inputs = InputList::from_iter([
                Some(case.signal.view().into()),
                Some(frame_step.view().into()),
                case.window.as_ref().map(|w| w.view().into()),
                frame_length.as_ref().map(|fl| fl.view().into()),
            ]);
            let result: Result<NdTensor<f32, 4>, _> = STFT {
                onesided: case.onesided,
            }
            .run_simple(inputs);

            match (&result, &case.expected) {
                (Ok(result), Ok(expected)) => {
                    expect_eq_1e4(result, expected).unwrap();
                }
                _ => assert_eq!(result, case.expected),
            }
        });
    }
}
