use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::ops::resolve_axis;

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

    // If both `frame_length` and `window` are set, their sizes must match.
    if let (Some(frame_length), Some(window)) = (frame_length, window)
        && frame_length != window.size(0)
    {
        return Err(OpError::InvalidValue(
            "window length must equal frame_length",
        ));
    }

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
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];

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

            fft.process_with_scratch(&mut tmp_buf, &mut scratch);

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

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    STFT,
    op,
    shape_ops::STFT {
        onesided: op.onesided,
    }
);

/// Compute the Discrete Fourier Transform of `input` along `axis`.
///
/// `input` has shape `[...][n_signal][n_components]` where the DFT is performed
/// along the dimension given by `axis` and the final dimension contains either
/// the real values (size 1) or interleaved real and imaginary values (size 2)
/// of a complex signal.
///
/// With `onesided` set, a forward transform (RFFT) returns the non-redundant
/// half of the conjugate-symmetric spectrum, and an inverse transform (IRFFT)
/// reconstructs the full real signal from such a half spectrum.
///
/// See <https://onnx.ai/onnx/operators/onnx__DFT.html>.
pub fn dft(
    pool: &BufferPool,
    input: TensorView<f32>,
    dft_length: Option<i32>,
    axis: isize,
    inverse: bool,
    onesided: bool,
) -> Result<Tensor<f32>, OpError> {
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(OpError::InvalidValue("DFT input must have at least 2 dims"));
    }

    let complex_dim = ndim - 1;
    let n_components = input.size(complex_dim);
    if n_components != 1 && n_components != 2 {
        return Err(OpError::InvalidValue(
            "Last dimension of DFT input must have size 1 or 2",
        ));
    }

    // A one-sided inverse transform (IRFFT) reconstructs a real signal from a
    // conjugate-symmetric half spectrum, so it requires complex input. A
    // one-sided forward transform (RFFT) produces such a spectrum from a real
    // signal, so it requires real input.
    let irfft = inverse && onesided;
    if irfft && n_components != 2 {
        return Err(OpError::InvalidValue(
            "Inverse one-sided DFT (IRFFT) requires complex input",
        ));
    }
    if onesided && !inverse && n_components == 2 {
        return Err(OpError::InvalidValue(
            "DFT cannot be one-sided if input is complex",
        ));
    }

    // Resolve the signal axis. Per the spec the accepted range is
    // `[-r, -2] ∪ [0, r-2]`, ie. the final (complex) dimension is excluded.
    let dft_axis = resolve_axis(ndim, axis)?;
    if dft_axis == complex_dim {
        return Err(OpError::InvalidValue(
            "DFT axis cannot be the complex dimension",
        ));
    }

    let signal_len = input.size(dft_axis);

    // The length (`N`) of the DFT, ie. the length of the real signal. The input
    // signal is zero-padded or truncated to this length along the DFT axis.
    //
    // For IRFFT the input is the half spectrum of length `floor(N/2) + 1`, so
    // the default `N` is `2 * (signal_len - 1)`. Matches NumPy's `irfft`.
    let n_fft = match dft_length {
        Some(len) if len >= 1 => len as usize,
        Some(_) => return Err(OpError::InvalidValue("dft_length must be >= 1")),
        None if irfft => 2 * signal_len.saturating_sub(1),
        None => signal_len,
    };

    // Number of output values along the DFT axis. RFFT returns only the unique
    // half of the conjugate-symmetric spectrum.
    //
    // If the signal is empty (`n_fft == 0`), the one-sided output has a single
    // frequency bin filled with zeros. This matches ONNX Runtime, although
    // NumPy raises an error.
    let out_len = if onesided && !inverse {
        n_fft / 2 + 1
    } else {
        n_fft
    };

    // IRFFT returns a real signal, all other cases return complex.
    let out_components = if irfft { 1 } else { 2 };

    // Permute the input so the DFT axis and complex dimension are the last two,
    // leaving lanes of shape `[n_fft][n_components]` that are contiguous in
    // memory.
    let perm: Vec<usize> = (0..ndim)
        .filter(|&d| d != dft_axis && d != complex_dim)
        .chain([dft_axis, complex_dim])
        .collect();
    let input = input.permuted(perm.as_slice());
    let input = input.to_contiguous_in(pool).auto_return(pool);
    let in_data = input.data();
    let num_lanes: usize = input.shape()[..ndim - 2].iter().product();

    let mut out_shape: Vec<usize> = input.shape().to_vec();
    out_shape[ndim - 2] = out_len;
    out_shape[ndim - 1] = out_components;
    let mut output = Tensor::zeros_in(pool, out_shape.as_slice());
    let out_data = output.data_mut().unwrap();

    let mut planner = FftPlanner::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n_fft)
    } else {
        planner.plan_fft_forward(n_fft)
    };
    // ONNX defines the inverse transform with a `1/N` normalization factor,
    // which `rustfft` does not apply.
    let scale = if inverse { 1.0 / n_fft as f32 } else { 1.0 };

    let mut buf: Vec<Complex32> = Vec::with_capacity(n_fft);
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    let get = |base: usize, i: usize| {
        let re = in_data[base + i * n_components];
        let im = if n_components == 2 {
            in_data[base + i * n_components + 1]
        } else {
            0.
        };
        Complex32 { re, im }
    };

    for lane in 0..num_lanes {
        let in_base = lane * signal_len * n_components;

        buf.clear();
        let n_in = signal_len.min(n_fft);
        if irfft {
            // Reconstruct the full conjugate-symmetric spectrum from the half
            // spectrum: `X[N-k] = conj(X[k])`. For even `N` the Nyquist bin
            // (`k == N/2`) is its own mirror and is left as the input value.
            buf.resize(n_fft, Complex32::default());
            for k in 0..n_in {
                buf[k] = get(in_base, k);
            }
            let conj_end = if n_fft % 2 == 0 {
                n_in.saturating_sub(1)
            } else {
                n_in
            };
            for k in 1..conj_end {
                buf[n_fft - k] = get(in_base, k).conj();
            }
        } else {
            // Zero-pad or truncate the signal to `n_fft` values.
            buf.extend((0..n_in).map(|k| get(in_base, k)));
            buf.resize(n_fft, Complex32::default());
        }

        fft.process_with_scratch(&mut buf, &mut scratch);

        let out_base = lane * out_len * out_components;
        if irfft {
            // IRFFT discards the imaginary part, which is zero up to rounding.
            for (i, val) in buf.iter().take(out_len).enumerate() {
                out_data[out_base + i] = val.re * scale;
            }
        } else {
            for (i, val) in buf.iter().take(out_len).enumerate() {
                let val = val * scale;
                out_data[out_base + i * 2] = val.re;
                out_data[out_base + i * 2 + 1] = val.im;
            }
        }
    }

    // If the DFT axis was already the second-from-last dimension, `perm` is
    // the identity and the output needs no further reordering.
    if dft_axis == ndim - 2 {
        return Ok(output);
    }

    // Restore the original axis order.
    let output = output.auto_return(pool);
    let mut inv_perm = vec![0usize; ndim];
    for (new_pos, &old_dim) in perm.iter().enumerate() {
        inv_perm[old_dim] = new_pos;
    }
    Ok(output.permuted(inv_perm.as_slice()).to_tensor_in(pool))
}

/// Default signal axis for the [`DFT`] operator.
///
/// This is the second-from-last axis, ie. the last axis excluding the complex
/// component dimension. It matches the default of the `axis` input added in
/// opset 20.
const DFT_DEFAULT_AXIS: i32 = -2;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct DFT {
    /// Whether to perform the inverse transform.
    pub inverse: bool,
    /// Whether to return only the non-redundant half of a conjugate-symmetric
    /// spectrum.
    pub onesided: bool,
}

impl Operator for DFT {
    fn name(&self) -> &str {
        "DFT"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as(0)?;
        let dft_length = ctx.inputs().get_as(1)?;
        let axis = ctx.inputs().get_as::<i32>(2)?.unwrap_or(DFT_DEFAULT_AXIS) as isize;

        dft(
            ctx.pool(),
            input,
            dft_length,
            axis,
            self.inverse,
            self.onesided,
        )
        .into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    DFT,
    op,
    shape_ops::DFT {
        inverse: op.inverse,
        onesided: op.onesided,
    }
);

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;

    use super::{DFT, STFT};
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
            // Conflicting frame_length and window sizes (window shorter)
            Case {
                signal: real_signal.clone(),
                frame_step: 4,
                frame_length: Some(4),
                window: Some([0., 0.5].into()),
                expected: Err(OpError::InvalidValue(
                    "window length must equal frame_length",
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

    #[test]
    fn test_dft() {
        // Reference results are computed with NumPy's `np.fft` module.
        #[derive(Debug)]
        struct Case {
            // Input of shape `[batch, signal, component]` (or higher rank).
            input: Tensor,
            dft_length: Option<i32>,
            axis: i32,
            inverse: bool,
            onesided: bool,
            expected: Result<Tensor, OpError>,
        }

        impl Default for Case {
            fn default() -> Case {
                Case {
                    input: Tensor::from(0.),
                    dft_length: None,
                    axis: -2,
                    inverse: false,
                    onesided: false,
                    expected: Err(OpError::InvalidValue("Invalid expectation")),
                }
            }
        }

        // Real signal `[1, 2, 3, 4]` with shape `[1, 4, 1]`.
        let real_signal = Tensor::from([[[1.], [2.], [3.], [4.]]]).into_dyn();

        // Complex signal `[1, 2+1j, 3-1j, 4+2j]` with shape `[1, 4, 2]`.
        let complex_signal = Tensor::from([[[1., 0.], [2., 1.], [3., -1.], [4., 2.]]]).into_dyn();

        let cases = [
            // One-sided real-to-complex (`np.fft.rfft`).
            Case {
                input: real_signal.clone(),
                onesided: true,
                expected: Ok(Tensor::from([[[10., 0.], [-2., 2.], [-2., 0.]]]).into_dyn()),
                ..Default::default()
            },
            // Two-sided real-to-complex (`np.fft.fft`).
            Case {
                input: real_signal.clone(),
                expected: Ok(
                    Tensor::from([[[10., 0.], [-2., 2.], [-2., 0.], [-2., -2.]]]).into_dyn(),
                ),
                ..Default::default()
            },
            // Complex-to-complex (`np.fft.fft`).
            Case {
                input: complex_signal.clone(),
                expected: Ok(
                    Tensor::from([[[10., 2.], [-3., 3.], [-2., -4.], [-1., -1.]]]).into_dyn(),
                ),
                ..Default::default()
            },
            // Inverse complex-to-complex (`np.fft.ifft`).
            Case {
                input: complex_signal.clone(),
                inverse: true,
                expected: Ok(Tensor::from([[
                    [2.5, 0.5],
                    [-0.25, -0.25],
                    [-0.5, -1.],
                    [-0.75, 0.75],
                ]])
                .into_dyn()),
                ..Default::default()
            },
            // Inverse one-sided real-to-complex (`np.fft.irfft`). The half
            // spectrum is `rfft([1, 2, 3, 4])` and the default `dft_length`
            // reconstructs the original length-4 signal.
            Case {
                input: Tensor::from([[[10., 0.], [-2., 2.], [-2., 0.]]]).into_dyn(),
                inverse: true,
                onesided: true,
                expected: Ok(Tensor::from([[[1.], [2.], [3.], [4.]]]).into_dyn()),
                ..Default::default()
            },
            // Inverse one-sided with an explicit odd `dft_length`
            // (`np.fft.irfft(rfft([1, 2, 3]), n=3)`).
            Case {
                input: Tensor::from([[[6., 0.], [-1.5, 0.8660254]]]).into_dyn(),
                dft_length: Some(3),
                inverse: true,
                onesided: true,
                expected: Ok(Tensor::from([[[1.], [2.], [3.]]]).into_dyn()),
                ..Default::default()
            },
            // IRFFT requires complex input.
            Case {
                input: real_signal.clone(),
                inverse: true,
                onesided: true,
                expected: Err(OpError::InvalidValue(
                    "Inverse one-sided DFT (IRFFT) requires complex input",
                )),
                ..Default::default()
            },
            // One-sided transform of a complex input is disallowed, as the
            // spectrum is not conjugate-symmetric.
            Case {
                input: complex_signal.clone(),
                onesided: true,
                expected: Err(OpError::InvalidValue(
                    "DFT cannot be one-sided if input is complex",
                )),
                ..Default::default()
            },
            // Zero-padding via `dft_length` (`np.fft.rfft(x, n=6)`).
            Case {
                input: real_signal.clone(),
                dft_length: Some(6),
                onesided: true,
                expected: Ok(Tensor::from([[
                    [10., 0.],
                    [-3.5, -4.330127],
                    [2.5, 0.866025],
                    [-2., 0.],
                ]])
                .into_dyn()),
                ..Default::default()
            },
            // Truncation via `dft_length` (`np.fft.rfft(x, n=2)`).
            Case {
                input: real_signal.clone(),
                dft_length: Some(2),
                onesided: true,
                expected: Ok(Tensor::from([[[3., 0.], [-1., 0.]]]).into_dyn()),
                ..Default::default()
            },
            // Transform along a non-default axis of a higher-rank input. The
            // size-2 dimension is a "feature" axis with the DFT performed along
            // the size-4 axis. This exercises the axis permutation.
            //
            // Feature 0 is `rfft([1, 2, 3, 4])` and feature 1 is
            // `rfft([4, 3, 2, 1])`.
            Case {
                // Shape `[1, 4, 2, 1]`: `[batch, signal, feature, component]`.
                input: Tensor::from_data(&[1, 4, 2, 1], vec![1., 4., 2., 3., 3., 2., 4., 1.]),
                axis: 1,
                onesided: true,
                // Shape `[1, 3, 2, 2]`: `[batch, freq, feature, component]`.
                expected: Ok(Tensor::from_data(
                    &[1, 3, 2, 2],
                    vec![10., 0., 10., 0., -2., 2., 2., -2., -2., 0., 2., 0.],
                )),
                ..Default::default()
            },
            // Invalid component count.
            Case {
                input: Tensor::from([[[1., 2., 3.]]]).into_dyn(),
                expected: Err(OpError::InvalidValue(
                    "Last dimension of DFT input must have size 1 or 2",
                )),
                ..Default::default()
            },
            // Invalid `dft_length`.
            Case {
                input: real_signal.clone(),
                dft_length: Some(0),
                expected: Err(OpError::InvalidValue("dft_length must be >= 1")),
                ..Default::default()
            },
        ];

        cases.test_each(|case| {
            // `axis` is passed as an input (input 2). `dft_length` (input 1) is
            // optional, so a `None` placeholder is used when it is unset.
            let dft_length = case.dft_length.map(Tensor::from);
            let axis = Tensor::from(case.axis);
            let inputs = InputList::from_iter([
                Some(case.input.view().into()),
                dft_length.as_ref().map(|d| d.view().into()),
                Some(axis.view().into()),
            ]);
            let result: Result<Tensor, _> = DFT {
                inverse: case.inverse,
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
