use std::iter::zip;

use crate::ops::{get_input, get_optional_input, Input, IntoOpResult, OpError, Operator, Output};
use crate::tensor::Tensor;

/// Specifies an output size for a resize operation.
pub enum ResizeTarget<'a> {
    /// Vector of scale factors for each dimension. The length should match the
    /// input rank.
    Scales(&'a Tensor),

    /// Vector of output sizes for each dimension. The length should match the
    /// input rank.
    Sizes(&'a Tensor<i32>),
}

struct Image<'a> {
    data: &'a [f32],
    height: usize,
    width: usize,
    h_stride: usize,
    w_stride: usize,
}

struct ImageMut<'a> {
    data: &'a mut [f32],
    height: usize,
    width: usize,
    h_stride: usize,
    w_stride: usize,
}

/// Compute the input image coordinate that corresponds to an output coordinate,
/// where `scale` is the scale factor from output to input.
///
/// ONNX supports several modes for transforming coords, specified by the
/// `coordinate_transformation_mode` attribute. The default, implemented here,
/// is the "half pixel" mode. The half pixel mode is consistent with how OpenCV
/// (`cv2.resize`) and PyTorch (`torch.nn.functional.interpolate`) work. See
/// https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
/// for rationale.
fn input_coord(dest_coord: usize, scale: f32) -> f32 {
    scale * (dest_coord as f32 + 0.5) - 0.5
}

fn nearest_resize(input: &Image, output: &mut ImageMut) {
    // Scale factors to map output coords to input coords.
    let inv_scale_y = input.height as f32 / output.height as f32;
    let inv_scale_x = input.width as f32 / output.width as f32;

    for y in 0..output.height {
        let in_y = (y as f32 * inv_scale_y) as usize;
        for x in 0..output.width {
            let in_x = (x as f32 * inv_scale_x) as usize;
            let out = input.data[in_y * input.h_stride + in_x * input.w_stride];
            output.data[y * output.h_stride + x * output.w_stride] = out;
        }
    }
}

fn bilinear_resize(input: &Image, output: &mut ImageMut) {
    // Scale factors to map output coords to input coords.
    let inv_scale_y = input.height as f32 / output.height as f32;
    let inv_scale_x = input.width as f32 / output.width as f32;

    for y in 0..output.height {
        let in_y = input_coord(y, inv_scale_y).clamp(0., input.height as f32 - 1.);
        let in_y1 = in_y as usize;
        let in_y2 = (in_y1 + 1).min(input.height - 1);
        let weight_y = in_y - (in_y1 as f32);

        for x in 0..output.width {
            let in_x = input_coord(x, inv_scale_x).clamp(0., input.width as f32 - 1.);
            let in_x1 = in_x as usize;
            let in_x2 = (in_x1 + 1).min(input.width - 1);
            let weight_x = in_x - (in_x1 as f32);

            let in_tl = input.data[in_y1 * input.h_stride + in_x1 * input.w_stride];
            let in_tr = input.data[in_y1 * input.h_stride + in_x2 * input.w_stride];
            let in_bl = input.data[in_y2 * input.h_stride + in_x1 * input.w_stride];
            let in_br = input.data[in_y2 * input.h_stride + in_x2 * input.w_stride];

            // Interpolate in X direction
            let out_top = (1. - weight_x) * in_tl + weight_x * in_tr;
            let out_bottom = (1. - weight_x) * in_bl + weight_x * in_br;

            // Interpolate in Y direction
            let out = (1. - weight_y) * out_top + weight_y * out_bottom;

            output.data[y * output.h_stride + x * output.w_stride] = out;
        }
    }
}

pub fn resize(input: &Tensor, target: ResizeTarget, mode: ResizeMode) -> Result<Tensor, OpError> {
    let scales = match target {
        ResizeTarget::Scales(s) => s.clone(),
        ResizeTarget::Sizes(sizes) => {
            // TODO - Check sizes shape is correct
            let scales = zip(input.shape().iter(), sizes.elements())
                .map(|(&in_size, out_size)| (out_size as f32) / (in_size) as f32)
                .collect();
            Tensor::from_vec(scales)
        }
    };

    if scales.ndim() != 1 || scales.len() != input.ndim() {
        return Err(OpError::IncompatibleInputShapes(
            "scales should be a vector with length equal to input rank",
        ));
    }

    // The current implementation only supports NCHW tensors with scale factors
    // other than 1.0 for the H and W dims.
    if input.ndim() != 4 {
        return Err(OpError::UnsupportedValue("input must be an NCHW tensor"));
    }
    let [batch, chans, height, width] = input.dims();

    let scales_valid = (0..input.ndim())
        .all(|dim| dim == input.ndim() - 1 || dim == input.ndim() - 2 || scales[[dim]] == 1.);
    if !scales_valid {
        return Err(OpError::UnsupportedValue(
            "only height and width dimensions can be resized",
        ));
    }

    let out_shape: Vec<_> = zip(input.shape().iter(), scales.elements())
        .map(|(&size, scale)| ((size as f32) * scale) as usize)
        .collect();
    let mut output = Tensor::zeros(out_shape.as_slice());

    if output.is_empty() {
        return Ok(output);
    }

    for n in 0..batch {
        for c in 0..chans {
            let in_offset = input.offset([n, c, 0, 0]);
            let in_image = Image {
                data: &input.data()[in_offset..],
                h_stride: input.stride(2),
                height,
                w_stride: input.stride(3),
                width,
            };
            let out_offset = output.offset([n, c, 0, 0]);
            let mut out_image = ImageMut {
                h_stride: output.stride(2),
                height: output.shape()[2],
                w_stride: output.stride(3),
                width: output.shape()[3],
                data: &mut output.data_mut()[out_offset..],
            };

            match mode {
                ResizeMode::Nearest => {
                    nearest_resize(&in_image, &mut out_image);
                }
                ResizeMode::Linear => {
                    bilinear_resize(&in_image, &mut out_image);
                }
            };
        }
    }

    Ok(output)
}

#[derive(Clone, Copy, Debug)]
pub enum ResizeMode {
    Nearest,
    Linear,
}

#[derive(Debug)]
pub struct Resize {
    pub mode: ResizeMode,
}

impl Operator for Resize {
    fn name(&self) -> &str {
        "Resize"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = get_input(inputs, 0)?;

        // The `roi` input is marked as optional in ONNX, but the spec also says
        // that one of the subsequent `scales` or `sizes` inputs must be provided.
        //
        // Wasnn doesn't yet support omitting an optional input and then specifying
        // a subsequent optional input, so for now we require that `roi` and
        // `scales` must be provided, but `sizes` can be omitted.
        //
        // The `roi` input is only used if the `coordinate_transformation_mode`
        // attr is `tf_crop_and_resize`, which is not currently supported.

        let _roi = get_input::<f32>(inputs, 1)?;
        let scales = get_input(inputs, 2)?;
        let sizes = get_optional_input(inputs, 3)?;

        let target = if let Some(sizes) = sizes {
            ResizeTarget::Sizes(sizes)
        } else {
            ResizeTarget::Scales(scales)
        };

        resize(input, target, self.mode).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{resize, OpError, ResizeMode, ResizeTarget};
    use crate::tensor::Tensor;
    use crate::test_util::expect_equal;

    // Reference values for these tests can be computed with either OpenCV
    // (`cv2.resize`) or PyTorch (`torch.nn.functional.interpolate`).

    #[test]
    fn test_resize_nearest() -> Result<(), String> {
        struct Case {
            image: Tensor,
            scales: Vec<f32>,
            expected: Tensor,
        }

        let cases = [
            // Scale width and height by 0x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 0., 0.],
                expected: Tensor::from_data(vec![1, 1, 0, 0], vec![]),
            },
            // Scale width and height by 0.5x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 0.5, 0.5],
                expected: Tensor::from_data(vec![1, 1, 1, 1], vec![0.2]),
            },
            // Scale width and height by 1x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 1., 1.],
                expected: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
            },
            // Scale width and height by 1.5x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 1.5, 1.5],
                expected: Tensor::from_data(
                    vec![1, 1, 3, 3],
                    vec![
                        0.2000, 0.2000, 0.7000, // Y=0
                        0.2000, 0.2000, 0.7000, // Y=1
                        0.3000, 0.3000, 0.8000, // Y=2
                    ],
                ),
            },
            // Scale width and height by 2x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 2., 2.],
                expected: Tensor::from_data(
                    vec![1, 1, 4, 4],
                    vec![
                        0.2, 0.2, 0.7, 0.7, // Y=0
                        0.2, 0.2, 0.7, 0.7, // Y=1
                        0.3, 0.3, 0.8, 0.8, // Y=2
                        0.3, 0.3, 0.8, 0.8, // Y=3
                    ],
                ),
            },
            // Scale width and height by 3x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 3., 3.],
                expected: Tensor::from_data(
                    vec![1, 1, 6, 6],
                    vec![
                        0.2000, 0.2000, 0.2000, 0.7000, 0.7000, 0.7000, // Y=0
                        0.2000, 0.2000, 0.2000, 0.7000, 0.7000, 0.7000, // Y=1
                        0.2000, 0.2000, 0.2000, 0.7000, 0.7000, 0.7000, // Y=2
                        0.3000, 0.3000, 0.3000, 0.8000, 0.8000, 0.8000, // Y=3
                        0.3000, 0.3000, 0.3000, 0.8000, 0.8000, 0.8000, // Y=4
                        0.3000, 0.3000, 0.3000, 0.8000, 0.8000, 0.8000, // Y=5
                    ],
                ),
            },
        ];

        for case in cases {
            let scales = Tensor::from_vec(case.scales);
            let result = resize(
                &case.image,
                ResizeTarget::Scales(&scales),
                ResizeMode::Nearest,
            )
            .unwrap();

            expect_equal(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_resize_bilinear() -> Result<(), String> {
        struct Case {
            image: Tensor,
            scales: Vec<f32>,
            expected: Tensor,
        }

        let cases = [
            // Scale width and height by 0x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 0., 0.],
                expected: Tensor::from_data(vec![1, 1, 0, 0], vec![]),
            },
            // Scale width and height by 0.5x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 0.5, 0.5],

                // OpenCV and PyTorch produce different results for this case.
                // This result matches OpenCV. This relates to the `half_pixel`
                // vs `pytorch_half_pixel` values for the `coordinate_transformation_mode`
                // attribute in the ONNX op.
                expected: Tensor::from_data(vec![1, 1, 1, 1], vec![0.5]),
            },
            // Scale width and height by 1x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 1., 1.],
                expected: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
            },
            // Scale width and height by 1.5x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 1.5, 1.5],
                expected: Tensor::from_data(
                    vec![1, 1, 3, 3],
                    vec![
                        0.2, 0.45, 0.7, // Y=0
                        0.25, 0.5, 0.75, // Y=1
                        0.3, 0.55, 0.8, // Y=2
                    ],
                ),
            },
            // Scale width and height by 2x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 2., 2.],
                expected: Tensor::from_data(
                    vec![1, 1, 4, 4],
                    vec![
                        0.2, 0.325, 0.575, 0.7, // Y=0
                        0.225, 0.35, 0.6, 0.725, // Y=1
                        0.275, 0.4, 0.65, 0.775, // Y=2
                        0.3, 0.425, 0.675, 0.8, // Y=3
                    ],
                ),
            },
            // Scale width and height by 3x
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 3., 3.],
                expected: Tensor::from_data(
                    vec![1, 1, 6, 6],
                    vec![
                        0.2000, 0.2000, 0.3667, 0.5333, 0.7000, 0.7000, // Y=0
                        0.2000, 0.2000, 0.3667, 0.5333, 0.7000, 0.7000, // Y=1
                        0.2333, 0.2333, 0.4000, 0.5667, 0.7333, 0.7333, // Y=2
                        0.2667, 0.2667, 0.4333, 0.6000, 0.7667, 0.7667, // Y=3
                        0.3000, 0.3000, 0.4667, 0.6333, 0.8000, 0.8000, // Y=4
                        0.3000, 0.3000, 0.4667, 0.6333, 0.8000, 0.8000, // Y=5
                    ],
                ),
            },
        ];

        for case in cases {
            let scales = Tensor::from_vec(case.scales);

            let result = resize(
                &case.image,
                ResizeTarget::Scales(&scales),
                ResizeMode::Linear,
            )
            .unwrap();

            expect_equal(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_resize_invalid_inputs() {
        struct Case {
            image: Tensor,
            scales: Tensor,
            expected: OpError,
        }

        let cases = [
            Case {
                image: Tensor::from_vec(vec![1., 1.]),
                scales: Tensor::from_vec(vec![1.]),
                expected: OpError::UnsupportedValue("input must be an NCHW tensor"),
            },
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: Tensor::from_data(vec![1, 1, 2, 2], vec![1., 1., 3., 3.]),
                expected: OpError::IncompatibleInputShapes(
                    "scales should be a vector with length equal to input rank",
                ),
            },
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: Tensor::from_vec(vec![3., 3.]),
                expected: OpError::IncompatibleInputShapes(
                    "scales should be a vector with length equal to input rank",
                ),
            },
            Case {
                image: Tensor::from_data(vec![1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: Tensor::from_vec(vec![2., 1., 3., 3.]),
                expected: OpError::UnsupportedValue(
                    "only height and width dimensions can be resized",
                ),
            },
        ];

        for case in cases {
            let result = resize(
                &case.image,
                ResizeTarget::Scales(&case.scales),
                ResizeMode::Linear,
            );
            assert_eq!(result.err(), Some(case.expected));
        }
    }
}
