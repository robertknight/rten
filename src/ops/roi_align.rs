use rayon::prelude::*;
use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};

/// Method used to combine the samples taken within each output bin.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum RoiAlignMode {
    /// Average the samples.
    #[default]
    Avg,

    /// Take the maximum of the samples.
    Max,
}

/// Method used to map ROI coordinates to input coordinates.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum RoiAlignCoordTransformMode {
    /// Shift the scaled ROI coordinates by -0.5, so that they refer to pixel
    /// centers.
    #[default]
    HalfPixel,

    /// Use the scaled ROI coordinates unmodified.
    OutputHalfPixel,
}

/// Positions and weights used to bilinearly sample one point from an input
/// image.
///
/// Positions are offsets into a contiguous `(height, width)` image.
#[derive(Default)]
struct BilinearSample {
    pos: [usize; 4],
    weight: [f32; 4],
}

impl BilinearSample {
    /// Compute the positions and weights for sampling an image of size
    /// `(height, width)` at `(y, x)`.
    ///
    /// Both `height` and `width` must be non-zero.
    fn new(height: usize, width: usize, y: f32, x: f32) -> BilinearSample {
        if y < -1.0 || y > height as f32 || x < -1.0 || x > width as f32 {
            // Points which lie more than one pixel outside the image
            // contribute nothing to the output.
            return BilinearSample::default();
        }

        let (y_low, y_high, ly) = resolve_coord(y, height);
        let (x_low, x_high, lx) = resolve_coord(x, width);
        let hy = 1.0 - ly;
        let hx = 1.0 - lx;

        BilinearSample {
            pos: [
                y_low * width + x_low,
                y_low * width + x_high,
                y_high * width + x_low,
                y_high * width + x_high,
            ],
            weight: [hy * hx, hy * lx, ly * hx, ly * lx],
        }
    }

    /// Return the weighted values of the four pixels adjacent to this sample.
    ///
    /// `image` is a contiguous `(height, width)` image with the size passed to
    /// [`BilinearSample::new`].
    fn weighted_pixels(&self, image: &[f32]) -> [f32; 4] {
        std::array::from_fn(|i| {
            // Safety: `BilinearSample::new` resolves positions within the
            // `(height, width)` image it was given, and out of range samples
            // use position zero, which is in bounds since the image is
            // non-empty.
            let pixel = unsafe { *image.get_unchecked(self.pos[i]) };
            self.weight[i] * pixel
        })
    }
}

/// Resolve a coordinate along an axis of size `size` to the positions of the
/// two adjacent pixels and the interpolation factor between them.
///
/// Coordinates beyond the end of the axis are clamped to the last pixel.
/// `size` must be non-zero.
fn resolve_coord(coord: f32, size: usize) -> (usize, usize, f32) {
    let coord = coord.max(0.);
    let low = coord as usize;
    if low >= size - 1 {
        let low = size - 1;
        (low, low, 0.)
    } else {
        (low, low + 1, coord - low as f32)
    }
}

#[derive(Debug)]
pub struct RoiAlign {
    pub coord_mode: RoiAlignCoordTransformMode,
    pub mode: RoiAlignMode,
    pub output_height: usize,
    pub output_width: usize,
    pub sampling_ratio: i32,
    pub spatial_scale: f32,
}

impl RoiAlign {
    /// Pool features from the regions of interest in `rois`.
    fn apply(
        &self,
        pool: &BufferPool,
        input: NdTensorView<f32, 4>,
        rois: NdTensorView<f32, 2>,
        batch_indices: NdTensorView<i32, 1>,
    ) -> Result<NdTensor<f32, 4>, OpError> {
        let &RoiAlign {
            coord_mode,
            mode,
            output_height: out_h,
            output_width: out_w,
            sampling_ratio,
            spatial_scale,
        } = self;

        let [batch, chans, in_h, in_w] = input.shape();
        let [n_rois, roi_len] = rois.shape();

        if roi_len != 4 {
            return Err(OpError::invalid_value(
                "`rois` must have shape (num_rois, 4)",
            ));
        }

        if batch_indices.size(0) != n_rois {
            return Err(OpError::incompatible_input_shapes(
                "`rois` and `batch_indices` must have the same length",
            ));
        }

        for batch_index in batch_indices.iter().copied() {
            if batch_index < 0 || batch_index as usize >= batch {
                return Err(OpError::invalid_value(format!(
                    "Batch index {} is out of range. Must be in [0, {})",
                    batch_index, batch
                )));
            }
        }

        let out_shape = [n_rois, chans, out_h, out_w];

        if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
            // Either the output is empty, or every sample is out of bounds and so
            // all outputs are zero.
            return Ok(NdTensor::zeros_in(pool, out_shape));
        }

        let input = input.to_contiguous_in(pool).auto_return(pool);
        let mut output = NdTensor::uninit_in(pool, out_shape);

        // For `half_pixel` mode the ROI coordinates refer to pixel centers, so
        // shift them to make them relative to the top-left of the input.
        let offset = match coord_mode {
            RoiAlignCoordTransformMode::HalfPixel => 0.5,
            RoiAlignCoordTransformMode::OutputHalfPixel => 0.,
        };
        let too_many_samples = || OpError::invalid_value("Number of ROI samples is too large");

        // Positions and weights of the samples taken for each output element of an
        // ROI, in `(out_y, out_x, grid_y, grid_x)` order. These are shared by all
        // channels.
        let mut samples: Vec<BilinearSample> = Vec::new();

        for n in 0..n_rois {
            let start_x = rois[[n, 0]] * spatial_scale - offset;
            let start_y = rois[[n, 1]] * spatial_scale - offset;
            let end_x = rois[[n, 2]] * spatial_scale - offset;
            let end_y = rois[[n, 3]] * spatial_scale - offset;

            let mut roi_w = end_x - start_x;
            let mut roi_h = end_y - start_y;
            if coord_mode == RoiAlignCoordTransformMode::OutputHalfPixel {
                // Force malformed ROIs to be 1x1.
                roi_w = roi_w.max(1.0);
                roi_h = roi_h.max(1.0);
            }

            let bin_h = roi_h / out_h as f32;
            let bin_w = roi_w / out_w as f32;

            // Size of the grid of samples taken within each output bin. If
            // `sampling_ratio` is unset, adapt the grid to the ROI size so that
            // each input pixel within the bin is sampled approximately once.
            let (grid_h, grid_w) = if sampling_ratio > 0 {
                let ratio = sampling_ratio as usize;
                (ratio, ratio)
            } else {
                (bin_h.ceil() as usize, bin_w.ceil() as usize)
            };
            let samples_per_bin = grid_h.checked_mul(grid_w).ok_or_else(too_many_samples)?;
            let n_samples = samples_per_bin
                .checked_mul(out_h)
                .and_then(|n| n.checked_mul(out_w))
                .ok_or_else(too_many_samples)?;
            let count = samples_per_bin.max(1) as f32;

            samples.clear();
            samples.reserve(n_samples);
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    for grid_y in 0..grid_h {
                        let y = start_y
                            + out_y as f32 * bin_h
                            + (grid_y as f32 + 0.5) * bin_h / grid_h as f32;
                        for grid_x in 0..grid_w {
                            let x = start_x
                                + out_x as f32 * bin_w
                                + (grid_x as f32 + 0.5) * bin_w / grid_w as f32;
                            samples.push(BilinearSample::new(in_h, in_w, y, x));
                        }
                    }
                }
            }

            let in_batch = input.slice(batch_indices[[n]] as usize);

            output
                .slice_mut(n)
                .axis_iter_mut(0)
                .into_par_iter()
                .zip(in_batch.axis_iter(0))
                .for_each(|(mut out_chan, in_chan)| {
                    // The input was made contiguous above, so each channel is a
                    // contiguous image.
                    let image = in_chan.data().unwrap();

                    for (bin, out) in out_chan.iter_mut().enumerate() {
                        let bin_samples = &samples[bin * samples_per_bin..][..samples_per_bin];
                        let val = match mode {
                            RoiAlignMode::Avg => {
                                let sum: f32 = bin_samples
                                    .iter()
                                    .map(|s| s.weighted_pixels(image).iter().sum::<f32>())
                                    .sum();
                                sum / count
                            }
                            RoiAlignMode::Max => bin_samples
                                .iter()
                                .map(|s| {
                                    let [a, b, c, d] = s.weighted_pixels(image);
                                    a.max(b).max(c).max(d)
                                })
                                .reduce(f32::max)
                                .unwrap_or(0.),
                        };
                        out.write(val);
                    }
                });
        }

        // Safety: We initialized all output values.
        Ok(unsafe { output.assume_init() })
    }
}

impl Operator for RoiAlign {
    fn name(&self) -> &str {
        "RoiAlign"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let rois = inputs.require_as(1)?;
        let batch_indices = inputs.require_as(2)?;

        self.apply(ctx.pool(), input, rois, batch_indices)
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
    RoiAlign,
    op,
    shape_ops::RoiAlign {
        output_height: op.output_height,
        output_width: op.output_width,
    }
);

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;
    use rten_testing::TestCases;

    use super::{RoiAlign, RoiAlignCoordTransformMode, RoiAlignMode};
    use crate::operator::{OpError, OperatorExt};
    use crate::ops::tests::expect_eq_1e4;

    /// Input with shape `(1, 2, 5, 5)`.
    fn input_1x2x5x5() -> NdTensor<f32, 4> {
        NdTensor::from([
            [
                [0.9794, 0.9767, 0.9879, 0.3802, 0.1714],
                [0.9232, 0.1049, 0.2617, 0.1388, 0.3191],
                [0.5360, 0.1181, 0.7959, 0.2418, 0.7861],
                [0.3185, 0.7935, 0.9641, 0.9583, 0.2636],
                [0.5556, 0.4410, 0.2547, 0.6099, 0.8971],
            ],
            [
                [0.8636, 0.6460, 0.8638, 0.5185, 0.6749],
                [0.9551, 0.6599, 0.0473, 0.7358, 0.7387],
                [0.2228, 0.8671, 0.1721, 0.7887, 0.8704],
                [0.4828, 0.0601, 0.1623, 0.6837, 0.0416],
                [0.6712, 0.1397, 0.6110, 0.5487, 0.0601],
            ],
        ])
        .with_new_axis(0)
    }

    /// Operator with the ONNX default attributes, except for a 2x2 output.
    fn default_op() -> RoiAlign {
        RoiAlign {
            coord_mode: RoiAlignCoordTransformMode::HalfPixel,
            mode: RoiAlignMode::Avg,
            output_height: 2,
            output_width: 2,
            sampling_ratio: 0,
            spatial_scale: 1.0,
        }
    }

    /// Input with shape `(2, 1, 4, 4)`.
    fn input_2x1x4x4() -> NdTensor<f32, 4> {
        NdTensor::from([
            [
                [0.1785, 0.9778, 0.4034, 0.4390],
                [0.4533, 0.5326, 0.1930, 0.0031],
                [0.6457, 0.2513, 0.6576, 0.8585],
                [0.4309, 0.4253, 0.1760, 0.7358],
            ],
            [
                [0.6989, 0.9220, 0.7760, 0.1535],
                [0.8035, 0.9923, 0.3576, 0.1823],
                [0.8519, 0.9401, 0.9749, 0.0869],
                [0.8790, 0.4682, 0.4318, 0.8290],
            ],
        ])
        .with_new_axis(1)
    }

    // Expected values in these tests were generated using the ONNX reference
    // implementation of RoiAlign.
    #[test]
    fn test_roi_align() {
        #[derive(Debug)]
        struct Case {
            input: NdTensor<f32, 4>,
            rois: NdTensor<f32, 2>,
            batch_indices: NdTensor<i32, 1>,
            op: RoiAlign,
            expected: NdTensor<f32, 4>,
        }

        let cases = [
            // Default attributes.
            Case {
                input: input_1x2x5x5(),
                rois: NdTensor::from([[0.5, 0.5, 3.5, 3.5]]),
                batch_indices: NdTensor::from([0]),
                op: default_op(),
                expected: NdTensor::from([
                    [[0.5554, 0.4189], [0.4100, 0.6735]],
                    [[0.7124, 0.4454], [0.4956, 0.3736]],
                ])
                .with_new_axis(0),
            },
            // `output_half_pixel` coordinate transform.
            Case {
                input: input_1x2x5x5(),
                rois: NdTensor::from([[0.5, 0.5, 3.5, 3.5]]),
                batch_indices: NdTensor::from([0]),
                op: RoiAlign {
                    coord_mode: RoiAlignCoordTransformMode::OutputHalfPixel,
                    ..default_op()
                },
                expected: NdTensor::from([
                    [[0.3007, 0.2929], [0.6463, 0.7455]],
                    [[0.5403, 0.5531], [0.2846, 0.5218]],
                ])
                .with_new_axis(0),
            },
            // Max pooling.
            Case {
                input: input_1x2x5x5(),
                rois: NdTensor::from([[0.5, 0.5, 3.5, 3.5]]),
                batch_indices: NdTensor::from([0]),
                op: RoiAlign {
                    mode: RoiAlignMode::Max,
                    ..default_op()
                },
                expected: NdTensor::from([
                    [[0.5341, 0.5403], [0.4339, 0.6094]],
                    [[0.5223, 0.4724], [0.6639, 0.4313]],
                ])
                .with_new_axis(0),
            },
            // Fixed sampling ratio.
            Case {
                input: input_1x2x5x5(),
                rois: NdTensor::from([[0., 0., 4., 4.]]),
                batch_indices: NdTensor::from([0]),
                op: RoiAlign {
                    sampling_ratio: 2,
                    ..default_op()
                },
                expected: NdTensor::from([
                    [[0.7460, 0.4422], [0.4415, 0.7400]],
                    [[0.7812, 0.5414], [0.4082, 0.4517]],
                ])
                .with_new_axis(0),
            },
            // Spatial scale. The ROI is twice the size of the one in the
            // previous case, so scaling it by 0.5 gives the same output.
            Case {
                input: input_1x2x5x5(),
                rois: NdTensor::from([[1., 1., 7., 7.]]),
                batch_indices: NdTensor::from([0]),
                op: RoiAlign {
                    spatial_scale: 0.5,
                    ..default_op()
                },
                expected: NdTensor::from([
                    [[0.5554, 0.4189], [0.4100, 0.6735]],
                    [[0.7124, 0.4454], [0.4956, 0.3736]],
                ])
                .with_new_axis(0),
            },
            // ROI which extends beyond the input. Samples which are out of
            // bounds are treated as zero.
            Case {
                input: input_1x2x5x5(),
                rois: NdTensor::from([[-3., -3., 1.5, 1.5]]),
                batch_indices: NdTensor::from([0]),
                op: RoiAlign {
                    sampling_ratio: 2,
                    ..default_op()
                },
                expected: NdTensor::from([[[0., 0.], [0., 0.9275]], [[0., 0.], [0., 0.8323]]])
                    .with_new_axis(0),
            },
            // Multiple ROIs sampling different images in the batch.
            Case {
                input: input_2x1x4x4(),
                rois: NdTensor::from([[0., 0., 2., 2.], [1., 1., 3., 3.]]),
                batch_indices: NdTensor::from([1, 0]),
                op: default_op(),
                expected: NdTensor::from([
                    [[0.6989, 0.9220], [0.8035, 0.9923]],
                    [[0.5326, 0.1930], [0.2513, 0.6576]],
                ])
                .with_new_axis(1),
            },
            // Non-square output.
            Case {
                input: input_1x2x5x5(),
                rois: NdTensor::from([[0.25, 0.75, 4.25, 3.75]]),
                batch_indices: NdTensor::from([0]),
                op: RoiAlign {
                    coord_mode: RoiAlignCoordTransformMode::OutputHalfPixel,
                    mode: RoiAlignMode::Max,
                    output_height: 3,
                    output_width: 3,
                    sampling_ratio: 1,
                    ..default_op()
                },
                expected: NdTensor::from([
                    [
                        [0.0721, 0.1492, 0.1396],
                        [0.1818, 0.4477, 0.3439],
                        [0.5455, 0.5423, 0.2995],
                    ],
                    [
                        [0.4537, 0.1380, 0.3232],
                        [0.5961, 0.1479, 0.3808],
                        [0.0413, 0.1282, 0.2137],
                    ],
                ])
                .with_new_axis(0),
            },
            // Empty input. All samples are out of bounds.
            Case {
                input: NdTensor::zeros([1, 2, 0, 0]),
                rois: NdTensor::from([[0., 0., 1., 1.]]),
                batch_indices: NdTensor::from([0]),
                op: default_op(),
                expected: NdTensor::zeros([1, 2, 2, 2]),
            },
            // No ROIs.
            Case {
                input: input_1x2x5x5(),
                rois: NdTensor::zeros([0, 4]),
                batch_indices: NdTensor::zeros([0]),
                op: default_op(),
                expected: NdTensor::zeros([0, 2, 2, 2]),
            },
        ];

        cases.test_each(|case| {
            let result: NdTensor<f32, 4> = case
                .op
                .run_simple((
                    case.input.view(),
                    case.rois.view(),
                    case.batch_indices.view(),
                ))
                .unwrap();
            assert_eq!(result.shape(), case.expected.shape());
            expect_eq_1e4(&result, &case.expected).unwrap();
        });
    }

    #[test]
    fn test_roi_align_invalid() {
        #[derive(Debug)]
        struct Case {
            rois: NdTensor<f32, 2>,
            batch_indices: NdTensor<i32, 1>,
            expected: OpError,
        }

        let cases = [
            Case {
                rois: NdTensor::zeros([1, 5]),
                batch_indices: NdTensor::from([0]),
                expected: OpError::invalid_value("`rois` must have shape (num_rois, 4)"),
            },
            Case {
                rois: NdTensor::zeros([2, 4]),
                batch_indices: NdTensor::from([0]),
                expected: OpError::incompatible_input_shapes(
                    "`rois` and `batch_indices` must have the same length",
                ),
            },
            Case {
                rois: NdTensor::zeros([1, 4]),
                batch_indices: NdTensor::from([1]),
                expected: OpError::invalid_value(
                    "Batch index 1 is out of range. Must be in [0, 1)",
                ),
            },
            Case {
                rois: NdTensor::zeros([1, 4]),
                batch_indices: NdTensor::from([-1]),
                expected: OpError::invalid_value(
                    "Batch index -1 is out of range. Must be in [0, 1)",
                ),
            },
        ];

        let input = input_1x2x5x5();

        cases.test_each(|case| {
            let result = default_op().run_simple::<_, NdTensor<f32, 4>>((
                input.view(),
                case.rois.view(),
                case.batch_indices.view(),
            ));
            assert_eq!(result.err().as_ref(), Some(&case.expected));
        });
    }
}
