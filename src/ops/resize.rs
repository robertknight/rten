use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut, Tensor, TensorView};

use crate::iter_util::range_chunks;
use crate::ops::{
    static_dims, Input, InputList, IntoOpResult, OpError, Operator, Output, OutputList,
};
use crate::tensor_pool::{AutoReturn, TensorPool};

/// Specifies an output size for a resize operation.
pub enum ResizeTarget<'a> {
    /// Scale factors for each dimension. The length should match the input rank.
    Scales(NdTensorView<'a, f32, 1>),

    /// Output sizes for each dimension. The length should match the input rank.
    Sizes(NdTensorView<'a, i32, 1>),
}

/// Compute the input image coordinate that corresponds to an output coordinate,
/// along an axis.
///
/// - `dest_coord` is the coordinate along the output axis
/// - `scale` is the scale factor from output to input along the axis
/// - `mode` specifies the coordinate transformation mode from the
///   `coordinate_transformation_mode` attribute.
/// - `length_original` is the size of the axis in the input
/// - `length_resized` is the size of the axis in the output
///
/// See https://github.com/onnx/onnx/blob/v1.15.0/docs/Operators.md#resize
/// for the formulae for different transform modes.
///
/// The default is half pixel, and is is consistent with how OpenCV
/// (`cv2.resize`) and PyTorch (`torch.nn.functional.interpolate`) work. See
/// https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/ for
/// rationale.
fn input_coord(
    dest_coord: usize,
    scale: f32,
    mode: CoordTransformMode,
    length_original: usize,
    length_resized: usize,
) -> f32 {
    type Ctm = CoordTransformMode;
    match mode {
        Ctm::HalfPixel => scale * (dest_coord as f32 + 0.5) - 0.5,
        Ctm::Asymmetric => scale * dest_coord as f32,
        Ctm::AlignCorners => {
            dest_coord as f32 * (length_original - 1) as f32 / (length_resized - 1) as f32
        }
    }
}

/// Specifies how resizing with `ResizeMode::Nearest` should map a fractional
/// input coordinate to an image coordinate.
#[derive(Copy, Clone, Debug, Default)]
pub enum NearestMode {
    Ceil,
    Floor,
    RoundPreferCeil,

    #[default]
    RoundPreferFloor,
}

/// Specifies how resizing maps output coordinates to input coordinates.
#[derive(Copy, Clone, Debug, Default)]
pub enum CoordTransformMode {
    #[default]
    HalfPixel,
    Asymmetric,
    AlignCorners,
}

const CHAN_GROUP_SIZE: usize = 4;

/// Interpolate between `a` and `b` according to `weight`.
fn lerp(a: f32, b: f32, weight: f32) -> f32 {
    (1. - weight) * a + weight * b
}

/// Resize a group of channels in a CHW tensor using nearest neighbor resizing.
///
/// This initializes all elements of `output`.
fn nearest_resize(
    input: NdTensorView<f32, 3>,
    mut output: NdTensorViewMut<MaybeUninit<f32>, 3>,
    mode: NearestMode,
    coord_mode: CoordTransformMode,
) {
    let [chans, rows, cols] = output.shape();
    let [_, in_rows, in_cols] = input.shape();

    // Scale factors to map output coords to input coords.
    let inv_scale_y = in_rows as f32 / rows as f32;
    let inv_scale_x = in_cols as f32 / cols as f32;

    let round_coord = |coord: f32| match mode {
        NearestMode::Ceil => coord.ceil() as usize,
        NearestMode::Floor => coord as usize,

        // `f32::round` has round-away-from-zero behavior. For `RoundPreferCeil`
        // and `RoundPreferFloor` we need to always round up or down.
        NearestMode::RoundPreferCeil => {
            if coord.fract() == 0.5 {
                coord.ceil() as usize
            } else {
                coord.round() as usize
            }
        }
        NearestMode::RoundPreferFloor => {
            if coord.fract() == 0.5 {
                coord.floor() as usize
            } else {
                coord.round() as usize
            }
        }
    };

    let mut n_init = 0;
    for y in 0..rows {
        let in_y = round_coord(
            input_coord(y, inv_scale_y, coord_mode, in_rows, rows).clamp(0., in_rows as f32 - 1.),
        );
        for x in 0..cols {
            let in_x = round_coord(
                input_coord(x, inv_scale_x, coord_mode, in_cols, cols)
                    .clamp(0., in_cols as f32 - 1.),
            );

            for c in 0..chans {
                output[[c, y, x]].write(input[[c, in_y, in_x]]);
                n_init += 1;
            }
        }
    }
    assert!(n_init == output.len());
}

/// Resize a group of channels in a CHW tensor using bilinear resizing.
///
/// This initializes all elements of `output`.
fn bilinear_resize(
    input: NdTensorView<f32, 3>,
    mut output: NdTensorViewMut<MaybeUninit<f32>, 3>,
    coord_mode: CoordTransformMode,
) {
    let [chans, rows, cols] = output.shape();
    let [_, in_rows, in_cols] = input.shape();

    // Scale factors to map output coords to input coords.
    let inv_scale_y = in_rows as f32 / rows as f32;
    let inv_scale_x = in_cols as f32 / cols as f32;

    let n_init = AtomicUsize::new(0);

    let row_chunk = rows.div_ceil(
        std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1),
    );

    output
        .axis_chunks_mut(1, row_chunk)
        .zip(range_chunks(0..rows, row_chunk))
        .par_bridge()
        .for_each(|(mut out_row_chunk, out_row_range)| {
            for y in out_row_range.clone() {
                let in_y = input_coord(y, inv_scale_y, coord_mode, in_rows, rows)
                    .clamp(0., in_rows as f32 - 1.);
                let in_y1 = in_y as usize;
                let in_y2 = (in_y1 + 1).min(in_rows - 1);
                let weight_y = in_y - (in_y1 as f32);

                for x in 0..cols {
                    let in_x = input_coord(x, inv_scale_x, coord_mode, in_cols, cols)
                        .clamp(0., in_cols as f32 - 1.);
                    let in_x1 = in_x as usize;
                    let in_x2 = (in_x1 + 1).min(in_cols - 1);
                    let weight_x = in_x - (in_x1 as f32);

                    const N: usize = CHAN_GROUP_SIZE;
                    if chans == N {
                        let in_tl = input.get_array::<N>([0, in_y1, in_x1], 0);
                        let in_tr = input.get_array::<N>([0, in_y1, in_x2], 0);
                        let in_bl = input.get_array::<N>([0, in_y2, in_x1], 0);
                        let in_br = input.get_array::<N>([0, in_y2, in_x2], 0);

                        let mut out = [MaybeUninit::new(0.); N];
                        for c in 0..chans {
                            // Interpolate in X direction
                            let out_top = lerp(in_tl[c], in_tr[c], weight_x);
                            let out_bottom = lerp(in_bl[c], in_br[c], weight_x);

                            // Interpolate in Y direction
                            out[c].write(lerp(out_top, out_bottom, weight_y));
                        }

                        out_row_chunk.set_array([0, y - out_row_range.start, x], 0, out);
                    } else {
                        for c in 0..chans {
                            let in_tl = input[[c, in_y1, in_x1]];
                            let in_tr = input[[c, in_y1, in_x2]];
                            let in_bl = input[[c, in_y2, in_x1]];
                            let in_br = input[[c, in_y2, in_x2]];

                            // Interpolate in X direction
                            let out_top = lerp(in_tl, in_tr, weight_x);
                            let out_bottom = lerp(in_bl, in_br, weight_x);

                            // Interpolate in Y direction
                            out_row_chunk[[c, y - out_row_range.start, x]]
                                .write(lerp(out_top, out_bottom, weight_y));
                        }
                    }
                }
            }
            n_init.fetch_add(out_row_chunk.len(), Ordering::SeqCst);
        });
    assert!(n_init.load(Ordering::SeqCst) == output.len());
}

/// Resize an NCHW image tensor to a given `[height, width]`.
///
/// This is a simplified API for [`resize`].
pub fn resize_image(input: TensorView, size: [usize; 2]) -> Result<Tensor, OpError> {
    let [batch, chans, _height, _width] = static_dims!(input, 4)?.shape();
    let [out_height, out_width] = size;
    let out_shape = [batch, chans, out_height, out_width].map(|x| x as i32);
    resize(
        &TensorPool::new(),
        input,
        ResizeTarget::Sizes(out_shape.as_slice().into()),
        ResizeMode::Linear,
        CoordTransformMode::default(),
        NearestMode::default(),
    )
}

/// Resolve the target output size, specified as either as scale factors or
/// fixed sizes, into a fixed size.
fn calc_output_size(input_shape: &[usize], target: ResizeTarget) -> Result<Vec<usize>, OpError> {
    let sizes: NdTensor<i32, 1> = match target {
        ResizeTarget::Scales(scales) => input_shape
            .iter()
            .zip(scales.iter())
            .map(|(&in_size, scale)| ((in_size as f32) * scale).floor() as i32)
            .collect(),
        ResizeTarget::Sizes(sizes) => sizes.to_tensor(),
    };

    if sizes.len() != input_shape.len() {
        return Err(OpError::IncompatibleInputShapes(
            "scales/sizes length should equal input rank",
        ));
    }
    if sizes.iter().any(|size| *size < 0) {
        return Err(OpError::InvalidValue("scales/sizes must be positive"));
    }

    Ok(sizes.into_data().into_iter().map(|x| x as usize).collect())
}

/// Compute the target output size from the `scales` and `sizes` inputs to a
/// Resize operator.
fn target_from_scale_size_inputs<'a>(
    inputs: &InputList<'a>,
    scales_input_idx: usize,
) -> Result<ResizeTarget<'a>, OpError> {
    let scales = get_optional_input(inputs, scales_input_idx)?
        .map(|scales| static_dims!(scales, 1))
        .transpose()?
        .map(ResizeTarget::Scales);
    let sizes = get_optional_input(inputs, scales_input_idx + 1)?
        .map(|sizes| static_dims!(sizes, 1))
        .transpose()?
        .map(ResizeTarget::Sizes);
    scales.or(sizes).ok_or(OpError::MissingInputs)
}

fn resize_impl(
    pool: &TensorPool,
    input: TensorView,
    output_size: &[usize],
    mode: ResizeMode,
    coord_mode: CoordTransformMode,
    nearest_mode: NearestMode,
) -> Result<Tensor, OpError> {
    // Fall back to a simple copy if this is a no-op resize.
    if input.shape() == output_size {
        return Ok(input.to_tensor_in(pool));
    }

    // The current implementation only supports NCHW tensors with scale factors
    // other than 1.0 for the H and W dims.
    let input = static_dims!(input, 4, "NCHW")?;
    let [batch, _chans, _height, _width] = input.shape();
    let sizes_valid = (0..input.ndim()).zip(input.shape()).all(|(dim, in_size)| {
        dim == input.ndim() - 1 || dim == input.ndim() - 2 || output_size[dim] == in_size
    });
    if !sizes_valid {
        return Err(OpError::UnsupportedValue(
            "only height and width dimensions can be resized",
        ));
    }

    let mut output = Tensor::uninit_in(pool, output_size);

    if output.is_empty() {
        // Safety: Empty output is already initialized.
        let output = unsafe { output.assume_init() };
        return Ok(output);
    }

    let n_init = AtomicUsize::new(0);
    for n in 0..batch {
        let in_image = input.slice([n]);
        let mut out_batch = output.nd_view_mut::<4>();
        let mut out_image = out_batch.slice_mut([n]);

        out_image
            .axis_chunks_mut(0, CHAN_GROUP_SIZE)
            .zip(in_image.axis_chunks(0, CHAN_GROUP_SIZE))
            .par_bridge()
            .for_each(|(mut out_chans, in_chans)| {
                match mode {
                    ResizeMode::Nearest => {
                        nearest_resize(in_chans, out_chans.view_mut(), nearest_mode, coord_mode);
                    }
                    ResizeMode::Linear => {
                        bilinear_resize(in_chans, out_chans.view_mut(), coord_mode);
                    }
                };
                n_init.fetch_add(out_chans.len(), Ordering::SeqCst);
            });
    }

    assert!(n_init.load(Ordering::SeqCst) == output.len());
    let output = unsafe { output.assume_init() };

    Ok(output)
}

pub fn resize(
    pool: &TensorPool,
    input: TensorView,
    target: ResizeTarget,
    mode: ResizeMode,
    coord_mode: CoordTransformMode,
    nearest_mode: NearestMode,
) -> Result<Tensor, OpError> {
    let sizes = calc_output_size(input.shape(), target)?;
    resize_impl(pool, input, &sizes, mode, coord_mode, nearest_mode)
}

/// Get an optional input for the Resize operator, treating empty tensors as
/// missing inputs.
///
/// This is needed for compatibility with ONNX models generated by PyTorch when
/// targeting opset < 13. See https://github.com/pytorch/pytorch/pull/50574.
fn get_optional_input<'a, T>(
    inputs: &InputList<'a>,
    index: usize,
) -> Result<Option<TensorView<'a, T>>, OpError>
where
    TensorView<'a, T>: TryFrom<Input<'a>, Error = OpError>,
{
    let tensor = inputs.get_as(index)?.filter(|t| !t.is_empty());
    Ok(tensor)
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ResizeMode {
    #[default]
    Nearest,
    Linear,
}

#[derive(Debug)]
pub struct Resize {
    pub mode: ResizeMode,
    pub coord_mode: CoordTransformMode,
    pub nearest_mode: NearestMode,
}

impl Default for Resize {
    fn default() -> Resize {
        Resize {
            mode: ResizeMode::Nearest,
            coord_mode: CoordTransformMode::default(),
            nearest_mode: NearestMode::default(),
        }
    }
}

impl Operator for Resize {
    fn name(&self) -> &str {
        "Resize"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;

        // The `roi` input is only used if the `coordinate_transformation_mode`
        // ONNX attr is `tf_crop_and_resize`, which is not currently supported.
        let _roi = get_optional_input::<f32>(&inputs, 1)?;
        let target = target_from_scale_size_inputs(&inputs, 2)?;

        resize(
            pool,
            input,
            target,
            self.mode,
            self.coord_mode,
            self.nearest_mode,
        )
        .into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        // Resize can run in place if the computed output size is the same
        // as the input size. In that case the in-place operation is a noop.
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        // See note in `run` about the `roi` input.

        let target = target_from_scale_size_inputs(&other, 1)?;
        let output_size = calc_output_size(input.shape(), target)?;

        // If this is a no-op resize, just return the input.
        if input.shape() == output_size {
            return Ok(input);
        }

        let input = Tensor::<f32>::try_from(input)?.auto_return(pool);
        resize_impl(
            pool,
            input.view(),
            &output_size,
            self.mode,
            self.coord_mode,
            self.nearest_mode,
        )
        .map(|t| t.into())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, NdTensorView, Tensor};

    use crate::ops::tests::expect_eq_1e4;
    use crate::ops::tests::new_pool;
    use crate::ops::{
        resize, CoordTransformMode, InputList, NearestMode, OpError, Operator, Resize, ResizeMode,
        ResizeTarget,
    };

    // Reference values for these tests can be computed with either OpenCV
    // (`cv2.resize`) or PyTorch (`torch.nn.functional.interpolate`).

    #[test]
    fn test_resize_nearest() -> Result<(), Box<dyn Error>> {
        struct Case {
            image: Tensor,
            scales: Vec<f32>,
            expected: Tensor,
        }

        let cases = [
            // Scale width and height by 0x
            Case {
                image: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 0., 0.],
                expected: Tensor::from_data(&[1, 1, 0, 0], vec![]),
            },
            // Scale width and height by 0.5x
            Case {
                image: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 0.5, 0.5],
                expected: Tensor::from_data(&[1, 1, 1, 1], vec![0.2]),
            },
            // Scale width and height by 1x
            Case {
                image: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 1., 1.],
                expected: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
            },
            // Scale width and height by 1.5x
            Case {
                image: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 1.5, 1.5],
                expected: Tensor::from_data(
                    &[1, 1, 3, 3],
                    vec![
                        0.2000, 0.2000, 0.7000, // Y=0
                        0.2000, 0.2000, 0.7000, // Y=1
                        0.3000, 0.3000, 0.8000, // Y=2
                    ],
                ),
            },
            // Scale width and height by 2x
            Case {
                image: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 2., 2.],
                expected: Tensor::from_data(
                    &[1, 1, 4, 4],
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
                image: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: vec![1., 1., 3., 3.],
                expected: Tensor::from_data(
                    &[1, 1, 6, 6],
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

        let pool = new_pool();
        for case in cases {
            let result = resize(
                &pool,
                case.image.view(),
                ResizeTarget::Scales(case.scales.as_slice().into()),
                ResizeMode::Nearest,
                CoordTransformMode::HalfPixel,
                NearestMode::RoundPreferFloor,
            )
            .unwrap();

            expect_equal(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_resize_nearest_mode() -> Result<(), Box<dyn Error>> {
        let image = Tensor::from_data(&[1, 1, 1, 2], vec![0.1, 0.2]);

        // Use a scale factor of 4 so that we have output pixels that map
        // to input coordinates with fractional values of 0, 0.25, 0.5 and 0.75.
        // This allows the same input to exercise all the rounding modes.
        let scales = &[1., 1., 1., 4.];

        struct Case {
            mode: NearestMode,

            // Expected output after nearest resizing using `mode` and the
            // "asymmetric" output => input coord transform. This coord transform
            // is used because it is the simplest (input_coord = output_coord / scale).
            expected: Tensor,
        }

        let cases = [
            Case {
                mode: NearestMode::Ceil,
                expected: Tensor::from_data(
                    &[1, 1, 1, 8],
                    vec![0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                ),
            },
            Case {
                mode: NearestMode::Floor,
                expected: Tensor::from_data(
                    &[1, 1, 1, 8],
                    vec![0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
                ),
            },
            Case {
                mode: NearestMode::RoundPreferCeil,
                expected: Tensor::from_data(
                    &[1, 1, 1, 8],
                    vec![0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                ),
            },
            Case {
                mode: NearestMode::RoundPreferFloor,
                expected: Tensor::from_data(
                    &[1, 1, 1, 8],
                    vec![0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2],
                ),
            },
        ];

        let pool = new_pool();
        for case in cases {
            let result = resize(
                &pool,
                image.view(),
                ResizeTarget::Scales(scales.into()),
                ResizeMode::Nearest,
                CoordTransformMode::Asymmetric,
                case.mode,
            )
            .unwrap();

            expect_equal(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_resize_bilinear() -> Result<(), Box<dyn Error>> {
        struct Case<'a> {
            image: NdTensorView<'a, f32, 4>,
            scales: Vec<f32>,
            expected: Tensor,
            coord_transform_mode: Option<CoordTransformMode>,
        }

        let image = NdTensor::from([0.2, 0.7, 0.3, 0.8]).into_shape([1, 1, 2, 2]);
        let image = image.view();

        let cases = [
            // Scale width and height by 0x
            Case {
                image,
                scales: vec![1., 1., 0., 0.],
                coord_transform_mode: None,
                expected: Tensor::from_data(&[1, 1, 0, 0], vec![]),
            },
            // Scale width and height by 0.5x
            Case {
                image,
                scales: vec![1., 1., 0.5, 0.5],
                coord_transform_mode: None,

                // OpenCV and PyTorch produce different results for this case.
                // This result matches OpenCV. This relates to the `half_pixel`
                // vs `pytorch_half_pixel` values for the `coordinate_transformation_mode`
                // attribute in the ONNX op.
                expected: Tensor::from_data(&[1, 1, 1, 1], vec![0.5]),
            },
            // Scale width and height by 1x
            Case {
                image,
                scales: vec![1., 1., 1., 1.],
                coord_transform_mode: None,
                expected: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
            },
            // Scale width and height by 1.5x
            Case {
                image,
                scales: vec![1., 1., 1.5, 1.5],
                coord_transform_mode: None,
                expected: Tensor::from_data(
                    &[1, 1, 3, 3],
                    vec![
                        0.2, 0.45, 0.7, // Y=0
                        0.25, 0.5, 0.75, // Y=1
                        0.3, 0.55, 0.8, // Y=2
                    ],
                ),
            },
            // Scale width and height by 2x
            Case {
                image,
                scales: vec![1., 1., 2., 2.],
                coord_transform_mode: None,
                expected: Tensor::from_data(
                    &[1, 1, 4, 4],
                    vec![
                        0.2, 0.325, 0.575, 0.7, // Y=0
                        0.225, 0.35, 0.6, 0.725, // Y=1
                        0.275, 0.4, 0.65, 0.775, // Y=2
                        0.3, 0.425, 0.675, 0.8, // Y=3
                    ],
                ),
            },
            // Scale width and height by 2x, align corners.
            Case {
                image,
                scales: vec![1., 1., 2., 2.],
                coord_transform_mode: Some(CoordTransformMode::AlignCorners),

                // Generated with `torch.functional.nn.interpolate(x, scale_factor=2,
                // mode='bilinear', align_corners=True)`.
                expected: Tensor::from([
                    [0.2000, 0.3667, 0.5333, 0.7000],
                    [0.2333, 0.4000, 0.5667, 0.7333],
                    [0.2667, 0.4333, 0.6000, 0.7667],
                    [0.3000, 0.4667, 0.6333, 0.8000],
                ])
                .into_shape([1, 1, 4, 4].as_slice()),
            },
            // Scale width and height by 3x
            Case {
                image,
                scales: vec![1., 1., 3., 3.],
                coord_transform_mode: None,
                expected: Tensor::from_data(
                    &[1, 1, 6, 6],
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

        let pool = new_pool();
        for case in cases {
            let result = resize(
                &pool,
                case.image.as_dyn(),
                ResizeTarget::Scales(case.scales.as_slice().into()),
                ResizeMode::Linear,
                case.coord_transform_mode
                    .unwrap_or(CoordTransformMode::HalfPixel),
                NearestMode::Floor,
            )
            .unwrap();

            expect_eq_1e4(&result, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_resize_scales_sizes() {
        enum CaseOutput {
            Shape(Vec<usize>),
            Error(OpError),
        }

        struct Case {
            image: Tensor,
            scales: Option<Tensor>,
            sizes: Option<Tensor<i32>>,
            expected: CaseOutput,
        }

        let cases = [
            // Specify output size via `scales`
            Case {
                image: Tensor::from_data(&[1, 1, 1, 1], vec![1.]),
                scales: Some(Tensor::from([1., 1., 2., 2.])),
                sizes: None,
                expected: CaseOutput::Shape(vec![1, 1, 2, 2]),
            },
            // Specify output size via `sizes`
            Case {
                image: Tensor::from_data(&[1, 1, 1, 1], vec![1.]),
                scales: None,
                sizes: Some(Tensor::from([1, 1, 2, 2])),
                expected: CaseOutput::Shape(vec![1, 1, 2, 2]),
            },
            // Identity resize via `scales`
            Case {
                image: Tensor::from_data(&[1, 1, 1, 1], vec![1.]),
                scales: Some(Tensor::from([1., 1., 1., 1.])),
                sizes: None,
                expected: CaseOutput::Shape(vec![1, 1, 1, 1]),
            },
            // Identity resize via `sizes`
            Case {
                image: Tensor::from_data(&[1, 1, 1, 1], vec![1.]),
                scales: None,
                sizes: Some(Tensor::from([1, 1, 1, 1])),
                expected: CaseOutput::Shape(vec![1, 1, 1, 1]),
            },
            // At least one of `scales` or `sizes` must be provided
            Case {
                image: Tensor::from_data(&[1, 1, 1, 1], vec![1.]),
                scales: None,
                sizes: None,
                expected: CaseOutput::Error(OpError::MissingInputs),
            },
            // Test empty tensors are also treated as missing inputs, for
            // compatibility with PyTorch targeting ONNX opset < 13.
            Case {
                image: Tensor::from_data(&[1, 1, 1, 1], vec![1.]),
                scales: Some(Tensor::from_vec(vec![])),
                sizes: Some(Tensor::from_vec(vec![])),
                expected: CaseOutput::Error(OpError::MissingInputs),
            },
            // Invalid values for scales/sizes
            Case {
                image: Tensor::from_data(&[1, 1, 1, 1], vec![1.]),
                scales: Some(Tensor::from([1., 1., 1.])),
                sizes: None,
                expected: CaseOutput::Error(OpError::IncompatibleInputShapes(
                    "scales/sizes length should equal input rank",
                )),
            },
            Case {
                image: Tensor::from_data(&[1, 1, 1, 1], vec![1.]),
                scales: Some(Tensor::from([1., 1., -1., 1.])),
                sizes: None,
                expected: CaseOutput::Error(OpError::InvalidValue("scales/sizes must be positive")),
            },
            Case {
                image: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: Some(Tensor::from_data(&[1, 1, 2, 2], vec![1., 1., 3., 3.])),
                sizes: None,
                expected: CaseOutput::Error(OpError::InvalidValue("scales must have 1 dims")),
            },
            // Values for scales/sizes and input shapes which are legal according to the spec,
            // but not currently supported in our implementation.
            Case {
                image: Tensor::from_data(&[1, 1, 2, 2], vec![0.2, 0.7, 0.3, 0.8]),
                scales: Some(Tensor::from([2., 1., 3., 3.])),
                sizes: None,
                expected: CaseOutput::Error(OpError::UnsupportedValue(
                    "only height and width dimensions can be resized",
                )),
            },
            // 1D input, with identity scale
            Case {
                image: [1., 1.].into(),
                scales: Some(Tensor::from([1.])),
                sizes: None,
                expected: CaseOutput::Shape(vec![2]),
            },
            // 1D input, with non-identity scale. This is not currently supported.
            Case {
                image: [1., 1.].into(),
                scales: Some(Tensor::from([2.])),
                sizes: None,
                expected: CaseOutput::Error(OpError::InvalidValue("input must have 4 dims (NCHW)")),
            },
        ];

        let pool = new_pool();
        for case in cases {
            let op = Resize {
                mode: ResizeMode::Linear,
                ..Resize::default()
            };
            let inputs = vec![
                Some((&case.image).into()),
                None, // `roi`
                case.scales.as_ref().map(|t| t.into()),
                case.sizes.as_ref().map(|t| t.into()),
            ];
            let result = op.run(&pool, InputList::from_optional(&inputs));
            match (case.expected, result) {
                (CaseOutput::Shape(shape), Ok(out)) => {
                    assert_eq!(out[0].shape(), &shape);
                }
                (CaseOutput::Error(expected_err), Err(err)) => {
                    assert_eq!(err, expected_err);
                }
                (CaseOutput::Shape(_), Err(err)) => {
                    panic!("Expected output but got error {:?}", err);
                }
                (CaseOutput::Error(_), Ok(_)) => {
                    panic!("Expected error but got output");
                }
            }
        }
    }
}
