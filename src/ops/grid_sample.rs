use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, NdTensorViewMut, Tensor, TensorView};

use crate::ops::{static_dims, IntoOpResult, OpError, OpRunContext, Operator, OutputList};
use crate::tensor_pool::TensorPool;

/// Interpolation mode for GridSample operation.
#[derive(Copy, Clone, Debug, Default)]
pub enum GridSampleMode {
    #[default]
    Bilinear,
    Nearest,
    Bicubic,
}

/// Padding mode for out-of-bounds grid locations.
#[derive(Copy, Clone, Debug, Default)]
pub enum GridSamplePaddingMode {
    #[default]
    Zeros,
    Border,
    Reflection,
}

/// GridSample operator for 2D images.
///
/// Given an input tensor and a grid of sampling locations, produces an output
/// tensor by sampling the input at the grid locations using interpolation.
///
/// This implementation currently supports 2D images (4D tensors with shape [N, C, H, W]).
#[derive(Debug)]
pub struct GridSample {
    pub mode: GridSampleMode,
    pub padding_mode: GridSamplePaddingMode,
    pub align_corners: bool,
}

impl Default for GridSample {
    fn default() -> Self {
        Self {
            mode: GridSampleMode::Bilinear,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: false,
        }
    }
}

impl GridSample {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mode(mut self, mode: GridSampleMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn padding_mode(mut self, padding_mode: GridSamplePaddingMode) -> Self {
        self.padding_mode = padding_mode;
        self
    }

    pub fn align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }
}

impl Operator for GridSample {
    fn name(&self) -> &str {
        "GridSample"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as::<TensorView<f32>>(0)?;
        let grid = ctx.inputs().require_as::<TensorView<f32>>(1)?;

        // Ensure input is 4D [N, C, H, W]
        let input_4d = static_dims!(input, 4, "NCHW")?;

        // Ensure grid is 4D [N, H_out, W_out, 2]
        let grid_4d = static_dims!(grid, 4)?;
        if grid_4d.size(3) != 2 {
            return Err(OpError::InvalidValue(
                "Grid tensor must have 2 coordinates in last dimension",
            ));
        }

        let [batch_size, channels, _in_height, _in_width] = input_4d.shape();
        let [grid_batch_size, out_height, out_width, _] = grid_4d.shape();

        if batch_size != grid_batch_size {
            return Err(OpError::IncompatibleInputShapes(
                "Input and grid batch sizes must match",
            ));
        }

        let mut output =
            Tensor::zeros_in(ctx.pool(), &[batch_size, channels, out_height, out_width]);

        let mut output_4d = output.nd_view_mut::<4>();

        // Process each batch item in parallel
        output_4d
            .axis_iter_mut(0)
            .into_par_iter()
            .zip(input_4d.axis_iter(0).into_par_iter())
            .zip(grid_4d.axis_iter(0).into_par_iter())
            .for_each(|((mut output_batch, input_batch), grid_batch)| {
                grid_sample_2d(
                    &mut output_batch,
                    &input_batch,
                    &grid_batch,
                    self.mode,
                    self.padding_mode,
                    self.align_corners,
                );
            });

        Ok(output.into_op_result()?)
    }
}

/// Perform grid sampling for a single batch item.
fn grid_sample_2d(
    output: &mut NdTensorViewMut<f32, 3>, // [C, H_out, W_out]
    input: &NdTensorView<f32, 3>,         // [C, H_in, W_in]
    grid: &NdTensorView<f32, 3>,          // [H_out, W_out, 2]
    mode: GridSampleMode,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) {
    let [channels, _, _] = input.shape();
    let [out_height, out_width, _] = grid.shape();

    // Process each output pixel
    for y_out in 0..out_height {
        for x_out in 0..out_width {
            // Get normalized grid coordinates [-1, 1]
            let grid_x = grid[[y_out, x_out, 0]];
            let grid_y = grid[[y_out, x_out, 1]];

            // Convert normalized coordinates to input tensor coordinates
            let (input_x, input_y) = denormalize_coordinates(
                grid_x,
                grid_y,
                input.size(2) as f32, // W_in
                input.size(1) as f32, // H_in
                align_corners,
            );

            // Sample each channel
            for c in 0..channels {
                let channel_input = input.slice((c, .., ..));
                let sampled_value =
                    sample_pixel(&channel_input, input_x, input_y, mode, padding_mode);
                output[[c, y_out, x_out]] = sampled_value;
            }
        }
    }
}

/// Convert normalized coordinates [-1, 1] to input tensor coordinates.
fn denormalize_coordinates(
    norm_x: f32,
    norm_y: f32,
    width: f32,
    height: f32,
    align_corners: bool,
) -> (f32, f32) {
    let input_x = if align_corners {
        0.5 * ((norm_x + 1.0) * (width - 1.0))
    } else {
        0.5 * ((norm_x + 1.0) * width - 1.0)
    };

    let input_y = if align_corners {
        0.5 * ((norm_y + 1.0) * (height - 1.0))
    } else {
        0.5 * ((norm_y + 1.0) * height - 1.0)
    };

    (input_x, input_y)
}

/// Sample a pixel value from the input tensor at the given coordinates.
fn sample_pixel(
    input: &NdTensorView<f32, 2>, // [H, W]
    x: f32,
    y: f32,
    mode: GridSampleMode,
    padding_mode: GridSamplePaddingMode,
) -> f32 {
    match mode {
        GridSampleMode::Nearest => sample_nearest(input, x, y, padding_mode),
        GridSampleMode::Bilinear => sample_bilinear(input, x, y, padding_mode),
        GridSampleMode::Bicubic => sample_bicubic(input, x, y, padding_mode),
    }
}

/// Sample using nearest neighbor interpolation.
fn sample_nearest(
    input: &NdTensorView<f32, 2>,
    x: f32,
    y: f32,
    padding_mode: GridSamplePaddingMode,
) -> f32 {
    let x_nearest = x.round() as i32;
    let y_nearest = y.round() as i32;

    get_pixel_value(input, x_nearest, y_nearest, padding_mode)
}

/// Sample using bilinear interpolation.
fn sample_bilinear(
    input: &NdTensorView<f32, 2>,
    x: f32,
    y: f32,
    padding_mode: GridSamplePaddingMode,
) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    let v00 = get_pixel_value(input, x0, y0, padding_mode);
    let v01 = get_pixel_value(input, x0, y1, padding_mode);
    let v10 = get_pixel_value(input, x1, y0, padding_mode);
    let v11 = get_pixel_value(input, x1, y1, padding_mode);

    let v0 = lerp(v00, v10, wx);
    let v1 = lerp(v01, v11, wx);
    lerp(v0, v1, wy)
}

/// Sample using bicubic interpolation.
fn sample_bicubic(
    input: &NdTensorView<f32, 2>,
    x: f32,
    y: f32,
    padding_mode: GridSamplePaddingMode,
) -> f32 {
    let x_floor = x.floor() as i32;
    let y_floor = y.floor() as i32;

    let wx = x - x_floor as f32;
    let wy = y - y_floor as f32;

    // Bicubic interpolation using 4x4 neighborhood
    let mut result = 0.0;
    for j in 0..4 {
        let mut row_result = 0.0;
        for i in 0..4 {
            let px = x_floor + i - 1;
            let py = y_floor + j - 1;
            let pixel_value = get_pixel_value(input, px, py, padding_mode);
            row_result += pixel_value * cubic_weight(wx - (i - 1) as f32);
        }
        result += row_result * cubic_weight(wy - (j - 1) as f32);
    }

    result
}

/// Bicubic interpolation weight function.
fn cubic_weight(t: f32) -> f32 {
    let a = -0.5;
    let abs_t = t.abs();

    if abs_t <= 1.0 {
        (a + 2.0) * abs_t.powi(3) - (a + 3.0) * abs_t.powi(2) + 1.0
    } else if abs_t <= 2.0 {
        a * abs_t.powi(3) - 5.0 * a * abs_t.powi(2) + 8.0 * a * abs_t - 4.0 * a
    } else {
        0.0
    }
}

/// Get pixel value with boundary handling.
fn get_pixel_value(
    input: &NdTensorView<f32, 2>,
    x: i32,
    y: i32,
    padding_mode: GridSamplePaddingMode,
) -> f32 {
    let height = input.size(0) as i32;
    let width = input.size(1) as i32;

    let (px, py) = match padding_mode {
        GridSamplePaddingMode::Zeros => {
            if x < 0 || x >= width || y < 0 || y >= height {
                return 0.0;
            }
            (x, y)
        }
        GridSamplePaddingMode::Border => {
            let px = x.clamp(0, width - 1);
            let py = y.clamp(0, height - 1);
            (px, py)
        }
        GridSamplePaddingMode::Reflection => {
            let px = if x < 0 {
                (-x).min(width - 1)
            } else if x >= width {
                (2 * width - 2 - x).max(0)
            } else {
                x
            };

            let py = if y < 0 {
                (-y).min(height - 1)
            } else if y >= height {
                (2 * height - 2 - y).max(0)
            } else {
                y
            };
            (px, py)
        }
    };

    input[[py as usize, px as usize]]
}

/// Interpolate between `a` and `b` according to `weight`.
fn lerp(a: f32, b: f32, weight: f32) -> f32 {
    (1.0 - weight) * a + weight * b
}

/// Standalone function for grid sampling.
pub fn grid_sample(
    input: TensorView<f32>,
    grid: TensorView<f32>,
    mode: GridSampleMode,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
    pool: &TensorPool,
) -> Result<Tensor<f32>, OpError> {
    let op = GridSample {
        mode,
        padding_mode,
        align_corners,
    };
    let inputs = (input, grid).into();
    let ctx = crate::ops::OpRunContext::new(pool, &inputs);
    let mut outputs = op.run(&ctx)?;
    outputs.remove(0).try_into().map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::Tensor;

    use crate::ops::grid_sample::{grid_sample, GridSampleMode, GridSamplePaddingMode};
    use crate::ops::tests::new_pool;

    #[test]
    fn test_grid_sample_identity() {
        let pool = new_pool();

        // Create a simple 2x2 input image
        let mut input = Tensor::from([1.0, 2.0, 3.0, 4.0]);
        input.reshape(&[1, 1, 2, 2]);

        // Identity grid (maps each output pixel to the same input location)
        let mut grid = Tensor::from([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        grid.reshape(&[1, 2, 2, 2]);

        let result = grid_sample(
            input.view(),
            grid.view(),
            GridSampleMode::Nearest,
            GridSamplePaddingMode::Zeros,
            true,
            &pool,
        )
        .unwrap();

        let mut expected = Tensor::from([1.0, 2.0, 3.0, 4.0]);
        expected.reshape(&[1, 1, 2, 2]);

        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    fn test_grid_sample_bilinear() {
        let pool = new_pool();

        // Create a simple input
        let mut input = Tensor::from([0.0, 1.0, 2.0, 3.0]);
        input.reshape(&[1, 1, 2, 2]);

        // Grid that samples at center points
        let mut grid = Tensor::from([0.0, 0.0]);
        grid.reshape(&[1, 1, 1, 2]);

        let result = grid_sample(
            input.view(),
            grid.view(),
            GridSampleMode::Bilinear,
            GridSamplePaddingMode::Zeros,
            false,
            &pool,
        )
        .unwrap();

        // Should interpolate between all four corners: (0+1+2+3)/4 = 1.5
        let mut expected = Tensor::from([1.5]);
        expected.reshape(&[1, 1, 1, 1]);

        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    fn test_grid_sample_out_of_bounds() {
        let pool = new_pool();

        let mut input = Tensor::from([1.0, 2.0, 3.0, 4.0]);
        input.reshape(&[1, 1, 2, 2]);

        // Grid with out-of-bounds coordinates
        let mut grid = Tensor::from([2.0, 2.0]);
        grid.reshape(&[1, 1, 1, 2]);

        let result = grid_sample(
            input.view(),
            grid.view(),
            GridSampleMode::Nearest,
            GridSamplePaddingMode::Zeros,
            false,
            &pool,
        )
        .unwrap();

        // Should return 0 for out-of-bounds with zeros padding
        let mut expected = Tensor::from([0.0]);
        expected.reshape(&[1, 1, 1, 1]);

        expect_equal(&result, &expected).unwrap();
    }
}
