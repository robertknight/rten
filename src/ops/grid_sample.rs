use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

use crate::buffer_pool::BufferPool;
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};

/// Interpolate between `x0` and `x1` according to the `factor` in range [0, 1].
fn lerp(x0: f32, x1: f32, factor: f32) -> f32 {
    x0 + (x1 - x0) * factor
}

fn grid_sample(
    pool: &BufferPool,
    input: NdTensorView<f32, 4>,
    grid: NdTensorView<f32, 4>,
    align_corners: bool,
) -> Result<NdTensor<f32, 4>, OpError> {
    let [batch, h_out, w_out, coord_ndim] = grid.shape();
    let [in_batch, in_c, in_h, in_w] = input.shape();

    if batch != in_batch {
        return Err(OpError::IncompatibleInputShapes(
            "Batch size of input and grid must match",
        ));
    }

    if coord_ndim != 2 {
        return Err(OpError::UnsupportedValue(
            "Unsupported grid coordinate size",
        ));
    }

    let out_shape = [batch, in_c, h_out, w_out];

    if in_h == 0 || in_w == 0 {
        // If input is empty, all grid coordinates will be out of bounds.
        return Ok(NdTensor::zeros(out_shape));
    }

    let mut output = NdTensor::uninit_in(pool, out_shape);

    for n in 0..batch {
        let grid = grid.slice(n);
        let input = input.slice(n);
        let mut output = output.slice_mut(n);

        for y in 0..h_out {
            for x in 0..w_out {
                // Get sample coordinates in the range [-1, 1].
                let grid_x = grid[[y, x, 0]];
                let grid_y = grid[[y, x, 1]];

                // Scale sample coordinates to [0, 1]
                let grid_x = (grid_x + 1.) * 0.5;
                let grid_y = (grid_y + 1.) * 0.5;

                // For `align_corners=false`, the extrema coordinates (-1, 1)
                // refer to the corners of the input (eg. 0, in_w on the X
                // axis). If true, they refer to the pixel centers (0.5, in_w -
                // 0.5)
                //
                // For the default case of `align_corners=false`, scale sample
                // coordinates to image size and subtract 0.5 so that a grid
                // coordinate of -1 maps to -0.5. The sampled pixels would have
                // coordinates of -1 and 0 with an interpolation factor of 0.5.
                // A grid coordinate of 1 maps to `in_w - 0.5` and the sampled
                // pixels would have coordinates of `in_w - 1` and `in_w` with
                // an interpolation factor of 0.5.
                //
                // For `align_corners=true`, the sample coordinates are shifted
                // by +0.5 at the start of the axis and -0.5 at the end.
                let scaled_x = if align_corners {
                    (in_w as f32 - 1.0) * grid_x
                } else {
                    in_w as f32 * grid_x - 0.5
                };
                let scaled_y = if align_corners {
                    (in_h as f32 - 1.0) * grid_y
                } else {
                    in_h as f32 * grid_y - 0.5
                };

                // Compute coordinates of the 4 pixels to sample and the
                // interpolation factor along each axis.
                let x_lerp = scaled_x - scaled_x.floor();
                let in_x = scaled_x.floor() as i32;
                let y_lerp = scaled_y - scaled_y.floor();
                let in_y = scaled_y.floor() as i32;

                for c in 0..in_c {
                    let get_pixel = |y: i32, x: i32| {
                        if y < 0 || y >= in_h as i32 || x < 0 || x >= in_w as i32 {
                            // Out of bounds coordinates are sampled as zero.
                            0.
                        } else {
                            // Safety: c, y and x are all in-bounds here.
                            unsafe { *input.get_unchecked([c, y as usize, x as usize]) }
                        }
                    };

                    let y0x0 = get_pixel(in_y, in_x);
                    let y0x1 = get_pixel(in_y, in_x + 1);
                    let y1x0 = get_pixel(in_y + 1, in_x);
                    let y1x1 = get_pixel(in_y + 1, in_x + 1);
                    let y0 = lerp(y0x0, y0x1, x_lerp);
                    let y1 = lerp(y1x0, y1x1, x_lerp);
                    let val = lerp(y0, y1, y_lerp);

                    // Safety: [c, y, x] coordinates are all in bounds.
                    unsafe {
                        output.get_unchecked_mut([c, y, x]).write(val);
                    }
                }
            }
        }
    }

    // Safety: We initialized all output values.
    Ok(unsafe { output.assume_init() })
}

#[derive(Debug)]
pub struct GridSample {
    pub align_corners: bool,
}

impl Operator for GridSample {
    fn name(&self) -> &str {
        "GridSample"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as(0)?;
        let grid = ctx.inputs().require_as(1)?;
        grid_sample(ctx.pool(), input, grid, self.align_corners).into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;
    use rten_testing::TestCases;

    use super::grid_sample;
    use crate::buffer_pool::BufferPool;
    use crate::operator::OpError;
    use crate::ops::tests::{IntoNDim, expect_eq_1e4};

    #[test]
    fn test_grid_sample() {
        #[derive(Debug)]
        struct Case {
            input: NdTensor<f32, 4>,
            grid: NdTensor<f32, 4>,
            expected: NdTensor<f32, 4>,
            align_corners: bool,
        }

        let row = NdTensor::from([0.1087, 0.9655]).into_ndim();
        let col = NdTensor::from([[0.1087], [0.9655]]).into_ndim();

        let cases = [
            // Grid point with center X coordinate.
            Case {
                input: row.clone(),
                grid: NdTensor::from([0., 0.]).into_ndim(),
                expected: NdTensor::from([0.5371]).into_ndim(),
                align_corners: false,
            },
            // Grid point with minimum X coordinate.
            Case {
                input: row.clone(),
                grid: NdTensor::from([-1., 0.]).into_ndim(),
                expected: NdTensor::from([0.05435]).into_ndim(),
                align_corners: false,
            },
            // Grid point with maximum X coordinate.
            Case {
                input: row.clone(),
                grid: NdTensor::from([1., 0.]).into_ndim(),
                expected: NdTensor::from([0.48275]).into_ndim(),
                align_corners: false,
            },
            // Grid point with out of range X coordinate (-ve).
            Case {
                input: row.clone(),
                grid: NdTensor::from([-2., 0.]).into_ndim(),
                expected: NdTensor::from([0.]).into_ndim(),
                align_corners: false,
            },
            // Grid point with out of range X coordinate (+ve).
            Case {
                input: row.clone(),
                grid: NdTensor::from([2., 0.]).into_ndim(),
                expected: NdTensor::from([0.]).into_ndim(),
                align_corners: false,
            },
            // Grid point with center Y coordinate.
            Case {
                input: col.clone(),
                grid: NdTensor::from([0., 0.]).into_ndim(),
                expected: NdTensor::from([0.5371]).into_ndim(),
                align_corners: false,
            },
            // Grid point with minimum Y coordinate.
            Case {
                input: col.clone(),
                grid: NdTensor::from([0., -1.]).into_ndim(),
                expected: NdTensor::from([0.05435]).into_ndim(),
                align_corners: false,
            },
            // Grid point with maximum Y coordinate.
            Case {
                input: col.clone(),
                grid: NdTensor::from([0., 1.]).into_ndim(),
                expected: NdTensor::from([0.48275]).into_ndim(),
                align_corners: false,
            },
            // Test case for align_corners=false, created with PyTorch's
            // `torch.nn.functional.grid_sample`.
            Case {
                input: NdTensor::from([
                    [0.9942, 0.4255, 0.9730, 0.5230],
                    [0.8417, 0.1245, 0.2245, 0.0774],
                    [0.9674, 0.5163, 0.3541, 0.0016],
                    [0.7593, 0.0594, 0.8754, 0.1339],
                ])
                .into_ndim(),
                grid: NdTensor::from([
                    [[0.3389, 0.0883], [0.9822, 0.6967], [0.3037, 0.8579]],
                    [[0.4092, 0.4664], [0.4346, 0.3142], [0.3880, 0.4060]],
                    [[0.0835, 0.1432], [0.5129, 0.7989], [0.2861, 0.7945]],
                ])
                .into_ndim(),
                expected: NdTensor::from([
                    [0.2613, 0.0642, 0.6241],
                    [0.4138, 0.2725, 0.3860],
                    [0.3618, 0.4381, 0.7487],
                ])
                .into_ndim(),
                align_corners: false,
            },
            // Test case for align_corners=true, created with PyTorch's
            // `torch.nn.functional.grid_sample`.
            Case {
                input: NdTensor::from([
                    [0.9942, 0.4255, 0.9730, 0.5230],
                    [0.8417, 0.1245, 0.2245, 0.0774],
                    [0.9674, 0.5163, 0.3541, 0.0016],
                    [0.7593, 0.0594, 0.8754, 0.1339],
                ])
                .into_ndim(),
                grid: NdTensor::from([
                    [[0.3389, 0.0883], [0.9822, 0.6967], [0.3037, 0.8579]],
                    [[0.4092, 0.4664], [0.4346, 0.3142], [0.3880, 0.4060]],
                    [[0.0835, 0.1432], [0.5129, 0.7989], [0.2861, 0.7945]],
                ])
                .into_ndim(),
                expected: NdTensor::from([
                    [0.3042, 0.0888, 0.7373],
                    [0.4092, 0.2977, 0.3785],
                    [0.3499, 0.5500, 0.6783],
                ])
                .into_ndim(),
                align_corners: true,
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = grid_sample(
                &pool,
                case.input.view(),
                case.grid.view(),
                case.align_corners,
            )
            .unwrap();
            expect_eq_1e4(&result, &case.expected).unwrap();
        });
    }

    #[test]
    fn test_grid_sample_invalid() {
        #[derive(Debug)]
        struct Case {
            input_shape: [usize; 4],
            grid_shape: [usize; 4],
            expected: OpError,
        }

        let cases = [
            Case {
                input_shape: [1, 1, 1, 1],
                grid_shape: [2, 1, 1, 2],
                expected: OpError::IncompatibleInputShapes(
                    "Batch size of input and grid must match",
                ),
            },
            Case {
                input_shape: [1, 1, 1, 1],
                grid_shape: [1, 1, 1, 3],
                expected: OpError::UnsupportedValue("Unsupported grid coordinate size"),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let input = NdTensor::zeros(case.input_shape);
            let grid = NdTensor::zeros(case.grid_shape);
            let align_corners = false;
            let result = grid_sample(&pool, input.view(), grid.view(), align_corners);
            assert_eq!(result.err().as_ref(), Some(&case.expected));
        });
    }
}
