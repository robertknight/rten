use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

use crate::buffer_pool::BufferPool;
use crate::ops::{IntoOpResult, OpError, OpRunContext, Operator, OutputList};

/// Interpolate between `x0` and `x1` according to the `factor` in range [0, 1].
fn lerp(x0: f32, x1: f32, factor: f32) -> f32 {
    x0 + (x1 - x0) * factor
}

fn grid_sample(
    pool: &BufferPool,
    input: NdTensorView<f32, 4>,
    grid: NdTensorView<f32, 4>,
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

                // Scale sample coordinates to image size and subtract 0.5 so
                // that a grid coordinate of -1 maps to -0.5. The sampled pixels
                // would have coordinates of -1 and 0 with an interpolation
                // factor of 0.5. A grid coordinate of 1 maps to `in_w - 0.5`
                // and the sampled pixels would have coordinates of `in_w - 1`
                // and `in_w` with an interpolation factor of 0.5.
                let scaled_x = in_w as f32 * grid_x - 0.5;
                let scaled_y = in_h as f32 * grid_y - 0.5;

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
pub struct GridSample {}

impl Operator for GridSample {
    fn name(&self) -> &str {
        "GridSample"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as(0)?;
        let grid = ctx.inputs().require_as(1)?;
        grid_sample(ctx.pool(), input, grid).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;
    use rten_testing::TestCases;

    use super::grid_sample;
    use crate::ops::tests::{expect_eq_1e4, new_pool};
    use crate::ops::OpError;

    /// Increase the rank of a tensor by inserting leading 1-sized dimensions.
    trait IntoNDim<const N: usize> {
        /// Variant of `Self` with N dimensions.
        type Output;

        /// Insert leading 1-sized dimensions into the shape of `self` so that
        /// it has N dimensions.
        ///
        /// Panics if `self` already has more than N dimensions.
        fn into_ndim(self) -> Self::Output;
    }

    impl<T: Clone, const M: usize, const N: usize> IntoNDim<N> for NdTensor<T, M> {
        type Output = NdTensor<T, N>;

        fn into_ndim(self) -> Self::Output {
            assert!(N >= M);
            let new_dims = N - M;
            let shape = self.shape();
            let new_shape =
                std::array::from_fn(|d| if d < new_dims { 1 } else { shape[d - new_dims] });
            self.into_shape(new_shape)
        }
    }

    #[test]
    fn test_grid_sample() {
        #[derive(Debug)]
        struct Case {
            input: NdTensor<f32, 4>,
            grid: NdTensor<f32, 4>,
            expected: NdTensor<f32, 4>,
        }

        let row = NdTensor::from([0.1087, 0.9655]).into_ndim();
        let col = NdTensor::from([[0.1087], [0.9655]]).into_ndim();

        let cases = [
            // Grid point with center X coordinate.
            Case {
                input: row.clone(),
                grid: NdTensor::from([0., 0.]).into_ndim(),
                expected: NdTensor::from([0.5371]).into_ndim(),
            },
            // Grid point with minimum X coordinate.
            Case {
                input: row.clone(),
                grid: NdTensor::from([-1., 0.]).into_ndim(),
                expected: NdTensor::from([0.05435]).into_ndim(),
            },
            // Grid point with maximum X coordinate.
            Case {
                input: row.clone(),
                grid: NdTensor::from([1., 0.]).into_ndim(),
                expected: NdTensor::from([0.48275]).into_ndim(),
            },
            // Grid point with out of range X coordinate (-ve).
            Case {
                input: row.clone(),
                grid: NdTensor::from([-2., 0.]).into_ndim(),
                expected: NdTensor::from([0.]).into_ndim(),
            },
            // Grid point with out of range X coordinate (+ve).
            Case {
                input: row.clone(),
                grid: NdTensor::from([2., 0.]).into_ndim(),
                expected: NdTensor::from([0.]).into_ndim(),
            },
            // Grid point with center Y coordinate.
            Case {
                input: col.clone(),
                grid: NdTensor::from([0., 0.]).into_ndim(),
                expected: NdTensor::from([0.5371]).into_ndim(),
            },
            // Grid point with minimum Y coordinate.
            Case {
                input: col.clone(),
                grid: NdTensor::from([0., -1.]).into_ndim(),
                expected: NdTensor::from([0.05435]).into_ndim(),
            },
            // Grid point with maximum Y coordinate.
            Case {
                input: col.clone(),
                grid: NdTensor::from([0., 1.]).into_ndim(),
                expected: NdTensor::from([0.48275]).into_ndim(),
            },
            // Test case created with PyTorch's
            // `torch.nn.functional.grid_sample`
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
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let result = grid_sample(&pool, case.input.view(), case.grid.view()).unwrap();
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
            let pool = new_pool();
            let input = NdTensor::zeros(case.input_shape);
            let grid = NdTensor::zeros(case.grid_shape);
            let result = grid_sample(&pool, input.view(), grid.view());
            assert_eq!(result.err().as_ref(), Some(&case.expected));
        });
    }
}
