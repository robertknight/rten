use fastrand::Rng;
use fastrand_contrib::RngExt;
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use crate::ops::{IntoOpResult, OpError, OpRunContext, Operator, OutputList, Value};

#[derive(Debug)]
pub struct RandomUniform {
    pub low: f32,
    pub high: f32,
    pub shape: Vec<usize>,

    /// Random seed.
    ///
    /// This unusually uses an `f32` value for consistency with the ONNX
    /// specification.
    pub seed: Option<f32>,
}

impl Operator for RandomUniform {
    fn name(&self) -> &str {
        "RandomUniform"
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let scale_value = |val: f32| self.low + val * (self.high - self.low);
        let shape = self.shape.as_slice();

        let mut rng = if let Some(seed) = self.seed {
            Rng::with_seed(seed.to_bits() as u64)
        } else {
            Rng::new()
        };
        Tensor::from_simple_fn_in(ctx.pool(), shape, || scale_value(rng.f32())).into_op_result()
    }
}

#[derive(Debug)]
pub struct RandomUniformLike {
    pub low: f32,
    pub high: f32,

    /// Random seed. See [`RandomUniform::seed`].
    pub seed: Option<f32>,
}

impl Operator for RandomUniformLike {
    fn name(&self) -> &str {
        "RandomUniformLike"
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let op = RandomUniform {
            low: self.low,
            high: self.high,
            seed: self.seed,
            shape: input.shape().to_vec(),
        };
        op.run(ctx)
    }
}

#[derive(Debug)]
pub struct RandomNormal {
    pub mean: f32,
    pub scale: f32,
    pub shape: Vec<usize>,

    /// Random seed.
    ///
    /// This unusually uses an `f32` value for consistency with the ONNX
    /// specification.
    pub seed: Option<f32>,
}

impl Operator for RandomNormal {
    fn name(&self) -> &str {
        "RandomNormal"
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let shape = self.shape.as_slice();

        let mut rng = if let Some(seed) = self.seed {
            Rng::with_seed(seed.to_bits() as u64)
        } else {
            Rng::new()
        };

        Tensor::from_simple_fn_in(ctx.pool(), shape, || rng.f32_normal(self.mean, self.scale))
            .into_op_result()
    }
}

#[derive(Debug)]
pub struct RandomNormalLike {
    pub mean: f32,
    pub scale: f32,

    /// Random seed. See [`RandomUniform::seed`].
    pub seed: Option<f32>,
}

impl Operator for RandomNormalLike {
    fn name(&self) -> &str {
        "RandomNormalLike"
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let op = RandomNormal {
            mean: self.mean,
            scale: self.scale,
            seed: self.seed,
            shape: input.shape().to_vec(),
        };
        op.run(ctx)
    }
}

#[derive(Debug)]
pub struct Dropout {
    pub seed: Option<i32>,
}

impl Operator for Dropout {
    fn name(&self) -> &str {
        "Dropout"
    }

    fn is_deterministic(&self) -> bool {
        self.seed.is_some()
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input: TensorView<f32> = inputs.require_as(0)?;

        // The spec (https://onnx.ai/onnx/operators/onnx__Dropout.html) says "If
        // this input was not set, or if it was set to 0, the output would be a
        // simple copy of the input", implying the default should be 0, but also
        // "It is an optional value, if not specified it will default to 0.5.".
        // The latter sentence matches the reference and ONNX Runtime.
        let ratio = inputs.get_as(1)?.unwrap_or(0.5);
        #[allow(clippy::manual_range_contains)]
        if ratio < 0. || ratio >= 1.0 {
            return Err(OpError::InvalidValue("ratio must be in the range [0, 1)"));
        }

        let training_mode = inputs.get_as::<i32>(2)?.unwrap_or(0) != 0;

        let (output, mask) = if !training_mode || ratio == 0. {
            let mask = Tensor::<i32>::full(input.shape(), 1);
            (input.to_tensor(), mask)
        } else {
            let mut rng = if let Some(seed) = self.seed {
                Rng::with_seed(seed as u64)
            } else {
                Rng::new()
            };
            let scale = 1. / (1. - ratio);

            let mask =
                Tensor::<i32>::from_simple_fn(
                    input.shape(),
                    || if rng.f32() < ratio { 0 } else { 1 },
                );
            let input = input.to_contiguous_in(ctx.pool());

            let output = Tensor::from_data(
                input.shape(),
                input
                    .data()
                    .unwrap()
                    .iter()
                    .zip(mask.data().unwrap())
                    .map(|(&x, &mask)| x * scale * mask as f32)
                    .collect::<Vec<_>>(),
            );
            (output, mask)
        };

        Ok([Value::from(output), Value::from(mask)]
            .into_iter()
            .collect())
    }

    // Ideally this operator should support running in place if:
    //
    // 1. The mask output is unused
    // 2. `ratio` is zero or `training_mode` is false, in which case the output
    //    is a copy of the input
    //
    // Operators currently do not have a way to check if an output is unused, so
    // we can't check condition (1).
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use crate::ops::operators::{FloatOperators, Operators};
    use crate::ops::tests::new_pool;
    use crate::ops::{InputList, OpRunContext, Operator, OperatorExt};

    use super::{Dropout, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike};

    #[test]
    fn test_random_uniform() {
        #[derive(Clone, Debug)]
        struct Case {
            low: f32,
            high: f32,
            shape: Vec<usize>,
            seed: Option<f32>,
        }

        let cases = [
            // Standard value range.
            Case {
                low: 0.,
                high: 1.,
                shape: vec![50, 50],
                seed: None,
            },
            // Non-standard low/high ranges.
            Case {
                low: -5.,
                high: -1.,
                shape: vec![50, 50],
                seed: None,
            },
            Case {
                low: 1.,
                high: 5.,
                shape: vec![50, 50],
                seed: None,
            },
            // Custom seed
            Case {
                low: 0.,
                high: 1.,
                shape: vec![50, 50],
                seed: Some(0.5),
            },
        ];

        cases.test_each_clone(|Case {
            low,
            high,
            shape,
            seed,
        }|
        {
            let op = RandomUniform {
                low,
                high,
                shape,
                seed,
            };
            let output: Tensor = op.run_simple(InputList::new()).unwrap();

            assert_eq!(output.shape(), op.shape);

            // Create buckets to count elements in N sub-intervals of
            // `[op.low, op.high]`.
            let mut buckets = vec![0; 10];
            let bucket_size = (op.high - op.low) as f32 / buckets.len() as f32;

            // Test generated outputs are within expected range.
            for el in output.iter().copied() {
                let low = op.low;
                let high = op.high;
                assert!(
                    el >= low && el <= high,
                    "value {el} outside range {low}..{high}"
                );

                let bucket_idx = ((el - low) / bucket_size) as usize;
                buckets[bucket_idx] += 1;
            }

            // Check that distribution is approximately uniform. A more
            // principled approach would be to do a chi-squared test.
            let expected_count_per_bucket = (output.len() / buckets.len()) as i32;
            let max_expected_count_diff = buckets
                .iter()
                .map(|count| (count - expected_count_per_bucket).abs())
                .max()
                .unwrap();
            let tolerance = (expected_count_per_bucket as f32) * 0.3;
            assert!(
                (max_expected_count_diff as f32) <= tolerance,
                "max deviation from expected bucket size {max_expected_count_diff} > tolerance {tolerance}"
            );

            // Test that repeated generation produces the same output if the
            // seed is fixed, or different output otherwise.
            let output_2: Tensor = op.run_simple(InputList::new()).unwrap();
            if let Some(_seed) = seed {
                assert_eq!(output, output_2);
            } else {
                assert_ne!(output, output_2);
            }
        })
    }

    #[test]
    fn test_random_uniform_like() {
        let input = Tensor::<f32>::zeros(&[5, 5]);
        let op = RandomUniformLike {
            low: 0.,
            high: 2.,
            seed: None,
        };
        let output: Tensor<f32> = op.run_simple(input.view()).unwrap();
        assert_eq!(output.shape(), &[5, 5]);
    }

    #[test]
    fn test_random_normal() {
        #[derive(Clone, Debug)]
        struct Case {
            mean: f32,
            scale: f32,
            shape: Vec<usize>,
            seed: Option<f32>,
        }

        let cases = [
            // Default mean/scale values.
            Case {
                mean: 0.,
                scale: 1.,
                shape: vec![10, 10],
                seed: None,
            },
            // Custom mean/scale values.
            Case {
                mean: 5.,
                scale: 0.5,
                shape: vec![10, 10],
                seed: None,
            },
            // Custom seed
            Case {
                mean: 0.,
                scale: 1.,
                shape: vec![10, 10],
                seed: Some(0.5),
            },
        ];

        cases.test_each_clone(|case| {
            let Case {
                mean,
                scale,
                shape,
                seed,
            } = case;

            let op = RandomNormal {
                mean,
                scale,
                shape,
                seed,
            };
            let output: Tensor = op.run_simple(InputList::new()).unwrap();
            assert_eq!(output.shape(), op.shape);

            // Test that outputs have expected distribution.
            let mean = output
                .reduce_mean(None, false /* keep_dims */)
                .unwrap()
                .item()
                .copied()
                .unwrap();
            let delta = (mean - op.mean).abs();

            // nb. This threshold is large because we're only generating a small
            // number of values.
            let threshold = 0.5;
            assert!(delta <= threshold, "delta {delta} > {threshold}");

            let var: f32 = output
                .map(|x| (x - mean) * (x - mean))
                .reduce_sum(None, false /* keep_dims */)
                .unwrap()
                .item()
                .unwrap()
                / output.len() as f32;
            let std_dev = var.sqrt();
            let delta = (std_dev - op.scale).abs();
            assert!(delta <= threshold, "delta {delta} > {threshold}");

            // Test that repeated generation produces the same output if the
            // seed is fixed, or different output otherwise.
            let output_2: Tensor = op.run_simple(InputList::new()).unwrap();
            if let Some(_seed) = seed {
                assert_eq!(output, output_2);
            } else {
                assert_ne!(output, output_2);
            }
        })
    }

    #[test]
    fn test_random_normal_like() {
        let input = Tensor::<f32>::zeros(&[5, 5]);
        let op = RandomNormalLike {
            mean: 0.,
            scale: 5.,
            seed: None,
        };
        let output: Tensor<f32> = op.run_simple(input.view()).unwrap();
        assert_eq!(output.shape(), &[5, 5]);
    }

    #[test]
    fn test_dropout_noop() {
        #[derive(Debug)]
        struct Case {
            ratio: Option<f32>,
            training_mode: Option<bool>,
        }

        let cases = [
            // No ratio or training_mode. training_mode defaults to false.
            Case {
                ratio: None,
                training_mode: None,
            },
            // Dropout ratio of zero.
            Case {
                ratio: Some(0.),
                training_mode: Some(true),
            },
            // Non-zero dropout, but training mode is disabled.
            Case {
                ratio: Some(0.5),
                training_mode: Some(false),
            },
        ];

        cases.test_each(|case| {
            let data = Tensor::from([[1., 2.], [3., 4.]]);
            let ratio_input = case.ratio.map(Tensor::from);
            let training_mode_input = case
                .training_mode
                .map(|tm| Tensor::from(if tm { 1i32 } else { 0 }));

            let op = Dropout { seed: None };
            let mut inputs = InputList::new();
            inputs.push(&data);
            inputs.push_optional(ratio_input.as_ref());
            inputs.push_optional(training_mode_input.as_ref());
            let pool = new_pool();
            let ctx = OpRunContext::new(&pool, &inputs);
            let mut outputs = op.run(&ctx).unwrap();
            let output: Tensor<f32> = outputs.remove(0).try_into().unwrap();
            assert_eq!(output, data);

            let mask: Tensor<i32> = outputs.remove(0).try_into().unwrap();
            assert_eq!(mask, Tensor::full(data.shape(), 1));
        });
    }

    #[test]
    fn test_dropout_active() {
        #[derive(Debug)]
        struct Case {
            ratio: Option<f32>,
            expected_dropout_ratio: f32, // Expected ratio in [0, 1]
            tolerance: f32,              // Tolerance for comparing actual vs expected
                                         // dropout ratio
        }

        let cases = [
            Case {
                // The spec disallows setting the dropout ratio to exactly 1.
                ratio: Some(0.99999),
                expected_dropout_ratio: 1.,
                tolerance: 0.,
            },
            Case {
                ratio: None, // Default ratio is 0.5
                expected_dropout_ratio: 0.5,
                tolerance: 0.1,
            },
        ];

        cases.test_each(|case| {
            let data = Tensor::full(&[10, 10], 1.0);
            let ratio_input = case.ratio.map(Tensor::from);
            let training_mode_input = Tensor::from(1i32);

            let op = Dropout {
                // Seed a fixed seed for consistent results
                seed: Some(1),
            };
            let mut inputs = InputList::new();
            inputs.push(&data);
            inputs.push_optional(ratio_input.as_ref());
            inputs.push(&training_mode_input);
            let pool = new_pool();
            let ctx = OpRunContext::new(&pool, &inputs);

            let mut outputs = op.run(&ctx).unwrap();
            let output: Tensor<f32> = outputs.remove(0).try_into().unwrap();
            let dropout_ratio =
                output.iter().filter(|x| **x == 0.0).count() as f32 / data.len() as f32;
            assert!(
                (dropout_ratio - case.expected_dropout_ratio).abs() <= case.tolerance,
                "dropout ratio {} is not close enough to {}",
                dropout_ratio,
                case.expected_dropout_ratio
            );

            let mask: Tensor<i32> = outputs.remove(0).try_into().unwrap();
            let mask_dropout_ratio =
                mask.iter().filter(|x| **x == 0).count() as f32 / data.len() as f32;
            assert_eq!(mask_dropout_ratio, dropout_ratio);
        });
    }
}
