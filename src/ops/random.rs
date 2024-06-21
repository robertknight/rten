use fastrand::Rng;
use fastrand_contrib::RngExt;
use rten_tensor::prelude::*;
use rten_tensor::Tensor;

use crate::ops::{InputList, IntoOpResult, OpError, Operator, OutputList};
use crate::tensor_pool::TensorPool;

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

    fn run(&self, pool: &TensorPool, _inputs: InputList) -> Result<OutputList, OpError> {
        let scale_value = |val: f32| self.low + val * (self.high - self.low);
        let shape = self.shape.as_slice();

        let mut rng = if let Some(seed) = self.seed {
            Rng::with_seed(seed.to_bits() as u64)
        } else {
            Rng::new()
        };
        Tensor::from_simple_fn_in(pool, shape, || scale_value(rng.f32())).into_op_result()
    }
}

#[derive(Debug)]
pub struct RandomUniformLike {
    pub low: f32,
    pub high: f32,

    /// Random seed. See [RandomUniform::seed].
    pub seed: Option<f32>,
}

impl Operator for RandomUniformLike {
    fn name(&self) -> &str {
        "RandomUniformLike"
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let op = RandomUniform {
            low: self.low,
            high: self.high,
            seed: self.seed,
            shape: input.shape().to_vec(),
        };
        op.run(pool, InputList::new())
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

    fn run(&self, pool: &TensorPool, _inputs: InputList) -> Result<OutputList, OpError> {
        let shape = self.shape.as_slice();

        let mut rng = if let Some(seed) = self.seed {
            Rng::with_seed(seed.to_bits() as u64)
        } else {
            Rng::new()
        };

        Tensor::from_simple_fn_in(pool, shape, || rng.f32_normal(self.mean, self.scale))
            .into_op_result()
    }
}

#[derive(Debug)]
pub struct RandomNormalLike {
    pub mean: f32,
    pub scale: f32,

    /// Random seed. See [RandomUniform::seed].
    pub seed: Option<f32>,
}

impl Operator for RandomNormalLike {
    fn name(&self) -> &str {
        "RandomNormalLike"
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let op = RandomNormal {
            mean: self.mean,
            scale: self.scale,
            seed: self.seed,
            shape: input.shape().to_vec(),
        };
        op.run(pool, InputList::new())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;

    use crate::ops::operators::FloatOperators;
    use crate::ops::tests::{new_pool, run_op};
    use crate::ops::{InputList, Operator};

    use super::{RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike};

    #[test]
    fn test_random_uniform() {
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

        let pool = new_pool();

        for Case {
            low,
            high,
            shape,
            seed,
        } in cases
        {
            let op = RandomUniform {
                low,
                high,
                shape,
                seed,
            };
            let output = op.run(&pool, InputList::new()).unwrap().remove(0);
            let output: Tensor = output.try_into().unwrap();

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
            let output_2 = op.run(&pool, InputList::new()).unwrap().remove(0);
            let output_2: Tensor = output_2.try_into().unwrap();
            if let Some(_seed) = seed {
                assert_eq!(output, output_2);
            } else {
                assert_ne!(output, output_2);
            }
        }
    }

    #[test]
    fn test_random_uniform_like() {
        let input = Tensor::<f32>::zeros(&[5, 5]);
        let op = RandomUniformLike {
            low: 0.,
            high: 2.,
            seed: None,
        };
        let output: Tensor<f32> = run_op(&op, input.view()).unwrap();
        assert_eq!(output.shape(), &[5, 5]);
    }

    #[test]
    fn test_random_normal() -> Result<(), Box<dyn Error>> {
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

        for Case {
            mean,
            scale,
            shape,
            seed,
        } in cases
        {
            let op = RandomNormal {
                mean,
                scale,
                shape,
                seed,
            };
            let output: Tensor = run_op(&op, InputList::new()).unwrap();
            assert_eq!(output.shape(), op.shape);

            // Test that outputs have expected distribution.
            let mean = output
                .reduce_mean(None, false /* keep_dims */)?
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
                .reduce_sum(None, false /* keep_dims */)?
                .item()
                .unwrap()
                / output.len() as f32;
            let std_dev = var.sqrt();
            let delta = (std_dev - op.scale).abs();
            assert!(delta <= threshold, "delta {delta} > {threshold}");

            // Test that repeated generation produces the same output if the
            // seed is fixed, or different output otherwise.
            let output_2: Tensor = run_op(&op, InputList::new()).unwrap();
            if let Some(_seed) = seed {
                assert_eq!(output, output_2);
            } else {
                assert_ne!(output, output_2);
            }
        }

        Ok(())
    }

    #[test]
    fn test_random_normal_like() {
        let input = Tensor::<f32>::zeros(&[5, 5]);
        let op = RandomNormalLike {
            mean: 0.,
            scale: 5.,
            seed: None,
        };
        let output: Tensor<f32> = run_op(&op, input.view()).unwrap();
        assert_eq!(output.shape(), &[5, 5]);
    }
}
