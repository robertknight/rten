use fastrand::Rng;
use rten_tensor::Tensor;

use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output};
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

    fn run(&self, pool: &TensorPool, _inputs: InputList) -> Result<Vec<Output>, OpError> {
        let scale_value = |val: f32| self.low + val * (self.high - self.low);
        let shape = self.shape.as_slice();

        let mut rng = if let Some(seed) = self.seed {
            Rng::with_seed(seed.to_bits() as u64)
        } else {
            Rng::new()
        };

        let len = shape.iter().product();
        let mut data = pool.alloc_vec(len);
        data.extend(std::iter::from_fn(|| Some(scale_value(rng.f32()))).take(len));

        Tensor::from_data(shape, data).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;

    use crate::ops::tests::new_pool;
    use crate::ops::{InputList, Operator};

    use super::RandomUniform;

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
}
