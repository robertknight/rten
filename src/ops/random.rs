use fastrand::Rng;
use fastrand_contrib::RngExt;
use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};

use crate::buffer_pool::AutoReturn;
use crate::infer_shapes::{InferShapes, UnaryOp, impl_infer_shapes};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::value::{DataType, Value, ValueType};

/// Create a random number generator, seeded with `seed` if it is provided.
///
/// The seed is an `f32` for consistency with the ONNX specification, which
/// specifies random seeds for operators as floats.
fn rng_from_seed(seed: Option<f32>) -> Rng {
    match seed {
        Some(seed) => Rng::with_seed(seed.to_bits() as u64),
        None => Rng::new(),
    }
}

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

    fn max_inputs(&self) -> Option<usize> {
        Some(0)
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let scale_value = |val: f32| self.low + val * (self.high - self.low);
        let shape = self.shape.as_slice();

        let mut rng = rng_from_seed(self.seed);
        Tensor::from_simple_fn_in(ctx.pool(), shape, || scale_value(rng.f32())).into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Float))].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    RandomUniform,
    op,
    shape_ops::FixedShape { shape: &op.shape }
);

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

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
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

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Float))].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
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

    fn max_inputs(&self) -> Option<usize> {
        Some(0)
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let shape = self.shape.as_slice();

        let mut rng = rng_from_seed(self.seed);

        // Use `Rng::f32_normal_approx` here rather than `Rng::f32_normal`
        // because the approximation is much faster and good enough for use
        // cases for random normal distributions in ML models.
        //
        // See https://marc-b-reynolds.github.io/distribution/2021/03/18/CheapGaussianApprox.html
        // for more info on the algorithm. The non-approximate version uses the
        // Box-Muller transform.
        Tensor::from_simple_fn_in(ctx.pool(), shape, || {
            rng.f32_normal_approx(self.mean, self.scale)
        })
        .into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Float))].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(RandomNormal, op, shape_ops::FixedShape { shape: &op.shape });

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

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
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

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Float))].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }
}

#[derive(Debug)]
pub struct Multinomial {
    /// Number of samples to draw for each row of the input.
    pub sample_size: usize,

    /// Random seed. See [`RandomUniform::seed`].
    pub seed: Option<f32>,
}

impl Operator for Multinomial {
    fn name(&self) -> &str {
        "Multinomial"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let pool = ctx.pool();
        let input: NdTensorView<f32, 2> = ctx.inputs().require_as(0)?;
        let [batch_size, class_size] = input.shape();

        if class_size == 0 {
            return Err(OpError::InvalidValue("input must have at least one class"));
        }

        let mut rng = rng_from_seed(self.seed);

        let input = input.to_contiguous_in(pool).auto_return(pool);
        let input = input.data();

        let mut output = pool.alloc(batch_size * self.sample_size);
        let mut cdf = pool.alloc(class_size).auto_return(pool);
        for row in input.chunks(class_size) {
            // Convert the unnormalized log probabilities for this row into a
            // cumulative distribution. The maximum is subtracted before
            // exponentiating, for numerical stability.
            //
            // `cdf` is left unnormalized to avoid a division per class. `sum`
            // ends up as the total weight, which is folded into the sampling
            // threshold below instead.
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            cdf.clear();
            let mut sum = 0.;
            for &logit in row {
                sum += (logit - max).exp();
                cdf.push(sum);
            }

            for _ in 0..self.sample_size {
                // Scale the `[0, 1)` draw by `sum` to match the scale of `cdf`.
                let threshold = rng.f32() * sum;
                // Sample the first class whose cumulative probability exceeds
                // the threshold. The last value in `cdf` is exactly `sum` and
                // `rng.f32() < 1`, so the maximum value of `idx` is
                // `class_size - 1`.
                let idx = cdf.partition_point(|&c| c <= threshold);
                output.push(idx as i32);
            }
        }

        Tensor::from_data(&[batch_size, self.sample_size], output).into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Int32))].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    Multinomial,
    op,
    shape_ops::Multinomial {
        sample_size: op.sample_size
    }
);

#[derive(Debug)]
pub struct Dropout {
    pub seed: Option<i32>,
}

impl Operator for Dropout {
    fn name(&self) -> &str {
        "Dropout"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
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

        let (output, mask) =
            if !training_mode || ratio == 0. {
                let mask = Tensor::<i32>::full(input.shape(), 1);
                (input.to_tensor(), mask)
            } else {
                let mut rng = if let Some(seed) = self.seed {
                    Rng::with_seed(seed as u64)
                } else {
                    Rng::new()
                };
                let scale = 1. / (1. - ratio);

                let mask = Tensor::<i32>::from_simple_fn(input.shape(), || {
                    if rng.f32() < ratio { 0 } else { 1 }
                });
                let input = input.to_contiguous_in(ctx.pool());

                let output = Tensor::from_data(
                    input.shape(),
                    input
                        .data()
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

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some(OutputTypeList::from_slice(&[
            OutputType::CopyFromInput(0),
            OutputType::Fixed(ValueType::Tensor(DataType::Int32)),
        ]))
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(Dropout, _op, shape_ops::Dropout);

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_testing::TestCases;

    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpError, OpRunContext, Operator, OperatorExt};
    use crate::ops::operators::{FloatOperators, Operators};

    use super::{
        Dropout, Multinomial, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike,
    };

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
                shape: vec![10, 100],
                seed: Some(0.1),
            },
            // Custom mean/scale values.
            Case {
                mean: 5.,
                scale: 0.5,
                shape: vec![10, 100],
                seed: Some(0.5),
            },
            // Auto-generate a seed
            Case {
                mean: 0.,
                scale: 1.,
                shape: vec![10, 100],
                seed: None,
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

            // Threshold is inversely proportional to number of samples (ie.
            // with more samples, values will approach expectation). For a
            // random seed use a higher threshold to avoid flakiness in case
            // mean/std-dev are far from expectation by chance.
            let threshold = if seed.is_some() { 0.05 } else { 0.5 };
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
    fn test_multinomial() {
        #[derive(Clone, Debug)]
        struct Case {
            // Per-class unnormalized log probabilities.
            logits: Vec<f32>,
            seed: Option<f32>,
        }

        let cases = [
            // Non-uniform distribution.
            Case {
                logits: vec![1.0, 2.0, 3.0, 4.0],
                seed: Some(0.5),
            },
            // Uniform distribution.
            Case {
                logits: vec![2.0, 2.0, 2.0, 2.0],
                seed: Some(1.0),
            },
            // All zeros
            Case {
                logits: vec![0.0, 0.0, 0.0, 0.0],
                seed: Some(1.0),
            },
            // Auto-generated seed.
            Case {
                logits: vec![-1.0, 0.0, 2.0],
                seed: None,
            },
        ];

        cases.test_each_clone(|Case { logits, seed }| {
            let class_size = logits.len();
            let sample_size = 5000;
            let input = Tensor::from_data(&[1, class_size], logits.clone());

            let op = Multinomial { sample_size, seed };
            let output: Tensor<i32> = op.run_simple(input.view()).unwrap();
            assert_eq!(output.shape(), &[1, sample_size]);

            // Expected per-class probabilities, computed via softmax.
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
            let exp_sum: f32 = exp.iter().sum();
            let expected: Vec<f32> = exp.iter().map(|x| x / exp_sum).collect();

            // Count how often each class was sampled.
            let mut counts = vec![0; class_size];
            for &idx in output.iter() {
                assert!(
                    (idx as usize) < class_size,
                    "sampled class {idx} out of range"
                );
                counts[idx as usize] += 1;
            }

            // Check that empirical frequencies approximately match the expected
            // distribution.
            for (cls, &count) in counts.iter().enumerate() {
                let freq = count as f32 / sample_size as f32;
                let delta = (freq - expected[cls]).abs();
                assert!(
                    delta <= 0.05,
                    "class {cls} frequency {freq} too far from expected {}",
                    expected[cls]
                );
            }
        })
    }

    #[test]
    fn test_multinomial_seed() {
        let input = Tensor::from([[1.0f32, 2.0, 3.0]]);

        // A fixed seed produces repeatable output.
        let seeded = Multinomial {
            sample_size: 20,
            seed: Some(0.5),
        };
        let a: Tensor<i32> = seeded.run_simple(input.view()).unwrap();
        let b: Tensor<i32> = seeded.run_simple(input.view()).unwrap();
        assert_eq!(a, b);

        // Without a seed, repeated runs (very likely) differ.
        let unseeded = Multinomial {
            sample_size: 20,
            seed: None,
        };
        let a: Tensor<i32> = unseeded.run_simple(input.view()).unwrap();
        let b: Tensor<i32> = unseeded.run_simple(input.view()).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_multinomial_empty_classes() {
        // An input with zero classes has no distribution to sample from. This
        // should report an error rather than panic.
        let input = Tensor::<f32>::zeros(&[2, 0]);
        let op = Multinomial {
            sample_size: 4,
            seed: None,
        };
        let result: Result<Tensor<i32>, _> = op.run_simple(input.view());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("input must have at least one class"))
        );
    }

    #[test]
    fn test_multinomial_empty_batch() {
        // An input with no rows yields an empty output without error.
        let input = Tensor::<f32>::zeros(&[0, 3]);
        let op = Multinomial {
            sample_size: 4,
            seed: None,
        };
        let output: Tensor<i32> = op.run_simple(input.view()).unwrap();
        assert_eq!(output.shape(), &[0, 4]);
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
            let inputs = InputList::from_iter([
                Some(data.view().into()),
                ratio_input.as_ref().map(|ri| ri.view().into()),
                training_mode_input.as_ref().map(|tm| tm.view().into()),
            ]);
            let pool = BufferPool::new();
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
            let inputs = InputList::from_iter([
                Some(data.view().into()),
                ratio_input.as_ref().map(|ri| ri.view().into()),
                Some(training_mode_input.view().into()),
            ]);
            let pool = BufferPool::new();
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
