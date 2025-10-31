//! Samplers which select a token from model outputs.

use std::cell::RefCell;

use rten::{FloatOperators, Operators};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

use crate::generator::TokenId;

/// Samplers take the output logits from a model and select a token ID.
pub trait Sampler {
    /// Sample a token ID from the output logits of a model.
    ///
    /// `logits` has shape `[n_vocab]`.
    ///
    /// # Panics
    ///
    /// `sample` will panic if `logits` is empty.
    fn sample(&self, logits: NdTensorView<f32, 1>) -> TokenId;
}

/// A [`Sampler`] which always chooses the token ID with the highest probability.
#[derive(Clone, Default)]
pub struct ArgMaxSampler {}

impl ArgMaxSampler {
    pub fn new() -> ArgMaxSampler {
        ArgMaxSampler {}
    }
}

impl Sampler for ArgMaxSampler {
    fn sample(&self, logits: NdTensorView<f32, 1>) -> TokenId {
        let next_id = logits
            .to_slice() // For slightly faster iteration, assuming `logits` is contiguous.
            .iter()
            .enumerate()
            .reduce(|(max_i, max_val), (i, val)| {
                if val > max_val {
                    (i, val)
                } else {
                    (max_i, max_val)
                }
            })
            .expect("logits should be non-empty")
            .0;
        next_id as TokenId
    }
}

/// A [`Sampler`] which samples from the top K tokens according to their
/// probabilities.
pub struct TopKSampler {
    k: usize,
    temperature: f32,
    rng: RefCell<fastrand::Rng>,
}

impl TopKSampler {
    /// Create a sampler which samples from the top `k` tokens with a given
    /// temperature.
    ///
    /// The `k` value must be > 0 and temperature must be >= 0.0.
    pub fn new(k: usize, temperature: f32) -> TopKSampler {
        Self::with_rng(fastrand::Rng::new(), k, temperature)
    }

    /// Create a sampler which samples from the top `k` tokens, using a seeded
    /// random number generator.
    pub fn with_rng(rng: fastrand::Rng, k: usize, temperature: f32) -> TopKSampler {
        assert!(temperature >= 0.);
        assert!(k > 0);

        TopKSampler {
            rng: RefCell::new(rng),
            k,
            temperature,
        }
    }
}

impl Sampler for TopKSampler {
    fn sample(&self, logits: NdTensorView<f32, 1>) -> TokenId {
        if self.temperature == 0. || self.k == 1 {
            return ArgMaxSampler::new().sample(logits);
        }

        let logits = if self.temperature != 1.0 {
            logits.map(|x| x / self.temperature).into_cow()
        } else {
            logits.as_cow()
        };

        let [n_vocab] = logits.shape();
        let (topk_logits, topk_indices) = logits
            .topk(
                self.k.min(n_vocab),
                Some(0),
                true,  /* largest */
                false, /* sorted */
            )
            .expect("logits should be non-empty");

        // Convert scores to normalized probabilities and sample a token ID
        // according to each token's probability.
        let probs = topk_logits.softmax(-1).unwrap();
        let topk_index = multinomial(&mut self.rng.borrow_mut(), probs.nd_view())
            .expect("probs should be non-empty and sum to 1");

        let token_id = topk_indices.slice(topk_index).item().copied().unwrap();
        token_id as TokenId
    }
}

/// A [`Sampler`] which samples from the smallest set of tokens whose cumulative
/// probability exceeds a threshold _p_.
///
/// See <https://en.wikipedia.org/wiki/Top-p_sampling>.
pub struct TopPSampler {
    temperature: f32,
    p: f32,
    rng: RefCell<fastrand::Rng>,
    normalize: bool,
}

impl TopPSampler {
    /// Create a sampler with cumulative probability threshold `p`.
    ///
    /// `temperature` specifies a scaling factor to apply before normalizing
    /// logits to probabilities.
    pub fn new(p: f32, temperature: f32) -> TopPSampler {
        Self::with_rng(fastrand::Rng::new(), p, temperature)
    }

    /// Create a sampler with cumulative probability threshold `p` and a
    /// pre-configured random number generator.
    pub fn with_rng(rng: fastrand::Rng, p: f32, temperature: f32) -> TopPSampler {
        assert!(temperature >= 0.);
        assert!((0. ..=1.0).contains(&p));

        Self {
            rng: RefCell::new(rng),
            p,
            temperature,
            normalize: true,
        }
    }

    // For testing, treat input logits as probabilities and don't normalize.
    #[cfg(test)]
    fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Sampler for TopPSampler {
    fn sample(&self, logits: NdTensorView<f32, 1>) -> TokenId {
        if self.temperature == 0. {
            return ArgMaxSampler::new().sample(logits);
        }

        let logits = if self.temperature != 1.0 {
            logits.map(|x| x / self.temperature).into_cow()
        } else {
            logits.as_cow()
        };

        // Convert logits to probabilities.
        let probs = if self.normalize {
            logits.softmax(-1).unwrap().into_cow()
        } else {
            logits.into_dyn()
        };

        // Create (token_id, prob) pairs sorted by ascending probability.
        let mut sorted_probs: Vec<(u32, f32)> = probs
            .data()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(idx, prob)| (idx as u32, *prob))
            .collect();
        sorted_probs.sort_by(|a, b| {
            let (_a_idx, a_prob) = a;
            let (_b_idx, b_prob) = b;
            a_prob.total_cmp(b_prob).reverse()
        });

        // Find k such that the top-K logits have a cumulative probability >= self.p.
        //
        // The threshold is set to be > 0 so the sampled set is non-empty.
        let mut cum_prob = 0.;
        let mut k = 0;
        let threshold = self.p.max(f32::EPSILON);
        while cum_prob < threshold && k < sorted_probs.len() {
            cum_prob += sorted_probs[k].1;
            k += 1;
        }

        // Select the top-K logits, re-normalize their probabilities and sample
        // a token.
        let topk_logits = NdTensor::from_fn([k], |[i]| sorted_probs[i].1);
        let probs = topk_logits.softmax(-1).unwrap();
        let topk_index = multinomial(&mut self.rng.borrow_mut(), probs.nd_view())
            .expect("probs should be non-empty and sum to 1");

        // Map index in sorted logits back to original index.
        sorted_probs[topk_index].0
    }
}

/// Sample an item from a vector of probabilities.
///
/// Returns the index of the selected item, or `None` if the vector is empty
/// or sums to less than 1.
fn multinomial(rng: &mut fastrand::Rng, probs: NdTensorView<f32, 1>) -> Option<usize> {
    let target = rng.f32();

    let mut cum_prob = 0.;
    for (idx, &prob) in probs.iter().enumerate() {
        cum_prob += prob;
        if target <= cum_prob {
            return Some(idx);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;
    use rten_testing::TestCases;

    use super::{ArgMaxSampler, Sampler, TopKSampler, TopPSampler};

    #[test]
    fn test_argmax_sampler() {
        let logits = NdTensor::from([0.1, 0.2, 0.8, 0.7]);
        let sampler = ArgMaxSampler::new();

        for _ in 0..5 {
            let tok_id = sampler.sample(logits.view());
            assert_eq!(tok_id, 2);
        }
    }

    #[test]
    fn test_topk_sampler() {
        #[derive(Debug)]
        struct Case<'a> {
            k: usize,
            temperature: f32,

            // Number of times each of the top `k` tokens should be sampled
            // ordered from least to most frequent. The rng seed is fixed to
            // make this consistent across runs.
            expected_counts: &'a [usize],
        }

        let cases = [
            Case {
                k: 3,
                temperature: 1.0,
                expected_counts: &[12, 25, 63],
            },
            Case {
                k: 1,
                temperature: 1.0,
                expected_counts: &[100],
            },
            Case {
                k: 3,
                temperature: 0.,
                expected_counts: &[0, 0, 100],
            },
            Case {
                k: 3,
                temperature: 0.5,
                expected_counts: &[5, 11, 84],
            },
        ];

        cases.test_each(|case| {
            let &Case {
                k,
                temperature,
                expected_counts: expected,
            } = case;

            let rng = fastrand::Rng::with_seed(1234);

            let logits = NdTensor::arange(0., 10., None);
            let vocab_dim = 0;
            let sampler = TopKSampler::with_rng(rng, k, temperature);

            let token_ids: Vec<_> = (0..100).map(|_| sampler.sample(logits.view())).collect();
            let mut counts = vec![0; logits.size(vocab_dim)];
            for tok_id in &token_ids {
                counts[*tok_id as usize] += 1;
            }

            // All samples should come from the top K tokens.
            for token_id in 0..logits.size(vocab_dim) - k {
                assert_eq!(counts[token_id], 0);
            }
            assert_eq!(
                counts
                    .iter()
                    .skip(logits.size(vocab_dim) - k)
                    .sum::<usize>(),
                token_ids.len()
            );

            // For the top K tokens the distribution should be in proportion to
            // their probabilities.
            assert_eq!(counts[logits.size(vocab_dim) - k..], *expected);
        })
    }

    #[test]
    fn test_topp_sampler() {
        #[derive(Clone, Debug)]
        struct Case {
            // Threshold
            p: f32,
            temperature: f32,

            // If false, treat input logits as probabilities.
            normalize: bool,

            // Logits or probabilities to sample from.
            probs: Vec<f32>,

            // Minimum probability of sampled token.
            min_prob: f32,
        }

        let cases = [
            // Threshold set so that sampled set has only one token.
            Case {
                p: 0.5,
                temperature: 1.0,
                normalize: false,
                probs: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02].into(),
                min_prob: 0.5,
            },
            // Threshold set so that sampled set has two tokens.
            Case {
                p: 0.6,
                temperature: 1.0,
                normalize: false,
                probs: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02].into(),
                min_prob: 0.3,
            },
            Case {
                p: 0.5,
                temperature: 1.0,
                normalize: true,
                // After softmax the probabilities are
                // [0.2288, 0.1873, 0.1534, 0.1459, 0.1430, 0.1416] so we'll
                // need to sample 3 tokens to reach the threshold.
                probs: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02].into(),
                min_prob: 0.1,
            },
            // Temperature of zero, causing fallback to greedy sampling.
            Case {
                temperature: 0.,
                p: 0.6,
                normalize: true,
                probs: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02].into(),
                min_prob: 0.5,
            },
            // Probability of zero.
            Case {
                p: 0.,
                temperature: 1.0,
                normalize: true,
                probs: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02].into(),
                min_prob: 0.5,
            },
        ];

        cases.test_each_clone(|case| {
            let Case {
                p,
                temperature,
                normalize,
                probs,
                min_prob,
            } = case;

            let rng = fastrand::Rng::with_seed(1234);
            let logits = NdTensor::from(probs);

            let sampler = TopPSampler::with_rng(rng, p, temperature).with_normalize(normalize);

            for _ in 0..10 {
                let token_id = sampler.sample(logits.view()) as usize;
                let prob = logits[[token_id]];
                assert!(
                    prob >= min_prob,
                    "sampled token prob {} is below threshold {}",
                    prob,
                    min_prob
                );
            }
        });
    }
}
