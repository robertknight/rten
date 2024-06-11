//! Samplers which select a token from model outputs.

use std::cell::RefCell;

use rten::{FloatOperators, Operators};
use rten_tensor::prelude::*;
use rten_tensor::NdTensorView;

/// Samplers take the output logits from a model and select a token ID.
pub trait Sampler {
    /// Sample a token ID from the output logits of a model.
    ///
    /// `logits` has shape `[n_vocab]`.
    fn sample(&self, logits: NdTensorView<f32, 1>) -> u32;
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
    fn sample(&self, logits: NdTensorView<f32, 1>) -> u32 {
        let next_id = logits
            .arg_max(-1, false /* keep_dims */)
            .expect("logits should be non-empty")
            .item()
            .copied()
            .expect("result should be scalar");
        next_id as u32
    }
}

/// A [`Sampler`] which samples from the top K tokens according to their
/// probabilities.
pub struct TopKSampler {
    k: usize,
    rng: RefCell<fastrand::Rng>,
}

impl TopKSampler {
    /// Create a sampler which samples from the top `k` tokens.
    pub fn new(k: usize) -> TopKSampler {
        Self::with_rng(fastrand::Rng::new(), k)
    }

    /// Create a sampler which samples from the top `k` tokens, using a seeded
    /// random number generator.
    pub fn with_rng(rng: fastrand::Rng, k: usize) -> TopKSampler {
        TopKSampler {
            rng: RefCell::new(rng),
            k,
        }
    }
}

impl Sampler for TopKSampler {
    fn sample(&self, logits: NdTensorView<f32, 1>) -> u32 {
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

        let token_id = topk_indices
            .slice::<0, _>(topk_index)
            .item()
            .copied()
            .unwrap();
        token_id as u32
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
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    use super::{ArgMaxSampler, Sampler, TopKSampler};

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
        // Use a fixed seed for reproducibility.
        let rng = fastrand::Rng::with_seed(1234);

        let logits = NdTensor::arange(0., 10., None);
        let vocab_dim = 0;
        let k = 3;
        let sampler = TopKSampler::with_rng(rng, k);

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
        assert_eq!(counts[logits.size(vocab_dim) - k..], [12, 25, 63]);
    }
}
