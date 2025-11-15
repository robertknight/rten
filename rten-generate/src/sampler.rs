//! Samplers which select a token from model outputs.

use std::cell::RefCell;

use rten_simd::SimdOp;
use rten_vecmath::Softmax;

use crate::Logits;
use crate::generator::TokenId;

/// Samplers take the output logits from a model and select a token ID.
pub trait Sampler {
    /// Sample a token ID from the output logits of a model.
    ///
    /// # Panics
    ///
    /// `sample` will panic if `logits` is empty.
    fn sample(&self, logits: &Logits) -> TokenId;
}

/// A [`Sampler`] which always chooses the token ID with the highest probability.
#[derive(Clone, Default)]
pub struct ArgMax {
    _private: (),
}

impl ArgMax {
    pub fn new() -> ArgMax {
        ArgMax { _private: () }
    }
}

impl Sampler for ArgMax {
    fn sample(&self, logits: &Logits) -> TokenId {
        let next_id = logits
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

/// A [`Sampler`] which chooses a token ID according to the probability of each
/// logit.
///
/// Input logits are first normalized using a softmax operation before a token
/// ID is sampled according to the probability of each logit.
///
/// By default sampling uses a random seed so results will vary for each run.
/// To get repeatable sampling, use [`with_seed`](Multinomial::with_seed).
#[derive(Clone, Default)]
pub struct Multinomial {
    rng: RefCell<fastrand::Rng>,

    // Scratch space for normalized logits.
    scratch: RefCell<Vec<f32>>,
}

impl Multinomial {
    /// Create a sampler with a random seed.
    pub fn new() -> Self {
        Self {
            rng: RefCell::new(fastrand::Rng::default()),
            scratch: RefCell::new(Vec::new()),
        }
    }

    /// Create a sampler with a fixed seed.
    ///
    /// This guarantees repeatable sampling.
    pub fn with_seed(seed: u64) -> Self {
        let rng = fastrand::Rng::with_seed(seed);
        Self {
            rng: RefCell::new(rng),
            scratch: RefCell::new(Vec::new()),
        }
    }
}

impl Sampler for Multinomial {
    fn sample(&self, logits: &Logits) -> TokenId {
        assert!(!logits.is_empty());

        // Normalize logits to probabilities.
        let mut scratch = self.scratch.borrow_mut();
        scratch.clear();
        scratch.reserve(logits.len());
        let scratch = &mut scratch.spare_capacity_mut()[..logits.len()];
        let probs = Softmax::new(logits.logits(), scratch).dispatch();

        let mut rng = self.rng.borrow_mut();

        // Sample ID according to probabilities.
        //
        // `multinomial` may return None if the input contains a NaN or
        // infinity. In that case we fall back to the ID zero.
        let idx = multinomial(&mut rng, probs).unwrap_or(0);

        logits.indices()[idx]
    }
}

/// Sample an item from a vector of probabilities.
///
/// Returns the index of the selected item, or `None` if the vector is empty
/// or sums to less than 1.
fn multinomial(rng: &mut fastrand::Rng, probs: &[f32]) -> Option<usize> {
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
    use rten_simd::SimdOp;
    use rten_testing::TestCases;
    use rten_vecmath::Softmax;

    use super::{ArgMax, Multinomial, Sampler};
    use crate::Logits;
    use crate::generator::TokenId;

    #[test]
    fn test_argmax() {
        let logits = Logits::dense(vec![0.1, 0.2, 0.8, 0.7]);
        let sampler = ArgMax::new();

        for _ in 0..5 {
            let tok_id = sampler.sample(&logits);
            assert_eq!(tok_id, 2);
        }
    }

    #[test]
    fn test_multinomial() {
        let logits = Logits::dense(vec![0.25, 0.25, 0.5]);
        let sampler = Multinomial::with_seed(1234);
        let n_iters = 512;

        let mut counts = vec![0u32; logits.len()];
        for _ in 0..n_iters {
            let tok_id = sampler.sample(&logits);
            counts[tok_id as usize] += 1;
        }

        let mut normalized_logits = logits.logits().to_vec();
        Softmax::new_mut(&mut normalized_logits).dispatch();

        // Check sample count for each token is within a threshold percentage
        // of expectations. Increasing the sample count should bring actual
        // closer to expected.
        let threshold = 0.12;
        for (prob, count) in normalized_logits.into_iter().zip(counts) {
            let expected = (prob * n_iters as f32).round() as i32;
            let delta = (count as i32 - expected).abs();
            let delta_frac = delta as f32 / expected as f32;

            assert!(
                delta_frac <= threshold,
                "sample count differs from expectation by {:.1}%, above threshold {}%",
                delta_frac * 100.0,
                threshold * 100.0
            );
        }
    }

    #[test]
    fn test_multinomial_nan_infinity() {
        #[derive(Debug)]
        struct Case {
            logits: Vec<f32>,
            expected: TokenId,
        }

        let cases = [
            // Softmax normalization spreads NaNs and positive infinities.
            Case {
                logits: vec![0.1, f32::NAN, 0.5],
                expected: 0,
            },
            Case {
                logits: vec![0.1, f32::INFINITY, 0.5],
                expected: 0,
            },
            // Negative infinity shrinks to zero after softmax.
            Case {
                logits: vec![0., f32::NEG_INFINITY, 100.0],
                expected: 2,
            },
        ];

        cases.test_each(|case| {
            let logits = Logits::dense(case.logits.clone());
            let sampler = Multinomial::with_seed(1234);
            let n_iters = 10;
            for _ in 0..n_iters {
                let token_id = sampler.sample(&logits);
                assert_eq!(token_id, case.expected);
            }
        });
    }
}
