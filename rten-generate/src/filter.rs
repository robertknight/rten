//! Filters for processing model outputs ("logits") prior to sampling.
//!
//! This module defines the [`LogitsFilter`] trait implemented by all filters,
//! plus convenience functions to simplify implementing filters.

use rten_simd::ops::{MaskOps, NumOps};
use rten_simd::{Isa, Simd, SimdIterable, SimdOp};
use rten_vecmath::Softmax;

use crate::Logits;
use crate::generator::TokenId;

/// Filter which modifies the output logits from a model.
///
/// Filters can remove tokens or alter their scores. Filters are stateless and
/// at each step they receive logits from the model or a previous filter, plus
/// the previously generated token IDs.
///
/// Filters can be chained together using [`Chain`].
pub trait LogitsFilter {
    /// Filter the model's output and return the modified logits.
    ///
    /// `prev_tokens` contains the previously sampled tokens, including the prompt.
    fn filter(&self, logits: Logits, prev_tokens: &[TokenId]) -> Logits;
}

struct TokenIdFilter<F: Fn(TokenId) -> bool> {
    predicate: F,
}

impl<F: Fn(TokenId) -> bool> LogitsFilter for TokenIdFilter<F> {
    fn filter(&self, logits: Logits, _prev_tokens: &[TokenId]) -> Logits {
        let (logits, indices) = logits.into_logits_indices();
        let (new_logits, new_indices) = logits
            .into_iter()
            .zip(indices)
            .filter(|(_logit, token_id)| (self.predicate)(*token_id))
            .unzip();
        Logits::sparse(new_logits, new_indices)
    }
}

/// Create a filter which suppresses all tokens that do not match a predicate by
/// setting the value to `f32::NEG_INFINITY`.
pub fn token_id_filter<F: Fn(TokenId) -> bool>(predicate: F) -> impl LogitsFilter {
    TokenIdFilter { predicate }
}

/// Filter which scales logits uniformly.
///
/// This updates the value of each input logit using the formula `logit /
/// temperature`.
pub struct Temperature {
    temperature: f32,
}

impl Temperature {
    /// Create a temperature filter which updates each logit by dividing by
    /// `temperature`.
    pub fn new(temperature: f32) -> Self {
        assert!(temperature >= 0.);
        Self { temperature }
    }
}

impl LogitsFilter for Temperature {
    fn filter(&self, logits: Logits, _prev_tokens: &[TokenId]) -> Logits {
        if self.temperature == 1.0 {
            return logits;
        }
        let (mut logits, indices) = logits.into_logits_indices();
        let inv_temp = 1. / self.temperature;
        for x in &mut logits {
            *x *= inv_temp;
        }
        Logits::sparse(logits, indices)
    }
}

/// Applies a sequence of logit filters in series.
pub struct Chain {
    filters: Vec<Box<dyn LogitsFilter>>,
}

impl Default for Chain {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain {
    /// Create an empty logits filter chain.
    ///
    /// An empty chain returns input logits unmodified.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a new filter to the chain.
    pub fn append<F: LogitsFilter + 'static>(mut self, filter: F) -> Self {
        self.filters.push(Box::new(filter));
        self
    }

    /// Add a temperature filter to the chain. See [`Temperature`].
    pub fn temperature(self, temp: f32) -> Self {
        self.append(Temperature::new(temp))
    }

    /// Add a top-P (nucleus sampling) filter to the chain. See [`TopP`].
    pub fn top_p(self, p: f32) -> Self {
        self.append(TopP::new(p))
    }

    /// Add a top-K filter to the chain. See [`TopK`].
    pub fn top_k(self, k: usize) -> Self {
        self.append(TopK::new(k))
    }
}

impl LogitsFilter for Chain {
    fn filter(&self, logits: Logits, prev_tokens: &[TokenId]) -> Logits {
        self.filters
            .iter()
            .fold(logits, |logits, f| f.filter(logits, prev_tokens))
    }
}

/// Filter which retains K logits with the highest values.
pub struct TopK {
    k: usize,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl LogitsFilter for TopK {
    fn filter(&self, logits: Logits, _prev_tokens: &[TokenId]) -> Logits {
        if logits.is_empty() {
            return logits;
        }

        let (logits, indices) = logits.into_logits_indices();

        let topk = SimdTopK {
            k: self.k,
            indices: &indices,
            logits: &logits,
        }
        .dispatch();

        let (indices, logits) = topk.into_iter().unzip();
        Logits::sparse(logits, indices)
    }
}

/// Vectorized top-K implementation which assumes that K is small relative to
/// the input length, and that values are distributed such that updates to the
/// running top-K are infrequent.
struct SimdTopK<'a> {
    k: usize,
    logits: &'a [f32],
    indices: &'a [u32],
}

impl<'a> SimdOp for SimdTopK<'a> {
    type Output = Vec<(u32, f32)>;

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let SimdTopK { logits, indices, k } = self;

        let ops = isa.f32();
        let mask_ops = isa.m32();
        let compare_gt = |a: f32, b: f32| a.total_cmp(&b).reverse();

        // Create an initial sorted top-K list from the first K entries.
        let mut topk: Vec<(u32, f32)> = indices
            .iter()
            .zip(logits)
            .take(k)
            .map(|(i, logit)| (*i, logit.clone()))
            .collect();
        topk.sort_by(|a, b| compare_gt(a.1, b.1));

        if k == 0 || logits.len() == k {
            return topk;
        }

        let mut kth_logit = topk.last().unwrap().1;
        let mut kth_logit_vec = ops.splat(kth_logit);

        let mut update_topk = |kth_logit: &mut f32, index: u32, logit: f32| {
            if logit > *kth_logit {
                *topk.last_mut().unwrap() = (index, logit);
                topk.sort_by(|a, b| compare_gt(a.1, b.1));
                *kth_logit = topk.last().unwrap().1;
            }
        };

        let indices = &indices[k..];
        let logits = &logits[k..];

        // Iterate over SIMD-sized chunks of remaining logits and update running
        // top-K.
        let mut indices_iter = indices.chunks_exact(ops.len());
        let mut logits_iter = logits.simd_iter(ops);
        for (index_chunk, logits_vec) in indices_iter.by_ref().zip(logits_iter.by_ref()) {
            if mask_ops.any(ops.gt(logits_vec, kth_logit_vec)) {
                for (&index, logit) in index_chunk.iter().zip(logits_vec.to_array()) {
                    update_topk(&mut kth_logit, index, logit);
                }
                kth_logit_vec = ops.splat(kth_logit);
            }
        }

        // Handle tail.
        if let Some((logits_tail, _mask)) = logits_iter.tail() {
            let indices_tail = indices_iter.remainder();
            for (&index, logit) in indices_tail.iter().zip(logits_tail.to_array()) {
                update_topk(&mut kth_logit, index, logit);
            }
        }

        topk
    }
}

/// Filter which retains the logits whose cumulative probability exceeds a
/// threshold _p_.
///
/// See <https://en.wikipedia.org/wiki/Top-p_sampling>.
pub struct TopP {
    cumulative_prob: f32,
    normalize: bool,
}

impl TopP {
    pub fn new(cumulative_prob: f32) -> Self {
        Self {
            cumulative_prob,
            normalize: false,
        }
    }

    /// Set whether input logits are normalized to probabilities using softmax
    /// before the top-P subset is computed.
    ///
    /// This is true by default.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl LogitsFilter for TopP {
    fn filter(&self, logits: Logits, _prev_tokens: &[TokenId]) -> Logits {
        if self.cumulative_prob == 1.0 {
            return logits;
        }

        let (mut logits, indices) = logits.into_logits_indices();

        // Normalize logits to probabilities.
        if self.normalize {
            Softmax::new_mut(&mut logits).dispatch();
        }

        // Combine into (logit, token_id) tuples and sort by probability
        // descending.
        let mut pairs: Vec<(f32, TokenId)> = logits.into_iter().zip(indices).collect();
        pairs.sort_by(|a, b| {
            let (a_prob, _a_id) = a;
            let (b_prob, _b_id) = b;
            a_prob.total_cmp(b_prob).reverse()
        });

        // Find k such that the top-K logits have a cumulative probability >= self.p.
        //
        // The threshold is set to be > 0 so the sampled set is non-empty.
        let mut cum_prob = 0.;
        let mut k = 0;
        let threshold = self.cumulative_prob.max(f32::MIN_POSITIVE);
        while cum_prob < threshold && k < pairs.len() {
            cum_prob += pairs[k].0;
            k += 1;
        }
        pairs.truncate(k);

        // Return the top-K logits.
        let (logits, indices) = pairs.into_iter().unzip();
        Logits::sparse(logits, indices)
    }
}

/// Filter which sorts logits in descending order of their scores.
#[derive(Default)]
pub struct Sort {
    _private: (),
}

impl Sort {
    pub fn new() -> Self {
        Sort { _private: () }
    }
}

impl LogitsFilter for Sort {
    fn filter(&self, logits: Logits, _prev_tokens: &[TokenId]) -> Logits {
        let (logits, indices) = logits.into_logits_indices();

        let mut pairs: Vec<(f32, TokenId)> = logits.into_iter().zip(indices).collect();
        pairs.sort_by(|(a_val, _), (b_val, _)| a_val.total_cmp(b_val).reverse());

        let (logits, indices) = pairs.into_iter().unzip();
        Logits::sparse(logits, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::{Chain, Logits, LogitsFilter, Sort, Temperature, TopK, TopP, token_id_filter};

    #[test]
    fn test_token_id_filter() {
        let logits = Logits::dense(vec![0., 1., 2., 3., 4.]);
        let filter = token_id_filter(|id| id % 2 == 0);
        let output = filter.filter(logits, &[]);
        assert_eq!(output.logits(), &[0., 2., 4.]);
        assert_eq!(output.indices(), &[0, 2, 4]);
    }

    #[test]
    fn test_temperature() {
        let logits = Logits::dense(vec![0., 1., 2., 3., 4.]);
        let filter = Temperature::new(2.0);
        let output = filter.filter(logits, &[]);
        assert_eq!(output.logits(), &[0., 0.5, 1., 1.5, 2.0]);
        assert_eq!(output.indices(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_chain() {
        let logits = Logits::dense(vec![0., 1., 2., 3., 4.]);
        let chain = Chain::new()
            .append(token_id_filter(|id| id % 2 == 0))
            .append(token_id_filter(|id| id > 0));
        let output = chain.filter(logits, &[]);
        assert_eq!(output.logits(), &[2., 4.]);
        assert_eq!(output.indices(), &[2, 4]);
    }

    fn reference_topk(logits: &Logits, k: usize) -> Logits {
        let mut pairs: Vec<(u32, f32)> = logits
            .indices()
            .iter()
            .zip(logits.logits())
            .map(|(idx, val)| (*idx, *val))
            .collect();
        pairs.sort_by(|a, b| a.1.total_cmp(&b.1).reverse());
        pairs.truncate(k);
        let (indices, logits) = pairs.into_iter().unzip();
        Logits::sparse(logits, indices)
    }

    #[test]
    fn test_top_k() {
        let sort = |logits| Sort::new().filter(logits, &[]);

        let logits = Logits::dense(vec![
            -1., 1., 0., 2., -2., 10., -3., 2., 1., 0., 20., -5., 5., 0.1, -0.2, 0.2, 0.1,
        ]);
        // Exceeds max common SIMD vector width (16 x f32 = 512 bits) and has
        // a tail.
        assert_eq!(logits.len(), 17);

        // Test cases where K <= logits length.
        for k in 0..=logits.len() {
            let topk = TopK::new(k).filter(logits.clone(), &[]);
            let sorted_topk = sort(topk);
            let expected_topk = reference_topk(&logits, k);
            assert_eq!(sorted_topk.logits(), expected_topk.logits());
            assert_eq!(sorted_topk.indices(), expected_topk.indices());
        }

        // Test empty logits
        let logits = Logits::dense(vec![]);
        let topk = TopK::new(1).filter(logits, &[]);
        assert!(topk.is_empty());
    }

    #[test]
    fn test_top_p() {
        // These tests disable normalization so the input logits are treated
        // directly as probabilities.

        let logits = Logits::dense(vec![0.1, 0.25, 0.15, 0.5]);
        let all_logits = TopP::new(1.0).normalize(false).filter(logits.clone(), &[]);
        assert_eq!(logits, all_logits);

        let top_p_logits = TopP::new(0.5).normalize(false).filter(logits.clone(), &[]);
        assert_eq!(top_p_logits.logits(), &[0.5]);
        assert_eq!(top_p_logits.indices(), &[3]);

        let top_p_logits = TopP::new(0.75).normalize(false).filter(logits.clone(), &[]);
        assert_eq!(top_p_logits.logits(), &[0.5, 0.25]);
        assert_eq!(top_p_logits.indices(), &[3, 1]);

        // As a special case, the probability is clamped to be > 0 so that at
        // least one token will be sampled, if the input is non-empty.
        let top_p_logits = TopP::new(0.).normalize(false).filter(logits.clone(), &[]);
        assert_eq!(top_p_logits.logits(), &[0.5]);
        assert_eq!(top_p_logits.indices(), &[3]);
    }
}
