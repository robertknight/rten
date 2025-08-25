//! Filters for processing model outputs prior to sampling.
//!
//! This module defines the [`LogitsFilter`] trait implemented by all filters,
//! plus convenience functions to simplify implementing filters.

use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

use crate::generator::TokenId;

/// Filter which modifies the output logits from a model.
///
/// The filter is applied to the model outputs before a token is sampled.
pub trait LogitsFilter {
    /// Filter the model's output and return the modified logits.
    ///
    /// If this method returns `None`, the input logits are passed unmodified
    /// to the sampler. `prev_tokens` contains the previously sampled tokens,
    /// including the prompt.
    fn filter(
        &self,
        logits: NdTensorView<f32, 1>,
        prev_tokens: &[TokenId],
    ) -> Option<NdTensor<f32, 1>>;
}

struct TokenIdFilter<F: Fn(TokenId) -> bool> {
    predicate: F,
}

impl<F: Fn(TokenId) -> bool> LogitsFilter for TokenIdFilter<F> {
    fn filter(
        &self,
        logits: NdTensorView<f32, 1>,
        _prev_tokens: &[TokenId],
    ) -> Option<NdTensor<f32, 1>> {
        Some(NdTensor::from_fn(logits.shape(), |[i]| {
            let token_id = i as TokenId;
            if (self.predicate)(token_id) {
                logits[[i]]
            } else {
                f32::NEG_INFINITY
            }
        }))
    }
}

/// Create a filter which suppresses all tokens that do not match a predicate by
/// setting the value to `f32::NEG_INFINITY`.
pub fn token_id_filter<F: Fn(TokenId) -> bool>(predicate: F) -> impl LogitsFilter {
    TokenIdFilter { predicate }
}

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;

    use super::{LogitsFilter, token_id_filter};

    #[test]
    fn test_token_id_filter() {
        let logits = NdTensor::from([0., 1., 2., 3., 4.]);
        let filter = token_id_filter(|id| id % 2 == 0);
        let output = filter.filter(logits.view(), &[]);
        assert_eq!(
            output,
            Some(NdTensor::from([
                0.,
                f32::NEG_INFINITY,
                2.,
                f32::NEG_INFINITY,
                4.
            ]))
        );
    }
}
