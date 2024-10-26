//! Filters for processing model outputs prior to sampling.

use rten_tensor::{NdTensor, NdTensorView};

/// Filter which modifies the output logits from a model.
///
/// The filter is applied to the model outputs before a token is sampled.
pub trait LogitsFilter {
    /// Filter the model's output and return the modified logits.
    ///
    /// If this method returns `None`, the input logits are passed unmodified
    /// to the sampler. `prev_tokens` contains the previously sampled tokens,
    /// including the prompt.
    fn filter(&self, logits: NdTensorView<f32, 1>, prev_tokens: &[u32])
        -> Option<NdTensor<f32, 1>>;
}
