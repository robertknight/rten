use crate::generator::TokenId;

/// A sparse vector of token scores.
///
/// At each step of generation, the model yields a dense vector of scores for
/// each token. These are filtered and processed using
/// [`LogitsFilter`](crate::filter::LogitsFilter)s before a single token is
/// sampled using a [`Sampler`](crate::sampler::Sampler).
#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    logits: Vec<f32>,
    indices: Vec<TokenId>,
}

impl Logits {
    /// Create a dense array of token scores.
    pub fn dense(logits: Vec<f32>) -> Logits {
        assert!(logits.len() <= u32::MAX as usize);
        let indices = (0..logits.len() as TokenId).collect();
        Self { logits, indices }
    }

    /// Create a sparse array of token scores.
    ///
    /// `logits` and `indices` must have the same length.
    pub fn sparse(logits: Vec<f32>, indices: Vec<TokenId>) -> Logits {
        assert_eq!(logits.len(), indices.len());
        Self { logits, indices }
    }

    /// Decompose the logits into a (logits, indices) tuple.
    pub fn into_logits_indices(self) -> (Vec<f32>, Vec<TokenId>) {
        (self.logits, self.indices)
    }

    /// Return the number of logits.
    ///
    /// This is equal to `self.logits().len()` and `self.indices().len()`.
    pub fn len(&self) -> usize {
        self.logits.len()
    }

    /// Return true if the set of logits is empty.
    pub fn is_empty(&self) -> bool {
        self.logits.is_empty()
    }

    /// Return the token scores corresponding to the token IDs yielded
    /// by [`indices`](Self::indices).
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Return the token IDs corresponding to the scores yielded by
    /// [`logits`](Self::indices).
    pub fn indices(&self) -> &[TokenId] {
        &self.indices
    }

    /// Return an iterator of `(token_id, score)` tuples.
    pub fn enumerate(&self) -> impl Iterator<Item = (TokenId, f32)> {
        self.indices
            .iter()
            .zip(&self.logits)
            .map(|(token_id, logit)| (*token_id, *logit))
    }
}
