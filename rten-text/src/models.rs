//! Popular tokenization models, including:
//!
//! - WordPiece
//! - Byte Pair Encoding or BPE

mod bpe;
mod wordpiece;

pub use bpe::{merge_pairs_from_lines, patterns, Bpe, BpeError};
pub use wordpiece::{WordPiece, WordPieceOptions};

use crate::tokenizers::{TokenId, TokenizerError};

/// Trait for tokenization models which convert words (or other string pieces)
/// into tokens with numeric IDs.
///
/// Models are not generally used directly but instead via a wrapping
/// [`Tokenizer`](crate::tokenizers::Tokenizer).
pub trait Model {
    /// Look up the numeric ID for a token given its canonical string
    /// representation. This is used eg. for looking up the IDs of special
    /// tokens.
    fn get_token_id(&self, token: &str) -> Result<TokenId, TokenizerError>;

    /// Convert a token ID to its canonical string representation.
    ///
    /// This is the representation of the token used in text-based
    /// representations of the vocabulary, such as the `tokenizer.json` file
    /// for Hugging Face tokenizers.
    ///
    /// For tokenizers such as [`Bpe`] where tokens correspond to sequences of
    /// bytes rather than strings, the canonical string representation is an
    /// encoding of the _bytes_, not the text string that the token logically
    /// corresponds to. To get text strings, pass a sequence of token IDs to
    /// [`decode`](Self::decode) instead.
    fn get_token_str(&self, id: TokenId) -> Result<String, TokenizerError>;

    /// Return the canonical strings that correspond to a sequence of token IDs.
    ///
    /// See [`get_token_str`](Self::get_token_str) for notes on what the
    /// "canonical string" is.
    fn get_tokens(&self, ids: &[TokenId]) -> Result<Vec<String>, TokenizerError> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            let token = self.get_token_str(id)?;
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Encode a string into a sequence of token IDs with source offsets.
    ///
    /// `on_token` is a callback with `(offset, token_id)` arguments that should
    /// be invoked for each token produced.
    fn encode_with_offsets(
        &self,
        text: &str,
        on_token: &mut dyn FnMut(usize, TokenId),
    ) -> Result<(), TokenizerError>;

    /// Encode a string into a sequence of token IDs.
    ///
    /// This is a convenience wrapper around
    /// [`encode_with_offsets`](Self::encode_with_offsets) for cases when the
    /// source offsets are not needed.
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        let mut token_ids = Vec::new();
        self.encode_with_offsets(text, &mut |_offset, token_id| token_ids.push(token_id))?;
        Ok(token_ids)
    }

    /// Decode a sequence of token IDs to a text string.
    ///
    /// For tokenizers which operate on byte sequences (eg. [`Bpe`]) this can
    /// fail if the token IDs don't correspond to a complete UTF-8 sequence.
    /// In that case the solution is to accumulate more token IDs and then
    /// retry decoding.
    ///
    /// Special tokens are decoded into their canonical string representations
    /// as returned by [`get_token_str`](Self::get_token_str).
    fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError>;
}
