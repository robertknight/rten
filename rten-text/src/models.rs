//! Implementations of popular tokenization models including WordPiece and
//! Byte Pair Encoding (BPE).

use std::error::Error;
use std::fmt;

mod bpe;
mod wordpiece;

pub use bpe::{
    Bpe, BpeError, BpeOptions, EncodedByteSlice, EncodedBytes, char_to_byte, merge_pairs_from_lines,
};
pub use wordpiece::{WordPiece, WordPieceOptions};

use crate::tokenizer::TokenId;

/// Errors that occur while encoding text pieces into token IDs, after
/// normalization and pre-tokenization.
#[derive(Clone, Debug, PartialEq)]
pub enum EncodeError {
    /// Encoding a string as a single token failed because no matching ID was
    /// found in the vocabulary.
    TokenIdNotFound(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl Error for EncodeError {}

/// Errors that occur while decoding token IDs into text.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodeError {
    /// The decoded byte sequence does not form a valid UTF-8 string.
    ///
    /// This can arise when working with tokenizers like [`Bpe`] where
    /// individual tokens do not always represent whole Unicode characters.
    ///
    /// If this error is encountered in the middle of a process that generates
    /// tokens, the solution is to accumulate more tokens and then try decoding
    /// again.
    InvalidUtf8,

    /// No token with a given ID exists in the vocabulary.
    InvalidTokenId(TokenId),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTokenId(id) => write!(f, "cannot decode unknown token ID {}", id),
            Self::InvalidUtf8 => write!(f, "decoded tokens do not form valid UTF-8 text"),
        }
    }
}

impl Error for DecodeError {}

/// Trait for tokenization models which convert words (or other string pieces)
/// into tokens with numeric IDs.
///
/// Models are not generally used directly but instead via a wrapping
/// [`Tokenizer`](crate::tokenizer::Tokenizer).
pub trait Model {
    /// Look up the numeric ID for a token given its canonical string
    /// representation. This is used eg. for looking up the IDs of special
    /// tokens.
    fn get_token_id(&self, token: &str) -> Option<TokenId>;

    /// Return the canonical string representation for a token or `None` if
    /// the token ID does not exist in the model's vocabulary.
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
    fn get_token_str(&self, id: TokenId) -> Option<String>;

    /// Return the canonical strings that correspond to a sequence of token IDs.
    ///
    /// See [`get_token_str`](Self::get_token_str) for notes on what the
    /// "canonical string" is.
    fn get_tokens(&self, ids: &[TokenId]) -> Result<Vec<String>, DecodeError> {
        ids.iter()
            .map(|&id| {
                self.get_token_str(id)
                    .ok_or(DecodeError::InvalidTokenId(id))
            })
            .collect()
    }

    /// Encode a string into a sequence of token IDs with source offsets.
    ///
    /// `on_token` is a callback with `(offset, token_id)` arguments that should
    /// be invoked for each token produced.
    fn encode_with_offsets(
        &self,
        text: &str,
        on_token: &mut dyn FnMut(usize, TokenId),
    ) -> Result<(), EncodeError>;

    /// Encode a string into a sequence of token IDs.
    ///
    /// This is a convenience wrapper around
    /// [`encode_with_offsets`](Self::encode_with_offsets) for cases when the
    /// source offsets are not needed.
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, EncodeError> {
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
    fn decode(&self, ids: &[TokenId]) -> Result<String, DecodeError>;
}
