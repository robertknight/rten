//! Module containing tokenizers for converting text into sequences of token
//! IDs that can be fed into models.

use std::error::Error;
use std::fmt;
use std::iter::repeat;
use std::ops::Range;

mod wordpiece;
pub use wordpiece::{WordPiece, WordPieceOptions};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EncoderInput<'a> {
    /// Encoder input with a single sequence.
    Item(&'a str),

    /// Encoder input with a pair of sequences. Used in tasks such as extractive
    /// question answering, where the sequence is `(query, context)`.
    Pair((&'a str, &'a str)),
}

/// Construct a tokenizer input with a single sequence.
impl<'a> From<&'a str> for EncoderInput<'a> {
    fn from(val: &'a str) -> EncoderInput<'a> {
        EncoderInput::Item(val)
    }
}

/// Construct a tokenizer input with a pair of sequences.
impl<'a> From<(&'a str, &'a str)> for EncoderInput<'a> {
    fn from(val: (&'a str, &'a str)) -> EncoderInput<'a> {
        EncoderInput::Pair(val)
    }
}

/// Output producd by a [Tokenizer::encode] implementation.
///
/// Use [Encoded::token_ids] to get the token IDs to feed to a model, and
/// [Encoded::text_for_token_range] to map token ID ranges back to the
/// corresponding input text.
#[derive(Debug)]
pub struct Encoded<'a> {
    input: EncoderInput<'a>,
    token_ids: Vec<usize>,

    /// Number of tokens in `token_ids` that were generated from the first
    /// sequence in the input. This includes the `[CLS]` and `[SEP]` tokens
    /// which come before and after the sequence respectively.
    first_seq_tokens: usize,

    /// Offsets of text corresponding to tokens in the input string. When the
    /// input contains two sentences, the offsets are relative to the string
    /// that a particular input that a token comes from.
    token_offsets: Vec<usize>,
}

impl<'a> Encoded<'a> {
    fn new(
        input: EncoderInput<'a>,
        ids: Vec<usize>,
        offsets: Vec<usize>,
        first_seq_tokens: usize,
    ) -> Encoded<'a> {
        Encoded {
            input,
            token_ids: ids,
            token_offsets: offsets,
            first_seq_tokens,
        }
    }

    /// Return the sequence of token IDs that the input was tokenized into.
    pub fn token_ids(&self) -> &[usize] {
        &self.token_ids
    }

    /// Return the byte offsets of the start of each token in the input
    /// sequence. If the input contained two sequences, the offsets are assigned
    /// as if the two sequences were concatenated.
    pub fn token_offsets(&self) -> &[usize] {
        &self.token_offsets
    }

    /// Return an iterator of the inputs for the `token_type_ids` input field
    /// in the model, if it has one.
    pub fn token_type_ids(&self) -> impl Iterator<Item = usize> {
        let second_seq_tokens = self.token_ids.len() - self.first_seq_tokens;
        repeat(0)
            .take(self.first_seq_tokens)
            .chain(repeat(1).take(second_seq_tokens))
    }

    /// Return the text from the input sequence(s) that corresponds to a range
    /// of token indices. If the input contained two sequences, the range must
    /// lie entirely within one of them.
    pub fn text_for_token_range(&self, range: Range<usize>) -> Option<&'a str> {
        let start_offset = self.token_offsets.get(range.start).copied()?;
        let input_len = match self.input {
            EncoderInput::Item(item) => item.len(),
            EncoderInput::Pair((query, context)) => query.len() + context.len(),
        };

        let end_offset = if range.end == self.token_offsets.len() {
            input_len
        } else {
            self.token_offsets.get(range.end).copied()?
        };

        match self.input {
            EncoderInput::Item(item) => item.get(start_offset..end_offset),
            EncoderInput::Pair((query, context)) => {
                if end_offset <= query.len() {
                    query.get(start_offset..end_offset)
                } else {
                    let offset = query.len();
                    context.get(start_offset - offset..end_offset - offset)
                }
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct EncodeOptions {
    /// Maximum number of tokens in each chunk, including any special tokens
    /// (eg. `[CLS]`, `[SEP]`) that are added.
    pub max_chunk_len: Option<usize>,

    /// The number of tokens that a chunk will overlap with the previous chunk.
    pub overlap: usize,
}

/// Common interface for tokenizers that take a sequence or pair of sequences
/// as input, and convert them to a sequence of token IDs and offsets.
pub trait Tokenizer {
    /// Encode one or two sequences into a sequence of tokens.
    fn encode<'a>(
        &self,
        input: EncoderInput<'a>,
        options: EncodeOptions,
    ) -> Result<Encoded<'a>, TokenizerError>;

    /// Encode one or two sequences into a sequence of tokens.
    ///
    /// The output is split into chunks such that the number of tokens in
    /// each chunk is less than the limit specified in [EncodeOptions].
    fn encode_chunks<'a>(
        &self,
        input: EncoderInput<'a>,
        options: EncodeOptions,
    ) -> Result<Vec<Encoded<'a>>, TokenizerError>;
}

/// Error type returned when tokenizing a string.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenizerError {
    /// A token was not found in the vocabulary.
    MissingToken(String),
    /// No token with a given ID exists in the vocabulary.
    InvalidTokenId(usize),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingToken(ref token) => write!(f, "missing vocab token {}", token),
            Self::InvalidTokenId(id) => write!(f, "unknown token id {}", id),
        }
    }
}

impl Error for TokenizerError {}
