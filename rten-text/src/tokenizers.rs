//! Tokenizers for converting text into sequences of token IDs.
//!
//! There are two ways to construct a tokenizer:
//!
//! 1. Load a preconfigured tokenizer from JSON, using [`Tokenizer::from_json`].
//!    This crate supports a subset of the `tokenizer.json` format that
//!    Hugging Face Tokenizers generates.
//!
//! 2. Manually configure a [`Tokenizer`] by creating an [`Model`] implementation,
//!    such as [`WordPiece`] and then wrap it with a tokenizer using
//!    [`Tokenizer::new`].

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::iter::repeat;
use std::ops::Range;

use crate::models::{merge_pairs_from_lines, Bpe, BpeError, WordPiece};
use crate::normalizer::{BertNormalizer, BertNormalizerOptions, Normalizer};
use crate::pretokenizers::{ByteLevelPreTokenizer, PreTokenizeError, PreTokenizer};
use crate::split::SliceExt;

mod json;

/// Input sequences for [`Tokenizer::encode`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EncoderInput<'a> {
    /// Input with a single sequence.
    Item(&'a str),

    /// Input with a pair of sequences. Used in tasks such as extractive
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

/// Integer type used to represent token IDs.
pub type TokenId = u32;

/// Output produced by a [`Tokenizer::encode`] implementation.
///
/// Use [`Encoded::token_ids`] to get the token IDs to feed to a model, and
/// [`Encoded::text_for_token_range`] to map token ID ranges back to the
/// corresponding input text.
#[derive(Debug)]
pub struct Encoded<'a> {
    input: EncoderInput<'a>,
    token_ids: Vec<TokenId>,

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
        ids: Vec<TokenId>,
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
    pub fn token_ids(&self) -> &[TokenId] {
        &self.token_ids
    }

    /// Consume `self` and return a list of token IDs.
    ///
    /// This is a convenient way to discard other information from the encoded
    /// output and get the token IDs as an owned vector.
    pub fn into_token_ids(self) -> Vec<TokenId> {
        self.token_ids
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

/// Options that control chunking and truncation by [`Tokenizer::encode`] and
/// [`Tokenizer::encode_chunks`].
#[derive(Clone, Default)]
pub struct EncodeOptions {
    /// Maximum number of tokens in each chunk, including any special tokens
    /// (eg. `[CLS]`, `[SEP]`) that are added.
    pub max_chunk_len: Option<usize>,

    /// The number of tokens that a chunk will overlap with the previous chunk.
    pub overlap: usize,
}

/// Trait for tokenization models which convert words (or other string pieces)
/// into tokens with numeric IDs.
///
/// Models are not generally used directly but instead via a wrapping
/// [`Tokenizer`].
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

/// Errors returned by [`Tokenizer::from_json`].
#[derive(Debug)]
pub enum FromJsonError {
    /// There was an error loading a BPE tokenizer.
    BpeError(BpeError),
    /// There was an error decoding the JSON data.
    JsonError(serde_json::Error),
    /// The model type isn't supported by this crate.
    UnsupportedModel,
}

impl fmt::Display for FromJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BpeError(err) => write!(f, "BPE tokenizer error: {}", err),
            Self::JsonError(err) => write!(f, "JSON error {}", err),
            Self::UnsupportedModel => write!(f, "unsupported model type"),
        }
    }
}

impl Error for FromJsonError {}

/// Configuration for a [`Tokenizer`].
#[derive(Clone, Default)]
pub struct TokenizerOptions<'a> {
    /// Token added at the start of the output. For BERT models, this is the
    /// `[CLS]` token.
    pub cls_token: Option<&'a str>,

    /// Token added after each encoded sequence in the output. For BERT models,
    /// this is the `[SEP]` token.
    pub sep_token: Option<&'a str>,
}

/// Tokenizes text inputs into sequences of token IDs that can be fed to a
/// machine learning model.
///
/// `Tokenizer` wraps a [`Model`] which handles specific methods of encoding of
/// individual sequences (eg. WordPiece, Byte Pair Encoding, Unigram) and adds
/// common functionality such as injecting special tokens, splitting sequences
/// into overlapping chunks and truncating long sequences.
pub struct Tokenizer {
    normalizer: Option<Box<dyn Normalizer>>,
    pretokenizer: Option<Box<dyn PreTokenizer>>,
    model: Box<dyn Model>,

    /// Token added at start of output.
    cls_token: Option<String>,

    /// Token added after end of each sequence.
    sep_token: Option<String>,
}

impl Tokenizer {
    /// Create a new tokenizer which wraps the given model.
    pub fn new<M: Model + 'static>(model: M, options: TokenizerOptions) -> Tokenizer {
        Tokenizer {
            model: Box::new(model),
            pretokenizer: None,
            normalizer: None,
            cls_token: options.cls_token.map(|t| t.to_string()),
            sep_token: options.sep_token.map(|t| t.to_string()),
        }
    }

    /// Configure the normalizer used by this tokenizer.
    pub fn with_normalizer(mut self, normalizer: Box<dyn Normalizer>) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Configure the pre-tokenizer used by this tokenizer.
    pub fn with_pre_tokenizer(mut self, pre_tokenizer: Box<dyn PreTokenizer>) -> Self {
        self.pretokenizer = Some(pre_tokenizer);
        self
    }

    /// Load a tokenizer from the contents of a Hugging Face `tokenizer.json`
    /// file.
    pub fn from_json(json: &str) -> Result<Tokenizer, FromJsonError> {
        let tokenizer_json = json::from_json(json).map_err(FromJsonError::JsonError)?;
        Self::from_parsed_json(tokenizer_json)
    }

    fn from_parsed_json(json: json::TokenizerJson) -> Result<Tokenizer, FromJsonError> {
        let normalizer: Option<Box<dyn Normalizer>> = json.normalizer.map(|normalizer| {
            let normalizer: Box<dyn Normalizer> = match normalizer {
                json::Normalizer::Bert(bert_norm) => {
                    Box::new(BertNormalizer::new(BertNormalizerOptions {
                        lowercase: bert_norm.lowercase,
                        strip_accents: bert_norm.strip_accents.unwrap_or(bert_norm.lowercase),
                    }))
                }

                // Dummy implementation of NFC normalization.
                json::Normalizer::Nfc => Box::new(BertNormalizer::new(BertNormalizerOptions {
                    lowercase: false,
                    strip_accents: false,
                })),
            };
            normalizer
        });

        let mut tokenizer = match json.model {
            json::Model::Bpe(model) => {
                let added_tokens: HashMap<TokenId, String> = json
                    .added_tokens
                    .as_ref()
                    .map(|tokens| {
                        tokens
                            .iter()
                            .map(|token| (token.id, token.content.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                let merges: Vec<(&str, &str)> = match &model.merges {
                    json::MergeList::Legacy(lines) => merge_pairs_from_lines(lines),
                    json::MergeList::Tuple(pairs) => pairs
                        .iter()
                        .map(|(a, b)| (a.as_str(), b.as_str()))
                        .collect(),
                };
                let model = Bpe::new(
                    &merges,
                    Some(model.vocab),
                    added_tokens,
                    model.end_of_word_suffix,
                )
                .map_err(FromJsonError::BpeError)?;

                let tokenizer = Tokenizer::new(
                    model,
                    TokenizerOptions {
                        cls_token: None,
                        sep_token: None,
                    },
                )
                .with_pre_tokenizer(Box::new(ByteLevelPreTokenizer::gpt2()));

                Ok(tokenizer)
            }
            json::Model::WordPiece(model) => {
                let model = WordPiece::from_vocab(model.vocab, Default::default());
                let tokenizer = Tokenizer::new(
                    model,
                    TokenizerOptions {
                        cls_token: Some("[CLS]"),
                        sep_token: Some("[SEP]"),
                    },
                );

                Ok(tokenizer)
            }
        }?;

        if let Some(normalizer) = normalizer {
            tokenizer = tokenizer.with_normalizer(normalizer)
        }

        Ok(tokenizer)
    }

    #[deprecated = "`encoder` was renamed to `model`"]
    pub fn encoder(&self) -> &dyn Model {
        self.model()
    }

    /// Return the model used to convert string pieces to token IDs.
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    fn cls_token(&self) -> Result<Option<TokenId>, TokenizerError> {
        self.cls_token
            .as_ref()
            .map(|cls| self.model.get_token_id(cls.as_str()))
            .transpose()
    }

    fn sep_token(&self) -> Result<Option<TokenId>, TokenizerError> {
        self.sep_token
            .as_ref()
            .map(|sep| self.model.get_token_id(sep.as_str()))
            .transpose()
    }

    /// Encode one or two sequences into a sequence of tokens.
    ///
    /// The input can be an `&str` or tuple of `(&str, &str)`.
    ///
    /// In addition to token IDs, the result also includes information about
    /// the corresponding offsets in the source text.
    pub fn encode<'a, I: Into<EncoderInput<'a>>>(
        &self,
        input: I,
        options: Option<EncodeOptions>,
    ) -> Result<Encoded<'a>, TokenizerError> {
        let options = options.unwrap_or_default();
        let input: EncoderInput = input.into();

        let cls_token = self.cls_token()?;
        let sep_token = self.sep_token()?;

        // To simplify the implementation, we tokenize the whole input and
        // just discard all chunks except the first. This could be optimized
        // to only generate one chunk.
        let chunks = self.encode_chunks(input, options)?;

        let chunk = chunks.into_iter().next().unwrap_or_else(|| {
            // If the input is empty after tokenization, generate a single
            // empty chunk.
            let mut tokens = Vec::new();
            let mut offsets = Vec::new();
            let mut first_seq_tokens = 0;

            if let Some(cls_token) = cls_token {
                tokens.push(cls_token);
                offsets.push(0);
                first_seq_tokens += 1;
            }
            if let Some(sep_token) = sep_token {
                tokens.push(sep_token);
                offsets.push(0);
                first_seq_tokens += 1;

                if matches!(input, EncoderInput::Pair(_)) {
                    tokens.push(sep_token);
                    offsets.push(0);
                }
            }

            Encoded::new(input, tokens, offsets, first_seq_tokens)
        });

        Ok(chunk)
    }

    /// Encode one or two sequences into a sequence of tokens.
    ///
    /// The output is split into chunks such that the number of tokens in
    /// each chunk is less than the limit specified in [`EncodeOptions`].
    pub fn encode_chunks<'a>(
        &self,
        input: EncoderInput<'a>,
        options: EncodeOptions,
    ) -> Result<Vec<Encoded<'a>>, TokenizerError> {
        let cls_token = self.cls_token()?;
        let sep_token = self.sep_token()?;

        let has_cls = cls_token.is_some() as usize;
        let has_sep = sep_token.is_some() as usize;

        // Number of non-content tokens added to each chunk.
        let non_content_tokens_per_chunk = has_cls
            + match input {
                EncoderInput::Item(_) => has_sep,     // [CLS] .. [SEP]
                EncoderInput::Pair(_) => has_sep * 2, // [CLS] .. [SEP] .. [SEP]
            };

        // Encode the full input sequences.
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let (first_seq, second_seq) = match input {
            EncoderInput::Item(first) => (first, None),
            EncoderInput::Pair((first, second)) => (first, Some(second)),
        };

        // Normalize text and return a (normalized_text, offset_mapping) pair.
        let normalize_text = |text: &str| match &self.normalizer {
            None => (text.to_string(), None),
            Some(normalizer) => {
                let (normalized_text, offsets) = normalizer.normalize(text);
                (normalized_text, Some(offsets))
            }
        };

        // Map an offset into the normalized string into an offset in the source
        // string.
        let map_offset = |offset: usize, mapping: Option<&[usize]>| {
            if let Some(mappings) = &mapping {
                mappings
                    .get(offset)
                    .copied()
                    .expect("invalid normalized offset")
            } else {
                offset
            }
        };

        let (normalized, offset_map) = normalize_text(first_seq);
        let chunks = self
            .pretokenizer
            .as_ref()
            .map(|pt| pt.pretokenize(&normalized))
            .transpose()
            .map_err(TokenizerError::PreTokenizeError)?
            .unwrap_or(Vec::from([normalized.as_str()]));

        for chunk in chunks {
            let base_offset = normalized
                .as_bytes()
                .subslice_offsets(chunk.as_bytes())
                .expect("should be a subslice")
                .start;
            self.model
                .encode_with_offsets(chunk, &mut |offset, token| {
                    offsets.push(base_offset + map_offset(offset, offset_map.as_deref()));
                    tokens.push(token);
                })?;
        }

        let first_seq_tokens = tokens.len();

        if let Some(second_seq) = second_seq {
            let (normalized, offset_map) = normalize_text(second_seq);
            let chunks = self
                .pretokenizer
                .as_ref()
                .map(|pt| pt.pretokenize(&normalized))
                .transpose()
                .map_err(TokenizerError::PreTokenizeError)?
                .unwrap_or(Vec::from([normalized.as_str()]));
            for chunk in chunks {
                let base_offset = normalized
                    .as_bytes()
                    .subslice_offsets(chunk.as_bytes())
                    .expect("should be a subslice")
                    .start;
                self.model
                    .encode_with_offsets(chunk, &mut |offset, token| {
                        offsets.push(
                            first_seq.len()
                                + base_offset
                                + map_offset(offset, offset_map.as_deref()),
                        );
                        tokens.push(token);
                    })?;
            }
        }

        let max_tokens_per_chunk = options
            .max_chunk_len
            .unwrap_or(tokens.len() + non_content_tokens_per_chunk)
            .saturating_sub(non_content_tokens_per_chunk);

        if max_tokens_per_chunk == 0 {
            // We can't "consume" tokens from the input in each chunk, so just
            // return an empty output.
            return Ok(vec![]);
        }

        // Split into chunks.
        let mut chunks = Vec::new();

        match input {
            // For single sequence inputs, create chunks with a maximum of
            // `max_seq_len` tokens each.
            EncoderInput::Item(item) => {
                let all_offsets = &offsets;
                for (chunk_idx, (tokens_chunk, offsets_chunk)) in tokens
                    .chunks_with_overlap(max_tokens_per_chunk, options.overlap)
                    .zip(offsets.chunks_with_overlap(max_tokens_per_chunk, options.overlap))
                    .enumerate()
                {
                    let mut tokens = Vec::new();
                    let mut offsets = Vec::new();

                    if let Some(cls_token) = cls_token {
                        tokens.push(cls_token);
                        offsets.push(offsets_chunk.first().copied().unwrap());
                    }

                    tokens.extend_from_slice(tokens_chunk);
                    offsets.extend_from_slice(offsets_chunk);

                    if let Some(sep_token) = sep_token {
                        tokens.push(sep_token);
                    }

                    // The offset for the final token is the offset of the first
                    // token in the next chunk, or the input length if this
                    // is the final chunk.
                    let chunk_start = chunk_idx * max_tokens_per_chunk;
                    offsets.push(
                        all_offsets
                            .get(chunk_start + offsets_chunk.len())
                            .copied()
                            .unwrap_or(item.len()),
                    );

                    let n_tokens = tokens.len();
                    chunks.push(Encoded::new(input, tokens, offsets, n_tokens));
                }
            }

            // For input sequence pairs, create chunks where the first part is
            // the same for each chunk and has a maximum of `max_seq_len` tokens,
            // and the second part contains chunks of the second sequence,
            // taking up the remaining available space in the chunk.
            EncoderInput::Pair((first, second)) => {
                let (first_tokens, second_tokens) = tokens.split_at(first_seq_tokens);
                let (first_offsets, second_offsets) = offsets.split_at(first_seq_tokens);

                let first_len = first_tokens.len().min(max_tokens_per_chunk);
                let second_len = second_tokens.len().min(max_tokens_per_chunk - first_len);

                if second_len == 0 {
                    // We can't "consume" tokens from the second sequence in
                    // each chunk, so just return an empty output.
                    return Ok(vec![]);
                }

                for (chunk_idx, (tokens_chunk, offsets_chunk)) in second_tokens
                    .chunks_with_overlap(second_len, options.overlap)
                    .zip(second_offsets.chunks_with_overlap(second_len, options.overlap))
                    .enumerate()
                {
                    let mut tokens = Vec::new();
                    let mut offsets = Vec::new();

                    // Add the first sequence. This is the same for every chunk.
                    if let Some(cls_token) = cls_token {
                        tokens.push(cls_token);
                        offsets.push(0);
                    }

                    tokens.extend_from_slice(&first_tokens[..first_len]);
                    offsets.extend_from_slice(&first_offsets[..first_len]);

                    if let Some(sep_token) = sep_token {
                        tokens.push(sep_token);
                        offsets.push(first.len());
                    }

                    let first_seq_len = tokens.len();

                    // Add the second sequence, which changes in each chunk.
                    tokens.extend_from_slice(tokens_chunk);
                    offsets.extend_from_slice(offsets_chunk);

                    // The offset for the final token is the offset of the first
                    // token from the second sequence in the next chunk, or
                    // the concatenated input length if this is the final chunk.
                    if let Some(sep_token) = sep_token {
                        tokens.push(sep_token);
                    }
                    let chunk_start = chunk_idx * second_len;
                    offsets.push(
                        second_offsets
                            .get(chunk_start + offsets_chunk.len())
                            .copied()
                            .unwrap_or(first.len() + second.len()),
                    );

                    chunks.push(Encoded::new(input, tokens, offsets, first_seq_len));
                }
            }
        }

        Ok(chunks)
    }

    /// Decode a sequence of token IDs to a text string.
    ///
    /// For tokenizers which operate on byte sequences (eg. [`Bpe`]) this can
    /// fail if the token IDs don't correspond to a complete UTF-8 sequence.
    /// In that case the solution is to accumulate more token IDs and then
    /// retry decoding.
    ///
    /// Special tokens are decoded into their canonical string representations
    /// as returned by [`Model::get_token_str`].
    pub fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        self.model.decode(ids)
    }
}

/// Error type returned when tokenizing a string.
#[derive(Clone, Debug)]
pub enum TokenizerError {
    /// A token was not found in the vocabulary.
    MissingToken(String),

    /// No token with a given ID exists in the vocabulary.
    InvalidTokenId(TokenId),

    /// An error occurred while performing pre-tokenization to split the input.
    PreTokenizeError(PreTokenizeError),

    /// Splitting the input with a regex failed.
    RegexSplitFailed(Box<fancy_regex::Error>),

    /// There was an error parsing a byte sequence as a UTF-8 string.
    ///
    /// This can arise when working with tokenizers like [`Bpe`] where
    /// individual tokens do not always represent whole characters.
    InvalidUtf8,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingToken(ref token) => write!(f, "missing vocab token {}", token),
            Self::InvalidTokenId(id) => write!(f, "unknown token id {}", id),
            Self::RegexSplitFailed(err) => write!(f, "regex failed {}", err),
            Self::PreTokenizeError(err) => write!(f, "pretokenization error: {}", err),
            Self::InvalidUtf8 => write!(f, "UTF-8 decode failed"),
        }
    }
}

impl Error for TokenizerError {}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs::read_to_string;
    use std::ops::Range;
    use std::path::PathBuf;

    use super::{EncodeOptions, EncoderInput, TokenId, Tokenizer, TokenizerOptions, WordPiece};
    use crate::normalizer::{BertNormalizer, BertNormalizerOptions, Normalizer};
    use serde::Deserialize;

    fn make_wordpiece(vocab: &[&str]) -> WordPiece {
        let vocab: HashMap<_, _> = vocab
            .iter()
            .enumerate()
            .map(|(i, token)| (token.to_string(), i as u32))
            .collect();
        WordPiece::from_vocab(vocab, Default::default())
    }

    fn lowercase_normalizer() -> Box<dyn Normalizer> {
        Box::new(BertNormalizer::new(BertNormalizerOptions {
            lowercase: true,
            ..Default::default()
        }))
    }

    // The tests below use the WordPiece model to exercise common Tokenizer
    // functionality. This is convenient as WordPiece is simple.

    #[test]
    fn test_encode_two_sequences() {
        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "This", "is", "a", "test", "sequence",
        ];
        let model = make_wordpiece(vocab);
        let tokenizer = Tokenizer::new(
            model,
            TokenizerOptions {
                cls_token: Some("[CLS]"),
                sep_token: Some("[SEP]"),
            },
        );

        // Two sequences, no subwords.
        let encoded = tokenizer
            .encode(("This is", "a test sequence"), None)
            .unwrap();
        assert_eq!(
            tokenizer.model().get_tokens(encoded.token_ids()).unwrap(),
            &["[CLS]", "This", "is", "[SEP]", "a", "test", "sequence", "[SEP]"]
        );

        let token_type_ids: Vec<_> = encoded.token_type_ids().collect();
        assert_eq!(token_type_ids, &[0, 0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn test_text_for_token_range() {
        struct Case<'a> {
            input: EncoderInput<'a>,
            range: Range<usize>,
            expected: Option<&'a str>,
        }

        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "This", "is", "a", "test", "sequence", "Word", "##Piece",
            "Piece", "of", "pie", ".", "!", "?", "Hey", "Hello",
        ];

        let cases = [
            // Part of a single sequence
            Case {
                input: "This is a test sequence".into(),
                range: 4..6,
                expected: Some("test sequence"),
            },
            // Whole of a single sentence
            Case {
                input: "This is a test sequence".into(),
                range: 1..6,
                expected: Some("This is a test sequence"),
            },
            // Part of first item in a pair
            Case {
                input: ("This is a test sequence", "Hey Hello").into(),
                range: 4..6,
                expected: Some("test sequence"),
            },
            // Whole of first item in a pair
            Case {
                input: "This is a test sequence".into(),
                range: 1..6,
                expected: Some("This is a test sequence"),
            },
            // Part of second item in a pair
            Case {
                input: ("This is a test sequence", "Hey Hello").into(),
                range: 8..9,
                expected: Some("Hello"),
            },
            // Whole of second item in a pair
            Case {
                input: ("This is a test sequence", "Hey Hello").into(),
                range: 7..9,
                expected: Some("Hey Hello"),
            },
            // Out of bounds range for a single sequence
            Case {
                input: "This is a test sequence".into(),
                range: 4..8,
                expected: None,
            },
            // Out of bounds range for a pair
            Case {
                input: ("This is a test sequence", "Hey Hello").into(),
                range: 7..12,
                expected: None,
            },
            // Range that spans first and second sequences in a pair
            Case {
                input: "This is a test sequence".into(),
                range: 1..8,
                expected: None,
            },
            // Range that intersects special tokens
            Case {
                input: "This is a test sequence".into(),
                range: 0..7,
                expected: Some("This is a test sequence"),
            },
        ];

        let model = make_wordpiece(vocab);
        let tokenizer = Tokenizer::new(
            model,
            TokenizerOptions {
                cls_token: Some("[CLS]"),
                sep_token: Some("[SEP]"),
            },
        );

        for Case {
            input,
            range,
            expected,
        } in cases
        {
            let encoded = tokenizer.encode(input, None).unwrap();
            let text = encoded.text_for_token_range(range.clone());
            assert_eq!(
                text, expected,
                "mismatch for input {:?} with range {:?}",
                input, range
            );
        }
    }

    #[test]
    fn test_encode_chunks_single_sequence() {
        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "This", "is", "a", "test", "sequence",
        ];

        struct Case<'a> {
            text: &'a str,
            max_chunk_len: Option<usize>,
            overlap: usize,
            tokens: Vec<&'a [&'a str]>,
            use_cls_sep: bool,
            lowercase: bool,
        }

        let cases = [
            // Unbounded chunk size
            Case {
                text: "This is a test sequence",
                max_chunk_len: None,
                overlap: 0,
                tokens: vec![&["[CLS]", "This", "is", "a", "test", "sequence", "[SEP]"]],
                use_cls_sep: true,
                lowercase: false,
            },
            // Encode with a normalizer
            Case {
                text: "A TEST SEQUENCE",
                max_chunk_len: None,
                overlap: 0,
                tokens: vec![&["[CLS]", "a", "test", "sequence", "[SEP]"]],
                use_cls_sep: true,
                lowercase: true,
            },
            // Two chunks
            Case {
                text: "This is a test sequence",
                max_chunk_len: Some(5),
                overlap: 0,
                tokens: vec![
                    &["[CLS]", "This", "is", "a", "[SEP]"],
                    &["[CLS]", "test", "sequence", "[SEP]"],
                ],
                use_cls_sep: true,
                lowercase: false,
            },
            // Three chunks
            Case {
                text: "This is a test sequence",
                max_chunk_len: Some(4),
                overlap: 0,
                tokens: vec![
                    &["[CLS]", "This", "is", "[SEP]"],
                    &["[CLS]", "a", "test", "[SEP]"],
                    &["[CLS]", "sequence", "[SEP]"],
                ],
                use_cls_sep: true,
                lowercase: false,
            },
            // Chunk size that is small enough that there is no room for
            // any content tokens in each chunk.
            Case {
                text: "This is a test sequence",
                max_chunk_len: Some(0),
                overlap: 0,
                tokens: vec![],
                use_cls_sep: true,
                lowercase: false,
            },
            // Overlap between chunks
            Case {
                text: "This is a test sequence",
                max_chunk_len: Some(5),
                overlap: 2,
                tokens: vec![
                    &["[CLS]", "This", "is", "a", "[SEP]"],
                    &["[CLS]", "is", "a", "test", "[SEP]"],
                    &["[CLS]", "a", "test", "sequence", "[SEP]"],
                ],
                use_cls_sep: true,
                lowercase: false,
            },
            // No special tokens
            Case {
                text: "This is a test sequence",
                max_chunk_len: None,
                overlap: 0,
                tokens: vec![&["This", "is", "a", "test", "sequence"]],
                use_cls_sep: false,
                lowercase: false,
            },
        ];

        let model = make_wordpiece(vocab);

        for Case {
            text,
            max_chunk_len,
            overlap,
            tokens,
            use_cls_sep,
            lowercase,
        } in cases
        {
            let mut tokenizer = Tokenizer::new(
                model.clone(),
                TokenizerOptions {
                    cls_token: use_cls_sep.then_some("[CLS]"),
                    sep_token: use_cls_sep.then_some("[SEP]"),
                },
            );

            if lowercase {
                tokenizer = tokenizer.with_normalizer(lowercase_normalizer());
            }

            let options = EncodeOptions {
                max_chunk_len,
                overlap,
            };
            let chunks = tokenizer.encode_chunks(text.into(), options).unwrap();
            let chunk_tokens: Vec<_> = chunks
                .into_iter()
                .map(|c| tokenizer.model().get_tokens(c.token_ids()).unwrap())
                .collect();
            assert_eq!(chunk_tokens, tokens);
        }
    }

    #[test]
    fn test_encode_chunks_sequence_pair() {
        let vocab = &[
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "What",
            "is",
            "Rust",
            "?",
            "a",
            "programming",
            "language",
            ".",
            "Its",
            "mascot",
            "is",
            "Ferris",
        ];

        let model = make_wordpiece(vocab);

        struct Case<'a> {
            query: &'a str,
            context: &'a str,
            max_chunk_len: Option<usize>,
            overlap: usize,
            tokens: Vec<&'a [&'a str]>,
            use_sep_cls: bool,
            lowercase: bool,
        }

        let cases = [
            // Unbounded chunk size
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language",
                max_chunk_len: None,
                overlap: 0,
                use_sep_cls: true,
                tokens: vec![&[
                    "[CLS]",
                    "What",
                    "is",
                    "Rust",
                    "?",
                    "[SEP]",
                    "Rust",
                    "is",
                    "a",
                    "programming",
                    "language",
                    "[SEP]",
                ]],
                lowercase: false,
            },
            // Apply normalization to both sequences
            Case {
                query: "PROGRAMMING",
                context: "LANGUAGE",
                max_chunk_len: None,
                overlap: 0,
                use_sep_cls: true,
                tokens: vec![&["[CLS]", "programming", "[SEP]", "language", "[SEP]"]],
                lowercase: true,
            },
            // Multiple chunks, no overlap
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language. Its mascot is Ferris.",
                max_chunk_len: Some(13),
                overlap: 0,
                use_sep_cls: true,
                tokens: vec![
                    &[
                        "[CLS]",
                        "What",
                        "is",
                        "Rust",
                        "?",
                        "[SEP]",
                        "Rust",
                        "is",
                        "a",
                        "programming",
                        "language",
                        ".",
                        "[SEP]",
                    ],
                    &[
                        "[CLS]", "What", "is", "Rust", "?", "[SEP]", "Its", "mascot", "is",
                        "Ferris", ".", "[SEP]",
                    ],
                ],
                lowercase: false,
            },
            // Multiple chunks with overlap
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language. Its mascot is Ferris",
                max_chunk_len: Some(13),
                overlap: 2,
                use_sep_cls: true,
                tokens: vec![
                    &[
                        "[CLS]",
                        "What",
                        "is",
                        "Rust",
                        "?",
                        "[SEP]",
                        "Rust",
                        "is",
                        "a",
                        "programming",
                        "language",
                        ".",
                        "[SEP]",
                    ],
                    &[
                        "[CLS]", "What", "is", "Rust", "?", "[SEP]", "language", ".", "Its",
                        "mascot", "is", "Ferris", "[SEP]",
                    ],
                ],
                lowercase: false,
            },
            // Chunk size too small for any tokens from the second sequence
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language",
                max_chunk_len: Some(7), // Tokens in query + special tokens (3)
                overlap: 0,
                use_sep_cls: true,
                tokens: vec![],
                lowercase: false,
            },
            // No special tokens
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language",
                max_chunk_len: None,
                overlap: 0,
                use_sep_cls: false,
                tokens: vec![&[
                    "What",
                    "is",
                    "Rust",
                    "?",
                    "Rust",
                    "is",
                    "a",
                    "programming",
                    "language",
                ]],
                lowercase: false,
            },
        ];

        for Case {
            query,
            context,
            max_chunk_len,
            overlap,
            tokens,
            use_sep_cls,
            lowercase,
        } in cases
        {
            let mut tokenizer = Tokenizer::new(
                model.clone(),
                TokenizerOptions {
                    cls_token: use_sep_cls.then_some("[CLS]"),
                    sep_token: use_sep_cls.then_some("[SEP]"),
                },
            );

            if lowercase {
                tokenizer = tokenizer.with_normalizer(lowercase_normalizer());
            }

            let options = EncodeOptions {
                max_chunk_len,
                overlap,
                ..Default::default()
            };
            let chunks = tokenizer
                .encode_chunks((query, context).into(), options)
                .unwrap();
            let chunk_tokens: Vec<_> = chunks
                .iter()
                .map(|c| tokenizer.model().get_tokens(c.token_ids()).unwrap())
                .collect();
            assert_eq!(chunk_tokens, tokens);

            // Check that the generated offsets are correct. Since none of the
            // tokens are subwords, and no normalization is being applied, the
            // source text for every token index should be the same as the
            // token's canonical string.
            for (chunk, chunk_tokens) in chunks.iter().zip(chunk_tokens.into_iter()) {
                for (i, token) in chunk_tokens.into_iter().enumerate() {
                    if !token.starts_with("[") {
                        let text = chunk
                            .text_for_token_range(i..i + 1)
                            .map(|t| t.trim())
                            .unwrap();
                        let text = if lowercase {
                            text.to_lowercase()
                        } else {
                            text.to_string()
                        };
                        assert_eq!(text, token);
                    }
                }
            }
        }
    }

    #[derive(Deserialize)]
    struct TokenizerJsonCase {
        text: String,
        token_ids: Vec<TokenId>,
    }

    #[derive(Deserialize)]
    struct TokenizerJsonTest {
        tokenizer: super::json::TokenizerJson,
        cases: Vec<TokenizerJsonCase>,
    }

    fn read_test_json(path: &str) -> Result<TokenizerJsonTest, Box<dyn Error>> {
        let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        abs_path.push("test-data/tokenizer-json/");
        abs_path.push(path);
        let content = read_to_string(abs_path)?;
        let json = serde_json::from_str(&content)?;
        Ok(json)
    }

    #[test]
    fn test_from_json() {
        let paths = ["wordpiece.json", "wordpiece-lower.json"];

        for path in paths.iter() {
            let config = read_test_json(path).unwrap();

            let tokenizer = Tokenizer::from_parsed_json(config.tokenizer).unwrap();
            for case in config.cases {
                let encoded = tokenizer.encode(case.text.as_str(), None).unwrap();
                assert_eq!(encoded.token_ids(), case.token_ids);
            }
        }
    }
}
