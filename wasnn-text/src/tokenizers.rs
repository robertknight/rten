//! Module containing tokenizers for converting text into sequences of token
//! IDs that can be fed into models.

use std::error::Error;
use std::fmt;
use std::iter::repeat;
use std::ops::Range;

use crate::split::SliceExt;

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

/// An Encoder implements a specific method of converting strings into token IDs
/// using a pre-computed model.
///
/// Encoders are not generally used directly but instead via a wrapping
/// [Tokenizer].
pub trait Encoder {
    /// Look up the numeric ID for a token given its canonical string
    /// representation. This is used eg. for looking up the IDs of special
    /// tokens.
    fn get_token_id(&self, token: &str) -> Result<usize, TokenizerError>;

    /// Convert a token ID to its canonical string representation.
    fn get_token_str(&self, id: usize) -> Result<String, TokenizerError>;

    /// Encode a string into a sequence of token IDs. `on_token` is a callback
    /// with `(offset, token_id)` arguments that should be invoked for each
    /// token produced.
    fn encode_sequence(
        &self,
        text: &str,
        on_token: &mut dyn FnMut(usize, usize),
    ) -> Result<(), TokenizerError>;

    /// Return the canonical strings that correspond to a set of token IDs.
    fn get_tokens(&self, ids: &[usize]) -> Result<Vec<String>, TokenizerError> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            let token = self.get_token_str(id)?;
            tokens.push(token);
        }
        Ok(tokens)
    }
}

/// Tokenizes text inputs into sequences of token IDs that can be fed to a
/// machine learning model.
///
/// `Tokenizer` wraps an [Encoder] which handles specific methods of encoding of
/// individual sequences (eg. WordPiece, Byte Pair Encoding, Unigram) and adds
/// common functionality such as injecting special tokens, splitting sequences
/// into overlapping chunks and truncating long sequences.
pub struct Tokenizer {
    encoder: Box<dyn Encoder>,
}

impl Tokenizer {
    /// Create a new tokenizer which wraps the given encoder.
    pub fn new<E: Encoder + 'static>(encoder: E) -> Tokenizer {
        Tokenizer {
            encoder: Box::new(encoder),
        }
    }

    /// Return the wrapped encoder.
    pub fn encoder(&self) -> &dyn Encoder {
        self.encoder.as_ref()
    }

    /// Encode one or two sequences into a sequence of tokens.
    pub fn encode<'a>(
        &self,
        input: EncoderInput<'a>,
        options: EncodeOptions,
    ) -> Result<Encoded<'a>, TokenizerError> {
        let cls_token = self.encoder.get_token_id("[CLS]")?;
        let sep_token = self.encoder.get_token_id("[SEP]")?;

        // To simplify the implementation, we tokenize the whole input and
        // just discard all chunks except the first. This could be optimized
        // to only generate one chunk.
        let chunks = self.encode_chunks(input, options)?;

        let chunk = chunks.into_iter().next().unwrap_or_else(|| {
            // If the input is empty after tokenization, generate a single
            // empty chunk.
            let (tokens, offsets) = match input {
                EncoderInput::Item(_) => (vec![cls_token, sep_token], vec![0, 0]),
                EncoderInput::Pair(_) => (vec![cls_token, sep_token, sep_token], vec![0, 0, 0]),
            };
            Encoded::new(input, tokens, offsets, 2)
        });

        Ok(chunk)
    }

    /// Encode one or two sequences into a sequence of tokens.
    ///
    /// The output is split into chunks such that the number of tokens in
    /// each chunk is less than the limit specified in [EncodeOptions].
    pub fn encode_chunks<'a>(
        &self,
        input: EncoderInput<'a>,
        options: EncodeOptions,
    ) -> Result<Vec<Encoded<'a>>, TokenizerError> {
        // Number of non-content tokens added to each chunk.
        let non_content_tokens_per_chunk = match input {
            EncoderInput::Item(_) => 2, // [CLS] .. [SEP]
            EncoderInput::Pair(_) => 3, // [CLS] .. [SEP] .. [SEP]
        };

        let cls_token = self.encoder.get_token_id("[CLS]")?;
        let sep_token = self.encoder.get_token_id("[SEP]")?;

        // Encode the full input sequences.
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let (first_seq, second_seq) = match input {
            EncoderInput::Item(first) => (first, None),
            EncoderInput::Pair((first, second)) => (first, Some(second)),
        };

        self.encoder
            .encode_sequence(first_seq, &mut |offset, token| {
                offsets.push(offset);
                tokens.push(token);
            })?;
        let first_seq_tokens = tokens.len();

        if let Some(second_seq) = second_seq {
            self.encoder
                .encode_sequence(second_seq, &mut |offset, token| {
                    offsets.push(offset + first_seq.len());
                    tokens.push(token);
                })?;
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

                    tokens.push(cls_token);
                    offsets.push(offsets_chunk.first().copied().unwrap());

                    tokens.extend_from_slice(tokens_chunk);
                    offsets.extend_from_slice(offsets_chunk);

                    tokens.push(sep_token);

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
                    tokens.push(cls_token);
                    offsets.push(0);

                    tokens.extend_from_slice(&first_tokens[..first_len]);
                    offsets.extend_from_slice(&first_offsets[..first_len]);

                    tokens.push(sep_token);
                    offsets.push(first.len());

                    let first_seq_len = tokens.len();

                    // Add the second sequence, which changes in each chunk.
                    tokens.extend_from_slice(tokens_chunk);
                    offsets.extend_from_slice(offsets_chunk);

                    // The offset for the final token is the offset of the first
                    // token from the second sequence in the next chunk, or
                    // the concatenated input length if this is the final chunk.
                    tokens.push(sep_token);
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

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::{EncodeOptions, EncoderInput, Tokenizer, WordPiece, WordPieceOptions};

    // The tests below use the WordPiece encoder to exercise common Tokenizer
    // functionality. This is convenient as WordPiece is simple.

    #[test]
    fn test_encode_two_sequences() {
        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "This", "is", "a", "test", "sequence",
        ];
        let encoder = WordPiece::from_vocab(vocab, Default::default());
        let tokenizer = Tokenizer::new(encoder);

        // Two sequences, no subwords.
        let encoded = tokenizer
            .encode(
                ("This is", "a test sequence").into(),
                EncodeOptions::default(),
            )
            .unwrap();
        assert_eq!(
            tokenizer.encoder().get_tokens(encoded.token_ids()).unwrap(),
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

        let encoder = WordPiece::from_vocab(vocab, WordPieceOptions::default());
        let tokenizer = Tokenizer::new(encoder);

        for Case {
            input,
            range,
            expected,
        } in cases
        {
            let encoded = tokenizer.encode(input, EncodeOptions::default()).unwrap();
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
        }

        let cases = [
            // Unbounded chunk size
            Case {
                text: "This is a test sequence",
                max_chunk_len: None,
                overlap: 0,
                tokens: vec![&["[CLS]", "This", "is", "a", "test", "sequence", "[SEP]"]],
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
            },
            // Chunk size that is small enough that there is no room for
            // any content tokens in each chunk.
            Case {
                text: "This is a test sequence",
                max_chunk_len: Some(0),
                overlap: 0,
                tokens: vec![],
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
            },
        ];

        let encoder = WordPiece::from_vocab(vocab, Default::default());
        let tokenizer = Tokenizer::new(encoder);

        for Case {
            text,
            max_chunk_len,
            overlap,
            tokens,
        } in cases
        {
            let options = EncodeOptions {
                max_chunk_len,
                overlap,
            };
            let chunks = tokenizer.encode_chunks(text.into(), options).unwrap();
            let chunk_tokens: Vec<_> = chunks
                .into_iter()
                .map(|c| tokenizer.encoder().get_tokens(c.token_ids()).unwrap())
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
        let encoder = WordPiece::from_vocab(vocab, Default::default());
        let tokenizer = Tokenizer::new(encoder);

        struct Case<'a> {
            query: &'a str,
            context: &'a str,
            max_chunk_len: Option<usize>,
            overlap: usize,
            tokens: Vec<&'a [&'a str]>,
        }

        let cases = [
            // Unbounded chunk size
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language",
                max_chunk_len: None,
                overlap: 0,
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
            },
            // Multiple chunks, no overlap
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language. Its mascot is Ferris.",
                max_chunk_len: Some(13),
                overlap: 0,
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
            },
            // Multiple chunks with overlap
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language. Its mascot is Ferris",
                max_chunk_len: Some(13),
                overlap: 2,
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
            },
            // Chunk size too small for any tokens from the second sequence
            Case {
                query: "What is Rust?",
                context: "Rust is a programming language",
                max_chunk_len: Some(7), // Tokens in query + special tokens (3)
                overlap: 0,
                tokens: vec![],
            },
        ];

        for Case {
            query,
            context,
            max_chunk_len,
            overlap,
            tokens,
        } in cases
        {
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
                .map(|c| tokenizer.encoder().get_tokens(c.token_ids()).unwrap())
                .collect();
            assert_eq!(chunk_tokens, tokens);

            // Check that the generated offsets are correct. Since none of the
            // tokens are subwords, and no normalization is being applied, the
            // source text for every token index should be the same as the
            // token's canonical string.
            for (chunk, chunk_tokens) in chunks.iter().zip(chunk_tokens.into_iter()) {
                for (i, token) in chunk_tokens.into_iter().enumerate() {
                    if !token.starts_with("[") {
                        let text = chunk.text_for_token_range(i..i + 1).map(|t| t.trim());
                        assert_eq!(text, Some(token).as_deref());
                    }
                }
            }
        }
    }
}
