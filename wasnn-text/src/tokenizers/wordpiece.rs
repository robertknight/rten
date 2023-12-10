use std::collections::HashMap;

use super::{EncodeOptions, Encoded, EncoderInput, Tokenizer, TokenizerError};
use crate::normalizer::Normalizer;
use crate::split::{SliceExt, SplitExt};

use unicode_categories::UnicodeCategories;

/// WordPiece tokenizer [1] used by BERT [2] models.
///
/// [1] Schuster, Mike, and Kaisuke Nakajima. "Japanese and korean voice
///     search." 2012 IEEE international conference on acoustics, speech and signal
///     processing (ICASSP). IEEE, 2012. Accessed at
///     https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf
///
/// [2] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional
///     transformers for language understanding." arXiv preprint arXiv:1810.04805
///     (2018). https://arxiv.org/abs/1810.04805
pub struct WordPiece {
    normalizer: Option<Normalizer>,
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    subword_prefix: String,
    max_word_len: usize,
}

#[derive(Default, Clone)]
pub struct WordPieceOptions {
    /// The normalizer that handles Unicode normalization, lower-casing the
    /// input etc.
    pub normalizer: Option<Normalizer>,

    /// The maximum length of words that can be tokenized. Any words longer than
    /// this are tokenized as `[UNK]`.
    ///
    /// Defaults to 100.
    pub max_word_len: Option<usize>,
}

impl WordPiece {
    /// Construct a WordPiece tokenizer from a vocabulary.
    ///
    /// The index of each entry in `vocab` is used as the token ID.
    pub fn from_vocab(vocab: &[&str], options: WordPieceOptions) -> WordPiece {
        let token_to_id: HashMap<_, _> = vocab
            .iter()
            .enumerate()
            .map(|(pos, token)| (token.to_string(), pos))
            .collect();
        let id_to_token: Vec<_> = vocab.iter().map(|s| s.to_string()).collect();
        let subword_prefix = "##".to_string();

        WordPiece {
            normalizer: options.normalizer,
            token_to_id,
            subword_prefix,
            max_word_len: options.max_word_len.unwrap_or(100),
            id_to_token,
        }
    }

    /// Return the token ID that corresponds to a given string.
    pub fn get_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    /// Return the canonical string that corresponds to a set of token IDs.
    pub fn get_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|t| t.as_str())
    }

    /// Return the canonical strings that correspond to a set of token IDs.
    pub fn get_tokens(&self, ids: &[usize]) -> Option<Vec<&str>> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id) {
                tokens.push(token.as_str());
            } else {
                return None;
            }
        }
        Some(tokens)
    }

    /// Tokenize a string. Invokes the `on_token` callback with `(offset,
    /// token_id)` arguments for each token produced.
    fn encode_sequence<F: FnMut(usize, usize)>(
        &self,
        text: &str,
        mut on_token: F,
    ) -> Result<(), TokenizerError> {
        let mut tmp_buf = String::with_capacity(self.max_word_len);

        // Apply normalization to the input text.
        let (text, normalized_to_source_offsets) = match &self.normalizer {
            None => (text.to_string(), None),
            Some(normalizer) => {
                let (normalized_text, offsets) = normalizer.normalize(text);
                (normalized_text, Some(offsets))
            }
        };

        // Map an offset into the normalized string into an offset in the source
        // string.
        let map_offset = |offset: usize| {
            if let Some(mappings) = &normalized_to_source_offsets {
                mappings
                    .get(offset)
                    .copied()
                    .expect("invalid normalized offset")
            } else {
                offset
            }
        };

        let is_punc_or_space =
            |ch: char| ch.is_ascii_punctuation() || ch.is_punctuation() || ch.is_whitespace();
        let words = text.split_keep_delimeters(is_punc_or_space);
        let mut offset = 0;

        macro_rules! add_unknown_token {
            () => {
                let unknown_token = self.get_required_token("[UNK]")?;
                on_token(map_offset(offset), unknown_token);
            };
        }

        for word in words {
            if word.trim().is_empty() {
                offset += word.len();
                continue;
            }

            if word.chars().count() > self.max_word_len {
                add_unknown_token!();
                continue;
            }

            let mut remainder = word;
            let mut word_tokens = 0;
            while !remainder.is_empty() {
                // Find longest prefix of `remainder` that is in the vocab.
                let mut len = remainder.len();
                while len > 0 {
                    let prefix = if word_tokens > 0 {
                        tmp_buf.clear();
                        tmp_buf.push_str(&self.subword_prefix);
                        tmp_buf.push_str(&remainder[..len]);
                        &tmp_buf[..]
                    } else {
                        &remainder[..len]
                    };

                    if let Some(id) = self.token_to_id.get(prefix) {
                        on_token(map_offset(offset), *id);
                        remainder = remainder.split_at(len).1;
                        word_tokens += 1;
                        break;
                    } else {
                        let last_char_bytes = prefix.chars().next_back().unwrap().len_utf8();
                        len -= last_char_bytes;
                    }
                }

                if len == 0 {
                    add_unknown_token!();
                    break;
                }
            }

            offset += word.len();
        }
        Ok(())
    }

    fn get_required_token(&self, tok: &str) -> Result<usize, TokenizerError> {
        self.token_to_id
            .get(tok)
            .copied()
            .ok_or(TokenizerError::MissingToken(tok.to_string()))
    }
}

impl Tokenizer for WordPiece {
    fn encode<'a>(
        &self,
        input: EncoderInput<'a>,
        options: EncodeOptions,
    ) -> Result<Encoded<'a>, TokenizerError> {
        let cls_token = self.get_required_token("[CLS]")?;
        let sep_token = self.get_required_token("[SEP]")?;

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

    fn encode_chunks<'a>(
        &self,
        input: EncoderInput<'a>,
        options: EncodeOptions,
    ) -> Result<Vec<Encoded<'a>>, TokenizerError> {
        // Number of non-content tokens added to each chunk.
        let non_content_tokens_per_chunk = match input {
            EncoderInput::Item(_) => 2, // [CLS] .. [SEP]
            EncoderInput::Pair(_) => 3, // [CLS] .. [SEP] .. [SEP]
        };

        let cls_token = self.get_required_token("[CLS]")?;
        let sep_token = self.get_required_token("[SEP]")?;

        // Encode the full input sequences.
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let (first_seq, second_seq) = match input {
            EncoderInput::Item(first) => (first, None),
            EncoderInput::Pair((first, second)) => (first, Some(second)),
        };

        self.encode_sequence(first_seq, |offset, token| {
            offsets.push(offset);
            tokens.push(token);
        })?;
        let first_seq_tokens = tokens.len();

        if let Some(second_seq) = second_seq {
            self.encode_sequence(second_seq, |offset, token| {
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

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::{EncodeOptions, EncoderInput, Tokenizer, WordPiece, WordPieceOptions};
    use crate::normalizer::{Normalizer, NormalizerOptions};

    #[test]
    fn test_wordpiece_encoder() {
        struct Case<'a> {
            text: &'a str,
            tokens: &'a [&'a str],
        }

        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "This", "is", "a", "test", "sequence", "Word", "##Piece",
            "Piece", "of", "pie", ".", "!", "?", "Hey", "Hello", "the", "game", "is", "set", "in",
            "Faerûn",
        ];
        let encoder = WordPiece::from_vocab(vocab, Default::default());

        let cases = [
            // Single sequence, no subwords.
            Case {
                text: "This is a test sequence",
                tokens: &["[CLS]", "This", "is", "a", "test", "sequence", "[SEP]"],
            },
            Case {
                text: "Piece of pie",
                tokens: &["[CLS]", "Piece", "of", "pie", "[SEP]"],
            },
            // Sequence with unknown word.
            Case {
                text: "This is unknown sequence",
                tokens: &["[CLS]", "This", "is", "[UNK]", "sequence", "[SEP]"],
            },
            // Sequence with subwords.
            Case {
                text: "WordPiece",
                tokens: &["[CLS]", "Word", "##Piece", "[SEP]"],
            },
            // Empty sequence.
            Case {
                text: "",
                tokens: &["[CLS]", "[SEP]"],
            },
            // Punctuation
            Case {
                text: "Hey! Hello?",
                tokens: &["[CLS]", "Hey", "!", "Hello", "?", "[SEP]"],
            },
            // Word that exceeds length limit.
            Case {
                // note that "a" on its own is in the vocab
                text: &"a".repeat(101),
                tokens: &["[CLS]", "[UNK]", "[SEP]"],
            },
            // Chars requiring multiple bytes in UTF-8
            Case {
                text: "the game is set in Faerûn",
                tokens: &["[CLS]", "the", "game", "is", "set", "in", "Faerûn", "[SEP]"],
            },
        ];

        for Case { text, tokens } in cases {
            let encoded = encoder
                .encode(text.into(), EncodeOptions::default())
                .unwrap();
            assert_eq!(encoder.get_tokens(encoded.token_ids()).unwrap(), tokens);
            assert!(encoded.token_type_ids().all(|ttid| ttid == 0));
        }
    }

    #[test]
    fn test_wordpiece_max_word_len() {
        let vocab = &["[CLS]", "[SEP]", "[UNK]", "foo", "##bar", "##foo"];
        let encoder = WordPiece::from_vocab(
            vocab,
            WordPieceOptions {
                max_word_len: Some(6),
                ..Default::default()
            },
        );

        // The third word should be tokenized to `[UNK]` because it exceeds
        // `max_word_len`.
        let text = "foobar foofoo foobarfoo";
        let encoded = encoder
            .encode(text.into(), EncodeOptions::default())
            .unwrap();

        assert_eq!(
            encoder.get_tokens(encoded.token_ids()).unwrap(),
            &["[CLS]", "foo", "##bar", "foo", "##foo", "[UNK]", "[SEP]"]
        );
    }

    #[test]
    fn test_wordpiece_encoder_two_sequences() {
        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "This", "is", "a", "test", "sequence",
        ];
        let encoder = WordPiece::from_vocab(vocab, Default::default());

        // Two sequences, no subwords.
        let encoded = encoder
            .encode(
                ("This is", "a test sequence").into(),
                EncodeOptions::default(),
            )
            .unwrap();
        assert_eq!(
            encoder.get_tokens(encoded.token_ids()).unwrap(),
            &["[CLS]", "This", "is", "[SEP]", "a", "test", "sequence", "[SEP]"]
        );

        let token_type_ids: Vec<_> = encoded.token_type_ids().collect();
        assert_eq!(token_type_ids, &[0, 0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn test_wordpiece_encoder_lowercase() {
        struct Case<'a> {
            text: &'a str,
            tokens: &'a [&'a str],
        }

        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "this", "is", "a", "test", "sequence",
        ];
        let encoder = WordPiece::from_vocab(
            vocab,
            WordPieceOptions {
                normalizer: Some(Normalizer::new(NormalizerOptions {
                    lowercase: true,
                    ..Default::default()
                })),
                ..Default::default()
            },
        );

        let cases = [
            // Single sequence, no subwords.
            Case {
                text: "this is a test sequence",
                tokens: &["[CLS]", "this", "is", "a", "test", "sequence", "[SEP]"],
            },
            Case {
                text: "THIS IS A TEST SEQUENCE",
                tokens: &["[CLS]", "this", "is", "a", "test", "sequence", "[SEP]"],
            },
        ];

        for Case { text, tokens } in cases {
            let encoded = encoder
                .encode(text.into(), EncodeOptions::default())
                .unwrap();
            assert_eq!(encoder.get_tokens(encoded.token_ids()).unwrap(), tokens);
            assert!(encoded.token_type_ids().all(|ttid| ttid == 0));
        }
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

        for Case {
            input,
            range,
            expected,
        } in cases
        {
            let encoded = encoder.encode(input, EncodeOptions::default()).unwrap();
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
        let encoder = WordPiece::from_vocab(vocab, Default::default());

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
            let chunks = encoder.encode_chunks(text.into(), options).unwrap();
            let chunk_tokens: Vec<_> = chunks
                .into_iter()
                .map(|c| encoder.get_tokens(c.token_ids()).unwrap())
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
            let chunks = encoder
                .encode_chunks((query, context).into(), options)
                .unwrap();
            let chunk_tokens: Vec<_> = chunks
                .iter()
                .map(|c| encoder.get_tokens(c.token_ids()).unwrap())
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
                        assert_eq!(text, Some(token));
                    }
                }
            }
        }
    }
}
