use std::collections::HashMap;

use super::Model;
use crate::tokenizers::{TokenId, TokenizerError};

/// WordPiece tokenizer [^1] used by BERT [^2] models.
///
/// [^1]: Schuster, Mike, and Kaisuke Nakajima. "Japanese and korean voice
///       search." 2012 IEEE international conference on acoustics, speech and signal
///       processing (ICASSP). IEEE, 2012. Accessed at
///       <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf>
///
/// [^2]: Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional
///       transformers for language understanding." arXiv preprint arXiv:1810.04805
///       (2018). <https://arxiv.org/abs/1810.04805>
#[derive(Clone)]
pub struct WordPiece {
    token_to_id: HashMap<String, TokenId>,
    id_to_token: HashMap<TokenId, String>,
    subword_prefix: String,
    max_word_len: usize,
}

/// Configuration for a [`WordPiece`] tokenizer.
#[derive(Debug, Default, Clone)]
pub struct WordPieceOptions {
    /// The maximum length of words that can be tokenized. Any words longer than
    /// this are tokenized as `[UNK]`.
    ///
    /// Defaults to 100.
    pub max_word_len: Option<usize>,
}

impl WordPiece {
    /// Construct a WordPiece tokenizer from a vocabulary.
    ///
    /// `vocab` is a mapping from word piece to token ID.
    pub fn from_vocab(vocab: HashMap<String, TokenId>, options: WordPieceOptions) -> WordPiece {
        let id_to_token: HashMap<TokenId, String> =
            vocab.iter().map(|(k, v)| (*v, k.to_string())).collect();

        let subword_prefix = "##".to_string();

        WordPiece {
            token_to_id: vocab,
            subword_prefix,
            max_word_len: options.max_word_len.unwrap_or(100),
            id_to_token,
        }
    }
}

impl Model for WordPiece {
    fn encode_with_offsets(
        &self,
        word: &str,
        on_token: &mut dyn FnMut(usize, TokenId),
    ) -> Result<(), TokenizerError> {
        let mut tmp_buf = String::with_capacity(self.max_word_len);
        let mut offset = 0;

        macro_rules! add_unknown_token {
            () => {
                let unknown_token = self.get_token_id("[UNK]")?;
                on_token(offset, unknown_token);
            };
        }

        if word.trim().is_empty() {
            return Ok(());
        }

        if word.chars().count() > self.max_word_len {
            add_unknown_token!();
            return Ok(());
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
                    on_token(offset, *id);
                    offset += prefix.len();
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

        Ok(())
    }

    fn get_token_str(&self, id: TokenId) -> Result<String, TokenizerError> {
        self.id_to_token
            .get(&id)
            .cloned()
            .ok_or(TokenizerError::InvalidTokenId(id))
    }

    fn get_token_id(&self, tok: &str) -> Result<TokenId, TokenizerError> {
        self.token_to_id
            .get(tok)
            .copied()
            .ok_or(TokenizerError::MissingToken(tok.to_string()))
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        let token_strings = self.get_tokens(ids)?;
        Ok(token_strings.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::models::{WordPiece, WordPieceOptions};
    use crate::normalizer::{BertNormalizer, BertNormalizerOptions, Normalizer};
    use crate::pre_tokenizers::BertPreTokenizer;
    use crate::tokenizers::{Tokenizer, TokenizerOptions};

    fn create_tokenizer(
        vocab: &[&str],
        normalizer: Option<Box<dyn Normalizer>>,
        options: WordPieceOptions,
    ) -> Tokenizer {
        let vocab: HashMap<_, _> = vocab
            .iter()
            .enumerate()
            .map(|(i, token)| (token.to_string(), i as u32))
            .collect();
        let model = WordPiece::from_vocab(vocab, options);
        let mut tokenizer = Tokenizer::new(
            model,
            TokenizerOptions {
                cls_token: Some("[CLS]"),
                sep_token: Some("[SEP]"),
            },
        )
        .with_pre_tokenizer(Box::new(BertPreTokenizer::new()));

        if let Some(normalizer) = normalizer {
            tokenizer = tokenizer.with_normalizer(normalizer);
        }

        tokenizer
    }

    #[test]
    fn test_wordpiece_model() {
        struct Case<'a> {
            text: &'a str,
            tokens: &'a [&'a str],
        }

        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "This", "is", "a", "test", "sequence", "Word", "##Piece",
            "Piece", "of", "pie", ".", "!", "?", "Hey", "Hello", "the", "game", "is", "set", "in",
            "Faerûn",
        ];
        let tokenizer = create_tokenizer(vocab, None, Default::default());

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
            let encoded = tokenizer.encode(text, None).unwrap();
            assert_eq!(
                tokenizer.model().get_tokens(encoded.token_ids()).unwrap(),
                tokens
            );
            assert!(encoded.token_type_ids().all(|ttid| ttid == 0));
        }
    }

    #[test]
    fn test_wordpiece_max_word_len() {
        let vocab = &["[CLS]", "[SEP]", "[UNK]", "foo", "##bar", "##foo"];
        let opts = WordPieceOptions {
            max_word_len: Some(6),
            ..Default::default()
        };
        let tokenizer = create_tokenizer(vocab, None, opts);

        // The third word should be tokenized to `[UNK]` because it exceeds
        // `max_word_len`.
        let text = "foobar foofoo foobarfoo";
        let encoded = tokenizer.encode(text, None).unwrap();

        assert_eq!(
            tokenizer.model().get_tokens(encoded.token_ids()).unwrap(),
            &["[CLS]", "foo", "##bar", "foo", "##foo", "[UNK]", "[SEP]"]
        );
    }

    #[test]
    fn test_wordpiece_model_lowercase() {
        struct Case<'a> {
            text: &'a str,
            tokens: &'a [&'a str],
        }

        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "this", "is", "a", "test", "sequence",
        ];

        let normalizer = BertNormalizer::new(BertNormalizerOptions {
            lowercase: true,
            ..Default::default()
        });
        let tokenizer = create_tokenizer(vocab, Some(Box::new(normalizer)), Default::default());

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
            let encoded = tokenizer.encode(text, None).unwrap();
            assert_eq!(
                tokenizer.model().get_tokens(encoded.token_ids()).unwrap(),
                tokens
            );
            assert!(encoded.token_type_ids().all(|ttid| ttid == 0));
        }
    }

    #[test]
    fn test_decode() {
        struct Case<'a> {
            input: &'a str,
            expected: &'a str,
        }

        let cases = [
            Case {
                input: "",
                expected: "[CLS] [SEP]",
            },
            Case {
                input: "this is a test sequence",
                expected: "[CLS] this is a test sequence [SEP]",
            },
            Case {
                input: "THIS IS A TEST SEQUENCE",
                expected: "[CLS] this is a test sequence [SEP]",
            },
        ];

        let vocab = &[
            "[CLS]", "[SEP]", "[UNK]", "this", "is", "a", "test", "sequence",
        ];

        let normalizer = BertNormalizer::new(BertNormalizerOptions {
            lowercase: true,
            ..Default::default()
        });
        let tokenizer = create_tokenizer(vocab, Some(Box::new(normalizer)), Default::default());

        for Case { input, expected } in cases {
            let encoded = tokenizer.encode(input, None).unwrap();
            let decoded = tokenizer.decode(encoded.token_ids()).unwrap();
            assert_eq!(decoded, expected);
        }
    }
}
