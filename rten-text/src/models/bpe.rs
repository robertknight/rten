use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};

use super::{DecodeError, EncodeError, Model};
use crate::tokenizer::TokenId;

/// Errors that can occur when building a [`Bpe`] tokenizer or encoding or
/// decoding text using it.
#[derive(Debug)]
pub enum BpeError {
    /// There was an invalid entry in the merge list. This means that either
    /// the entry doesn't have the expected `<token> [SPACE] <token>` format
    /// or the `<token>` is not either a single character or the concatenation
    /// of another pair in the merge list.
    InvalidMergeEntry(String),

    /// An entry in the vocab (token string to ID map) is not either a known
    /// special token or an entry in the merge list.
    InvalidVocabEntry(String),
}

impl Display for BpeError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BpeError::InvalidMergeEntry(entry) => write!(fmt, "invalid merge entry: {}", entry),
            BpeError::InvalidVocabEntry(entry) => write!(fmt, "invalid vocab entry: {}", entry),
        }
    }
}

impl Error for BpeError {}

/// Rank of a token in the merge list. A token is formed either of a pair of
/// smaller tokens, or a single byte.
type Rank = u32;

/// A sequence of UTF-8 bytes, encoded as a string of printable characters.
/// [`char_to_byte`] provides the mapping between characters and bytes.
///
/// Unlike a Rust `str`, the sequence of bytes do not necessarily form a
/// complete sequence of Unicode characters. The bytes may end in the middle of
/// a character.
pub type EncodedByteSlice<'a> = &'a str;

/// Like [`EncodedByteSlice`], but owned.
pub type EncodedBytes = String;

/// Return true if `c` is considered a printable character.
///
/// This matches the output of Python's `str.isprintable` for code points < 256,
/// except for ASCII space.
fn is_printable(c: char) -> bool {
    !c.is_control() && !c.is_whitespace() && c != '\u{ad}' /* soft hyphen */
}

/// Return a mapping from byte value to token rank / ID.
fn byte_to_rank() -> [Rank; 256] {
    let mut ranks = [0; 256];

    let mut r = 0;
    for b in 0..=255u8 {
        if is_printable(char::from(b)) {
            ranks[b as usize] = r;
            r += 1;
        }
    }

    for b in 0..=255u8 {
        if !is_printable(char::from(b)) {
            ranks[b as usize] = r;
            r += 1;
        }
    }

    ranks
}

/// Return a mapping between the printable characters used in the GPT 2 merge
/// list and vocabulary, and the byte values they represent.
///
/// Based on the `bytes_to_unicode` function in the original GPT-2 encoder -
/// <https://github.com/openai/gpt-2/blob/master/src/encoder.py>.
pub fn char_to_byte() -> HashMap<char, u8> {
    let mut n = 0;
    (0..=255u8)
        .map(|b| {
            let ch = char::from(b);
            if is_printable(ch) {
                (ch, b)
            } else {
                let pair = (char::from_u32(256 + n).unwrap(), b);
                n += 1;
                pair
            }
        })
        .collect()
}

/// Iteratively merge pairs of tokens in `tokens`, using the mappings in `ranks`,
/// until no more merges are possible.
///
/// Returns the number of merged tokens.
fn bpe_merge(tokens: &mut Vec<Rank>, ranks: &HashMap<(Rank, Rank), Rank>) -> usize {
    loop {
        // Find the pair of tokens with the lowest rank and merge all occurences
        // of the pair.
        let min_pair: Option<((Rank, Rank), Rank)> = tokens
            .windows(2)
            .filter_map(|pair| {
                let [first, second] = pair.try_into().unwrap();
                ranks
                    .get(&(first, second))
                    .map(|&rank| ((first, second), rank))
            })
            .min_by_key(|((_first, _second), rank)| *rank);

        let Some(((first, second), rank)) = min_pair else {
            break;
        };

        let mut i = 0;
        while i < tokens.len() - 1 {
            if tokens[i] == first && tokens[i + 1] == second {
                tokens[i] = rank;
                tokens.remove(i + 1);
            }
            i += 1;
        }
    }
    tokens.len()
}

struct BpeBuilder {
    /// See [`ByteLevelBpe::merges`].
    ranks: HashMap<(Rank, Rank), Rank>,

    /// Mapping between encoded tokens and their rank in the BPE merge list. In
    /// addition to entries created from the merge list, this also has
    /// single-character entries that correspond to byte values.
    token_ranks: HashMap<EncodedBytes, Rank>,

    /// Ranks assigned to individual bytes.
    byte_to_rank: [Rank; 256],

    /// True if different tokens are generated depending on whether a token
    /// appears at the end of a word or not, where a "word" is a string piece
    /// produced after initial splitting of the input (eg. using a regex).
    end_of_word_suffix: bool,
}

struct BpeBuilderOptions<'a> {
    end_of_word_suffix: Option<&'a str>,
}

impl BpeBuilder {
    fn new(options: BpeBuilderOptions) -> BpeBuilder {
        let char_to_byte = char_to_byte();
        let byte_to_rank = byte_to_rank();
        let mut token_ranks: HashMap<EncodedBytes, u32> = char_to_byte
            .iter()
            .map(|(ch, byte)| (ch.to_string(), byte_to_rank[*byte as usize]))
            .collect();

        if let Some(suffix) = options.end_of_word_suffix {
            let end_of_word_start = token_ranks.len() as u32;
            token_ranks.extend(char_to_byte.iter().map(|(ch, byte)| {
                (
                    format!("{}{}", ch, suffix),
                    byte_to_rank[*byte as usize] + end_of_word_start,
                )
            }));
        }

        BpeBuilder {
            ranks: HashMap::new(),
            byte_to_rank,
            token_ranks,
            end_of_word_suffix: options.end_of_word_suffix.is_some(),
        }
    }

    /// Return the rank of a token in the merge list, ie. the pair whose
    /// concatenated parts equal `token`.
    fn get_token_rank(&self, token: EncodedByteSlice) -> Option<Rank> {
        self.token_ranks.get(token).copied()
    }

    /// Build the BPE merge map that assigns a rank to pairs of tokens.
    ///
    /// `merges` contains entries of the BPE merge table. Each entry is a pair
    /// of tokens. Each token is a sequence of byte values encoded using the
    /// scheme described in [`char_to_byte`].
    fn add_merges(
        &mut self,
        merges: &[(EncodedByteSlice, EncodedByteSlice)],
    ) -> Result<(), BpeError> {
        // The first 256 ranks are assigned to individual byte values.
        let mut rank = 256 + self.ranks.len() as u32;

        // If using an EOW suffix, the next 256 ranks are assigned to bytes
        // occurring at the end of a word.
        if self.end_of_word_suffix {
            rank += 256;
        }

        self.ranks.reserve(merges.len());
        self.token_ranks.reserve(merges.len());

        for (a, b) in merges.iter().copied() {
            let invalid_entry = || BpeError::InvalidMergeEntry(format!("{} {}", a, b));
            let a_rank = self.get_token_rank(a).ok_or_else(invalid_entry)?;
            let b_rank = self.get_token_rank(b).ok_or_else(invalid_entry)?;
            self.ranks.insert((a_rank, b_rank), rank);
            self.token_ranks.insert([a, b].concat(), rank);

            rank += 1;
        }

        Ok(())
    }
}

/// Parse a list of space-separated BPE merge entries into pairs of tokens.
///
/// Lines that are empty or contain only a `#version` marker are ignored.
pub fn merge_pairs_from_lines(
    lines: &[impl AsRef<str>],
) -> Vec<(EncodedByteSlice<'_>, EncodedByteSlice<'_>)> {
    lines
        .iter()
        .filter_map(|line| {
            let line = line.as_ref();
            if line.starts_with("#version") || line.trim().is_empty() {
                None
            } else {
                line.split_once(' ')
            }
        })
        .collect()
}

/// Configuration for a [`Bpe`] tokenization model.
#[derive(Default)]
pub struct BpeOptions<'a> {
    /// Ordered entries of the merge list. Each entry is a pair of strings
    /// representing byte sequences. See also [`merge_pairs_from_lines`] which
    /// can be used to extract pairs from the space-separated format used in eg.
    /// `merges.txt` files.
    pub merges: &'a [(EncodedByteSlice<'a>, EncodedByteSlice<'a>)],

    /// Mapping between token strings and IDs. If not provided, the
    /// ID of a token is 256 + the index of the pair in the merge list which
    /// form the token string when concatenated. For example, if index 10 in the
    /// merge list is "foo bar", then the token ID of "foobar" would be 266.
    /// Token IDs below 256 are reserved for individual bytes.
    pub vocab: Option<HashMap<EncodedBytes, TokenId>>,

    /// Set of tokens which don't appear in `merges` but do have a mapping in
    /// `vocab`. These are used for special purposes such as representing the
    /// end of output.
    pub added_tokens: HashMap<TokenId, String>,

    /// A string which is implicitly appended to each substring that is
    /// tokenized, after initial splitting.
    pub end_of_word_suffix: Option<String>,
}

/// Byte Pair Encoding tokenizer used by GPT-2 [^1] and subsequently used by
/// many other models.
///
/// Byte Pair Encoding was introduced by [^2]. Despite the name, the original
/// version operated on characters. The variant used by GPT-2 and other OpenAI
/// models operates on bytes instead. This avoids needing a huge base vocabulary
/// to support Unicode.
///
/// [^1]: Radford, Alec, et al. (2019) "Language models are unsupervised multitask learners."
///       <https://openai.com/research/better-language-models>
///
/// [^2]: Sennrich, Rico, Barry Haddow, and Alexandra Birch. "Neural machine
///       translation of rare words with subword units." arXiv preprint
///       arXiv:1508.07909 (2015).
pub struct Bpe {
    /// Map from pairs of tokens, to the rank of the pair. Each token in the
    /// pair is either the rank of another pair, or the rank for a single byte
    /// according to `byte_to_rank`.
    ///
    /// Values in this map start at 256, as lower values are reserved for single
    /// byte tokens.
    merges: HashMap<(Rank, Rank), Rank>,

    /// Map from byte values to token rank. Ranks are in the range [0, 255].
    byte_to_rank: [Rank; 256],

    /// Map from rank in `merges` or `byte_to_rank`, to token ID.
    ///
    /// If `None`, the token ID is the same as the rank. This is the case with
    /// GPT-2 and other OpenAI models for example.
    rank_to_token_id: Option<HashMap<Rank, TokenId>>,

    /// Map from token ID to encoded bytes.
    ///
    /// If `None`, the token ID is the same as the rank, and the bytes are
    /// obtained by recursively replacing the token ID with the pair of token
    /// IDs that make it up (see `merges`) until we have a sequence of token IDs
    /// that each represent single byte values.
    token_id_to_encoded_bytes: Option<HashMap<TokenId, EncodedBytes>>,

    /// Map from token ID to content for special tokens (eg. end-of-string).
    added_tokens: HashMap<TokenId, String>,

    /// A suffix which is implicitly appended to each string piece to be
    /// tokenized.
    ///
    /// This was originally introduced for CLIP's tokenizer.
    /// See <https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py>.
    end_of_word_suffix: Option<String>,
}

impl Bpe {
    /// Create a new Byte Pair Encoding tokenizer using the given configuration.
    pub fn new(config: BpeOptions) -> Result<Bpe, BpeError> {
        let BpeOptions {
            merges,
            vocab,
            added_tokens,
            mut end_of_word_suffix,
        } = config;

        // Normalize empty end-of-word suffix to `None`.
        end_of_word_suffix.take_if(|suffix| suffix.is_empty());

        let bb_opts = BpeBuilderOptions {
            end_of_word_suffix: end_of_word_suffix.as_deref(),
        };
        let mut builder = BpeBuilder::new(bb_opts);
        builder.add_merges(merges)?;

        let (rank_to_token_id, token_id_to_encoded_bytes) = if let Some(vocab) = vocab {
            let mut token_id_to_encoded_bytes = HashMap::with_capacity(vocab.len());
            let mut rank_to_token_id = HashMap::with_capacity(vocab.len());
            for (token, id) in vocab.into_iter() {
                token_id_to_encoded_bytes.insert(id, token.clone());

                if let Some(rank) = builder.get_token_rank(&token) {
                    rank_to_token_id.insert(rank, id);
                }
            }
            (Some(rank_to_token_id), Some(token_id_to_encoded_bytes))
        } else {
            (None, None)
        };

        Ok(Bpe {
            merges: builder.ranks,
            byte_to_rank: builder.byte_to_rank,
            rank_to_token_id,
            added_tokens,
            token_id_to_encoded_bytes,
            end_of_word_suffix,
        })
    }

    /// Decode a token ID to a byte sequence. Be aware that the returned bytes
    /// may end in the middle of a UTF-8 character.
    fn get_token_bytes(&self, id: TokenId) -> Option<Vec<u8>> {
        if id < 256 {
            let byte = self
                .byte_to_rank
                .iter()
                .enumerate()
                .find(|(_b, rank)| **rank == id)
                .map(|(b, _rank)| b)
                .unwrap();
            return Some(vec![byte as u8]);
        }

        if let Some(eow_suffix) = self.end_of_word_suffix.as_deref() {
            if id < 512 {
                let mut bytes = self.get_token_bytes(id - 256)?;
                bytes.extend(eow_suffix.as_bytes());
                return Some(bytes);
            }
        }

        let (first, second) = self
            .merges
            .iter()
            .find(|(_k, v)| **v == id)
            .map(|(k, _v)| k)?;
        let mut out = self.get_token_bytes(*first)?;
        let second_bytes = self.get_token_bytes(*second)?;
        out.extend(&second_bytes);
        Some(out)
    }

    /// Encode a string as a sequence of tokens.
    ///
    /// `end_of_word` specifies whether to apply end-of-word processing rules
    /// to the initial tokenization of piece.
    fn encode_piece(&self, piece: &str, end_of_word: bool) -> Vec<TokenId> {
        // Start with one token per byte.
        let mut tokens: Vec<Rank> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_rank[b as usize])
            .collect();

        // If the end-of-word suffix is enabled, replace the last byte's token
        // with the one that corresponds to "{byte}{end_of_word_suffix}".
        if self.end_of_word_suffix.is_some() && end_of_word {
            if let Some(last) = tokens.pop() {
                tokens.push(last + 256);
            }
        }

        // Iteratively merge tokens together until no more are possible.
        bpe_merge(&mut tokens, &self.merges);

        // Convert ranks to token IDs.
        let unknown_token_id = 0;
        if let Some(id_map) = self.rank_to_token_id.as_ref() {
            tokens
                .into_iter()
                .map(|rank| id_map.get(&rank).copied().unwrap_or(unknown_token_id))
                .collect()
        } else {
            tokens
        }
    }
}

impl Model for Bpe {
    fn get_token_str(&self, id: TokenId) -> Option<String> {
        if let Some(tok_str) = self.added_tokens.get(&id) {
            return Some(tok_str.to_string());
        }

        if let Some(tok_str) = self
            .token_id_to_encoded_bytes
            .as_ref()
            .and_then(|map| map.get(&id))
        {
            return Some(tok_str.clone());
        }

        // nb. The current implementation is inefficient as it does recursive
        // calls to `get_token_bytes` and creates the byte-to-char lookup table
        // on every call.

        let bytes = self.get_token_bytes(id)?;

        let byte_to_char: HashMap<u8, char> = char_to_byte()
            .into_iter()
            .map(|(ch, byte)| (byte, ch))
            .collect();

        let token_str: String = bytes
            .into_iter()
            .map(|byte| {
                byte_to_char
                    .get(&byte)
                    .expect("should have char for all bytes")
            })
            .collect();
        Some(token_str)
    }

    fn get_token_id(&self, mut text: &str) -> Option<TokenId> {
        if let Some((&id, _str)) = self.added_tokens.iter().find(|(_id, str)| *str == text) {
            return Some(id);
        }

        // Determine the end-of-word context. eg. In CLIP's tokenizer, the
        // trailing "</w>" in "from</w>" indicates that it should be treated as
        // occurring at the end of a piece from the initial split.
        let mut end_of_word = false;
        if let Some(suffix) = self.end_of_word_suffix.as_deref() {
            if text.ends_with(suffix) {
                text = &text[..text.len() - suffix.len()];
                end_of_word = true;
            }
        }

        let tokens = self.encode_piece(text, end_of_word);
        if tokens.len() == 1 {
            Some(tokens[0])
        } else {
            None
        }
    }

    fn encode_with_offsets(
        &self,
        piece: &str,
        on_token: &mut dyn FnMut(usize, TokenId),
    ) -> Result<(), EncodeError> {
        if piece.is_empty() {
            return Ok(());
        }
        for token in self.encode_piece(piece, true /* end_of_word */) {
            on_token(0, token)
        }
        Ok(())
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String, DecodeError> {
        let char_to_byte = char_to_byte();

        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok_str) = self.added_tokens.get(&id) {
                bytes.extend(tok_str.as_bytes());
            } else if let Some(encoded_bytes) = self
                .token_id_to_encoded_bytes
                .as_ref()
                .and_then(|map| map.get(&id))
            {
                bytes.extend(
                    encoded_bytes
                        .chars()
                        .map(|ch| char_to_byte.get(&ch).copied().unwrap()),
                );
            } else {
                let token_bytes = self
                    .get_token_bytes(id)
                    .ok_or(DecodeError::InvalidTokenId(id))?;
                bytes.extend(token_bytes);
            }
        }
        String::from_utf8(bytes).map_err(|_| DecodeError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rten_testing::TestCases;

    use super::{merge_pairs_from_lines, Bpe, BpeOptions, EncodedBytes};
    use crate::pre_tokenizers::Split;
    use crate::tokenizer::{TokenId, Tokenizer};

    // The first ~25 lines of the merge list from GPT 2.
    const MINI_GPT2: &str = "
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
o n
Ġt he
e r
Ġ s
a t
Ġ w
Ġ o
e n
Ġ c
i t
i s
a n
o r
e s
Ġ b
e d
Ġ f
in g";

    fn added_tokens() -> HashMap<TokenId, String> {
        [(50256, "<|endoftext|>")]
            .into_iter()
            .map(|(id, str)| (id, str.to_string()))
            .collect()
    }

    /// Generate a map from encoded token string to token ID.
    ///
    /// The token IDs are chosen to be different than the ones that would be
    /// automatically generated based on the merge list, if the vocabulary was
    /// not supplied.
    fn gen_vocab() -> HashMap<EncodedBytes, TokenId> {
        let mut vocab = HashMap::new();
        let mut next_token_id = 1000;

        for ch in super::char_to_byte().keys() {
            vocab.insert(ch.to_string(), next_token_id);
            next_token_id += 1;
        }

        for line in MINI_GPT2.lines().map(|l| l.trim()) {
            if line.starts_with("#version") || line.is_empty() {
                continue;
            }
            let token_str: EncodedBytes = line.chars().filter(|ch| *ch != ' ').collect();
            vocab.insert(token_str, next_token_id);
            next_token_id += 1;
        }

        vocab
    }

    #[test]
    fn test_encode() {
        #[derive(Debug)]
        struct Case<'a> {
            text: &'a str,
            expected_tokens: &'a [&'a str],
            merges: &'a str,
            vocab: Option<HashMap<EncodedBytes, TokenId>>,
            end_of_word_suffix: Option<String>,
        }

        let cases = [
            // Minimal test using a snippet of the GPT-2 merge list.
            Case {
                text: "the cat is in the bed",
                expected_tokens: &[
                    "t", "he", "Ġc", "at", "Ġ", "is", "Ġ", "in", "Ġthe", "Ġb", "ed",
                ],
                merges: MINI_GPT2,
                vocab: None,
                end_of_word_suffix: None,
            },
            // Test several levels of merging.
            Case {
                text: "--------",
                expected_tokens: &["--------"],
                merges: "
- -
-- --
---- ----
-------- --------
",
                vocab: None,
                end_of_word_suffix: None,
            },
            // End-of-word suffix
            Case {
                text: "barbar",
                expected_tokens: &["bar", "bar</w>"],
                merges: "
b a
ba r
ba r</w>
",
                vocab: None,
                end_of_word_suffix: Some("</w>".to_string()),
            },
            // Empty end-of-word suffix. Treated as `None` for compatibility
            // with some tokenizer.json files which represent the EOW suffix
            // using `""` instead of `null`.
            Case {
                text: "barbar",
                expected_tokens: &["bar", "bar"],
                merges: "
b a
ba r",
                vocab: None,
                end_of_word_suffix: Some("".to_string()),
            },
        ];

        cases.test_each(|case| {
            let Case {
                text,
                expected_tokens: tokens,
                merges,
                vocab,
                end_of_word_suffix,
            } = case;

            let merges: Vec<&str> = merges.lines().collect();
            let merge_pairs = merge_pairs_from_lines(&merges);
            let bpe_opts = BpeOptions {
                merges: &merge_pairs,
                vocab: vocab.clone(),
                end_of_word_suffix: end_of_word_suffix.clone(),
                ..Default::default()
            };
            let model = Bpe::new(bpe_opts).unwrap();
            let tokenizer = Tokenizer::new(model, Default::default())
                .with_pre_tokenizer(Box::new(Split::gpt2()));
            let encoded = tokenizer.encode(*text, None).unwrap();
            assert_eq!(
                tokenizer.model().get_tokens(encoded.token_ids()).unwrap(),
                *tokens
            );
        })
    }

    #[test]
    fn test_get_token_str() {
        #[derive(Debug)]
        struct Case<'a> {
            input: &'a str,
            encoded_str: &'a str,
        }

        let cases = [
            // Printable ASCII text. Encoded string is same as input.
            Case {
                input: "a",
                encoded_str: "a",
            },
            // Non-printable or non-ASCII text. Encoded string will use
            // printable characters to represent these bytes.
            Case {
                input: " ",
                encoded_str: "Ġ",
            },
            // Added tokens.
            Case {
                input: "<|endoftext|>",
                encoded_str: "<|endoftext|>",
            },
        ];

        let merges: Vec<&str> = MINI_GPT2.lines().collect();
        let merge_pairs = merge_pairs_from_lines(&merges);

        cases.test_each(|case| {
            let bpe_opts = BpeOptions {
                merges: &merge_pairs,
                added_tokens: added_tokens(),
                ..Default::default()
            };
            let model = Bpe::new(bpe_opts).unwrap();
            let tokenizer = Tokenizer::new(model, Default::default())
                .with_pre_tokenizer(Box::new(Split::gpt2()));

            let tok_id = tokenizer.model().get_token_id(case.input).unwrap();
            let token_str = tokenizer.model().get_token_str(tok_id).unwrap();
            assert_eq!(token_str, case.encoded_str);
        })
    }

    #[test]
    fn test_decode() {
        #[derive(Debug)]
        struct Case<'a> {
            text: &'a str,
            add_eos: bool,
            expected: &'a str,
            vocab: Option<HashMap<EncodedBytes, TokenId>>,
        }

        let vocab = gen_vocab();

        let cases = [
            Case {
                text: "foo bar",
                add_eos: false,
                expected: "foo bar",
                vocab: None,
            },
            Case {
                text: "foo bar",
                add_eos: true,
                expected: "foo bar<|endoftext|>",
                vocab: None,
            },
            Case {
                text: "the cat is in the bed",
                add_eos: false,
                expected: "the cat is in the bed",
                vocab: None,
            },
            Case {
                text: "the cat is in the bed",
                add_eos: false,
                expected: "the cat is in the bed",
                vocab: Some(vocab),
            },
        ];

        cases.test_each(|case| {
            let Case {
                text,
                add_eos,
                expected,
                vocab,
            } = case;

            let merges: Vec<&str> = MINI_GPT2.lines().collect();
            let merge_pairs = merge_pairs_from_lines(&merges);
            let bpe_opts = BpeOptions {
                merges: &merge_pairs,
                vocab: vocab.clone(),
                added_tokens: added_tokens(),
                ..Default::default()
            };
            let model = Bpe::new(bpe_opts).unwrap();
            let tokenizer = Tokenizer::new(model, Default::default())
                .with_pre_tokenizer(Box::new(Split::gpt2()));

            let encoded = tokenizer.encode(*text, None).unwrap();
            let mut token_ids = encoded.token_ids().to_vec();
            if *add_eos {
                // The `<|endoftext|>` token ID from GPT-2.
                token_ids.push(50256);
            }
            let decoded = tokenizer.decode(&token_ids).unwrap();
            assert_eq!(decoded, *expected);
        })
    }
}
