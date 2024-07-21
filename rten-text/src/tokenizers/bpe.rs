use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};

use fancy_regex::Regex;

use crate::tokenizers::{Encoder, TokenizerError};

/// Errors that can occur when building a [Bpe] tokenizer or encoding or
/// decoding text using it.
#[derive(Debug)]
pub enum BpeError {
    /// There was an invalid entry in the merge list. This means that either
    /// the entry doesn't have the expected `<token> [SPACE] <token>` format
    /// or the `<token>` is not either a single character or the concatenation
    /// of another pair in the merge list.
    InvalidMergeEntry(String),

    /// The regex for splitting tokens is invalid.
    InvalidPattern(fancy_regex::Error),

    /// An entry in the vocab (token string to ID map) is not either a known
    /// special token or an entry in the merge list.
    InvalidVocabEntry(String),
}

impl Display for BpeError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BpeError::InvalidMergeEntry(entry) => write!(fmt, "invalid merge entry: {}", entry),
            BpeError::InvalidPattern(err) => write!(fmt, "invalid regex: {}", err),
            BpeError::InvalidVocabEntry(entry) => write!(fmt, "invalid vocab entry: {}", entry),
        }
    }
}

impl Error for BpeError {}

/// Rank of a token in the merge list. A token is formed either of a pair of
/// smaller tokens, or a single byte.
type Rank = u32;

/// A sequence of UTF-8 bytes, encoded as a string of characters.
/// [`char_to_byte`] provides the mapping between characters and bytes.
///
/// Unlike a Rust `str`, the sequence of bytes do not necessarily form a
/// complete sequence of Unicode characters. The bytes may end in the middle of
/// a character.
type EncodedByteSlice<'a> = &'a str;

/// Like [EncodedByteSlice], but owned.
type EncodedBytes = String;

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

/// Return a mapping between the characters used in the GPT 2 merge list
/// and vocabulary, and the byte values they represent.
///
/// Based on the `bytes_to_unicode` function in the original GPT-2 encoder -
/// https://github.com/openai/gpt-2/blob/master/src/encoder.py.
fn char_to_byte() -> HashMap<char, u8> {
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
    /// See [ByteLevelBpe::merges].
    ranks: HashMap<(Rank, Rank), Rank>,

    /// Mapping between encoded tokens and their rank in the BPE merge list. In
    /// addition to entries created from the merge list, this also has
    /// single-character entries that correspond to byte values.
    token_ranks: HashMap<EncodedBytes, Rank>,

    /// Ranks assigned to individual bytes.
    byte_to_rank: [Rank; 256],
}

impl BpeBuilder {
    fn new() -> BpeBuilder {
        let char_to_byte = char_to_byte();
        let byte_to_rank = byte_to_rank();
        let token_ranks: HashMap<EncodedBytes, u32> = char_to_byte
            .iter()
            .map(|(ch, byte)| (ch.to_string(), byte_to_rank[*byte as usize]))
            .collect();

        BpeBuilder {
            ranks: HashMap::new(),
            byte_to_rank,
            token_ranks,
        }
    }

    /// Return the rank of a token in the merge list, ie. the pair whose
    /// concatenated parts equal `token`.
    fn get_token_rank(&self, token: EncodedByteSlice) -> Option<Rank> {
        self.token_ranks.get(token).copied()
    }

    /// Build the BPE merge map that assigns a rank to pairs of tokens.
    ///
    /// `merges` contains entries of the BPE merge table. Each entry is a
    /// space-separated pair of tokens. Each token is a sequence of byte values
    /// encoded using the scheme described in [`char_to_byte`].
    fn add_merges(&mut self, merges: &[EncodedByteSlice]) -> Result<(), BpeError> {
        // The first 256 ranks are assigned to individual byte values.
        let mut rank = 256 + self.ranks.len() as u32;
        self.ranks.reserve(merges.len());
        self.token_ranks.reserve(merges.len());

        for entry in merges.iter() {
            if entry.starts_with("#version") || entry.trim().is_empty() {
                continue;
            }

            let invalid_entry = || BpeError::InvalidMergeEntry(entry.to_string());
            let (a, b) = entry.split_once(' ').ok_or_else(invalid_entry)?;
            let a_rank = self.get_token_rank(a).ok_or_else(invalid_entry)?;
            let b_rank = self.get_token_rank(b).ok_or_else(invalid_entry)?;
            self.ranks.insert((a_rank, b_rank), rank);
            self.token_ranks.insert([a, b].concat(), rank);

            rank += 1;
        }

        Ok(())
    }
}

/// Regex patterns used by popular tokenizer models.
///
/// Some models (eg. GPT-2) use a regex to split input text into pieces prior
/// to applying the trained tokenizer model. This module contains some widely
/// used patterns.
pub mod patterns {
    /// Tokenization regex used by GPT-2.
    ///
    /// See <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>.
    pub const GPT2: &str =
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
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
    rank_to_token_id: Option<HashMap<Rank, usize>>,

    /// Map from token ID to encoded bytes.
    ///
    /// If `None`, the token ID is the same as the rank, and the bytes are
    /// obtained by recursively replacing the token ID with the pair of token
    /// IDs that make it up (see `merges`) until we have a sequence of token IDs
    /// that each represent single byte values.
    token_id_to_encoded_bytes: Option<HashMap<usize, EncodedBytes>>,

    /// Pattern used to split the text into pieces prior to applying BPE
    /// tokenization.
    splitter: Regex,

    /// Map from token ID to content for special tokens (eg. end-of-string).
    added_tokens: HashMap<usize, String>,
}

impl Bpe {
    /// Create a new Byte Pair Encoding tokenizer.
    ///
    /// `merges` are the ordered entries of the merge list. Each entry is a
    /// space-separated pair of strings representing byte sequences.
    ///
    /// `pattern` is a regex used to split input text into pieces before BPE
    /// encoding is applied. The supported syntax is that supported by the
    /// [fancy_regex](https://crates.io/crates/fancy-regex) crate. The
    /// [patterns] module contains patterns used by popular models.
    ///
    /// `vocab` is a mapping between token strings and IDs. If not provided, the
    /// ID of a token is 256 + the index of the pair in the merge list which
    /// form the token string when concatenated. For example, if index 10 in the
    /// merge list is "foo bar", then the token ID of "foobar" would be 266.
    /// Token IDs below 256 are reserved for individual bytes.
    ///
    /// `added_tokens` is a set of tokens which don't appear in `merges` but
    /// do have a mapping in `vocab`. These are used for special purposes such
    /// as representing the end of output.
    pub fn new(
        merges: &[EncodedByteSlice],
        pattern: &str,
        vocab: Option<HashMap<EncodedBytes, usize>>,
        added_tokens: HashMap<usize, String>,
    ) -> Result<Bpe, BpeError> {
        let splitter = Regex::new(pattern).map_err(BpeError::InvalidPattern)?;

        let mut builder = BpeBuilder::new();
        builder.add_merges(merges)?;

        let (rank_to_token_id, token_id_to_encoded_bytes) = if let Some(vocab) = vocab {
            let mut token_id_to_encoded_bytes = HashMap::with_capacity(vocab.len());
            let mut rank_to_token_id = HashMap::with_capacity(vocab.len());
            for (token, id) in vocab.into_iter() {
                token_id_to_encoded_bytes.insert(id, token.clone());

                if let Some(rank) = builder.get_token_rank(&token) {
                    rank_to_token_id.insert(rank, id);
                } else if !added_tokens.values().any(|s| *s == token.as_str()) {
                    return Err(BpeError::InvalidVocabEntry(token));
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
            splitter,
            added_tokens,
            token_id_to_encoded_bytes,
        })
    }

    /// Decode a token ID to a byte sequence. Be aware that the returned bytes
    /// may end in the middle of a UTF-8 character.
    fn get_token_bytes(&self, id: u32) -> Option<Vec<u8>> {
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
    fn encode_piece(&self, piece: &str) -> Vec<usize> {
        // Start with one token per byte.
        let mut tokens: Vec<Rank> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_rank[b as usize])
            .collect();

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
            tokens.into_iter().map(|rank| rank as usize).collect()
        }
    }
}

impl Encoder for Bpe {
    fn get_token_str(&self, id: usize) -> Result<String, TokenizerError> {
        if let Some(tok_str) = self.added_tokens.get(&id) {
            return Ok(tok_str.to_string());
        }

        if let Some(tok_str) = self
            .token_id_to_encoded_bytes
            .as_ref()
            .and_then(|map| map.get(&id))
        {
            return Ok(tok_str.clone());
        }

        // nb. The current implementation is inefficient as it does recursive
        // calls to `get_token_bytes` and creates the byte-to-char lookup table
        // on every call.

        let bytes = self
            .get_token_bytes(id as u32)
            .ok_or(TokenizerError::InvalidTokenId(id))?;

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
        Ok(token_str)
    }

    fn get_token_id(&self, text: &str) -> Result<usize, TokenizerError> {
        if let Some((&id, _str)) = self.added_tokens.iter().find(|(_id, str)| *str == text) {
            return Ok(id);
        }

        let tokens = self.encode_piece(text);
        if tokens.len() == 1 {
            Ok(tokens[0])
        } else {
            Err(TokenizerError::MissingToken(text.to_string()))
        }
    }

    fn encode_with_offsets(
        &self,
        text: &str,
        on_token: &mut dyn FnMut(usize, usize),
    ) -> Result<(), TokenizerError> {
        for piece in self.splitter.find_iter(text) {
            let piece = piece.map_err(TokenizerError::RegexSplitFailed)?;
            if piece.range().is_empty() {
                continue;
            }

            let piece_str = piece.as_str();
            for token in self.encode_piece(piece_str) {
                on_token(piece.start(), token)
            }
        }

        Ok(())
    }

    fn decode(&self, ids: &[usize]) -> Result<String, TokenizerError> {
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
                    .get_token_bytes(id as u32)
                    .ok_or(TokenizerError::InvalidTokenId(id))?;
                bytes.extend(token_bytes);
            }
        }
        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::patterns::GPT2 as GPT2_SPLIT_PATTERN;
    use super::{Bpe, EncodedBytes};
    use crate::tokenizers::Tokenizer;

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

    fn added_tokens() -> HashMap<usize, String> {
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
    fn gen_vocab() -> HashMap<EncodedBytes, usize> {
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
        struct Case<'a> {
            text: &'a str,
            tokens: &'a [&'a str],
            merges: &'a str,
        }

        let cases = [
            // Minimal test using a snippet of the GPT-2 merge list.
            Case {
                text: "the cat is in the bed",
                tokens: &[
                    "t", "he", "Ġc", "at", "Ġ", "is", "Ġ", "in", "Ġthe", "Ġb", "ed",
                ],
                merges: MINI_GPT2,
            },
            // Test several levels of merging.
            Case {
                text: "--------",
                tokens: &["--------"],
                merges: "
- -
-- --
---- ----
-------- --------
",
            },
        ];

        for Case {
            text,
            tokens,
            merges,
        } in cases
        {
            let merges: Vec<&str> = merges.lines().collect();
            let encoder = Bpe::new(&merges, GPT2_SPLIT_PATTERN, None, HashMap::new()).unwrap();
            let tokenizer = Tokenizer::new(encoder, Default::default());
            let encoded = tokenizer.encode(text.into(), Default::default()).unwrap();
            assert_eq!(
                tokenizer.encoder().get_tokens(encoded.token_ids()).unwrap(),
                tokens
            );
        }
    }

    #[test]
    fn test_get_token_str() {
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
        let encoder = Bpe::new(&merges, GPT2_SPLIT_PATTERN, None, added_tokens()).unwrap();
        let tokenizer = Tokenizer::new(encoder, Default::default());

        for Case { input, encoded_str } in cases {
            let tok_id = tokenizer.encoder().get_token_id(input).unwrap();
            let token_str = tokenizer.encoder().get_token_str(tok_id).unwrap();
            assert_eq!(token_str, encoded_str);
        }
    }

    #[test]
    fn test_decode() {
        struct Case<'a> {
            text: &'a str,
            add_eos: bool,
            expected: &'a str,
            vocab: Option<HashMap<EncodedBytes, usize>>,
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

        for Case {
            text,
            add_eos,
            expected,
            vocab,
        } in cases
        {
            let merges: Vec<&str> = MINI_GPT2.lines().collect();
            let encoder = Bpe::new(&merges, GPT2_SPLIT_PATTERN, vocab, added_tokens()).unwrap();
            let tokenizer = Tokenizer::new(encoder, Default::default());

            let encoded = tokenizer.encode(text.into(), Default::default()).unwrap();
            let mut token_ids = encoded.token_ids().to_vec();
            if add_eos {
                // The `<|endoftext|>` token ID from GPT-2.
                token_ids.push(50256);
            }
            let decoded = tokenizer.encoder().decode(&token_ids).unwrap();
            assert_eq!(decoded, expected);
        }
    }
}
