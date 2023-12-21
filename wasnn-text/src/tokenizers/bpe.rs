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
    /// There was an invalid entry in the merge table.
    InvalidMergeEntry(String),

    /// The regex for splitting tokens is invalid.
    InvalidPattern(fancy_regex::Error),
}

impl Display for BpeError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BpeError::InvalidMergeEntry(entry) => write!(fmt, "invalid merge entry: {}", entry),
            BpeError::InvalidPattern(err) => write!(fmt, "invalid regex: {}", err),
        }
    }
}

impl Error for BpeError {}

/// Rank of a token in the merge list. A token is formed either of a pair of
/// smaller tokens, or a single byte.
type Rank = u32;

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

/// Build the BPE merge map that assigns a rank to pairs of tokens.
///
/// `merges` contains entries of the BPE merge table. Each entry is a
/// space-separated pair of tokens. Each token is a sequence of byte values
/// encoded using the scheme described in [char_to_byte].
fn build_merge_map(merges: &[&str]) -> Result<HashMap<(Rank, Rank), Rank>, BpeError> {
    let char_to_byte = char_to_byte();
    let byte_to_rank = byte_to_rank();

    let mut tmp_tokens: Vec<Rank> = Vec::new();
    let mut merge_ranks: HashMap<(Rank, Rank), Rank> = HashMap::new();

    // The first 256 ranks are assigned to individual byte values.
    let mut rank = 256;

    for entry in merges.iter() {
        if entry.starts_with("#version") || entry.trim().is_empty() {
            continue;
        }

        let (a, b) = entry
            .split_once(' ')
            .ok_or_else(|| BpeError::InvalidMergeEntry(entry.to_string()))?;

        let mut get_token_rank = |token: &str| -> Result<Rank, BpeError> {
            tmp_tokens.clear();
            for ch in token.chars() {
                let Some(&byte) = char_to_byte.get(&ch) else {
                    return Err(BpeError::InvalidMergeEntry(entry.to_string()));
                };
                tmp_tokens.push(byte_to_rank[byte as usize]);
            }

            let n_merged = bpe_merge(&mut tmp_tokens, &merge_ranks);
            if n_merged == 1 {
                Ok(tmp_tokens[0])
            } else {
                Err(BpeError::InvalidMergeEntry(entry.to_string()))
            }
        };

        let a_rank = get_token_rank(a)?;
        let b_rank = get_token_rank(b)?;
        merge_ranks.insert((a_rank, b_rank), rank.try_into().unwrap());
        rank += 1;
    }

    Ok(merge_ranks)
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

/// Byte-level Byte Pair Encoding tokenizer used by GPT-2 [^1].
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
pub struct ByteLevelBpe {
    /// Map from pairs of tokens, to the rank of the pair. Each token in the
    /// pair is either the rank of another pair, or the rank for a single byte
    /// according to `byte_to_rank`.
    ///
    /// Values in this map start at 256, as lower values are reserved for single
    /// byte tokens.
    merges: HashMap<(Rank, Rank), Rank>,

    /// Map from byte values to token rank. Ranks are in the range [0, 255].
    byte_to_rank: [Rank; 256],

    /// Pattern used to split into text into pieces prior to applying BPE
    /// tokenization.
    splitter: Regex,
}

impl ByteLevelBpe {
    /// Create a new Byte Pair Encoding tokenizer.
    ///
    /// `merges` are the ordered entries of the merge list. Each entry is a
    /// space-separated pair of strings representing byte sequences. The ID
    /// of each token is its order within the merge list, plus 256, as the
    /// first 256 token IDs are reserved for individual bytes.
    ///
    /// `pattern` is a regex used to split input text into pieces before BPE
    /// encoding is applied. The supported syntax is that supported by the
    /// [fancy_regex](https://crates.io/crates/fancy-regex) crate. The
    /// [patterns] module contains patterns used by popular models.
    pub fn new(merges: &[&str], pattern: &str) -> Result<ByteLevelBpe, BpeError> {
        let splitter = Regex::new(pattern).map_err(BpeError::InvalidPattern)?;
        let merges = build_merge_map(merges)?;

        Ok(ByteLevelBpe {
            merges,
            byte_to_rank: byte_to_rank(),
            splitter,
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

    /// Encode a word piece as a sequence of tokens.
    fn encode_piece(&self, piece: &str) -> Vec<Rank> {
        // Start with one token per byte.
        let mut tokens: Vec<Rank> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_rank[b as usize])
            .collect();

        // Iteratively merge tokens together until no more are possible.
        bpe_merge(&mut tokens, &self.merges);

        tokens
    }
}

impl Encoder for ByteLevelBpe {
    fn get_token_str(&self, id: usize) -> Result<String, TokenizerError> {
        let bytes = self
            .get_token_bytes(id as u32)
            .ok_or(TokenizerError::InvalidTokenId(id))?;
        String::from_utf8(bytes).map_err(|err| TokenizerError::InvalidUtf8(err.utf8_error()))
    }

    fn get_token_id(&self, text: &str) -> Result<usize, TokenizerError> {
        let tokens = self.encode_piece(text);
        if tokens.len() == 1 {
            Ok(tokens[0] as usize)
        } else {
            Err(TokenizerError::MissingToken(text.to_string()))
        }
    }

    fn encode_sequence(
        &self,
        text: &str,
        on_token: &mut dyn FnMut(usize, usize),
    ) -> Result<(), TokenizerError> {
        for piece in self.splitter.find_iter(text) {
            let piece = piece.map_err(TokenizerError::RegexSplitFailed)?;
            if piece.range().is_empty() {
                continue;
            }

            for token in self.encode_piece(piece.as_str()) {
                on_token(piece.start(), token as usize)
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::patterns::GPT2 as GPT2_SPLIT_PATTERN;
    use super::ByteLevelBpe;
    use crate::tokenizers::Tokenizer;

    // TODO - Test that all single char tokens are decoded correctly.
    // TODO - Test handling of errors in merge entries

    // The first ~25 lines of the merge map from GPT 2.
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
                    "t", "he", " c", "at", " ", "is", " ", "in", " the", " b", "ed",
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
            let encoder = ByteLevelBpe::new(&merges, GPT2_SPLIT_PATTERN).unwrap();
            let tokenizer = Tokenizer::new(encoder, Default::default());
            let encoded = tokenizer.encode(text.into(), Default::default()).unwrap();
            assert_eq!(
                tokenizer.encoder().get_tokens(encoded.token_ids()).unwrap(),
                tokens
            );
        }
    }
}
