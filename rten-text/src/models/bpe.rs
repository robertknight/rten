use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};

use super::{DecodeError, EncodeError, Model};
use crate::tokenizer::TokenId;
use rustc_hash::FxHashMap;

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

    /// An entry was not found in the vocabulary.
    MissingVocabEntry(String),
}

impl Display for BpeError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BpeError::InvalidMergeEntry(entry) => write!(fmt, "invalid merge entry: {}", entry),
            BpeError::InvalidVocabEntry(entry) => write!(fmt, "invalid vocab entry: {}", entry),
            BpeError::MissingVocabEntry(entry) => write!(fmt, "missing vocab entry: {}", entry),
        }
    }
}

impl Error for BpeError {}

/// Rank of an entry in the BPE merge list.
///
/// A newtype is used here to avoid confusing ranks and token IDs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Rank(u32);

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

/// Return a mapping from byte value to printable character used to represent
/// the byte.
///
/// Based on the `bytes_to_unicode` function in the original GPT-2 encoder -
/// <https://github.com/openai/gpt-2/blob/master/src/encoder.py>.
fn byte_to_char() -> [char; 256] {
    let mut chars = ['\x00'; 256];

    for b in 0..=255u8 {
        let ch = char::from(b);
        if is_printable(ch) {
            chars[b as usize] = ch;
        }
    }

    let mut non_printable_count = 0;
    for b in 0..=255u8 {
        if !is_printable(char::from(b)) {
            chars[b as usize] = char::from_u32(256 + non_printable_count).unwrap();
            non_printable_count += 1;
        }
    }

    chars
}

/// Return a mapping from printable character used to represent bytes to the
/// corresponding byte value.
pub fn char_to_byte() -> HashMap<char, u8> {
    byte_to_char()
        .iter()
        .copied()
        .enumerate()
        .map(|(byte, ch)| (ch, byte as u8))
        .collect()
}

/// Iteratively merge pairs of tokens in `tokens`, using the mappings in `ranks`,
/// until no more merges are possible.
///
/// Returns the number of merged tokens.
fn bpe_merge(
    tokens: &mut Vec<TokenId>,
    merges: &HashMap<(TokenId, TokenId), (Rank, TokenId)>,
) -> usize {
    loop {
        // Find the pair of tokens with the lowest rank and merge all occurences
        // of the pair.
        let min_pair: Option<((TokenId, TokenId), (Rank, TokenId))> = tokens
            .windows(2)
            .filter_map(|pair| {
                let [first, second] = pair.try_into().unwrap();
                merges
                    .get(&(first, second))
                    .map(|&rank_id| ((first, second), rank_id))
            })
            .min_by_key(|((_first, _second), (rank, _merged_id))| *rank);

        let Some(((first, second), (_rank, merged_id))) = min_pair else {
            break;
        };

        let mut i = 0;
        while i < tokens.len() - 1 {
            if tokens[i] == first && tokens[i + 1] == second {
                tokens[i] = merged_id;
                tokens.remove(i + 1);
            }
            i += 1;
        }
    }
    tokens.len()
}

/// Mapping from pairs of tokens to the rank and ID of the merged pair.
type MergeMap = HashMap<(TokenId, TokenId), (Rank, TokenId)>;

/// Build the BPE merge map that associates a rank and token ID to merged pairs
/// of tokens.
fn build_merge_map(
    vocab: &FxHashMap<EncodedBytes, TokenId>,
    merges: &[(EncodedByteSlice, EncodedByteSlice)],
) -> Result<MergeMap, BpeError> {
    let mut merge_map = HashMap::with_capacity(merges.len());

    for (i, (a, b)) in merges.iter().copied().enumerate() {
        let a_id = *vocab.get(a).ok_or_else(|| {
            BpeError::InvalidMergeEntry(format!(
                "first entry in merge pair \"{a} {b}\" not found in vocab"
            ))
        })?;
        let b_id = *vocab.get(b).ok_or_else(|| {
            BpeError::InvalidMergeEntry(format!(
                "second entry in merge pair \"{a} {b}\" not found in vocab"
            ))
        })?;
        let merged_str = [a, b].concat();
        let merged_id = *vocab.get(&merged_str).ok_or_else(|| {
            BpeError::InvalidMergeEntry(format!("merged pair \"{a} {b}\" not found in vocab"))
        })?;
        let rank = Rank(i as u32);
        merge_map.insert((a_id, b_id), (rank, merged_id));
    }

    Ok(merge_map)
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

/// Build a mapping from token bytes to ID using the merge list.
///
/// This is used as a fallback when the tokenizer configuration doesn't have a
/// vocabulary.
fn build_vocab(
    merges: &[(EncodedByteSlice, EncodedByteSlice)],
    end_of_word_suffix: Option<EncodedByteSlice>,
) -> FxHashMap<EncodedBytes, TokenId> {
    let mut vocab = FxHashMap::default();

    fn byte_to_rank() -> [Rank; 256] {
        let mut ranks = [Rank(0); 256];

        let mut rank = 0;
        for byte in 0..=255u8 {
            if is_printable(char::from(byte)) {
                ranks[byte as usize] = Rank(rank);
                rank += 1;
            }
        }

        for byte in 0..=255u8 {
            if !is_printable(char::from(byte)) {
                ranks[byte as usize] = Rank(rank);
                rank += 1;
            }
        }

        ranks
    }

    // The first 256 token IDs are reserved for individual bytes.
    for (ch, rank) in byte_to_char().into_iter().zip(byte_to_rank()) {
        vocab.insert(ch.into(), rank.0);
    }

    // If an end-of-word suffix is used, the next 256 token IDs are bytes that
    // occur at the end of a word.
    if let Some(eow_suffix) = end_of_word_suffix {
        let start_id = vocab.len() as u32;
        for (ch, rank) in byte_to_char().into_iter().zip(byte_to_rank()) {
            let mut bytes: EncodedBytes = ch.into();
            bytes.push_str(eow_suffix);
            vocab.insert(bytes, start_id + rank.0);
        }
    }

    // Assign token IDs to concatenated pairs from the merge list.
    let start_id = vocab.len() as u32;
    vocab.extend(
        merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ([*a, *b].concat(), start_id + i as u32)),
    );

    vocab
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
    pub vocab: Option<FxHashMap<EncodedBytes, TokenId>>,

    /// Set of tokens which don't appear in `merges` but do have a mapping in
    /// `vocab`. These are used for special purposes such as representing the
    /// end of output.
    pub added_tokens: FxHashMap<TokenId, String>,

    /// A string which is implicitly appended to each substring that is
    /// tokenized, after initial splitting.
    pub end_of_word_suffix: Option<String>,

    /// When encoding a string piece, match the entire piece against the
    /// vocabulary before applying merge rules.
    pub ignore_merges: bool,
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
    merges: MergeMap,

    /// Map from byte values to token IDs.
    byte_to_token_id: [TokenId; 256],

    /// Map from byte values to printable character representation used in
    /// vocabulary.
    byte_to_char: [char; 256],

    token_id_to_encoded_bytes: FxHashMap<TokenId, EncodedBytes>,

    vocab: Option<FxHashMap<EncodedBytes, TokenId>>,

    /// Map from token ID to content for special tokens (eg. end-of-string).
    added_tokens: FxHashMap<TokenId, String>,

    /// A suffix which is implicitly appended to each string piece to be
    /// tokenized.
    ///
    /// This was originally introduced for CLIP's tokenizer.
    /// See <https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py>.
    end_of_word_suffix: Option<String>,

    /// When encoding a string piece, match the entire piece against the
    /// vocabulary before applying merge rules.
    ignore_merges: bool,
}

impl Bpe {
    /// Create a new Byte Pair Encoding tokenizer using the given configuration.
    pub fn new(config: BpeOptions) -> Result<Bpe, BpeError> {
        let BpeOptions {
            merges,
            vocab,
            added_tokens,
            mut end_of_word_suffix,
            ignore_merges,
        } = config;

        // Normalize empty end-of-word suffix to `None`.
        end_of_word_suffix.take_if(|suffix| suffix.is_empty());

        let vocab = vocab.unwrap_or_else(|| build_vocab(merges, end_of_word_suffix.as_deref()));

        let merges = build_merge_map(&vocab, merges)?;

        // Build byte -> token ID mapping for encoding.
        let mut byte_to_token_id = [0; 256];
        for (i, ch) in byte_to_char().into_iter().enumerate() {
            let mut ch_buf = [0u8; 4];
            let ch_str = ch.encode_utf8(&mut ch_buf);
            if let Some(id) = vocab.get(ch_str).copied() {
                byte_to_token_id[i] = id;
            } else {
                return Err(BpeError::MissingVocabEntry(ch_str.to_string()));
            }
        }

        // If the `ignore_merges` flag is set for this tokenizer, we'll need
        // to use the vocabulary during encoding. Otherwise we can save some
        // memory by discarding it.
        let vocab_copy = if ignore_merges {
            Some(vocab.clone())
        } else {
            None
        };

        // Build token ID -> encoded byte mapping for decoding.
        let token_id_to_encoded_bytes = vocab.into_iter().map(|(token, id)| (id, token)).collect();

        Ok(Bpe {
            added_tokens,
            byte_to_char: byte_to_char(),
            byte_to_token_id,
            end_of_word_suffix,
            ignore_merges,
            merges,
            token_id_to_encoded_bytes,
            vocab: vocab_copy,
        })
    }

    /// Encode a string as a sequence of tokens.
    ///
    /// `end_of_word` specifies whether to apply end-of-word processing rules
    /// to the initial tokenization of piece.
    fn encode_piece(&self, piece: &str, end_of_word: bool) -> Vec<TokenId> {
        // If `ignore_merges` is set, check for the entire string in the vocab
        // before using merges.
        if self.ignore_merges
            && let Some(vocab) = self.vocab.as_ref()
        {
            let encoded: EncodedBytes = piece
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize])
                .collect();
            if let Some(&id) = vocab.get(&encoded) {
                return [id].into();
            }
        }

        // Start with one token per byte.
        let mut tokens: Vec<TokenId> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_token_id[b as usize])
            .collect();

        // If the end-of-word suffix is enabled, replace the last byte's token
        // with the one that corresponds to "{byte}{end_of_word_suffix}".
        if self.end_of_word_suffix.is_some()
            && end_of_word
            && let Some(last) = tokens.pop()
        {
            tokens.push(last + 256);
        }

        // Iteratively merge tokens together until no more are possible.
        bpe_merge(&mut tokens, &self.merges);

        tokens
    }
}

impl Model for Bpe {
    fn get_token_str(&self, id: TokenId) -> Option<String> {
        if let Some(tok_str) = self.added_tokens.get(&id) {
            return Some(tok_str.to_string());
        }
        self.token_id_to_encoded_bytes.get(&id).cloned()
    }

    fn get_token_id(&self, mut text: &str) -> Option<TokenId> {
        if let Some((&id, _str)) = self.added_tokens.iter().find(|(_id, str)| *str == text) {
            return Some(id);
        }

        // Determine the end-of-word context. eg. In CLIP's tokenizer, the
        // trailing "</w>" in "from</w>" indicates that it should be treated as
        // occurring at the end of a piece from the initial split.
        let mut end_of_word = false;
        if let Some(suffix) = self.end_of_word_suffix.as_deref()
            && text.ends_with(suffix)
        {
            text = &text[..text.len() - suffix.len()];
            end_of_word = true;
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
            } else if let Some(encoded_bytes) = self.token_id_to_encoded_bytes.get(&id) {
                bytes.extend(
                    encoded_bytes
                        .chars()
                        .map(|ch| char_to_byte.get(&ch).copied().unwrap()),
                );
            } else {
                return Err(DecodeError::InvalidTokenId(id));
            }
        }

        String::from_utf8(bytes).map_err(|_| DecodeError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;
    use rustc_hash::FxHashMap;

    use super::{Bpe, BpeOptions, EncodedBytes, merge_pairs_from_lines};
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

    fn added_tokens() -> FxHashMap<TokenId, String> {
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
    fn gen_vocab() -> FxHashMap<EncodedBytes, TokenId> {
        let mut next_token_id = 1000;
        let mut vocab = minimal_vocab(next_token_id);
        next_token_id += vocab.len() as u32;

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

    /// Generate the simplest valid vocabulary.
    fn minimal_vocab(start_token_id: u32) -> FxHashMap<EncodedBytes, TokenId> {
        let mut vocab = FxHashMap::default();
        let mut next_token_id = start_token_id;
        for ch in super::char_to_byte().keys() {
            vocab.insert(ch.to_string(), next_token_id);
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
            vocab: Option<FxHashMap<EncodedBytes, TokenId>>,
            end_of_word_suffix: Option<String>,
            ignore_merges: bool,
        }

        impl<'a> Default for Case<'a> {
            fn default() -> Self {
                Self {
                    text: "",
                    expected_tokens: &[],
                    merges: "",
                    vocab: None,
                    end_of_word_suffix: None,
                    ignore_merges: false,
                }
            }
        }

        let cases = [
            // Minimal test using a snippet of the GPT-2 merge list.
            Case {
                text: "the cat is in the bed",
                expected_tokens: &[
                    "t", "he", "Ġc", "at", "Ġ", "is", "Ġ", "in", "Ġthe", "Ġb", "ed",
                ],
                merges: MINI_GPT2,
                ..Default::default()
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
                ..Default::default()
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
                end_of_word_suffix: Some("</w>".to_string()),
                ..Default::default()
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
                end_of_word_suffix: Some("".to_string()),
                ..Default::default()
            },
            // `ignore_merges` option enabled
            Case {
                text: "foobar",
                expected_tokens: &["foobar"],
                ignore_merges: true,
                vocab: {
                    let mut vocab = minimal_vocab(0);
                    vocab.insert("foobar".to_string(), vocab.len() as u32);
                    Some(vocab)
                },
                ..Default::default()
            },
        ];

        cases.test_each(|case| {
            let Case {
                text,
                expected_tokens: tokens,
                merges,
                vocab,
                end_of_word_suffix,
                ignore_merges,
            } = case;

            let merges: Vec<&str> = merges.lines().collect();
            let merge_pairs = merge_pairs_from_lines(&merges);
            let bpe_opts = BpeOptions {
                merges: &merge_pairs,
                vocab: vocab.clone(),
                end_of_word_suffix: end_of_word_suffix.clone(),
                ignore_merges: *ignore_merges,
                added_tokens: Default::default(),
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
            vocab: Option<FxHashMap<EncodedBytes, TokenId>>,
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
