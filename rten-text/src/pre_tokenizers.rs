use std::fmt;

use fancy_regex::Regex;
use unicode_categories::UnicodeCategories;

use crate::split::SplitExt;

/// Errors occuring while constructing a [`PreTokenizer`] or splitting input
/// using one.
#[derive(Clone, Debug)]
pub enum PreTokenizeError {
    /// An error occurred while constructing a regex from a pattern or
    /// splitting a string using a regex.
    RegexError(Box<fancy_regex::Error>),
}

impl fmt::Display for PreTokenizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegexError(err) => write!(f, "regex failed {}", err),
        }
    }
}

/// A pre-tokenizer splits input text into chunks ("words") which are then
/// tokenized by a [`Model`](crate::tokenizers::Model) individually.
pub trait PreTokenizer {
    /// Split `text` into chunks and return a vector of sub-slices.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizeError>;
}

/// Tokenization regex used by GPT-2.
///
/// See <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>.
pub const GPT2_REGEX: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct ByteLevelPreTokenizer {
    /// Pattern used to split the text into pieces.
    splitter: Regex,
}

impl ByteLevelPreTokenizer {
    /// Construct a pre-tokenizer which splits input using a given regex
    /// pattern.
    pub fn new(regex_pattern: &str) -> Result<Self, PreTokenizeError> {
        let splitter =
            Regex::new(regex_pattern).map_err(|err| PreTokenizeError::RegexError(err.into()))?;
        Ok(ByteLevelPreTokenizer { splitter })
    }

    /// Return a `ByteLevelPreTokenizer` configured using the standard regex
    /// originating from GPT-2.
    ///
    /// Use [`new`](Self::new) to specify a custom pattern.
    pub fn gpt2() -> Self {
        Self::new(GPT2_REGEX).expect("should be a valid pattern")
    }
}

impl PreTokenizer for ByteLevelPreTokenizer {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizeError> {
        self.splitter
            .find_iter(text)
            .filter_map(|piece| match piece {
                Ok(piece) => {
                    if piece.range().is_empty() {
                        None
                    } else {
                        Some(Ok(piece.as_str()))
                    }
                }
                Err(err) => Some(Err(PreTokenizeError::RegexError(Box::new(err)))),
            })
            .collect()
    }
}

pub struct BertPreTokenizer {}

impl BertPreTokenizer {
    pub fn new() -> Self {
        BertPreTokenizer {}
    }
}

impl Default for BertPreTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizeError> {
        let is_punc_or_space =
            |ch: char| ch.is_ascii_punctuation() || ch.is_punctuation() || ch.is_whitespace();
        let words = text.split_keep_delimeters(is_punc_or_space).collect();
        Ok(words)
    }
}
