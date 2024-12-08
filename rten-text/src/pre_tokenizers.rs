//! Pre-tokenizers which split text after normalization and before encoding
//! into token IDs by models.

use std::error::Error;
use std::fmt;

use fancy_regex::Regex;
use unicode_categories::UnicodeCategories;

use crate::split::{SliceExt, SplitExt};

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

impl Error for PreTokenizeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RegexError(err) => Some(err),
        }
    }
}

impl From<fancy_regex::Error> for PreTokenizeError {
    fn from(val: fancy_regex::Error) -> Self {
        PreTokenizeError::RegexError(Box::new(val))
    }
}

/// A pre-tokenizer splits input text into chunks ("words") which are then
/// tokenized by a [`Model`](crate::models::Model) individually.
pub trait PreTokenizer {
    /// Split `text` into chunks and return a vector of sub-slices.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizeError>;
}

/// Split into tokens containing either digits or non-digits.
pub struct Digits {
    split: Split,
}

impl Digits {
    /// Construct a digit splitter.
    ///
    /// `individual_digits` specifies whether each digit in a sequence of digits
    /// should be its own token or not.
    pub fn new(individual_digits: bool) -> Digits {
        let pattern = if individual_digits {
            r"[0-9]|[^0-9]+"
        } else {
            r"[0-9]+|[^0-9]+"
        };

        Digits {
            split: Split::new(SplitOptions {
                pattern,
                invert: true,
                delimiter: SplitDelimiterBehavior::Remove,
            })
            .expect("pattern should be valid"),
        }
    }
}

impl PreTokenizer for Digits {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizeError> {
        self.split.pre_tokenize(text)
    }
}

/// Tokenization regex used by GPT-2.
///
/// See <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>.
pub const GPT2_REGEX: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Specifies how [`Split`] should handle delimiters between chunks.
#[derive(Copy, Clone, Default, PartialEq)]
pub enum SplitDelimiterBehavior {
    /// Exclude the delimiter from the output.
    #[default]
    Remove,

    /// Add the delimiter to the output as its own chunk.
    Isolate,
}

#[derive(Default)]
pub struct SplitOptions<'a> {
    pub pattern: &'a str,
    pub delimiter: SplitDelimiterBehavior,
    pub invert: bool,
}

/// Split input strings using a pattern.
pub struct Split {
    regex: Regex,
    delimiter: SplitDelimiterBehavior,
    invert: bool,
}

impl Split {
    /// Construct a pre-tokenizer which splits input using a given regex
    /// pattern.
    pub fn new(opts: SplitOptions) -> Result<Self, PreTokenizeError> {
        let SplitOptions {
            pattern,
            delimiter,
            invert,
        } = opts;
        let regex = Regex::new(pattern).map_err(|err| PreTokenizeError::RegexError(err.into()))?;

        Ok(Split {
            regex,
            delimiter,
            invert,
        })
    }

    /// Split input strings into chunks using the [`GPT2_REGEX`] pattern
    /// originating from GPT-2 and subsequently used by many other models.
    ///
    /// Use [`new`](Self::new) to specify a custom pattern.
    pub fn gpt2() -> Self {
        Self::new(SplitOptions {
            pattern: GPT2_REGEX,
            delimiter: SplitDelimiterBehavior::Remove,
            invert: true,
        })
        .expect("should be a valid pattern")
    }
}

impl PreTokenizer for Split {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizeError> {
        let mut chunks = Vec::new();
        let mut last_match_end = 0;

        if self.invert {
            for match_ in self.regex.find_iter(text) {
                let match_ = match_?;

                match self.delimiter {
                    SplitDelimiterBehavior::Isolate => {
                        let delim_text = &text[last_match_end..match_.range().start];
                        if !delim_text.is_empty() {
                            chunks.push(delim_text);
                        }
                    }
                    SplitDelimiterBehavior::Remove => {}
                }

                if !match_.range().is_empty() {
                    chunks.push(match_.as_str());
                }

                last_match_end = match_.range().end;
            }
        } else {
            for match_ in self.regex.split(text) {
                let match_ = match_?;
                let match_range = text
                    .as_bytes()
                    .subslice_offsets(match_.as_bytes())
                    .expect("should be sub-slice");

                match self.delimiter {
                    SplitDelimiterBehavior::Isolate => {
                        let delim_text = &text[last_match_end..match_range.start];
                        if !delim_text.is_empty() {
                            chunks.push(delim_text);
                        }
                    }
                    SplitDelimiterBehavior::Remove => {}
                }

                if !match_.is_empty() {
                    chunks.push(match_);
                }

                last_match_end = match_range.end;
            }
        }

        match self.delimiter {
            SplitDelimiterBehavior::Isolate => {
                let delim_text = &text[last_match_end..];
                if !delim_text.is_empty() {
                    chunks.push(delim_text);
                }
            }
            SplitDelimiterBehavior::Remove => {}
        }

        Ok(chunks)
    }
}

/// Pre-tokenizer that implements the pre-tokenization rules used by BERT.
///
/// This splits the input into tokens consisting of either punctuation,
/// white-space or non-punctuation.
pub struct Bert {}

impl Bert {
    pub fn new() -> Self {
        Bert {}
    }
}

impl Default for Bert {
    fn default() -> Self {
        Self::new()
    }
}

impl PreTokenizer for Bert {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizeError> {
        let is_punc_or_space =
            |ch: char| ch.is_ascii_punctuation() || ch.is_punctuation() || ch.is_whitespace();
        let words = text.split_keep_delimeters(is_punc_or_space).collect();
        Ok(words)
    }
}

/// Compose a sequence of pre-tokenizers.
pub struct Sequence {
    pre_tokenizers: Vec<Box<dyn PreTokenizer>>,
}

impl Sequence {
    pub fn from_vec(pre_tokenizers: Vec<Box<dyn PreTokenizer>>) -> Self {
        Sequence { pre_tokenizers }
    }
}

impl PreTokenizer for Sequence {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, PreTokenizeError> {
        let mut chunks = Vec::from([text]);
        for pre_tokenizer in &self.pre_tokenizers {
            let mut next_chunks = Vec::new();
            for chunk in chunks {
                let sub_chunks = pre_tokenizer.pre_tokenize(chunk)?;
                next_chunks.extend(sub_chunks);
            }
            chunks = next_chunks;
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Bert, Digits, PreTokenizer, Sequence, Split, SplitDelimiterBehavior, SplitOptions,
    };

    #[test]
    fn test_bert() {
        struct Case<'a> {
            input: &'a str,
            expected: Vec<&'a str>,
        }

        let cases = [Case {
            input: "foo. bar baz, meep",
            expected: ["foo", ".", " ", "bar", " ", "baz", ",", " ", "meep"].into(),
        }];

        for Case { input, expected } in cases {
            let bert = Bert::new();
            let chunks = bert.pre_tokenize(input).unwrap();
            assert_eq!(chunks, expected);
        }
    }

    #[test]
    fn test_digits() {
        struct Case<'a> {
            individual_digits: bool,
            input: &'a str,
            expected: Vec<&'a str>,
        }

        let cases = [
            // Examples from
            // https://huggingface.co/docs/tokenizers/en/api/pre-tokenizers#tokenizers.pre_tokenizers.Digits.
            Case {
                individual_digits: false,
                input: "Call 123 please",
                expected: ["Call ", "123", " please"].into(),
            },
            Case {
                individual_digits: true,
                input: "Call 123 please",
                expected: ["Call ", "1", "2", "3", " please"].into(),
            },
        ];

        for Case {
            individual_digits,
            input,
            expected,
        } in cases
        {
            let digits = Digits::new(individual_digits);
            let chunks = digits.pre_tokenize(input).unwrap();
            assert_eq!(chunks, expected);
        }
    }

    #[test]
    fn test_split() {
        struct Case<'a> {
            opts: SplitOptions<'a>,
            input: &'a str,
            expected: Vec<&'a str>,
        }

        let cases = [
            // Non-inverted
            Case {
                opts: SplitOptions {
                    pattern: r"\s+",
                    ..Default::default()
                },
                input: "foo bar   baz meep",
                expected: ["foo", "bar", "baz", "meep"].into(),
            },
            Case {
                opts: SplitOptions {
                    pattern: r"\s+",
                    delimiter: SplitDelimiterBehavior::Isolate,
                    ..Default::default()
                },
                input: " foo bar   baz meep ",
                expected: [" ", "foo", " ", "bar", "   ", "baz", " ", "meep", " "].into(),
            },
            // Inverted
            Case {
                opts: SplitOptions {
                    pattern: r"\s+",
                    invert: true,
                    ..Default::default()
                },
                input: "foo bar   baz meep",
                expected: [" ", "   ", " "].into(),
            },
            Case {
                opts: SplitOptions {
                    pattern: r"\s+",
                    invert: true,
                    delimiter: SplitDelimiterBehavior::Isolate,
                    ..Default::default()
                },
                input: "foo bar   baz meep",
                expected: ["foo", " ", "bar", "   ", "baz", " ", "meep"].into(),
            },
        ];

        for Case {
            opts,
            input,
            expected,
        } in cases
        {
            let split = Split::new(opts).unwrap();
            let chunks = split.pre_tokenize(input).unwrap();
            assert_eq!(chunks, expected);
        }
    }

    #[test]
    fn test_sequence() {
        let split_space: Box<dyn PreTokenizer> = Box::new(
            Split::new(SplitOptions {
                pattern: r"\s+",
                ..Default::default()
            })
            .unwrap(),
        );
        let split_punct = Box::new(
            Split::new(SplitOptions {
                pattern: r"\.",
                ..Default::default()
            })
            .unwrap(),
        );
        let seq = Sequence::from_vec([split_space, split_punct].into());

        let chunks = seq.pre_tokenize("foo.bar baz meep").unwrap();

        assert_eq!(chunks, ["foo", "bar", "baz", "meep"]);
    }
}
