//! Tools for performing string normalization prior to tokenization.

use unicode_categories::UnicodeCategories;
use unicode_normalization::char::decompose_canonical;

struct CharNormalizer {
    normalized: Vec<char>,

    /// Temporary buffer that holds the output of a normalization step until
    /// it is copied back to `normalized`.
    tmp: Vec<char>,
}

impl CharNormalizer {
    fn new() -> CharNormalizer {
        CharNormalizer {
            normalized: Vec::new(),
            tmp: Vec::new(),
        }
    }

    /// Set the input character to normalize.
    fn set_char(&mut self, ch: char) {
        self.tmp.push(ch);
        self.update_normalized_from_tmp();
    }

    /// Lowercase the normalized characters.
    fn lower_case(&mut self) {
        for ch in &self.normalized {
            for lower_ch in ch.to_lowercase() {
                self.tmp.push(lower_ch);
            }
        }
        self.update_normalized_from_tmp();
    }

    /// Decompose the input into NFD form and then remove any characters in
    /// the Unicode non-spacing mark ("Mn") category.
    fn strip_accents(&mut self) {
        for ch in &self.normalized {
            decompose_canonical(*ch, |decomposed| {
                if !decomposed.is_mark_nonspacing() {
                    self.tmp.push(decomposed);
                }
            });
        }
        self.update_normalized_from_tmp();
    }

    /// Return the normalized characters.
    fn normalized(&self) -> &[char] {
        &self.normalized
    }

    fn update_normalized_from_tmp(&mut self) {
        self.normalized.clear();
        self.normalized.extend(self.tmp.iter());
        self.tmp.clear();
    }
}

/// Normalizer applies normalization such as Unicode normalization and
/// lower-casing to strings.
///
/// In addition to the normalized text, Normalizer methods also return mappings
/// from positions in the normalized string back to the original string. This
/// is useful for post-processing in NLP tasks to map machine learning model
/// outputs back to the location in the original text.
#[derive(Clone, Debug)]
pub struct Normalizer {
    lowercase: bool,
    strip_accents: bool,
}

/// Configuration for a [Normalizer].
#[derive(Clone, Debug, Default)]
pub struct NormalizerOptions {
    /// If true, convert all text to lowercase using [char::to_lowercase].
    pub lowercase: bool,

    /// Whether to strip accents when tokenizing. An "accent" is defined as
    /// any unicode character in the Nonspacing Mark ("Mn") category.
    pub strip_accents: bool,
}

impl Normalizer {
    pub fn new(opts: NormalizerOptions) -> Normalizer {
        Normalizer {
            lowercase: opts.lowercase,
            strip_accents: opts.strip_accents,
        }
    }

    /// Apply normalization to a string.
    ///
    /// Returns a tuple of `(normalized_string, offset_map)` where `offset_map`
    /// is a mapping from byte offsets in the normalized string to corresponding
    /// offsets in the original string.
    pub fn normalize(&self, text: &str) -> (String, Vec<usize>) {
        if self.is_noop() {
            let offsets = (0..text.len()).collect();
            return (text.to_string(), offsets);
        }

        let mut normalized = String::with_capacity(text.len());
        let mut offsets = Vec::with_capacity(text.len());
        let mut char_normalizer = CharNormalizer::new();

        for (offset, ch) in text.char_indices() {
            char_normalizer.set_char(ch);

            if self.strip_accents {
                char_normalizer.strip_accents();
            }

            if self.lowercase {
                char_normalizer.lower_case();
            }

            for ch in char_normalizer.normalized() {
                normalized.push(*ch);
                for _ in 0..ch.len_utf8() {
                    offsets.push(offset);
                }
            }
        }

        (normalized, offsets)
    }

    /// Return true if this normalizer doesn't alter its input.
    fn is_noop(&self) -> bool {
        !self.lowercase && !self.strip_accents
    }
}

#[cfg(test)]
mod tests {
    use super::{Normalizer, NormalizerOptions};

    #[test]
    fn test_normalizer_noop() {
        let normalizer = Normalizer::new(NormalizerOptions::default());
        let inputs = [
            "Hello world!", // Mixed case
            "Motörhead",    // Accented
            "lowercase",
        ];
        for input in inputs {
            let (normalized, offsets) = normalizer.normalize(input);
            assert_eq!(normalized, input);
            assert_eq!(offsets, (0..input.len()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_normalizer_lowercase() {
        let normalizer = Normalizer::new(NormalizerOptions {
            lowercase: true,
            ..Default::default()
        });

        struct Case<'a> {
            input: &'a str,
            expected: &'a str,
            expected_offsets: Vec<usize>,
        }

        let cases = [
            // Simple text where chars map 1:1 to lower-case version
            Case {
                input: "Hello World!",
                expected: "hello world!",
                expected_offsets: (0.."hello world!".len()).collect(),
            },
            // Text with chars which expand when lower-cased
            Case {
                input: "İİAB",
                expected: "i\u{307}i\u{307}ab",

                // The "İ" char requires two bytes in the input and expands into
                // two characters which require one and three bytes
                // respectively. Hence the offsets contain two groups of three
                // equal offsets, with values separated by two.
                expected_offsets: vec![0, 0, 0, 2, 2, 2, 4, 5],
            },
        ];

        for Case {
            input,
            expected,
            expected_offsets,
        } in cases
        {
            let (normalized, offsets) = normalizer.normalize(input);
            assert_eq!(normalized, expected);
            assert_eq!(offsets, expected_offsets);
        }
    }

    #[test]
    fn test_normalizer_strip_accepts() {
        struct Case<'a> {
            input: &'a str,
            lowercase: bool,
            expected: &'a str,
            expected_offsets: Vec<usize>,
        }

        let cases = [
            // Strip accents only
            Case {
                input: "Motörhead",
                lowercase: false,
                expected: "Motorhead",
                // Note jump in offset where the two UTF-8 char "ö" is replaced
                // with "o".
                expected_offsets: vec![0, 1, 2, 3, 5, 6, 7, 8, 9],
            },
            // Combined lowercase + strip accents
            Case {
                input: "Motörhead",
                lowercase: true,
                expected: "motorhead",
                // Note jump in offset where the two UTF-8 char "ö" is replaced
                // with "o".
                expected_offsets: vec![0, 1, 2, 3, 5, 6, 7, 8, 9],
            },
        ];

        for Case {
            input,
            lowercase,
            expected,
            expected_offsets,
        } in cases
        {
            let normalizer = Normalizer::new(NormalizerOptions {
                lowercase,
                strip_accents: true,
                ..Default::default()
            });

            let (normalized, offsets) = normalizer.normalize(input);
            assert_eq!(normalized, expected);
            assert_eq!(offsets, expected_offsets);
        }
    }
}
