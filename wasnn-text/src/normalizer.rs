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
}

#[derive(Clone, Debug, Default)]
pub struct NormalizerOptions {
    /// If true, convert all text to lowercase using [char::to_lowercase].
    pub lowercase: bool,
}

impl Normalizer {
    pub fn new(opts: NormalizerOptions) -> Normalizer {
        Normalizer {
            lowercase: opts.lowercase,
        }
    }

    /// Apply normalization to a string.
    ///
    /// Returns a tuple of `(normalized_string, offset_map)` where `offset_map`
    /// is a mapping from byte offsets in the normalized string to corresponding
    /// offsets in the original string.
    pub fn normalize(&self, text: &str) -> (String, Vec<usize>) {
        if self.lowercase {
            let mut normalized = String::with_capacity(text.len());
            let mut offsets = Vec::with_capacity(text.len());

            for (offset, ch) in text.char_indices() {
                for lower_ch in ch.to_lowercase() {
                    normalized.push(lower_ch);
                    for _ in 0..lower_ch.len_utf8() {
                        offsets.push(offset);
                    }
                }
            }

            (normalized, offsets)
        } else {
            let offsets = (0..text.len()).collect();
            (text.to_string(), offsets)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Normalizer, NormalizerOptions};

    #[test]
    fn test_normalizer_noop() {
        let normalizer = Normalizer::new(NormalizerOptions::default());
        let input = "Hello world!";
        let (normalized, offsets) = normalizer.normalize(input);
        assert_eq!(normalized, input);
        assert_eq!(offsets, (0..input.len()).collect::<Vec<_>>());
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
}
