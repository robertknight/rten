use std::collections::HashMap;

/// A naive text-to-phoneme converter.
///
/// This converts input text to sequences of [IPA][ipa] phonemes by looking up
/// words in a dictionary. For example the text "This is a text to speech
/// system" is converted to the IPA phoneme sequence "ðɪs ɪz ɐ tˈɛkst tə spˈiːtʃ
/// sˈɪstəm.".
///
/// The Piper project uses the [piper-phonemize][pp] library, which does a
/// more sophisticated translation.
///
/// [ipa]: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet
/// [pp]: https://github.com/rhasspy/piper-phonemize
pub struct Phonemizer {
    dict: HashMap<String, String>,
}

impl Phonemizer {
    /// Create a phonemizer from the contents of a dictionary file.
    ///
    /// The format of the dictionary should be a text file with one translation
    /// per line. Each line should have the form `<word> TAB <phonemes>`.
    /// Dictionaries can be generated using the `pronounciation-dict.py` Python
    /// script.
    pub fn load_dict(dict_content: &str) -> Phonemizer {
        let dict = dict_content
            .lines()
            .filter_map(|line| {
                line.trim()
                    .split_once('\t')
                    .map(|(text, phonemes)| (text.to_lowercase(), phonemes.to_string()))
            })
            .collect();
        Phonemizer { dict }
    }

    /// Translate an input text string to a sequence of phonemes.
    ///
    /// Any words that do not appear in the dictionary will be skipped.
    pub fn translate(&self, text: &str) -> String {
        let mut phonemes = String::new();
        for word in text.trim().split_whitespace() {
            if let Some(word_phonemes) = self.dict.get(word.to_lowercase().as_str()) {
                if !phonemes.is_empty() {
                    phonemes.push(' ');
                }
                phonemes.push_str(word_phonemes);
            }
        }
        phonemes
    }
}
