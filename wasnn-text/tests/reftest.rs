use std::collections::HashMap;
use std::error::Error;
use std::fs::read_to_string;
use std::io;
use std::path::PathBuf;

use serde::Deserialize;
use wasnn_text::normalizer::{Normalizer, NormalizerOptions};
use wasnn_text::tokenizers::{Tokenizer, WordPiece, WordPieceOptions};

/// Load a vocabulary from a text file with one token per line (ie. the
/// vocab.txt files that come with Hugging Face models).
fn read_vocab_text_file(path: &str) -> Result<HashMap<String, usize>, io::Error> {
    let content = read_test_file(path)?;
    Ok(content
        .lines()
        .enumerate()
        .map(|(i, line)| (line.to_string(), i))
        .collect())
}

/// Struct representing the JSON files in `reftests/`.
#[derive(Deserialize)]
struct ReferenceTokenization {
    token_ids: Vec<usize>,
}

impl ReferenceTokenization {
    /// Load a reference tokenization from a JSON input file.
    fn from_file(path: &str) -> Result<ReferenceTokenization, Box<dyn Error>> {
        let json = read_test_file(path)?;
        let ref_tok = serde_json::from_str(&json)?;
        Ok(ref_tok)
    }
}

fn read_test_file(path: &str) -> Result<String, io::Error> {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push("test-data/reftests/");
    abs_path.push(path);
    read_to_string(abs_path)
}

/// Compare two slices of token IDs and return an error if there are any
/// mismatches.
fn compare_tokens(actual: &[usize], expected: &[usize]) -> Result<(), Box<dyn Error>> {
    for (i, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if actual != expected {
            return Err(
                format!("Tokens differ at index {i}. Actual {actual} expected {expected}").into(),
            );
        }
    }

    // Check for length mismatch after comparing tokens so that errors about
    // too many / too few tokens are reported after earlier tokens have been
    // compared.
    if actual.len() != expected.len() {
        return Err(format!(
            "Lengths of token slices do not match. Actual {} expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }

    Ok(())
}

#[test]
fn test_wordpiece_bert_cased() -> Result<(), Box<dyn Error>> {
    let vocab = read_vocab_text_file("models/bert-base-cased/vocab.txt")?;
    let text = read_test_file("Rust_(programming_language).txt")?;
    let expected =
        ReferenceTokenization::from_file("Rust_(programming_language)-bert-base-cased.json")?;

    let encoder = WordPiece::from_vocab(vocab, Default::default());
    let tokenizer = Tokenizer::new(encoder);
    let encoded = tokenizer.encode(text.as_str().into(), Default::default())?;

    compare_tokens(encoded.token_ids(), &expected.token_ids)?;

    Ok(())
}

#[test]
fn test_wordpiece_bert_uncased() -> Result<(), Box<dyn Error>> {
    struct Case<'a> {
        text: &'a str,
        reference: &'a str,
    }

    let cases = [
        // ASCII text
        Case {
            text: "Rust_(programming_language).txt",
            reference: "Rust_(programming_language)-bert-base-uncased.json",
        },
        // Non-ASCII text
        Case {
            text: "monty-python-credits.txt",
            reference: "monty-python-credits-bert-base-uncased.json",
        },
        // Accents
        Case {
            text: "Metal_umlaut.txt",
            reference: "Metal_umlaut-bert-base-uncased.json",
        },
    ];

    let vocab = read_vocab_text_file("models/bert-base-uncased/vocab.txt")?;

    let normalizer = Normalizer::new(NormalizerOptions {
        lowercase: true,
        strip_accents: true,
        ..Default::default()
    });
    let encoder = WordPiece::from_vocab(
        vocab,
        WordPieceOptions {
            normalizer: Some(normalizer),
            ..Default::default()
        },
    );
    let tokenizer = Tokenizer::new(encoder);

    for Case { text, reference } in cases {
        let text = read_test_file(text)?;
        let expected = ReferenceTokenization::from_file(reference)?;
        let encoded = tokenizer.encode(text.as_str().into(), Default::default())?;

        compare_tokens(encoded.token_ids(), &expected.token_ids)?;
    }

    Ok(())
}
