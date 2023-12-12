use std::error::Error;
use std::fs::read_to_string;
use std::io;
use std::path::PathBuf;

use serde::Deserialize;
use wasnn_text::tokenizers::{Tokenizer, WordPiece};

fn read_test_file(path: &str) -> Result<String, io::Error> {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push("reftests/");
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

#[derive(Deserialize)]
struct ReferenceTokenization {
    token_ids: Vec<usize>,
}

#[test]
fn test_wordpiece_bert_cased() -> Result<(), Box<dyn Error>> {
    let vocab = read_test_file("models/bert-base-cased/vocab.txt")?;
    let vocab: Vec<_> = vocab.lines().collect();
    let text = read_test_file("Rust_(programming_language).txt")?;
    let expected_json = read_test_file("Rust_(programming_language)-bert-base-cased.json")?;
    let expected: ReferenceTokenization = serde_json::from_str(&expected_json)?;

    let tokenizer = WordPiece::from_vocab(&vocab, Default::default());
    let encoded = tokenizer.encode(text.as_str().into(), Default::default())?;

    compare_tokens(encoded.token_ids(), &expected.token_ids)?;

    Ok(())
}
