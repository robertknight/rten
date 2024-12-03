use std::collections::HashMap;
use std::error::Error;
use std::fs::read_to_string;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;

use rten_text::models::{
    merge_pairs_from_lines, patterns::GPT2 as GPT2_SPLIT_PATTERN, Bpe, WordPiece, WordPieceOptions,
};
use rten_text::normalizer::{BertNormalizer, BertNormalizerOptions};
use rten_text::tokenizers::{TokenId, Tokenizer, TokenizerOptions};
use serde::Deserialize;

/// Load a vocabulary from a text file with one token per line (ie. the
/// vocab.txt files that come with Hugging Face models).
fn read_vocab_text_file(path: &str) -> Result<HashMap<String, TokenId>, io::Error> {
    let content = read_test_file(path)?;
    Ok(content
        .lines()
        .enumerate()
        .map(|(i, line)| (line.to_string(), i as TokenId))
        .collect())
}

/// Struct representing the JSON files in `reftests/`.
#[derive(Deserialize)]
struct ReferenceTokenization {
    token_ids: Vec<TokenId>,
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
fn compare_tokens(actual: &[TokenId], expected: &[TokenId]) -> Result<(), Box<dyn Error>> {
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

fn wordpiece_tokenizer_opts() -> TokenizerOptions<'static> {
    TokenizerOptions {
        cls_token: Some("[CLS]"),
        sep_token: Some("[SEP]"),
    }
}

#[test]
fn test_wordpiece_bert_cased() -> Result<(), Box<dyn Error>> {
    let vocab = read_vocab_text_file("models/bert-base-cased/vocab.txt")?;
    let text = read_test_file("Rust_(programming_language).txt")?;
    let expected =
        ReferenceTokenization::from_file("Rust_(programming_language)-bert-base-cased.json")?;

    let model = WordPiece::from_vocab(vocab, Default::default());
    let tokenizer = Tokenizer::new(model, wordpiece_tokenizer_opts());
    let encoded = tokenizer.encode(text.as_str(), None)?;

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

    let normalizer = BertNormalizer::new(BertNormalizerOptions {
        lowercase: true,
        strip_accents: true,
        ..Default::default()
    });
    let model = WordPiece::from_vocab(
        vocab,
        WordPieceOptions {
            normalizer: Some(Rc::new(normalizer)),
            ..Default::default()
        },
    );
    let tokenizer = Tokenizer::new(model, wordpiece_tokenizer_opts());

    for Case { text, reference } in cases {
        let text = read_test_file(text)?;
        let expected = ReferenceTokenization::from_file(reference)?;
        let encoded = tokenizer.encode(text.as_str(), None)?;

        compare_tokens(encoded.token_ids(), &expected.token_ids)?;
    }

    Ok(())
}

#[test]
fn test_bpe_gpt2() -> Result<(), Box<dyn Error>> {
    struct Case<'a> {
        text: &'a str,
        reference: &'a str,
    }

    let cases = [Case {
        text: "monty-python-credits.txt",
        reference: "monty-python-credits-gpt2.json",
    }];

    // Create tokenizer manually.
    let merges = read_test_file("models/gpt2/merges.txt")?;
    let merge_lines: Vec<_> = merges.lines().collect();
    let merge_pairs = merge_pairs_from_lines(&merge_lines);
    let model = Bpe::new(
        &merge_pairs,
        GPT2_SPLIT_PATTERN,
        None,
        Default::default(),
        None,
    )?;
    let tokenizer = Tokenizer::new(model, Default::default());

    // Create tokenizer from a `tokenizers.json` file.
    let tokenizer_json = read_test_file("models/gpt2/tokenizer.json")?;
    let tokenizer_from_json = Tokenizer::from_json(&tokenizer_json)?;

    for Case { text, reference } in cases {
        let text = read_test_file(text)?;
        let expected = ReferenceTokenization::from_file(reference)?;

        let encoded = tokenizer.encode(text.as_str(), None)?;
        compare_tokens(encoded.token_ids(), &expected.token_ids)?;

        let encoded = tokenizer_from_json.encode(text.as_str(), None)?;
        compare_tokens(encoded.token_ids(), &expected.token_ids)?;
    }

    Ok(())
}
