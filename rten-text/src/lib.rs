//! This crate provides text tokenizers for preparing inputs for
//! inference of machine-learning models. It provides implementations of
//! popular tokenization methods such as WordPiece (used by BERT),
//! and Byte Pair Encoding (used by GPT-2).
//!
//! It does not support training new vocabularies and isn't optimized for
//! processing very large volumes of text. If you need a tokenization crate
//! with more complete functionality, see
//! [HuggingFace tokenizers](https://github.com/huggingface/tokenizers).

pub mod models;
pub mod normalizer;
pub mod pretokenizers;
pub mod tokenizers;

mod split;
