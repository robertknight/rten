//! This crate provides tools for pre and post-processing text inputs and
//! outputs of models. This primarily means tokenizing and de-tokenizing text.
//!
//! If you need a more featureful set of tokenizers, see the
//! [tokenizers](https://github.com/huggingface/tokenizers) project.

pub mod normalizer;
pub mod tokenizers;

mod split;
