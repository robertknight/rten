use std::collections::VecDeque;
use std::error::Error;
use std::fs;

use wasnn::ops::{concat, OpError};
use wasnn::{FloatOperators, Input, Model, NodeId, Operators};
use wasnn_tensor::prelude::*;
use wasnn_tensor::{NdTensor, NdTensorView, Tensor};
use wasnn_text::normalizer::{Normalizer, NormalizerOptions};
use wasnn_text::tokenizers::{EncodeOptions, Tokenizer, WordPiece, WordPieceOptions};

struct Args {
    model: String,
    vocab: String,
    first_sentence: String,
    second_sentence: String,

    #[allow(dead_code)]
    verbose: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();
    let mut verbose = false;

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Short('v') | Long("verbose") => verbose = true,
            Long("help") => {
                println!(
                    "Estimate semantic similarity of two sentences.

Usage: {bin_name} <model> <vocab> <first_sentence> <second_sentence>

Args:

  <model>       - Input model
  <vocab>       - Vocabulary for tokenization (vocab.txt)
  <first_sentence>  - First input sentence to process
  <second_sentence> - Second input sentence to process

Options:

  -v, --verbose - Output debug info
",
                    bin_name = parser.bin_name().unwrap_or("jina_similarity")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let vocab = values.pop_front().ok_or("missing `vocab` arg")?;
    let first_sentence = values.pop_front().ok_or("missing `first_sentence` arg")?;
    let second_sentence = values.pop_front().ok_or("missing `second_sentence` arg")?;

    let args = Args {
        model,
        vocab,
        first_sentence,
        second_sentence,
        verbose,
    };

    Ok(args)
}

/// Generate embeddings for a slice of sentences.
///
/// Returns a `(batch, embed_dim)` tensor where `batch` is equal to `sentences.len()`.
fn embed_sentence_batch(
    sentences: &[&str],
    tokenizer: &WordPiece,
    model: &Model,
    max_seq_len: usize,
) -> Result<NdTensor<f32, 2>, Box<dyn Error>> {
    // Tokenize input sequences
    let mut encoded = Vec::new();
    for &sentence in sentences {
        encoded.push(tokenizer.encode(
            sentence.into(),
            EncodeOptions {
                max_chunk_len: Some(max_seq_len),
                ..Default::default()
            },
        )?);
    }

    // Generate (batch, token_id) input.
    let max_sequence_len = encoded
        .iter()
        .map(|enc| enc.token_ids().len())
        .max()
        .unwrap_or(0);
    let batch = sentences.len();
    let mut input_ids = Tensor::zeros(&[batch, max_sequence_len]);
    for (i, encoded) in encoded.iter().enumerate() {
        let token_ids = encoded.token_ids();
        for (tid, input_id) in token_ids
            .iter()
            .zip(input_ids.slice_mut((i, ..token_ids.len())).iter_mut())
        {
            *input_id = *tid as i32;
        }
    }

    // Generate attention mask, set to 1 for non-padding tokens and 0 for
    // padding tokens.
    let mut attention_mask = Tensor::zeros(&[batch, max_sequence_len]);
    for (i, encoded) in encoded.iter().enumerate() {
        attention_mask
            .slice_mut((i, ..encoded.token_ids().len()))
            .fill(1i32);
    }

    let input_ids_id = model.node_id("input_ids")?;
    let attention_mask_id = model.node_id("attention_mask")?;

    let mut inputs: Vec<(NodeId, Input)> = vec![
        (input_ids_id, input_ids.view().into()),
        (attention_mask_id, attention_mask.view().into()),
    ];

    // Generate token type IDs if this model needs them. These are all zeros
    // since each item has just one sequence.
    let type_ids: Tensor<i32>;
    if let Some(type_ids_id) = model.find_node("token_type_ids") {
        type_ids = Tensor::zeros(&[batch, max_sequence_len]);
        inputs.push((type_ids_id, type_ids.view().into()));
    }

    let output_id = model.node_id("last_hidden_state")?;
    let [last_hidden_state] = model.run_n(&inputs, [output_id], None)?;
    let last_hidden_state = last_hidden_state.into_float().ok_or("wrong output type")?;

    // Mean pool each item in the batch. We process each batch item separately
    // since they can have different lengths.
    let mean_pooled: Vec<_> = last_hidden_state
        .axis_iter(0)
        .zip(encoded.iter())
        .map(|(item, input)| {
            // Take the mean of the non-padding elements along the sequence
            // dimension.
            let seq_len = input.token_ids().len();
            item.slice(..seq_len)
                .reduce_mean(Some(&[0]), false /* keep_dims */)
                .unwrap()
        })
        .collect();
    let mean_pooled_views: Vec<_> = mean_pooled
        .iter()
        .map(|mp| {
            // Re-add batch dim.
            let mut view = mp.view();
            view.insert_dim(0);
            view
        })
        .collect();
    let mean_pooled: NdTensor<f32, 2> = concat(&mean_pooled_views, 0)?.try_into()?;
    Ok(mean_pooled)
}

/// Return the cosine similarity between two vectors.
///
/// Fails if the vectors are not of equal length.
fn cosine_similarity(a: NdTensorView<f32, 1>, b: NdTensorView<f32, 1>) -> Result<f32, OpError> {
    let dot_prod: f32 = a.mul(b.as_dyn())?.iter().sum();
    let a_len = a
        .reduce_l2(None /* axes */, false /* keep_dims */)?
        .item()
        .copied()
        .unwrap();
    let b_len = b
        .reduce_l2(None /* axes */, false /* keep_dims */)?
        .item()
        .copied()
        .unwrap();
    Ok(dot_prod / (a_len * b_len))
}

/// This example computes the semantic similarity between two sentences or
/// documents.
///
/// It works with BERT-based models designed for generating embeddings,
/// such as https://huggingface.co/jinaai/jina-embeddings-v2-small-en.
///
/// You can download the Jina embeddings model in ONNX format, along with the
/// vocab.txt vocabulary file from https://huggingface.co/jinaai/jina-embeddings-v2-small-en/tree/main.
///
/// Convert the model using:
///
/// ```
/// tools/convert-onnx.py jina-embed.onnx jina-embed.model
/// ```
///
/// Then run the example with:
///
/// ```
/// cargo run -r --example jina_similarity jina-embed.model jina-vocab.txt
///   <first_sentence> <second_sentence>
/// ```
///
/// Where `<first_sentence>` and `<second_sentence>` are two quoted sentences
/// to compare. For example "How is the weather today?" and "What is the current
/// weather like today?".
///
/// [1] https://huggingface.co/tasks/question-answering
/// [2] https://huggingface.co/docs/optimum/index
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model_bytes = fs::read(args.model)?;
    let model = Model::load(&model_bytes)?;

    let vocab_text = std::fs::read_to_string(&args.vocab)?;
    let vocab: Vec<_> = vocab_text.lines().collect();

    let normalizer = Normalizer::new(NormalizerOptions {
        lowercase: true,
        ..Default::default()
    });
    let tokenizer = WordPiece::from_vocab(
        &vocab,
        WordPieceOptions {
            normalizer: Some(normalizer),
            ..Default::default()
        },
    );

    let first_sentence = args.first_sentence;
    let second_sentence = args.second_sentence;

    // Max sequence length supported by Jina embeddings.
    // See notes in https://huggingface.co/jinaai/jina-embeddings-v2-base-en.
    let max_sequence_len = 8192;

    let embeddings = embed_sentence_batch(
        &[first_sentence.as_str(), second_sentence.as_str()],
        &tokenizer,
        &model,
        max_sequence_len,
    )?;
    let similarity = cosine_similarity(embeddings.slice(0), embeddings.slice(1))?;

    println!("First sentence: \"{}\"", first_sentence);
    println!("Second sentence: \"{}\"", second_sentence);
    println!("Similarity: {similarity}");

    Ok(())
}
