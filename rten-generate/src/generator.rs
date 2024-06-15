//! Tools to run the generation loop for an auto-regressive model.

use std::error::Error;
use std::fmt;

use rten::{Dimension, Input, InputOrOutput, Model, NodeId, Output};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};
use rten_text::tokenizers::{Tokenizer, TokenizerError};

use crate::metrics::Metrics;
use crate::sampler::{ArgMaxSampler, Sampler};

/// Errors that occur when creating or running a [`Generator`].
#[derive(Debug)]
pub enum GeneratorError {
    /// An expected model input was not found.
    InputNotFound(String),

    /// An expected model output was not found.
    OutputNotFound(String),

    /// An input or output did not have the expected shape.
    ShapeMismatch(String),

    /// An error occurred while generating the next token.
    GenerateError(Box<dyn Error>),

    /// An error occurred while decoding tokens.
    DecodeError(TokenizerError),
}

impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GeneratorError::InputNotFound(name) => write!(f, "model input not found: {}", name),
            GeneratorError::OutputNotFound(name) => write!(f, "model output not found: {}", name),
            GeneratorError::ShapeMismatch(err) => write!(f, "shape mismatch: {}", err),
            GeneratorError::GenerateError(err) => write!(f, "generation error: {}", err),
            GeneratorError::DecodeError(err) => write!(f, "decode error: {}", err),
        }
    }
}

impl Error for GeneratorError {}

/// Key-value cache for a single layer of a transformer model.
struct KvCache {
    /// Input ID for this cache entry.
    input_id: NodeId,

    /// Output ID for this cache entry.
    output_id: NodeId,

    /// The cached keys and values, with shape [batch, heads, seq_len, size].
    cache: NdTensor<f32, 4>,
}

/// Generates a token ID sequence using an auto-regressive language model.
///
/// This is an iterator that runs the model on each call to [`Iterator::next`]
/// and yields a result containing the next token ID or an error.
///
/// The token ID sequence can be converted to text using the
/// [`decode`](GeneratorUtils::decode) method of the [`GeneratorUtils`] trait.
/// This trait also provides useful wrappers for the output, such as stopping
/// generation when an end-of-text token is reached. You can also use all of
/// the standard iterator adapters. For example `generator.take(30)` will
/// return an iterator that stops generation after 30 tokens have been produced).
///
/// ## Sampling
///
/// The token ID is sampled from the outputs of the model (the "logits") using
/// a [`Sampler`]. By default this is an [`ArgMaxSampler`] which simply chooses
/// the token with the highest probability. The sampler can be configured using
/// [`with_sampler`](Self::with_sampler).
pub struct Generator<'a> {
    model: &'a Model,

    /// Additional model inputs (eg. encoder outputs) passed to each model
    /// step.
    constant_inputs: Vec<(NodeId, InputOrOutput<'a>)>,

    /// Additional model inputs computed using constant propagation. This
    /// effectively caches parts of the graph that don't change in each
    /// generation step. This is `None` if the cache is out of date.
    constant_prop_inputs: Option<Vec<(NodeId, Output)>>,

    /// Input token IDs for the next run of the model.
    input_ids: Vec<u32>,

    // Input node IDs
    input_ids_input: NodeId,
    attention_mask_input: Option<NodeId>,
    position_ids_input: Option<NodeId>,

    // Output node IDs
    logits_output: NodeId,

    // Sampler used to get the next token ID from the output logits.
    sampler: Box<dyn Sampler>,

    /// Length of the sequence generated so far.
    seq_len: u32,

    /// Key-value cache.
    kv_cache: Vec<KvCache>,
}

impl<'a> Generator<'a> {
    /// Create a generator that iteratively produces tokens using a model.
    ///
    /// The model must have the required inputs:
    ///
    ///  - `input_ids` - (batch, sequence) tensor of token IDs
    ///
    /// The model may have the optional inputs:
    ///
    ///  - `attention_mask` - (batch, sequence) tensor of booleans
    ///  - `position_ids` - (batch, sequence) tensor of position indices
    ///  - `past_key_values.N.key` - (batch, head, past_seq_len, size) key vector cache
    ///    where `N` is the layer index
    ///  - `past_key_values.N.value` - (batch, head, past_key_values, size) value vector cache,
    ///    where `N` is the layer index
    ///
    /// The model must have the outputs:
    ///
    ///  - `logits` - output (batch, sequence, vocab) tensor of next token probabilities
    ///
    /// The model may have the optional outputs:
    ///
    ///  - `present.N.key` - (batch, head, past_seq_len + 1, size) updated key vector cache
    ///  - `present.N.value` - (batch, head, past_seq_len + 1, size) updated value vector cache
    pub fn from_model(model: &'a Model) -> Result<Generator<'a>, GeneratorError> {
        let input_ids_input = model
            .find_node("input_ids")
            .ok_or(GeneratorError::InputNotFound("input_ids".to_string()))?;
        let attention_mask_input = model.find_node("attention_mask");
        let position_ids_input = model.find_node("position_ids");

        let logits_output = model
            .find_node("logits")
            .ok_or(GeneratorError::OutputNotFound("logits".to_string()))?;

        // Find inputs and corresponding outputs for key-value cache.
        let batch_size = 1;
        let mut kv_cache = Vec::new();
        for &input_id in model.input_ids() {
            let input_info = model
                .node_info(input_id)
                .ok_or(GeneratorError::InputNotFound(format!(
                    "input ID {}",
                    input_id
                )))?;
            let Some(name) = input_info.name() else {
                continue;
            };

            if !name.starts_with("past_key_values.") {
                continue;
            }

            if !name.ends_with(".key") && !name.ends_with(".value") {
                continue;
            }

            let [n_heads, size] = match input_info.shape().as_deref() {
                Some(&[_, Dimension::Fixed(n_heads), _, Dimension::Fixed(size)]) => [n_heads, size],
                _ => {
                    return Err(GeneratorError::ShapeMismatch(format!("input \"{}\" has unexpected shape. expected (batch, heads, past_seq_len, size) where `heads` and `size` are fixed", name)));
                }
            };

            let cache_type = if name.ends_with(".key") {
                "key"
            } else {
                "value"
            };

            let layer_index_start = "past_key_values.".len();
            let layer_index_str: String = name[layer_index_start..]
                .chars()
                .take_while(|ch| ch.is_ascii_digit())
                .collect();
            let Ok(layer_index) = layer_index_str.parse::<u32>() else {
                continue;
            };

            let output_name = format!("present.{}.{}", layer_index, cache_type);
            let output_id = model
                .find_node(&output_name)
                .ok_or(GeneratorError::OutputNotFound(output_name))?;

            // This value should be configurable.
            let max_seq_len = 512;

            kv_cache.push(KvCache {
                input_id,
                output_id,
                cache: NdTensor::with_capacity(
                    [batch_size, n_heads, max_seq_len, size],
                    2, /* seq dim */
                ),
            });
        }

        Ok(Generator {
            model,
            constant_inputs: Vec::new(),
            constant_prop_inputs: None,
            input_ids: vec![],
            input_ids_input,
            attention_mask_input,
            position_ids_input,
            logits_output,
            kv_cache,
            seq_len: 0,
            sampler: Box::new(ArgMaxSampler {}),
        })
    }

    /// Set the initial sequence of tokens (aka. the prompt) passed to the model
    /// when it is first run.
    pub fn with_prompt(mut self, prompt: &[u32]) -> Self {
        self.input_ids = prompt.to_vec();
        self
    }

    /// Add a constant input which is provided to the model at each iteration.
    ///
    /// A common use case is to pass the outputs of an encoder model to
    /// an auto-regressive decoder.
    pub fn with_constant_input(mut self, input_id: NodeId, value: Input<'a>) -> Self {
        self.constant_prop_inputs = None;
        self.constant_inputs.push((input_id, value.into()));
        self
    }

    /// Set the sampler used to sample the next token ID from the output logits.
    pub fn with_sampler<S: Sampler + 'static>(mut self, sampler: S) -> Self {
        self.sampler = Box::new(sampler);
        self
    }

    /// Run the model and generate the next token.
    fn generate_next_token(&mut self) -> Result<u32, GeneratorError> {
        fn wrap_error<E>(e: E) -> GeneratorError
        where
            E: Into<Box<dyn Error>>,
        {
            GeneratorError::GenerateError(e.into())
        }

        let batch_size = 1;
        let input_ids: NdTensor<i32, 2> = self
            .input_ids
            .iter()
            .map(|id| *id as i32)
            .collect::<Tensor<_>>()
            .into_shape([batch_size, self.input_ids.len()]);

        let attention_mask = NdTensor::full([batch_size, self.input_ids.len()], 1i32);
        let position_ids = NdTensor::from_fn([batch_size, input_ids.len()], |[_batch, pos]| {
            self.seq_len as i32 + pos as i32
        });

        let mut model_inputs: Vec<(NodeId, InputOrOutput)> =
            vec![(self.input_ids_input, input_ids.view().into())];

        if let Some(attention_mask_input) = self.attention_mask_input {
            model_inputs.push((attention_mask_input, attention_mask.view().into()));
        }

        if let Some(position_ids_input) = self.position_ids_input {
            model_inputs.push((position_ids_input, position_ids.view().into()));
        }

        // Propagate constants on the first run.
        if self.constant_prop_inputs.is_none() {
            let inputs = match self.model.partial_run(
                self.constant_inputs.clone(),
                &[self.logits_output],
                None,
            ) {
                Ok(inputs) => inputs,
                Err(err) => {
                    return Err(wrap_error(err));
                }
            };
            self.constant_prop_inputs = Some(inputs);
        }

        if let Some(constants) = self.constant_prop_inputs.as_ref() {
            model_inputs.extend(
                constants
                    .iter()
                    .map(|(node_id, output)| (*node_id, output.as_input().into())),
            );
        }

        // Add key-value cache from previous run. The model takes ownership
        // of the KV-cache tensor during the run so it can efficiently append
        // the entry for the current step, without copying the existing buffer.
        for entry in self.kv_cache.iter_mut() {
            let empty_tensor = NdTensor::zeros([0, 0, 0, 0]);
            let cache = std::mem::replace(&mut entry.cache, empty_tensor);
            model_inputs.push((entry.input_id, cache.into()));
        }

        // Run the model and collect outputs and updated KV cache.
        let model_outputs: Vec<NodeId> = [self.logits_output]
            .into_iter()
            .chain(self.kv_cache.iter().map(|entry| entry.output_id))
            .collect();

        let mut outputs = self
            .model
            .run(model_inputs, &model_outputs, None)
            .map_err(wrap_error)?;

        // Sample output token.
        let logits: NdTensor<f32, 3> = outputs.remove(0).try_into().map_err(wrap_error)?;
        let next_id = self.sampler.sample(logits.slice::<1, _>((0, -1)));

        // Update the key-value cache.
        //
        // The KV cache tensors returned from the model should be the same as
        // the passed in tensors, but extended by one element along the sequence
        // axis.
        for cache_entry in self.kv_cache.iter_mut() {
            cache_entry.cache = outputs.remove(0).try_into().map_err(wrap_error)?;
        }

        // Update the token IDs for the next iteration.
        self.seq_len += self.input_ids.len() as u32;
        self.input_ids = vec![next_id];

        Ok(next_id)
    }
}

/// Output items from a [`Generator`].
pub type GeneratorItem = Result<u32, GeneratorError>;

impl<'a> Iterator for Generator<'a> {
    type Item = Result<u32, GeneratorError>;

    /// Run the model and generate the next output token.
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.generate_next_token())
    }
}

/// Iterator utilities that wrap a [`Generator`] to perform common tasks such
/// as stopping generation when an end-of-text token is encountered.
pub trait GeneratorUtils: Iterator<Item = GeneratorItem> + Sized {
    /// Stop the generator when `eos_token` or an error is encountered.
    fn stop_on_token(self, eos_token: u32) -> impl Iterator<Item = GeneratorItem> {
        self.take_while(move |tok| match tok {
            Ok(tok_id) => *tok_id != eos_token,
            Err(_) => false,
        })
    }

    /// Decode the tokens to text using a tokenizer.
    fn decode(self, tokenizer: &Tokenizer) -> TextGenerator<Self> {
        TextGenerator::wrap(self, tokenizer)
    }

    /// Record timing metrics.
    ///
    /// Metrics such as the number of tokens generated per second will be
    /// available from `metrics` after generation has finished.
    fn profile(self, metrics: &mut Metrics) -> impl Iterator<Item = Self::Item> {
        Profiler::wrap(self, metrics)
    }
}

impl<I: Iterator<Item = GeneratorItem>> GeneratorUtils for I {}

/// Wraps a [`Generator`] to decode the output token IDs from the model into
/// text using a [`Tokenizer`].
pub struct TextGenerator<'a, G: Iterator<Item = GeneratorItem>> {
    generator: G,
    tokenizer: &'a Tokenizer,
}

impl<'a, G> TextGenerator<'a, G>
where
    G: Iterator<Item = GeneratorItem>,
{
    /// Wrap a token generator and decode its outputs using `tokenizer`.
    pub fn wrap(generator: G, tokenizer: &'a Tokenizer) -> TextGenerator<'a, G> {
        TextGenerator {
            generator,
            tokenizer,
        }
    }
}

impl<'a, G: Iterator<Item = GeneratorItem>> Iterator for TextGenerator<'a, G> {
    /// The generated token string, or the error that occurred during generation.
    type Item = Result<String, GeneratorError>;

    /// Run the model repeatedly until it generates a sequence of tokens which
    /// can be decoded into a valid UTF-8 sequence.
    ///
    /// Returns `Some(Ok(text))` if successful, `Some(Err(error))` if an error
    /// occurs during generation or `None` if the end of output has been
    /// reached.
    fn next(&mut self) -> Option<Self::Item> {
        // Buffer that holds model output tokens until it forms a valid UTF-8
        // sequence.
        let mut token_buf = Vec::new();

        for token in self.generator.by_ref() {
            let token = match token {
                Ok(tok) => tok,
                Err(err) => return Some(Err(err)),
            };

            token_buf.push(token as usize);

            let text = self.tokenizer.encoder().decode(&token_buf);
            match text {
                Ok(text) => return Some(Ok(text)),
                Err(TokenizerError::InvalidUtf8) => {
                    // If the current token sequence doesn't correspond to a
                    // complete UTF-8 sequence, add more tokens until it does.
                    continue;
                }
                Err(err) => {
                    return Some(Err(GeneratorError::DecodeError(err)));
                }
            }
        }

        None
    }
}

/// Wraps a [`Generator`] to record timing metrics into a [`Metrics`] struct.
struct Profiler<'a, G: Iterator> {
    generator: G,
    metrics: &'a mut Metrics,
}

impl<'a, G: Iterator> Profiler<'a, G> {
    fn wrap(generator: G, metrics: &'a mut Metrics) -> Profiler<'a, G> {
        Profiler { generator, metrics }
    }
}

impl<'a, G: Iterator> Iterator for Profiler<'a, G> {
    type Item = G::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let start = std::time::Instant::now();
        let item = self.generator.next()?;
        self.metrics.add_step_duration(start.elapsed());
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{GeneratorError, GeneratorUtils};
    use rten_text::tokenizers::patterns::GPT2;
    use rten_text::tokenizers::{Bpe, Tokenizer, WordPiece};

    /// Create a simple WordPiece tokenizer. This is essentially just a lookup
    /// from token ID to string.
    fn create_tokenizer() -> Tokenizer {
        let vocab: HashMap<String, usize> = [("one", 1), ("two", 2), ("three", 3)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let encoder = WordPiece::from_vocab(vocab, Default::default());
        Tokenizer::new(encoder, Default::default())
    }

    /// Create a BPE tokenizer with an empty vocab. This can encode and decode
    /// arbitrary Unicode characters, by using one token per UTF-8 byte.
    fn create_bpe_tokenizer() -> Tokenizer {
        let encoder = Bpe::new(&[], GPT2, None, Default::default()).unwrap();
        Tokenizer::new(encoder, Default::default())
    }

    #[test]
    fn test_text_generator() {
        let tokenizer = create_tokenizer();
        let generator = [1, 2, 3].into_iter().map(Ok);
        let tokens: Vec<_> = generator
            .decode(&tokenizer)
            .map(|tok| tok.map_err(|e| e.to_string()))
            .collect();
        assert_eq!(tokens, ["one", "two", "three"].map(|s| Ok(s.to_string())));
    }

    #[test]
    fn test_text_generator_partial_utf8() {
        let tokenizer = create_bpe_tokenizer();

        // Encode a character which will require multiple token IDs. This means
        // the text decoder will need to loop until accumulated tokens decode
        // to a valid UTF-8 sequence.
        let token_ids = tokenizer.encoder().encode("😊").unwrap();
        assert!(token_ids.len() > 1);
        let generator = token_ids.into_iter().map(|tok_id| Ok(tok_id as u32));

        let tokens: Vec<_> = generator
            .decode(&tokenizer)
            .map(|tok| tok.map_err(|e| e.to_string()))
            .collect();

        assert_eq!(tokens, ["😊"].map(|s| Ok(s.to_string())));
    }

    #[test]
    fn test_text_generator_generate_error() {
        let tokenizer = create_tokenizer();
        let generator = [
            Ok(1),
            Err(GeneratorError::GenerateError("oh no".to_string().into())),
            Ok(3),
        ]
        .into_iter();

        let tokens: Vec<_> = generator
            .decode(&tokenizer)
            .map(|tok| tok.map_err(|e| e.to_string()))
            .collect();

        assert_eq!(
            tokens,
            [
                Ok("one".to_string()),
                Err("generation error: oh no".to_string()),
                Ok("three".to_string())
            ]
        );
    }

    #[test]
    fn test_text_generator_decode_error() {
        let tokenizer = create_tokenizer();
        let generator = [1, 5, 3].into_iter().map(Ok);

        let tokens: Vec<_> = generator
            .decode(&tokenizer)
            .map(|tok| tok.map_err(|e| e.to_string()))
            .collect();

        assert_eq!(
            tokens,
            [
                Ok("one".to_string()),
                Err("decode error: unknown token id 5".to_string()),
                Ok("three".to_string())
            ]
        );
    }
}
