//! Tools to run the generation loop for an auto-regressive model.

use std::error::Error;
use std::fmt;
use std::ops::Range;

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

/// Specifies a pattern for the name of a key-value cache input or output.
///
/// These inputs are expected to have the form `{prefix}{layer_number}{suffix}`,
/// with one input and output per layer for the key cache and the value cache.
pub struct KVCachePattern<'a> {
    pub prefix: &'a str,
    pub suffix: &'a str,
}

impl<'a> From<(&'a str, &'a str)> for KVCachePattern<'a> {
    /// Construct a [`KVCachePattern`] from a `(prefix, suffix)` tuple.
    fn from(value: (&'a str, &'a str)) -> Self {
        let (prefix, suffix) = value;
        KVCachePattern { prefix, suffix }
    }
}

/// Specifies the names of model inputs and outputs.
///
/// The [`Default`] impl for this struct returns an instance whose names
/// follow the configuration of Hugging Face's Optimum tool.
///
/// Any inputs that are not present in the model are ignored.
pub struct ModelInputsConfig<'a> {
    /// Model input that contains the token IDs of the prompt and output
    /// generated so far.
    pub input_ids: &'a str,

    /// Model output that contains logits.
    pub logits: &'a str,

    /// Model input that contains an attention mask.
    pub attention_mask: &'a str,

    /// Model input that contains position IDs for each position.
    pub position_ids: &'a str,

    /// Pattern for key cache inputs.
    pub key_cache: KVCachePattern<'a>,

    /// Pattern for key cache outputs.
    pub key_cache_output: KVCachePattern<'a>,

    /// Pattern for value cache inputs.
    pub value_cache: KVCachePattern<'a>,

    /// Pattern for value cache outputs.
    pub value_cache_output: KVCachePattern<'a>,
}

/// Contains essential configuration needed for a `Generator` to execute a
/// model, such as the roles of different inputs and outputs.
pub struct GeneratorConfig<'a> {
    /// Specifies names and roles of model inputs and outputs.
    pub model_inputs: ModelInputsConfig<'a>,
}

impl<'a> Default for ModelInputsConfig<'a> {
    /// Return default model input names.
    ///
    /// These are based on [Hugging Face's
    /// Optimum](https://huggingface.co/docs/optimum/en/index) model exporter.
    fn default() -> Self {
        ModelInputsConfig {
            input_ids: "input_ids",
            logits: "logits",
            attention_mask: "attention_mask",
            position_ids: "position_ids",
            key_cache: ("past_key_values.", ".key").into(),
            key_cache_output: ("present.", ".key").into(),
            value_cache: ("past_key_values.", ".value").into(),
            value_cache_output: ("present.", ".value").into(),
        }
    }
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

    /// Additional constant model inputs (eg. encoder outputs) passed to the
    /// model at each step.
    constant_inputs: Vec<(NodeId, InputOrOutput<'a>)>,

    /// Additional model inputs computed using constant propagation. This
    /// effectively caches parts of the graph that don't change in each
    /// generation step. This is `None` if the cache is out of date.
    constant_prop_inputs: Option<Vec<(NodeId, Output)>>,

    /// Additional varying model inputs computed and passed to the model at
    /// each step. The functions receive `(batch_size, sequence_positions)` as inputs.
    #[allow(clippy::type_complexity)]
    varying_inputs: Vec<(NodeId, &'a dyn Fn(usize, Range<usize>) -> InputOrOutput<'a>)>,

    /// Input token IDs for the next run of the model.
    input_ids: Vec<u32>,

    // Input node IDs
    input_ids_input: NodeId,

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
    /// This function assumes default names for model inputs and outputs
    /// based on the conventions of Hugging Face's Optimum exporter. These
    /// can be customized using [`from_model_config`](Self::from_model_config).
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
        let config = GeneratorConfig {
            model_inputs: ModelInputsConfig::default(),
        };
        Self::from_model_config(model, config)
    }

    /// Create a generator that iteratively produces tokens using a model.
    ///
    /// This is a variant of [`from_model`](Self::from_model) that allows
    /// specifying custom names for model inputs.
    pub fn from_model_config(
        model: &'a Model,
        config: GeneratorConfig,
    ) -> Result<Generator<'a>, GeneratorError> {
        let model_inputs = &config.model_inputs;

        let input_ids_input =
            model
                .find_node(model_inputs.input_ids)
                .ok_or(GeneratorError::InputNotFound(
                    model_inputs.input_ids.to_string(),
                ))?;

        let logits_output =
            model
                .find_node(model_inputs.logits)
                .ok_or(GeneratorError::OutputNotFound(
                    model_inputs.logits.to_string(),
                ))?;

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

            let is_key_cache = name.starts_with(model_inputs.key_cache.prefix)
                && name.ends_with(model_inputs.key_cache.suffix);
            let is_value_cache = name.starts_with(model_inputs.value_cache.prefix)
                && name.ends_with(model_inputs.value_cache.suffix);

            if !is_key_cache && !is_value_cache {
                continue;
            }

            let [n_heads, size] = match input_info.shape().as_deref() {
                Some(&[_, Dimension::Fixed(n_heads), _, Dimension::Fixed(size)]) => [n_heads, size],
                _ => {
                    return Err(GeneratorError::ShapeMismatch(format!("input \"{}\" has unexpected shape. expected (batch, heads, past_seq_len, size) where `heads` and `size` are fixed", name)));
                }
            };

            let prefix = if is_key_cache {
                model_inputs.key_cache.prefix
            } else {
                model_inputs.value_cache.prefix
            };

            let layer_index_start = prefix.len();
            let layer_index_str: String = name[layer_index_start..]
                .chars()
                .take_while(|ch| ch.is_ascii_digit())
                .collect();
            let Ok(layer_index) = layer_index_str.parse::<u32>() else {
                continue;
            };

            let (output_prefix, output_suffix) = if is_key_cache {
                (
                    model_inputs.key_cache_output.prefix,
                    model_inputs.key_cache_output.suffix,
                )
            } else {
                (
                    model_inputs.value_cache_output.prefix,
                    model_inputs.value_cache_output.suffix,
                )
            };

            let output_name = format!("{}{}{}", output_prefix, layer_index, output_suffix);
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

        let mut generator = Generator {
            model,
            constant_inputs: Vec::new(),
            varying_inputs: Vec::new(),

            // Constant propagation is performed as a graph optimization when
            // the model is loaded, so we only need to re-do it if additional
            // constant inputs are added.
            constant_prop_inputs: Some(Vec::new()),

            input_ids: vec![],
            input_ids_input,
            logits_output,
            kv_cache,
            seq_len: 0,
            sampler: Box::new(ArgMaxSampler {}),
        };

        let attention_mask_input = model.find_node(model_inputs.attention_mask);
        if let Some(attention_mask_input) = attention_mask_input {
            generator = generator
                .with_varying_input(attention_mask_input, &|batch_size, positions| {
                    NdTensor::full([batch_size, positions.len()], 1i32).into()
                });
        }

        let position_ids_input = model.find_node(model_inputs.position_ids);
        if let Some(position_ids_input) = position_ids_input {
            generator =
                generator.with_varying_input(position_ids_input, &|batch_size, positions| {
                    NdTensor::from_fn([batch_size, positions.len()], |[_batch, pos]| {
                        (positions.start + pos) as i32
                    })
                    .into()
                });
        }

        Ok(generator)
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

    /// Add an input which varies with the sequence position.
    ///
    /// `value_fn` receives `(batch_size, sequence_positions)` as input and
    /// computes the value for the input at the given positions.
    ///
    /// A common use case is to pass position embeddings, if they are not
    /// computed internally by the model.
    pub fn with_varying_input<F: Fn(usize, Range<usize>) -> InputOrOutput<'a>>(
        mut self,
        input_id: NodeId,
        value_fn: &'a F,
    ) -> Self {
        self.varying_inputs.push((input_id, value_fn));
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

        let seq_range = (self.seq_len as usize)..(self.seq_len as usize + self.input_ids.len());

        let mut model_inputs: Vec<(NodeId, InputOrOutput)> =
            vec![(self.input_ids_input, input_ids.view().into())];

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

        if !self.varying_inputs.is_empty() {
            model_inputs.extend(
                self.varying_inputs
                    .iter()
                    .map(|(node_id, value_fn)| (*node_id, value_fn(batch_size, seq_range.clone()))),
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
