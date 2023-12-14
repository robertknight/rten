use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::io::prelude::*;

use wasnn::{Model, NodeId, Operators};
use wasnn_tensor::prelude::*;
use wasnn_tensor::{NdTensor, Tensor};
use wasnn_text::tokenizers::patterns::GPT2 as GPT2_REGEX;
use wasnn_text::tokenizers::{ByteLevelBpe, Tokenizer};

struct Args {
    model: String,
    merges: String,
    prompt: String,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Generate text using a prompt.

Usage: {bin_name} <model> <bpe_merges> <prompt> [options]

Args:

  <model>       - Input GPT-2 model
  <bpe_merges>  - merges.txt file containing token merge rules
  <prompt>      - Text generation prompt
",
                    bin_name = parser.bin_name().unwrap_or("gpt2")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let merges = values.pop_front().ok_or("missing `bpe_merges` arg")?;
    let prompt = values.make_contiguous().join(" ");

    let args = Args {
        model,
        merges,
        prompt,
    };

    Ok(args)
}

/// Generate a token sequence using an auto-regressive language model.
struct LmSequenceGenerator<'a> {
    model: &'a Model,

    /// Input token IDs for the next run of the model.
    input_ids: Vec<u32>,

    node_ids: LmSequenceGenConfig,
}

pub struct LmSequenceGenConfig {
    /// Model input for token IDs.
    pub input_ids: NodeId,

    /// Model input for attention mask.
    pub attention_mask: NodeId,

    /// Model output for logits.
    pub logits: NodeId,
}

impl<'a> LmSequenceGenerator<'a> {
    /// Run the model and generate the next token.
    fn generate_next_token(&mut self) -> Result<u32, Box<dyn Error>> {
        let input_ids: Tensor<i32> = self
            .input_ids
            .iter()
            .map(|id| *id as i32)
            .collect::<Tensor<_>>()
            .into_shape(&[1, self.input_ids.len()]);
        let attention_mask: Tensor<i32> = Tensor::full(&[1, self.input_ids.len()], 1);

        let [logits] = self.model.run_n(
            &[
                (self.node_ids.input_ids, input_ids.view().into()),
                (self.node_ids.attention_mask, attention_mask.view().into()),
            ],
            [self.node_ids.logits],
            None,
        )?;
        let logits: NdTensor<f32, 3> = logits.try_into()?;
        let next_ids = logits.arg_max(-1, false /* keep_dims */)?;
        let next_id = next_ids
            .slice((0, -1))
            .item()
            .map(|it| *it as u32)
            .expect("expected scalar");

        self.input_ids.push(next_id);

        Ok(next_id)
    }
}

impl<'a> Iterator for LmSequenceGenerator<'a> {
    type Item = Result<u32, Box<dyn Error>>;

    /// Run the model and generate the next output token.
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.generate_next_token())
    }
}

/// Generate a sequence using an auto-regressive model.
///
/// Returns an iterator that runs the model on each call to [Iterator::next]
/// and yields a result containing the next token ID or an error.
///
/// The model is expected to have the following inputs and outputs:
///
///   - (input) input_ids - (batch, sequence) tensor of token IDs
///   - (input) attention_mask - (batch, sequence) tensor of booleans
///   - (output) logits - (batch, sequence, vocab) tensor of next token probabilities
///
/// At each step, the input IDs are fed into the model and the output logits
/// corresponding to the final input are sampled to get the next token ID.
fn generate_sequence<'a>(
    model: &'a Model,
    node_ids: LmSequenceGenConfig,
    input_ids: &[u32],
) -> LmSequenceGenerator<'a> {
    LmSequenceGenerator {
        model,
        node_ids,
        input_ids: input_ids.iter().copied().collect(),
    }
}

/// Generates text using GPT-2 [1] and a prompt.
///
/// To obtain the model from Hugging Face, use Optimum [2], then convert it:
///
/// ```sh
/// optimum-cli export onnx --model gpt2 gpt2_onnx/
/// tools/convert-onnx.py gpt2_onnx/decoder_model.onnx gpt2_onnx/decoder_model.model
/// ```
///
/// Run the converted model with a prompt:
///
/// ```sh
/// cargo run -r --example gpt2 gpt2_onnx/decoder_model.model gp2_onnx/merges.txt <prompt>
/// ```
///
/// Where `<prompt>` is the start of a sentence that the model should complete.
///
/// [1] https://openai.com/research/better-language-models
/// [2] https://huggingface.co/docs/optimum/index
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model_bytes = fs::read(args.model)?;
    let model = Model::load(&model_bytes)?;

    let node_ids = LmSequenceGenConfig {
        input_ids: model.node_id("input_ids")?,
        attention_mask: model.node_id("attention_mask")?,
        logits: model.node_id("logits")?,
    };

    let merges_file = fs::read_to_string(&args.merges)?;
    let merges: Vec<_> = merges_file.lines().collect();
    let encoder = ByteLevelBpe::new(&merges, GPT2_REGEX)?;
    let tokenizer = Tokenizer::new(encoder, Default::default());

    let prompt = args.prompt.as_str();
    let encoded_prompt = tokenizer.encode(prompt.into(), Default::default())?;
    let token_ids: Vec<u32> = encoded_prompt
        .token_ids()
        .iter()
        .map(|id| *id as u32)
        .collect();

    // The output starts with the user's prompt.
    print!("{}", prompt);

    // Buffer that holds model output tokens until it forms a valid UTF-8
    // sequence.
    let mut token_buf = Vec::new();

    for token in generate_sequence(&model, node_ids, &token_ids).take(30) {
        let token = token?;
        token_buf.push(token as usize);

        let token_strings = tokenizer.encoder().get_tokens(&token_buf);
        if let Ok(strings) = token_strings {
            for s in strings {
                print!("{}", s);
            }
            let _ = std::io::stdout().flush();
            token_buf.clear();
        }
    }

    Ok(())
}
