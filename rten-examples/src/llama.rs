use std::error::Error;
use std::io;
use std::io::prelude::*;

use argh::FromArgs;
use rten::ModelOptions;
use rten_generate::filter::Chain;
use rten_generate::metrics::Metrics;
use rten_generate::sampler::Multinomial;
use rten_generate::{Generator, GeneratorUtils};
use rten_text::{TokenId, Tokenizer, TokenizerError};

/// Chat with the Llama 3 language model.
#[derive(FromArgs)]
struct Args {
    /// input model
    #[argh(positional)]
    model: String,

    /// tokenizer.json file
    #[argh(positional)]
    tokenizer_config: String,

    /// generation temperature (must be >= 0, default: 0.6). Smaller values make output less "creative" by concentrating the probability distribution more. A value of 0.0 causes sampling to be greedy.
    #[argh(option, short = 't', default = "0.6")]
    temperature: f32,

    /// additions to the initial system prompt.
    ///
    /// Use this for instructions on how the chatbot should handle each prompt
    /// (eg. "keep responses under 10 words").
    #[argh(option, short = 's')]
    system: Option<String>,

    /// user prompt.
    ///
    /// If specified, the LLM will respond to this prompt and then exit.
    #[argh(option, short = 'p')]
    prompt: Option<String>,
}

/// Chatbot using Llama 3 [2].
///
/// To obtain the model from Hugging Face, use Optimum [1]. The model is
/// available in several sizes (1B, 3B, 8B...). To export the 3B model, use:
///
/// ```sh
/// # Export to ONNX
/// optimum-cli export onnx --model meta-llama/Llama-3.2-3B-Instruct llama3-3b
///
/// # Quantize weights to 4-bits.
/// python tools/ort-quantize.py nbits llama3-3b/model.onnx
/// ```
///
/// Note that the export will take some time.
///
/// Then run the model and enter a prompt:
///
/// ```sh
/// cargo run --release --bin llama llama3-3b/model.quant.onnx llama3-3b/tokenizer.json
/// ```
///
/// [1] https://huggingface.co/docs/optimum/index
/// [2] https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Args = argh::from_env();
    args.temperature = args.temperature.max(0.);

    let mut model_opts = ModelOptions::with_all_ops();

    // Enable shape inference. This enables additional optimizations that are
    // useful for this model. See https://github.com/robertknight/rten/issues/1091.
    model_opts.enable_shape_inference(true);

    // `load_mmap` reduces model load/free time and process memory usage, at the
    // cost of making the first execution slower.
    let model = unsafe { model_opts.load_mmap(args.model) }?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;
    let special = SpecialTokens::new(&tokenizer)?;

    // System prompt based on the `chat_template.jinja` file.
    let mut prompt = PromptBuilder::new(&tokenizer, special)
        .append_id(special.begin_of_text)
        .append("system", Some("You are a helpful assistant."));
    if let Some(sys) = args.system.as_deref() {
        prompt = prompt.append("system", Some(sys));
    }
    let system_prompt = prompt.encode()?;

    // From `generation_config.json`.
    let top_p = 0.9;
    let top_k = 40;

    let mut generator = Generator::from_model(&model)?
        .with_prompt(&system_prompt)
        .with_logits_filter(
            Chain::new()
                .top_k(top_k)
                .temperature(args.temperature)
                .top_p(top_p),
        )
        .with_sampler(Multinomial::new());
    let mut metrics = Metrics::new();

    loop {
        let mut user_input = String::new();

        if let Some(prompt) = &args.prompt {
            user_input = prompt.to_string();
        } else {
            print!("> ");
            let _ = std::io::stdout().flush();

            let n_read = io::stdin().read_line(&mut user_input)?;
            if n_read == 0 {
                // EOF
                break;
            }

            // If the user presses Enter without typing a message, enter
            // multi-line mode.
            if user_input.trim() == "" {
                println!(
                    ">> Entering multi-line mode. Press Ctrl-D on an empty line to end message."
                );
                user_input.clear();
                loop {
                    let n_read = io::stdin().read_line(&mut user_input)?;
                    if n_read == 0 {
                        // EOF
                        break;
                    }
                }
                if user_input.trim().is_empty() {
                    // Empty message returns to single-line mode.
                    continue;
                }
            }
        }

        // Turn prompt from chat_template.jinja.
        let turn_prompt = PromptBuilder::new(&tokenizer, special)
            .append("user", Some(&user_input))
            .append("assistant", None)
            .encode()?;
        generator.append_prompt(&turn_prompt);

        let decoder = generator
            .by_ref()
            // See `eos_token_id` in `generation_config.json`
            .stop_on_tokens([special.end_of_text])
            .profile(&mut metrics)
            .decode(&tokenizer);
        for token in decoder {
            let token = token?;
            print!("{}", token);
            let _ = std::io::stdout().flush();
        }

        if args.prompt.is_some() {
            break;
        }

        println!();
    }

    if let Some(mean_dur) = metrics.mean_duration()
        && let Some(tps) = metrics.tokens_per_second()
    {
        println!(
            "\n\nmetrics: tokens {}, {:.0}ms/token, {:.1} tok/s",
            metrics.token_count(),
            mean_dur,
            tps,
        );
    }

    Ok(())
}

#[derive(Copy, Clone)]
struct SpecialTokens {
    /// Token for start of a conversation.
    begin_of_text: TokenId,

    /// Token for start of a role ID.
    start_header: TokenId,

    /// Token for end of a role ID.
    end_header: TokenId,

    /// Token for the end of a turn.
    end_of_text: TokenId,
}

impl SpecialTokens {
    fn new(tokenizer: &Tokenizer) -> Result<SpecialTokens, TokenizerError> {
        let begin_of_text = tokenizer.get_token_id("<|begin_of_text|>")?;
        let start_header = tokenizer.get_token_id("<|start_header_id|>")?;
        let end_header = tokenizer.get_token_id("<|end_header_id|>")?;
        let end_of_text = tokenizer.get_token_id("<|eot_id|>")?;
        Ok(Self {
            begin_of_text,
            start_header,
            end_header,
            end_of_text,
        })
    }
}

enum MessageChunk {
    Text(String),
    Token(TokenId),
}

struct PromptBuilder<'a> {
    tokenizer: &'a Tokenizer,
    special: SpecialTokens,
    chunks: Vec<MessageChunk>,
}

impl<'a> PromptBuilder<'a> {
    fn new(tokenizer: &'a Tokenizer, special: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special,
            chunks: Vec::new(),
        }
    }

    /// Add a token to the message by ID.
    fn append_id(mut self, id: TokenId) -> Self {
        self.chunks.push(MessageChunk::Token(id));
        self
    }

    /// Add a message to the prompt.
    ///
    /// If `content` is None, the turn is left open for the LLM to complete.
    fn append(mut self, role: &str, content: Option<&str>) -> Self {
        self.chunks.extend([
            MessageChunk::Token(self.special.start_header),
            MessageChunk::Text(role.to_string()),
            MessageChunk::Token(self.special.end_header),
            MessageChunk::Text("\n\n".to_string()),
        ]);
        if let Some(content) = content {
            self.chunks.push(MessageChunk::Text(content.to_string()));
            self.chunks
                .push(MessageChunk::Token(self.special.end_of_text));
        }
        self
    }

    /// Encode a message into token IDs.
    fn encode(self) -> Result<Vec<TokenId>, TokenizerError> {
        let mut token_ids = Vec::new();
        for chunk in &self.chunks {
            match chunk {
                MessageChunk::Token(tok_id) => token_ids.push(*tok_id),
                MessageChunk::Text(text) => {
                    let encoded = self.tokenizer.encode(text, None)?;
                    token_ids.extend(encoded.token_ids());
                }
            }
        }
        Ok(token_ids)
    }
}
