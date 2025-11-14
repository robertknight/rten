use std::error::Error;
use std::io;
use std::io::prelude::*;

use argh::FromArgs;
use rten::Model;
use rten_generate::filter::Chain;
use rten_generate::sampler::Multinomial;
use rten_generate::{Generator, GeneratorUtils};
use rten_text::{Tokenizer, TokenizerError};

/// Chat with a large language model.
#[derive(FromArgs)]
struct Args {
    /// input model
    #[argh(positional)]
    model: String,

    /// tokenizer.json file
    #[argh(positional)]
    tokenizer_config: String,

    /// generation temperature (must be >= 0, default: 0.7). Smaller values make output less "creative" by concentrating the probability distribution more. A value of 0.0 causes sampling to be greedy.
    #[argh(option, short = 't', default = "0.7")]
    temperature: f32,
}

enum MessageChunk<'a> {
    Text(&'a str),
    Token(u32),
}

/// Encode a message consisting of a mix of text and special token IDs into a
/// sequence of token IDs.
///
/// Special tokens need to be passed as IDs because `Tokenizer::encode` will not
/// generate them (eg. it would treat a string such as "<|endoftext|>" as
/// ordinary text).
fn encode_message(
    tokenizer: &Tokenizer,
    chunks: &[MessageChunk],
) -> Result<Vec<u32>, TokenizerError> {
    let mut token_ids = Vec::new();
    for chunk in chunks {
        match chunk {
            MessageChunk::Token(tok_id) => token_ids.push(*tok_id),
            MessageChunk::Text(text) => {
                let encoded = tokenizer.encode(*text, None)?;
                token_ids.extend(encoded.token_ids());
            }
        }
    }
    Ok(token_ids)
}

/// Chatbot using Qwen 2 [2].
///
/// This example also works with some other models which use the same prompt
/// format as Qwen 2, such as SmolLM v3 [3].
///
/// To obtain the model from Hugging Face, use Optimum [1].
/// The model is available in various sizes. The larger models are smarter
/// but slower. To export the smallest 0.5B model, use:
///
/// ```sh
/// optimum-cli export onnx --model Qwen/Qwen2-0.5B-Instruct qwen2-0.5b
/// ```
///
/// The model can optionally be quantized using:
///
/// ```
/// python tools/ort-quantize.py nbits qwen2-0.5/model.onnx
/// ```
///
/// When using quantization, this will generate `model.quant.onnx` in the same
/// directory, which should be used in subsequent steps.
///
/// Then run the model and enter a prompt:
///
/// ```sh
/// cargo run --release --bin qwen2_chat qwen2-0.5b/model.onnx qwen2-0.5b/tokenizer.json
/// ```
///
/// [1] https://huggingface.co/docs/optimum/index
/// [2] https://github.com/QwenLM/Qwen2
/// [3] https://huggingface.co/HuggingFaceTB/SmolLM3-3B
fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Args = argh::from_env();
    args.temperature = args.temperature.max(0.);

    // `load_mmap` reduces model load/free time and process memory usage, at the
    // cost of making the first execution slower.
    let model = unsafe { Model::load_mmap(args.model) }?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;

    let im_start_token = tokenizer.get_token_id("<|im_start|>")?;
    let im_end_token = tokenizer.get_token_id("<|im_end|>")?;

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    // End of turn token used by some models that are supported by this example.
    // See the `eos_token_id` field in the model's `generation_config.json`.
    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }

    // From `chat_template` in tokenizer_config.json.
    let prompt_tokens = encode_message(
        &tokenizer,
        &[
            MessageChunk::Token(im_start_token),
            MessageChunk::Text("system\nYou are a helpful assistant."),
            MessageChunk::Token(im_end_token),
        ],
    )?;

    // From Qwen2's `generation_config.json`
    let top_k = 20;

    let mut generator = Generator::from_model(&model)?
        .with_prompt(&prompt_tokens)
        .with_logits_filter(Chain::new().top_k(top_k).temperature(args.temperature))
        .with_sampler(Multinomial::new());

    loop {
        print!("> ");
        let _ = std::io::stdout().flush();

        let mut user_input = String::new();
        let n_read = io::stdin().read_line(&mut user_input)?;
        if n_read == 0 {
            // EOF
            break;
        }

        // From `chat_template` in tokenizer_config.json.
        let token_ids = encode_message(
            &tokenizer,
            &[
                MessageChunk::Token(im_start_token),
                MessageChunk::Text("user\n"),
                MessageChunk::Text(&user_input),
                MessageChunk::Token(im_end_token),
                MessageChunk::Text("\n"),
                MessageChunk::Token(im_start_token),
                MessageChunk::Text("assistant\n"),
            ],
        )?;

        generator.append_prompt(&token_ids);

        let decoder = generator
            .by_ref()
            .stop_on_tokens(&end_of_turn_tokens)
            .decode(&tokenizer);
        for token in decoder {
            let token = token?;
            print!("{}", token);
            let _ = std::io::stdout().flush();
        }

        println!();
    }

    Ok(())
}
