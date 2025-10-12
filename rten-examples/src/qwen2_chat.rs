use std::collections::VecDeque;
use std::error::Error;
use std::io;
use std::io::prelude::*;

use rten::Model;
use rten_generate::sampler::TopKSampler;
use rten_generate::{Generator, GeneratorUtils};
use rten_text::{Tokenizer, TokenizerError};

struct Args {
    model: String,
    tokenizer_config: String,
    temperature: f32,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();

    // Default from Qwen2's `generate_config.json`.
    let mut temperature: f32 = 0.7;

    while let Some(arg) = parser.next()? {
        match arg {
            Short('t') | Long("temperature") => {
                temperature = parser.value()?.parse()?;
                temperature = temperature.max(0.);
            }
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Chat with a large language model.

Usage: {bin_name} [options] <model> <tokenizer>

Args:

  <model>       - Input model
  <tokenizer>   - `tokenizer.json` file

Options:

 -t, --temperature TEMP

    Set the generation temperature. Must be >= 0. Smaller values make the
    output less \"creative\" by concentrating the output probability distribution
    more. A value of 0.0 causes sampling to be greedy.
",
                    bin_name = parser.bin_name().unwrap_or("chat")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let tokenizer_config = values.pop_front().ok_or("missing `tokenizer` arg")?;

    let args = Args {
        model,
        tokenizer_config,
        temperature,
    };

    Ok(args)
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
/// To obtain the model from Hugging Face, use Optimum [1].
/// The model is available in various sizes. The larger models are smarter
/// but slower. To export the smallest 0.5B model, use:
///
/// ```sh
/// optimum-cli export onnx --model Qwen/Qwen2-0.5B-Instruct qwen2-0.5b
/// ```
///
/// Then run the model and enter a prompt:
///
/// ```sh
/// cargo run --release --bin qwen2_chat qwen2-0.5b/model.onnx qwen2-0.5b/tokenizer.json
/// ```
///
/// For better output, but generated more slowly, use the "1.5b" model.
///
/// [1] https://huggingface.co/docs/optimum/index
/// [2] https://github.com/QwenLM/Qwen2
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    // `load_mmap` reduces model load/free time and process memory usage, at the
    // cost of making the first execution slower.
    let model = unsafe { Model::load_mmap(args.model) }?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;

    let im_start_token = tokenizer.get_token_id("<|im_start|>")?;
    let im_end_token = tokenizer.get_token_id("<|im_end|>")?;
    let end_of_text_token = tokenizer.get_token_id("<|endoftext|>")?;

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
        .with_sampler(TopKSampler::new(top_k, args.temperature));

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
            // See `eos_token_id` in `generation_config.json`
            .stop_on_tokens([im_end_token, end_of_text_token])
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
