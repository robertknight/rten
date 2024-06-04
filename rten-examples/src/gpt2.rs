use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::io::prelude::*;

use rten::Model;
use rten_examples::generator::Generator;
use rten_text::tokenizers::Tokenizer;

struct Args {
    model: String,
    tokenizer_config: String,
    prompt: String,
    output_length: usize,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();
    let mut output_length = 30;

    while let Some(arg) = parser.next()? {
        match arg {
            Short('l') | Long("length") => {
                output_length = parser.value()?.parse()?;
            }
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Generate text using a prompt.

Usage: {bin_name} [options] <model> <tokenizer> <prompt>

Args:

  <model>       - Input GPT-2 model
  <tokenizer>   - `tokenizer.json` file
  <prompt>      - Text generation prompt

Options:

 -l, --length   - Set max output length (in tokens)
",
                    bin_name = parser.bin_name().unwrap_or("gpt2")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let tokenizer_config = values.pop_front().ok_or("missing `tokenizer` arg")?;
    let prompt = values.make_contiguous().join(" ");

    let args = Args {
        model,
        tokenizer_config,
        prompt,
        output_length,
    };

    Ok(args)
}

/// Generates text using GPT-2 [1] and a prompt.
///
/// To obtain the model from Hugging Face, use Optimum [2], then convert it:
///
/// ```sh
/// optimum-cli export onnx --model gpt2 gpt2_onnx/
/// rten-convert gpt2_onnx/model.onnx
/// ```
///
/// Run the converted model with a prompt:
///
/// ```sh
/// cargo run --release --bin gpt2 gpt2_onnx/model.rten gp2_onnx/tokenizer.json <prompt>
/// ```
///
/// Where `<prompt>` is the start of a sentence that the model should complete.
///
/// [1] https://openai.com/research/better-language-models
/// [2] https://huggingface.co/docs/optimum/index
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.model)?;

    let tokenizer_config = fs::read_to_string(&args.tokenizer_config)?;
    let tokenizer = Tokenizer::from_json(&tokenizer_config)?;

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

    let generator = Generator::from_model(&model)?.with_prompt(&token_ids);
    for token in generator.take(args.output_length) {
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
