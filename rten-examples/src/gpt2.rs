use std::error::Error;
use std::io::prelude::*;

use argh::FromArgs;
use rten::Model;
use rten_generate::filter::Chain;
use rten_generate::metrics::Metrics;
use rten_generate::sampler::Multinomial;
use rten_generate::{Generator, GeneratorUtils};
use rten_text::Tokenizer;

/// Generate text using a prompt.
#[derive(FromArgs)]
struct Args {
    /// input GPT-2 model
    #[argh(positional)]
    model: String,

    /// tokenizer.json file
    #[argh(positional)]
    tokenizer_config: String,

    /// text generation prompt
    #[argh(positional)]
    prompt: String,

    /// max output length (in tokens)
    #[argh(option, short = 'l', default = "30")]
    length: usize,

    /// sample from top K tokens at each step
    #[argh(option, short = 'k', default = "50")]
    top_k: usize,
}

/// Generates text using GPT-2 [1] and a prompt.
///
/// First, export the model using Optimum [2]:
///
/// ```sh
/// optimum-cli export onnx --model gpt2 gpt2_onnx/
/// ```
///
/// Then run the model with a prompt:
///
/// ```sh
/// cargo run --release --bin gpt2 gpt2_onnx/model.onnx gp2_onnx/tokenizer.json <prompt>
/// ```
///
/// Where `<prompt>` is the start of a sentence that the model should complete.
///
/// [1] https://openai.com/research/better-language-models
/// [2] https://huggingface.co/docs/optimum/index
fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();
    let model = Model::load_file(args.model)?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;

    let prompt = args.prompt.as_str();
    let encoded_prompt = tokenizer.encode(prompt, None)?;

    // The output starts with the user's prompt.
    print!("{}", prompt);

    let mut metrics = Metrics::new();
    let temperature = 1.0;
    let generator = Generator::from_model(&model)?
        .with_prompt(encoded_prompt.token_ids())
        .with_logits_filter(Chain::new().top_k(args.top_k).temperature(temperature))
        .with_sampler(Multinomial::new())
        .take(args.length)
        .profile(&mut metrics)
        .decode(&tokenizer);

    for token in generator {
        let token = token?;
        print!("{}", token);
        let _ = std::io::stdout().flush();
    }
    println!();

    println!(
        "Metrics: {:.2}s total, {:.2}s warmup, {:.2} tokens/sec, {:.2} ms/token.",
        metrics.total_duration().as_secs_f32(),
        metrics
            .warmup_duration()
            .map(|dur| dur.as_secs_f32())
            .unwrap_or(0.),
        metrics.tokens_per_second().unwrap_or(0.),
        metrics.mean_duration().unwrap_or(0.)
    );

    Ok(())
}
