use std::error::Error;
use std::io::prelude::*;

use argh::FromArgs;
use rten::{FloatOperators, Model};
use rten_generate::metrics::Metrics;
use rten_generate::{Generator, GeneratorUtils};
use rten_imageio::read_image;
use rten_imageproc::normalize_image;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use rten_text::Tokenizer;

/// Extract text from academic PDFs as Markdown.
#[derive(FromArgs)]
struct Args {
    /// image encoder model
    #[argh(positional)]
    encoder_model: String,

    /// text decoder model
    #[argh(positional)]
    decoder_model: String,

    /// tokenizer.json file
    #[argh(positional)]
    tokenizer_config: String,

    /// image path
    #[argh(positional)]
    image_path: String,
}

/// Extract text as markdown from academic PDFs using Nougat.
///
/// First use Hugging Face's Optimum tool to download and export the models to
/// ONNX:
///
/// ```
/// optimum-cli export onnx --model facebook/nougat-base nougat-base
/// ```
///
/// Run the model, specifying the image to recognize:
///
/// ```sh
/// cargo run --release --bin nougat nougat-base/encoder_model.onnx nougat-base/decoder_model_merged.onnx nougat-base/tokenizer.json <image>
/// ```
///
/// [^1]: https://arxiv.org/abs/2308.13418
fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();

    // Load the models. For faster load times/reduced memory, consider
    // converting the ONNX models either to external data or .rten format and
    // using `load_mmap`.
    let encoder = Model::load_file(args.encoder_model)?;
    let decoder = Model::load_file(args.decoder_model)?;

    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;
    let mut image = read_image(args.image_path)?.into_dyn();
    image.insert_axis(0); // Add batch dim

    // Values taken from `preprocessor_config.json`.
    let img_size = [896, 672];
    let mean = [0.485, 0.456, 0.406];
    let std_dev = [0.229, 0.224, 0.225];

    let mut image: NdTensor<_, 4> = image.resize_image(img_size)?.try_into()?;
    normalize_image(image.slice_mut(0), mean, std_dev);

    let encoded_image: NdTensor<f32, 3> = encoder.run_one(image.view().into(), None)?.try_into()?;

    // `bos_token_id` from `generation_config.json`.
    // token.
    let decoder_start_token = 0;
    let eos_token = 2;

    let max_tokens = 2_000;

    let prompt = [decoder_start_token];
    let mut metrics = Metrics::new();
    let generator = Generator::from_model(&decoder)?
        .with_prompt(&prompt)
        .with_constant_input(
            decoder.node_id("encoder_hidden_states")?,
            encoded_image.view().into(),
        )
        .stop_on_tokens([eos_token])
        .profile(&mut metrics)
        .take(max_tokens)
        .decode(&tokenizer);

    for token in generator {
        let token = token?;

        print!("{}", token);
        let _ = std::io::stdout().flush();
    }
    println!("\n");
    println!(
        "Generated {} tokens in {:.2}s ({:.2} tokens/sec).",
        metrics.token_count(),
        metrics.total_duration().as_secs_f64(),
        metrics.tokens_per_second().unwrap_or(0.),
    );

    Ok(())
}
