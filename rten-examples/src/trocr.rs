use std::error::Error;
use std::io::prelude::*;

use argh::FromArgs;
use rten::{FloatOperators, Model};
use rten_generate::{Generator, GeneratorUtils};
use rten_imageio::read_image;
use rten_imageproc::normalize_image;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use rten_text::Tokenizer;

/// Read text from an image containing a single text line.
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

/// Recognize text line images using TrOCR [^1].
///
/// First use Hugging Face's Optimum tool to download and export the models to
/// ONNX:
///
/// ```
/// optimum-cli export onnx --model microsoft/trocr-base-printed trocr-base-printed
/// ```
///
/// Run the model, specifying the path of an image to process. The image should
/// contain a single line of printed text.
///
/// ```sh
/// cargo run --release --bin trocr trocr-base-printed/encoder_model.onnx trocr-base-printed/decoder_model_merged.onnx tokenizer.json <image>
/// ```
///
/// There are variants of the model that support handwritten text. Note that
/// some model variants use SentencePiece tokenizers which are not supported
/// by `rten-text` yet. You can use the `tokenizers` crate instead for these.
///
/// [^1]: https://arxiv.org/abs/2109.10282
fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();

    // `Model::load_mmap` could be used here to reduce memory usage, but see
    // notes in documentation about compatible model formats.
    let encoder_model = Model::load_file(args.encoder_model)?;
    let decoder_model = Model::load_file(args.decoder_model)?;

    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;
    let mut image = read_image(args.image_path)?.into_dyn();
    image.insert_axis(0); // Add batch dim

    // From `image_size` in config.json.
    let mut image: NdTensor<_, 4> = image.resize_image([384, 384])?.try_into()?;

    // Values taken from `preprocessor_config.json`.
    let mean = [0.5, 0.5, 0.5];
    let std_dev = [0.5, 0.5, 0.5];
    normalize_image(image.slice_mut(0), mean, std_dev);

    let encoded_image: NdTensor<f32, 3> = encoder_model
        .run_one(image.view().into(), None)?
        .try_into()?;

    // `decoder_start_token_id` from `generation_config.json`. This is the `</s>`
    // token.
    let decoder_start_token = 2;
    let eos_token = 2;

    let max_tokens = 100;

    let prompt = vec![decoder_start_token];
    let generator = Generator::from_model(&decoder_model)?
        .with_prompt(&prompt)
        .with_constant_input(
            decoder_model.node_id("encoder_hidden_states")?,
            encoded_image.view().into(),
        )
        .stop_on_tokens([eos_token])
        .take(max_tokens)
        .decode(&tokenizer);

    for token in generator {
        let token = token?;

        print!("{}", token);
        let _ = std::io::stdout().flush();
    }

    Ok(())
}
