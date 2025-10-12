use std::error::Error;
use std::io::prelude::*;

use argh::FromArgs;
use rten::{FloatOperators, Model};
use rten_generate::{Generator, GeneratorUtils};
use rten_imageio::read_image;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use rten_text::Tokenizer;

/// Generate a caption for an image.
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

/// Generates captions for an image using Mozilla's DistilViT.
///
/// 1. Download the `onnx/encoder_model.onnx` and
///    `onnx/decoder_model_merged.onnx` ONNX models from
///    https://huggingface.co/Mozilla/distilvit/tree/main, as well as the
///    `tokenizer.json` file.
///
/// 2. Run the model, specifying the image to caption:
///
/// ```sh
/// cargo run --release --bin distilvit encoder_model.onnx decoder_model_merged.onnx tokenizer.json <image>
/// ```
fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();
    let encoder_model = Model::load_file(args.encoder_model)?;
    let decoder_model = Model::load_file(args.decoder_model)?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;
    let mut image = read_image(args.image_path)?.into_dyn();
    image.insert_axis(0); // Add batch dim
    let image = image.resize_image([224, 224])?;

    let encoded_image: NdTensor<f32, 3> = encoder_model
        .run_one(image.view().into(), None)?
        .try_into()?;

    let encoder_hidden_states_id = decoder_model.node_id("encoder_hidden_states")?;

    // `decoder_start_token_id` value from
    // https://huggingface.co/Mozilla/distilvit/blob/main/config.json.
    let bos_token = 50256;
    let eos_token = bos_token;

    // Taken from https://github.com/mozilla/distilvit/blob/9c301fd5ba1f62ab407ca0a342642666a1ec13c5/distilvit/infere.py#L45
    let max_tokens = 40;

    let prompt = vec![bos_token];
    let generator = Generator::from_model(&decoder_model)?
        .with_prompt(&prompt)
        .with_constant_input(encoder_hidden_states_id, encoded_image.view().into())
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
