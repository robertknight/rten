use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::io::prelude::*;

use rten::{FloatOperators, Model};
use rten_generate::{Generator, GeneratorUtils};
use rten_imageio::read_image;
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;
use rten_text::tokenizer::Tokenizer;

struct Args {
    encoder_model: String,
    decoder_model: String,
    tokenizer_config: String,
    image_path: String,
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
                    "Generate a caption for an image.

Usage: {bin_name} [options] <encoder_model> <decoder_model> <tokenizer> <image>

Args:

  <encoder_model>  - Image encoder model
  <decoder_model>  - Text decoder model
  <tokenizer>      - `tokenizer.json` file
  <image>          - Image path
",
                    bin_name = parser.bin_name().unwrap_or("distilvit")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let encoder_model = values.pop_front().ok_or("missing `encoder_model` arg")?;
    let decoder_model = values.pop_front().ok_or("missing `decoder_model` arg")?;
    let tokenizer_config = values.pop_front().ok_or("missing `tokenizer` arg")?;
    let image_path = values.pop_front().ok_or("missing `image_path` arg")?;

    let args = Args {
        encoder_model,
        decoder_model,
        tokenizer_config,
        image_path,
    };

    Ok(args)
}

/// Generates captions for an image using Mozilla's DistilViT.
///
/// 1. Download the `onnx/encoder.onnx` and `onnx/decoder_with_past.onnx` ONNX
///    models from https://huggingface.co/Mozilla/distilvit/tree/main, as well
///    as the `tokenizer.json` file.
/// 2. Convert the models
///
/// ```sh
/// rten-convert encoder_model.onnx
/// rten-convert decoder_model_with_past.onnx
/// ```
///
/// 3. Run the converted model, specifying the image to caption:
///
/// ```sh
/// cargo run --release --bin distilvit encoder_model.rten decoder_model.rten tokenizer.json <image>
/// ```
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let encoder_model = Model::load_file(args.encoder_model)?;
    let decoder_model = Model::load_file(args.decoder_model)?;
    let tokenizer_config = fs::read_to_string(&args.tokenizer_config)?;
    let tokenizer = Tokenizer::from_json(&tokenizer_config)?;
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
