use std::collections::VecDeque;
use std::error::Error;
use std::io::prelude::*;

use rten::{FloatOperators, Model};
use rten_generate::{Generator, GeneratorUtils};
use rten_imageio::read_image;
use rten_imageproc::normalize_image;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use rten_text::Tokenizer;

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
                    "Read text from an image containing a single text line.

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

/// Recognize text line images using TrOCR [^1].
///
/// First use Hugging Face's Optimum tool to download and export the models to
/// ONNX:
///
/// ```
/// optimum-cli export onnx --model microsoft/trocr-base-printed trocr-base-printed
/// ```
///
/// Convert the models to `.rten` format. For the decoder you need to use the
/// "merged" model.
///
/// ```
/// rten-convert trocr-base-printed/encoder_model.onnx
/// rten-convert trocr-base-printed/decoder_model_merged.onnx
/// ```
///
/// Run the model, specifying the image to recognize:
///
/// ```sh
/// cargo run --release --bin trocr trocr-base-printed/encoder_model.rten trocr-base-printed/decoder_model_merged.rten tokenizer.json <image>
/// ```
///
/// [^1]: https://arxiv.org/abs/2109.10282
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let encoder_model = unsafe { Model::load_mmap(args.encoder_model)? };
    let decoder_model = unsafe { Model::load_mmap(args.decoder_model)? };
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
