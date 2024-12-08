use std::collections::VecDeque;
use std::error::Error;
use std::io::prelude::*;

use rten::{FloatOperators, Model};
use rten_generate::{Generator, GeneratorUtils};
use rten_imageio::read_image;
use rten_imageproc::normalize_image;
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;
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
                    "Extract text from academic PDFs as Markdown.

Usage: {bin_name} [options] <encoder_model> <decoder_model> <tokenizer> <image>

Args:

  <encoder_model>  - Image encoder model
  <decoder_model>  - Text decoder model
  <tokenizer>      - `tokenizer.json` file
  <image>          - Image path
",
                    bin_name = parser.bin_name().unwrap_or("nougat")
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

/// Extract text as markdown from academic PDFs using Nougat.
///
/// First use Hugging Face's Optimum tool to download and export the models to
/// ONNX:
///
/// ```
/// optimum-cli export onnx --model facebook/nougat-base nougat-base
/// ```
///
/// Convert the models to `.rten` format. For the decoder you need to use the
/// "merged" model.
///
/// ```
/// rten-convert nougat-base/encoder_model.onnx
/// rten-convert nougat-base/decoder_model_merged.onnx
/// ```
///
/// Run the model, specifying the image to recognize:
///
/// ```sh
/// cargo run --release --bin nougat nougat-base/encoder_model.rten nougat-base/decoder_model_merged.rten tokenizer.json <image>
/// ```
///
/// [^1]: https://arxiv.org/abs/2308.13418
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let encoder_model = unsafe { Model::load_mmap(args.encoder_model)? };
    let decoder_model = unsafe { Model::load_mmap(args.decoder_model)? };
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;
    let mut image = read_image(args.image_path)?.into_dyn();
    image.insert_axis(0); // Add batch dim

    // Values taken from `preprocessor_config.json`.
    let img_size = [896, 672];
    let mean = [0.485, 0.456, 0.406];
    let std_dev = [0.229, 0.224, 0.225];

    let mut image: NdTensor<_, 4> = image.resize_image(img_size)?.try_into()?;
    normalize_image(image.slice_mut(0), mean, std_dev);

    let encoded_image: NdTensor<f32, 3> = encoder_model
        .run_one(image.view().into(), None)?
        .try_into()?;

    let encoder_hidden_states_id = decoder_model.node_id("encoder_hidden_states")?;

    // `bos_token_id` from `generation_config.json`.
    // token.
    let decoder_start_token = 0;
    let eos_token = 2;

    let max_tokens = 2_000;

    let prompt = [decoder_start_token];
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
