use std::collections::VecDeque;
use std::error::Error;
use std::fs;

use rten::{FloatOperators, Model};
use rten_imageio::read_image;
use rten_imageproc::normalize_image;
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;
use rten_text::tokenizers::Tokenizer;

struct Args {
    model: String,
    tokenizer: String,
    images: Vec<String>,
    captions: Vec<String>,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();

    let mut images = Vec::new();
    let mut captions = Vec::new();

    while let Some(arg) = parser.next()? {
        match arg {
            Short('i') | Long("image") => {
                images.push(parser.value()?.string()?);
            }
            Short('c') | Long("caption") => {
                captions.push(parser.value()?.string()?);
            }
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Match images against text captions.

Usage: {bin_name} [options] <model> <tokenizer> [-i image] [-c caption]

At least one image and one caption must be provided.

Args:

  <model>     - CLIP model
  <tokenizer> - tokenizer.json path 

Options:

  -i <image>    - Path to an image
  -c <caption>  - Text caption
",
                    bin_name = parser.bin_name().unwrap_or("clip")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let tokenizer = values.pop_front().ok_or("missing `tokenizer` arg")?;

    let args = Args {
        model,
        tokenizer,
        images,
        captions,
    };

    Ok(args)
}

/// Computes similarity between images and text captions.
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.model)?;
    let tokenizer_config = fs::read_to_string(&args.tokenizer)?;
    let tokenizer = Tokenizer::from_json(&tokenizer_config)?;

    // From preprocessor_config.json
    let image_width = 224;
    let image_height = 224;

    let mut pixel_values = NdTensor::zeros([args.images.len(), 3, image_width, image_height]);
    for (i, img_path) in args.images.iter().enumerate() {
        let mut image = read_image(img_path)?;

        // Values taken from `preprocessor_config.json`.
        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std_dev = [0.26862954, 0.26130258, 0.27577711];
        normalize_image(image.view_mut(), mean, std_dev);

        let mut image = image.into_dyn();

        image.insert_axis(0); // Add batch dim
        let image = image.resize_image([image_width, image_height])?;
        pixel_values
            .slice_mut::<4, _>(i..i + 1)
            .copy_from(&image.nd_view());
    }

    let start_of_text = 49406; // <|startoftext|>
    let end_of_text = 49407; // <|endoftext|>
    let mut encoded_captions = Vec::new();
    let mut max_seq_len = 0;
    for caption in &args.captions {
        let mut tokens = tokenizer.encoder().encode(caption)?;
        tokens.insert(0, start_of_text);
        tokens.push(end_of_text);
        max_seq_len = max_seq_len.max(tokens.len());

        // TESTING
        let decoded = tokenizer.encoder().decode(&tokens).unwrap();
        println!("tokens {:?} decoded \"{}\"", tokens, decoded);

        encoded_captions.push(tokens);
    }

    let input_ids = NdTensor::from_fn([encoded_captions.len(), max_seq_len], |[i, t]| {
        encoded_captions[i].get(t).copied().unwrap_or(end_of_text) as i32
    });
    let attention_mask = NdTensor::from_fn([encoded_captions.len(), max_seq_len], |[i, t]| {
        if encoded_captions[i].len() < t {
            1i32
        } else {
            0i32
        }
    });

    let input_ids_id = model.node_id("input_ids")?;
    let pixel_values_id = model.node_id("pixel_values")?;
    let attention_mask_id = model.node_id("attention_mask")?;
    let logits_per_image_id = model.node_id("logits_per_image")?;

    let [logits_per_image] = model.run_n(
        [
            (input_ids_id, input_ids.into()),
            (pixel_values_id, pixel_values.into()),
            (attention_mask_id, attention_mask.into()),
        ]
        .into(),
        [logits_per_image_id],
        None,
    )?;
    let logits_per_image: NdTensor<f32, 2> = logits_per_image.try_into()?;
    let probs_per_image = logits_per_image.softmax(1)?;

    for (img_idx, image) in (0..probs_per_image.size(0)).zip(&args.images) {
        for (caption_idx, caption) in (0..probs_per_image.size(1)).zip(&args.captions) {
            println!(
                "image \"{}\" caption \"{}\" score {:.3}",
                image,
                caption,
                probs_per_image[[img_idx, caption_idx]]
            );
        }
    }

    Ok(())
}
