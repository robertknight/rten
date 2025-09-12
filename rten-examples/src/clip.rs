use std::collections::VecDeque;
use std::error::Error;

use rten::{FloatOperators, Model};
use rten_imageio::read_image;
use rten_imageproc::normalize_image;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use rten_text::Tokenizer;

struct Args {
    model: String,
    tokenizer: String,
    images: Vec<String>,
    captions: Vec<String>,
    debug_tokens: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();

    let mut images = Vec::new();
    let mut captions = Vec::new();
    let mut debug_tokens = false;

    while let Some(arg) = parser.next()? {
        match arg {
            Short('i') | Long("image") => {
                images.push(parser.value()?.string()?);
            }
            Short('c') | Long("caption") => {
                captions.push(parser.value()?.string()?);
            }
            Short('t') | Long("tokens") => {
                debug_tokens = true;
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
  -t, --tokens  - Print tokenized captions
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
        debug_tokens,
    };

    Ok(args)
}

// Preprocess input CHW image as follows:
//
// - Resize shortest side to `target_width` or `target_height`
// - Crop out the center portion of size (target_width, target_height)
// - Normalize pixel values
fn preprocess_image(
    image: NdTensor<f32, 3>,
    target_width: u32,
    target_height: u32,
) -> Result<NdTensor<f32, 3>, Box<dyn Error>> {
    let [_chans, input_height, input_width] = image.shape();

    // Resize so shortest side matches target, preserving aspect ratio.
    let (resized_width, resized_height) = if input_width < input_height {
        let scale = target_width as f32 / input_width as f32;
        (target_width, (input_height as f32 * scale) as u32)
    } else {
        let scale = target_height as f32 / input_height as f32;
        ((input_width as f32 * scale) as u32, target_height)
    };
    debug_assert!(resized_width >= target_width);
    debug_assert!(resized_height >= target_height);

    let mut image = image.into_dyn();
    image.insert_axis(0); // Add batch dim
    let mut image = image.resize_image([resized_height as usize, resized_width as usize])?;
    image.remove_axis(0);
    let mut image: NdTensor<_, 3> = image.try_into()?;

    // Center crop image.
    let crop_top = resized_height.saturating_sub(target_height) / 2;
    let crop_left = resized_width.saturating_sub(target_width) / 2;
    let crop_bottom = crop_top + target_height;
    let crop_right = crop_left + target_width;
    let mut cropped_image = image.slice_mut((.., crop_top..crop_bottom, crop_left..crop_right));

    // Normalize image, using values taken from `preprocessor_config.json`.
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std_dev = [0.26862954, 0.2613026, 0.2757771];
    normalize_image(cropped_image.view_mut(), mean, std_dev);

    Ok(cropped_image.to_tensor())
}

/// Compute similarity between images and text captions using OpenAI's CLIP [^1].
///
/// The CLIP model [^2] can be obtained from Hugging Face and converted to RTen
/// format using Optimum:
///
/// ```
/// optimum-cli export onnx --model openai/clip-vit-base-patch32 clip-vit-base-patch32
/// rten-convert clip-vit-base-patch32/model.onnx
/// ```
///
/// Run this program specifying at least one image and at least one caption:
///
/// ```
/// cargo run --release --bin clip clip-vit-base-patch32/model.rten clip-vit-base-patch32/tokenizer.json -i ../tools/test-images/horses.jpeg -c "horses" -c "ducks"
/// ```
///
/// [^1]: https://github.com/openai/CLIP
/// [^2]: https://huggingface.co/openai/clip-vit-base-patch32
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.model)?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;

    // From preprocessor_config.json
    let image_width = 224;
    let image_height = 224;

    let mut pixel_values = NdTensor::zeros([args.images.len(), 3, image_width, image_height]);
    for (i, img_path) in args.images.iter().enumerate() {
        let image = read_image(img_path)?;
        let image = preprocess_image(image, image_width as u32, image_height as u32)?;
        pixel_values.slice_mut(i).copy_from(&image);
    }

    let start_of_text = tokenizer.get_token_id("<|startoftext|>")?;
    let end_of_text = tokenizer.get_token_id("<|endoftext|>")?;

    let mut encoded_captions = Vec::new();
    for caption in &args.captions {
        let mut tokens = tokenizer.encode(caption.as_str(), None)?.into_token_ids();
        tokens.insert(0, start_of_text);
        tokens.push(end_of_text);

        if args.debug_tokens {
            let decoded = tokenizer.decode(&tokens).unwrap();
            println!("tokens {:?} decoded \"{}\"", tokens, decoded);
        }

        encoded_captions.push(tokens);
    }
    let max_seq_len = encoded_captions
        .iter()
        .map(|tokens| tokens.len())
        .max()
        .unwrap_or(0);

    let input_ids = NdTensor::from_fn(
        [encoded_captions.len(), max_seq_len],
        |[caption_idx, pos]| {
            encoded_captions[caption_idx]
                .get(pos)
                .copied()
                .unwrap_or(end_of_text) as i32
        },
    );
    let attention_mask = NdTensor::from_fn(
        [encoded_captions.len(), max_seq_len],
        |[caption_idx, pos]| {
            if encoded_captions[caption_idx].len() > pos {
                1i32
            } else {
                0i32
            }
        },
    );

    let [logits_per_image] = model.run_n(
        [
            (model.node_id("input_ids")?, input_ids.into()),
            (model.node_id("pixel_values")?, pixel_values.into()),
            (model.node_id("attention_mask")?, attention_mask.into()),
        ]
        .into(),
        [model.node_id("logits_per_image")?],
        None,
    )?;
    let logits_per_image: NdTensor<f32, 2> = logits_per_image.try_into()?;
    let probs_per_image = logits_per_image.softmax(1)?;

    for (img_idx, image) in (0..probs_per_image.size(0)).zip(&args.images) {
        for (caption_idx, caption) in (0..probs_per_image.size(1)).zip(&args.captions) {
            println!(
                "image \"{}\" caption \"{}\" score {:.2}",
                image,
                caption,
                probs_per_image[[img_idx, caption_idx]]
            );
        }
    }

    Ok(())
}
