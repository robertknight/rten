use std::collections::VecDeque;
use std::error::Error;

use rten::{Dimension, FloatOperators, Model};
use rten_imageio::{read_image, write_image};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

struct Args {
    /// Path to image encoder model.
    encoder_model: String,

    /// Path to prompt encoder / mask decoder model.
    decoder_model: String,

    /// Path to input image to segment.
    image: String,

    /// (x, y) query points identifying the object(s) to generate segmentation
    /// masks for.
    points: Vec<(u32, u32)>,
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
                    "Segment an image.

Usage: {bin_name} <encoder_model> <decoder_model> <image> <points>

Args:

  <encoder_model> - Image encoder model
  <decoder_model> - Prompt decoder model
  <image> - Image to process
  <points> -

    List of points identifying the object to segment.

    This has the form `x1,y1;x2,y2;...`. At least one point must be provided.
",
                    bin_name = parser.bin_name().unwrap_or("segment_anything")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let encoder_model = values.pop_front().ok_or("missing `encoder_model` arg")?;
    let decoder_model = values.pop_front().ok_or("missing `decoder_model` arg")?;
    let image = values.pop_front().ok_or("missing `image` arg")?;
    let points_str = values.pop_front().ok_or("missing `points` arg")?;

    let mut points: Vec<(u32, u32)> = Vec::new();
    for xy_str in points_str.split(";") {
        let Some(xy_coords) = xy_str.trim().split_once(",") else {
            return Err(lexopt::Error::Custom(
                "points should be x,y coordinate pairs".into(),
            ));
        };
        let (Ok(x), Ok(y)) = (xy_coords.0.parse(), xy_coords.1.parse()) else {
            return Err(lexopt::Error::Custom(
                "points should be positive integer values".into(),
            ));
        };
        points.push((x, y));
    }

    let args = Args {
        image,
        encoder_model,
        decoder_model,
        points,
    };

    Ok(args)
}

/// Perform image segmentation using Segment Anything [^1].
///
/// First export the ONNX model using Hugging Face's Optimum tool:
///
/// ```
/// optimum-cli export onnx --model facebook/sam-vit-base sam-vit-base
/// ```
///
/// Then convert the models to `.rten` format and run the demo, specifying a
/// path to the image to segment and one or more points in the image identifying
/// the object of interest.
///
/// ```
/// rten-convert sam-vit-base/vision_encoder.onnx
/// rten-convert sam-vit-base/prompt_encoder_mask_decoder.rten
/// cargo run --release --bin segment_anything sam-vit-base/vision_encoder.rten sam-vit-base/prompt_encoder_mask_decoder.rten image.jpg points
/// ```
///
/// Where `points` is a semi-colon separated list of x,y pixel coordinates
/// identifying the objects to segment. For example `200,300;205,305` generates
/// a segmentation mask for the object containing the points (200, 300) and
/// (205, 305). At least one point must be specified.
///
/// ## Alternative models
///
/// The original SAM model uses a computationally expensive vision encoder
/// paired with a lightweight prompt decoder. Since its release various teams
/// have created alternatives with faster image encoders. For faster generation
/// of image embeddings, you can try alternatives such as:
///
/// - [SlimSAM](https://huggingface.co/Zigeng/SlimSAM-uniform-50)
///
/// The process for exporting and converting the models is the same as for
/// the `facebook/sam-vit-base` model.
///
/// [^1]: https://segment-anything.com
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    println!("Loading model...");
    let encoder = Model::load_file(args.encoder_model)?;
    let decoder = Model::load_file(args.decoder_model)?;

    println!("Reading image...");
    let mut image: Tensor = read_image(&args.image)?.into();
    let image_h = image.size(1);
    let image_w = image.size(2);
    image.insert_axis(0);

    // Prepare the input image.
    //
    // This currently does the mandatory resizing of the input image, but
    // doesn't normalize the pixel values.
    let pixel_values_id = encoder.node_id("pixel_values")?;
    let [input_h, input_w] = match encoder
        .node_info(pixel_values_id)
        .and_then(|ni| ni.shape())
        .as_deref()
    {
        Some(&[_, _, Dimension::Fixed(h), Dimension::Fixed(w)]) => [h, w],
        _ => [1024, 1024],
    };
    let image = image.resize_image([input_h, input_w])?;

    // Generate image embeddings.
    println!("Generating image embedding...");

    let [image_embeddings, image_pos_embeddings] = encoder.run_n(
        vec![(pixel_values_id, image.view().into())],
        [
            encoder.node_id("image_embeddings")?,
            encoder.node_id("image_positional_embeddings")?,
        ],
        None,
    )?;

    println!("Segmenting image with {} points...", args.points.len());

    // Prepare decoder inputs.
    let h_scale = input_h as f32 / image_h as f32;
    let w_scale = input_w as f32 / image_w as f32;

    let point_batch = 1;
    let nb_points_per_image = args.points.len();
    let input_points = NdTensor::from_fn(
        [1, point_batch, nb_points_per_image, 2],
        |[_, _, point, coord]| {
            if coord == 0 {
                args.points[point].0 as f32 * w_scale
            } else {
                args.points[point].1 as f32 * h_scale
            }
        },
    );

    const MATCH_POINT: i32 = 1;
    const _NON_MATCH_POINT: i32 = 0;
    const _BACKGROUND_POINT: i32 = -1;
    let input_labels = NdTensor::<i32, 3>::full([1, point_batch, nb_points_per_image], MATCH_POINT);

    // Run decoder and generate segmentation masks.
    let [_iou_scores, pred_masks] = decoder.run_n(
        vec![
            (decoder.node_id("input_points")?, input_points.into()),
            (decoder.node_id("input_labels")?, input_labels.into()),
            (
                decoder.node_id("image_embeddings")?,
                image_embeddings.into(),
            ),
            (
                decoder.node_id("image_positional_embeddings")?,
                image_pos_embeddings.into(),
            ),
        ],
        [
            decoder.node_id("iou_scores")?,
            decoder.node_id("pred_masks")?,
        ],
        None,
    )?;

    // Resize the output mask to match the original image and save to disk.
    let pred_masks: NdTensor<f32, 5> = pred_masks.try_into()?;
    let [_batch, _point_batch, _mask, mask_h, mask_w] = pred_masks.shape();
    let best_mask = pred_masks.slice((0, 0, 0)).reshaped([1, 1, mask_h, mask_w]);
    let resized_mask = best_mask.resize_image([image_h, image_w])?;
    write_image("segmented.png", resized_mask.nd_view::<4>().slice(0))?;

    Ok(())
}
