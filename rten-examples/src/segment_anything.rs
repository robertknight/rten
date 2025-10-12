use std::error::Error;

use argh::FromArgs;
use rten::{Dimension, FloatOperators, Model};
use rten_imageio::{read_image, write_image};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

/// segment an image
#[derive(FromArgs)]
struct Args {
    /// path to image encoder model
    #[argh(positional)]
    encoder_model: String,

    /// path to prompt encoder / mask decoder model
    #[argh(positional)]
    decoder_model: String,

    /// path to input image to segment
    #[argh(positional)]
    image: String,

    /// list of points identifying the object to segment.
    ///
    /// this has the form `x1,y1;x2,y2;...`. At least one point must be provided
    #[argh(positional)]
    points: String,
}

/// Perform image segmentation using Segment Anything [^1].
///
/// First export the ONNX model using Hugging Face's Optimum tool:
///
/// ```
/// optimum-cli export onnx --model facebook/sam-vit-base sam-vit-base
/// ```
///
/// Then run the example, specifying a path to the image to segment and one or
/// more points in the image identifying the object of interest.
///
/// ```
/// cargo run --release --bin segment_anything sam-vit-base/vision_encoder.onnx sam-vit-base/prompt_encoder_mask_decoder.onnx image.jpg points
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
    let args: Args = argh::from_env();

    // Parse points string
    let mut points: Vec<(u32, u32)> = Vec::new();
    for xy_str in args.points.split(";") {
        let Some(xy_coords) = xy_str.trim().split_once(",") else {
            return Err("points should be x,y coordinate pairs".into());
        };
        let (Ok(x), Ok(y)) = (xy_coords.0.parse(), xy_coords.1.parse()) else {
            return Err("points should be positive integer values".into());
        };
        points.push((x, y));
    }

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

    println!("Segmenting image with {} points...", points.len());

    // Prepare decoder inputs.
    let h_scale = input_h as f32 / image_h as f32;
    let w_scale = input_w as f32 / image_w as f32;

    let point_batch = 1;
    let nb_points_per_image = points.len();
    let input_points = NdTensor::from_fn(
        [1, point_batch, nb_points_per_image, 2],
        |[_, _, point, coord]| {
            if coord == 0 {
                points[point].0 as f32 * w_scale
            } else {
                points[point].1 as f32 * h_scale
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
