use std::collections::VecDeque;
use std::error::Error;
use std::fs;

use wasnn::{FloatOperators, Model, Operators};
use wasnn_imageio::{normalize_image, read_image, write_image};
use wasnn_imageproc::{Painter, Rect};
use wasnn_tensor::prelude::*;
use wasnn_tensor::NdTensor;

struct Args {
    model: String,
    image: String,
    annotated_image: Option<String>,
    min_size: Option<u32>,
    max_size: Option<u32>,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();
    let mut annotated_image = None;
    let mut min_size = None;
    let mut max_size = None;

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Detect objects in images.

Usage: {bin_name} <model> <image>

Options:

  --annotate <path>

    Annotate image with bounding boxes and save to <path>

  --max <length>

    Set the maximum length of the longest side of the model's input. Defaults to
    `1333 / 800` * `<shortest side or 800>`.

    Models may support smaller sizes, allowing a trade-off between inference
    time and accuracy.

  --min <length>

    Set the minimum length of the shortest side of the model's input. Defaults to
    `800 / 1333` * `<longest side or 1333>`.
",
                    bin_name = parser.bin_name().unwrap_or("detr")
                );
                std::process::exit(0);
            }
            Long("annotate") => {
                annotated_image = Some(parser.value()?.string()?);
            }
            Long("max") => {
                max_size = Some(parser.value()?.parse()?);
            }
            Long("min") => {
                min_size = Some(parser.value()?.parse()?);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let image = values.pop_front().ok_or("missing `image` arg")?;

    let args = Args {
        model,
        image,
        annotated_image,
        min_size,
        max_size,
    };

    Ok(args)
}

/// Resolve the min/max size to use as inputs for [rescaled_size] based on
/// the min/max CLI args and defaults for the model configuration.
fn resolve_min_max_size(
    min: Option<u32>,
    max: Option<u32>,
    default_min: u32,
    default_max: u32,
) -> (u32, u32) {
    let default_ratio = default_max as f32 / default_min as f32;
    let min = min.map(|x| x as f32);
    let max = max.map(|x| x as f32);

    let (min, max) = match (min, max) {
        (Some(min), None) => (min, min * default_ratio),
        (None, Some(max)) => (max / default_ratio, max),
        (Some(min), Some(max)) => (min, max),
        (None, None) => (default_min as f32, default_max as f32),
    };

    (min.round() as u32, max.round() as u32)
}

/// Calculate rescaled size for an image which currently has dimensions `(width,
/// height)` and needs to be scaled such that the shortest side is >= min_size
/// and the longest side is <= max_size.
fn rescaled_size(
    original_size: (usize, usize),
    min_size: usize,
    max_size: usize,
) -> (usize, usize) {
    let (w, h) = (original_size.0 as f32, original_size.1 as f32);
    let (short, long) = if w < h { (w, h) } else { (h, w) };
    let aspect_ratio = long / short;

    // Calculate new size by scaling up the short side.
    let scaled_up = if short < min_size as f32 {
        let scale = min_size as f32 / short;
        let (new_short, new_long) = (short * scale, (long * scale).min(max_size as f32));
        let new_aspect_ratio = new_long / new_short;
        Some((new_short, new_long, new_aspect_ratio))
    } else {
        None
    };

    // Calculate new size by scaling down the long side.
    let scaled_down = if long > max_size as f32 {
        let scale = max_size as f32 / long;
        let (new_short, new_long) = ((short * scale).max(min_size as f32), long * scale);
        let new_aspect_ratio = new_long / new_short;
        Some((new_short, new_long, new_aspect_ratio))
    } else {
        None
    };

    // Pick the new sizes that minimize the change in aspect ratio.
    let (new_short, new_long) = match (scaled_up, scaled_down) {
        (None, None) => (short, long),
        (Some((su_short, su_long, _)), None) => (su_short, su_long),
        (None, Some((sd_short, sd_long, _))) => (sd_short, sd_long),
        (Some((su_short, su_long, su_ar)), Some((sd_short, sd_long, sd_ar))) => {
            if (aspect_ratio - su_ar).abs() < (aspect_ratio - sd_ar).abs() {
                (su_short, su_long)
            } else {
                (sd_short, sd_long)
            }
        }
    };

    if w < h {
        (new_short.ceil() as usize, new_long.floor() as usize)
    } else {
        (new_long.floor() as usize, new_short.ceil() as usize)
    }
}

// Labels obtained from `id2label` map in
// https://huggingface.co/facebook/detr-resnet-50/blob/main/config.json.
const LABELS: &[&str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Detect objects in images using DETR [1].
///
/// The DETR model [2] can be obtained from Hugging Face and converted to this
/// library's format using Optimum [3]:
///
/// ```
/// optimum-cli export onnx --model facebook/detr-resnet-50 detr
/// tools/convert-onnx.py detr/model.onnx detr.model
/// ```
///
/// This model also works with YOLOS. Use `hustvl/yolos-tiny` or
/// `hustvl/yolos-small` as the `--model` argument in the command above.
///
/// Run this program on an image:
///
/// ```
/// cargo run --release --bin detr detr.model image.jpg
/// ```
///
/// [1] https://arxiv.org/abs/2005.12872
/// [2] https://huggingface.co/facebook/detr-resnet-50
/// [3] https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let model_data = fs::read(args.model)?;
    let model = Model::load(&model_data)?;

    let mut image = read_image(&args.image)?;

    // Save a copy of the input before normalization and scaling
    let mut annotated_image = args.annotated_image.as_ref().map(|_| image.clone());

    normalize_image(image.view_mut());

    let [_, image_height, image_width] = image.shape();

    let mut image = image.as_dyn().to_tensor();
    image.insert_dim(0); // Add batch dim

    // Resize input image according to min/max side length constraints.
    //
    // The DETR defaults are (800, 1333) for the shortest/longest sides, but
    // models may work with smaller sizes at lower accuracy but faster inference.
    let (min_size, max_size) = resolve_min_max_size(args.min_size, args.max_size, 800, 1333);
    let (rescaled_width, rescaled_height) = rescaled_size(
        (image_width, image_height),
        min_size as usize,
        max_size as usize,
    );

    println!("Input image size: {} x {}", rescaled_width, rescaled_height);

    if rescaled_width != image_width || rescaled_height != image_height {
        image = image.resize_image([rescaled_height, rescaled_width])?;
    }

    let pixel_input_id = model.node_id("pixel_values")?;
    let logits_output_id = model.node_id("logits")?;
    let boxes_output_id = model.node_id("pred_boxes")?;

    let [logits, boxes] = model.run_n(
        &[(pixel_input_id, image.view().into())],
        [logits_output_id, boxes_output_id],
        None,
    )?;
    let logits: NdTensor<f32, 3> = logits.try_into()?;
    let boxes: NdTensor<f32, 3> = boxes.try_into()?;

    let probs: NdTensor<f32, 3> = logits.softmax(-1 /* axis */)?.try_into()?;
    let classes: NdTensor<i32, 2> = logits
        .arg_max(-1 /* axis */, false /* keep_dims */)?
        .try_into()?;

    let [cls_batch, n_objects] = classes.shape();
    let [boxes_batch, n_boxes, n_coords] = boxes.shape();

    assert!(cls_batch == 1 && boxes_batch == 1);
    assert!(n_objects == n_boxes);
    assert!(n_coords == 4);

    let mut painter = annotated_image
        .as_mut()
        .map(|img| Painter::new(img.view_mut()));
    let stroke_width = 2;

    if let Some(painter) = painter.as_mut() {
        painter.set_stroke([1., 0., 0.]);
        painter.set_stroke_width(stroke_width);
    }

    for obj in 0..n_objects {
        let cls = classes[[0, obj]] as usize;
        let prob = probs[[0, obj, cls]];

        let Some(label) = LABELS.get(cls - 1) else {
            continue;
        };

        let [center_x, center_y, width, height] = boxes.slice([0, obj]).to_array();

        let rect = Rect::from_tlhw(
            (center_y - 0.5 * height) * image_height as f32,
            (center_x - 0.5 * width) * image_width as f32,
            height * image_height as f32,
            width * image_width as f32,
        );

        let int_rect = rect.integral_bounding_rect().clamp(Rect::from_tlhw(
            stroke_width as i32,
            stroke_width as i32,
            image_height as i32 - 2 * stroke_width as i32,
            image_width as i32 - 2 * stroke_width as i32,
        ));

        if let Some(painter) = painter.as_mut() {
            painter.draw_polygon(&int_rect.corners());
        }

        println!(
            "object: {obj} class: {cls} ({label}) prob: {prob:.2} coords: [{:?}]",
            [center_x, center_y, width, height]
        );
    }

    if let (Some(annotated_image), Some(path)) = (annotated_image, args.annotated_image) {
        write_image(&path, annotated_image.view())?;
    }

    Ok(())
}
