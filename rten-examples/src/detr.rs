use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::path::Path;

use rten::{FloatOperators, Model, Operators};
use rten_imageio::{read_image, write_image};
use rten_imageproc::{Painter, Rect, normalize_image};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use serde::Deserialize;

struct Args {
    model_dir: String,
    image: String,
    annotated_image: Option<String>,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();
    let mut annotated_image = None;

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Detect objects in images.

Usage: {bin_name} <model_dir>

Arguments:
  
  <model_dir>

    Path to directory containing converted model and configuration.

    This should contain:

    - `model.rten` - The DETR or RT-DETR model
    - `config.json` - JSON file containing class ID to label mappings
    - `preprocessor_config.json` - JSON file containing preprocessor configuration

Options:

  --annotate <path>

    Annotate image with bounding boxes and save to <path>
",
                    bin_name = parser.bin_name().unwrap_or("detr")
                );
                std::process::exit(0);
            }
            Long("annotate") => {
                annotated_image = Some(parser.value()?.string()?);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model_dir = values.pop_front().ok_or("missing `model_dir` arg")?;
    let image = values.pop_front().ok_or("missing `image` arg")?;

    let args = Args {
        model_dir,
        image,
        annotated_image,
    };

    Ok(args)
}

/// Model configuration from a Hugging Face `config.json` file.
#[derive(Deserialize)]
struct ModelConfig {
    id2label: HashMap<u32, String>,
}

/// Read model configuration from a Hugging Face `config.json` file.
fn read_config(path: &Path) -> Result<ModelConfig, Box<dyn Error>> {
    let config_json = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    let config = serde_json::from_str(&config_json)?;
    Ok(config)
}

#[derive(Deserialize)]
struct SizeSpec {
    height: Option<u32>,
    width: Option<u32>,
    longest_edge: Option<u32>,
    shortest_edge: Option<u32>,
}

/// Image preprocessing configuration from a Hugging Face `preprocessor_config.json` file.
#[derive(Deserialize)]
struct PreprocessorConfig {
    do_normalize: bool,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    size: SizeSpec,
}

/// Read image pre-processing configuration from a Hugging Face
/// `preprocessor_config.json` file.
fn read_preprocessor_config(path: &Path) -> Result<PreprocessorConfig, Box<dyn Error>> {
    let config_json = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    let config = serde_json::from_str(&config_json)?;
    Ok(config)
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

/// Labels which represent no detection.
const NO_OBJECT_LABELS: &[&str] = &["N/A"];

/// Detect objects in images using DETR [^1] or RT-DETR.
///
/// The DETR model [^2] can be obtained from Hugging Face and converted to this
/// library's format using Optimum [^3]:
///
/// ```
/// optimum-cli export onnx --model facebook/detr-resnet-50 detr
/// rten-convert detr/model.onnx
/// ```
///
/// [^1]: <https://arxiv.org/abs/2005.12872>
/// [^2]: <https://huggingface.co/facebook/detr-resnet-50>
/// [^3]: <https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model>
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let model_dir = Path::new(&args.model_dir);
    let model_file = model_dir.join("model.rten");
    let config_file = model_dir.join("config.json");
    let preprocessor_config_file = model_dir.join("preprocessor_config.json");

    let model = Model::load_file(&model_file)
        .map_err(|err| format!("failed to load {}: {}", model_file.display(), err))?;
    let config = read_config(&config_file)?;
    let preprocessor_config = read_preprocessor_config(&preprocessor_config_file)?;

    let mut image = read_image(&args.image)?;

    // Save a copy of the input before normalization and scaling
    let mut annotated_image = args.annotated_image.as_ref().map(|_| image.clone());

    if preprocessor_config.do_normalize {
        normalize_image(
            image.view_mut(),
            preprocessor_config.image_mean,
            preprocessor_config.image_std,
        );
    }

    let [_, image_height, image_width] = image.shape();

    let mut image = image.as_dyn().to_tensor();
    image.insert_axis(0); // Add batch dim

    // Resize image according to preprocessing configuration.
    let (rescaled_width, rescaled_height) = if let (Some(min_size), Some(max_size)) = (
        preprocessor_config.size.shortest_edge,
        preprocessor_config.size.longest_edge,
    ) {
        rescaled_size(
            (image_width, image_height),
            min_size as usize,
            max_size as usize,
        )
    } else if let (Some(width), Some(height)) = (
        preprocessor_config.size.width,
        preprocessor_config.size.height,
    ) {
        (width as usize, height as usize)
    } else {
        return Err("Preprocessor config file does not specify image size".into());
    };

    println!("Input image size: {} x {}", rescaled_width, rescaled_height);

    if rescaled_width != image_width || rescaled_height != image_height {
        image = image.resize_image([rescaled_height, rescaled_width])?;
    }

    let [logits, boxes] = model.run_n(
        vec![("pixel_values", image.into())],
        ["logits", "pred_boxes"],
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

    let threshold = 0.5;

    struct Match<'a> {
        object_id: u32,
        class: u32,
        label: &'a str,
        prob: f32,

        /// Object coordinates as [center_x, center_y, width, height]. Values
        /// are relative to the image size.
        coords: [f32; 4],
    }

    let mut matches = Vec::new();

    // Extract matches above detection threshold.
    for object_id in 0..n_objects {
        let class = classes[[0, object_id]] as u32;
        let prob = probs[[0, object_id, class as usize]];

        let Some(label) = config.id2label.get(&class) else {
            continue;
        };
        if NO_OBJECT_LABELS.contains(&label.as_str()) {
            continue;
        }
        if prob < threshold {
            continue;
        }

        let [center_x, center_y, width, height] = boxes.slice([0, object_id]).to_array();

        matches.push(Match {
            object_id: object_id as u32,
            class,
            label,
            prob,
            coords: [center_x, center_y, width, height],
        });
    }

    // Sort matches in descending order of probability.
    matches.sort_by(|a, b| a.prob.total_cmp(&b.prob).reverse());

    // Print match details and save annotated image.
    let mut painter = annotated_image
        .as_mut()
        .map(|img| Painter::new(img.view_mut()));
    let stroke_width = 2;

    if let Some(painter) = painter.as_mut() {
        painter.set_stroke([1., 0., 0.]);
        painter.set_stroke_width(stroke_width);
    }

    for Match {
        object_id,
        class,
        label,
        prob,
        coords: [center_x, center_y, width, height],
    } in matches
    {
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
            "object: {object_id} class: {class} ({label}) prob: {prob:.2} coords: [{:?}]",
            [center_x, center_y, width, height]
        );
    }

    if let (Some(annotated_image), Some(path)) = (annotated_image, args.annotated_image) {
        write_image(&path, annotated_image.view())?;
    }

    Ok(())
}
