use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rten::ops::{non_max_suppression, BoxOrder};
use rten::{Dimension, FloatOperators, Model, TensorPool};
use rten_imageio::{read_image, write_image};
use rten_imageproc::{Painter, Rect};
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;

struct Args {
    model: String,
    image: String,
    annotated_image: Option<String>,
    summary: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();
    let mut annotated_image = None;
    let mut summary = false;

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

  -s, --summary

    Print only a summary of the objects found
",
                    bin_name = parser.bin_name().unwrap_or("detr")
                );
                std::process::exit(0);
            }
            Long("annotate") => {
                annotated_image = Some(parser.value()?.string()?);
            }
            Short('s') | Long("summary") => summary = true,
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let image = values.pop_front().ok_or("missing `image` arg")?;

    let args = Args {
        model,
        image,
        annotated_image,
        summary,
    };

    Ok(args)
}

fn resource_path(path: &str) -> PathBuf {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push("data/");
    abs_path.push(path);
    abs_path
}

/// Detect objects in images using the YOLO v8 model.
///
/// See <https://docs.ultralytics.com/modes/export/> for current instructions on
/// how to get YOLO v8 models in ONNX format.
///
/// Note that this model is AGPL-licensed. If you need an object detection model
/// for commercial purposes, you can either buy a license from Ultralytics, or
/// use an alternate model such as DETR.
///
/// ```
/// pip install ultralytics
/// yolo mode=export model=yolov8s.pt format=onnx
/// rten-convert yolov8s.onnx yolov8.rten
/// ```
///
/// Run this program on an image:
///
/// ```
/// cargo run --release --bin yolo yolov8.rten image.jpg
/// ```
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let model = Model::load_file(args.model)?;

    let image = read_image(&args.image)?;
    let labels: Vec<_> = fs::read_to_string(resource_path("coco.names"))?
        .lines()
        .map(|s| s.to_string())
        .collect();

    // Save a copy of the input before normalization and scaling
    let mut annotated_image = args.annotated_image.as_ref().map(|_| image.clone());

    let [_, image_height, image_width] = image.shape();

    let mut image = image.as_dyn().to_tensor();
    image.insert_axis(0); // Add batch dim

    let input_shape = model
        .input_shape(0)
        .ok_or("model does not specify expected input shape")?;
    let (input_h, input_w) = match &input_shape[..] {
        &[_, _, Dimension::Fixed(h), Dimension::Fixed(w)] => (h, w),

        // If dimensions are not fixed, use the defaults from when this
        // example was created.
        _ => (640, 640),
    };
    let image = image.resize_image([input_h, input_w])?;

    let input_id = model.node_id("images")?;
    let output_id = model.node_id("output0")?;

    let [output] = model.run_n(&[(input_id, image.view().into())], [output_id], None)?;

    // Output format is [N, 84, B] where `B` is the number of boxes. The second
    // dimension contains `[x, y, w, h, class_0 ... class 79]` where `(x, y)`
    // are the coordinates of the box center, `(h, w)` are the box size and
    // the `class_{i}` fields are class probabilities.
    let output: NdTensor<f32, 3> = output.try_into()?;
    let [_batch, box_attrs, _n_boxes] = output.shape();
    assert!(box_attrs == 84);

    let model_in_h = image.size(2);
    let model_in_w = image.size(3);
    let scale_y = image_height as f32 / model_in_h as f32;
    let scale_x = image_width as f32 / model_in_w as f32;

    // [batch, n_boxes, coord]
    let boxes = output.slice::<3, _>((.., ..4, ..)).permuted([0, 2, 1]);

    // [batch, n_classes, n_boxes]. The `n_boxes` coord is last because that
    // is what `non_max_suppression` requires.
    let scores = output.slice::<3, _>((.., 4.., ..));

    let iou_threshold = 0.3;
    let score_threshold = 0.25;

    // nms_boxes is [n_selected, 3];
    let nms_boxes = non_max_suppression(
        &TensorPool::new(),
        boxes.view(),
        scores,
        BoxOrder::CenterWidthHeight,
        None, /* max_output_boxes_per_class */
        iou_threshold,
        score_threshold,
    )?;
    let [n_selected_boxes, _] = nms_boxes.shape();

    let mut painter = annotated_image
        .as_mut()
        .map(|img| Painter::new(img.view_mut()));
    let stroke_width = 2;

    if let Some(painter) = painter.as_mut() {
        painter.set_stroke([1., 0., 0.]);
        painter.set_stroke_width(stroke_width);
    }

    println!("Found {n_selected_boxes} objects in image.");

    for b in 0..n_selected_boxes {
        let [batch_idx, cls, box_idx] = nms_boxes.slice(b).to_array();
        let [cx, cy, box_w, box_h] = boxes.slice([batch_idx, box_idx]).to_array();
        let score = scores[[batch_idx as usize, cls as usize, box_idx as usize]];

        let rect = Rect::from_tlhw(
            (cy - 0.5 * box_h) * scale_y,
            (cx - 0.5 * box_w) * scale_x,
            box_h * scale_y as f32,
            box_w * scale_x as f32,
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

        let label = labels
            .get(cls as usize)
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        if !args.summary {
            println!(
            "object: {label} score: {score:.3} left: {} top: {} right: {} bottom: {} box index: {box_idx}",
            int_rect.left(),
            int_rect.top(),
            int_rect.right(),
            int_rect.bottom()
        );
        }
    }

    if let (Some(annotated_image), Some(path)) = (annotated_image, args.annotated_image) {
        write_image(&path, annotated_image.view())?;
    }

    Ok(())
}
