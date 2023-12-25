use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::fs;

use wasnn::{Dimension, FloatOperators, Model, Operators};
use wasnn_imageio::{normalize_image, read_image, write_image};
use wasnn_tensor::prelude::*;
use wasnn_tensor::{NdTensor, Tensor};

struct Args {
    model: String,
    image: String,
    output: String,
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
                    "Perform semantic segmentation on an image.

Usage: {bin_name} <model> <image> [<output>]

Args:

  <model> - Input DeepLab model
  <image> - Image to segment
  <output> - Path to save annotated image to. Defaults to \"out.png\".
",
                    bin_name = parser.bin_name().unwrap_or("deeplab")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let image = values.pop_front().ok_or("missing `image` arg")?;
    let output = values.pop_front().unwrap_or("out.png".into());

    let args = Args {
        image,
        model,
        output,
    };

    Ok(args)
}

type Rgb = (f32, f32, f32);

/// Labels and colors for the different categories of object that DeepLabv3 can
/// detect.
///
/// For the labels, see https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
const PASCAL_VOC_LABELS: [(&str, Rgb); 21] = [
    ("background", (1.0, 0.0, 0.0)),   // Red
    ("aeroplane", (0.0, 1.0, 0.0)),    // Green
    ("bicycle", (0.0, 0.0, 1.0)),      // Blue
    ("bird", (1.0, 1.0, 0.0)),         // Yellow
    ("boat", (1.0, 0.0, 1.0)),         // Magenta
    ("bottle", (0.0, 1.0, 1.0)),       // Cyan
    ("bus", (0.5, 0.0, 0.0)),          // Dark Red
    ("car", (0.0, 0.5, 0.0)),          // Dark Green
    ("cat", (0.0, 0.0, 0.5)),          // Dark Blue
    ("chair", (0.5, 0.5, 0.0)),        // Olive
    ("cow", (0.5, 0.0, 0.5)),          // Purple
    ("diningtable", (0.0, 0.5, 0.5)),  // Teal
    ("dog", (0.75, 0.75, 0.75)),       // Light Gray
    ("horse", (0.5, 0.5, 0.5)),        // Gray
    ("motorbike", (0.25, 0.25, 0.25)), // Dark Gray
    ("person", (1.0, 0.5, 0.0)),       // Orange
    ("pottedplant", (0.5, 1.0, 0.5)),  // Pastel Green
    ("sheep", (0.5, 0.5, 1.0)),        // Pastel Blue
    ("sofa", (1.0, 0.75, 0.8)),        // Pink
    ("train", (0.64, 0.16, 0.16)),     // Brown
    ("tvmonitor", (1.0, 1.0, 1.0)),    // White
];

/// Perform semantic segmentation of an image using DeepLabv3 [1].
///
/// This classifies each of the pixels in an image as belonging to one of the 20
/// Pascal VOC object categories or no object category.
///
/// DeepLabV3 models can be obtained from torchvision, using the
/// `export-deeplab.py` script:
///
/// ```
/// python examples/export-deeplab.py
/// tools/convert-onnx.py deeplab.onnx deeplab.model
/// ```
///
/// Run this program on an image with:
///
/// ```
/// cargo run --release --example deeplab deeplab.model image.jpg out.png
/// ```
///
/// [1] https://arxiv.org/abs/1706.05587
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model_bytes = fs::read(args.model)?;
    let model = Model::load(&model_bytes)?;

    let mut image: Tensor = read_image(&args.image)?.into();
    normalize_image(image.nd_view_mut());
    image.insert_dim(0); // Add batch dim

    // Resize image according to metadata in the model.
    let input_shape = model
        .input_shape(0)
        .ok_or("model does not specify expected input shape")?;
    let (input_h, input_w) = match &input_shape[..] {
        &[_, _, Dimension::Fixed(h), Dimension::Fixed(w)] => (h, w),

        // If dimensions are not fixed, use the defaults from when this
        // example was created.
        _ => (520, 780),
    };
    let image = image.resize_image([input_h, input_w])?;

    // Run model to classify each pixel
    let mut output: Tensor = model.run_one(image.view().into(), None)?.try_into()?;
    output.permute(&[0, 2, 3, 1]); // (N,class,H,W) => (N,H,W,class)

    let seg_classes: NdTensor<i32, 2> = output
        .slice(0)
        .arg_max(-1, false /* keep_dims */)?
        .try_into()?;
    let [out_height, out_width] = seg_classes.shape();

    // Generate image with pixels colored according to class label.
    let mut annotated_image = NdTensor::zeros([3, out_height, out_width]);
    let mut found_classes = HashSet::new();
    for y in 0..out_height {
        for x in 0..out_width {
            let cls = seg_classes[[y, x]];
            if cls != 0 {
                found_classes.insert(cls);

                let (r, g, b) = PASCAL_VOC_LABELS
                    .get(cls as usize)
                    .map(|(_, color)| color)
                    .copied()
                    .unwrap_or((1., 1., 1.));

                annotated_image
                    .slice_mut((.., y, x))
                    .assign_array([r, g, b]);
            }
        }
    }

    // Print a list of labels of categories found in the image.
    let classes: Vec<_> = found_classes
        .into_iter()
        .map(|cls| PASCAL_VOC_LABELS.get(cls as usize).map(|(label, _)| *label))
        .flatten()
        .collect();
    if !classes.is_empty() {
        println!("found objects: {}", classes.join(", "));
    } else {
        println!("no recognized objects found");
    }

    write_image(&args.output, annotated_image.view())?;

    Ok(())
}
