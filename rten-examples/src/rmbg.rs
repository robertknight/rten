use std::collections::VecDeque;
use std::error::Error;

use rten::{FloatOperators, Model};
use rten_imageio::{read_image, write_image};
use rten_imageproc::normalize_image;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut};

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
                    "Perform background removal on an image.

Usage: {bin_name} <model> <image> [<output>]

Args:

  <model> - Background removal model
  <image> - Image to process
  <output> - Path to save image to. Defaults to \"output.png\".
",
                    bin_name = parser.bin_name().unwrap_or("rmbg")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let image = values.pop_front().ok_or("missing `image` arg")?;
    let output = values.pop_front().unwrap_or("output.png".into());

    let args = Args {
        image,
        model,
        output,
    };

    Ok(args)
}

/// Fill a CHW image with `color` using a mask.
fn fill_mask(mut image: NdTensorViewMut<f32, 3>, mask: NdTensorView<bool, 2>, color: [f32; 3]) {
    let [chans, rows, cols] = image.shape();
    assert_eq!(chans, 3);
    for y in 0..rows {
        for x in 0..cols {
            if mask[[y, x]] {
                image.set_array([0, y, x], 0, color);
            }
        }
    }
}

/// Remove the background of an image using [BRIA Background
/// Removal](https://huggingface.co/briaai/RMBG-1.4).
///
/// The ONNX models can be obtained from https://huggingface.co/briaai/RMBG-1.4.
/// See the "Files and Versions" page.
///
/// Assuming the model has been downloaded and named as "rmbg.onnx", it can be
/// run on an image using:
///
/// ```
/// rten-convert rmbg.onnx
/// cargo run --release --bin rmbg rmbg.rten image.jpg
/// ```
///
/// This will generate `output.png`, a copy of the input image with the
/// background replaced with a fixed color.
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.model)?;

    let mut image: NdTensor<f32, 3> = read_image(&args.image)?;

    let mut normalized_image = image.clone();
    let mean = [0.5, 0.5, 0.5];
    let std_dev = [1.0, 1.0, 1.0];
    normalize_image(normalized_image.view_mut(), mean, std_dev);

    let [_, orig_height, orig_width] = image.shape();

    let mut normalized_image = normalized_image.into_dyn();
    normalized_image.insert_axis(0); // Add batch dim

    let [input_h, input_w] = [1024, 1024];
    let resized_image = normalized_image.resize_image([input_h, input_w])?;

    // Run the model to get the probability of each pixel being part of the
    // image's foreground, then apply a threshold to get a mask indicating which
    // pixels are part of the image's background.
    let foreground_prob: NdTensor<f32, 4> = model
        .run_one(resized_image.view().into(), None)?
        .try_into()?;
    let foreground_prob: NdTensor<f32, 4> = foreground_prob
        .resize_image([orig_height, orig_width])?
        .try_into()?;
    let foreground_threshold = 0.5;
    let background_mask = foreground_prob.map(|x| *x < foreground_threshold);

    // Replace background with a fixed color.
    let bg_color = [0., 1., 0.]; // RGB
    fill_mask(
        image.view_mut(),
        background_mask.slice([0, 0]), // Extract first mask and channel
        bg_color,
    );

    write_image(&args.output, image.nd_view())?;

    Ok(())
}
