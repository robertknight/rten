use std::collections::VecDeque;
use std::error::Error;

use rten::{FloatOperators, Model, Operators};
use rten_imageio::{read_image, write_image};
use rten_imageproc::{IMAGENET_MEAN, IMAGENET_STD_DEV, normalize_image};
use rten_tensor::Tensor;
use rten_tensor::prelude::*;

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
                    "Perform monocular depth estimation on an image.

Usage: {bin_name} <model> <image> [<output>]

Args:

  <model> - Input Depth Anything model
  <image> - Image to process
  <output> - Path to save depth image to. Defaults to \"depth-map.png\".
",
                    bin_name = parser.bin_name().unwrap_or("depth_anything")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let image = values.pop_front().ok_or("missing `image` arg")?;
    let output = values.pop_front().unwrap_or("depth-map.png".into());

    let args = Args {
        image,
        model,
        output,
    };

    Ok(args)
}

/// Perform monocular depth estimation using [Depth Anything][depth_anything].
///
/// The ONNX models can be obtained from
/// https://github.com/fabio-sim/Depth-Anything-ONNX. See the
/// [releases](https://github.com/fabio-sim/Depth-Anything-ONNX/releases) page
/// for pre-trained model links. This example was tested with the V1 release of
/// the model. The small ("vits") model is recommended for CPU inference.
///
/// After downloading the model, it can be run on an image using:
///
/// ```
/// cargo run --release --bin depth_anything depth_anything.onnx image.jpg
/// ```
///
/// This will generate a depth map as `depth-map.png`.
///
/// [depth_anything]: <https://github.com/LiheYoung/Depth-Anything>
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.model)?;

    let mut image: Tensor = read_image(&args.image)?.into();
    let [_, orig_height, orig_width] = image.shape().try_into()?;
    normalize_image(image.nd_view_mut(), IMAGENET_MEAN, IMAGENET_STD_DEV);
    image.insert_axis(0); // Add batch dim

    // Input size taken from README in https://github.com/fabio-sim/Depth-Anything-ONNX.
    let [input_h, input_w] = [518, 518];
    let image = image.resize_image([input_h, input_w])?;

    // Run model to estimate depth for each pixel.
    //
    // Depending on the model variant used, the output will be either
    // 3D (batch, height, width) or 4D (batch, 1, height, width).
    let mut output: Tensor<f32> = model.run_one(image.view().into(), None)?.try_into()?;
    if output.ndim() == 3 {
        output.insert_axis(1); // Add channel dim
    }

    // Normalize depth values to be in the range [0, 1].
    let min = output
        .reduce_min(None, false /* keep_dims */)?
        .item()
        .copied()
        .unwrap();
    let max = output
        .reduce_max(None, false /* keep_dims */)?
        .item()
        .copied()
        .unwrap();
    output.apply(|x| (x - min) / (max - min));

    // Resize output map back to original input size and write to file.
    let resized = output.resize_image([orig_height, orig_width])?;
    let resized = resized.nd_view::<4>().slice(0);
    write_image(&args.output, resized)?;

    Ok(())
}
