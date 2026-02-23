use std::error::Error;

use argh::FromArgs;
use rten::{FloatOperators, Model, Operators};
use rten_imageio::{read_image, write_image};
use rten_imageproc::{IMAGENET_MEAN, IMAGENET_STD_DEV, normalize_image};
use rten_tensor::Tensor;
use rten_tensor::prelude::*;

/// Perform monocular depth estimation on an image.
#[derive(FromArgs)]
struct Args {
    /// input Depth Anything model
    #[argh(positional)]
    model: String,

    /// image to process
    #[argh(positional)]
    image: String,

    /// path to save depth image to (default: "depth-map.png")
    #[argh(positional, default = "String::from(\"depth-map.png\")")]
    output: String,
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
    let args: Args = argh::from_env();
    let model = Model::load_file(args.model)?;

    let mut image = read_image(&args.image)?;
    let [_, orig_height, orig_width] = image.shape();
    normalize_image(image.view_mut(), IMAGENET_MEAN, IMAGENET_STD_DEV);
    let image = image.with_new_axis(0); // Add batch dim

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
