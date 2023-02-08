extern crate png;
extern crate wasnn;

use std::error::Error;
use std::fs;
use std::io::BufWriter;
use std::iter::zip;

use wasnn::ops::{resize, CoordTransformMode, NearestMode, ResizeMode, ResizeTarget};
use wasnn::{tensor, Dimension, Model, RunOptions, Tensor, TensorLayout};

/// Read a PNG image from `path` into an NCHW tensor with one channel.
fn read_greyscale_image<N: Fn(f32) -> f32>(
    path: &str,
    normalize_pixel: N,
) -> Result<Tensor<f32>, Box<dyn Error>> {
    let input_img = fs::File::open(path)?;
    let decoder = png::Decoder::new(input_img);
    let mut reader = decoder.read_info()?;

    let (in_chans, in_color_chans) = match reader.output_color_type() {
        (png::ColorType::Rgb, png::BitDepth::Eight) => (3, 3),
        (png::ColorType::Rgba, png::BitDepth::Eight) => (4, 3),
        (png::ColorType::Grayscale, png::BitDepth::Eight) => (1, 1),
        _ => return Err("Unsupported input image format".into()),
    };

    let info = reader.info();
    let width = info.width as usize;
    let height = info.height as usize;

    let mut img_data = vec![0; reader.output_buffer_size()];
    let frame_info = reader.next_frame(&mut img_data).unwrap();
    img_data.truncate(frame_info.buffer_size());

    let img = Tensor::from_data(&[height, width, in_chans], img_data);
    let mut output = Tensor::zeros(&[1, 1, height, width]);
    for y in 0..height {
        for x in 0..width {
            let mut pixel: u32 = 0;
            for c in 0..in_color_chans {
                let component: u32 = img[[y, x, c]].into();
                pixel += component;
            }
            // TODO: Correct RGB => Greyscale conversion.
            pixel /= in_color_chans as u32;
            let in_val = normalize_pixel(pixel as f32 / 255.0);
            output[[0, 0, y, x]] = in_val;
        }
    }

    Ok(output)
}

/// Convert an NCHW float tensor with values in the range [0, 1] to an
/// 8-bit grayscale image.
fn image_from_tensor(tensor: &Tensor) -> Vec<u8> {
    tensor
        .iter()
        .map(|x| (x.clamp(0., 1.) * 255.0) as u8)
        .collect()
}

/// Compute color of composited pixel using source-over compositing.
///
/// See https://en.wikipedia.org/wiki/Alpha_compositing#Description.
fn composite_pixel(src_color: f32, src_alpha: f32, dest_color: f32, dest_alpha: f32) -> f32 {
    let alpha_out = src_alpha + dest_alpha * (1. - src_alpha);
    let color_out =
        (src_color * src_alpha + (dest_color * dest_alpha) * (1. - src_alpha)) / alpha_out;
    color_out
}

/// This example loads a PNG image and uses a binary image segmentation model
/// to classify pixels, for tasks such as detecting text, faces or
/// foreground/background separation. The output is the result of multiplying
/// input image pixels by the corresponding probabilities from the segmentation
/// mask.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() <= 1 {
        println!(
            "Usage: {} <model> <image> <output>",
            args.get(0).map(|s| s.as_str()).unwrap_or("")
        );
        // Exit with non-zero status, but also don't show an error.
        std::process::exit(1);
    }

    let model_name = args.get(1).ok_or("model name not specified")?;
    let image_path = args.get(2).ok_or("image path not specified")?;
    let output_path = args.get(3).ok_or("output path not specified")?;

    let model_bytes = fs::read(model_name)?;
    let model = Model::load(&model_bytes)?;

    let input_id = model
        .input_ids()
        .get(0)
        .copied()
        .expect("model has no inputs");
    let input_shape = model
        .node_info(input_id)
        .and_then(|info| info.shape())
        .ok_or("model does not specify expected input shape")?;
    let output_id = model
        .output_ids()
        .get(0)
        .copied()
        .expect("model has no outputs");

    let normalize_pixel = |pixel| pixel - 0.5;
    let img =
        read_greyscale_image(image_path, normalize_pixel).expect("failed to read input image");
    let [_, _, img_height, img_width] = img.dims();

    let bilinear_resize = |img, height, width| {
        resize(
            img,
            ResizeTarget::Sizes(&tensor!([1, 1, height, width])),
            ResizeMode::Linear,
            CoordTransformMode::default(),
            NearestMode::default(),
        )
    };

    let (in_height, in_width) = match input_shape[..] {
        [_, _, Dimension::Fixed(h), Dimension::Fixed(w)] => (h, w),
        _ => {
            return Err("failed to get model dims".into());
        }
    };

    let resized_img = bilinear_resize(&img, in_height as i32, in_width as i32)?;

    let outputs = model.run(
        &[(input_id, (&resized_img).into())],
        &[output_id],
        Some(RunOptions {
            timing: true,
            verbose: false,
        }),
    )?;
    let text_mask = &outputs[0].as_float_ref().unwrap();
    let text_mask = bilinear_resize(text_mask, img_height as i32, img_width as i32)?;

    let mut combined_img_mask = Tensor::<f32>::zeros(text_mask.shape());
    for (out, (img, prob)) in zip(
        combined_img_mask.iter_mut(),
        zip(img.iter(), text_mask.iter()),
    ) {
        // Convert normalized image pixel back to greyscale value in [0, 1]
        let in_grey = img + 0.5;
        let threshold = 0.2;
        let mask_pixel = if prob > threshold { 1. } else { 0. };
        *out = composite_pixel(mask_pixel, 0.5, in_grey, 1.);
    }
    let out_img = image_from_tensor(&combined_img_mask);

    let file = fs::File::create(output_path)?;
    let writer = BufWriter::new(file);

    let encoder = png::Encoder::new(writer, img_width as u32, img_height as u32);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&out_img)?;

    Ok(())
}
