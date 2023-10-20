use std::error::Error;
use std::fs;
use std::io::BufWriter;
use std::iter::zip;

use wasnn_tensor::prelude::*;
use wasnn_tensor::{NdTensor, NdTensorView, NdTensorViewMut};

/// Apply standard ImageNet normalization to a pixel value.
/// See https://huggingface.co/facebook/detr-resnet-50#preprocessing.
pub fn normalize_pixel(value: f32, channel: usize) -> f32 {
    assert!(channel < 3, "channel index is invalid");
    let imagenet_mean = [0.485, 0.456, 0.406];
    let imagenet_std_dev = [0.229, 0.224, 0.225];
    (value - imagenet_mean[channel]) / imagenet_std_dev[channel]
}

/// Apply standard ImageNet normalization to all pixel values in an image.
pub fn normalize_image(mut img: NdTensorViewMut<f32, 3>) {
    for ([chan, _y, _x], pixel) in zip(img.indices(), img.iter_mut()) {
        *pixel = normalize_pixel(*pixel, chan);
    }
}

/// Read an image from `path` into a CHW tensor.
pub fn read_image(path: &str) -> Result<NdTensor<f32, 3>, Box<dyn Error>> {
    let input_img = image::open(path)?;
    let input_img = input_img.into_rgb8();

    let (width, height) = input_img.dimensions();

    let in_chans = 3;
    let mut float_img = NdTensor::zeros([in_chans, height as usize, width as usize]);
    for c in 0..in_chans {
        let mut chan_img = float_img.slice_mut([c]);
        for y in 0..height {
            for x in 0..width {
                let pixel_value = input_img.get_pixel(x, y)[c] as f32 / 255.0;
                chan_img[[y as usize, x as usize]] = pixel_value;
            }
        }
    }
    Ok(float_img)
}

/// Convert a CHW float tensor with values in the range [0, 1] to `Vec<u8>` with
/// values scaled to [0, 255].
fn image_from_tensor(tensor: NdTensorView<f32, 3>) -> Vec<u8> {
    tensor
        .iter()
        .map(|x| (x.clamp(0., 1.) * 255.0) as u8)
        .collect()
}

/// Write a CHW image to a PNG file in `path`.
pub fn write_image(path: &str, img: NdTensorView<f32, 3>) -> Result<(), Box<dyn Error>> {
    let img_width = img.size(2);
    let img_height = img.size(1);
    let color_type = match img.size(0) {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => return Err("Unsupported channel count".into()),
    };

    let hwc_img = img.permuted([1, 2, 0]); // CHW => HWC

    let out_img = image_from_tensor(hwc_img);
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, img_width as u32, img_height as u32);
    encoder.set_color(color_type);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&out_img)?;

    Ok(())
}
