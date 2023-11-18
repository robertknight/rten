use std::error::Error;
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
    let layout = input_img.sample_layout();

    let chw_tensor = NdTensorView::from_slice(
        input_img.as_raw().as_slice(),
        [height as usize, width as usize, 3],
        Some([
            layout.height_stride,
            layout.width_stride,
            layout.channel_stride,
        ]),
    )?
    .permuted([2, 0, 1]) // HWC => CHW
    .to_tensor() // Make tensor contiguous, which makes `map` faster
    .map(|x| *x as f32 / 255.); // Rescale from [0, 255] to [0, 1]

    Ok(chw_tensor)
}

/// Write a CHW image to an image file in `path`.
pub fn write_image(path: &str, img: NdTensorView<f32, 3>) -> Result<(), Box<dyn Error>> {
    let [channels, height, width] = img.shape();
    let color_type = match channels {
        1 => image::ColorType::L8,
        3 => image::ColorType::Rgb8,
        4 => image::ColorType::Rgba8,
        _ => return Err("Unsupported channel count".into()),
    };

    let hwc_img = img
        .permuted([1, 2, 0]) // CHW => HWC
        .map(|x| (x.clamp(0., 1.) * 255.0) as u8)
        .to_tensor();

    image::save_buffer(
        path,
        hwc_img.data().unwrap(),
        width as u32,
        height as u32,
        color_type,
    )?;

    Ok(())
}
