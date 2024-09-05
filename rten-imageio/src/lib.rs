//! Provides utilities for loading, saving and preprocessing images for use with
//! [RTen](https://github.com/robertknight/rten).
//!
//! The APIs are limited to keep them simple for the most common use cases.
//! If you need more flexibility from a function, copy and adjust the
//! implementation.

use std::error::Error;
use std::path::Path;

use rten_tensor::errors::FromDataError;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

/// Errors reported when creating a tensor from an image.
#[derive(Debug)]
pub enum ReadImageError {
    /// The image could not be loaded.
    ImageError(image::ImageError),
    /// The loaded image could not be converted to a tensor.
    ConvertError(FromDataError),
}

impl std::fmt::Display for ReadImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadImageError::ImageError(e) => write!(f, "failed to read image: {}", e),
            ReadImageError::ConvertError(e) => write!(f, "failed to create tensor: {}", e),
        }
    }
}

impl Error for ReadImageError {}

/// Convert an image into a CHW tensor with 3 channels and values in the range
/// [0, 1].
pub fn image_to_tensor(image: image::DynamicImage) -> Result<NdTensor<f32, 3>, ReadImageError> {
    let image = image.into_rgb8();
    let (width, height) = image.dimensions();
    let layout = image.sample_layout();

    let chw_tensor = NdTensorView::from_data_with_strides(
        [height as usize, width as usize, 3],
        image.as_raw().as_slice(),
        [
            layout.height_stride,
            layout.width_stride,
            layout.channel_stride,
        ],
    )
    .map_err(ReadImageError::ConvertError)?
    .permuted([2, 0, 1]) // HWC => CHW
    .map(|x| *x as f32 / 255.); // Rescale from [0, 255] to [0, 1]

    Ok(chw_tensor)
}

/// Read an image from a file into a CHW tensor.
///
/// To load an image from a byte buffer or other source, use [`image::open`]
/// and pass the result to [`image_to_tensor`].
pub fn read_image<P: AsRef<Path>>(path: P) -> Result<NdTensor<f32, 3>, ReadImageError> {
    image::open(path)
        .map_err(ReadImageError::ImageError)
        .and_then(image_to_tensor)
}

/// Errors returned when writing a tensor to an image.
#[derive(Debug)]
pub enum WriteImageError {
    /// The number of channels in the image tensor is unsupported.
    UnsupportedChannelCount,
    /// The image could not be written.
    ImageError(image::ImageError),
}

impl std::fmt::Display for WriteImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageError(e) => write!(f, "failed to write image: {}", e),
            Self::UnsupportedChannelCount => write!(f, "image has unsupported number of channels"),
        }
    }
}

impl Error for WriteImageError {}

/// Convert a CHW tensor to an image and write it to a PNG file.
pub fn write_image(path: &str, img: NdTensorView<f32, 3>) -> Result<(), WriteImageError> {
    let [channels, height, width] = img.shape();
    let color_type = match channels {
        1 => image::ColorType::L8,
        3 => image::ColorType::Rgb8,
        4 => image::ColorType::Rgba8,
        _ => return Err(WriteImageError::UnsupportedChannelCount),
    };

    let hwc_img = img
        .permuted([1, 2, 0]) // CHW => HWC
        .map(|x| (x.clamp(0., 1.) * 255.0) as u8);

    image::save_buffer(
        path,
        hwc_img.data().unwrap(),
        width as u32,
        height as u32,
        color_type,
    )
    .map_err(WriteImageError::ImageError)?;

    Ok(())
}
