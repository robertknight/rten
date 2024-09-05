use rten_tensor::prelude::*;
use rten_tensor::NdTensorViewMut;

/// Standard ImageNet normalization mean values, for use with
/// [`normalize_image`].
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// Standard ImageNet normalization standard deviation values, for use with
/// [`normalize_image`].
pub const IMAGENET_STD_DEV: [f32; 3] = [0.229, 0.224, 0.225];

/// Normalize the mean and standard deviation of all pixels in an image.
///
/// `img` should be a CHW tensor with `C` channels. For each channel `c`, the
/// output pixel values are computed as `y = (x - mean[c]) / std_dev[c]`.
///
/// This is a common preprocessing step for inputs to machine learning models.
/// Many models use standard "ImageNet" constants ([`IMAGENET_MEAN`],
/// [`IMAGENET_STD_DEV`]), but check the expected values for the model you are
/// using.
pub fn normalize_image<const C: usize>(
    mut img: NdTensorViewMut<f32, 3>,
    mean: [f32; C],
    std_dev: [f32; C],
) {
    let n_chans = img.size(0);
    assert_eq!(
        n_chans, C,
        "expected image to have {} channels but found {}",
        C, n_chans
    );

    for chan in 0..n_chans {
        let inv_std_dev = 1. / std_dev[chan];
        img.slice_mut::<2, _>(chan)
            .apply(|x| (x - mean[chan]) * inv_std_dev);
    }
}
