use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use rten::{Dimension, FloatOperators, Model, Operators};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

#[derive(Clone, Copy, PartialEq)]
enum PixelNorm {
    /// Do not apply any per-pixel normalization, other than converting u8
    /// pixel values in [0, 255] to floats in [0-1].
    NoNorm,

    /// Apply the standard ImageNet normalization `(input - mean) / std_dev`
    /// where `mean` and `std_dev` are per-channel constants.
    ImageNetNorm,
}

#[derive(Clone, Copy, PartialEq)]
enum ChannelOrder {
    Rgb,
    Bgr,
}

#[derive(Clone, Copy, PartialEq)]
enum DimOrder {
    /// Use "channels-first" order
    Nchw,
    /// Use "channels-last" order
    Nhwc,
}

/// Specifies the input format that the model expects.
struct InputConfig {
    /// Normalization for input pixels.
    norm: PixelNorm,

    /// Expected order of channels within the channel dimension.
    chan_order: ChannelOrder,

    /// Expected order of dimensions in the input tensor.
    dim_order: DimOrder,

    /// Default input width if the model does not specify a fixed width.
    default_width: u16,

    /// Default input height if the model does not specify a fixed height.
    default_height: u16,
}

/// Read an image from `path` into an NCHW or NHWC tensor, depending on
/// `out_dim_order`.
fn read_image<N: Fn(usize, f32) -> f32>(
    path: &str,
    normalize_pixel: N,
    out_chan_order: ChannelOrder,
    out_dim_order: DimOrder,
    out_height: u32,
    out_width: u32,
) -> Result<Tensor<f32>, Box<dyn Error>> {
    let input_img = image::open(path)?;
    let input_img = input_img.into_rgb8();

    // Resize the image using the `imageops::resize` function from the `image`
    // crate rather than using RTen's `resize` operator because
    // `imageops::resize` supports antialiasing. This significantly improves
    // output image quality and thus prediction accuracy when the output is
    // small (eg. 224 or 256px).
    //
    // The outputs of `imageops::resize` still don't match PyTorch exactly
    // though, which can lead to small differences in prediction outputs.
    let input_img = image::imageops::resize(
        &input_img,
        out_width,
        out_height,
        image::imageops::FilterType::Triangle,
    );

    let (width, height) = input_img.dimensions();

    // Map input channel index, in RGB order, to output channel index
    let out_chans = match out_chan_order {
        ChannelOrder::Rgb => [0, 1, 2],
        ChannelOrder::Bgr => [2, 1, 0],
    };

    let mut img_tensor = Tensor::zeros(&[1, 3, height as usize, width as usize]);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let pixel_value = input_img.get_pixel(x, y)[c] as f32 / 255.0;
                let in_val = normalize_pixel(c, pixel_value);
                img_tensor[[0, out_chans[c], y as usize, x as usize]] = in_val;
            }
        }
    }

    if out_dim_order == DimOrder::Nhwc {
        // NCHW => NHWC
        img_tensor.permute(&[0, 3, 2, 1]);
    }

    Ok(img_tensor)
}

fn resource_path(path: &str) -> PathBuf {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push("data/");
    abs_path.push(path);
    abs_path
}

// Config for MobileViT model exported from https://huggingface.co/apple/mobilevit-small
const MOBILEVIT_CONFIG: InputConfig = InputConfig {
    chan_order: ChannelOrder::Bgr,
    norm: PixelNorm::NoNorm,
    dim_order: DimOrder::Nchw,
    default_width: 256,
    default_height: 256,
};

// Config for EfficientNet model from ONNX Model Zoo.
//
// See https://github.com/onnx/models/tree/main/validated/vision/classification/efficientnet-lite4
const EFFICIENTNET_CONFIG: InputConfig = InputConfig {
    chan_order: ChannelOrder::Rgb,
    norm: PixelNorm::ImageNetNorm,
    dim_order: DimOrder::Nhwc,
    default_width: 224,
    default_height: 224,
};

// Config for MobileNet model from ONNX Model Zoo.
//
// This also works with MobileNet v3 exported from torchvision.
//
// See https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet
const MOBILENET_CONFIG: InputConfig = InputConfig {
    chan_order: ChannelOrder::Rgb,
    norm: PixelNorm::ImageNetNorm,
    dim_order: DimOrder::Nchw,
    default_width: 224,
    default_height: 224,
};

struct Args {
    config: InputConfig,
    model: String,
    image: String,
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
                    "Classify images.

Usage: {bin_name} <config> <model> <image>

Where config is one of:

  - efficientnet
  - mobilenet
  - mobilevit
",
                    bin_name = parser.bin_name().unwrap_or("imagenet")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let config = values.pop_front().ok_or("missing `config` arg")?;
    let config = match config.as_str() {
        "efficientnet" => EFFICIENTNET_CONFIG,
        "mobilenet" => MOBILENET_CONFIG,
        "mobilevit" => MOBILEVIT_CONFIG,
        _ => {
            return Err(format!("Unknown config {config}").into());
        }
    };

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let image = values.pop_front().ok_or("missing `image` arg")?;

    let args = Args {
        config,
        image,
        model,
    };

    Ok(args)
}

/// This example loads an image and uses a classification model such as
/// MobileNet to classify it into one of the 1000 ImageNet classes.
///
/// Steps to run:
///
/// 1. Download an ImageNet image classification ONNX model. See the `_CONFIG`
///    constants above for available models and links.
///
/// 2. Convert the model to .rten format using:
///
///    rten-convert mobilenet.onnx
///
/// 3. Run this example specifying the model type, model and path to an image:
///
///    ```
///    cargo run --release --bin imagenet mobilenet mobilenet.rten image.jpg
///    ```
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.model)?;

    let normalize_pixel = match args.config.norm {
        PixelNorm::NoNorm => |_, value| value,
        PixelNorm::ImageNetNorm => |chan, value| {
            let imagenet_mean = &[0.485, 0.456, 0.406];
            let imagenet_std_dev = &[0.229, 0.224, 0.225];
            (value - imagenet_mean[chan]) / imagenet_std_dev[chan]
        },
    };

    let input_id = model
        .input_ids()
        .first()
        .copied()
        .ok_or("model has no inputs")?;
    let input_shape = model
        .node_info(input_id)
        .and_then(|info| info.shape())
        .ok_or("model does not specify expected input shape")?;
    let (in_height, in_width) = match &input_shape[..] {
        [_, _, h, w] => {
            let h = if let Dimension::Fixed(h) = h {
                *h
            } else {
                args.config.default_height as usize
            };
            let w = if let Dimension::Fixed(w) = w {
                *w
            } else {
                args.config.default_width as usize
            };
            (h, w)
        }
        _ => {
            return Err("failed to get model dims".into());
        }
    };

    let img_tensor = read_image(
        &args.image,
        normalize_pixel,
        args.config.chan_order,
        args.config.dim_order,
        in_height as u32,
        in_width as u32,
    )?;

    let logits: NdTensor<f32, 2> = model.run_one(img_tensor.view().into(), None)?.try_into()?;

    let (top_probs, top_classes) = logits
        .softmax(-1)?
        .topk(5, None, true /* largest */, true /* sorted */)?;

    // Determine which class label mapping to use based on the number of
    // labels in the predictions.
    let n_classes = logits.size(1);
    let index_to_wordnet_id_file = match n_classes {
        1000 => Some(resource_path("imagenet_synsets.txt")),
        21841 => Some(resource_path("imagenet22k_synsets.txt")),
        _ => None,
    };

    let labels = if let Some(index_to_wordnet_id_file) = index_to_wordnet_id_file {
        let labels_path = resource_path("imagenet_synset_to_lemma.txt");
        Some(ImageNetLabels::read(
            index_to_wordnet_id_file.as_path(),
            labels_path.as_path(),
        )?)
    } else {
        println!(
            "Unable to determine class labels for model with {} classes",
            n_classes
        );
        None
    };

    println!("Top classes:");
    for (&cls, &score) in top_classes.iter().zip(top_probs.iter()) {
        let label = labels
            .as_ref()
            .and_then(|labels| labels.label_for_index(cls as usize))
            .unwrap_or("unknown");
        println!("  {} ({}) ({})", label, cls, score);
    }

    Ok(())
}

/// Class label map for ImageNet.
///
/// Data files for the most common ImageNet variants are included in the
/// examples/data/ directory. Additional labels can be found in
/// <https://github.com/huggingface/pytorch-image-models/tree/main/timm/data/_info>.
struct ImageNetLabels {
    /// Map of class index to WordNet synset ID.
    wordnet_ids: Vec<String>,

    /// Map of WordNet synset ID to label.
    wordnet_id_to_label: HashMap<String, String>,
}

impl ImageNetLabels {
    /// Load ImageNet class mappings and labels.
    ///
    /// `id_mapping_file` is a file containing a list of WordNet synset IDs, one
    /// per line. The line index in this file is treated as a class index.
    /// `label_file` is a file containing lines of the form
    /// `[synset_id]<TAB>[definition]`.
    fn read(id_mapping_file: &Path, label_file: &Path) -> Result<ImageNetLabels, Box<dyn Error>> {
        let id_file = File::open(id_mapping_file)?;
        let id_reader = BufReader::new(id_file);

        let mut wordnet_ids = Vec::new();
        for id in id_reader.lines() {
            let id = id?;
            wordnet_ids.push(id);
        }

        let label_file = File::open(label_file)?;
        let label_reader = BufReader::new(label_file);

        let mut wordnet_id_to_label = HashMap::new();
        for line in label_reader.lines() {
            let line = line?;
            if let Some((wordnet_id, label)) = line.split_once('\t') {
                wordnet_id_to_label.insert(wordnet_id.into(), label.into());
            }
        }

        Ok(ImageNetLabels {
            wordnet_ids,
            wordnet_id_to_label,
        })
    }

    /// Return the label for a given class index or `None` if the class index is
    /// out of bounds for the ID mapping file used when `self` was constructed.
    fn label_for_index(&self, idx: usize) -> Option<&str> {
        let wordnet_id = self.wordnet_ids.get(idx)?;
        self.wordnet_id_to_label.get(wordnet_id).map(|x| x.as_str())
    }
}
