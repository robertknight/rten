"""Classify an image using an ImageNet model and RTen.

Downloads MobileNetV2 from the ONNX model zoo, the ImageNet class labels and a
test image (unless a path to an image is given), then prints the most likely
classes for the image.

Build the Python bindings with `just build-py`, then run this example from the
root of the repository with:

    uv run --project rten-py --no-sync --with pillow python rten-py/examples/classify_image.py
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import rten

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"

# Directory to store downloaded files in.
CACHE_DIR = Path.home() / ".cache" / "rten-examples"

# Per-channel normalization constants used when training ImageNet models.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download(url: str, dir: Path) -> Path:
    """Download `url` into `dir`, unless it was downloaded previously."""
    path = dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        print(f"Downloading {url}")
        dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.rename(path)
    return path


def preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
    """Convert an image to a normalized NCHW float32 array.

    The image is resized so that its shorter side is `size / 0.875` pixels,
    then center-cropped to `size x size`, matching the standard ImageNet
    evaluation transform.
    """
    image = image.convert("RGB")
    scale = (size / 0.875) / min(image.size)
    width, height = round(image.width * scale), round(image.height * scale)
    image = image.resize((width, height), Image.Resampling.BILINEAR)

    left, top = (width - size) // 2, (height - size) // 2
    image = image.crop((left, top, left + size, top + size))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return pixels.transpose(2, 0, 1)[np.newaxis]


def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max())
    return exp / exp.sum()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", help="Image to classify")
    args = parser.parse_args()

    model_path = download(MODEL_URL, CACHE_DIR)
    labels_path = download(LABELS_URL, CACHE_DIR)
    image_path = Path(args.image) if args.image else download(IMAGE_URL, CACHE_DIR)

    labels = labels_path.read_text().splitlines()
    model = rten.Model(model_path)
    (input_info,) = model.get_inputs()

    input = preprocess(Image.open(image_path))
    (logits,) = model.run(None, {input_info.name: input})
    probs = softmax(logits[0])

    print(f"Predictions for {image_path}:")
    for cls in np.argsort(probs)[::-1][:5]:
        print(f"  {labels[cls]}: {probs[cls]:.3f}")


if __name__ == "__main__":
    main()
