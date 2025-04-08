# Reference inference for DeepLab example using ONNX Runtime.
#
# To use this, first export the DeepLab model then run inference:
#
# ```
# python export-deeplab.py
# python deeplab_reference.py deeplab.onnx path/to/test_image.jpeg
# ```
#
# This will produce an `out_reference.png` image containing the segmentation map.
from argparse import ArgumentParser

from PIL import Image
import numpy as np
import onnxruntime

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD_DEV = [0.229, 0.224, 0.225]

# Labels and colors for the different categories of object that DeepLabv3 can
# detect.
#
# For the labels, see https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt.
PASCAL_VOC_LABELS = [
    ("background", (0.0, 0.0, 0.0)),  # Black
    ("aeroplane", (0.0, 1.0, 0.0)),  # Green
    ("bicycle", (0.0, 0.0, 1.0)),  # Blue
    ("bird", (1.0, 1.0, 0.0)),  # Yellow
    ("boat", (1.0, 0.0, 1.0)),  # Magenta
    ("bottle", (0.0, 1.0, 1.0)),  # Cyan
    ("bus", (0.5, 0.0, 0.0)),  # Dark Red
    ("car", (0.0, 0.5, 0.0)),  # Dark Green
    ("cat", (0.0, 0.0, 0.5)),  # Dark Blue
    ("chair", (0.5, 0.5, 0.0)),  # Olive
    ("cow", (0.5, 0.0, 0.5)),  # Purple
    ("diningtable", (0.0, 0.5, 0.5)),  # Teal
    ("dog", (0.75, 0.75, 0.75)),  # Light Gray
    ("horse", (0.5, 0.5, 0.5)),  # Gray
    ("motorbike", (0.25, 0.25, 0.25)),  # Dark Gray
    ("person", (1.0, 0.5, 0.0)),  # Orange
    ("pottedplant", (0.5, 1.0, 0.5)),  # Pastel Green
    ("sheep", (0.5, 0.5, 1.0)),  # Pastel Blue
    ("sofa", (1.0, 0.75, 0.8)),  # Pink
    ("train", (0.64, 0.16, 0.16)),  # Brown
    ("tvmonitor", (1.0, 1.0, 1.0)),  # White
]

parser = ArgumentParser()
parser.add_argument("model", help="Path to DeepLab ONNX model")
parser.add_argument("image", help="Image to segment")
args = parser.parse_args()

session = onnxruntime.InferenceSession(args.model)

# Input image size expected by model
input_width = 693
input_height = 520

# Load image, normalize and convert to NHWC layout
image = Image.open(args.image)
image = image.resize([input_width, input_height])
image = np.asarray(image).astype("float32") / 255.0
image = np.transpose(image, (2, 0, 1))  # HWC => CHW

norm_mean = np.array(IMAGENET_MEAN, dtype="float32").reshape(-1, 1, 1)
norm_std_dev = np.array(IMAGENET_STD_DEV, dtype="float32").reshape(-1, 1, 1)
image = (image - norm_mean) / norm_std_dev
image = np.expand_dims(image, axis=0)  # Insert batch dim

# Segment image, producing an HW tensor containing the class index for each pixel.
seg_classes = session.run(["output"], {"input": image})[0]
seg_classes = np.transpose(seg_classes, (0, 2, 3, 1))  # (N,class,H,W) => (N,H,W,class)
seg_classes = np.argmax(seg_classes[0], axis=-1)

# Produce a segmentation map with pixels colored based on predicted class for
# each pixel.
out_height, out_width = seg_classes.shape
seg_map = np.zeros((out_height, out_width, 3), dtype="float32")
for cls_id, cls_info in enumerate(PASCAL_VOC_LABELS):
    cls_name, cls_color = cls_info
    cls_mask = seg_classes == cls_id
    for chan in range(3):
        seg_map[cls_mask, chan] = cls_color[chan]

out_im = Image.fromarray(np.uint8(seg_map * 255))
out_im.save("out_reference.png")
