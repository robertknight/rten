from argparse import ArgumentParser

import torch
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

# Load model
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights)
model.eval()

# Load transforms. These will resize the input to what the model expects.
preprocess = weights.transforms()

# Generate a random input and resize it.
img = torch.rand((3, 480, 640))
batch = preprocess(img).unsqueeze(0)

parser = ArgumentParser()
parser.add_argument("-f", "--filename", default="deeplab.onnx")
parser.add_argument(
    "--dynamo", action="store_true", help="Use TorchDynamo-based exporter"
)
args = parser.parse_args()

if args.dynamo:
    print("Exporting model using TorchDynamo...")
    onnx_prog = torch.onnx.export(
        model,
        args=(batch),
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
    )
    onnx_prog.optimize()
    onnx_prog.save(args.filename)
else:
    print("Exporting model using TorchScript...")
    torch.onnx.export(
        model,
        args=(batch),
        f=args.filename,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
    )
