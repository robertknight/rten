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

# Export to ONNX
torch.onnx.export(
    model,
    args=(batch),
    f="deeplab.onnx",
    verbose=False,
    input_names=["input"],
    output_names=["output"],
)
