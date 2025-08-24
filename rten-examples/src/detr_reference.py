# Reference implementation of RT-DETR based on example in
# https://huggingface.co/docs/transformers/en/model_doc/rt_detr.

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")

image_path = Path(__file__).parent / Path("../../tools/test-images/sofa-cats.jpg")
image = Image.open(image_path)
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs, target_sizes=[(image.height, image.width)], threshold=0.5
)

for result in results:
    for score, label_id, box in zip(
        result["scores"], result["labels"], result["boxes"]
    ):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
