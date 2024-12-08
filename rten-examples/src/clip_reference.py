from argparse import ArgumentParser

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

parser = ArgumentParser(description="Reference implementation for the CLIP example.")
parser.add_argument("-i", "--image", type=str, action="append", help="Path to image")
parser.add_argument("-c", "--caption", type=str, action="append", help="Text caption")
parser.add_argument("-t", "--tokens", action="store_true", help="Print text token IDs")
args = parser.parse_args()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True
)

images = [Image.open(img_path) for img_path in args.image]

inputs = processor(text=args.caption, images=images, return_tensors="pt", padding=True)
if args.tokens:
    print("Tokens", inputs["input_ids"])

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)

for img_idx, img_path in enumerate(args.image):
    for cap_idx, caption in enumerate(args.caption):
        prob = probs[img_idx, cap_idx]
        print(f'image "{img_path}" caption "{caption}" probability {prob:.2f}')
