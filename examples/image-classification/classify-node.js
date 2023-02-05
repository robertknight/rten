import { readFileSync } from "fs";

import sharp from "sharp";
import { initSync, binaryName } from "wasnn";

import { ImageClassifier } from "./image-classifier.js";
import { IMAGENET_CLASSES } from "./imagenet-classes.js";

/**
 * Load a JPEG or PNG image from `path`, resize it to `width`x`height` and
 * return the RGB image data as an `ImageData`-like object.
 */
async function loadImage(path, width, height) {
  const image = await sharp(path)
    .removeAlpha()
    .resize(width, height, { fit: "fill" });
  return {
    data: new Uint8Array(await image.raw().toBuffer()),
    width,
    height,
  };
}

const path = process.argv[2];

// Initialize Wasnn.
const wasnnBinary = readFileSync("node_modules/wasnn/dist/" + binaryName());
initSync(wasnnBinary);

// Load the MobileNet classification model.
const modelData = new Uint8Array(readFileSync("./mobilenet.model"));
const classifier = new ImageClassifier(modelData);
const { width, height } = classifier.inputSize();
const image = await loadImage(path, width, height);
const top5 = classifier.classify(image);

const topCategories = top5.map(
  ([classIndex, score]) => IMAGENET_CLASSES[classIndex]
);

console.log("Most likely categories:");
for (let category of topCategories) {
  console.log("  - " + category);
}
