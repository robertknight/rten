import { Model, TensorList } from "./node_modules/wasnn/index.js";

/**
 * Convert a 224x224 RGB or RGBA image loaded with `loadImage` into HWC tensor
 * data ready for input into an ImageNet classification model, such as MobileNet.
 *
 * @param {ImageData} image
 * @return {Float32Array}
 */
function tensorFromImage(image) {
  const { width, height, data } = image;
  const inChannels = data.length / (width * height);

  if (
    width !== 224 ||
    height !== 224 ||
    (inChannels !== 3 && inChannels !== 4)
  ) {
    throw new Error("Input image is not a 224x224 RGB or RGBA image");
  }

  const outChannels = 3;
  const tensor = new Float32Array(height * width * outChannels);

  const shape = new Uint32Array(4);
  shape[0] = 1;
  shape[1] = outChannels;
  shape[2] = height;
  shape[3] = width;

  // Standard values for normalizing inputs to ImageNet models.
  const chanMeans = [0.485, 0.456, 0.406];
  const chanStdDev = [0.229, 0.224, 0.225];

  // The input image is a sequence of RGB bytes in HWC order. Convert it to a
  // CHW tensor where each input is normalized using the standard per-channel
  // mean and standard deviation for ImageNet models.
  let inOffset = 0;

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      for (let channel = 0; channel < outChannels; channel++) {
        const pixel = data[inOffset + channel];
        tensor[channel * (width * height) + row * width + col] =
          (pixel / 255 - chanMeans[channel]) / chanStdDev[channel];
      }
      inOffset += inChannels;
    }
  }

  return { shape, tensor };
}

/**
 * Return `k` indices and values from `array` with the highest values.
 *
 * @param {Float32Array} array
 * @return {Array<[index: number, score: number]>}
 */
function topK(array, k) {
  return [...array.entries()]
    .sort(([i, valA], [j, valB]) => valB - valA)
    .slice(0, k);
}

/**
 * Classifies the content of images into the 1000 ImageNet categories (see
 * imagenet-classes.js) using a Wasnn model.
 */
export class ImageClassifier {
  /**
   * Initialize a classifier using a serialized Wasnn model.
   *
   * The Wasnn engine must be initialized before this method is called.
   *
   * @param {Uint8Array} modelData - Serialized Wasnn model
   */
  constructor(modelData) {
    this.model = new Model(modelData);
  }

  /**
   * Classify the content of an image.
   *
   * @param {ImageData} image - The input image. This should be a 224x224 RGB
   *   image.
   * @return {number[]} - Returns the 5 most likely ImageNet categories according
   *   to the model
   */
  classify(image) {
    const inputId = this.model.findNode("input");
    const outputId = this.model.findNode("output");

    const { shape: inputShape, tensor: inputData } = tensorFromImage(image);
    const inputs = new TensorList();
    inputs.push(inputShape, inputData);

    const outputs = this.model.run([inputId], inputs, [outputId]);

    // `scores` has shape [1, 1000] where the second dimension are the scores for each
    // ImageNet category.
    const scores = outputs.getData(0);
    return topK(scores, 5);
  }
}
