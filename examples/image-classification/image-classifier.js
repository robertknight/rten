import { Model, Tensor, TensorList } from "./node_modules/wasnn/index.js";

/**
 * Convert an RGB or RGBA image loaded with `loadImage` into an NCHW
 * tensor ready for input into an ImageNet classification model, such as
 * MobileNet.
 *
 * @param {ImageData} image
 */
function tensorFromImage(image) {
  const { width, height, data } = image;
  const inChannels = data.length / (width * height);

  if (inChannels !== 3 && inChannels !== 4) {
    throw new Error("Input image is not an RGB or RGBA image");
  }

  const outChannels = 3;
  const outData = new Float32Array(height * width * outChannels);

  const shape = new Uint32Array(4);
  shape[0] = 1;
  shape[1] = outChannels;
  shape[2] = height;
  shape[3] = width;

  // Standard values for normalizing inputs to ImageNet models.
  const chanMeans = [0.485, 0.456, 0.406];
  const chanStdDev = [0.229, 0.224, 0.225];

  // The input image is a sequence of RGB bytes in HWC order. Convert it to an
  // NCHW tensor where each input is normalized using the standard per-channel
  // mean and standard deviation for ImageNet models.
  let inOffset = 0;

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      for (let channel = 0; channel < outChannels; channel++) {
        const pixel = data[inOffset + channel];
        outData[channel * (width * height) + row * width + col] =
          (pixel / 255 - chanMeans[channel]) / chanStdDev[channel];
      }
      inOffset += inChannels;
    }
  }

  return Tensor.floatTensor(shape, outData);
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
    try {
      this.model = new Model(modelData);
    } catch (err) {
      throw new Error(`Failed to load model: ${err}`);
    }
  }

  /**
   * Returns the expected size of input images for the model.
   *
   * @return {{ width: number|null, height: number|null }}
   */
  inputSize() {
    const inputIds = this.model.inputIds();
    if (inputIds.length < 1) {
      throw new Error("Model has no inputs");
    }
    const shape = this.model.nodeInfo(inputIds[0]).shape();
    const [width, height] = shape.slice(shape.length - 2);
    if (width < 0 || height < 0) {
      throw new Error("Model does not specify expected size");
    }
    return { width, height };
  }

  /**
   * Classify the content of an image.
   *
   * @param {ImageData} image - The input image. This should be an RGB image
   *   matching the size returned by {@link inputSize}.
   * @return {number[]} - Returns the 5 most likely ImageNet categories according
   *   to the model
   */
  classify(image) {
    const inputIds = this.model.inputIds();
    const outputIds = this.model.outputIds();
    const { width: expectedWidth, height: expectedHeight } = this.inputSize();

    if (
      (image.width !== null && image.width !== expectedWidth) ||
      (image.height !== null && image.height !== expectedHeight)
    ) {
      throw new Error("Image size does not match expected size");
    }

    const inputs = TensorList.from([tensorFromImage(image)]);
    const outputs = this.model.run(inputIds, inputs, outputIds);
    const output = outputs.item(0);

    // `scores` has shape [1, 1000] where the second dimension are the scores for each
    // ImageNet category.
    const scores = output.floatData();
    return topK(scores, 5);
  }
}
