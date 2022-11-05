import { init as initWasnn, binaryName } from "./node_modules/wasnn/index.js";

import { ImageClassifier } from "./image-classifier.js";
import { IMAGENET_CLASSES } from "./imagenet-classes.js";

/**
 * Fetch a binary file from `url`.
 *
 * @param {string} url
 * @return {Promise<Uint8Array>}
 */
async function fetchBinary(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}`);
  }
  const buffer = await response.arrayBuffer();
  return new Uint8Array(buffer);
}

/**
 * Extract the pixel data from an ImageBitmap.
 *
 * @param {ImageBitmap} bitmap
 * @return {ImageData}
 */
function imageDataFromBitmap(bitmap) {
  let canvas;
  if (typeof OffscreenCanvas !== "undefined") {
    canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  } else if (typeof HTMLCanvasElement !== "undefined") {
    const canvasEl = document.createElement("canvas");
    canvasEl.width = bitmap.width;
    canvasEl.height = bitmap.height;
    canvas = canvasEl;
  } else {
    throw new Error("No canvas implementation available");
  }

  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height);
  return context.getImageData(0, 0, bitmap.width, bitmap.height);
}

/**
 * Initialize an image classifier using the Wasnn engine and MobileNet v2
 * model.
 */
async function createClassifier() {
  // Fetch the Wasnn engine and MobileNet model in parallel.
  const [, modelData] = await Promise.all([
    fetchBinary("./node_modules/wasnn/dist/" + binaryName()).then(initWasnn),
    fetchBinary("./mobilenet.model"),
  ]);

  // Initialize the classifier. This must be done after Wasnn is initialized.
  return new ImageClassifier(modelData);
}

async function init() {
  // Start to initialize the classifier pre-emptively, before an image is
  // selected. This reduces the delay for the user after the initial selection.
  const classifierPromise = createClassifier();

  const fileInput = document.querySelector("#file");
  const resultList = document.querySelector("#result-list");
  const statusInfo = document.querySelector("#status");

  fileInput.onchange = async () => {
    const bitmap = await createImageBitmap(fileInput.files[0], {
      // Resize image to input dimensions expected by model.
      resizeWidth: 224,
      resizeHeight: 224,
    });

    statusInfo.textContent = 'Downloading brain...';
    const classifier = await classifierPromise;

    statusInfo.textContent = 'Thinking...';
    const imageData = imageDataFromBitmap(bitmap);
    const classes = classifier.classify(imageData);

    statusInfo.textContent = 'Things that may be in this image:';

    resultList.innerHTML = "";
    const listItems = classes.map(([classIndex, score]) => {
      const item = document.createElement("li");
      item.textContent = IMAGENET_CLASSES[classIndex];
      return item;
    });
    resultList.append(...listItems);
  };
}

init().catch((err) => {
  console.error("Error initializing classifier:", err);
});
