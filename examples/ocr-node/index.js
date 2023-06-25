import { readFileSync } from "fs";

import sharp from "sharp";

import { OcrEngine, OcrEngineInit, initSync } from "../../dist/wasnn_ocr.js";

/**
 * Load a JPEG or PNG image from `path` and return the RGB image data as an
 * `ImageData`-like object.
 */
async function loadImage(path) {
  const image = await sharp(path);
  const { width, height } = await image.metadata();
  const data = await image.raw().toBuffer();
  return {
    data: new Uint8Array(data),
    width,
    height,
  };
}

async function saveImage(
  path,
  width,
  height,
  data,
  channels = data.length / (width * height)
) {
  const image = await sharp(data, {
    raw: { width, height, channels },
  });
  await image.toFile(path);
}

const detectionModelPath = process.argv[2];
const recognitionModelPath = process.argv[3];
const imagePath = process.argv[4];

const ocrBin = readFileSync("dist/wasnn_ocr_bg.wasm");
initSync(ocrBin);

const detectionModel = new Uint8Array(readFileSync(detectionModelPath));
const recognitionModel = new Uint8Array(readFileSync(recognitionModelPath));

const image = await loadImage(imagePath);

const ocrInit = new OcrEngineInit();
ocrInit.setDetectionModel(detectionModel);
ocrInit.setRecognitionModel(recognitionModel);

const ocrEngine = new OcrEngine(ocrInit);
const ocrInput = ocrEngine.loadImage(image.width, image.height, image.data);

const text = ocrEngine.getText(ocrInput);

console.log(text);
