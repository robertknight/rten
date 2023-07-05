import { readFile } from "fs/promises";

import { program } from "commander";
import sharp from "sharp";

import {
  OcrEngine,
  OcrEngineInit,
  default as initOcrLib,
} from "../../dist/wasnn_ocr.js";

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

/**
 * Convert a list-like object returned by the OCR library into an iterator
 * that can be used with `for ... of` or `Array.from`.
 */
function* listItems(list) {
  for (let i = 0; i < list.length; i++) {
    yield list.item(i);
  }
}

/**
 * Perform OCR on an image and return the result as a JSON-serialiable object.
 *
 * @param {OcrEngine} ocrEngine
 * @param {OcrInput} ocrInput
 */
function generateJSON(ocrEngine, ocrInput) {
  const textLines = ocrEngine.getTextLines(ocrInput);
  const lines = Array.from(listItems(textLines)).map((line) => {
    const words = Array.from(listItems(line.words())).map((word) => {
      return {
        text: word.text(),
        rect: Array.from(word.rotatedRect().boundingRect()),
      };
    });

    return {
      text: line.text(),
      words,
    };
  });
  return {
    lines,
  };
}

program
  .name("ocr")
  .argument("<detection_model>", "Text detection model path")
  .argument("<recognition_model>", "Text recognition model path")
  .argument("<image>", "Input image path")
  .option("-j, --json", "Output JSON")
  .action(
    async (detectionModelPath, recognitionModelPath, imagePath, options) => {
      // Concurrently load the OCR library, text detection and recognition models,
      // and input image.
      const [_, detectionModel, recognitionModel, image] = await Promise.all([
        readFile("dist/wasnn_ocr_bg.wasm").then(initOcrLib),
        readFile(detectionModelPath).then((data) => new Uint8Array(data)),
        readFile(recognitionModelPath).then((data) => new Uint8Array(data)),
        loadImage(imagePath),
      ]);

      const ocrInit = new OcrEngineInit();
      ocrInit.setDetectionModel(detectionModel);
      ocrInit.setRecognitionModel(recognitionModel);

      const ocrEngine = new OcrEngine(ocrInit);
      const ocrInput = ocrEngine.loadImage(
        image.width,
        image.height,
        image.data
      );

      if (options.json) {
        const json = generateJSON(ocrEngine, ocrInput);
        console.log(JSON.stringify(json, null, 2));
      } else {
        const text = ocrEngine.getText(ocrInput);
        console.log(text);
      }
    }
  )
  .parse();
