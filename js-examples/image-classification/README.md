# MobileNet image classification

This demo classifies the content of images using MobileNet v2. The known
categories are listed in `imagenet-classes.js`. The model works best when
there is an obvious central subject in the image, and the subject is a common
kind of object.

## Setup

1. Build the main RTen project for WebAssembly. See the README.md file at the
   root of the repository.
2. In this directory, run `npm install`
3. Download the ONNX MobileNet model from the ONNX Model Zoo and convert it
   to `.rten` format:

   ```sh
   curl -L https://github.com/onnx/models/raw/main/Computer_Vision/mobilenetv2_110d_Opset18_timm/mobilenetv2_110d_Opset18.onnx -o mobilenet.onnx

   rten-convert mobilenet.onnx mobilenet.rten
   ```
4. Follow either of the subsections below to run the example in Node or the
   browser

## Running in Node

```sh
$ node classify-node.js espresso.png

# Example output
Most likely categories:
  - espresso
  - chocolate sauce, chocolate syrup
  - cup
  - ice cream, icecream
  - plate
```

## Running in a browser

1. Start a web server:

   ```
   python -m http.server 3010
   ```

2. Open http://localhost:3010/
3. Click "Choose file" and select a photo or image to classify
