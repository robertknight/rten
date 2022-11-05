# MobileNet image classification

This demo classifies the content of images using MobileNet v2. The known
categories are listed in `imagenet-classes.js`. The model works best when
there is an obvious central subject in the image, and that subject is familiar
to the model.

## Setup

```
npm install
```

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
