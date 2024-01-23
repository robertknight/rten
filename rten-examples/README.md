# rten-examples

This crate contains example projects showing how to convert and run models for
common tasks across various modalities. See [Example
descriptions](#example-descriptions) for a summary of what each example does.

## Running an example

Each example has a `main` function with a comment above it describing the steps
to fetch the ONNX model, convert it to the format used by this library and run
the example.

The general steps to run an example are:

1. Download the ONNX model. These usually come from [Hugging
   Face](https://huggingface.co/docs/optimum/exporters/onnx/overview),
   the [ONNX Model Zoo](https://github.com/onnx/models) or pre-created ONNX
   models by the model authors.
2. Convert the ONNX model to this library's format using the `convert-onnx.py`
   script:

   ```sh
   $ ../tools/convert-onnx.py <onnx_model> <output_model>
   ```

3. Run the example using:

   ```sh
   $ cargo run -r --bin <example_name> <model_path> <...args>
   ```

   Where `...args` refers to the example-specific arguments, such as input data.
   The syntax and flags for an individual example can be displayed using its
   `--help` command:

   Note the `-r` flag to create a release build. This is required as the
   examples will run very slowly in debug builds.

   ```sh
   $ cargo run -r --bin <example_name> -- --help
   ```

   Note the `--` before `--help`. Without this `cargo` will print its own help
   info.

## Example descriptions

The examples have been chosen to cover common tasks and popular models.

### Vision

- **imagenet** - Classification of images using models trained on ImageNet.
  This example works with a wide variety of models, such as ResNet, MobileNet,
  ConvNeXt, ViT.
- **deeplab** - Semantic segmentation of images using [DeepLabv3](https://arxiv.org/abs/1706.05587)
- **depth_anything** - Monocular depth estimation using [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
- **detr** - Object detection using [DETR](https://research.facebook.com/publications/end-to-end-object-detection-with-transformers/)
- **yolo** - Object detection using [YOLO v8](https://github.com/ultralytics/ultralytics)

### Text

- **bert_qa** - Extractive question answering using
[BERT](https://arxiv.org/abs/1810.04805)-based models which have been fine-tuned
on the [SQuAD](https://paperswithcode.com/dataset/squad) dataset
- **jina_similarity** - Sentence similarity using vector embeddings of sentences

### Audio

- **wav2vec2** - Speech recognition of .wav audio using [wav2vec2](https://arxiv.org/abs/2006.11477)
