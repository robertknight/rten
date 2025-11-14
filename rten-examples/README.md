# rten-examples

This crate contains example projects showing how to convert and run models for
common tasks across various modalities. See [Example
descriptions](#example-descriptions) for a summary of what each example does.

## ONNX format support

The example instructions use models in `.onnx` format. This is only supported
by rten v0.23 and later. To run examples with earlier versions of rten, you
will need to convert the models to `.rten` format first:

```sh
$ pip install rten-convert
$ rten-convert model.onnx
```

This will create a `model.rten` file in the same directory as the model.

## Running an example

Each example has a `main` function with a comment above it describing the steps
to fetch the ONNX model and run the example.

The general steps to run an example are:

1. Download the ONNX model. These usually come from [Hugging
   Face](https://huggingface.co/docs/optimum/exporters/onnx/overview),
   or ONNX format models pre-created by the model authors or community members.

2. Run the example from the `rten-examples` directory using:

   ```sh
   $ cargo run -r --bin <example_name> -- <...args>
   ```

   Where `...args` refers to the example-specific arguments, such as model
   path(s), tokenizer configuration and input data.

   Note the `-r` flag to create a **release build**. This is required as the
   examples will run very slowly in debug builds.

   The syntax and flags for an individual example can be displayed using its
   `--help` command:

   ```sh
   $ cargo run -r --bin <example_name> -- --help
   ```

   Note the `--` before `--help`. Without this `cargo` will print its own help
   info.

   You can also run examples from the root of rten repository by specifying
   the rten-examples package name:

   ```sh
   $ cargo run -p rten-examples -r --bin <example_name> -- <...args>
   ```

## Reference implementations

Some of the examples have reference implementations in Python using PyTorch and
[Transformers](https://github.com/huggingface/transformers). These are found in
`src/{example_name}_reference.py` and enable comparison of RTen outputs with the
original models.

## Example descriptions

The examples have been chosen to cover common tasks and popular models.

### Vision

- **clip** - Match images against text descriptions using [CLIP](https://github.com/openai/CLIP)
- **imagenet** - Classification of images using models trained on ImageNet.
  This example works with a wide variety of models, such as ResNet, MobileNet,
  ConvNeXt, ViT.
- **deeplab** - Semantic segmentation of images using [DeepLabv3](https://arxiv.org/abs/1706.05587)
- **depth_anything** - Monocular depth estimation using [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
- **detr** - Object detection using [DETR](https://research.facebook.com/publications/end-to-end-object-detection-with-transformers/)
  or [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (Real-time DETR).
- **distilvit** - Image captioning using [Mozilla's DistilViT](https://hacks.mozilla.org/2024/05/experimenting-with-local-alt-text-generation-in-firefox-nightly/)
- **nougat** - Extract text from academic PDFs as Markdown using [Nougat](https://github.com/facebookresearch/nougat/)
- **rmbg** - Background removal using [BRIA Background Removal](https://huggingface.co/briaai/RMBG-1.4)
- **segment_anything** - Image segmentation using [Segment Anything](https://segment-anything.com)
- **trocr** - Recognize text using [TrOCR](https://arxiv.org/abs/2109.10282)
- **yolo** - Object detection using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

### Text

- **bert_qa** - Extractive question answering using
  [BERT](https://arxiv.org/abs/1810.04805)-based models which have been fine-tuned
  on the [SQuAD](https://paperswithcode.com/dataset/squad) dataset
- **gpt2** - Text generation using the [GPT-2](https://openai.com/index/better-language-models/)
  language model.
- **jina_similarity** - Sentence similarity using vector embeddings of sentences
- **llama** - Chatbot using [Llama 3](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **modernbert** - Masked word prediction using [ModernBERT](https://huggingface.co/blog/modernbert). Also works with the base version of the original BERT model.
- **qwen2_chat** - Chatbot using [Qwen2](https://github.com/QwenLM/Qwen2). Also works with some other
  chat models that use the same prompt format such as [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

### Audio

- **byt5_g2p** - Convert text to phonemes using [ByT5](https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak).
  The phonemes can be used as inputs for the **piper** example.
- **kokoro** - Text-to-speech using [Kokoro](https://github.com/hexgrad/kokoro).
- **piper** - Text-to-speech using [Piper](https://github.com/rhasspy/piper) models
- **silero** - Speech detection using [Silero VAD](https://github.com/snakers4/silero-vad)
- **wav2vec2** - Speech recognition of .wav audio using [wav2vec2](https://arxiv.org/abs/2006.11477)
- **whisper** - Speech recognition of .wav audio using [OpenAI's Whisper](https://github.com/openai/whisper)
