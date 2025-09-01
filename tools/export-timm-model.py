#!/usr/bin/env python

from argparse import ArgumentParser
import os
from time import perf_counter

import onnxruntime as ort
from PIL import Image
import timm
import torch


def extract_config_name(config: str) -> str:
    """
    Extract the name of a timm model config from a name, path or URL.

    This accepts configs in several formats:

    - A config name that can be used with `timm.create_model`, eg.
      "mobilenetv3_large_100.ra_in1k"
    - A Hugging Face model URL, eg.
      "https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k"
    - The path of a Hugging Face model URL, eg.
      "timm/mobilenetv3_large_100.ra_in1k"
    """
    prefixes = ["timm/", "https://huggingface.co/timm/"]
    for prefix in prefixes:
        if config.startswith(prefix):
            config = config[len(prefix) :]
            break
    return config


def print_predictions(scores: torch.Tensor):
    """
    Display predictions from ImageNet classification model output.

    :param scores: (N, C) or (C,) tensor of ImageNet class scores
    """

    if scores.ndim > 1:
        scores = scores[0]  # Strip batch dim

    probs, classes = torch.topk(scores.softmax(dim=0) * 100, k=5)
    imagenet_info = timm.data.imagenet_info.ImageNetInfo("imagenet-1k")
    for i in range(probs.size(0)):
        cls = classes[i]
        prob = probs[i]
        label_id = imagenet_info.index_to_label_name(cls)
        label_desc = imagenet_info.label_name_to_description(label_id)
        print(f"  {label_desc} ({prob:.2f})")


def export_timm_model(
    config: str, onnx_path: str, dynamo: bool = False, save_tensors: bool = False
):
    """
    Export a PyTorch model from timm to ONNX.

    :param config: Name of the model configuration to pass to the
        `timm.create_model` function.
    :param onnx_path: Path of exported ONNX model
    :param save_tensors: Save input and output tensors
    """
    test_img_path = os.path.join(os.path.dirname(__file__), "test-images/horses.jpeg")
    img = Image.open(test_img_path)

    prefixes = ["timm/", "https://huggingface.co/timm/"]
    for prefix in prefixes:
        if config.startswith(prefix):
            config = config[prefix:]
            break

    print(f"Loading model {config}...")
    model = timm.create_model(config, pretrained=True)
    model = model.eval()

    # Get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Transform and add batch dimm
    input_img = transforms(img).unsqueeze(0)

    # Make sure model works in PyTorch
    print(f"Testing model in PyTorch...")
    start = perf_counter()
    output = model(input_img)
    elapsed = perf_counter() - start
    print(f"PyTorch eval time {elapsed:.3f}s")
    print(f"Predictions from PyTorch:")
    print_predictions(output)

    print(f"Exporting model to {onnx_path}")
    if dynamo:
        onnx_prog = torch.onnx.export(model, input_img, dynamo=True)
        # Optimize model to inline functions. See https://github.com/robertknight/rten/issues/655.
        onnx_prog.optimize()
        onnx_prog.save(onnx_path)
    else:
        torch.onnx.export(model, input_img, onnx_path)

    # Test exported model with ONNX Runtime as a reference implementation.
    #
    # We test both with graph optimizations disabled and enabled, to show the
    # impact of running the ONNX model "as is" vs. with the various fusions that
    # ONNX Runtime does.
    print(f"Testing model with ONNX Runtime (unoptimized)...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = ort.InferenceSession(
        onnx_path, sess_options, providers=ort.get_available_providers()
    )
    input_name = ort_session.get_inputs()[0].name
    start = perf_counter()
    ort_output = ort_session.run(None, {input_name: input_img.numpy()})
    elapsed = perf_counter() - start
    print(f"ONNX Runtime unoptimized eval time {elapsed:.3f}s")
    ort_scores = torch.tensor(ort_output[0])

    # Optionally save input and output tensors.
    #
    # This is useful for verifying that RTen produces the same results as PyTorch.
    # The resulting safetensors files can be passed to the RTen CLI as reference
    # inputs and outputs via `rten --inputs inputs.safetensors --check-outputs outputs.safetensors`.
    if save_tensors:
        import safetensors.torch

        output_name = ort_session.get_outputs()[0].name
        safetensors.torch.save_file({input_name: input_img}, "inputs.safetensors")
        safetensors.torch.save_file({output_name: output}, "outputs.safetensors")

    print(f"Predictions from ONNX Runtime (unoptimized):")
    print_predictions(ort_scores)

    print(f"Testing model with ONNX Runtime (default optimizations)...")
    ort_session = ort.InferenceSession(
        onnx_path, providers=ort.get_available_providers()
    )
    input_name = ort_session.get_inputs()[0].name
    start = perf_counter()
    ort_output = ort_session.run(None, {input_name: input_img.numpy()})
    elapsed = perf_counter() - start
    print(f"ONNX Runtime optimized eval time {elapsed:.3f}s")
    ort_scores = torch.tensor(ort_output[0])

    print(f"Predictions from ONNX Runtime (optimized):")
    print_predictions(ort_scores)


def main():
    parser = ArgumentParser(
        description="""
Export image models from the timm library to ONNX.

See https://huggingface.co/docs/hub/timm for documentation and
https://huggingface.co/models?library=timm&sort=downloads for a directory of
available models.
"""
    )
    parser.add_argument(
        "model_config",
        help="Name of the model configuration or Hugging Face model URL or path",
    )
    parser.add_argument("onnx_path", nargs="?", help="Path to ONNX file")
    parser.add_argument(
        "-d",
        "--dynamo",
        action="store_true",
        help="Use PyTorch's newer TorchDynamo-based ONNX exporter",
    )
    parser.add_argument(
        "--save-tensors",
        action="store_true",
        help="Save test model inputs and outputs to SafeTensors files",
    )
    args = parser.parse_args()

    config_name = extract_config_name(args.model_config)
    onnx_path = args.onnx_path
    if onnx_path is None:
        onnx_path = config_name + ".onnx"

    export_timm_model(
        config_name, onnx_path, dynamo=args.dynamo, save_tensors=args.save_tensors
    )


if __name__ == "__main__":
    main()
