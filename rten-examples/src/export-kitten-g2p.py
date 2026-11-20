"""
Export the grapheme-to-phoneme (G2P) model used by Kitten TTS to ONNX.

Kitten TTS uses a small character-level BART model to predict the phonemes of
words that are not in its pronunciation lexicon. The model is distributed as
PyTorch weights in the `g2p/g2p_bart_en_us` directory of
https://huggingface.co/KittenML/kitten-tts-2-mini. This script exports it as
two ONNX models, `g2p_encoder.onnx` and `g2p_decoder.onnx`, for use by the
`kitten_tts` example.

Usage:

    python export-kitten-g2p.py path/to/g2p/g2p_bart_en_us

Requires the `torch` and `transformers` packages.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import BartForConditionalGeneration


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids).last_hidden_state


class Decoder(torch.nn.Module):
    """Decoder which returns logits for every position of `decoder_input_ids`."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.final_logits_bias = model.final_logits_bias

    def forward(self, decoder_input_ids, encoder_hidden_states):
        hidden = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        ).last_hidden_state
        return self.lm_head(hidden) + self.final_logits_bias


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Path to `g2p/g2p_bart_en_us` directory")
    parser.add_argument(
        "--output-dir",
        help="Directory to write ONNX models to. Defaults to `model_dir`.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or args.model_dir)
    model = BartForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, attn_implementation="eager"
    ).eval()

    seq_len = 8
    input_ids = torch.ones((1, seq_len), dtype=torch.int64)
    hidden = torch.zeros((1, seq_len, model.config.d_model))

    with torch.no_grad():
        torch.onnx.export(
            Encoder(model),
            (input_ids,),
            output_dir / "g2p_encoder.onnx",
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {1: "seq"},
                "last_hidden_state": {1: "seq"},
            },
            dynamo=False,
        )
        torch.onnx.export(
            Decoder(model),
            (input_ids, hidden),
            output_dir / "g2p_decoder.onnx",
            input_names=["decoder_input_ids", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "decoder_input_ids": {1: "out_seq"},
                "encoder_hidden_states": {1: "seq"},
                "logits": {1: "out_seq"},
            },
            dynamo=False,
        )
    print(f"Saved g2p_encoder.onnx and g2p_decoder.onnx to {output_dir}")


if __name__ == "__main__":
    main()
