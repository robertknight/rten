from argparse import ArgumentParser
from os.path import splitext
import json

from tokenizers import Tokenizer


def main():
    parser = ArgumentParser(
        description="""
Create a reference tokenization of text using the `tokenizers` package.
"""
    )
    parser.add_argument("model_name", help="Name of pretrained model from Hugging Face")
    parser.add_argument("text_file", help="Text to tokenize")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_pretrained(args.model_name)

    with open(args.text_file) as text_fp:
        text = text_fp.read()

    encoded = tokenizer.encode(text)

    output = {
        "input_file": args.text_file,
        "model_name": args.model_name,
        "token_ids": encoded.ids,
        "tokens": encoded.tokens,
    }
    json_output = json.dumps(output, indent=2)

    text_file_base, _ = splitext(args.text_file)
    model_fname = args.model_name.replace("/", "_")
    output_fname = f"{text_file_base}-{model_fname}.json"
    with open(output_fname, "w") as output:
        output.write(json_output)


if __name__ == "__main__":
    main()
