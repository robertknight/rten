from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForMaskedLM

parser = ArgumentParser(
    description="Replace [MASK] tokens in the input text with model predictions."
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="Name of the model to use",

    # nb. If you get an error that this model is not supported, see
    # https://huggingface.co/answerdotai/ModernBERT-base/discussions/3.
    #
    # You can also use an older BERT model such as "bert-base-uncased".
    default="answerdotai/ModernBERT-base",
)
parser.add_argument("text", type=str, help="Input text containing [MASK] tokens")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForMaskedLM.from_pretrained(args.model)
inputs = tokenizer(args.text, return_tensors="pt")

# Print input and output token IDs to enable comparison against RTen's
# tokenization and model output.
input_ids = inputs["input_ids"][0].tolist()
print("Input IDs:", input_ids)

outputs = model(**inputs)

raw_output_ids = outputs.logits[0].argmax(axis=-1)
print("Output IDs:", raw_output_ids.tolist())

# Keep only the output IDs for positions where the input contained a mask token.
output_ids = input_ids.copy()
for pos in range(len(output_ids)):
    if output_ids[pos] == tokenizer.mask_token_id:
        output_ids[pos] = raw_output_ids[pos]

predicted_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(predicted_text)
