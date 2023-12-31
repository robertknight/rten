# Reference implementation of sentence similarity estimation using a BERT
# embedding model. Adapted from example on
# https://huggingface.co/jinaai/jina-embeddings-v2-small-en

from argparse import ArgumentParser

from numpy.linalg import norm
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("first_sentence")
    parser.add_argument("second_sentence")
    args = parser.parse_args()

    cos_sim = lambda a, b: a.dot(b) / (norm(a) * norm(b))
    model_name = "jinaai/jina-embeddings-v2-small-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True
    )  # trust_remote_code is needed to use the encode method

    sentences = [args.first_sentence, args.second_sentence]
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    # FIXME - Is this needed
    embeddings = F.normalize(embeddings, p=2, dim=1)

    similarity = cos_sim(embeddings[0], embeddings[1])

    print(f'First sentence: "{args.first_sentence}"')
    print(f'Second sentence: "{args.second_sentence}"')
    print(f"Similarity: {similarity}")


if __name__ == "__main__":
    main()
