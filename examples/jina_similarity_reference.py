# Reference implementation of sentence similarity estimation using a BERT
# embedding model. Adapted from example on
# https://huggingface.co/jinaai/jina-embeddings-v2-small-en

from argparse import ArgumentParser

from numpy.linalg import norm
from transformers import AutoModel


def main():
    parser = ArgumentParser()
    parser.add_argument("first_sentence")
    parser.add_argument("second_sentence")
    args = parser.parse_args()

    cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
    )  # trust_remote_code is needed to use the encode method

    sentence_a = args.first_sentence
    sentence_b = args.second_sentence
    embeddings = model.encode([sentence_a, sentence_b])
    similarity = cos_sim(embeddings[0], embeddings[1])

    print(f'First sentence: "{sentence_a}"')
    print(f'Second sentence: "{sentence_b}"')
    print(f"Similarity: {similarity}")


if __name__ == "__main__":
    main()
