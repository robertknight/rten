from argparse import ArgumentParser
from transformers import pipeline, set_seed


def main():
    parser = ArgumentParser(description="Generate text using GPT-2 and a prompt")
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    prompt = " ".join(args.prompt)
    if args.seed is not None:
        set_seed(args.seed)

    print(f'prompt: "{prompt}"')
    generator = pipeline("text-generation", model="gpt2")

    sequences = generator(prompt, max_length=30, num_return_sequences=1, do_sample=False)
    for seq in sequences:
        print(seq)


if __name__ == "__main__":
    main()
