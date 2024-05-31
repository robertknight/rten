from argparse import ArgumentParser
import piper_phonemize as pp
import tqdm

parser = ArgumentParser(
    description="Generate pronounciation dictionary using piper-phonemize"
)
parser.add_argument("word_list", help="Path to file containing words to translate")
parser.add_argument("espeak_voice", help='Name of the espeak-ng voice, eg. "en-us"')
args = parser.parse_args()


with open(args.word_list) as word_list:
    # Read the entire file so tqdm can show a more useful progress meter.
    lines = [line for line in word_list]
    for line in tqdm.tqdm(lines):
        line = line.strip()
        sequences = pp.phonemize_espeak(line, args.espeak_voice)
        phonemes = "".join(sequences[0])
        print(f"{line}\t{phonemes}")
