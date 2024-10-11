import librosa.filters
import json

import sys


def ndarray_to_dict(array):
    """Return a JSON-serializable representation of an ndarray."""
    return {
        "shape": array.shape,
        "data": array.flatten().tolist(),
    }

# Generate mel filter matrices using the same method as Whisper's original
# preprocessing code and export them to JSON.
#
# See https://github.com/openai/whisper/blob/25639fc17ddc013d56c594bfbf7644f2185fad84/whisper/audio.py#L92
mel_80 = librosa.filters.mel(sr=16_000, n_fft=400, n_mels=80)
mel_128 = librosa.filters.mel(sr=16_000, n_fft=400, n_mels=128)
data = {
    "_note": "Generated with dump_mel_filters.py",
    "mel_80": ndarray_to_dict(mel_80),
    "mel_128": ndarray_to_dict(mel_128),
}
with open("mel_filters.json", "w") as fp:
    json.dump(data, fp)
