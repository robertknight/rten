import json
import sys

import librosa.filters


def ndarray_to_dict(array):
    """
    Return a JSON-serializable representation of an ndarray.

    This representation is compatible with rten-tensor's serde deserialization.
    """
    return {
        "shape": array.shape,
        "data": array.flatten().tolist(),
    }


# Generate mel filter matrices using the same method as Whisper's original
# preprocessing code and export them to JSON.
#
# See https://github.com/openai/whisper/blob/25639fc17ddc013d56c594bfbf7644f2185fad84/whisper/audio.py#L92
#
# Most of the entries are zero so the output could be made smaller by
# representing it as an (index, value) array. However, the non-sparse
# representation is pretty small when compressed.
mel_80 = librosa.filters.mel(sr=16_000, n_fft=400, n_mels=80)
mel_128 = librosa.filters.mel(sr=16_000, n_fft=400, n_mels=128)
data = {
    "_note": "Generated with dump_mel_filters.py",
    "mel_80": ndarray_to_dict(mel_80),
    "mel_128": ndarray_to_dict(mel_128),
}
with open("mel_filters.json", "w") as fp:
    json.dump(data, fp)
