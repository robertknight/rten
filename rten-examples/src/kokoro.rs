use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

use argh::FromArgs;
use hound::{SampleFormat, WavSpec, WavWriter};
use rten::{Model, RunOptions};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

/// Convert text to speech using Kokoro.
#[derive(FromArgs)]
struct Args {
    /// path to ONNX model
    #[argh(positional)]
    model: String,

    /// path to voice data (f32 vector serialized as bytes in little-endian order)
    #[argh(positional)]
    voice: String,

    /// phonemes to speak (optional)
    #[argh(positional)]
    phonemes: Option<String>,
}

/// Return a char => token ID map for tokenizing input phonemes.
///
/// This was extracted from the "vocab" field in
/// https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/blob/main/tokenizer.json.
fn create_vocab() -> HashMap<String, u32> {
    let entries = [
        ("$", 0),
        (";", 1),
        (":", 2),
        (",", 3),
        (".", 4),
        ("!", 5),
        ("?", 6),
        ("\u{2014}", 9),
        ("\u{2026}", 10),
        ("\"", 11),
        ("(", 12),
        (")", 13),
        ("\u{201C}", 14),
        ("\u{201D}", 15),
        (" ", 16),
        ("\u{0303}", 17),
        ("\u{02A3}", 18),
        ("\u{02A5}", 19),
        ("\u{02A6}", 20),
        ("\u{02A8}", 21),
        ("\u{1D5D}", 22),
        ("\u{AB67}", 23),
        ("A", 24),
        ("I", 25),
        ("O", 31),
        ("Q", 33),
        ("S", 35),
        ("T", 36),
        ("W", 39),
        ("Y", 41),
        ("\u{1D4A}", 42),
        ("a", 43),
        ("b", 44),
        ("c", 45),
        ("d", 46),
        ("e", 47),
        ("f", 48),
        ("h", 50),
        ("i", 51),
        ("j", 52),
        ("k", 53),
        ("l", 54),
        ("m", 55),
        ("n", 56),
        ("o", 57),
        ("p", 58),
        ("q", 59),
        ("r", 60),
        ("s", 61),
        ("t", 62),
        ("u", 63),
        ("v", 64),
        ("w", 65),
        ("x", 66),
        ("y", 67),
        ("z", 68),
        ("\u{0251}", 69),
        ("\u{0250}", 70),
        ("\u{0252}", 71),
        ("\u{00E6}", 72),
        ("\u{03B2}", 75),
        ("\u{0254}", 76),
        ("\u{0255}", 77),
        ("\u{00E7}", 78),
        ("\u{0256}", 80),
        ("\u{00F0}", 81),
        ("\u{02A4}", 82),
        ("\u{0259}", 83),
        ("\u{025A}", 85),
        ("\u{025B}", 86),
        ("\u{025C}", 87),
        ("\u{025F}", 90),
        ("\u{0261}", 92),
        ("\u{0265}", 99),
        ("\u{0268}", 101),
        ("\u{026A}", 102),
        ("\u{029D}", 103),
        ("\u{026F}", 110),
        ("\u{0270}", 111),
        ("\u{014B}", 112),
        ("\u{0273}", 113),
        ("\u{0272}", 114),
        ("\u{0274}", 115),
        ("\u{00F8}", 116),
        ("\u{0278}", 118),
        ("\u{03B8}", 119),
        ("\u{0153}", 120),
        ("\u{0279}", 123),
        ("\u{027E}", 125),
        ("\u{027B}", 126),
        ("\u{0281}", 128),
        ("\u{027D}", 129),
        ("\u{0282}", 130),
        ("\u{0283}", 131),
        ("\u{0288}", 132),
        ("\u{02A7}", 133),
        ("\u{028A}", 135),
        ("\u{028B}", 136),
        ("\u{028C}", 138),
        ("\u{0263}", 139),
        ("\u{0264}", 140),
        ("\u{03C7}", 142),
        ("\u{028E}", 143),
        ("\u{0292}", 147),
        ("\u{0294}", 148),
        ("\u{02C8}", 156),
        ("\u{02CC}", 157),
        ("\u{02D0}", 158),
        ("\u{02B0}", 162),
        ("\u{02B2}", 164),
        ("\u{2193}", 169),
        ("\u{2192}", 171),
        ("\u{2197}", 172),
        ("\u{2198}", 173),
        ("\u{1D7B}", 177),
    ];

    entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

/// Start/end of input token.
const BOS_ID: i32 = 0;

/// Convert text to speech using Kokoro.
///
/// Download the ONNX model from https://huggingface.co/robertknight/kokoro-onnx/tree/main.
///
/// Then run with:
///
/// ```txt
/// cargo run -p rten-examples -r --bin kokoro -- kokoro.onnx voice.bin [<phonemes>]
/// ```
///
/// The voice files can be downloaded from
/// https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices.
///
/// (Note: Don't use the version of the ONNX model from the onnx-community repository.
///  It relies on some particular behaviors of ONNX Runtime which do not work
///  in other runtimes)
///
/// _phonemes_ is an optional argument that specifies the text to read as a
/// sequence of phonemes. See [`Args::phonemes`].
fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();
    let model = Model::load_file(&args.model)?;

    let phonemes = args
        .phonemes
        .as_deref()
        .unwrap_or("laɪf ɪz laɪk ɐ bɑːks ʌv tʃɑːkləts");

    // Encode phonemes into token IDs.
    let vocab = create_vocab();
    let mut input_ids = vec![BOS_ID];
    let mut ch_buf = vec![0u8; 8];
    for ch in phonemes.chars() {
        let ch_str = ch.encode_utf8(&mut ch_buf);
        if let Some(token) = vocab.get(ch_str) {
            input_ids.push(*token as i32);
        }
    }
    input_ids.push(BOS_ID);

    let tokens = NdTensor::from_data([1, input_ids.len()], input_ids);

    // Load the voice.
    //
    // Voice files contain the f32 values of a tensor of shape
    // `[input_len, 1, style_dim]`.
    let voice_data: Vec<f32> = std::fs::read(&args.voice)?
        .as_chunks()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    let style_dim = 256;
    let max_tokens = voice_data.len() / style_dim;
    let voice_data = NdTensor::from_data([max_tokens, style_dim], voice_data);
    let num_tokens = tokens.size(1).saturating_sub(2).min(max_tokens - 1);
    let style = voice_data.slice((num_tokens..num_tokens + 1, ..));

    // Set playback rate of generated audio.
    let speed = NdTensor::from([1.]);

    // nb. Some variants of the Kokoro model use the name `tokens` instead of
    // `input_ids` in the inputs and `audio` instead of `waveform` in the outputs.
    let [output] = model.run_n(
        [
            (model.node_id("input_ids")?, tokens.into()),
            (model.node_id("style")?, style.into()),
            (model.node_id("speed")?, speed.into()),
        ]
        .into(),
        [model.node_id("waveform")?],
        Some(RunOptions {
            ..Default::default()
        }),
    )?;

    // Either (batch, seq) or (seq) depending on the model variant.
    let audio: Tensor<f32> = output.try_into()?;

    let wav_file = BufWriter::new(File::create("output.wav")?);
    let mut wav_writer = WavWriter::new(
        wav_file,
        WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    )?;
    for sample in audio.iter() {
        wav_writer.write_sample(*sample)?;
    }
    wav_writer.finalize()?;

    Ok(())
}
