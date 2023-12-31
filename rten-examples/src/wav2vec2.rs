use std::collections::VecDeque;
use std::error::Error;
use std::fs;

use rten::ctc::CtcDecoder;
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

struct Args {
    model: String,
    wav_file: String,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Recognize speech in .wav files.

Usage: {bin_name} <model_path> <wav_file>
",
                    bin_name = parser.bin_name().unwrap_or("wav2vec2")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let wav_file = values.pop_front().ok_or("missing `wav_file` arg")?;

    let args = Args { model, wav_file };

    Ok(args)
}

/// Read a .wav audio file into a sequence of samples with values in [1, -1].
///
/// nb. The wav2vec2 model expects the sample rate to be 16 kHz.
fn read_wav_file(path: &str) -> Result<Vec<f32>, hound::Error> {
    let mut reader = hound::WavReader::open(path)?;

    let spec = reader.spec();
    let expected_sample_rate = 16_000;

    if spec.sample_rate != expected_sample_rate {
        println!(
            "WARNING: Sample rate is {} kHz, this model expects {} kHz.",
            spec.sample_rate / 1_000,
            expected_sample_rate
        );
    }

    let mut samples = Vec::new();
    for sample in reader.samples::<i16>() {
        samples.push(sample?);
    }
    let float_samples: Vec<f32> = samples
        .into_iter()
        .map(|x| (x as f32) / i16::MAX as f32)
        .collect();
    Ok(float_samples)
}

/// Recognize speech in .wav audio files using Wav2Vec 2 [1]
///
/// The wav2vec2 speech recognition model [2] can be obtained from Hugging Face
/// and converted to this library's format using Optimum [3].
///
/// ```
/// optimum-cli export onnx --model facebook/wav2vec2-base-960h wav2vec2
/// tools/convert-onnx.py wav2vec2/model.onnx wav2vec2.rten
/// ```
///
/// To record a .wav file and test this app:
///
/// 1. Use a tool such as QuickTime (on macOS) to record audio from your
///    microphone and save the result.
///
/// 2. Use ffmpeg to convert the audio file to a 16 kHz .wav file:
///
///    ```
///    ffmpeg -i saved-file.m4a -ar 16000 output.wav
///    ```
///
/// 3. Run this program on the generated .wav file:
///
///    ```
///    cargo run --release --bin wav2vec2 wav2vec.rten output.wav
///    ```
///
/// [1] https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/
/// [2] https://huggingface.co/facebook/wav2vec2-base-960h
/// [3] https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    // Vocabulary converted from the vocab.json file at
    // https://huggingface.co/facebook/wav2vec2-base-960h/tree/main.
    //
    // The special characters have been replaced with "?" and the character
    // with index 0 is omitted, as that is used for a CTC blank.
    let vocab = "???|ETAONIHSRDLUMWCFGYPBVK'XJQZ";

    let model_data = fs::read(args.model)?;
    let model = Model::load(&model_data)?;

    let samples = read_wav_file(&args.wav_file)?;

    let mut sample_batch = Tensor::from_vec(samples);
    sample_batch.insert_dim(0);

    let result: NdTensor<f32, 3> = model
        .run_one(sample_batch.view().into(), None)?
        .try_into()?;

    let decoder = CtcDecoder::new();
    let hypothesis = decoder.decode_beam(result.slice([0]), 10 /* beam_size */);
    let text = hypothesis.to_string(vocab);

    println!("{}", text);

    Ok(())
}
