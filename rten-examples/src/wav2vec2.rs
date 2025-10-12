use std::collections::VecDeque;
use std::error::Error;

use rten::Model;
use rten::ctc::CtcDecoder;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

struct Args {
    model: String,
    wav_file: String,
    raw_text: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut raw_text = false;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Short('r') | Long("raw") => raw_text = true,
            Long("help") => {
                println!(
                    "Recognize speech in .wav files.

Usage: {bin_name} <model_path> <wav_file>

Options:
  -r, --raw    Output the characters predicted by the model without any cleanup
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

    let args = Args {
        model,
        wav_file,
        raw_text,
    };

    Ok(args)
}

/// Normalize values in `data` to have zero mean and unit variance.
fn normalize_mean_variance(data: &mut [f32]) {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();
    for x in data {
        *x = (*x - mean) / std_dev;
    }
}

/// Read a .wav audio file into a sequence of normalized samples.
///
/// The samples are normalized to have zero mean and unit variance. See section
/// 2 of https://arxiv.org/pdf/2006.11477 ("The raw waveform input to the
/// encoder is normalized to zero mean and unit variance") and
/// https://github.com/facebookresearch/fairseq/issues/3277.
fn read_wav_file(path: &str, expected_sample_rate: u32) -> Result<Vec<f32>, hound::Error> {
    let mut reader = hound::WavReader::open(path)?;

    let spec = reader.spec();
    if spec.sample_rate != expected_sample_rate {
        println!(
            "WARNING: Sample rate is {} kHz, this model expects {} kHz.",
            spec.sample_rate / 1_000,
            expected_sample_rate
        );
    }

    let samples = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
    let mut float_samples: Vec<f32> = samples.into_iter().map(|x| x as f32).collect();

    normalize_mean_variance(&mut float_samples);

    Ok(float_samples)
}

/// Recognize speech in .wav audio files using Wav2Vec 2 [^1]
///
/// The wav2vec2 speech recognition model [^2] can be obtained from Hugging Face
/// and converted to this library's format using Optimum [^3].
///
/// ```
/// optimum-cli export onnx --model facebook/wav2vec2-base-960h wav2vec2
/// ```
///
/// For better accuracy at the cost of slower transcription, you can use larger
/// models such as "facebook/wav2vec2-large-960h-lv60-self".
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
///    cargo run --release --bin wav2vec2 wav2vec.onnx output.wav
///    ```
///
/// [^1]: <https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/>
/// [^2]: <https://huggingface.co/facebook/wav2vec2-base-960h>
/// [^3]: <https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model>
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    // Vocabulary converted from the vocab.json file at
    // https://huggingface.co/facebook/wav2vec2-base-960h/tree/main.
    //
    // The special characters have been replaced with "?" and the character
    // with index 0 is omitted, as that is used for a CTC blank.
    let vocab = "???|ETAONIHSRDLUMWCFGYPBVK'XJQZ";

    let sample_rate = 16_000;
    let chunk_length = 10;

    let model = Model::load_file(args.model)?;
    let samples = read_wav_file(&args.wav_file, sample_rate)?;

    // Chunk the audio into fixed-length chunks to support transcription of
    // longer audio clips. This approach to chunking is very simple, but there
    // are better approaches. See https://huggingface.co/blog/asr-chunking.
    for sample_chunk in samples.chunks(sample_rate as usize * chunk_length) {
        let mut sample_batch = Tensor::from(sample_chunk.to_vec());
        sample_batch.insert_axis(0);

        let result: NdTensor<f32, 3> = model
            .run_one(sample_batch.view().into(), None)?
            .try_into()?;

        let decoder = CtcDecoder::new();
        let hypothesis = decoder.decode_beam(result.slice(0), 10 /* beam_size */);
        let raw_text = hypothesis.to_string(vocab);

        // Lower-case output for readibility.
        let text = raw_text.replace("|", " ").to_lowercase();

        if args.raw_text {
            println!("{}", raw_text);
        } else {
            println!("{}", text);
        }
    }

    Ok(())
}
