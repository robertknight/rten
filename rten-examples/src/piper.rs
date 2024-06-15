use std::collections::HashMap;
use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

use hound::{SampleFormat, WavSpec, WavWriter};
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use serde::Deserialize;

/// Convert a float audio sample to a 16-bit int, suitable for writing to a
/// .wav file with format `WAVE_FORMAT_PCM`.
///
/// Converted from https://github.com/rhasspy/piper/blob/master/src/python_run/piper/util.py.
fn audio_float_to_int16(
    audio: NdTensorView<f32, 1>,
    max_wav_value: Option<f32>,
) -> NdTensor<i16, 1> {
    let max_wav_value = max_wav_value.unwrap_or(32767.0);
    let audio_max = audio
        .iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.)
        .max(0.01);
    audio.map(|x| {
        let sample = x * (max_wav_value / audio_max);
        sample.clamp(-max_wav_value, max_wav_value) as i16
    })
}

/// Deserialized JSON config for a voice model.
///
/// See https://github.com/rhasspy/piper?tab=readme-ov-file#voices.
#[derive(Deserialize)]
struct ModelConfig {
    audio: AudioConfig,
    inference: InferenceConfig,

    /// Map of IPA phoneme character to model input IDs.
    phoneme_id_map: HashMap<char, Vec<i32>>,
}

#[derive(Deserialize)]
struct AudioConfig {
    /// Output sample rate in Hz.
    sample_rate: u32,
}

#[derive(Deserialize)]
struct InferenceConfig {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

struct Args {
    /// Path to converted Piper voice model.
    model: String,

    /// Path to configuration JSON for the Piper model.
    model_config: String,

    /// Custom string of phonemes to speak.
    ///
    /// The Piper project generates these using piper-phonemize
    /// (https://pypi.org/project/piper-phonemize/). You can run this locally
    /// with Python:
    ///
    /// ```text
    /// pip install piper-phonemize
    /// python
    ///
    /// >> import piper_phonemize as pp
    /// >> ''.join(pp.phonemize_espeak('This is a text to speech system', 'en-US')[0])
    /// ```
    ///
    /// The voice name ("en-US" here) can be found in the `espeak.voice`
    /// property of the voice model config.
    phonemes: Option<String>,
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
                    "Convert text to speech.

Usage: {bin_name} <model> <model_config> [<phonemes>]
",
                    bin_name = parser.bin_name().unwrap_or("piper")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let model_config = values.pop_front().ok_or("missing `model_config` arg")?;
    let phonemes = values.pop_front();

    let args = Args {
        model,
        model_config,
        phonemes,
    };

    Ok(args)
}

/// Convert a string of phonemes, characters representing speech sounds,
/// to input IDs for a Piper model.
///
/// For example, the phrase "This is a text to speech system" can be encoded
/// into the phonemes "ðɪs ɪz ɐ tˈɛkst tə spˈiːtʃ sˈɪstəm.". This function
/// will then convert that string into an ID sequence such as
/// `[1,  41,   0,  74,   0,  31, ... 2]` where `1` and `2` represent the
/// start and end of the input, and `0` is a separator.
fn phonemes_to_ids(phonemes: &str, config: &ModelConfig) -> NdTensor<i32, 1> {
    let start_ids = config
        .phoneme_id_map
        .get(&'^')
        .expect("missing ID for start char");
    let end_ids = config
        .phoneme_id_map
        .get(&'$')
        .expect("missing ID for end char");

    // Replacement IDs for unknown phonemes.
    let replacement = [];
    let separator = [0];
    let mut ids: Vec<i32> = start_ids.to_vec();
    ids.extend(phonemes.chars().flat_map(|ch| {
        if let Some(ids) = config.phoneme_id_map.get(&ch) {
            ids.iter().chain(separator.iter())
        } else {
            println!("Warning: Skipping unknown phoneme \"{}\"", ch);
            replacement.iter().chain(separator.iter())
        }
    }));
    ids.extend(end_ids);
    NdTensor::from_vec(ids)
}

/// Text to speech demo using models from Piper [1].
///
/// 1. Download the `en_US-lessac-medium.onnx` voice model and JSON config file
///    linked at https://github.com/rhasspy/piper?tab=readme-ov-file#voices.
///
///    Other voice models should also work, but have not been tested
///    extensively.
///
/// 2. Convert the model to `.rten` format using `rten-convert`
/// 3. Run the demo with:
///
///    ```
///    cargo run -p rten-examples -r --bin piper \
///      en_US-lessac-medium.rten en_US-lessac-medium.onnx.json
///    ```
///
///    This will generate an `output.wav` file, which you can play using
///    ffmpeg or another audio application:
///
///    ```
///    ffplay output.wav
///    ```
///
/// [1] https://github.com/rhasspy/piper
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let model = Model::load_file(args.model)?;

    let config_json = std::fs::read_to_string(args.model_config)?;
    let config: ModelConfig = serde_json::from_str(&config_json)?;

    // Phoenemes for "This is a text to speech system."
    //
    // The phonemes were generated by getting Piper's Python inference script
    // to log them before running the model.
    //
    // See https://github.com/rhasspy/piper/blob/a0f09cdf9155010a45c243bc8a4286b94f286ef4/src/python_run/piper/voice.py#L165
    let default_phonemes = "ðɪs ɪz ɐ tˈɛkst tə spˈiːtʃ sˈɪstəm.";

    // Encode phonemes to IDs and prepare other model inputs.
    let phonemes = args.phonemes.as_deref().unwrap_or(default_phonemes);
    let phoneme_ids = phonemes_to_ids(phonemes, &config);
    let phoneme_ids_len = phoneme_ids.size(0);
    let phoneme_ids = phoneme_ids.into_shape([1, phoneme_ids_len]); // Add batch dim
    let input_lengths = NdTensor::from([phoneme_ids_len as i32]);
    let scales = NdTensor::from([
        config.inference.noise_scale,
        config.inference.length_scale,
        config.inference.noise_w,
    ]);

    // Run inference and generate audio samples as floats.
    let input_id = model.find_node("input").unwrap();
    let input_lengths_id = model.find_node("input_lengths").unwrap();
    let output_id = model.find_node("output").unwrap();
    let scales_id = model.find_node("scales").unwrap();

    let [samples] = model.run_n(
        vec![
            (input_id, phoneme_ids.into()),
            (input_lengths_id, input_lengths.into()),
            (scales_id, scales.into()),
        ],
        [output_id],
        None,
    )?;
    let samples: NdTensor<f32, 4> = samples.try_into()?; // (batch, time, 1, sample)

    // Convert audio samples from float to 16-bit ints and write to output .wav
    // file.
    let int_samples = audio_float_to_int16(samples.slice::<1, _>((0, 0, 0)), None);
    let wav_file = BufWriter::new(File::create("output.wav")?);

    let mut wav_writer = WavWriter::new(
        wav_file,
        WavSpec {
            channels: 1,
            sample_rate: config.audio.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        },
    )?;
    let mut wav_16_writer = wav_writer.get_i16_writer(int_samples.len() as u32);
    for sample in int_samples.iter().copied() {
        wav_16_writer.write_sample(sample);
    }
    wav_16_writer.flush()?;
    wav_writer.finalize()?;

    Ok(())
}
