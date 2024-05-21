use std::collections::HashMap;
use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use serde::Deserialize;

mod phonemes;
use phonemes::Phonemizer;

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

/// Input to convert to speech.
enum TtsInput {
    Text(String),

    /// A string of IPA phonemes.
    Phonemes(String),
}

struct Args {
    /// Path to converted Piper voice model.
    model: String,

    /// Path to configuration JSON for the Piper model.
    model_config: String,

    /// Path to pronounciation dictionary.
    pronounciation_dict: Option<String>,

    /// Text or phonemes to speak. See the [Phonemizer] docs for more details
    /// on how text are converted to phonemes.
    input: Option<TtsInput>,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();

    let mut pronounciation_dict = None;
    let mut input_is_phonemes = false;

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Short('d') | Long("dict") => {
                let dict_path = parser.value()?.string()?;
                pronounciation_dict = Some(dict_path);
            }
            Short('p') | Long("phonemes") => input_is_phonemes = true,
            Long("help") => {
                println!(
                    "Convert text to speech.

Usage: {bin_name} [options] <model> <model_config> [<text>]

Options:
   -d, --dict <path>

     Path to custom pronounciation dictionary.

   -p, --phonemes

     Interpret `<text>` input as a string of phonemes.
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
    let input = values.pop_front();

    let args = Args {
        model,
        model_config,
        input: input.map(|input| {
            if input_is_phonemes {
                TtsInput::Phonemes(input)
            } else {
                TtsInput::Text(input)
            }
        }),
        pronounciation_dict,
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

fn resource_path(path: &str) -> PathBuf {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push("src/piper");
    abs_path.push(path);
    abs_path
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

    // Load voice model and associated configuration.
    let model = Model::load_file(args.model)?;
    let config_json = std::fs::read_to_string(args.model_config)?;
    let config: ModelConfig = serde_json::from_str(&config_json)?;

    // Load pronounciation dictionary.
    let default_dict_path = resource_path("en-us.txt").to_string_lossy().into_owned();
    let dict_path = args
        .pronounciation_dict
        .as_deref()
        .unwrap_or(&default_dict_path);
    let dict = std::fs::read_to_string(&dict_path)?;
    let phonemizer = Phonemizer::load_dict(&dict);

    // Encode text as phonemes.
    let default_text = "This is a text to speech system";
    let phonemes = match args.input {
        Some(TtsInput::Phonemes(phonemes)) => phonemes,
        Some(TtsInput::Text(text)) => phonemizer.translate(text.as_str()),
        _ => phonemizer.translate(default_text),
    };

    // Encode phonemes to IDs and prepare other model inputs.
    let phoneme_ids = phonemes_to_ids(&phonemes, &config);
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
        &[
            (input_id, phoneme_ids.view().into()),
            (input_lengths_id, input_lengths.view().into()),
            (scales_id, scales.view().into()),
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
