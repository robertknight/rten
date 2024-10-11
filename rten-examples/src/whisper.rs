use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rten::{Dimension, FloatOperators, Model};
use rten_generate::{Generator, GeneratorUtils};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor};
use rten_text::tokenizers::Tokenizer;
use serde::Deserialize;

struct Args {
    encoder_model: String,
    decoder_model: String,
    tokenizer_config: String,
    audio_path: String,
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
                    "Recognize speech in an audio file.

Usage: {bin_name} [options] <encoder_model> <decoder_model> <tokenizer> <audio>

Args:

  <encoder_model>  - Audio encoder model
  <decoder_model>  - Text decoder model
  <tokenizer>      - `tokenizer.json` file
  <audio>          - Audio path
",
                    bin_name = parser.bin_name().unwrap_or("whisper")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let encoder_model = values.pop_front().ok_or("missing `encoder_model` arg")?;
    let decoder_model = values.pop_front().ok_or("missing `decoder_model` arg")?;
    let tokenizer_config = values.pop_front().ok_or("missing `tokenizer` arg")?;
    let audio_path = values.pop_front().ok_or("missing `audio_path` arg")?;

    let args = Args {
        encoder_model,
        decoder_model,
        tokenizer_config,
        audio_path,
    };

    Ok(args)
}

/// Read a .wav audio file into a sequence of samples with values in [1, -1].
///
/// nb. The Whisper model expects the sample rate to be 16 kHz.
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

/// Minimal implementation of a complex number.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Complex<T: Copy> {
    pub real: T,
    pub imag: T,
}

impl<T: Copy + Default> Complex<T> {
    pub fn new(real: T, imag: T) -> Self {
        Complex { real, imag }
    }

    pub fn from_real(real: T) -> Self {
        Complex {
            real,
            imag: T::default(),
        }
    }
}

impl Complex<f32> {
    /// Evaluate euler's formula for angle `theta`.
    pub fn from_polar(radius: f32, theta: f32) -> Self {
        Complex::new(radius * theta.cos(), radius * theta.sin())
    }

    pub fn abs(self) -> f32 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
}

impl<T: Copy + Default> Default for Complex<T> {
    fn default() -> Self {
        Complex {
            real: T::default(),
            imag: T::default(),
        }
    }
}

impl std::ops::Add for Complex<f32> {
    type Output = Complex<f32>;

    fn add(self, rhs: Self) -> Self {
        Complex::new(self.real + rhs.real, self.imag + rhs.imag)
    }
}

impl std::ops::Sub for Complex<f32> {
    type Output = Complex<f32>;

    fn sub(self, rhs: Self) -> Self {
        Complex::new(self.real - rhs.real, self.imag - rhs.imag)
    }
}

impl std::ops::Mul for Complex<f32> {
    type Output = Complex<f32>;

    fn mul(self, rhs: Self) -> Self {
        Complex::new(
            (self.real * rhs.real) - (self.imag * rhs.imag),
            (self.real * rhs.imag) + (self.imag * rhs.real),
        )
    }
}

/// Compute the Hann window function.
///
/// See https://pytorch.org/docs/stable/generated/torch.hann_window.html.
fn hann_window(size: usize) -> NdTensor<f32, 1> {
    NdTensor::from_fn([size], |[i]| {
        ((std::f32::consts::PI * i as f32) / (size as f32 - 1.) as f32)
            .sin()
            .powf(2.)
    })
}

/// Compute the Discrete Fourier Transform of a signal.
///
/// The length of `signal` must be a power of 2.
fn fft(signal: &mut [Complex<f32>]) {
    let n = signal.len();
    if n <= 1 {
        return;
    }

    let mut even: smallvec::SmallVec<[Complex<f32>; 512]> =
        signal.iter().step_by(2).copied().collect();
    let mut odd: smallvec::SmallVec<[Complex<f32>; 512]> =
        signal.iter().skip(1).step_by(2).copied().collect();

    fft(&mut even);
    fft(&mut odd);

    for k in 0..n / 2 {
        let t = odd[k] * Complex::from_polar(1., -2. * std::f32::consts::PI * k as f32 / n as f32);
        signal[k] = even[k] + t;
        signal[k + n / 2] = even[k] - t;
    }
}

/// Compute the Short Time Fourier Transform (STFT) of an input signal.
///
/// Returns a matrix with shape `[n_fft / 2 + 1, n_windows]`.
///
/// See https://pytorch.org/docs/stable/generated/torch.stft.html.
fn stft(
    signal: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: Option<NdTensorView<f32, 1>>,
) -> NdTensor<Complex<f32>, 2> {
    let window_length = n_fft;
    let n_windows = signal.len() / hop_length;
    let out_freqs = n_fft / 2 + 1;
    let mut output = NdTensor::zeros([out_freqs, n_windows]);

    let padded_window_length = window_length.next_power_of_two();

    let mut win_signal = Vec::with_capacity(padded_window_length);

    for w in 0..n_windows {
        win_signal.clear();
        win_signal.extend((0..window_length).map(|k| {
            let weight = window.as_ref().map(|win| win[[k]]).unwrap_or(0.);
            Complex::from_real(weight * signal.get(w * hop_length + k).copied().unwrap_or(0.))
        }));
        win_signal.resize(padded_window_length, Complex::default());

        fft(&mut win_signal);

        let win_signal = NdTensorView::from_data([out_freqs], &win_signal[..out_freqs]);
        output.slice_mut((.., w)).copy_from(&win_signal);
    }

    output
}

#[derive(Deserialize)]
struct TensorData {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl TensorData {
    fn to_tensor(&self) -> Tensor {
        Tensor::from_data(&self.shape, self.data.clone())
    }
}

/// JSON-serialized mel filter bank.
///
/// See `data/dump_mel_filters.py`.
#[derive(Deserialize)]
struct MelFilters {
    mel_80: TensorData,
    mel_128: TensorData,
}

fn resource_path(path: &str) -> PathBuf {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push("data/");
    abs_path.push(path);
    abs_path
}

/// Compute the log-mel spectrogram of an audio waveform.
///
/// This is a Rust implementation of the `log_mel_spectrogram` function from
/// https://github.com/openai/whisper/blob/25639fc17ddc013d56c594bfbf7644f2185fad84/whisper/audio.py#L110.
fn log_mel_spectrogram(
    audio: &[f32],
    padded_len: usize,
    n_mels: u32,
    sample_rate: u32,
    mel_filter_map: &MelFilters,
) -> Result<NdTensor<f32, 2>, Box<dyn Error>> {
    let hop_length = 160;
    let n_fft = 400;

    // Pad input with zero samples.
    let n_pad = padded_len.saturating_sub(audio.len());
    let padded_audio: Vec<f32> = audio
        .iter()
        .copied()
        .chain(std::iter::repeat(0.).take(n_pad))
        .collect();

    // Compute Fourier transform of input, with a Hann window applied to prevent
    // spectral leakage.
    let window = hann_window(n_fft);
    let audio_fft = stft(&padded_audio, n_fft, hop_length, Some(window.view()));

    // Get power spectrum of input.
    let magnitudes: NdTensor<f32, 2> = audio_fft.map(|&x| {
        let x = x.abs();
        x * x
    });

    let mel_filters: NdTensor<f32, 2> = match (n_mels, sample_rate, n_fft) {
        (80, 16_000, 400) => mel_filter_map.mel_80.to_tensor(),
        (128, 16_000, 400) => mel_filter_map.mel_128.to_tensor(),
        _ => return Err("unsupported mel filter parameters".into()),
    }
    .to_tensor()
    .try_into()?;

    // Convert from hz to mels.
    let mels = mel_filters.matmul(magnitudes.as_dyn()).unwrap();
    let mels = mels.nd_view::<2>();

    // Convert amplitudes to log scale
    let log_mels = mels.map(|x| x.max(1e-10).log10());
    let log_mels_max = log_mels
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .clone()
        .unwrap();
    let log_mels = log_mels.map(|x| {
        let x = x.max(log_mels_max - 8.0);
        (x + 4.0) / 4.0
    });

    Ok(log_mels)
}

/// Recognize speech using OpenAI's Whisper model.
///
/// First use Hugging Face's Optimum tool to download and export the models to
/// ONNX:
///
/// ```
/// optimum-cli export onnx --model openai/whisper-base models/whisper-base
/// ```
///
/// Convert the models to `.rten` format. For the decoder you need to use the
/// "merged" model.
///
/// ```
/// rten-convert whisper-base/encoder_model.onnx
/// rten-convert whisper-base/decoder_model_merged.onnx
/// ```
///
/// Run the model, specifying the audio file to recognize. The audio file
/// must have a sample rate of 16 kHz. See the `wav2vec` example's comments
/// for details on how to create a suitable file.
///
/// ```sh
/// cargo run --release --bin whisper whisper-base/encoder_model.rten whisper-base/decoder_model_merged.rten whisper-base/tokenizer.json <audio>
/// ```
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let encoder_model = unsafe { Model::load_mmap(args.encoder_model)? };
    let decoder_model = unsafe { Model::load_mmap(args.decoder_model)? };
    let tokenizer_config = fs::read_to_string(&args.tokenizer_config)?;
    let tokenizer = Tokenizer::from_json(&tokenizer_config)?;

    let sample_rate = 16_000;
    let chunk_length = 30;
    let n_mels = match encoder_model.input_shape(0).as_deref() {
        // Encoder input shape is [batch, n_mels, sequence].
        Some([_, Dimension::Fixed(mels), _]) => *mels as u32,
        _ => 80,
    };
    let samples_per_chunk = sample_rate * chunk_length;

    // Load mel filter matrices exported from librosa
    let mel_filters_json = std::fs::read_to_string(resource_path("mel_filters.json"))?;
    let mel_filters: MelFilters = serde_json::from_str(&mel_filters_json)?;

    let audio = read_wav_file(&args.audio_path)?;

    let start = std::time::Instant::now();
    for audio_chunk in audio.chunks(samples_per_chunk) {
        let mut mel_spec = log_mel_spectrogram(
            &audio_chunk,
            samples_per_chunk,
            n_mels,
            sample_rate as u32,
            &mel_filters,
        )?
        .into_dyn();

        mel_spec.insert_axis(0); // Add batch dim

        let encoded_audio: NdTensor<f32, 3> = encoder_model
            .run_one(mel_spec.view().into(), None)?
            .try_into()?;

        let encoder_hidden_states_id = decoder_model.node_id("encoder_hidden_states")?;

        // Values from `generation_config.json`.
        let decoder_start_token = 50258; // <|startoftranscript|>
        let eos_token = 50257; // <|endoftext|>

        let prompt = vec![decoder_start_token];
        let generator = Generator::from_model(&decoder_model)?
            .with_prompt(&prompt)
            .with_constant_input(encoder_hidden_states_id, encoded_audio.view().into())
            .stop_on_tokens([eos_token])
            .decode(&tokenizer);

        let mut transcription = String::new();
        for token in generator {
            transcription.push_str(&token?);
        }
        println!("{}", transcription);
    }

    // Start new line after transcription.
    println!();
    println!("Decoding took {:.3}s", start.elapsed().as_secs_f64());

    Ok(())
}
