use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use microfft::Complex32;
use rten::{Dimension, FloatOperators, Model};
use rten_generate::filter::{token_id_filter, LogitsFilter};
use rten_generate::{Generator, GeneratorUtils};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use rten_text::tokenizers::Tokenizer;
use serde::Deserialize;

struct Args {
    /// Path to Whisper encoder model.
    encoder_model: String,

    /// Path to Whisper decoder model.
    decoder_model: String,

    /// Path to tokenizer.json file.
    tokenizer_config: String,

    /// Path to input audio file. This must be a .wav file with a 16 kHz sample rate.
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
                    "Recognize speech in an audio file using OpenAI's Whisper.

Usage: {bin_name} [options] <encoder_model> <decoder_model> <tokenizer> <audio>

Args:

  <encoder_model>  - Audio encoder model
  <decoder_model>  - Text decoder model
  <tokenizer>      - `tokenizer.json` file
  <audio>          - Path to audio file (16 kHz .wav)
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

/// Compute the Hann window function.
///
/// See https://pytorch.org/docs/stable/generated/torch.hann_window.html.
fn hann_window(size: usize) -> NdTensor<f32, 1> {
    NdTensor::from_fn([size], |[i]| {
        ((std::f32::consts::PI * i as f32) / (size as f32 - 1.))
            .sin()
            .powf(2.)
    })
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
) -> NdTensor<Complex32, 2> {
    assert!(n_fft <= 512);

    let window_length = n_fft;
    let n_windows = signal.len() / hop_length;
    let out_freqs = n_fft / 2 + 1;
    let mut output = NdTensor::zeros([out_freqs, n_windows]);

    for w in 0..n_windows {
        let mut window_signal = std::array::from_fn(|k| {
            if k < window_length {
                let weight = window.as_ref().map(|win| win[[k]]).unwrap_or(0.);
                weight * signal.get(w * hop_length + k).copied().unwrap_or(0.)
            } else {
                0.
            }
        });

        let fft = microfft::real::rfft_512(&mut window_signal);

        let win_signal = NdTensorView::from_data([out_freqs], &fft[..out_freqs]);
        output.slice_mut((.., w)).copy_from(&win_signal);
    }

    output
}

/// JSON-serialized mel filter bank.
///
/// See `data/dump_mel_filters.py`.
#[derive(Deserialize)]
struct MelFilters {
    mel_80: NdTensor<f32, 2>,
    mel_128: NdTensor<f32, 2>,
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
    let magnitudes: NdTensor<f32, 2> = audio_fft.map(|&x| x.norm_sqr());

    let mel_filters: NdTensorView<f32, 2> = match (n_mels, sample_rate, n_fft) {
        (80, 16_000, 400) => mel_filter_map.mel_80.view(),
        (128, 16_000, 400) => mel_filter_map.mel_128.view(),
        _ => return Err("unsupported mel filter parameters".into()),
    };

    // Convert from hz to mels.
    let mels = mel_filters.matmul(magnitudes.as_dyn()).unwrap();
    let mels = mels.nd_view::<2>();

    // Convert amplitudes to log scale
    let log_mels = mels.map(|x| x.max(1e-10).log10());
    let log_mels_max = log_mels.iter().copied().max_by(f32::total_cmp).unwrap();
    let log_mels = log_mels.map(|x| {
        let x = x.max(log_mels_max - 8.0);
        (x + 4.0) / 4.0
    });

    Ok(log_mels)
}

/// Processor for decoder model outputs that applies Whisper's processing rules
/// for timestamp tokens.
///
/// See the
/// [`ApplyTimestampRules`](https://github.com/openai/whisper/blob/cdb81479623391f0651f4f9175ad986e85777f31/whisper/decoding.py#L441) filter in the original Python code.
struct TimestampFilter {
    /// Token ID of first timestamp token (`<|0.00|>`)
    timestamp_min: u32,

    /// Token ID of last timestamp token (`<|30.00|>`)
    timestamp_max: u32,

    /// Token ID of the `<|notimestamps|>` token.
    no_timestamps: u32,

    /// Length of the prompt for the current generation.
    prompt_len: usize,
}

impl TimestampFilter {
    fn new(
        timestamp_min: u32,
        timestamp_max: u32,
        no_timestamps: u32,
        prompt_len: usize,
    ) -> TimestampFilter {
        TimestampFilter {
            timestamp_min,
            timestamp_max,
            no_timestamps,
            prompt_len,
        }
    }

    fn is_timestamp_token(&self, token_id: u32) -> bool {
        token_id >= self.timestamp_min && token_id <= self.timestamp_max
    }
}

impl LogitsFilter for TimestampFilter {
    fn filter(
        &self,
        logits: NdTensorView<f32, 1>,
        prev_tokens: &[u32],
    ) -> Option<NdTensor<f32, 1>> {
        // TODO - Implement remaining parts of `ApplyTimestampRules` filter:
        // - Enforce that timestamp tokens appear in pairs, except before EOT.

        let probs = logits.softmax(-1).unwrap();
        let sum_timestamp_probs: f32 = probs
            .iter()
            .enumerate()
            .filter(|(i, _x)| self.is_timestamp_token(*i as u32))
            .map(|(_i, x)| *x)
            .sum();
        let max_non_timestamp_prob = probs
            .iter()
            .enumerate()
            .filter(|(i, _x)| !self.is_timestamp_token(*i as u32))
            .map(|(_i, x)| *x)
            .max_by(f32::total_cmp)?;
        let suppress_non_timestamp_tokens = sum_timestamp_probs > max_non_timestamp_prob;

        // Get the token ID of the most recently sampled timestamp token,
        // excluding any that appear in the prompt, as that may contain part
        // of the transcription for a previous chunk.
        let prev_timestamp = prev_tokens
            .iter()
            .skip(self.prompt_len)
            .filter(|t| self.is_timestamp_token(**t))
            .last()
            .copied()
            .unwrap_or(self.timestamp_min);

        Some(NdTensor::from_fn(logits.shape(), |[i]| {
            // If the total probability of a timestamp is greater than a
            // non-timestamp, pick the timestamp.
            let mut suppress = if suppress_non_timestamp_tokens {
                !self.is_timestamp_token(i as u32)
            } else {
                false
            };

            // The `<|notimestamps|>` token must never occur in output.
            suppress = suppress || i as u32 == self.no_timestamps;

            // Timestamps must increase within a segment.
            if self.is_timestamp_token(i as u32) && (i as u32) < prev_timestamp {
                suppress = true;
            }

            if !suppress {
                logits[[i]]
            } else {
                f32::NEG_INFINITY
            }
        }))
    }
}

/// Format a timestamp in seconds in `mm:ss.xxx` or `hh:mm:ss.xxx` format.
fn format_timestamp(sec: f32) -> String {
    let n_msec = (sec * 1000.0).round() as u32;
    let n_sec = n_msec / 1_000;
    let n_min = n_sec / 60;
    let n_hours = n_min / 60;
    let (ts_msec, ts_sec, ts_min) = (n_msec % 1000, n_sec % 60, n_min % 60);
    if n_hours > 0 {
        format!("{:02}:{:02}:{:02}.{:03}", n_hours, ts_min, ts_sec, ts_msec)
    } else {
        format!("{:02}:{:02}.{:03}", ts_min, ts_sec, ts_msec)
    }
}

/// Recognize speech using OpenAI's Whisper [^1][^2] model.
///
/// First use Hugging Face's Optimum tool to download and export the models to
/// ONNX:
///
/// ```
/// optimum-cli export onnx --model openai/whisper-base whisper-base
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
/// The audio input must be a 16 kHz WAV file. To convert an existing audio
/// file to this format you can use ffmpeg:
///
/// ```
/// ffmpeg -i input.mp3 -ar 16000 audio.wav
/// ```
///
/// Run the model, specifying the audio file to recognize.
///
/// ```sh
/// cargo run --release --bin whisper whisper-base/encoder_model.rten whisper-base/decoder_model_merged.rten whisper-base/tokenizer.json audio.wav
/// ```
///
/// [^1]: https://arxiv.org/abs/2212.04356
/// [^2]: https://github.com/openai/whisper
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let encoder_model = unsafe { Model::load_mmap(args.encoder_model)? };
    let decoder_model = unsafe { Model::load_mmap(args.decoder_model)? };
    let tokenizer_config = fs::read_to_string(&args.tokenizer_config)?;
    let tokenizer = Tokenizer::from_json(&tokenizer_config)?;

    // Length of audio chunks expected by model, in seconds.
    let chunk_length = 30;

    // Expected audio sample rate in hz.
    let sample_rate = 16_000;

    // Granularity of timestamp tokens in milliseconds.
    let timestamp_unit_ms = 20;

    // Maximum number of tokens that may be generated for each 30-second chunk.
    // This value is taken from HF Transformers.
    let max_tokens_per_chunk: usize = 448;

    // Number of mel bins used by the spectrogram.
    let n_mels = match encoder_model.input_shape(0).as_deref() {
        // Encoder input shape is [batch, n_mels, sequence].
        Some([_, Dimension::Fixed(mels), _]) => *mels as u32,
        _ => 80,
    };
    let samples_per_chunk = (sample_rate * chunk_length) as usize;

    // Load mel filter matrices exported from librosa.
    //
    // We could re-implement the mel filter generation in Rust, but the
    // matrices are quite small and static, so exporting them was easier.
    let mel_filters_json = std::fs::read_to_string(resource_path("mel_filters.json"))?;
    let mel_filters: MelFilters = serde_json::from_str(&mel_filters_json)?;

    let audio: Vec<f32> = read_wav_file(&args.audio_path, sample_rate)?;

    let start = std::time::Instant::now();

    // The text (ie. non-special) tokens from the previous chunk.
    let mut prev_chunk_tokens = Vec::new();

    // Offset of the start of the next chunk in milliseconds.
    let mut chunk_offset_ms = 0;

    // ID of the language identification token (eg. "<|en|>").
    let mut lang_id_token: Option<u32> = None;

    // Process 30-second chunks of audio.
    loop {
        let sample_offset = ((chunk_offset_ms as f32 / 1000.0) * sample_rate as f32) as usize;
        let audio_chunk =
            &audio[sample_offset..(sample_offset + samples_per_chunk).min(audio.len())];
        if audio_chunk.is_empty() {
            break;
        }

        let mut mel_spec = log_mel_spectrogram(
            audio_chunk,
            samples_per_chunk,
            n_mels,
            sample_rate,
            &mel_filters,
        )?
        .into_dyn();
        mel_spec.insert_axis(0); // Add batch dim

        let input_features_id = encoder_model.node_id("input_features")?;
        let output_id = encoder_model.node_id("last_hidden_state")?;
        let [encoded_audio] = encoder_model.run_n(
            [(input_features_id, mel_spec.view().into())].into(),
            [output_id],
            None,
        )?;
        let encoded_audio: NdTensor<f32, 3> = encoded_audio.try_into()?;

        let encoder = tokenizer.encoder();
        let start_of_transcript = encoder.get_token_id("<|startoftranscript|>")?;
        let encoder_hidden_states_id = decoder_model.node_id("encoder_hidden_states")?;

        // Get the language ID token (eg. "<|en|>").
        //
        // If unknown, run one iteration of the decoder to identify it.
        let lang_id_token = match lang_id_token {
            Some(token) => token,
            None => {
                let lang_id_min = encoder.get_token_id("<|en|>")?;
                let lang_id_max = encoder.get_token_id("<|su|>")?;
                let prompt = [start_of_transcript];
                let mut generator = Generator::from_model(&decoder_model)?
                    .with_prompt(&prompt)
                    .with_constant_input(encoder_hidden_states_id, encoded_audio.view().into())
                    .with_logits_filter(token_id_filter(|token| {
                        // Keep only language ID tokens
                        token >= lang_id_min && token <= lang_id_max
                    }))
                    .take(1);
                let token = generator.next().unwrap()?;

                // Get language ID token and strip special-token markers.
                let token_str = encoder.get_token_str(token)?;
                let lang_id_str = token_str.trim_start_matches("<|").trim_end_matches("|>");
                println!("Detected language: {}", lang_id_str);

                lang_id_token = Some(token);
                token
            }
        };

        // Construct prompt.
        //
        // See https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
        // for details.
        let eos_token = encoder.get_token_id("<|endoftext|>")?;
        let start_of_prev = encoder.get_token_id("<|startofprev|>")?;
        let transcribe = encoder.get_token_id("<|transcribe|>")?;
        let timestamp_min = encoder.get_token_id("<|0.00|>")?;
        let timestamp_max = encoder.get_token_id("<|30.00|>")?;
        let no_timestamps = encoder.get_token_id("<|notimestamps|>")?;

        let mut prompt = Vec::new();
        if !prev_chunk_tokens.is_empty() {
            prompt.push(start_of_prev);
            prompt.extend(prev_chunk_tokens.iter().copied())
        }
        prompt.extend([start_of_transcript, lang_id_token, transcribe]);
        prev_chunk_tokens.clear();

        // Decode audio chunk into transcript segments. Each segment starts
        // and ends with a timestamp token.
        let generator = Generator::from_model(&decoder_model)?
            .with_prompt(&prompt)
            .with_constant_input(encoder_hidden_states_id, encoded_audio.view().into())
            .with_logits_filter(TimestampFilter::new(
                timestamp_min,
                timestamp_max,
                no_timestamps,
                prompt.len(),
            ))
            .take(max_tokens_per_chunk.saturating_sub(prompt.len()))
            .stop_on_tokens([eos_token]);

        struct TranscriptChunk {
            start_ms: u32,
            end_ms: u32,
            caption: String,
        }

        let timestamp_token_id_to_ms =
            |token_id: u32| (token_id - timestamp_min) * timestamp_unit_ms;

        let mut transcript_chunks = Vec::new();

        let mut curr_chunk_start = None;
        let mut curr_chunk_tokens = Vec::new();
        for token in generator {
            let token = token?;
            if token != eos_token {
                // Save token for use in prompt for next chunk.
                prev_chunk_tokens.push(token);
            }

            if token >= timestamp_min && token <= timestamp_max {
                if let Some(start_timestamp) = curr_chunk_start {
                    transcript_chunks.push(TranscriptChunk {
                        start_ms: start_timestamp,
                        end_ms: timestamp_token_id_to_ms(token),
                        caption: tokenizer.encoder().decode(&curr_chunk_tokens)?,
                    });
                    curr_chunk_start = None;
                    curr_chunk_tokens.clear();
                } else {
                    curr_chunk_start = Some(timestamp_token_id_to_ms(token));
                }
            } else {
                curr_chunk_tokens.push(token);
            }
        }

        // Output the transcription for the current chunk.
        for chunk in &transcript_chunks {
            let start = (chunk_offset_ms + chunk.start_ms) as f32 / 1000.0;
            let end = (chunk_offset_ms + chunk.end_ms) as f32 / 1000.0;
            println!(
                "[{} --> {}]  {}",
                format_timestamp(start),
                format_timestamp(end),
                chunk.caption.trim()
            );
        }

        // Increment audio sample offset for next chunk based on the last
        // timestamp in the transcription output.
        let last_timestamp_ms = transcript_chunks
            .last()
            .map(|c| c.end_ms)
            .unwrap_or(chunk_length * 1000);
        chunk_offset_ms += last_timestamp_ms;

        if audio_chunk.len() < samples_per_chunk {
            break;
        }
    }

    let decode_duration = start.elapsed().as_secs_f64();
    let audio_duration: f64 = audio.len() as f64 / sample_rate as f64;
    let real_time_factor = audio_duration / decode_duration;
    println!(
        "Transcribed {:.0}s of audio in {:.2}s, {:.1}x real-time",
        audio_duration, decode_duration, real_time_factor
    );

    Ok(())
}
