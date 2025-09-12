use std::collections::VecDeque;
use std::error::Error;
use std::ops::Range;

use rten::Model;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;

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
                    "Detect speech in .wav files.

Usage: {bin_name} <model_path> <wav_file>
",
                    bin_name = parser.bin_name().unwrap_or("silero")
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
/// The Silero model supports a sample rate of either 8 or 16 kHz. This example
/// only supports 16 kHz.
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

/// Time offset or duration in milliseconds.
type Millis = usize;

/// Configuration for the speech detection state machine.
struct VadConfig {
    /// Probability threshold indicating presence of speech
    pos_threshold: f32,

    /// Probability threshold indicating absence of speech
    neg_threshold: f32,

    /// Minimum duration of speech segments
    min_speech_duration: Millis,

    /// Minimum duration of a gap between two speech segments
    min_silence_duration: Millis,
}

impl Default for VadConfig {
    fn default() -> VadConfig {
        // These are based on the defaults for the `get_speech_timestamps` function
        // at https://github.com/snakers4/silero-vad/blob/46f94b7d6029e19b482eebdfff0c18012fa84675/src/silero_vad/utils_vad.py#L187.
        VadConfig {
            pos_threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_duration: 250,
            min_silence_duration: 100,
        }
    }
}

/// State machine which tracks whether voice activity is currently detected.
struct VadState {
    config: VadConfig,

    /// True if we're currently in a speaking state.
    speaking: bool,

    /// Offset of start of chunk where last transition between speaking and
    /// non-speaking occurred.
    last_transition_time: Millis,

    /// True if speech was detected in the previous audio chunk.
    prev_chunk_speaking: bool,

    /// End offset of previous audio chunk.
    prev_time: Millis,
}

impl VadState {
    fn new(config: VadConfig) -> VadState {
        VadState {
            config,
            speaking: false,
            last_transition_time: 0,
            prev_chunk_speaking: false,
            prev_time: 0,
        }
    }

    /// Update the speech state using the speech probability output for an
    /// audio chunk from the Silero model.
    fn update(&mut self, current_time: Millis, speech_prob: f32) {
        assert!(current_time >= self.prev_time);

        let curr_chunk_speaking = if speech_prob > self.config.pos_threshold {
            true
        } else if speech_prob < self.config.neg_threshold {
            false
        } else {
            self.prev_chunk_speaking
        };
        if curr_chunk_speaking != self.prev_chunk_speaking {
            self.last_transition_time = self.prev_time;
        }

        if curr_chunk_speaking {
            if !self.speaking {
                let speech_duration = current_time - self.last_transition_time;
                if speech_duration >= self.config.min_speech_duration {
                    self.speaking = true;
                }
            }
        } else if self.speaking {
            let silence_duration = current_time - self.last_transition_time;
            if silence_duration >= self.config.min_silence_duration {
                self.speaking = false;
            }
        }

        self.prev_chunk_speaking = curr_chunk_speaking;
        self.prev_time = current_time;
    }

    /// Return the speech start time if speech was detected.
    fn speech_start(&self) -> Option<Millis> {
        self.speaking.then_some(self.last_transition_time)
    }
}

/// Detect speech in .wav audio files using Silero VAD [^1].
///
/// Download the ONNX model from the Silero VAD repository at
/// https://github.com/snakers4/silero-vad/tree/master/src/silero_vad/data,
/// then convert it using:
///
/// ```
/// rten-convert silero_vad.onnx
/// ```
///
/// To record a .wav file and run this example:
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
///    cargo run --release --bin silero_vad silero.rten output.wav
///    ```
///
/// [^1]: <https://github.com/snakers4/silero-vad>
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let model = Model::load_file(args.model)?;
    let samples = read_wav_file(&args.wav_file)?;

    // Create initial internal state for the model.
    let mut state: NdTensor<f32, 3> = NdTensor::zeros([2, 1, 128]); // [2, batch, 128]
    let sample_rate = 16_000;
    let chunk_duration = 30;

    // Initialize state machine that tracks whether speech is present.
    let mut speech_state = VadState::new(VadConfig::default());
    let mut speech_segments = Vec::new();
    let mut current_segment: Option<Range<Millis>> = None;

    let samples_per_chunk = (sample_rate * chunk_duration) / 1_000;
    for (i, chunk) in samples.chunks(samples_per_chunk).enumerate() {
        let padded_chunk: Vec<_> = chunk
            .iter()
            .copied()
            .chain(std::iter::repeat(0.))
            .take(samples_per_chunk)
            .collect();

        let [output, next_state] = model.run_n(
            [
                (
                    "input",
                    NdTensor::from_data([1, samples_per_chunk], padded_chunk).into(),
                ),
                ("sr", NdTensor::from(sample_rate as i32).into()),
                ("state", state.view().into()),
            ]
            .into(),
            ["output", "stateN"],
            None,
        )?;

        let curr_chunk_duration = (chunk_duration * samples_per_chunk) / chunk.len();
        let start_time = i * chunk_duration;
        let end_time = start_time + curr_chunk_duration;

        let output: NdTensor<f32, 2> = output.try_into()?;
        let prob = output
            .get([0, 0])
            .copied()
            .ok_or::<&str>("empty probability output")?;
        speech_state.update(end_time, prob);

        if let Some(speech_start) = speech_state.speech_start() {
            if let Some(current_segment) = current_segment.as_mut() {
                current_segment.end = end_time;
            } else {
                current_segment = Some(speech_start..end_time);
            }
        } else if let Some(segment) = current_segment {
            speech_segments.push(segment);
            current_segment = None;
        }

        println!(
            "time: {}..{}ms speech prob: {:.3} chunk speech: {} speaking: {}",
            start_time,
            end_time,
            prob,
            speech_state.prev_chunk_speaking,
            speech_state.speech_start().is_some(),
        );

        state = next_state.try_into()?;
    }
    if let Some(current_segment) = current_segment {
        speech_segments.push(current_segment);
    }

    for segment in speech_segments {
        let start_sec = segment.start as f32 / 1000.0;
        let end_sec = segment.end as f32 / 1000.0;
        println!("Speech segment: {:.2}..{:.2}", start_sec, end_sec);
    }

    Ok(())
}
