use std::collections::VecDeque;
use std::error::Error;
use std::time::Instant;

use rten::Model;
use rten_generate::Generator;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;

struct Args {
    encoder_model: String,
    decoder_model: String,
    text: String,
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
                    "Convert text to phonemes.

Usage: {bin_name} <encder_model> <decoder_model> <text>
",
                    bin_name = parser.bin_name().unwrap_or("byt5_g2p")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let encoder_model = values.pop_front().ok_or("missing `encoder_model` arg")?;
    let decoder_model = values.pop_front().ok_or("missing `decoder_model` arg")?;
    let text = values.pop_front().ok_or("missing `text` arg")?;

    let args = Args {
        encoder_model,
        decoder_model,
        text,
    };

    Ok(args)
}

/// Convert a sequence of characters (graphemes) to phonemes using a ByT5 model [^1].
///
/// To export the model:
///
/// ```
/// optimum-cli export onnx --model fdemelo/g2p-mbyt5-12l-ipa-childes-espeak g2p --task text2text-generation-with-past
/// rten-convert g2p/encoder_model.onnx
/// rten-convert g2p/decoder_model_merged.onnx
/// ```
///
/// To run the example:
///
/// ```
/// cargo run -p rten-examples --release --bin byt5_g2p -- g2p/encoder_model.rten g2p/decoder_model_merged.rten "This is some text to convert."
/// ```
///
/// The example assumes US English text. To use a different language, change
/// the prompt. See https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak#language-tags.
///
/// [^1]: https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let encoder = Model::load_file(&args.encoder_model)?;
    let decoder = Model::load_file(&args.decoder_model)?;

    let prompt = format!("<en-US>: {}", args.text);
    let input_ids = encode_text(&prompt);

    let input_ids = NdTensor::from_data([1, input_ids.len()], input_ids);
    let attention_mask = NdTensor::full([1, input_ids.len()], 1);

    // Encode the input.
    let encode_start = Instant::now();
    let [encoded_state] = encoder
        .run_n(
            [
                (encoder.node_id("input_ids")?, input_ids.into()),
                (
                    encoder.node_id("attention_mask")?,
                    attention_mask.view().into(),
                ),
            ]
            .into(),
            [encoder.node_id("last_hidden_state")?],
            None,
        )
        .map_err(|e| format!("failed to run encoder: {}", e))?;
    let encoded_state: NdTensor<f32, 3> = encoded_state.try_into()?; // (batch, seq, dim)

    let decode_start = Instant::now();

    // Run the decoder to generate phonemes.
    let generator = Generator::from_model(&decoder)?
        .with_constant_input(
            decoder.node_id("encoder_attention_mask")?,
            attention_mask.view().into(),
        )
        .with_constant_input(
            decoder.node_id("encoder_hidden_states")?,
            encoded_state.view().into(),
        )
        .with_prompt(&[BOS_ID])
        .take(MAX_TOKENS);

    let mut token_ids = Vec::new();
    for token_id in generator {
        let token = token_id?;
        if token == EOS_ID {
            break;
        }
        token_ids.push(token);
    }

    let encode_dur = (decode_start - encode_start).as_secs_f32();
    let decode_dur = decode_start.elapsed().as_secs_f32();
    let elapsed = encode_start.elapsed().as_secs_f32();

    let phonemes = decode_ids(&token_ids);
    println!(
        "predicted phonemes in {:.3}s ({:.3} enc, {:.3} dec): {}",
        elapsed, encode_dur, decode_dur, phonemes
    );

    Ok(())
}

/// Start-of-text token.
const BOS_ID: u32 = 0;

/// End-of-text token.
const EOS_ID: u32 = 1;

/// Number of special tokens.
const SPECIAL_TOKEN_COUNT: u32 = 3;

const MAX_TOKENS: usize = 512;

/// Encode text to ByT5 token IDs.
///
/// ByT5 tokens are byte values but shifted to allow for a few special tokens.
/// See https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak#example-2-inference-without-tokenizer.
fn encode_text(text: &str) -> Vec<i32> {
    let mut ids = Vec::from([BOS_ID as i32]);
    ids.extend(
        text.chars()
            .map(|c| c as u32 + SPECIAL_TOKEN_COUNT)
            .map(|id| id as i32),
    );
    ids.push(EOS_ID as i32);
    ids
}

/// Decode ByT5 token IDs to text.
///
/// ByT5 tokens are byte values but shifted to allow for a few special tokens.
fn decode_ids(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .filter(|x| **x >= SPECIAL_TOKEN_COUNT)
        .map(|x| (*x - SPECIAL_TOKEN_COUNT) as u8)
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}
