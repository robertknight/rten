use std::error::Error;
use std::time::Instant;

use argh::FromArgs;
use rten::Model;
use rten_generate::Generator;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;

/// Convert text to phonemes.
#[derive(FromArgs)]
struct Args {
    /// encoder model path
    #[argh(positional)]
    encoder_model: String,

    /// decoder model path
    #[argh(positional)]
    decoder_model: String,

    /// text to convert to phonemes
    #[argh(positional)]
    text: String,

    /// language tag (default: en-US). see https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak#language-tags
    #[argh(option, short = 'l')]
    lang: Option<String>,
}

/// Convert a sequence of characters (graphemes) to phonemes using a ByT5 model [^1].
///
/// To export the model:
///
/// ```
/// optimum-cli export onnx --model fdemelo/g2p-mbyt5-12l-ipa-childes-espeak g2p --task text2text-generation-with-past
/// ```
///
/// To run the example:
///
/// ```
/// cargo run -p rten-examples --release --bin byt5_g2p -- g2p/encoder_model.onnx g2p/decoder_model_merged.onnx "This is some text to convert."
/// ```
///
/// The example assumes US English text. To use a different language, specify
/// a language tag with `--lang`. See https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak#language-tags.
///
/// [^1]: https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak
fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();

    if let Some(ref lang_tag) = args.lang
        && !SUPPORTED_LANGS.contains(&lang_tag.as_str())
    {
        eprintln!("WARNING: {} is an unrecognized language tag", lang_tag);
    }

    let encoder = Model::load_file(&args.encoder_model)?;
    let decoder = Model::load_file(&args.decoder_model)?;

    // Per model card: "The tag must be prepended to the prompt as a prefix
    // using the format <{tag}>: (e.g., <pt-BR>: ). Note: a space between the
    // prefix colon (:) and the beginning of the text is mandatory."
    let language_tag = args.lang.as_deref().unwrap_or("en-US");
    let prompt = format!("<{}>: {}", language_tag, args.text);
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

/// Supported language tags from
/// https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak#language-tags.
const SUPPORTED_LANGS: [&str; 31] = [
    "ca-ES", "cy-GB", "da-DK", "de-DE", "en-US", "en-GB", "es-ES", "et-EE", "eu-ES", "fa-IR",
    "fr-FR", "ga-IE", "hr-HR", "hu-HU", "id-ID", "is-IS", "it-IT", "ja-JP", "ko-KR", "nb-NO",
    "nl-NL", "pl-PL", "pt-BR", "pt-PT", "qu-PE", "ro-RO", "sr-RS", "sv-SE", "tr-TR", "yue-CN",
    "zh-CN",
];

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
    text.as_bytes()
        .iter()
        .map(|c| *c as u32 + SPECIAL_TOKEN_COUNT)
        .map(|id| id as i32)
        .collect()
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
