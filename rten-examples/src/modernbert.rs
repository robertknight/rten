use std::collections::VecDeque;
use std::error::Error;

use rten::{Model, Operators};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use rten_text::{TokenId, Tokenizer, TokenizerError};

struct Args {
    model: String,
    tokenizer: String,
    input: String,
    show_token_ids: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();
    let mut show_token_ids = false;

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Predict masked words in a sentence.

Usage: {bin_name} [options] <model> <tokenizer> <input>

Args:

  <model>       - Input BERT model
  <tokenizer>   - `tokenizer.json` file
  <input>       - Text with \"[MASK]\" spans to fill in

Options:

 -t, --token-ids  - Show token IDs for input and output text
",
                    bin_name = parser.bin_name().unwrap_or("modernbert")
                );
                std::process::exit(0);
            }
            Short('t') | Long("token-ids") => {
                show_token_ids = true;
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let tokenizer = values.pop_front().ok_or("missing `tokenizer` arg")?;
    let input = values.make_contiguous().join(" ");

    let args = Args {
        model,
        tokenizer,
        input,
        show_token_ids,
    };

    Ok(args)
}

/// Predict masked words in a sentence using [ModernBERT].
///
/// First download the ModernBERT model in ONNX format from
/// https://huggingface.co/answerdotai/ModernBERT-base/tree/main, as well
/// as the `tokenizer.json` file.
///
/// Convert the model using `rten-convert`:
///
/// ```
/// rten-convert modernbert.onnx
/// ```
///
/// Run the example using:
///
/// ```
/// cargo run --release --bin modernbert modernbert.rten tokenizer.json "Earth is a [MASK]."
/// ```
///
/// This should print "Earth is a planet." Note the period at the end of the
/// input sentence. Without that the model may predict the masked token is
/// simply a new line.
///
/// This example also works with classic BERT models such as
/// https://huggingface.co/google-bert/bert-base-uncased.
///
/// [ModernBERT]: https://huggingface.co/answerdotai/ModernBERT-base
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.model)?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let cls_token = tokenizer.get_token_id("[CLS]")?;
    let sep_token = tokenizer.get_token_id("[SEP]")?;
    let mask_token = tokenizer.get_token_id("[MASK]")?;

    // Encode input, replacing occurrences of "[MASK]" by the corresponding
    // special token ID.

    let tokenize_piece = |piece| -> Result<Vec<TokenId>, TokenizerError> {
        let ids = tokenizer
            .encode(piece, None)?
            .into_token_ids()
            .into_iter()
            // `Tokenizer::encode` adds [CLS] and [SEP] tokens. Remove these
            // as we only want to insert them around the whole input ID sequence.
            .filter(|id| ![cls_token, sep_token].contains(id))
            .collect();
        Ok(ids)
    };

    let mut input_ids = Vec::from([cls_token]);
    let mut mask_indices = Vec::new();
    let mut remainder = args.input.as_str();
    while let Some(mask_pos) = remainder.find("[MASK]") {
        // `trim_end` replicates the behavior of the `lstrip` attribute for
        // the `[MASK]` special token in tokenizer.json.
        let piece_str = remainder[..mask_pos].trim_end();
        input_ids.extend(tokenize_piece(piece_str)?);
        mask_indices.push(input_ids.len());
        input_ids.push(mask_token);
        remainder = &remainder[mask_pos + "[MASK]".len()..];
    }
    if !remainder.is_empty() {
        input_ids.extend(tokenize_piece(remainder)?);
    }
    input_ids.push(sep_token);

    if args.show_token_ids {
        println!("Input IDs: {:?}", input_ids);
    }

    // Prepare model inputs
    let input_ids = NdTensor::from_data([1, input_ids.len()], input_ids).map(|id| *id as i32);
    let attention_mask = NdTensor::full(input_ids.shape(), 1);

    let input_ids_id = model.node_id("input_ids")?;
    let attention_mask_id = model.node_id("attention_mask")?;
    let logits_id = model.node_id("logits")?;

    let mut model_inputs = Vec::from([
        (input_ids_id, input_ids.view().into()),
        (attention_mask_id, attention_mask.into()),
    ]);

    // ModernBERT doesn't have a `token_type_ids` input, but this example also
    // works with older BERT models that do. If using such a model, such as
    // "bert-base-uncased", provide this input.
    if let Ok(token_type_ids_id) = model.node_id("token_type_ids") {
        let token_type_ids = NdTensor::full(input_ids.shape(), 0);
        model_inputs.push((token_type_ids_id, token_type_ids.into()));
    }

    // Run model and predict masked words.
    let [logits] = model.run_n(model_inputs, [logits_id], None)?;

    // Get the most likely token for each position, filter out special tokens
    // and decode into text.
    let logits: NdTensor<f32, 3> = logits.try_into()?;
    let mut output_ids: NdTensor<i32, 2> = logits
        .arg_max(2 /* axis */, false /* keep_dims */)?
        .try_into()?;
    let mut output_ids = output_ids.slice_mut(0); // Remove batch dim

    if args.show_token_ids {
        println!("Output IDs: {:?}", output_ids.to_vec());
    }

    // Discard output tokens other than those corresponding to `[MASK]` tokens
    // in the input.
    for pos in 0..output_ids.size(0) {
        if !mask_indices.contains(&pos) {
            output_ids[pos] = input_ids[[0, pos]];
        }
    }

    let output_ids: Vec<TokenId> = output_ids
        .iter()
        .filter_map(|id| {
            let id = *id as TokenId;

            // Strip special tokens.
            if ![cls_token, sep_token].contains(&id) {
                Some(id)
            } else {
                None
            }
        })
        .collect();

    let text = tokenizer.decode(&output_ids)?;

    println!("{}", text);

    Ok(())
}
