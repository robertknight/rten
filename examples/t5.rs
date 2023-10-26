use std::collections::VecDeque;
use std::error::Error;
use std::fs;

use wasnn::{Model, NodeId, Operators, RunOptions};
use wasnn_tensor::prelude::*;
use wasnn_tensor::{tensor, NdTensor, Tensor};

struct Args {
    encoder_model: String,
    decoder_model: String,
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
                    "Perform semantic segmentation on an image.

Usage: {bin_name} <encoder_model> <decoder_model>

Args:

  <encoder_model> - T5 encoder model
  <decoder_model> - T5 decoder model
",
                    bin_name = parser.bin_name().unwrap_or("t5")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let encoder_model = values.pop_front().ok_or("missing `encoder_model` arg")?;
    let decoder_model = values.pop_front().ok_or("missing `decoder_model` arg")?;

    let args = Args {
        encoder_model,
        decoder_model,
    };

    Ok(args)
}

fn find_node(model: &Model, name: &str) -> Result<NodeId, String> {
    model
        .find_node(name)
        .ok_or(format!("model has no `{name}` node"))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let encoder_model_bytes = fs::read(args.encoder_model)?;
    let encoder_model = Model::load(&encoder_model_bytes)?;
    let decoder_model_bytes = fs::read(args.decoder_model)?;
    let decoder_model = Model::load(&decoder_model_bytes)?;

    let input_ids_id = find_node(&encoder_model, "input_ids")?;
    let attention_mask_id = find_node(&encoder_model, "attention_mask")?;
    let output_id = find_node(&encoder_model, "last_hidden_state")?;

    // "summarize: studies have shown that owning a dog is good for you"
    let mut input_ids =
        tensor!([21603, 10, 2116, 43, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1]);
    input_ids.insert_dim(0);
    let attention_mask = Tensor::full(input_ids.shape(), 1i32);

    let [encoder_state] = encoder_model.run_n(
        &[
            (input_ids_id, (&input_ids).into()),
            (attention_mask_id, (&attention_mask).into()),
        ],
        [output_id],
        Some(RunOptions {
            timing: false,
            verbose: false,
        }),
    )?;

    let encoder_state: Tensor<f32> = encoder_state.try_into()?;

    let mut output_tokens: Vec<i32> = vec![0];

    let eos_token = 1;
    while output_tokens.last() != Some(&eos_token) {
        let mut decoder_input_ids: Tensor<i32> = Tensor::from_vec(output_tokens.clone());
        decoder_input_ids.insert_dim(0);
        let decoder_input_ids_id = find_node(&decoder_model, "input_ids")?;
        let encoder_attention_mask_id = find_node(&decoder_model, "encoder_attention_mask")?;
        let encoder_hidden_states_id = find_node(&decoder_model, "encoder_hidden_states")?;
        let decoder_logits_id = find_node(&decoder_model, "logits")?;

        let [logits] = decoder_model.run_n(
            &[
                (decoder_input_ids_id, (&decoder_input_ids).into()),
                (encoder_attention_mask_id, (&attention_mask).into()),
                (encoder_hidden_states_id, (&encoder_state).into()),
            ],
            [decoder_logits_id],
            Some(RunOptions {
                timing: false,
                verbose: false,
            }),
        )?;
        let logits: NdTensor<f32, 3> = logits.try_into()?;
        println!("logits shape {:?}", logits.shape());
        let tokens = logits.arg_max(-1, false /* keep_dims */)?;
        let new_token = *logits
            .arg_max(-1, false /* keep_dims */)?
            .slice([0, tokens.size(1) - 1])
            .item()
            .unwrap();
        output_tokens.push(new_token);

        println!("tokens {:?}", output_tokens);
    }

    println!("output tokens {:?}", output_tokens);

    Ok(())
}
