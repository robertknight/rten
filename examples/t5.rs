use std::collections::VecDeque;
use std::error::Error;
use std::fs;

use wasnn::{Model, NodeId, Operators, RunOptions};
use wasnn_tensor::prelude::*;
use wasnn_tensor::{tensor, NdTensor, Tensor};

struct Args {
    model: String,
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

Usage: {bin_name} <model>

Args:

  <model> - T5 combined encoder + decoder model
",
                    bin_name = parser.bin_name().unwrap_or("t5")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;

    let args = Args { model };

    Ok(args)
}

fn find_node(model: &Model, name: &str) -> Result<NodeId, String> {
    model
        .find_node(name)
        .ok_or(format!("model has no `{name}` node"))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model_bytes = fs::read(args.model)?;
    let model = Model::load(&model_bytes)?;

    // "translate to french: studies have shown that owning a dog is good for you"
    let mut encoder_tokens = tensor!([
        13959, 12, 20609, 10, 2116, 43, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1
    ]);
    encoder_tokens.insert_dim(0);

    // expected output:
    //
    // "Des études ont montré que la propriété d’un chien est bonne pour vous"
    //
    // [
    //     0, 2973, 17868, 30, 17, 29625, 238, 50, 19713, 3, 26, 22, 202, 17826, 259, 4079, 171, 327,
    //     1,
    // ];

    let mut output_tokens: Vec<i32> = vec![0];
    let max_tokens = 20;

    let eos_token = 1;
    while output_tokens.last() != Some(&eos_token) && output_tokens.len() < max_tokens {
        let mut decoder_tokens: Tensor<i32> = Tensor::from_vec(output_tokens.clone());
        decoder_tokens.insert_dim(0);

        let encoder_tokens_id = find_node(&model, "encoder_tokens")?;
        let decoder_tokens_id = find_node(&model, "decoder_tokens")?;
        let logits_id = find_node(&model, "logits")?;

        let [logits] = model.run_n(
            &[
                (encoder_tokens_id, (&encoder_tokens).into()),
                (decoder_tokens_id, (&decoder_tokens).into()),
            ],
            [logits_id],
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
