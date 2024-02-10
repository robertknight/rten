use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fs;
use std::time::Instant;

use rten::{Dimension, Input, Model, ModelMetadata, NodeId, Output, RunOptions};
use rten_tensor::prelude::*;
use rten_tensor::Tensor;

struct Args {
    /// Model file to load.
    model: String,

    /// Show operator timing stats.
    timing: bool,

    /// Enable verbose logging for model execution.
    verbose: bool,

    /// Map of `(input_name, dims)` with custom shapes for inputs.
    input_shapes: HashMap<String, Vec<usize>>,
}

/// Parse an input shape specifier in the form `input_name=dim0,dim1,...`.
///
/// Returns a tuple of (name, shape).
fn parse_shape_spec(spec: &str) -> Result<(String, Vec<usize>), lexopt::Error> {
    let parts: Vec<&str> = spec.split('=').collect();
    if parts.len() != 2 {
        return Err(lexopt::Error::Custom(
            "Invalid input format. Expected input_name=dim0,dim1,...".into(),
        ));
    }

    let name = parts[0].to_string();
    let dims_str = parts[1];
    let parsed_dims: Result<Vec<usize>, _> = dims_str.split(',').map(|dim| dim.parse()).collect();

    match parsed_dims {
        Ok(dims) => Ok((name, dims)),
        Err(e) => Err(lexopt::Error::Custom(e.into())),
    }
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut timing = false;
    let mut verbose = false;
    let mut input_shapes = HashMap::new();

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Short('v') | Long("verbose") => verbose = true,
            Short('t') | Long("timing") => timing = true,
            Short('s') | Long("shape") => {
                let value = parser.value()?.string()?;
                let (name, shape) = parse_shape_spec(&value)?;
                input_shapes.insert(name, shape);
            }
            Short('h') | Long("help") => {
                println!(
                    "Inspect and run RTen models.

Usage: {bin_name} [OPTIONS] <model>

Args:
  <model>
    Path to '.rten' model to inspect and run.

Options:
  -t, --timing   Output timing info
  -v, --verbose  Enable verbose logging
  -h, --help     Print help

  -s, --shape <shape>
                 Specify shape for an input in the form `name=dim0,dim1,...`
",
                    bin_name = parser.bin_name().unwrap_or("rten")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `<model>` arg")?;

    Ok(Args {
        model,
        timing,
        verbose,
        input_shapes,
    })
}

fn format_param_count(n: usize) -> String {
    if n > 1_000_000 {
        format!("{:.1} M", n as f32 / 1_000_000.)
    } else {
        format!("{:.1} K", n as f32 / 1000.)
    }
}

fn print_metadata(metadata: &ModelMetadata) {
    fn print_field<T: std::fmt::Display>(name: &str, value: Option<T>) {
        if let Some(value) = value {
            println!("  {}: {}", name, value);
        }
    }

    println!("Metadata:");
    print_field("ONNX hash", metadata.onnx_hash());
    print_field("Description", metadata.description());
    print_field("License", metadata.license());
    print_field("Commit", metadata.commit());
    print_field("Repository", metadata.code_repository());
    print_field("Model repository", metadata.model_repository());
    print_field("Run ID", metadata.run_id());
    print_field("Run URL", metadata.run_url());
}

/// Generate random inputs for `model` using shape metadata and heuristics,
/// run it, and print details of the output.
///
/// `custom_shapes` is a map of (input_name, dims) to use as shapes for inputs.
/// If a shape is not specified for an input, one is generated using heuristics
/// and the shape information specified by the model.
fn run_with_random_input(
    model: &Model,
    custom_shapes: &HashMap<String, Vec<usize>>,
    run_opts: RunOptions,
) -> Result<(), Box<dyn Error>> {
    let mut rng = fastrand::Rng::new();

    // Generate random ints that are likely to be valid token IDs in a language
    // model.
    let generate_token_id = |rng: &mut fastrand::Rng| rng.i32(0..1000);

    // Generate random model inputs. The `Output` type here is used as an
    // enum that can hold tensors of different types.
    let inputs: Vec<(NodeId, Output)> = model.input_ids().iter().copied().try_fold(
        Vec::<(NodeId, Output)>::new(),
        |mut inputs, id| {
            let info = model.node_info(id).ok_or("Unable to get input info")?;
            let name = info.name().unwrap_or("(unnamed input)");
            let shape = info
                .shape()
                .ok_or(format!("Unable to get shape for input {}", name))?;

            let resolved_shape = if let Some(shape) = custom_shapes.get(name) {
                shape.clone()
            } else {
                shape
                    .iter()
                    .map(|dim| {
                        match dim {
                            // Guess a suitable size for an input dimension based on
                            // the name.
                            Dimension::Symbolic(name) => match name.as_str() {
                                "batch" | "batch_size" => 1,
                                "sequence" | "sequence_length" => 128,
                                _ => 256,
                            },
                            Dimension::Fixed(size) => *size,
                        }
                    })
                    .collect()
            };

            // Guess suitable content for the input based on its name.
            let tensor = match name {
                // If this is a mask, use all ones on the assumption that we
                // don't want to mask anything out.
                name if name.ends_with("_mask") => {
                    Output::from(Tensor::full(&resolved_shape, 1i32))
                }

                // For BERT-style models from Hugging Face, `token_type_ids`
                // must be 0 or 1.
                "token_type_ids" => Output::from(Tensor::<i32>::zeros(&resolved_shape)),

                // For input names such as `input_ids`, generate some input that
                // is likely to be a valid token ID.
                name if name.ends_with("_ids") => {
                    Output::from(Tensor::from_simple_fn(&resolved_shape, || {
                        generate_token_id(&mut rng)
                    }))
                }

                // For anything else, random floats in [0, 1].
                //
                // TODO - Value nodes in the model should include data types,
                // so we can at least be sure to generate values of the correct
                // type.
                _ => Output::from(Tensor::from_simple_fn(&resolved_shape, || rng.f32())),
            };

            inputs.push((id, tensor));

            Ok::<_, Box<dyn Error>>(inputs)
        },
    )?;

    // Convert inputs from `Output` (owned) to `Input` (view).
    let inputs: Vec<(NodeId, Input)> = inputs
        .iter()
        .map(|(id, output)| (*id, Input::from(output)))
        .collect();

    for (id, input) in inputs.iter() {
        let info = model.node_info(*id);
        let name = info
            .as_ref()
            .and_then(|ni| ni.name())
            .unwrap_or("(unnamed)");
        println!("  Input \"{name}\" generated shape {:?}", input.shape());
    }

    // Run model and summarize outputs.
    let start = Instant::now();
    let outputs = model.run(&inputs, model.output_ids(), Some(run_opts))?;
    let elapsed = start.elapsed().as_millis();

    println!();
    println!(
        "  Model returned {} outputs in {:.2}ms.",
        outputs.len(),
        elapsed
    );
    println!();

    let output_names: Vec<String> = model
        .output_ids()
        .iter()
        .map(|id| {
            model
                .node_info(*id)
                .and_then(|ni| ni.name().map(|n| n.to_string()))
                .unwrap_or("(unnamed)".to_string())
        })
        .collect();

    for (i, (output, name)) in outputs.iter().zip(output_names).enumerate() {
        let dtype = match output {
            Output::FloatTensor(_) => "f32",
            Output::IntTensor(_) => "i32",
        };
        println!(
            "  Output {i} \"{name}\" data type {} shape: {:?}",
            dtype,
            output.shape()
        );
    }

    Ok(())
}

/// Format an input or output shape as a `[dim0, dim1, ...]` string, where each
/// dimension is represented by its fixed size or symbolic name.
fn format_shape(shape: &[Dimension]) -> String {
    let dims = shape
        .iter()
        .map(|dim| match dim {
            Dimension::Fixed(value) => value.to_string(),
            Dimension::Symbolic(name) => name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", dims)
}

/// Print a summary of the names and shapes of a list of input or output node IDs.
fn print_input_output_list(model: &Model, node_ids: &[NodeId]) {
    for &node_id in node_ids {
        let Some(info) = model.node_info(node_id) else {
            continue;
        };
        println!(
            "  {}: {}",
            info.name().unwrap_or("(unnamed)"),
            info.shape()
                .map(|dims| format_shape(&dims))
                .unwrap_or("(unknown shape)".to_string())
        );
    }
}

/// Tool for inspecting converted ONNX models and running them with randomly
/// generated inputs.
///
/// ```
/// pip install -e rten-convert
/// rten-convert model.onnx output.rten
/// cargo run -p rten-cli --release output.rten
/// ```
///
/// To get detailed timing information set the `RTEN_TIMING` env var before
/// running. See `docs/profiling.md`.
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model_bytes = fs::read(args.model)?;
    let model = Model::load(&model_bytes)?;

    println!(
        "Model summary: {} inputs, {} outputs, {} params",
        model.input_ids().len(),
        model.output_ids().len(),
        format_param_count(model.total_params()),
    );
    println!();

    println!("Inputs");
    print_input_output_list(&model, model.input_ids());
    println!();

    println!("Outputs");
    print_input_output_list(&model, model.output_ids());
    println!();

    print_metadata(model.metadata());

    println!();
    println!("Running model with random inputs...");
    run_with_random_input(
        &model,
        &args.input_shapes,
        RunOptions {
            timing: args.timing,
            verbose: args.verbose,
            ..Default::default()
        },
    )?;

    Ok(())
}
