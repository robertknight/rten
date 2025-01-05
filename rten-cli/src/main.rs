use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::time::Instant;

use rten::{
    DataType, Dimension, InputOrOutput, Model, ModelMetadata, ModelOptions, NodeId, Output,
    RunOptions,
};
use rten_tensor::prelude::*;
use rten_tensor::Tensor;

struct Args {
    /// Model file to load.
    model: String,

    /// Whether to enable graph optimizations
    optimize: bool,

    /// Whether to enable prepacking of weights
    prepack_weights: bool,

    /// Run model and don't produce other output
    quiet: bool,

    /// Show operator timing stats.
    timing: bool,

    /// Enable verbose logging for model execution.
    verbose: bool,

    /// Sizes for dynamic dimensions of inputs.
    input_sizes: Vec<DimSize>,

    /// Number of times to run model.
    n_iters: u32,

    /// Load model using `Model::load_mmap`.
    mmap: bool,
}

/// Specifies the size for a dynamic input dimension.
struct DimSize {
    /// Name of model input. If `None`, this matches all inputs.
    input_name: Option<String>,

    /// Name of the dynamically-sized dimension.
    dim_name: String,

    /// Dimension size
    size: usize,
}

impl DimSize {
    /// Return true if `self` specifies the size for a given input dimension.
    fn matches(&self, input_name: &str, dim_name: &str) -> bool {
        match self {
            DimSize {
                input_name: Some(in_name),
                dim_name: dn,
                size: _,
            } if in_name == input_name && dn == dim_name => true,
            DimSize {
                input_name: None,
                dim_name: dn,
                size: _,
            } if dn == dim_name => true,
            _ => false,
        }
    }

    /// Parse a dimension size specifier in the form `dim_name=size` or
    /// `input_name.dim_name=size`.
    fn parse(spec: &str) -> Result<DimSize, String> {
        let parts: Vec<&str> = spec.split('=').collect();
        let (name_spec, size_spec) = match parts[..] {
            [name, size] => (name, size),
            _ => {
                return Err(
                    "Invalid input format. Expected dim_name=size or input_name.dim_name=size"
                        .into(),
                );
            }
        };

        let name_parts: Vec<_> = name_spec.split('.').collect();
        let (input_name, dim_name) = match &name_parts[..] {
            [dim_name] => (None, dim_name),
            [input_name, dim_name] => (Some(input_name), dim_name),
            _ => {
                return Err(
                    "Invalid input input name format. Expected dim_name or input_name.dim_name"
                        .into(),
                );
            }
        };

        let size: usize = size_spec
            .parse()
            .map_err(|_| format!("Failed to parse dimension size \"{}\"", parts[1]))?;

        Ok(DimSize {
            input_name: input_name.map(|s| s.to_string()),
            dim_name: dim_name.to_string(),
            size,
        })
    }
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();

    let mut mmap = false;
    let mut n_iters = 1;
    let mut quiet = false;
    let mut timing = false;
    let mut verbose = false;
    let mut input_sizes = Vec::new();
    let mut optimize = true;
    let mut prepack_weights = false;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("mmap") => mmap = true,
            Short('n') | Long("n_iters") => {
                let value = parser.value()?.string()?;
                n_iters = value
                    .parse()
                    .map_err(|_| "Unable to parse `n_iters`".to_string())?;
            }
            Long("no-optimize") => optimize = false,
            Short('p') | Long("prepack") => prepack_weights = true,
            Short('q') | Long("quiet") => quiet = true,
            Short('v') | Long("verbose") => verbose = true,
            Short('V') | Long("version") => {
                println!("rten {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            Short('t') | Long("timing") => timing = true,
            Short('s') | Long("shape") => {
                let value = parser.value()?.string()?;
                let size =
                    DimSize::parse(&value).map_err(|err| lexopt::Error::Custom(err.into()))?;
                input_sizes.push(size);
            }
            Short('h') | Long("help") => {
                println!(
                    "Inspect and run RTen models.

Usage: {bin_name} [OPTIONS] <model>

Args:
  <model>
    Path to '.rten' model to inspect and run.

Options:
  -h, --help     Print help

  --mmap         Load model via memory mapping

  -n, --n_iters <n>
                 Number of times to evaluate model

  --no-optimize  Disable graph optimizations

  -q, --quiet    Run model and don't produce other output

  -p, --prepack  Enable prepacking of weights.
                 This requires additional memory but makes inference faster.

  -s, --size <spec>
                 Specify size for a dynamic dimension in the form `dim_name=size`
                 or `input_name.dim_name=size`

  -t, --timing   Output timing info

  -v, --verbose  Enable verbose logging
  -V, --version  Display RTen version
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
        input_sizes,
        mmap,
        model,
        n_iters,
        optimize,
        prepack_weights,
        quiet,
        timing,
        verbose,
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
/// `dim_sizes` specifies the sizes for input dimensions with dynamic sizes.
fn run_with_random_input(
    model: &Model,
    dim_sizes: &[DimSize],
    run_opts: RunOptions,
    n_iters: u32,
    quiet: bool,
) -> Result<(), Box<dyn Error>> {
    let mut rng = fastrand::Rng::new();

    // Names of all dynamic dimensions for which no size was explicitly
    // specified.
    let mut dynamic_dims_using_default_size: HashSet<String> = HashSet::new();

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
            let dtype = info.dtype();

            let resolved_shape: Vec<usize> = shape
                .iter()
                .map(|dim| match dim {
                    Dimension::Symbolic(dim_name) => {
                        let dim_size = dim_sizes.iter().find(|ds| ds.matches(name, dim_name));
                        if let Some(ds) = dim_size {
                            ds.size
                        } else {
                            dynamic_dims_using_default_size.insert(dim_name.to_string());
                            1
                        }
                    }
                    Dimension::Fixed(size) => *size,
                })
                .collect();

            fn random_ints<T, F: FnMut() -> T>(shape: &[usize], gen: F) -> Output
            where
                Output: From<Tensor<T>>,
            {
                Tensor::from_simple_fn(shape, gen).into()
            }

            // Guess suitable content for the input based on its name.
            let tensor = match name {
                // If this is a mask, use all ones on the assumption that we
                // don't want to mask anything out.
                name if name.ends_with("_mask") => {
                    Output::from(Tensor::full(&resolved_shape, 1i32))
                }

                // Inputs such as `token_type_ids`, `position_ids`, `input_ids`.
                // We use zero as a value that is likely to be valid for all
                // of these.
                name if name.ends_with("_ids") => {
                    Output::from(Tensor::<i32>::zeros(&resolved_shape))
                }

                // Optimum can export "merged" transformer models which have two
                // branches. One accepts KV-cache inputs and the other does not.
                // Set this to false as a "safer" value because we don't have
                // cached outputs from a previous run.
                "use_cache_branch" => Output::from(Tensor::from(0i32)),

                // For anything else, random values.
                _ => match dtype {
                    // Generate floats in [0, 1]
                    Some(DataType::Float) | None => {
                        Output::from(Tensor::from_simple_fn(&resolved_shape, || rng.f32()))
                    }
                    // Generate random values for int types. The default ranges
                    // are intended to be suitable for many models, but there
                    // ought to be a way to override them.
                    Some(DataType::Int32) => random_ints(&resolved_shape, || rng.i32(0..256)),
                    Some(DataType::Int8) => random_ints(&resolved_shape, || rng.i8(0..=127)),
                    Some(DataType::UInt8) => random_ints(&resolved_shape, || rng.u8(0..=255)),
                },
            };

            inputs.push((id, tensor));

            Ok::<_, Box<dyn Error>>(inputs)
        },
    )?;

    // Warn about any dynamic dims for which sizes were generated.
    //
    // Some models may have many inputs with the same dim name. To be less
    // verbose, we only warn once per dim name.
    if !quiet && !dynamic_dims_using_default_size.is_empty() {
        for dim_name in dynamic_dims_using_default_size {
            println!(
                "  Size not specified for dim \"{}\". Defaulting to 1.",
                dim_name
            );
        }
    }

    // Convert inputs from `Output` (owned) to `Input` (view).
    let inputs: Vec<(NodeId, InputOrOutput)> = inputs
        .iter()
        .map(|(id, output)| (*id, InputOrOutput::from(output)))
        .collect();

    if !quiet {
        for (id, input) in inputs.iter() {
            let info = model.node_info(*id);
            let name = info
                .as_ref()
                .and_then(|ni| ni.name())
                .unwrap_or("(unnamed)");
            println!("  Input \"{name}\" shape {:?}", input.shape());
        }
    }

    // Run model and summarize outputs.
    if !quiet {
        println!();
    }

    let n_iters = n_iters.max(1);
    let mut iter_num = 1;
    let mut outputs;
    let mut durations = Vec::new();

    // `loop` instead of `for` to guarantee `outputs` is initialized.
    loop {
        let start = Instant::now();
        outputs = model.run(inputs.clone(), model.output_ids(), Some(run_opts.clone()))?;
        let elapsed = start.elapsed().as_millis();

        if !quiet {
            println!(
                "  #{} - Model returned {} outputs in {:.2}ms.",
                iter_num,
                outputs.len(),
                elapsed
            );
        }
        durations.push(elapsed);

        if iter_num >= n_iters {
            break;
        }
        iter_num += 1;
    }
    if !quiet {
        if n_iters > 1 {
            let n_iters_float = n_iters as f32;
            let duration_floats: Vec<_> = durations.into_iter().map(|dur| dur as f32).collect();
            let mean = duration_floats.iter().sum::<f32>() / n_iters_float;
            let variance = duration_floats
                .iter()
                .map(|dur| (dur - mean) * (dur - mean))
                .sum::<f32>()
                / n_iters_float;
            let std_dev = variance.sqrt();
            let min = duration_floats
                .iter()
                .min_by(|a, b| f32::total_cmp(a, b))
                .unwrap();
            let max = duration_floats
                .iter()
                .max_by(|a, b| f32::total_cmp(a, b))
                .unwrap();

            println!();
            println!(
                "  Duration stats: mean {:.2}ms, min {:.2}ms, max {:.2}ms, std dev {:.2}ms",
                mean, min, max, std_dev
            );
        }
        println!();
    }

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

    if !quiet {
        for (i, (output, name)) in outputs.iter().zip(output_names).enumerate() {
            let dtype = match output {
                Output::FloatTensor(_) => "f32",
                Output::Int32Tensor(_) => "i32",
                Output::Int8Tensor(_) => "i8",
                Output::UInt8Tensor(_) => "u8",
            };
            println!(
                "  Output {i} \"{name}\" data type {} shape: {:?}",
                dtype,
                output.shape()
            );
        }
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
            "  {}: {} {}",
            info.name().unwrap_or("(unnamed)"),
            info.dtype()
                .map(|dt| dt.to_string())
                .unwrap_or("(unknown dtype)".to_string()),
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
/// rten-convert model.onnx output.rten
/// cargo run -p rten-cli --release output.rten
/// ```
///
/// To get detailed timing information set the `RTEN_TIMING` env var before
/// running. See `docs/profiling.md`.
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let mut model_opts = ModelOptions::with_all_ops();
    model_opts.enable_optimization(args.optimize);
    model_opts.prepack_weights(args.prepack_weights);

    let model = if args.mmap {
        unsafe { model_opts.load_mmap(args.model)? }
    } else {
        model_opts.load_file(args.model)?
    };

    if !args.quiet {
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
    }

    run_with_random_input(
        &model,
        &args.input_sizes,
        RunOptions {
            timing: args.timing,
            verbose: args.verbose,
            ..Default::default()
        },
        args.n_iters,
        args.quiet,
    )?;

    Ok(())
}
