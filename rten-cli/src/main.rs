use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::time::Instant;

use rten::{
    DataType, Dimension, InputOrOutput, Model, ModelMetadata, ModelOptions, NodeId, Output,
    RunOptions, ThreadPool,
};
use rten_tensor::prelude::*;
use rten_tensor::Tensor;

mod dim_size;
use dim_size::DimSize;

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

    /// Number of threads to use.
    num_threads: Option<u32>,
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
    let mut num_threads = None;

    let parse_uint = |parser: &mut lexopt::Parser, opt_name| -> Result<u32, lexopt::Error> {
        let value = parser.value()?.string()?;
        value
            .parse()
            .map_err(|_| format!("Unable to parse `{}`", opt_name).into())
    };

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("mmap") => mmap = true,
            Short('n') | Long("num-iters") => {
                n_iters = parse_uint(&mut parser, "num-iters")?;
            }
            Long("no-optimize") => optimize = false,
            Long("num-threads") => {
                num_threads = Some(parse_uint(&mut parser, "num-threads")?);
            }
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

  -n, --num-iters <n>
                 Number of times to evaluate model

  --no-optimize  Disable graph optimizations

  --num-threads  Specify number of threads to use

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

    DimSize::sort_dedup(&mut input_sizes);

    Ok(Args {
        input_sizes,
        mmap,
        model,
        n_iters,
        num_threads,
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

    // Indexes of entries in `dim_sizes` that didn't match any inputs.
    let mut unused_dim_sizes: HashSet<usize> = (0..dim_sizes.len()).collect();

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
                        if let Some((idx, dim_size)) = dim_sizes
                            .iter()
                            .enumerate()
                            .find(|(_i, ds)| ds.matches(name, dim_name))
                        {
                            unused_dim_sizes.remove(&idx);
                            dim_size.size
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

    // Error if specified dimension sizes were unused. This likely indicates a
    // typo in the name. Running the model with a default dimension size might
    // cause errors or less work (because a dimension has a smaller value than
    // intended).
    if let Some(idx) = unused_dim_sizes.into_iter().next() {
        let dim_size = &dim_sizes[idx];
        let err = if let Some(input_name) = &dim_size.input_name {
            format!(
                "Input and dim name \"{}.{}\" did not match any inputs",
                input_name, dim_size.dim_name
            )
        } else {
            format!(
                "Dim name \"{}\" did not match any inputs",
                dim_size.dim_name
            )
        };
        return Err(err.into());
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

    let run_opts = RunOptions {
        timing: args.timing,
        verbose: args.verbose,
        thread_pool: args
            .num_threads
            .map(|nt| ThreadPool::with_num_threads(nt as usize).into()),
        ..Default::default()
    };

    if let Err(err) = run_with_random_input(
        &model,
        &args.input_sizes,
        run_opts,
        args.n_iters,
        args.quiet,
    ) {
        // For readability, add a blank line after any output before the final
        // error.
        println!();
        return Err(err);
    }

    Ok(())
}
