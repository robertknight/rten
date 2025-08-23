use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

use rten::{
    DataType, Dimension, Model, ModelMetadata, ModelOptions, NodeId, RunOptions, ThreadPool, Value,
    ValueOrView,
};
use rten_tensor::prelude::*;
use rten_tensor::Tensor;
use safetensors::SafeTensors;

mod dim_size;
use dim_size::DimSize;

#[derive(Clone, Copy, Default, PartialEq)]
enum ProfileMode {
    #[default]
    None,

    /// Show a simple breakdown of execution time by operator.
    Basic,

    /// Show a detailed breakdown of execution time by operator and input shape.
    Detailed,
}

impl ProfileMode {
    fn next_level(self) -> ProfileMode {
        match self {
            Self::None => Self::Basic,
            Self::Basic => Self::Detailed,
            Self::Detailed => Self::Detailed,
        }
    }
}

struct Args {
    /// Model file to load.
    model: String,

    /// Whether to enable graph optimizations
    optimize: bool,

    /// Whether to enable prepacking of weights
    prepack_weights: bool,

    /// Run model and don't produce other output
    quiet: bool,

    /// Print output values produced by inference.
    print_outputs: bool,

    /// Record and display operator timing stats.
    profile_mode: ProfileMode,

    /// Enable verbose logging for model execution.
    verbose: bool,

    /// Sizes for dynamic dimensions of inputs.
    input_sizes: Vec<DimSize>,

    /// Path to Safetensors file containing map of input name to value.
    input_data: Option<String>,

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

    let mut input_data = None;
    let mut input_sizes = Vec::new();
    let mut mmap = false;
    let mut n_iters = 1;
    let mut num_threads = None;
    let mut optimize = true;
    let mut prepack_weights = false;
    let mut print_outputs = false;
    let mut profile_mode = ProfileMode::None;
    let mut quiet = false;
    let mut verbose = false;

    let parse_uint = |parser: &mut lexopt::Parser, opt_name| -> Result<u32, lexopt::Error> {
        let value = parser.value()?.string()?;
        value
            .parse()
            .map_err(|_| format!("Unable to parse numeric value for option '{}'", opt_name).into())
    };

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Short('d') | Long("data") => {
                input_data = Some(parser.value()?.string()?);
            }
            Long("mmap") => mmap = true,
            Short('n') | Long("num-iters") => {
                n_iters = parse_uint(&mut parser, "num-iters")?;
            }
            Long("no-optimize") => optimize = false,
            Short('t') | Long("num-threads") => {
                num_threads = Some(parse_uint(&mut parser, "num-threads")?);
            }
            Short('k') | Long("prepack") => prepack_weights = true,
            Short('p') | Long("profile") => profile_mode = profile_mode.next_level(),
            Long("print") => print_outputs = true,
            Short('q') | Long("quiet") => quiet = true,
            Short('v') | Long("verbose") => verbose = true,
            Short('V') | Long("version") => {
                println!("rten {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
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

  -d, --data <path>
                 Read values for input tensors from Safetensors file at the
                 given path. Tensor names in the file are used as input names.

  --mmap         Load model via memory mapping

  -n, --num-iters <n>
                 Number of times to evaluate model.

                 If zero, the model will be loaded and optimized, but not run.

  --no-optimize  Disable graph optimizations

  -k, --prepack  Enable prepacking of weights.

                 This requires additional memory but makes inference faster.

  --print        Print output tensor values.

  -p, --profile  Record and display operator timings.

                 If this flag is repeated, more detailed profiling information
                 is displayed.

  -q, --quiet    Run model and don't produce other output

  -s, --size <spec>
                 Specify size for a dynamic dimension in the form `dim_name=size`
                 or `input_name.dim_name=size`

  -t, --num-threads
                 Specify number of threads to use

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
        input_data,
        input_sizes,
        mmap,
        model,
        n_iters,
        num_threads,
        optimize,
        prepack_weights,
        print_outputs,
        quiet,
        profile_mode,
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

struct InputConfig {
    /// Dimension sizes to use when generating inputs with dimensions that have
    /// a dynamic size.
    dim_sizes: Vec<DimSize>,

    /// Map of input name to value.
    ///
    /// Inputs use values from this map if present, otherwise a random input is
    /// generated.
    values: HashMap<String, Value>,
}

/// Generate random inputs for `model` using shape metadata and heuristics,
/// run it, and print details of the output.
///
/// `dim_sizes` specifies the sizes for input dimensions with dynamic sizes.
fn run_model(
    model: &Model,
    input_config: &InputConfig,
    run_opts: RunOptions,
    n_iters: u32,
    quiet: bool,
    print_outputs: bool,
) -> Result<(), Box<dyn Error>> {
    // Names of all dynamic dimensions for which no size was explicitly
    // specified.
    let mut dynamic_dims_using_default_size: HashSet<String> = HashSet::new();

    // Indexes of entries in `dim_sizes` that didn't match any inputs.
    let mut unused_dim_sizes: HashSet<usize> = (0..input_config.dim_sizes.len()).collect();

    let mut input_generator = RandomInputGenerator::new();

    // Fetch or generate model inputs
    let inputs: Vec<(NodeId, ValueOrView)> = model.input_ids().iter().copied().try_fold(
        Vec::<(NodeId, ValueOrView)>::new(),
        |mut inputs, id| {
            let info = model.node_info(id).ok_or("Unable to get input info")?;
            let name = info.name().unwrap_or("(unnamed input)");
            let shape = info
                .shape()
                .ok_or(format!("Unable to get shape for input {}", name))?;
            let dtype = info.dtype();

            let value_or_view = if let Some(value) = input_config.values.get(name) {
                ValueOrView::View(value.as_view())
            } else {
                let tensor = input_generator.generate(
                    name,
                    dtype,
                    &shape,
                    &input_config.dim_sizes,
                    |dim_name, dim_size_idx| {
                        if let Some(idx) = dim_size_idx {
                            unused_dim_sizes.remove(&idx);
                        } else {
                            dynamic_dims_using_default_size.insert(dim_name.to_string());
                        }
                    },
                );
                ValueOrView::Value(tensor)
            };
            inputs.push((id, value_or_view));

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
        let dim_size = &input_config.dim_sizes[idx];
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

    let mut last_outputs = None;

    // Run duration in milliseconds.
    let mut durations: Vec<f32> = Vec::new();

    for iter_num in 1..=n_iters {
        let start = Instant::now();
        let outputs = model.run(inputs.clone(), model.output_ids(), Some(run_opts.clone()))?;
        let elapsed_ms = (start.elapsed().as_secs_f64() * 1000.0) as f32;

        if !quiet {
            println!(
                "  #{} - Model returned {} outputs in {:.2}ms.",
                iter_num,
                outputs.len(),
                elapsed_ms
            );
        }
        durations.push(elapsed_ms);
        last_outputs = Some(outputs);
    }

    if !quiet {
        // Print run timing variation statistics if we had multiple runs.
        if n_iters > 1 {
            let n_iters_float = n_iters as f32;
            let mean = durations.iter().sum::<f32>() / n_iters_float;
            let variance = durations
                .iter()
                .map(|dur| (dur - mean) * (dur - mean))
                .sum::<f32>()
                / n_iters_float;
            let std_dev = variance.sqrt();
            let min = durations
                .iter()
                .min_by(|a, b| f32::total_cmp(a, b))
                .unwrap();
            let max = durations
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
        if let Some(outputs) = last_outputs {
            for (i, (output, name)) in outputs.iter().zip(output_names).enumerate() {
                println!(
                    "  Output {i} \"{name}\" data type {} shape: {:?}",
                    output.dtype(),
                    output.shape()
                );

                if print_outputs {
                    println!("  Output {} value: {:?}", name, output);
                }
            }
        }
    }

    Ok(())
}

struct RandomInputGenerator {
    rng: fastrand::Rng,
}

impl RandomInputGenerator {
    fn new() -> Self {
        RandomInputGenerator {
            rng: fastrand::Rng::new(),
        }
    }

    /// Generate a random value for an input using the name, shape and dtype
    /// properties from the model as well as configuration provided when
    /// running the CLI.
    ///
    /// `on_resolve_size` is invoked for each dynamic dimension size that
    /// is resolved, specifying the dimension name and index of the entry in
    /// `dim_sizes` that was used, if any.
    fn generate(
        &mut self,
        name: &str,
        dtype: Option<DataType>,
        shape: &[Dimension],
        dim_sizes: &[DimSize],
        mut on_resolve_size: impl FnMut(&str, Option<usize>),
    ) -> Value {
        let resolved_shape: Vec<usize> = shape
            .iter()
            .map(|dim| match dim {
                Dimension::Symbolic(dim_name) => {
                    if let Some((idx, dim_size)) = dim_sizes
                        .iter()
                        .enumerate()
                        .find(|(_i, ds)| ds.matches(name, dim_name))
                    {
                        on_resolve_size(dim_name, Some(idx));
                        dim_size.size
                    } else {
                        on_resolve_size(dim_name, None);
                        1
                    }
                }
                Dimension::Fixed(size) => *size,
            })
            .collect();

        fn random_ints<T, F: FnMut() -> T>(shape: &[usize], gen: F) -> Value
        where
            Value: From<Tensor<T>>,
        {
            Tensor::from_simple_fn(shape, gen).into()
        }

        // Guess suitable content for the input based on its name.
        match name {
            // If this is a mask, use all ones on the assumption that we
            // don't want to mask anything out.
            name if name.ends_with("_mask") => Value::from(Tensor::full(&resolved_shape, 1i32)),

            // Inputs such as `token_type_ids`, `position_ids`, `input_ids`.
            // We use zero as a value that is likely to be valid for all
            // of these.
            name if name.ends_with("_ids") => Value::from(Tensor::<i32>::zeros(&resolved_shape)),

            // Optimum can export "merged" transformer models which have two
            // branches. One accepts KV-cache inputs and the other does not.
            // Set this to false as a "safer" value because we don't have
            // cached outputs from a previous run.
            "use_cache_branch" => Value::from(Tensor::from(0i32)),

            // For anything else, random values.
            _ => match dtype {
                // Generate floats in [0, 1]
                Some(DataType::Float) | None => {
                    Value::from(Tensor::from_simple_fn(&resolved_shape, || self.rng.f32()))
                }
                // Generate random values for int types. The default ranges
                // are intended to be suitable for many models, but there
                // ought to be a way to override them.
                Some(DataType::Int32) => random_ints(&resolved_shape, || self.rng.i32(0..256)),
                Some(DataType::Int8) => random_ints(&resolved_shape, || self.rng.i8(0..=127)),
                Some(DataType::UInt8) => random_ints(&resolved_shape, || self.rng.u8(0..=255)),
            },
        }
    }
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

/// Convert a tensor from a Safetensors file into an rten Tensor.
fn read_tensor<T, const ELEM_BYTES: usize>(
    view: safetensors::tensor::TensorView,
    convert: impl Fn([u8; ELEM_BYTES]) -> T,
) -> Tensor<T> {
    // We assume that safetensors has validated the length of the data.
    let (chunks, remainder) = view.data().as_chunks::<ELEM_BYTES>();
    assert!(remainder.is_empty());
    let data: Vec<T> = chunks.iter().copied().map(convert).collect();
    Tensor::from_data(view.shape(), data)
}

/// Read values for model inputs from a Safetensors file.
///
/// Returns a map of input name to value.
fn read_inputs_from_safetensors(path: &Path) -> Result<HashMap<String, Value>, Box<dyn Error>> {
    use safetensors::tensor::Dtype;

    let data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let mut result = HashMap::new();
    for (name, view) in tensors.iter() {
        let value: Value = match view.dtype() {
            Dtype::F32 => read_tensor::<f32, _>(view, f32::from_le_bytes).into(),
            Dtype::I32 => read_tensor::<i32, _>(view, i32::from_le_bytes).into(),
            _ => {
                return Err(format!("Unsupported tensor dtype {:?}", view.dtype()).into());
            }
        };
        result.insert(name.to_string(), value);
    }
    Ok(result)
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
fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("Invalid arguments: {}", err);
            std::process::exit(1);
        }
    };

    let mut model_opts = ModelOptions::with_all_ops();
    model_opts.enable_optimization(args.optimize);
    model_opts.prepack_weights(args.prepack_weights);

    let model = if args.mmap {
        unsafe { model_opts.load_mmap(&args.model) }
    } else {
        model_opts.load_file(&args.model)
    };

    let model = match model {
        Ok(model) => model,
        Err(err) => {
            eprintln!("Failed to load model \"{}\": {}", args.model, err);
            std::process::exit(1);
        }
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
        timing: args.profile_mode != ProfileMode::None,
        timing_by_shape: args.profile_mode == ProfileMode::Detailed,
        verbose: args.verbose,
        thread_pool: args
            .num_threads
            .map(|nt| ThreadPool::with_num_threads(nt as usize).into()),
        ..Default::default()
    };

    let mut input_values = HashMap::new();
    if let Some(data_path) = args.input_data {
        input_values = match read_inputs_from_safetensors(Path::new(&data_path)) {
            Ok(values) => values,
            Err(err) => {
                eprintln!("Reading inputs failed: {}", err);
                std::process::exit(1);
            }
        };
    }

    let inputs = InputConfig {
        dim_sizes: args.input_sizes,
        values: input_values,
    };

    if let Err(err) = run_model(
        &model,
        &inputs,
        run_opts,
        args.n_iters,
        args.quiet,
        args.print_outputs,
    ) {
        // For readability, add a blank line after any output before the final
        // error.
        println!();
        eprintln!("Model run failed: {}", err);
        std::process::exit(1);
    }
}
