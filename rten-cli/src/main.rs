use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::Debug;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use rten::{
    Model, ModelMetadata, ModelOptions, NodeId, RunOptions, ThreadPool, Value, ValueOrView,
};
use rten_base::num::AsUsize;
use rten_tensor::TensorView;
use rten_tensor::prelude::*;

mod dim_size;
use dim_size::DimSize;
mod input_generator;
use input_generator::RandomInputGenerator;
mod input_info;
use input_info::{print_input_output_list, print_input_shapes, print_output_shapes};
mod input_range;
use input_range::InputRange;
mod name_value;

#[derive(Clone, Copy, Default, PartialEq)]
enum ProfileMode {
    #[default]
    None,

    /// Show a simple breakdown of execution time by operator.
    Basic,

    /// Show a detailed breakdown of execution time by operator and input shape.
    Detailed,
}

/// Inspect and run ONNX or RTen models.
#[derive(argh::FromArgs)]
struct Args {
    /// path to '.onnx' or '.rten' model to inspect and run
    #[argh(positional)]
    model: Option<String>,

    /// check outputs against the values provided in the Safetensors file specified by the given path. This must be used together with `--inputs`.
    #[argh(option)]
    check_outputs: Option<String>,

    /// set shape and type inference mode.
    ///
    /// Can be "off", "on" (best-effort) or "strict". If "strict", model will
    /// fail to load if shape inference is not complete.
    #[argh(option)]
    infer_shapes: Option<InferShapesMode>,

    /// read values for input tensors from Safetensors file at the given path. Tensor names in the file are used as input names.
    #[argh(option, short = 'i')]
    inputs: Option<String>,

    /// load model via memory mapping
    #[argh(switch)]
    mmap: bool,

    /// number of times to evaluate model. If zero, the model will be loaded and optimized, but not run.
    #[argh(option, short = 'n', default = "1")]
    n_iters: u32,

    /// disable graph optimizations
    #[argh(switch)]
    no_optimize: bool,

    /// enable prepacking of weights. This requires additional memory but makes inference faster.
    #[argh(switch, short = 'k')]
    prepack: bool,

    /// print output tensor values
    #[argh(switch)]
    print: bool,

    /// record and display operator timings. Repeat for more detailed profiling.
    #[argh(switch, short = 'p')]
    profile: u32,

    /// run model and don't produce other output
    #[argh(switch, short = 'q')]
    quiet: bool,

    /// specify the range of randomly generated values for an input in the form `input_name=min:max`. Can be specified multiple times.
    /// Input names may be quoted (eg. `"input.one"=0:10`).
    #[argh(option, short = 'r')]
    range: Vec<String>,

    /// specify size for a dynamic dimension in the form `dim_name=size` or `input_name.dim_name=size`. Can be specified multiple times.
    /// Input and dimension names may be quoted (eg. `"input.one"."dim.two"=3`).
    #[argh(option, short = 's')]
    size: Vec<String>,

    /// specify number of threads to use
    #[argh(option, short = 't')]
    num_threads: Option<u32>,

    /// enable verbose logging
    #[argh(switch, short = 'v')]
    verbose: bool,

    /// display RTen version
    #[argh(switch, short = 'V')]
    version: bool,
}

#[derive(Clone, Copy, Default, PartialEq)]
enum InferShapesMode {
    Off,
    #[default] // Matches ModelOptions defaults.
    On,
    Strict,
}

impl FromStr for InferShapesMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            "strict" => Ok(Self::Strict),
            text => Err(format!(
                "Unsupported shape inference mode \"{text}\". Valid options are 'on', 'off' or 'strict'."
            )),
        }
    }
}

fn format_param_count(n: usize) -> String {
    if n > 1_000_000 {
        format!("{:.1} M", n as f32 / 1_000_000.)
    } else {
        format!("{:.1} K", n as f32 / 1000.)
    }
}

fn print_metadata(metadata: &ModelMetadata) {
    println!("Metadata:");

    let mut fields: Vec<_> = metadata.fields().collect();
    fields.sort_by_key(|(field, _val)| *field);

    for (name, value) in fields {
        println!("  {}: {}", name, value);
    }
}

struct InputConfig {
    /// Dimension sizes to use when generating inputs with dimensions that have
    /// a dynamic size.
    dim_sizes: Vec<DimSize>,

    /// Ranges of values to use when generating random inputs.
    ranges: Vec<InputRange>,

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
    expected_outputs: Option<HashMap<String, Value>>,
) -> Result<(), Box<dyn Error>> {
    // Names of all dynamic dimensions for which no size was explicitly
    // specified.
    let mut dynamic_dims_using_default_size: HashSet<String> = HashSet::new();

    // Indexes of entries in `dim_sizes` that didn't match any inputs.
    let mut unused_dim_sizes: HashSet<usize> = (0..input_config.dim_sizes.len()).collect();

    // Indexes of entries in `ranges` that didn't match any inputs.
    let mut unused_ranges: HashSet<usize> = (0..input_config.ranges.len()).collect();

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
                let range = input_config
                    .ranges
                    .iter()
                    .enumerate()
                    .find(|(_i, range)| range.matches(name));
                if let Some((idx, _)) = range {
                    unused_ranges.remove(&idx);
                }

                let tensor = input_generator.generate(
                    name,
                    dtype,
                    &shape,
                    &input_config.dim_sizes,
                    range.map(|(_i, range)| range),
                    |dim_name, dim_size_idx| {
                        if let Some(idx) = dim_size_idx {
                            unused_dim_sizes.remove(&idx);
                        } else {
                            dynamic_dims_using_default_size.insert(dim_name.to_string());
                        }
                    },
                )?;
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
                "Input and dim name \"{}.{}\" does not match any inputs",
                input_name, dim_size.dim_name
            )
        } else {
            format!(
                "Dim name \"{}\" does not match any inputs",
                dim_size.dim_name
            )
        };
        return Err(err.into());
    }

    // Error if specified value ranges were unused, as this likely indicates a
    // typo in the input name.
    if let Some(idx) = unused_ranges.into_iter().next() {
        let range = &input_config.ranges[idx];
        let err = format!(
            "Input name \"{}\" does not match any inputs",
            range.input_name
        );
        return Err(err.into());
    }

    if !quiet {
        print_input_shapes(model, &inputs);
    }

    // Run model and summarize outputs.
    if !quiet {
        println!();
    }

    let mut last_outputs = None;

    // Run duration in milliseconds.
    let mut durations: Vec<f32> = Vec::new();

    // Take views of owned inputs to avoid copying potentially large input
    // values in the loop, which might distort timings.
    let input_views: Vec<(NodeId, ValueOrView)> = inputs
        .iter()
        .map(|(id, value)| (*id, value.as_view().into()))
        .collect();
    for iter_num in 1..=n_iters {
        let start = Instant::now();
        let outputs = model.run(
            input_views.clone(),
            model.output_ids(),
            Some(run_opts.clone()),
        )?;
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

    if !quiet && let Some(outputs) = last_outputs {
        // Print basic information about the output.
        print_output_shapes(model, &outputs);

        for (output, name) in outputs.iter().zip(output_names) {
            // Print a debug representation of the output.
            if print_outputs {
                println!("  Output {} value: {:?}", name, output);
            }

            // Compare output against expected value.
            if let Some(expected) = expected_outputs.as_ref().and_then(|eo| eo.get(&name)) {
                if expected.shape() != output.shape() {
                    println!(
                        "  Output \"{name}\" shape {:?} does not match expected {:?}",
                        output.shape(),
                        expected.shape()
                    );
                    continue;
                } else if expected.dtype() != output.dtype() {
                    println!(
                        "  Output \"{name}\" dtype {:?} does not match expected {:?}",
                        output.dtype(),
                        expected.dtype()
                    );
                    continue;
                }

                let compare_result = match (output, expected) {
                    (Value::FloatTensor(actual), Value::FloatTensor(expected)) => {
                        compare_tensors(actual.view(), expected.view(), |x, y| (x - y).abs())
                    }
                    _ => {
                        eprintln!("  Unable to compare outputs. Unsupported tensor types.");
                        continue;
                    }
                };
                println!(
                    "  Output \"{name}\" vs expected: max diff {:.6}",
                    compare_result.max_diff
                );
            }
        }
    }

    Ok(())
}

struct CompareMetrics {
    /// Maximum absolute difference between any corresponding pair of elements.
    max_diff: f32,
}

/// Compute metrics for the difference between elements of `actual` and
/// `expected`, which must have the same shape.
fn compare_tensors<T: Copy + Debug>(
    actual: TensorView<T>,
    expected: TensorView<T>,
    diff: impl Fn(T, T) -> f32,
) -> CompareMetrics {
    assert_eq!(actual.shape(), expected.shape());

    let mut max_diff = 0.0f32;
    for (x, y) in actual.iter().zip(expected.iter()) {
        let diff = diff(*x, *y);
        max_diff = max_diff.max(diff);
    }
    CompareMetrics { max_diff }
}

/// Convert a tensor read from a Safetensors file into an rten [`Value`].
fn rten_value(value: rten_serialize::Value) -> Result<Value, Box<dyn Error>> {
    use rten_serialize::Value as SV;

    let value = match value {
        SV::Float32(tensor) => Value::from(tensor),
        SV::Int32(tensor) => Value::from(tensor),
        SV::Int8(tensor) => Value::from(tensor),
        SV::UInt8(tensor) => Value::from(tensor),
        other => {
            return Err(format!("Unsupported tensor dtype {:?}", other.dtype()).into());
        }
    };
    Ok(value)
}

/// Read tensor values from a Safetensors file.
///
/// Returns a map of input name to value.
fn read_safetensors(path: &Path) -> Result<HashMap<String, Value>, Box<dyn Error>> {
    let tensors = rten_serialize::safetensors::read_from_file(path)?;
    tensors
        .into_iter()
        .map(|(name, value)| Ok((name, rten_value(value)?)))
        .collect()
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
    let args: Args = argh::from_env();

    // Handle --version flag
    if args.version {
        println!("rten {}", env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }

    // Require model argument if not showing version
    let model_path = match &args.model {
        Some(m) => m,
        None => {
            eprintln!("Error: missing required argument: <model>");
            eprintln!("Run with --help for usage information");
            std::process::exit(1);
        }
    };

    // Parse dimension sizes from string arguments
    let mut input_sizes = Vec::new();
    for size_str in &args.size {
        match DimSize::parse(size_str) {
            Ok(size) => input_sizes.push(size),
            Err(err) => {
                eprintln!("Invalid --size argument: {}", err);
                std::process::exit(1);
            }
        }
    }
    DimSize::sort_dedup(&mut input_sizes);

    // Parse input value ranges from string arguments
    let mut input_ranges = Vec::new();
    for range_str in &args.range {
        match InputRange::parse(range_str) {
            Ok(range) => input_ranges.push(range),
            Err(err) => {
                eprintln!("Invalid --range argument: {}", err);
                std::process::exit(1);
            }
        }
    }
    InputRange::sort_dedup(&mut input_ranges);

    // Parse profile mode from switch count
    let profile_mode = match args.profile {
        0 => ProfileMode::None,
        1 => ProfileMode::Basic,
        _ => ProfileMode::Detailed,
    };

    let mut model_opts = ModelOptions::with_all_ops();
    model_opts.enable_optimization(!args.no_optimize);
    model_opts.shape_inference(match args.infer_shapes.unwrap_or_default() {
        InferShapesMode::Off => rten::ShapeInferenceMode::Off,
        InferShapesMode::On => rten::ShapeInferenceMode::On,
        InferShapesMode::Strict => rten::ShapeInferenceMode::Strict,
    });
    model_opts.prepack_weights(args.prepack);

    let model = if args.mmap {
        unsafe { model_opts.load_mmap(model_path) }
    } else {
        model_opts.load_file(model_path)
    };

    let model = match model {
        Ok(m) => m,
        Err(err) => {
            eprintln!("Failed to load model \"{}\": {}", model_path, err);
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

    let thread_pool = args
        .num_threads
        .map(|nt| ThreadPool::with_num_threads(nt.as_usize()).into());
    let run_opts = RunOptions::default()
        .with_timing(profile_mode != ProfileMode::None)
        .with_timing_by_shape(profile_mode == ProfileMode::Detailed)
        .with_verbose(args.verbose)
        .with_thread_pool(thread_pool);

    // Read values for inputs, if provided.
    let mut input_values = HashMap::new();
    if let Some(data_path) = args.inputs {
        input_values = match read_safetensors(Path::new(&data_path)) {
            Ok(values) => values,
            Err(err) => {
                eprintln!("Reading inputs failed: {}", err);
                std::process::exit(1);
            }
        };
    }

    // Read expected values for outputs, if provided.
    let mut expected_outputs = None;
    if let Some(data_path) = args.check_outputs {
        expected_outputs = match read_safetensors(Path::new(&data_path)) {
            Ok(values) => Some(values),
            Err(err) => {
                eprintln!("Reading expected outputs failed: {}", err);
                std::process::exit(1);
            }
        };
    }

    let inputs = InputConfig {
        dim_sizes: input_sizes,
        ranges: input_ranges,
        values: input_values,
    };

    if let Err(err) = run_model(
        &model,
        &inputs,
        run_opts,
        args.n_iters,
        args.quiet,
        args.print,
        expected_outputs,
    ) {
        // For readability, add a blank line after any output before the final
        // error.
        println!();
        eprintln!("Model run failed: {}", err);
        std::process::exit(1);
    }
}
