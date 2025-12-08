use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::Debug;
use std::path::Path;
use std::time::Instant;

use rten::{
    DataType, Dimension, Model, ModelMetadata, ModelOptions, NodeId, RunOptions, ThreadPool, Value,
    ValueOrView,
};
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};
use safetensors::SafeTensors;

mod dim_size;
use dim_size::DimSize;
mod input_info;
use input_info::{print_input_output_list, print_input_shapes, print_output_shapes};

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

    /// run shape and type inference prior to optimization.
    ///
    /// This is an experimental option that can enable more effective model
    /// optimization. See https://github.com/robertknight/rten/pull/1124.
    #[argh(switch)]
    infer_shapes: bool,

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
        print_input_shapes(model, &inputs);
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

#[derive(Debug)]
enum GenerateError {
    UnsupportedDataType(DataType),
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDataType(dtype) => {
                write!(f, "generation of {dtype} inputs is not supported")
            }
        }
    }
}

impl Error for GenerateError {}

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
    ) -> Result<Value, GenerateError> {
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

        fn random_ints<T, F: FnMut() -> T>(shape: &[usize], generate: F) -> Value
        where
            Value: From<Tensor<T>>,
        {
            Tensor::from_simple_fn(shape, generate).into()
        }

        // Guess suitable content for the input based on its name.
        let value = match name {
            // If this is a mask, use all ones on the assumption that we
            // don't want to mask anything out.
            name if name.ends_with("_mask") && matches!(dtype, Some(DataType::Int32) | None) => {
                Value::from(Tensor::full(&resolved_shape, 1i32))
            }

            // Inputs such as `token_type_ids`, `position_ids`, `input_ids`.
            // We use zero as a value that is likely to be valid for all
            // of these.
            name if name.ends_with("_ids") && matches!(dtype, Some(DataType::Int32) | None) => {
                Value::from(Tensor::<i32>::zeros(&resolved_shape))
            }

            // Optimum can export "merged" transformer models which have two
            // branches. One accepts KV-cache inputs and the other does not.
            // Set this to false as a "safer" value because we don't have
            // cached outputs from a previous run.
            "use_cache_branch" if matches!(dtype, Some(DataType::Int32) | None) => {
                Value::from(Tensor::from(0i32))
            }

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
                Some(dtype) => {
                    return Err(GenerateError::UnsupportedDataType(dtype));
                }
            },
        };

        Ok(value)
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

/// Read tensor values from a Safetensors file.
///
/// Returns a map of input name to value.
fn read_safetensors(path: &Path) -> Result<HashMap<String, Value>, Box<dyn Error>> {
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
                eprintln!("Invalid size specification '{}': {}", size_str, err);
                std::process::exit(1);
            }
        }
    }
    DimSize::sort_dedup(&mut input_sizes);

    // Parse profile mode from switch count
    let profile_mode = match args.profile {
        0 => ProfileMode::None,
        1 => ProfileMode::Basic,
        _ => ProfileMode::Detailed,
    };

    let mut model_opts = ModelOptions::with_all_ops();
    model_opts.enable_optimization(!args.no_optimize);
    model_opts.enable_shape_inference(args.infer_shapes);
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
        .map(|nt| ThreadPool::with_num_threads(nt as usize).into());
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
