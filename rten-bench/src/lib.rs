use std::fmt::Display;
use std::time::Instant;

/// Statistics from a benchmark run. All fields are durations in milliseconds.
#[derive(Default)]
pub struct BenchStats {
    /// Duration of longest run.
    pub max: f32,

    /// Mean duration.
    pub mean: f32,

    /// Median duration.
    pub median: f32,

    /// Minimum duration.
    pub min: f32,

    /// Variance of durations.
    pub var: f32,
}

/// Run a benchmark function `f` for `trials` iterations and print statistics
/// about the run.
pub fn run_bench<F: FnMut(), D: Display>(trials: usize, description: D, mut f: F) -> BenchStats {
    if trials == 0 {
        return BenchStats::default();
    }

    let mut times = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();

        f();

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        times.push(duration_ms as f32);
    }

    times.sort_by(|a, b| a.total_cmp(b));
    let min = times.first().copied().unwrap();
    let max = times.last().copied().unwrap();

    let mid = times.len() / 2;
    let median = if times.len() % 2 == 1 {
        times[mid]
    } else {
        (times[mid] + times[mid + 1]) / 2.
    };
    let mean = times.iter().sum::<f32>() / times.len() as f32;
    let var = times.iter().map(|x| (x - mean).abs()).sum::<f32>() / times.len() as f32;

    println!(
        "{}. mean {:.3}ms median {:.3} var {:.3} min {:.3} max {:.3}",
        description, mean, median, var, min, max
    );

    BenchStats {
        max,
        mean,
        median,
        min,
        var,
    }
}
