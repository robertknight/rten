use crate::timer::Timer;

/// Run a benchmark function `f` for `trials` iterations and print statistics
/// about the run.
pub fn run_bench<F: FnMut()>(trials: usize, description: &str, mut f: F) {
    if trials == 0 {
        return;
    }

    let mut times = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut t = Timer::new();
        t.start();

        f();

        t.end();
        times.push(t.elapsed_ms());
    }

    times.sort_by(|a, b| a.total_cmp(b));
    let min = times.first().unwrap();
    let max = times.last().unwrap();

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
}
