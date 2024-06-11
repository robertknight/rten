//! Record timing metrics during generation.

use std::time::Duration;

/// Records timing metrics for a generation loop.
///
/// Metrics are separated into a _warmup_ phase and a _main_ phase. The
/// _warmup_ phase consists of the first step and the _main_ phase consists
/// of the remaining steps. This is because the first step will do extra
/// processing which is then reused for the remaining steps. As a result the
/// first step can be much slower.
///
/// This is used via [`GeneratorUtils::profile`](crate::GeneratorUtils::profile).
#[derive(Clone)]
pub struct Metrics {
    /// Duration times for each step in microseconds, excluding the warmup.
    durations: Vec<Duration>,

    /// Duration for the warmup step.
    warmup_duration: Option<Duration>,
}

impl Metrics {
    /// Create an empty metrics container.
    pub fn new() -> Metrics {
        Metrics {
            durations: Vec::new(),
            warmup_duration: None,
        }
    }

    /// Record the duration of a single run, in microseconds.
    ///
    /// The first run is recorded as the warmup step, subsequent steps are
    /// recorded as main steps.
    pub fn add_step_duration(&mut self, duration: Duration) {
        if self.warmup_duration.is_some() {
            self.durations.push(duration);
        } else {
            self.warmup_duration = Some(duration);
        }
    }

    /// Return the duration of the first or "warmup" step.
    pub fn warmup_duration(&self) -> Option<Duration> {
        self.warmup_duration
    }

    /// Return the durations recorded for each step, excluding the warmup.
    pub fn step_durations(&self) -> &[Duration] {
        &self.durations
    }

    /// Return the total duration, including the warmup.
    pub fn total_duration(&self) -> Duration {
        self.durations.iter().sum::<Duration>() + self.warmup_duration.unwrap_or(Duration::ZERO)
    }

    /// Return the total duration, excluding the warmup.
    pub fn total_main_duration(&self) -> Duration {
        self.durations.iter().sum()
    }

    /// Return the mean generation time in milliseconds, excluding the warmup.
    pub fn mean_duration(&self) -> f32 {
        let total_ms = self.total_main_duration().as_secs_f64() * 1000.0;
        (total_ms / self.durations.len() as f64) as f32
    }

    /// Return the mean number of tokens generated for each second, excluding
    /// the warmup.
    pub fn tokens_per_second(&self) -> f32 {
        self.durations.len() as f32 / self.total_main_duration().as_secs_f32()
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics::new()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::Metrics;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr, $threshold: expr) => {
            let a = $a;
            let b = $b;
            let threshold = $threshold;
            assert!(
                (a - b).abs() < threshold,
                "values {} and {} not approximately equal",
                a,
                b
            )
        };
    }

    #[test]
    fn test_metrics() {
        let ms = Duration::from_millis;

        let mut metrics = Metrics::new();

        // Add slower warmup step.
        metrics.add_step_duration(ms(200));

        // Add faster subsequent steps.
        metrics.add_step_duration(ms(110));
        metrics.add_step_duration(ms(90));

        assert_eq!(metrics.warmup_duration(), Some(ms(200)));
        assert_eq!(metrics.step_durations(), &[ms(110), ms(90)]);
        assert_eq!(metrics.total_duration(), ms(400));
        assert_eq!(metrics.total_main_duration(), ms(200));
        assert_approx_eq!(metrics.mean_duration(), 100.0, 1e-5);
        assert_approx_eq!(metrics.tokens_per_second(), 10.0, 1e-5);
    }
}
