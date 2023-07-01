use std::time::Instant;

/// Utility for recording the cumulative time spent in an operation.
pub struct Timer {
    start: Option<Instant>,
    elapsed: u64,
}

impl Timer {
    /// Create a new, inactive timer with zero elapsed time
    pub fn new() -> Timer {
        Timer {
            start: None,
            elapsed: 0,
        }
    }

    /// Start the timer, or reset it if already active
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop active timer and add elapsed time to the total returned by `elapsed`
    pub fn end(&mut self) {
        if let Some(start) = self.start {
            self.elapsed += start.elapsed().as_micros() as u64;
        }
    }

    /// Return the cumulative elapsed time between calls to `start` and `end`
    /// in milliseconds.
    pub fn elapsed_ms(&self) -> f32 {
        (self.elapsed as f32) / 1000.0
    }

    /// Return the cumulative elapsed time between calls to `start` and `end`
    /// in seconds.
    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed as f32 / 1000_000.0
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}
