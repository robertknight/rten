use std::env;
use std::sync::OnceLock;

/// A wrapper around the Rayon thread pool used to run models.
///
/// On platforms where threads are not supported (eg. WebAssembly) this runs
/// operations directly on the main thread.
pub struct ThreadPool {
    /// The wrapped thread pool, or None if we failed to construct one.
    pool: Option<rayon::ThreadPool>,
}

impl ThreadPool {
    /// Run a function in the thread pool.
    ///
    /// This corresponds to [`rayon::ThreadPool::install`], except on platforms
    /// where threading is not supported, where it just runs `op` directly.
    pub fn run<R: Send, Op: FnOnce() -> R + Send>(&self, op: Op) -> R {
        if let Some(pool) = self.pool.as_ref() {
            pool.install(op)
        } else {
            op()
        }
    }

    /// Create a thread pool with a given number of threads.
    pub fn with_num_threads(num_threads: usize) -> ThreadPool {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("rten-{}", index))
            .build();

        ThreadPool { pool: pool.ok() }
    }
}

/// Return the optimal number of cores to use for maximum performance.
///
/// This may be less than the total number of cores on systems with heterogenous
/// cores (eg. a mix of performance and efficiency).
fn optimal_core_count() -> u32 {
    #[allow(unused_mut)]
    let mut core_count = num_cpus::get_physical().max(1) as u32;

    #[cfg(target_os = "macos")]
    {
        use rten_simd::isa_detection::macos::sysctl_int;
        if let Ok(perf_core_count) = sysctl_int(c"hw.perflevel0.physicalcpu") {
            core_count = core_count.clamp(1, perf_core_count as u32);
        }
    }

    core_count
}

/// Return the [Rayon][rayon] thread pool which is used to execute RTen models.
///
/// This differs from Rayon's default global thread pool in that it is tuned for
/// CPU rather than IO-bound work by choosing a thread count based on the number
/// of physical rather than logical cores.
///
/// The thread count can be overridden at the process level by setting the
/// `RTEN_NUM_THREADS` environment variable, whose value must be a number
/// between 1 and the logical core count.
///
/// The thread count can be overridden for each model run by configuring a
/// custom thread pool in [`RunOptions`](crate::RunOptions).
///
/// To run your own tasks in this thread pool, you can use
/// [`ThreadPool::run`].
///
/// [rayon]: https://github.com/rayon-rs/rayon
pub fn thread_pool() -> &'static ThreadPool {
    static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();
    THREAD_POOL.get_or_init(|| {
        let physical_cpus = optimal_core_count();

        let num_threads = if let Some(threads_var) = env::var_os("RTEN_NUM_THREADS") {
            let requested_threads: Result<u32, _> = threads_var.to_string_lossy().parse();
            match requested_threads {
                Ok(n_threads) => n_threads.clamp(1, num_cpus::get() as u32),
                Err(_) => physical_cpus,
            }
        } else {
            physical_cpus
        };

        ThreadPool::with_num_threads(num_threads as usize)
    })
}

#[cfg(test)]
mod tests {
    use super::optimal_core_count;

    #[test]
    fn test_optimal_core_count() {
        let max_cores = num_cpus::get_physical() as u32;
        let opt_cores = optimal_core_count();
        assert!(opt_cores >= 1 && opt_cores <= max_cores);
    }
}
