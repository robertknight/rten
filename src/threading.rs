use std::env;
use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

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
/// To run your own tasks in this thread pool, you can use
/// [`ThreadPool::install`].
///
/// [rayon]: https://github.com/rayon-rs/rayon
pub fn thread_pool() -> &'static ThreadPool {
    static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();
    THREAD_POOL.get_or_init(|| {
        let physical_cpus = num_cpus::get_physical();

        let num_threads = if let Some(threads_var) = env::var_os("RTEN_NUM_THREADS") {
            let requested_threads: Result<usize, _> = threads_var.to_string_lossy().parse();
            match requested_threads {
                Ok(n_threads) => n_threads.clamp(1, num_cpus::get()),
                Err(_) => physical_cpus,
            }
        } else {
            physical_cpus
        };

        ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("rten-{}", index))
            .build()
            .expect("failed to initialize RTen thread pool")
    })
}
