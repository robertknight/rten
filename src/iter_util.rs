use rayon::prelude::*;

/// Wrapper around either a serial or parallel iterator, returned by
/// [`MaybeParIter::maybe_par_iter`].
pub enum MaybeParallel<PI: ParallelIterator, SI: Iterator<Item = PI::Item>> {
    Serial(SI),
    Parallel(PI),
}

impl<PI: ParallelIterator, SI: Iterator<Item = PI::Item>> MaybeParallel<PI, SI> {
    pub fn for_each<F: Fn(PI::Item) + Send + Sync>(self, f: F) {
        match self {
            MaybeParallel::Serial(iter) => iter.for_each(f),
            MaybeParallel::Parallel(iter) => iter.for_each(f),
        }
    }
}

/// Trait which allows use of Rayon parallelism to be conditionally enabled.
///
/// See https://crates.io/crates/rayon-cond for a more full-featured alternative.
pub trait MaybeParIter {
    type Item;
    type ParIter: ParallelIterator<Item = Self::Item>;
    type Iter: Iterator<Item = Self::Item>;

    /// Return an iterator which executes either in serial on the current
    /// thread, or in parallel in a Rayon thread pool if `parallel` is true.
    fn maybe_par_iter(self, parallel: bool) -> MaybeParallel<Self::ParIter, Self::Iter>;
}

impl<Item, I: rayon::iter::IntoParallelIterator<Item = Item> + IntoIterator<Item = Item>>
    MaybeParIter for I
{
    type Item = Item;
    type ParIter = I::Iter;
    type Iter = I::IntoIter;

    fn maybe_par_iter(self, parallel: bool) -> MaybeParallel<Self::ParIter, Self::Iter> {
        if parallel {
            MaybeParallel::Parallel(self.into_par_iter())
        } else {
            MaybeParallel::Serial(self.into_iter())
        }
    }
}

/// Unroll a loop 4x.
///
/// This is very similar to [`unroll_loop`] but uses a more aggressive approach
/// to unrolling which only supports a fixed unroll factor. Whereas
/// `unroll_loop` uses a hint (a `for` loop with a fixed iteration count) which
/// the compiler follows most of the time, this macro actually duplicates the
/// body 4x.
macro_rules! unroll_loop_x4 {
    ($range:expr, $loop_var:ident, $block:tt) => {
        let mut n = $range.len();
        let mut $loop_var = $range.start;

        while n >= 4 {
            $block;
            $loop_var += 1;
            $block;
            $loop_var += 1;
            $block;
            $loop_var += 1;
            $block;
            $loop_var += 1;
            n -= 4;
        }

        while n > 0 {
            $block;
            $loop_var += 1;
            n -= 1;
        }
    };
}

/// Generate an unrolled loop.
///
/// `$range` is a `Range` specifying the loop start and end. `$loop_var` is the
/// name of the variable containing the current iteration inside `$block`.
/// `$factor` should be a constant expression specifying the unroll factor,
/// typically a small value such as 4 or 8.
///
/// This macro generates a "hint" in the form of a `for` loop with a const
/// iteration count which the compiler follows in most cases. If it doesn't,
/// and you're sure you still need unrolling, consider [`unroll_loop_x4`]
/// instead.
macro_rules! unroll_loop {
    ($range:expr, $loop_var:ident, $factor: expr, $block:tt) => {
        let mut n = $range.len();
        let mut $loop_var = $range.start;
        while n >= $factor {
            for _i in 0..$factor {
                $block;
                $loop_var += 1;
            }
            n -= $factor;
        }
        while n > 0 {
            $block;

            $loop_var += 1;
            n -= 1;
        }
    };
}

#[allow(unused_imports)]
pub(crate) use {unroll_loop, unroll_loop_x4};

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::{unroll_loop, MaybeParIter};

    #[test]
    fn test_maybe_par_iter() {
        let count = AtomicU32::new(0);
        (0..1000).maybe_par_iter(false).for_each(|_| {
            count.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(count.load(Ordering::SeqCst), 1000);

        let count = AtomicU32::new(0);
        (0..1000).maybe_par_iter(true).for_each(|_| {
            count.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(count.load(Ordering::SeqCst), 1000);
    }

    #[test]
    fn test_unroll_loop() {
        let mut items: Vec<i32> = Vec::new();
        unroll_loop!(0..10, i, 4, {
            items.push(i);
        });
        assert_eq!(items, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
