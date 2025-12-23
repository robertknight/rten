//! Loop-unrolling macros.

/// Unroll a loop 4x.
///
/// This is very similar to [`unroll_loop`] but uses a more aggressive approach
/// to unrolling which only supports a fixed unroll factor. Whereas
/// `unroll_loop` uses a hint (a `for` loop with a fixed iteration count) which
/// the compiler follows most of the time, this macro actually duplicates the
/// body 4x.
#[macro_export]
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
#[macro_export]
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

pub use {unroll_loop, unroll_loop_x4};

#[cfg(test)]
mod tests {
    #[test]
    fn test_unroll_loop() {
        let mut items: Vec<i32> = Vec::new();
        unroll_loop!(0..10, i, 4, {
            items.push(i);
        });
        assert_eq!(items, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
