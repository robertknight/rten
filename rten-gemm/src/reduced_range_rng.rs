use rten_tensor::RandomSource;
use rten_tensor::rng::XorShiftRng;

/// Random number generator which produces values with an optionally reduced
/// range.
///
/// This works around an issue under AVX2 where the `vpmaddubsw` instruction
/// can encounter saturation when adding two signed 16-bit values into a
/// 16-bit result. Each of the two 16-bit inputs are the result of a `u8 x
/// i8` multiplication. By limiting the range of either the u8 or i8 input,
/// saturation is avoided. This issue does not affect the VNNI instruction
/// used on newer x64 systems. It also does not affect Arm.
///
/// To match the behavior in ONNX Runtime's quantizer when
/// `reduce_range=True` is enabled, the range of whichever input are the
/// weights (usually the RHS) should be limited.
///
/// To avoid saturation we require `i16::MIN >= u8_val * i8_val * 2 <=
/// i16::MAX`. A suitable choice is to use i7/u7 values with ranges [-64,
/// 63] and [0, 127].
///
/// See also <https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html>.
pub struct ReducedRangeRng {
    reduce_range: bool,
    rng: XorShiftRng,
}

impl ReducedRangeRng {
    pub fn new(reduce_range: bool, seed: u64) -> Self {
        Self {
            rng: XorShiftRng::new(seed),
            reduce_range,
        }
    }
}

impl RandomSource<i8> for ReducedRangeRng {
    /// Return a random value in `[-64, 63]` (the i7 range).
    fn next(&mut self) -> i8 {
        if self.reduce_range {
            ((self.rng.next_u64() % 128) as i16 - 64i16) as i8
        } else {
            self.rng.next_u64() as i8
        }
    }
}

impl RandomSource<u8> for ReducedRangeRng {
    /// Return a random value in `[0, 127]` (the u7 range).
    fn next(&mut self) -> u8 {
        if self.reduce_range {
            (self.rng.next_u64() % 128) as u8
        } else {
            self.rng.next_u64() as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::RandomSource;

    use super::ReducedRangeRng;

    #[test]
    fn test_reduced_range_rng() {
        let mut rng = ReducedRangeRng::new(true, 1234);
        for _ in 0..100 {
            let x: i8 = rng.next();
            assert!(x >= -64 && x <= 63);

            let x: u8 = rng.next();
            assert!(x <= 127);
        }
    }
}
