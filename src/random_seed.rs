//! Seeds for random number generation in operators.

use std::hash::{BuildHasher, Hasher, RandomState};

/// Seed for a random number generator.
///
/// Seeds can be combined with other values to derive new seeds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RandomSeed(u64);

impl RandomSeed {
    /// Create a seed from a `u64` value.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Generate a new random seed.
    pub fn random() -> Self {
        // `RandomState` keys are randomly initialized per thread and change on
        // each call, so this produces a different seed each time.
        Self(RandomState::new().build_hasher().finish())
    }

    /// Create a seed from an `f32` value.
    ///
    /// This is used for the seed attributes of ONNX operators, which are
    /// specified as floats.
    pub fn from_f32(seed: f32) -> Self {
        Self(seed.to_bits() as u64)
    }

    /// Return the seed as a `u64` value.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Derive a new seed by combining this seed with `value`.
    pub fn mix(self, value: u64) -> Self {
        // Combine using the SplitMix64 finalizer.
        let mut z = self.0 ^ value.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        Self(z ^ (z >> 31))
    }

    /// Derive a new seed by combining this seed with a hash of `s`.
    pub fn mix_str(self, s: &str) -> Self {
        self.mix(fnv1a_hash(s))
    }
}

/// Return the 64-bit FNV-1a hash of a string.
fn fnv1a_hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::{RandomSeed, fnv1a_hash};

    #[test]
    fn test_random() {
        assert_ne!(RandomSeed::random(), RandomSeed::random());
    }

    #[test]
    fn test_from_f32() {
        assert_eq!(RandomSeed::from_f32(1.0).as_u64(), 0x3f800000);
    }

    #[test]
    fn test_mix() {
        // Compare against fixed values to detect unintended changes to seed
        // derivation.
        assert_eq!(RandomSeed::new(0).mix(0).as_u64(), 0xe220a8397b1dcdaf);
        assert_ne!(RandomSeed::new(1).mix(0), RandomSeed::new(0).mix(1));
    }

    #[test]
    fn test_mix_str() {
        assert_eq!(
            RandomSeed::new(0).mix_str("a"),
            RandomSeed::new(0).mix(0xaf63dc4c8601ec8c)
        );
        assert_ne!(
            RandomSeed::new(0).mix_str("a"),
            RandomSeed::new(0).mix_str("b")
        );
    }

    #[test]
    fn test_fnv1a_hash() {
        // Reference values for FNV-1a.
        assert_eq!(fnv1a_hash(""), 0xcbf29ce484222325);
        assert_eq!(fnv1a_hash("a"), 0xaf63dc4c8601ec8c);
    }
}
