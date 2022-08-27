/// Simple, non-cryptographically secure random number generator.
///
/// See https://en.wikipedia.org/wiki/Xorshift
pub struct XorShiftRNG {
    state: u64,
}

impl XorShiftRNG {
    pub fn new(seed: u64) -> XorShiftRNG {
        XorShiftRNG { state: seed }
    }

    /// Return a random value in the range [0, 2^64]
    pub fn next(&mut self) -> u64 {
        let mut tmp = self.state;
        tmp ^= tmp << 13;
        tmp ^= tmp >> 7;
        tmp ^= tmp << 17;
        self.state = tmp;
        tmp
    }

    /// Return a random value in the range [0, 1]
    pub fn next_f32(&mut self) -> f32 {
        // Number of most significant bits to use
        let n_bits = 40;
        let scale = 1.0 / (1u64 << n_bits) as f32;
        let val = self.next() >> (64 - n_bits);
        (val as f32) * scale
    }
}

