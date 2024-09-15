use crate::RandomSource;

/// Simple, non-cryptographically secure random number generator.
///
/// See <https://en.wikipedia.org/wiki/Xorshift>.
pub struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    pub fn new(seed: u64) -> XorShiftRng {
        XorShiftRng { state: seed }
    }

    /// Return a random value in the range [0, 2^64]
    pub fn next_u64(&mut self) -> u64 {
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
        let val = self.next_u64() >> (64 - n_bits);
        (val as f32) * scale
    }

    /// Return an infinite iterator that yields random values of type `T`.
    pub fn iter<T>(&mut self) -> impl Iterator<Item = T> + '_
    where
        Self: RandomSource<T>,
    {
        std::iter::from_fn(|| Some(self.next()))
    }
}

impl RandomSource<f32> for XorShiftRng {
    fn next(&mut self) -> f32 {
        self.next_f32()
    }
}

macro_rules! impl_random_source {
    ($ty:ty) => {
        impl RandomSource<$ty> for XorShiftRng {
            fn next(&mut self) -> $ty {
                // Take the least significant bits of the 64bit value as the
                // result.
                self.next_u64() as $ty
            }
        }
    };
}

impl_random_source!(u8);
impl_random_source!(i8);
impl_random_source!(i16);
impl_random_source!(u16);
impl_random_source!(i32);
impl_random_source!(u32);

#[cfg(test)]
mod tests {
    use super::XorShiftRng;

    #[test]
    fn test_f32() {
        let mut rng = XorShiftRng::new(1234);
        let x: Vec<f32> = rng.iter().take(10).collect();
        assert_eq!(
            x,
            &[
                7.2381226e-8,
                0.12971127,
                0.44675463,
                6.69676e-5,
                0.44387037,
                0.24518594,
                0.84056354,
                0.9960614,
                0.32433507,
                0.9239961
            ]
        );
    }

    #[test]
    fn test_i8() {
        let mut rng = XorShiftRng::new(1234);
        let x: Vec<i8> = rng.iter().take(10).collect();
        assert_eq!(x, &[91, 123, 3, -73, 8, -102, -19, 118, 88, 58]);
    }

    #[test]
    fn test_u8() {
        let mut rng = XorShiftRng::new(1234);
        let x: Vec<u8> = rng.iter().take(10).collect();
        assert_eq!(x, &[91, 123, 3, 183, 8, 154, 237, 118, 88, 58]);
    }

    #[test]
    fn test_i32() {
        let mut rng = XorShiftRng::new(1234);
        let x: Vec<i32> = rng.iter().take(10).collect();
        assert_eq!(
            x,
            &[
                -533893029,
                -1874043781,
                -2014135805,
                -1501708361,
                330844424,
                1872264090,
                -1812926995,
                -306325642,
                692957528,
                -1439925190
            ]
        );
    }
}
