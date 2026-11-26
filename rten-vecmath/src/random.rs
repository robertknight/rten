//! Vectorized random number generation.
//!
//! Random bits are generated using the Philox4x32-10 counter-based generator
//! (see "Parallel Random Numbers: As Easy as 1, 2, 3", Salmon et al. 2011).
//! Philox maps a 128-bit counter and 64-bit key to 128 random bits. Element
//! `i` of the output sequence depends only on the seed and `i`, so the
//! sequence is independent of the SIMD vector width and can be generated in
//! chunks, in parallel.

use std::f32::consts::TAU;
use std::mem::MaybeUninit;

use rten_simd::functional::array_map;
use rten_simd::ops::{BitOps, FloatOps, IntOps, MulHigh, NumOps, ToFloat};
use rten_simd::{Isa, Simd, SimdOp, SimdUnaryOp, SliceWriter};

use crate::{Cos, Ln, Sin};

// Philox4x32 multipliers and Weyl sequence constants for key schedule.
const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;
const PHILOX_W0: u32 = 0x9E3779B9;
const PHILOX_W1: u32 = 0xBB67AE85;
const PHILOX_ROUNDS: usize = 10;

/// Number of Philox counters used to generate each block of output.
///
/// This must be a multiple of the number of `u32` lanes in a SIMD vector for
/// all ISAs.
const BLOCK_COUNTERS: usize = 16;

/// Number of output elements in each block.
///
/// Element `k` of block `b` is generated from word `k / BLOCK_COUNTERS` of the
/// output for counter `b * BLOCK_COUNTERS + k % BLOCK_COUNTERS`. This layout
/// allows each SIMD vector of output to be generated from contiguous counters
/// in a vector, without transposing the generator's output.
const BLOCK_SIZE: usize = BLOCK_COUNTERS * 4;

/// Compute the Philox4x32-10 function for `G` vectors of counters.
///
/// `ctr` contains the four 32-bit words of the counter for each lane. Rounds
/// are interleaved across the `G` vectors, which are independent, to increase
/// instruction-level parallelism.
#[inline(always)]
fn philox4x32<I: Isa, const G: usize>(
    isa: I,
    mut ctr: [[I::U32; 4]; G],
    key: [u32; 2],
) -> [[I::U32; 4]; G] {
    let ops = isa.u32();

    let m0 = ops.splat(PHILOX_M0);
    let m1 = ops.splat(PHILOX_M1);
    let w0 = ops.splat(PHILOX_W0);
    let w1 = ops.splat(PHILOX_W1);
    let mut k0 = ops.splat(key[0]);
    let mut k1 = ops.splat(key[1]);

    for _ in 0..PHILOX_ROUNDS {
        ctr = array_map(
            ctr,
            #[inline(always)]
            |c| {
                let lo0 = ops.mul(m0, c[0]);
                let hi0 = ops.mul_high(m0, c[0]);
                let lo1 = ops.mul(m1, c[2]);
                let hi1 = ops.mul_high(m1, c[2]);
                [
                    ops.xor(ops.xor(hi1, c[1]), k0),
                    lo1,
                    ops.xor(ops.xor(hi0, c[3]), k1),
                    lo0,
                ]
            },
        );
        k0 = ops.add(k0, w0);
        k1 = ops.add(k1, w1);
    }
    ctr
}

/// Scale factor which maps the top 24 bits of a `u32` to `[0, 1)`.
const UNIT_SCALE: f32 = 1.0 / (1 << 24) as f32;

/// Convert the top 24 bits of each lane to a float in `[0, 2^24)`.
///
/// The conversion is exact.
#[inline(always)]
fn top_bits_to_float<I: Isa>(isa: I, x: I::U32) -> I::F32 {
    let x: I::I32 = isa.u32().shift_right::<8>(x).reinterpret_cast();
    isa.i32().to_float(x)
}

/// Transforms random bits into samples from a distribution.
trait Distribution {
    /// Convert four vectors of random bits into four vectors of samples.
    fn sample<I: Isa>(&self, isa: I, bits: [I::U32; 4]) -> [I::F32; 4];
}

struct Uniform {
    low: f32,
    high: f32,
}

impl Distribution for Uniform {
    #[inline(always)]
    fn sample<I: Isa>(&self, isa: I, bits: [I::U32; 4]) -> [I::F32; 4] {
        let ops = isa.f32();
        let low = ops.splat(self.low);

        // Fold the scale from `[0, 2^24)` to `[0, 1)` into the range. Scaling
        // by a power of two is exact, so this produces the same result as
        // scaling separately.
        let scaled_range = ops.splat((self.high - self.low) * UNIT_SCALE);
        array_map(
            bits,
            #[inline(always)]
            |x| ops.mul_add(top_bits_to_float(isa, x), scaled_range, low),
        )
    }
}

struct Normal {
    mean: f32,
    std_dev: f32,
}

impl Normal {
    /// Convert a pair of random bit vectors to a pair of samples from the
    /// standard normal distribution using the Box-Muller transform.
    #[inline(always)]
    fn box_muller<I: Isa>(isa: I, bits1: I::U32, bits2: I::U32) -> [I::F32; 2] {
        let ops = isa.f32();
        let scale = ops.splat(UNIT_SCALE);
        let u1 = ops.mul(top_bits_to_float(isa, bits1), scale);
        let u2 = ops.mul(top_bits_to_float(isa, bits2), scale);

        // Map `u1` to `(0, 1]` so the log is finite.
        let u1 = ops.sub(ops.one(), u1);
        let r = ops.sqrt(ops.mul(ops.splat(-2.), Ln::default().eval(isa, u1)));
        let theta = ops.mul(u2, ops.splat(TAU));

        [
            ops.mul(r, Cos::new().eval(isa, theta)),
            ops.mul(r, Sin::new().eval(isa, theta)),
        ]
    }
}

impl Distribution for Normal {
    #[inline(always)]
    fn sample<I: Isa>(&self, isa: I, bits: [I::U32; 4]) -> [I::F32; 4] {
        let ops = isa.f32();
        let [z0, z1] = Self::box_muller(isa, bits[0], bits[1]);
        let [z2, z3] = Self::box_muller(isa, bits[2], bits[3]);

        let mean = ops.splat(self.mean);
        let std_dev = ops.splat(self.std_dev);
        array_map(
            [z0, z1, z2, z3],
            #[inline(always)]
            |z| ops.mul_add(z, std_dev, mean),
        )
    }
}

/// Fill `dest` with samples from a distribution.
///
/// `offset` is the index of the first element of `dest` in the random
/// sequence.
#[inline(always)]
fn fill<'d, I: Isa, D: Distribution>(
    isa: I,
    dest: &'d mut [MaybeUninit<f32>],
    seed: u64,
    offset: u64,
    dist: &D,
) -> &'d mut [f32] {
    // Dispatch on the number of SIMD vectors of counters needed per block.
    match BLOCK_COUNTERS / isa.u32().len() {
        1 => fill_impl::<I, D, 1>(isa, dest, seed, offset, dist),
        2 => fill_impl::<I, D, 2>(isa, dest, seed, offset, dist),
        4 => fill_impl::<I, D, 4>(isa, dest, seed, offset, dist),
        _ => unreachable!("SIMD vector width should divide `BLOCK_COUNTERS`"),
    }
}

/// Implementation of [`fill`] for an ISA which needs `G` vectors of counters
/// per block.
#[inline(always)]
fn fill_impl<'d, I: Isa, D: Distribution, const G: usize>(
    isa: I,
    dest: &'d mut [MaybeUninit<f32>],
    seed: u64,
    offset: u64,
    dist: &D,
) -> &'d mut [f32] {
    let u32_ops = isa.u32();
    let f32_ops = isa.f32();
    let n_lanes = u32_ops.len();

    let key = [seed as u32, (seed >> 32) as u32];
    let lane_index: [u32; BLOCK_COUNTERS] = std::array::from_fn(|i| i as u32);
    let lane_index = u32_ops.load(&lane_index);
    let zero = u32_ops.zero();
    let groups: [usize; G] = std::array::from_fn(|i| i);

    let block_size = BLOCK_SIZE as u64;
    let end = offset
        .checked_add(dest.len() as u64)
        .expect("random sequence index should not overflow");
    let mut writer = SliceWriter::new(dest);

    for block in offset / block_size..end.div_ceil(block_size) {
        let ctr = array_map(
            groups,
            #[inline(always)]
            |group| {
                let ctr = block * BLOCK_COUNTERS as u64 + (group * n_lanes) as u64;

                // `ctr` is a multiple of `n_lanes`, so adding the lane index to
                // the low word cannot overflow.
                let ctr_lo = u32_ops.add(u32_ops.splat(ctr as u32), lane_index);
                let ctr_hi = u32_ops.splat((ctr >> 32) as u32);
                [ctr_lo, ctr_hi, zero, zero]
            },
        );

        // Generate 4 vectors of samples for each group of counters, one for
        // each word of the generator output.
        let samples = array_map(
            philox4x32(isa, ctr, key),
            #[inline(always)]
            |bits| dist.sample(isa, bits),
        );

        let block_start = block * block_size;
        let block_end = block_start + block_size;
        if block_start >= offset && block_end <= end {
            for word in 0..4 {
                for group_samples in &samples {
                    writer.write_vec(f32_ops, group_samples[word]);
                }
            }
        } else {
            // Block is partially outside the range of `dest`.
            let mut tmp = [0.; BLOCK_SIZE];
            let mut tmp_chunks = tmp.chunks_exact_mut(n_lanes);
            for word in 0..4 {
                for (group_samples, chunk) in samples.iter().zip(tmp_chunks.by_ref()) {
                    f32_ops.store(group_samples[word], chunk);
                }
            }
            let lo = offset.saturating_sub(block_start) as usize;
            let hi = (end - block_start).min(block_size) as usize;
            for &x in &tmp[lo..hi] {
                writer.write_scalar(x);
            }
        }
    }

    writer.into_mut_slice()
}

/// Fill a slice with random values uniformly distributed between `low` and
/// `high`.
///
/// Values are generated as `low + u * (high - low)` where `u` is in `[0, 1)`.
/// Due to rounding the result can be equal to `high`.
///
/// The output for a given seed may differ slightly across platforms, due to
/// differences in rounding.
pub struct RandomUniform<'d> {
    dest: &'d mut [MaybeUninit<f32>],
    seed: u64,
    offset: u64,
    dist: Uniform,
}

impl<'d> RandomUniform<'d> {
    pub fn new(dest: &'d mut [MaybeUninit<f32>], seed: u64, low: f32, high: f32) -> Self {
        RandomUniform {
            dest,
            seed,
            offset: 0,
            dist: Uniform { low, high },
        }
    }

    /// Set the index of the first element of `dest` in the random sequence.
    ///
    /// Filling consecutive chunks of a slice, with offsets set to the start of
    /// each chunk, produces the same output as filling the whole slice at once.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }
}

impl<'d> SimdOp for RandomUniform<'d> {
    type Output = &'d mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        fill(isa, self.dest, self.seed, self.offset, &self.dist)
    }
}

/// Fill a slice with random values from a normal distribution.
///
/// Values are generated using the Box-Muller transform. Samples are limited to
/// within about 5.8 standard deviations of the mean.
///
/// The output for a given seed may differ slightly across platforms, due to
/// differences in rounding.
pub struct RandomNormal<'d> {
    dest: &'d mut [MaybeUninit<f32>],
    seed: u64,
    offset: u64,
    dist: Normal,
}

impl<'d> RandomNormal<'d> {
    pub fn new(dest: &'d mut [MaybeUninit<f32>], seed: u64, mean: f32, std_dev: f32) -> Self {
        RandomNormal {
            dest,
            seed,
            offset: 0,
            dist: Normal { mean, std_dev },
        }
    }

    /// Set the index of the first element of `dest` in the random sequence.
    ///
    /// See [`RandomUniform::with_offset`].
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }
}

impl<'d> SimdOp for RandomNormal<'d> {
    type Output = &'d mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        fill(isa, self.dest, self.seed, self.offset, &self.dist)
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::isa::GenericIsa;
    use rten_simd::ops::BitOps;
    use rten_simd::{Isa, Simd, SimdOp};

    use super::{BLOCK_COUNTERS, BLOCK_SIZE, RandomNormal, RandomUniform, philox4x32};
    use crate::testing::{IntoUninit, check_f32s_are_equal_atol};

    /// Scalar reference implementation of Philox4x32-10.
    fn reference_philox4x32(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
        let mulhilo = |a: u32, b: u32| {
            let prod = a as u64 * b as u64;
            ((prod >> 32) as u32, prod as u32)
        };
        let mut c = ctr;
        let mut k = key;
        for round in 0..10 {
            if round > 0 {
                k[0] = k[0].wrapping_add(0x9E3779B9);
                k[1] = k[1].wrapping_add(0xBB67AE85);
            }
            let (hi0, lo0) = mulhilo(0xD2511F53, c[0]);
            let (hi1, lo1) = mulhilo(0xCD9E8D57, c[2]);
            c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
        }
        c
    }

    // Known-answer test vectors from the Random123 library.
    // See <https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors>.
    const PHILOX_KAT: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
        (
            [0, 0, 0, 0],
            [0, 0],
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
        ),
        (
            [u32::MAX; 4],
            [u32::MAX; 2],
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
        ),
        (
            [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
            [0xa4093822, 0x299f31d0],
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
        ),
    ];

    #[test]
    fn test_reference_philox() {
        for (ctr, key, expected) in PHILOX_KAT {
            assert_eq!(reference_philox4x32(ctr, key), expected);
        }
    }

    #[test]
    fn test_philox() {
        struct PhiloxTest {}

        impl SimdOp for PhiloxTest {
            type Output = ();

            fn eval<I: Isa>(self, isa: I) {
                let ops = isa.u32();
                for (ctr, key, expected) in PHILOX_KAT {
                    let ctr = ctr.map(|c| ops.splat(c));
                    let [actual] = philox4x32(isa, [ctr], key);
                    for (actual, expected) in actual.into_iter().zip(expected) {
                        let actual = actual.to_array();
                        assert!(actual.as_ref().iter().all(|x| *x == expected));
                    }
                }
            }
        }

        PhiloxTest {}.dispatch();
        PhiloxTest {}.eval(GenericIsa::new());
    }

    fn uniform(len: usize, seed: u64, offset: u64) -> Vec<f32> {
        let mut buf = vec![0.; len];
        RandomUniform::new(buf.as_mut_slice().into_uninit(), seed, 0., 1.)
            .with_offset(offset)
            .dispatch();
        buf
    }

    fn normal(len: usize, seed: u64, offset: u64) -> Vec<f32> {
        let mut buf = vec![0.; len];
        RandomNormal::new(buf.as_mut_slice().into_uninit(), seed, 0., 1.)
            .with_offset(offset)
            .dispatch();
        buf
    }

    #[test]
    fn test_random_uniform_matches_philox() {
        let seed = 0x1234_5678_9abc_def0;
        let key = [seed as u32, (seed >> 32) as u32];
        let output = uniform(BLOCK_SIZE * 2, seed, 0);

        for (i, x) in output.into_iter().enumerate() {
            let block = i / BLOCK_SIZE;
            let word = (i % BLOCK_SIZE) / BLOCK_COUNTERS;
            let ctr = (block * BLOCK_COUNTERS + i % BLOCK_COUNTERS) as u32;
            let bits = reference_philox4x32([ctr, 0, 0, 0], key)[word];
            let expected = (bits >> 8) as f32 / (1 << 24) as f32;
            assert_eq!(x, expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_random_uniform_isa_independent() {
        let len = BLOCK_SIZE * 3 + 5;
        let seed = 42;

        let mut buf = vec![0.; len];
        RandomUniform::new(buf.as_mut_slice().into_uninit(), seed, -2., 3.).eval(GenericIsa::new());

        let mut native = vec![0.; len];
        RandomUniform::new(native.as_mut_slice().into_uninit(), seed, -2., 3.).dispatch();

        // Results differ by up to 1 ULP if only one ISA uses fused
        // multiply-add.
        check_f32s_are_equal_atol(
            buf.into_iter()
                .zip(native)
                .enumerate()
                .map(|(i, (x, y))| (i as f32, x, y)),
            2.4e-7,
        );
    }

    #[test]
    fn test_random_normal_isa_independent() {
        let len = BLOCK_SIZE * 3 + 5;
        let seed = 42;

        let mut buf = vec![0.; len];
        RandomNormal::new(buf.as_mut_slice().into_uninit(), seed, 0., 1.).eval(GenericIsa::new());
        let native = normal(len, seed, 0);

        check_f32s_are_equal_atol(
            buf.into_iter()
                .zip(native)
                .enumerate()
                .map(|(i, (x, y))| (i as f32, x, y)),
            9.6e-7,
        );
    }

    #[test]
    fn test_offset() {
        #[derive(Debug)]
        struct Case {
            len: usize,
            chunk_size: usize,
        }

        let cases = [
            // Chunks aligned to blocks.
            Case {
                len: BLOCK_SIZE * 4,
                chunk_size: BLOCK_SIZE,
            },
            // Chunks not aligned to blocks.
            Case {
                len: BLOCK_SIZE * 4 + 3,
                chunk_size: 7,
            },
            // Chunks spanning multiple blocks.
            Case {
                len: BLOCK_SIZE * 4 + 3,
                chunk_size: BLOCK_SIZE + 17,
            },
        ];

        for Case { len, chunk_size } in cases {
            let seed = 1234;
            let expected_uniform = uniform(len, seed, 0);
            let expected_normal = normal(len, seed, 0);

            let mut actual_uniform = Vec::new();
            let mut actual_normal = Vec::new();
            for start in (0..len).step_by(chunk_size) {
                let chunk_len = chunk_size.min(len - start);
                actual_uniform.extend(uniform(chunk_len, seed, start as u64));
                actual_normal.extend(normal(chunk_len, seed, start as u64));
            }

            assert_eq!(actual_uniform, expected_uniform);
            assert_eq!(actual_normal, expected_normal);
        }
    }

    #[test]
    fn test_seed() {
        assert_eq!(uniform(100, 1, 0), uniform(100, 1, 0));
        assert_ne!(uniform(100, 1, 0), uniform(100, 2, 0));
        assert_ne!(normal(100, 1, 0), normal(100, 2, 0));
    }

    #[test]
    fn test_random_uniform_distribution() {
        let low = -3.;
        let high = 5.;
        let len = 100_000;
        let mut buf = vec![0.; len];
        RandomUniform::new(buf.as_mut_slice().into_uninit(), 7, low, high).dispatch();

        let mut buckets = [0usize; 10];
        let bucket_size = (high - low) / buckets.len() as f32;
        for &x in &buf {
            assert!(x >= low && x <= high, "{x} out of range");
            let idx = (((x - low) / bucket_size) as usize).min(buckets.len() - 1);
            buckets[idx] += 1;
        }

        let expected = len / buckets.len();
        for count in buckets {
            let delta = count.abs_diff(expected) as f32 / expected as f32;
            assert!(delta < 0.05, "bucket count {count} far from {expected}");
        }
    }

    #[test]
    fn test_random_normal_distribution() {
        let mean = 2.;
        let std_dev = 0.5;
        let len = 100_000;
        let mut buf = vec![0.; len];
        RandomNormal::new(buf.as_mut_slice().into_uninit(), 7, mean, std_dev).dispatch();

        assert!(buf.iter().all(|x| x.is_finite()));

        let n = len as f64;
        let actual_mean = buf.iter().map(|&x| x as f64).sum::<f64>() / n;
        let actual_var = buf
            .iter()
            .map(|&x| (x as f64 - actual_mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(
            (actual_mean - mean as f64).abs() < 0.01,
            "mean {actual_mean}"
        );
        assert!(
            (actual_var.sqrt() - std_dev as f64).abs() < 0.01,
            "std dev {}",
            actual_var.sqrt()
        );

        // Check the fraction of samples within 1, 2 and 3 standard deviations.
        for (k, expected) in [(1., 0.6827), (2., 0.9545), (3., 0.9973)] {
            let within = buf
                .iter()
                .filter(|&&x| ((x - mean) / std_dev).abs() < k)
                .count() as f64
                / n;
            assert!(
                (within - expected).abs() < 0.01,
                "fraction within {k} std devs is {within}, expected {expected}"
            );
        }
    }

    #[ignore]
    #[test]
    fn bench_random() {
        use std::time::Instant;

        let len = 1 << 22;
        let iters = 50;
        let mut buf = vec![0.; len];

        let mut time = |name: &str, f: &mut dyn FnMut(&mut [f32])| {
            f(&mut buf); // Warm up
            let start = Instant::now();
            for _ in 0..iters {
                f(&mut buf);
            }
            let elapsed = start.elapsed().as_secs_f64() * 1e9 / (iters * len) as f64;
            println!("{name}: {elapsed:.3} ns/elem");
        };

        let mut rng = fastrand::Rng::with_seed(1);
        time("scalar fastrand uniform", &mut |buf| {
            buf.fill_with(|| rng.f32());
        });
        time("vectorized uniform", &mut |buf| {
            RandomUniform::new(buf.into_uninit(), 1, 0., 1.).dispatch();
        });
        time("vectorized normal", &mut |buf| {
            RandomNormal::new(buf.into_uninit(), 1, 0., 1.).dispatch();
        });
    }
}
