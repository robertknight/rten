use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A hash map which uses keys directly as hash values.
///
/// This is intended for use with u32 keys that come from a sequence, so the
/// keys can be used directly as hash values without encountering too many
/// collisions.
pub type NoopHashMap<K, V> = HashMap<K, V, BuildHasherDefault<NoopHasher>>;

#[derive(Default)]
pub struct NoopHasher {
    hash: u64,
}

impl Hasher for NoopHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    /// Hash bytes such that hashing `unsigned_int.to_ne_bytes()` sets the
    /// hash value to `unsigned_int as u64` (on a little-endian arch as least).
    fn write(&mut self, bytes: &[u8]) {
        let mut new_hash = 0;
        for (i, b) in bytes.iter().enumerate() {
            new_hash |= (*b as u64) << (i * 8);
        }
        self.hash = new_hash;
    }

    // Implement u32 hashing directly, since `NodeId`s are u32 values.
    fn write_u32(&mut self, i: u32) {
        self.hash = i as u64;
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use super::NoopHasher;

    #[test]
    fn test_noop_hasher() {
        let mut hasher = NoopHasher::default();
        hasher.write_u32(1234);
        assert_eq!(hasher.finish(), 1234);

        let mut hasher = NoopHasher::default();
        hasher.write(&(4567u64).to_ne_bytes());
        assert_eq!(hasher.finish(), 4567);
    }
}
