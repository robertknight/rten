#[derive(Copy, Clone, Default, PartialEq)]
pub struct BitSet(u32);

impl BitSet {
    pub const BITS: usize = u32::BITS as usize;

    /// Return a bit set with the first `n` positions set.
    pub fn ones(n: u32) -> Self {
        let bits = if n >= 32 { u32::MAX } else { (1 << n) - 1 };
        Self(bits)
    }

    /// Unset the bit at position `pos`.
    pub fn delete(&mut self, pos: u32) {
        self.0 &= !(1 << pos)
    }

    /// Return true if position `pos` is set.
    pub fn get(&self, pos: u32) -> bool {
        self.0 & (1 << pos) != 0
    }

    /// Return the number of bits set.
    pub fn len(&self) -> u32 {
        self.0.count_ones()
    }

    /// Return true if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Return an iterator over the indices of set positions.
    pub fn iter(&self) -> impl Iterator<Item = usize> {
        (0..u32::BITS as usize).filter(|pos| self.get(*pos as u32))
    }
}

impl std::fmt::Debug for BitSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:b}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::BitSet;

    #[test]
    fn test_bit_set() {
        let mut set = BitSet::ones(5);
        assert_eq!(set.len(), 5);
        assert!(!set.is_empty());
        for i in 0..5 {
            assert!(set.get(i));
            set.delete(i);
            assert!(!set.get(i));
        }
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());

        let all_zeros = BitSet::default();
        assert_eq!(all_zeros.len(), 0);
    }

    #[test]
    fn test_bit_set_ones() {
        for i in 0..=32 {
            let all_ones = BitSet::ones(i);
            assert_eq!(all_ones.len(), i);
        }
    }

    #[test]
    fn test_bit_set_iter() {
        let mut set = BitSet::ones(6);
        set.delete(0);
        set.delete(5);

        let positions: Vec<usize> = set.iter().collect();
        assert_eq!(positions, [1, 2, 3, 4]);
    }
}
