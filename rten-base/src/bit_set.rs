/// Wraps an integer type `B` to treat it as a set of bit flags.
#[derive(Copy, Clone, Default, PartialEq)]
pub struct BitSet<B: BitOps = u32>(B);

impl<B: BitOps> BitSet<B> {
    /// Return a bit set with all positions cleared.
    pub fn new() -> Self {
        Self(B::ZERO)
    }

    /// Return a bit set with the first `n` positions set.
    pub fn ones(n: u32) -> Self {
        let bits = if n >= B::BITS {
            B::MAX
        } else {
            B::nth(n) - B::ONE
        };
        Self(bits)
    }

    /// Return a bit set with given indices set.
    pub fn from_indices<I: IntoIterator<Item = u32>>(indices: I) -> Self {
        let mut bits = Self(B::ZERO);
        for pos in indices {
            bits.set(pos);
        }
        bits
    }

    /// Unset the bit at position `pos`.
    pub fn delete(&mut self, pos: u32) {
        self.0 &= !B::nth(pos)
    }

    /// Set the bit at position `pos`.
    pub fn set(&mut self, pos: u32) {
        self.0 |= B::nth(pos);
    }

    /// Return true if position `pos` is set.
    pub fn get(&self, pos: u32) -> bool {
        self.0 & B::nth(pos) != B::ZERO
    }

    /// Return the number of bits set.
    pub fn count_true(&self) -> u32 {
        self.0.count_ones()
    }

    /// Return true if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.0 == B::ZERO
    }

    /// Return an iterator over the indices of set positions.
    pub fn iter(&self) -> impl Iterator<Item = u32> {
        (0..B::BITS).filter(|pos| self.get(*pos))
    }
}

impl<B: BitOps> std::fmt::Debug for BitSet<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:b}", self.0)
    }
}

/// Operations needed for types that can be used as a [`BitSet`].
pub trait BitOps:
    Copy
    + Default
    + Eq
    + std::fmt::Binary
    + std::ops::BitAnd<Self, Output = Self>
    + std::ops::BitAndAssign
    + std::ops::BitOr<Self, Output = Self>
    + std::ops::BitOrAssign
    + std::ops::Not<Output = Self>
    + std::ops::Sub<Self, Output = Self>
{
    /// Number of bits in this type.
    const BITS: u32;

    /// Maximum value of this type.
    const MAX: Self;

    const ZERO: Self;
    const ONE: Self;

    /// Return the number of bits set to one.
    fn count_ones(self) -> u32;

    /// Return `Self` with bit `idx` set to one.
    fn nth(idx: u32) -> Self;
}

macro_rules! impl_bitops {
    ($ty:ty) => {
        impl BitOps for $ty {
            const BITS: u32 = <$ty>::BITS;
            const MAX: $ty = <$ty>::MAX;
            const ZERO: $ty = 0;
            const ONE: $ty = 1;

            fn count_ones(self) -> u32 {
                <$ty>::count_ones(self)
            }

            fn nth(idx: u32) -> Self {
                (1 << idx) as Self
            }
        }
    };
}

impl_bitops!(u8);
impl_bitops!(u16);
impl_bitops!(u32);
impl_bitops!(u64);
impl_bitops!(u128);
impl_bitops!(usize);

#[cfg(test)]
mod tests {
    use super::BitSet;

    #[test]
    fn test_bit_set() {
        let mut set = BitSet::<u32>::ones(5);
        assert_eq!(set.count_true(), 5);
        assert!(!set.is_empty());
        for i in 0..5 {
            assert!(set.get(i));
            set.delete(i);
            assert!(!set.get(i));
        }
        assert_eq!(set.count_true(), 0);
        assert!(set.is_empty());

        let all_zeros = BitSet::<u32>::default();
        assert_eq!(all_zeros.count_true(), 0);
        assert_eq!(BitSet::<u32>::new(), all_zeros);
    }

    #[test]
    fn test_bit_set_ones() {
        for i in 0..=32 {
            let all_ones = BitSet::<u32>::ones(i);
            assert_eq!(all_ones.count_true(), i);
        }
    }

    #[test]
    fn test_bit_set_iter() {
        let mut set = BitSet::<u32>::ones(6);
        set.delete(0);
        set.delete(5);

        let positions: Vec<_> = set.iter().collect();
        assert_eq!(positions, [1, 2, 3, 4]);
    }

    #[test]
    fn test_bit_set_from_indices() {
        let set = BitSet::<u32>::from_indices([0, 3]);
        for i in 0..u32::BITS {
            assert_eq!(set.get(i), i == 0 || i == 3);
        }
    }
}
