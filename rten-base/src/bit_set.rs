/// Wraps an integer type `B` to treat it as a set of bit flags.
#[derive(Copy, Clone, Default, PartialEq)]
pub struct BitSet<B: BitOps = u32>(B);

impl<B: BitOps> BitSet<B> {
    const BITS: u32 = B::BITS;

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
                (1 as $ty) << idx
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

type Block = BitSet<u64>;

#[derive(Clone, Debug, PartialEq)]
enum BitVecData {
    Inline { bits: Block, len: usize },
    Heap { blocks: Box<[Block]>, len: usize },
}

/// A set of bit flags with a length chosen at runtime.
///
/// Vectors of 64 bits or less are stored inline without allocating.
#[derive(Clone, Debug, PartialEq)]
pub struct BitVec(BitVecData);

impl BitVec {
    /// Return a bit vector of length `len` with all positions cleared.
    pub fn new(len: usize) -> Self {
        let blocks = len.div_ceil(Block::BITS as usize);
        if blocks <= 1 {
            Self(BitVecData::Inline {
                bits: BitSet::new(),
                len,
            })
        } else {
            Self(BitVecData::Heap {
                blocks: std::iter::repeat_n(BitSet::new(), blocks).collect(),
                len,
            })
        }
    }

    /// Return a bit vector of length `len` with all positions set.
    pub fn ones(len: usize) -> Self {
        let mut vec = Self::new(len);
        match &mut vec.0 {
            BitVecData::Inline { bits, len } => *bits = Block::ones(*len as u32),
            BitVecData::Heap { blocks, len } => {
                let mut remaining = *len;
                for block in blocks {
                    *block = Block::ones(remaining.min(Block::BITS as usize) as u32);
                    remaining = remaining.saturating_sub(Block::BITS as usize);
                }
            }
        }
        vec
    }

    /// Return a bit vector of length `len` with given indices set.
    pub fn from_indices<I: IntoIterator<Item = u32>>(len: usize, indices: I) -> Self {
        let mut bits = Self::new(len);
        for pos in indices {
            bits.set(pos);
        }
        bits
    }

    /// Return the number of bits in this vector.
    pub fn len(&self) -> usize {
        match &self.0 {
            BitVecData::Inline { bits: _, len } => *len,
            BitVecData::Heap { blocks: _, len } => *len,
        }
    }

    /// Return the block containing bit `pos` and the offset within it.
    ///
    /// `pos` must be less than `self.len()`.
    fn block(&self, pos: u32) -> (&Block, u32) {
        match &self.0 {
            BitVecData::Inline { bits, len: _ } => (bits, pos),
            BitVecData::Heap { blocks, len: _ } => {
                let (block_idx, block_off) = (pos / Block::BITS, pos % Block::BITS);
                (&blocks[block_idx as usize], block_off)
            }
        }
    }

    /// Return the block containing bit `pos` and the offset within it.
    ///
    /// `pos` must be less than `self.len()`.
    fn block_mut(&mut self, pos: u32) -> (&mut Block, u32) {
        match &mut self.0 {
            BitVecData::Inline { bits, len: _ } => (bits, pos),
            BitVecData::Heap { blocks, len: _ } => {
                let (block_idx, block_off) = (pos / Block::BITS, pos % Block::BITS);
                (&mut blocks[block_idx as usize], block_off)
            }
        }
    }

    fn blocks(&self) -> &[Block] {
        match &self.0 {
            BitVecData::Inline { bits, len: _ } => std::slice::from_ref(bits),
            BitVecData::Heap { blocks, len: _ } => blocks,
        }
    }

    /// Set the bit at position `pos`.
    ///
    /// Panics if `pos` is out of bounds.
    pub fn set(&mut self, pos: u32) {
        assert!((pos as usize) < self.len(), "position out of bounds");
        let (blk, blk_off) = self.block_mut(pos);
        blk.set(blk_off);
    }

    /// Unset the bit at position `pos`.
    ///
    /// Panics if `pos` is out of bounds.
    pub fn delete(&mut self, pos: u32) {
        assert!((pos as usize) < self.len(), "position out of bounds");
        let (blk, blk_off) = self.block_mut(pos);
        blk.delete(blk_off);
    }

    /// Return true if position `pos` is set.
    ///
    /// Returns false if `pos` is out of bounds.
    pub fn get(&self, pos: u32) -> bool {
        if pos as usize >= self.len() {
            return false;
        }
        let (blk, blk_off) = self.block(pos);
        blk.get(blk_off)
    }

    /// Return the number of bits set.
    pub fn count_true(&self) -> u32 {
        self.blocks().iter().map(|blk| blk.count_true()).sum()
    }

    /// Return true if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.blocks().iter().all(|blk| blk.is_empty())
    }

    /// Return an iterator over the indices of set positions.
    pub fn iter(&self) -> impl Iterator<Item = usize> {
        (0..self.len()).filter(|pos| self.get(*pos as u32))
    }
}

impl Default for BitVec {
    fn default() -> Self {
        BitVec::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::{BitSet, BitVec};

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

    // Lengths which use the inline and heap-allocated representations
    // respectively.
    const BIT_VEC_LENS: [usize; 2] = [5, 100];

    #[test]
    fn test_bit_vec_new() {
        for len in BIT_VEC_LENS {
            let vec = BitVec::new(len);
            assert_eq!(vec.len(), len);
            assert_eq!(vec.count_true(), 0);
            assert!(vec.is_empty());
            assert_eq!(vec, BitVec::new(len));
            assert_ne!(vec, BitVec::ones(len));
        }
        assert_eq!(BitVec::default(), BitVec::new(0));
    }

    #[test]
    fn test_bit_vec_ones() {
        for len in BIT_VEC_LENS {
            let vec = BitVec::ones(len);
            assert_eq!(vec.len(), len);
            assert_eq!(vec.count_true(), len as u32);
            for i in 0..len + 10 {
                assert_eq!(vec.get(i as u32), i < len);
            }
        }
    }

    #[test]
    fn test_bit_vec_set_get_delete() {
        for len in BIT_VEC_LENS {
            let mut vec = BitVec::new(len);
            for i in 0..len as u32 {
                assert!(!vec.get(i));
                vec.set(i);
                assert!(vec.get(i));
            }
            assert_eq!(vec.count_true(), len as u32);
            assert!(!vec.is_empty());

            for i in 0..len as u32 {
                vec.delete(i);
                assert!(!vec.get(i));
            }
            assert!(vec.is_empty());
        }
    }

    #[test]
    #[should_panic(expected = "position out of bounds")]
    fn test_bit_vec_set_out_of_bounds() {
        let mut vec = BitVec::new(5);
        vec.set(5);
    }

    #[test]
    fn test_bit_vec_iter() {
        for len in BIT_VEC_LENS {
            let last = len as u32 - 1;
            let vec = BitVec::from_indices(len, [0, 3, last]);
            let positions: Vec<_> = vec.iter().collect();
            assert_eq!(positions, [0, 3, last as usize]);
        }
    }
}
