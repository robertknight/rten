use std::ops::Range;

use smallvec::{smallvec, SmallVec};

pub trait IndexArray: AsMut<[usize]> + AsRef<[usize]> + Clone {}
impl<const N: usize> IndexArray for SmallVec<[usize; N]> {}
impl<const N: usize> IndexArray for [usize; N] {}

/// Iterator over a range of N-dimensional indices, where N may be known at
/// compile time (see [NdIndices]) or only at runtime ([DynIndices]).
///
/// The number of dimensions may be zero, in which case the iterator will yield
/// a single empty index. This is consistent with eg. `ndindex` in NumPy.
pub struct Indices<Index: IndexArray>
where
    Index: IndexArray,
{
    /// Start index along each dimension.
    start: Index,

    /// End index (exclusive) along each dimension.
    end: Index,

    next: Option<Index>,
}

/// Iterator over a range of N-dimensional indices, where N is not known at
/// compile time.
pub type DynIndices = Indices<SmallVec<[usize; 5]>>;

/// Iterator over a range of N-dimensional indices, where N is known at compile
/// time.
pub type NdIndices<const N: usize> = Indices<[usize; N]>;

impl<Index: IndexArray> Indices<Index> {
    fn from_start_and_end(start: Index, end: Index) -> Indices<Index> {
        Indices {
            // Note that if the index is empty, `start == end` but the iterator
            // should yield a single empty element in that case.
            next: if start.as_ref() != end.as_ref() || start.as_ref().is_empty() {
                Some(start.clone())
            } else {
                None
            },
            start,
            end,
        }
    }
}

impl<const N: usize> Indices<SmallVec<[usize; N]>> {
    /// Return an iterator over all the indices where each dimension lies
    /// within the corresponding range in `ranges`.
    pub fn from_ranges(ranges: &[Range<usize>]) -> Indices<SmallVec<[usize; N]>> {
        let start: SmallVec<[usize; N]> = ranges.iter().map(|r| r.start).collect();
        let end = ranges.iter().map(|r| r.end).collect();
        Self::from_start_and_end(start, end)
    }

    /// Return an iterator over all the indices where each dimension is between
    /// `0` and `shape[dim]`.
    pub fn from_shape(shape: &[usize]) -> Indices<SmallVec<[usize; N]>> {
        let start = smallvec![0; shape.len()];
        let end = shape.iter().copied().collect();
        Self::from_start_and_end(start, end)
    }
}

impl<const N: usize> Indices<[usize; N]> {
    /// Return an iterator over all the indices where each dimension lies
    /// within the corresponding range in `ranges`.
    pub fn from_ranges(ranges: [Range<usize>; N]) -> Indices<[usize; N]> {
        let start = ranges.clone().map(|r| r.start);
        let end = ranges.map(|r| r.end);
        Self::from_start_and_end(start, end)
    }

    /// Return an iterator over all the indices where each dimension is between
    /// `0` and `shape[dim]`.
    pub fn from_shape(shape: [usize; N]) -> Indices<[usize; N]> {
        Self::from_ranges(shape.map(|size| 0..size))
    }
}

impl<Index: IndexArray> Iterator for Indices<Index> {
    type Item = Index;

    /// Return the next index in the sequence, or `None` after all indices
    /// have been returned.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = self.next.clone() {
            if current.as_ref().is_empty() {
                self.next = None;
                return Some(current);
            }

            // Find dimension where the last element has not been reached.
            let mut next = current.clone();
            let mut dim = next.as_ref().len() - 1;
            while dim > 0 && next.as_ref()[dim] >= self.end.as_ref()[dim] - 1 {
                next.as_mut()[dim] = self.start.as_ref()[dim];
                dim -= 1;
            }
            next.as_mut()[dim] += 1;

            if next.as_ref()[dim] < self.end.as_ref()[dim] {
                self.next = Some(next);
            } else {
                self.next = None;
            }

            Some(current)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DynIndices, NdIndices};

    #[test]
    fn test_nd_indices() {
        // Empty iterator
        let mut iter = NdIndices::from_ranges([0..0]);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        // Scalar index iterator
        let mut iter = NdIndices::from_ranges([]);
        assert_eq!(iter.next(), Some([]));
        assert_eq!(iter.next(), None);

        // 1D index iterator
        let iter = NdIndices::from_ranges([0..5]);
        let visited: Vec<_> = iter.collect();
        assert_eq!(visited, &[[0], [1], [2], [3], [4]]);

        // 2D index iterator
        let iter = NdIndices::from_ranges([2..4, 2..4]);
        let visited: Vec<_> = iter.collect();
        assert_eq!(visited, &[[2, 2], [2, 3], [3, 2], [3, 3]]);
    }

    #[test]
    fn test_dyn_indices() {
        type Index = <DynIndices as Iterator>::Item;

        // Empty iterator
        let mut iter = DynIndices::from_ranges(&[0..0]);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        // Scalar index iterator
        let mut iter = DynIndices::from_ranges(&[]);
        assert_eq!(iter.next(), Some(Index::new()));
        assert_eq!(iter.next(), None);

        // 1D index iterator
        let iter = DynIndices::from_ranges(&[0..5]);
        let visited: Vec<Vec<usize>> = iter.map(|ix| ix.into_iter().collect()).collect();
        assert_eq!(visited, vec![vec![0], vec![1], vec![2], vec![3], vec![4]]);

        // 2D index iterator
        let iter = DynIndices::from_ranges(&[2..4, 2..4]);
        let visited: Vec<Vec<usize>> = iter.map(|ix| ix.into_iter().collect()).collect();
        assert_eq!(
            visited,
            vec![vec![2, 2], vec![2, 3], vec![3, 2], vec![3, 3],]
        );
    }
}
