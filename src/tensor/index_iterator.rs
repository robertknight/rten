use std::ops::Range;

/// An iterator over indices within a given range.
///
/// This struct is for indices with a statically known dimension count. See
/// [IndexIterator] for dynamic dimension counts.
pub struct NdIndexIterator<const N: usize> {
    first: bool,
    current: [usize; N],
    ranges: [Range<usize>; N],
}

impl<const N: usize> NdIndexIterator<N> {
    /// Return an iterator over all the indices where each dimension lies
    /// within the corresponding range in `ranges`.
    ///
    /// If `ranges` is empty, the iterator yields a single empty index. This
    /// is consistent with `ndindex` in eg. numpy.
    pub fn from_ranges(ranges: [Range<usize>; N]) -> NdIndexIterator<N> {
        NdIndexIterator {
            first: true,
            current: ranges.clone().map(|r| r.start),
            ranges,
        }
    }

    /// Return an iterator over all the indices where each dimension is between
    /// `0` and `shape[dim]`.
    pub fn from_shape(shape: [usize; N]) -> NdIndexIterator<N> {
        Self::from_ranges(shape.map(|size| 0..size))
    }
}

impl<const N: usize> Iterator for NdIndexIterator<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_empty() {
            if self.first {
                self.first = false;
                return Some(self.current);
            } else {
                return None;
            }
        }

        // Find dimension where the last element has not been reached.
        let mut dim = self.current.len() - 1;
        while dim > 0 && self.current[dim] >= self.ranges[dim].end - 1 {
            self.current[dim] = self.ranges[dim].start;
            dim -= 1;
        }

        if self.first {
            self.first = false;
        } else {
            self.current[dim] += 1;
        }

        if self.current[dim] >= self.ranges[dim].end {
            return None;
        }

        Some(self.current)
    }
}

/// An iterator over indices within a given range.
///
/// This struct does not implement the `Iterator` trait because such iterators
/// cannot return references to data they contain. Use a `while` loop instead.
pub struct IndexIterator {
    first: bool,
    current: Vec<usize>,
    ranges: Vec<Range<usize>>,
}

impl IndexIterator {
    /// Return an iterator over all the indices where each dimension lies
    /// within the corresponding range in `ranges`.
    ///
    /// If `ranges` is empty, the iterator yields a single empty index. This
    /// is consistent with `ndindex` in eg. numpy.
    pub fn from_ranges(ranges: &[Range<usize>]) -> IndexIterator {
        let current = ranges.iter().map(|r| r.start).collect();
        IndexIterator {
            first: true,
            current,
            ranges: ranges.into(),
        }
    }

    /// Return an iterator over all the indices where each dimension is between
    /// `0` and `shape[dim]`.
    pub fn from_shape(shape: &[usize]) -> IndexIterator {
        let ranges = shape.iter().map(|&size| 0..size).collect();
        let current = vec![0; shape.len()];
        IndexIterator {
            first: true,
            current,
            ranges,
        }
    }

    /// Reset the iterator back to the first index.
    pub fn reset(&mut self) {
        self.first = true;
        for i in 0..self.ranges.len() {
            self.current[i] = self.ranges[i].start;
        }
    }

    /// Return the index index in the sequence, or `None` after all indices
    /// have been returned.
    pub fn next(&mut self) -> Option<&[usize]> {
        if self.current.is_empty() {
            if self.first {
                self.first = false;
                return Some(&self.current[..]);
            } else {
                return None;
            }
        }

        // Find dimension where the last element has not been reached.
        let mut dim = self.current.len() - 1;
        while dim > 0 && self.current[dim] >= self.ranges[dim].end - 1 {
            self.current[dim] = self.ranges[dim].start;
            dim -= 1;
        }

        if self.first {
            self.first = false;
        } else {
            self.current[dim] += 1;
        }

        if self.current[dim] >= self.ranges[dim].end {
            return None;
        }

        Some(&self.current[..])
    }
}

#[cfg(test)]
mod tests {
    use super::{IndexIterator, NdIndexIterator};

    #[test]
    fn test_nd_index_iterator() {
        // Empty iterator
        let mut iter = NdIndexIterator::from_ranges([0..0]);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        // Scalar index iterator
        let mut iter = NdIndexIterator::from_ranges([]);
        assert_eq!(iter.next(), Some([]));
        assert_eq!(iter.next(), None);

        // 1D index iterator
        let iter = NdIndexIterator::from_ranges([0..5]);
        let visited: Vec<_> = iter.collect();
        assert_eq!(visited, &[[0], [1], [2], [3], [4]]);

        // 2D index iterator
        let iter = NdIndexIterator::from_ranges([2..4, 2..4]);
        let visited: Vec<_> = iter.collect();
        assert_eq!(visited, &[[2, 2], [2, 3], [3, 2], [3, 3]]);
    }

    #[test]
    fn test_index_iterator() {
        // Empty iterator
        let mut iter = IndexIterator::from_ranges(&[0..0]);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        // Scalar index iterator
        let mut iter = IndexIterator::from_ranges(&[]);
        assert_eq!(iter.next(), Some(&[] as &[usize]));
        assert_eq!(iter.next(), None);

        // 1D index iterator
        let mut iter = IndexIterator::from_ranges(&[0..5]);
        let mut visited: Vec<Vec<usize>> = Vec::new();
        while let Some(index) = iter.next() {
            visited.push(index.into());
        }
        assert_eq!(visited, vec![vec![0], vec![1], vec![2], vec![3], vec![4]]);

        // 2D index iterator
        let mut iter = IndexIterator::from_ranges(&[2..4, 2..4]);
        let mut visited: Vec<Vec<usize>> = Vec::new();
        while let Some(index) = iter.next() {
            visited.push(index.into());
        }

        assert_eq!(
            visited,
            vec![vec![2, 2], vec![2, 3], vec![3, 2], vec![3, 3],]
        );
    }
}
