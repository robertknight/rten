use std::iter::FusedIterator;
use std::ops::Range;

use smallvec::{smallvec, SmallVec};

pub trait IndexArray: AsMut<[usize]> + AsRef<[usize]> + Clone {}
impl<const N: usize> IndexArray for SmallVec<[usize; N]> {}
impl<const N: usize> IndexArray for [usize; N] {}

/// The index type used for dynamic-rank tensors.
pub type DynIndex = SmallVec<[usize; 5]>;

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

    /// Remaining iteration steps.
    steps: usize,
}

/// Return the number of steps for an index iterator over the range of indices
/// from `from` to `to`.
///
/// If any index in `from` is greater than the corresponding index in `to`,
/// this returns zero.
fn steps(from: &[usize], to: &[usize]) -> usize {
    assert!(from.len() == to.len());
    let mut product = 1;
    for (&from, &to) in from.iter().zip(to.iter()).rev() {
        let size = to.saturating_sub(from);
        product *= size;
    }
    product
}

impl<Index: IndexArray> Indices<Index> {
    fn from_start_and_end(start: Index, end: Index) -> Indices<Index> {
        let steps = steps(start.as_ref(), end.as_ref());
        Indices {
            // Note that if the index is empty, `start == end` but the iterator
            // should yield a single empty element in that case.
            next: if steps > 0 || start.as_ref().is_empty() {
                Some(start.clone())
            } else {
                None
            },
            start,
            end,
            steps,
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
        let current = self.next.clone()?;

        let mut next = current.clone();
        let mut has_next = false;
        for ((&dim_end, &dim_start), index) in self
            .end
            .as_ref()
            .iter()
            .zip(self.start.as_ref())
            .zip(next.as_mut().iter_mut())
            .rev()
        {
            *index += 1;
            if *index == dim_end {
                *index = dim_start;
            } else {
                has_next = true;
                break;
            }
        }

        self.next = has_next.then_some(next);

        Some(current)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.steps, Some(self.steps))
    }
}

impl<Index: IndexArray> ExactSizeIterator for Indices<Index> {}

impl<Index: IndexArray> FusedIterator for Indices<Index> {}

/// Iterator over a range of N-dimensional indices, where N is known at compile
/// time.
pub struct NdIndices<const N: usize> {
    inner: Indices<[usize; N]>,
}

impl<const N: usize> NdIndices<N> {
    pub fn from_ranges(ranges: [Range<usize>; N]) -> NdIndices<N> {
        NdIndices {
            inner: Indices::<[usize; N]>::from_ranges(ranges),
        }
    }

    pub fn from_shape(shape: [usize; N]) -> NdIndices<N> {
        NdIndices {
            inner: Indices::<[usize; N]>::from_shape(shape),
        }
    }
}

impl<const N: usize> Iterator for NdIndices<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize> ExactSizeIterator for NdIndices<N> {}
impl<const N: usize> FusedIterator for NdIndices<N> {}

/// Max tensor rank supported by the variant of [DynIndices] that is optimized
/// for small-rank tensors.
const DYN_SMALL_LEN: usize = 4;

enum DynIndicesInner {
    Small {
        iter: NdIndices<DYN_SMALL_LEN>,
        pad: usize,
    },
    Large(Indices<DynIndex>),
}

/// Iterator over a range of N-dimensional indices, where N is not known at
/// compile time.
pub struct DynIndices {
    inner: DynIndicesInner,
}

/// Left-pad a shape with 1s to size N (eg. [32, 32] => [1, 1, 32, 32]).
fn left_pad_shape<const N: usize>(shape: &[usize]) -> (usize, [usize; N]) {
    assert!(shape.len() <= N);
    let mut padded_shape = [0; N];
    let pad = N - shape.len();
    for i in 0..pad {
        padded_shape[i] = 1;
    }
    for i in pad..N {
        padded_shape[i] = shape[i - pad];
    }
    (N - shape.len(), padded_shape)
}

/// Left-pad ranges with `[0..1]` to size N.
fn left_pad_ranges<const N: usize>(ranges: &[Range<usize>]) -> (usize, [Range<usize>; N]) {
    assert!(ranges.len() <= N);

    // We use a `SmallVec` here because sadly `[elem; N]` doesn't work with
    // Range, which is a non-Copy type :(
    let mut padded_ranges = SmallVec::<[Range<usize>; N]>::from_elem(0..1, N);
    let pad = N - ranges.len();
    for i in 0..pad {
        padded_ranges[i] = 0..1;
    }
    for i in pad..N {
        padded_ranges[i] = ranges[i - pad].clone();
    }
    (N - ranges.len(), padded_ranges.into_inner().unwrap())
}

impl DynIndices {
    pub fn from_shape(shape: &[usize]) -> DynIndices {
        let inner = if shape.len() <= DYN_SMALL_LEN {
            let (pad, padded) = left_pad_shape(shape);
            DynIndicesInner::Small {
                iter: NdIndices::from_shape(padded),
                pad,
            }
        } else {
            DynIndicesInner::Large(Indices::<DynIndex>::from_shape(shape))
        };
        DynIndices { inner }
    }

    pub fn from_ranges(ranges: &[Range<usize>]) -> DynIndices {
        let inner = if ranges.len() <= DYN_SMALL_LEN {
            let (pad, padded) = left_pad_ranges(ranges);
            DynIndicesInner::Small {
                iter: NdIndices::from_ranges(padded),
                pad,
            }
        } else {
            DynIndicesInner::Large(Indices::<DynIndex>::from_ranges(ranges))
        };
        DynIndices { inner }
    }
}

impl Iterator for DynIndices {
    type Item = DynIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner {
            DynIndicesInner::Small { ref mut iter, pad } => {
                iter.next().map(|idx| SmallVec::from_slice(&idx[pad..]))
            }
            DynIndicesInner::Large(ref mut inner) => inner.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.inner {
            DynIndicesInner::Small { ref iter, .. } => iter.size_hint(),
            DynIndicesInner::Large(ref inner) => inner.size_hint(),
        }
    }
}

impl ExactSizeIterator for DynIndices {}
impl FusedIterator for DynIndices {}

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

        // 5D index iterator. This exercises the path for tensors with more
        // than 4 dims.
        let iter = DynIndices::from_shape(&[2, 1, 1, 2, 2]);
        let visited: Vec<Vec<usize>> = iter.map(|ix| ix.into_iter().collect()).collect();
        assert_eq!(
            visited,
            vec![
                vec![0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 1],
                vec![0, 0, 0, 1, 0],
                vec![0, 0, 0, 1, 1],
                //
                vec![1, 0, 0, 0, 0],
                vec![1, 0, 0, 0, 1],
                vec![1, 0, 0, 1, 0],
                vec![1, 0, 0, 1, 1],
            ]
        );
    }

    #[test]
    #[ignore]
    fn bench_indices() {
        use std::time::Instant;

        // Shape taken from GatherElements usage in
        // https://huggingface.co/microsoft/deberta-v3-large.
        //
        // `black_box` is not necessary for the current implementations, but in
        // an experiment with some less branch-y implementations of NdIndices,
        // Rust was able to precompute the iteration count (!).
        let shape = std::hint::black_box([16, 128, 128]);

        // Dynamic rank
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..100 {
            let indices = DynIndices::from_shape(&shape);
            for _ in indices {
                count += 1;
            }
        }
        let elapsed = start.elapsed().as_millis();
        println!("DynIndices stepped {} times in {} ms", count, elapsed);

        // Same shape, static rank
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..100 {
            let indices = NdIndices::from_shape(shape);
            for _ in indices {
                count += 1;
            }
        }
        let elapsed = start.elapsed().as_millis();
        println!("NdIndices stepped {} times in {} ms", count, elapsed);
    }
}
