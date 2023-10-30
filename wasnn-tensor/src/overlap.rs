use std::iter::zip;

use smallvec::SmallVec;

/// Return true if a given shape and strides describe a contiguous layout in
/// "C" order.
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    let mut product = 1;
    for (dim, len) in shape.iter().enumerate().rev() {
        if strides[dim] != product {
            return false;
        }
        product *= len;
    }
    true
}

/// Return true if multiple indices may map to the same offset.
///
/// Determining whether arbitrary shapes and strides will overlap is
/// difficult [1][2] so this method is conservative. It verifies that, after
/// sorting dimensions in order of increasing stride, each dimension's
/// stride is larger than the maximum offset that is reachable by indexing
/// the previous dimensions. This correctly reports that there is no overlap
/// for layouts that are contiguous or produced by slicing other
/// non-overlapping layouts. However it is possible to construct
/// combinations of shapes and strides for which no two indicies map to the
/// same offset, but for which this method returns true. For example when
/// `shape == [4, 4]` and `strides == [3, 4]` the offsets are `[0, 3, 4, 6,
/// 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21]`. The maximum offset for
/// the dimension with the smallest stride is `(4-1)*3 == 9`, which is
/// greater than the next-smallest stride. Hence this method will report
/// there may be overlap, even though there is not.
///
/// [1] See https://github.com/numpy/numpy/blob/main/numpy/core/src/common/mem_overlap.c
///     and in particular references to internal overlap.
/// [2] See also references to "memory overlap" in PyTorch source and
///     issues.
pub fn may_have_internal_overlap(shape: &[usize], strides: &[usize]) -> bool {
    // If the tensor is empty (ie. there are no valid indices), there can't be
    // any overlap.
    if shape.iter().any(|&size| size == 0) {
        return false;
    }

    // Fast path for common case of contiguous tensor.
    if is_contiguous(shape, strides) {
        return false;
    }

    // Sort dimensions in order of increasing stride.
    let mut stride_shape: SmallVec<[(usize, usize); 8]> =
        zip(strides.iter().copied(), shape.iter().copied()).collect();
    stride_shape.sort_unstable();

    // Verify that the stride for each dimension fully "steps over" the
    // previous dimension.
    let mut max_offset = 0;
    for (stride, shape) in stride_shape {
        if stride <= max_offset {
            return true;
        }
        max_offset += (shape - 1) * stride;
    }
    false
}
