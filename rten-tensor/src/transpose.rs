use crate::{AsView, Layout};
use crate::{NdLayout, NdTensorView, TensorView};

/// Return the elements of `src` as a contiguous vector, in the same order they
/// would be yielded by `src.iter()`.
///
/// This is equivalent to `src.iter().cloned().collect::<Vec<_>>()` but
/// faster.
pub fn contiguous_data<T: Clone>(mut src: TensorView<T>) -> Vec<T> {
    if src.ndim() > 4 {
        // Fallback for tensors with too many dims.
        return src.iter().cloned().collect();
    }

    let src_len = src.len();

    let mut data = Vec::with_capacity(src.len());

    while src.ndim() < 4 {
        src.insert_axis(0);
    }

    let src_data = src.non_contiguous_data();

    let src_4d: NdTensorView<T, 4> = src.nd_view();
    let dest_strides = NdLayout::contiguous_strides(src_4d.shape());

    struct Dim {
        size: usize,
        src_stride: usize,
        dest_stride: usize,
    }

    let mut dims = [0, 1, 2, 3].map(|d| Dim {
        size: src_4d.size(d),
        src_stride: src_4d.stride(d),
        dest_stride: dest_strides[d],
    });

    // Sort dims in order of decreasing source stride. This makes the traversal
    // order over the source as contiguous as possible, which is important for
    // performance.
    dims.sort_by_key(|dim| dim.src_stride);
    dims.reverse();

    // Sanity check before unsafe code below.
    assert!(src_data.len() >= src_4d.layout().min_data_len());

    for i0 in 0..dims[0].size {
        for i1 in 0..dims[1].size {
            for i2 in 0..dims[2].size {
                for i3 in 0..dims[3].size {
                    let src_offset = i0 * dims[0].src_stride
                        + i1 * dims[1].src_stride
                        + i2 * dims[2].src_stride
                        + i3 * dims[3].src_stride;
                    let dest_offset = i0 * dims[0].dest_stride
                        + i1 * dims[1].dest_stride
                        + i2 * dims[2].dest_stride
                        + i3 * dims[3].dest_stride;

                    // Safety: We checked data length > max offset produced
                    // by layout.
                    let elt = unsafe { src_data.get_unchecked(src_offset) };
                    unsafe {
                        *data.get_unchecked_mut(dest_offset) = elt.clone();
                    };
                }
            }
        }
    }

    // Safety: Length here matches capacity passed to `Vec::with_capacity`.
    unsafe { data.set_len(src_len) }

    data
}

#[cfg(test)]
mod tests {
    use super::contiguous_data;
    use crate::{AsView, Tensor};

    #[test]
    fn test_contiguous_data() {
        // <= 4 dims
        let x = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        assert_eq!(contiguous_data(x.view()), [1, 2, 3, 4]);
        assert_eq!(contiguous_data(x.transposed()), [1, 3, 2, 4]);

        // > 4 dims
        let x = Tensor::from_data(&[1, 1, 1, 2, 2], vec![1, 2, 3, 4]);
        assert_eq!(contiguous_data(x.view()), [1, 2, 3, 4]);
        assert_eq!(contiguous_data(x.transposed()), [1, 3, 2, 4]);
    }
}
