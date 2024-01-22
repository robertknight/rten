use crate::{AsView, Layout};
use crate::{NdTensorView, TensorView};

/// Call `f` with every element in `x` in logical order.
///
/// This is equivalent to `x.iter().for_each(f)` but is faster that Rust's
/// standard iteration protocol when `x` is non-contiguous and has <= 4
/// dimensions.
fn fast_for_each_element<T, F: FnMut(&T)>(mut x: TensorView<T>, mut f: F) {
    if x.ndim() > 4 {
        x.iter().for_each(f)
    } else {
        while x.ndim() < 4 {
            x.insert_axis(0);
        }

        let x_data = x.non_contiguous_data();
        let x: NdTensorView<T, 4> = x.nd_view();
        let shape = x.shape();
        let strides = x.strides();

        assert!(x_data.len() >= x.layout().min_data_len());

        for i0 in 0..shape[0] {
            for i1 in 0..shape[1] {
                for i2 in 0..shape[2] {
                    for i3 in 0..shape[3] {
                        let offset =
                            i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];

                        // Safety: We checked data length > max offset produced
                        // by layout.
                        let elt = unsafe { x_data.get_unchecked(offset) };
                        f(elt)
                    }
                }
            }
        }
    }
}

/// Return the elements of `src` as a contiguous vector, in the same order they
/// would be yielded by `src.iter()`.
///
/// This is equivalent to `src.iter().cloned().collect::<Vec<_>>()` but
/// faster.
pub fn contiguous_data<T: Clone>(src: TensorView<T>) -> Vec<T> {
    let src_len = src.len();

    // This is equivalent to `x.iter().cloned().collect::<Vec<_>>()` but uses a
    // faster iteration method that is optimized for tensors with few (<= 4)
    // dimensions.
    let mut data = Vec::with_capacity(src.len());
    let ptr: *mut T = data.as_mut_ptr();

    let mut offset = 0;
    fast_for_each_element(src, |elt| {
        // Safety: `fast_for_each_element` calls fn `self.len()` times,
        // matching the buffer capacity.
        unsafe { *ptr.add(offset) = elt.clone() };
        offset += 1;
    });

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
