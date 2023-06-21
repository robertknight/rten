/// Construct a tensor.
///
/// ```
/// use wasnn_tensor::tensor;
///
/// // Create a scalar (0D tensor)
/// tensor!(2.);
///
/// // Create a vector (1D tensor)
/// tensor!([1, 2, 3]);
///
/// // Create a 3D tensor with shape [1, 2, 2] and elements [1, 2, 3, 4].
/// tensor!((1, 2, 2); [1, 2, 3, 4]);
/// ```
#[macro_export]
macro_rules! tensor {
    [[$($elem:expr),*]] => {
        {
            use $crate::Tensor;
            Tensor::from_vec(vec![$($elem),*])
        }
    };

    // As above, but with trailing comma.
    [[$($elem:expr),*,]] => {
        tensor!([$($elem),*])
    };

    (($($dim:expr),+); [$($elem:expr),*]) => {
        {
            use $crate::Tensor;
            Tensor::from_data(&[$($dim),+], vec![$($elem),*])
        }
    };

    ($elem:expr) => {
        {
            use $crate::Tensor;
            Tensor::from_scalar($elem)
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{tensor, Tensor};

    #[test]
    fn test_scalar() {
        let x = tensor!(5.);
        assert_eq!(x, Tensor::from_scalar(5.));
    }

    #[test]
    fn test_vector() {
        let x = tensor!([1, 2, 3]);
        assert_eq!(x, Tensor::from_vec(vec![1, 2, 3]));
    }

    #[test]
    fn test_shape_data() {
        let x = tensor!((1, 2, 2); [1, 2, 3, 4]);
        assert_eq!(x, Tensor::from_data(&[1, 2, 2], vec![1, 2, 3, 4]));
    }
}
